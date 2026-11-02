# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""CPU-only: the AdaLN memo is exact on a real (tiny) MiniMax-H3 transformer."""

from __future__ import annotations

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from diffusers import MiniMaxH3Transformer3DModel  # noqa: E402

from tokenspeed.runtime.diffusion.adaln_memo import (  # noqa: E402
    AdaLNMemoTable,
    install_adaln_memo,
)

TEXT, VIDEO, AUDIO = 3, 4, 2


def _tiny_transformer() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=3,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=16,
        time_embed_hidden_dim=24,
        time_embed_dim=12,
        rope_freq_dim=2,
    )
    return model.eval()


def _inputs(timestep: list[float]) -> dict:
    seq = TEXT + VIDEO + AUDIO
    g = torch.Generator().manual_seed(1)
    text_indices = torch.arange(0, TEXT)
    video_indices = torch.arange(TEXT, TEXT + VIDEO)
    audio_indices = torch.arange(TEXT + VIDEO, seq)
    token_tags = torch.tensor([1] * TEXT + [0] * VIDEO + [2] * AUDIO)
    # Conditioning rows (text) at timestep 0, generated rows at timestep 1.
    timestep_indices = torch.tensor([0] * TEXT + [1] * (VIDEO + AUDIO))
    return {
        "hidden_states": torch.randn(1, VIDEO, 4, generator=g),
        "audio_hidden_states": torch.randn(1, AUDIO, 4, generator=g),
        "encoder_hidden_states": torch.randn(1, TEXT, 8, generator=g),
        "timestep": torch.tensor(timestep),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": torch.randint(0, 4, (seq, 3), generator=g),
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "return_dict": False,
    }


@torch.no_grad()
def test_memoized_forward_matches_the_original_bitwise():
    model = _tiny_transformer()
    steps = [[0.0, 0.9], [0.0, 0.5]]
    expected = [model(**_inputs(t)) for t in steps]

    table = AdaLNMemoTable(max_bytes=1 << 20)
    assert install_adaln_memo(model, table) == 3
    # The projection weights left the module tree (they stay on the host).
    assert not any("adaln_proj" in name for name, _ in model.named_parameters())

    for _ in range(2):  # first pass fills the memo, second pass reads it
        for t, (video, audio) in zip(steps, expected):
            got_video, got_audio = model(**_inputs(t))
            assert torch.equal(got_video, video)
            assert torch.equal(got_audio, audio)
    assert (table.misses, table.hits) == (2 * 3, 2 * 3)


@torch.no_grad()
def test_distinct_timesteps_never_share_an_entry():
    model = _tiny_transformer()
    table = AdaLNMemoTable(max_bytes=1 << 20)
    install_adaln_memo(model, table)
    model(**_inputs([0.0, 0.9]))
    model(**_inputs([0.0, 0.90000004]))  # one float32 ulp away
    assert table.hits == 0 and table.misses == 6


@torch.no_grad()
def test_lru_bound_evicts_oldest_steps():
    model = _tiny_transformer()
    table = AdaLNMemoTable(max_bytes=1 << 20)
    install_adaln_memo(model, table)
    model(**_inputs([0.0, 0.9]))
    one_step = table.bytes
    table.max_bytes = one_step  # room for exactly one step
    model(**_inputs([0.0, 0.5]))
    assert table.bytes <= one_step
    model(**_inputs([0.0, 0.5]))  # newest step still cached
    assert table.hits == 3


def test_memo_outside_a_forward_fails_closed():
    table = AdaLNMemoTable(max_bytes=1 << 20)
    with pytest.raises(RuntimeError, match="no step key"):
        table.lookup(0)
