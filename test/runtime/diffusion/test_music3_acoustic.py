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

"""CPU: the streamed, batched acoustic stage equals the Diffusers denoise + decode."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from diffusers import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from diffusers.modular_pipelines import SequentialPipelineBlocks  # noqa: E402
from diffusers.modular_pipelines.minimax_music3 import MiniMaxMusic3Blocks  # noqa: E402

from tokenspeed.runtime.diffusion.music3_acoustic import (  # noqa: E402
    AcousticSong,
    Music3Acoustic,
)

HIDDEN, LAYERS = 8, 8  # frame hidden = 8 condition layers x 8
STEPS = 3


def _components():
    torch.manual_seed(0)
    encoder = MiniMaxMusic3ConditionEncoder(
        condition_hidden_dim=HIDDEN, num_condition_layers=LAYERS, out_dim=16
    ).eval()
    transformer = MiniMaxMusic3Transformer1DModel(
        in_channels=8,
        condition_dim=16,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        ff_inner_dim=32,
        rotary_dim=4,
        fourier_embedding_dim=16,
    ).eval()
    # Upsampling must stay 512x: the crops are in units of the 512-sample hop.
    vocoder = MiniMaxMusic3Vocoder(
        latent_channels=8, decoder_input_dim=16, decoder_hidden_dim=16
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1, shift=1.0, invert_sigmas=True
    )
    return encoder, transformer, vocoder, scheduler


def _frames(num_frames: int, seed: int) -> torch.Tensor:
    return torch.randn(
        1, num_frames, HIDDEN * LAYERS, generator=torch.Generator().manual_seed(seed)
    )


@torch.no_grad()
def _diffusers_audio(frames: torch.Tensor, seed: int) -> np.ndarray:
    encoder, transformer, vocoder, scheduler = _components()
    blocks = MiniMaxMusic3Blocks()
    rest = SequentialPipelineBlocks.from_blocks_dict(
        {"denoise": blocks.sub_blocks["denoise"], "decode": blocks.sub_blocks["decode"]}
    )
    pipe = rest.init_pipeline()
    pipe.update_components(
        condition_encoder=encoder,
        transformer=transformer,
        vocoder=vocoder,
        scheduler=scheduler,
    )
    audio = pipe(
        frame_hiddens=frames,
        generator=torch.Generator().manual_seed(seed),
        num_inference_steps=STEPS,
        output="audios",
    )
    return np.asarray(audio)[0]


def _engine():
    encoder, transformer, vocoder, scheduler = _components()
    return Music3Acoustic(
        encoder,
        transformer,
        vocoder,
        scheduler,
        device="cpu",
        latent_hop=512,
        low_priority=False,
    )


def _song(frames: torch.Tensor, seed: int) -> AcousticSong:
    return AcousticSong(
        seed=seed,
        num_inference_steps=STEPS,
        generator=torch.Generator().manual_seed(seed),
        frames=[f[None] for f in frames[0]],
    )


@torch.no_grad()
def _stream(
    engine: Music3Acoustic, songs: list[AcousticSong], totals: list[int]
) -> list[np.ndarray]:
    """Feed frames 50 at a time like the AR stage; batch every ready window."""
    pieces: list[list[np.ndarray]] = [[] for _ in songs]
    step = 0
    while not all(s.finished for s in songs):
        for song, total in zip(songs, totals):
            song.frames_available = min(total, 50 * (step + 1))
            if song.frames_available == total:
                song.total_frames = total
        step += 1
        ready = [(i, s.ready_window()) for i, s in enumerate(songs)]
        ready = [(i, w) for i, w in ready if w is not None]
        while ready:
            length = ready[0][1][1] - ready[0][1][0]
            group = [(i, w) for i, w in ready if w[1] - w[0] == length]
            outputs = engine.run_windows([(songs[i], *w) for i, w in group])
            for (i, _), (piece, _last) in zip(group, outputs):
                pieces[i].append(piece)
            ready = [(i, s.ready_window()) for i, s in enumerate(songs)]
            ready = [(i, w) for i, w in ready if w is not None]
    return [np.concatenate(p, axis=-1) for p in pieces]


@pytest.mark.parametrize("num_frames", [150, 450])  # one window / four windows
def test_streamed_windows_equal_diffusers(num_frames):
    frames = _frames(num_frames, seed=1)
    expected = _diffusers_audio(frames, seed=5)
    (got,) = _stream(_engine(), [_song(frames, 5)], [num_frames])
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_batched_songs_equal_each_alone():
    a, b = _frames(450, seed=2), _frames(360, seed=3)
    alone_a = _stream(_engine(), [_song(a, 7)], [450])[0]
    alone_b = _stream(_engine(), [_song(b, 8)], [360])[0]
    both = _stream(_engine(), [_song(a, 7), _song(b, 8)], [450, 360])
    np.testing.assert_allclose(both[0], alone_a, rtol=0, atol=1e-5)
    np.testing.assert_allclose(both[1], alone_b, rtol=0, atol=1e-5)


@torch.no_grad()
def test_folded_weight_norm_vocoder_is_the_same_decoder():
    import copy

    from tokenspeed.runtime.diffusion.music3_pipeline import _fold_weight_norm

    _, _, vocoder, _ = _components()
    latents = torch.randn(1, 8, 40, generator=torch.Generator().manual_seed(9))
    expected = vocoder(latents)
    folded = vocoder
    assert _fold_weight_norm(folded) > 0
    assert not any(hasattr(m, "weight_g") for m in folded.modules())
    copy.deepcopy(folded)  # a replica for a second device can now be made
    torch.testing.assert_close(folded(latents), expected, rtol=0, atol=1e-6)


def test_a_window_waits_until_its_frames_exist():
    song = _song(_frames(450, seed=4), 1)
    song.frames_available = 200  # window 0 covers [0, 200): one more frame needed
    assert song.ready_window() is None
    song.frames_available = 201
    assert song.ready_window() == (0, 200, False)
    song.next_window = 3  # [300, 450) is the last window once the total is known
    song.frames_available = 450
    assert song.ready_window() is None
    song.total_frames = 450
    assert song.ready_window() == (300, 450, True)
