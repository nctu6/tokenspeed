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

"""CPU-only: text-encoder truncation is exact on a real (tiny) Qwen3-VL."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from tokenspeed.runtime.diffusion.residency import (  # noqa: E402
    text_encoder_device_map,
    truncate_text_encoder,
)

NUM_LAYERS = 6
READ_INDEX = 3  # stands in for H3's hidden_states[50]


def _tiny_qwen3_vl() -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 96,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
            "deepstack_visual_indexes": [0],
        },
    )
    return Qwen3VLForConditionalGeneration(config).eval()


def _hidden_states(model) -> tuple[torch.Tensor, ...]:
    ids = torch.tensor([[5, 9, 17, 33, 2, 41, 7]])
    out = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        output_hidden_states=True,
    )
    return out.hidden_states


@torch.no_grad()
def test_truncation_keeps_the_read_hidden_state_exact():
    model = _tiny_qwen3_vl()
    expected = _hidden_states(model)
    assert truncate_text_encoder(model, READ_INDEX) == READ_INDEX + 1
    assert model.config.text_config.num_hidden_layers == READ_INDEX + 1
    got = _hidden_states(model)
    for index in range(READ_INDEX + 1):
        assert torch.equal(got[index], expected[index])


@torch.no_grad()
def test_one_layer_fewer_would_return_the_post_norm_state():
    # Guards the "+ 1": with exactly READ_INDEX layers the entry is post-norm.
    model = _tiny_qwen3_vl()
    expected = _hidden_states(model)[READ_INDEX]
    truncate_text_encoder(model, READ_INDEX - 1)
    assert not torch.equal(_hidden_states(model)[READ_INDEX], expected)


def test_truncation_beyond_the_stack_fails_closed():
    with pytest.raises(ValueError, match="needs"):
        truncate_text_encoder(_tiny_qwen3_vl(), NUM_LAYERS)


def test_device_map_covers_every_weight_in_contiguous_ranges():
    model = _tiny_qwen3_vl()
    truncate_text_encoder(model, READ_INDEX)
    device_map = text_encoder_device_map(model, ["cuda:0", "cuda:1"])
    layers = [device_map[f"model.language_model.layers.{i}"] for i in range(4)]
    assert layers == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert device_map["model.language_model.embed_tokens"] == "cuda:0"
    assert device_map["model.language_model.norm"] == "cuda:1"
    assert device_map["model.visual"] == "cuda:0"
