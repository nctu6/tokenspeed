# Copyright (c) 2026 LightSeek Foundation
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch

from tokenspeed.runtime.layers.rotary_embedding import _yarn_get_mscale
from tokenspeed.runtime.models.mistral3 import (
    Mistral3ForConditionalGeneration,
    hf_yarn_rope_scaling,
    llama4_query_scale,
)


def test_llama4_query_scale_matches_hf_formula():
    pos = torch.tensor([0, 1, 4095, 4096, 8191, 8192, 40000])
    got = llama4_query_scale(pos, 0.1, 4096)
    want = [1 + 0.1 * math.log(1 + math.floor(p / 4096)) for p in pos.tolist()]
    torch.testing.assert_close(got, torch.tensor(want), rtol=0, atol=1e-6)


def test_checkpoint_names_map_to_text_model():
    r = Mistral3ForConditionalGeneration._rename
    assert (
        r("model.language_model.layers.3.self_attn.q_proj.weight")
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        r("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")
        == "model.layers.0.mlp.down_proj.weight_scale"
    )
    assert (
        r("model.language_model.layers.0.self_attn.k_proj.activation_scale")
        == "model.layers.0.self_attn.k_proj.input_scale"
    )
    assert r("lm_head.weight") == "lm_head.weight"
    # Vision tower / projector are unsupported and dropped.
    assert r("model.vision_tower.patch_conv.weight") is None
    assert r("model.multi_modal_projector.linear_1.weight") is None


def _hf_attention_factor(p):
    def m(scale, ms=1.0):
        return 1.0 if scale <= 1 else 0.1 * ms * math.log(scale) + 1.0

    if p.get("mscale") and p.get("mscale_all_dim"):
        return m(p["factor"], p["mscale"]) / m(p["factor"], p["mscale_all_dim"])
    return m(p["factor"])


def test_yarn_attention_factor_matches_hf():
    # Ministral-3-14B (mscale == mscale_all_dim -> 1.0) and the tiny config
    # (mscale_all_dim == 0 -> mscale(factor)).
    for p in (
        {
            "rope_type": "yarn",
            "factor": 16.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
        },
        {
            "rope_type": "yarn",
            "factor": 64.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 4.0,
        },
    ):
        out = hf_yarn_rope_scaling(p)
        got = _yarn_get_mscale(p["factor"]) * out["attn_factor"]
        assert math.isclose(got, _hf_attention_factor(p), rel_tol=1e-9)
