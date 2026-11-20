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

"""RoPE scaling of DeepSeek-V3-architecture MLA checkpoints without scaling.

transformers 5 normalizes a config with no ``rope_scaling`` (Moonlight-16B,
DeepSeek-V2-Lite) into ``{"rope_theta": ..., "rope_type": "default"}``. The
MLA layer turned every truthy scaling into deepseek-yarn and then failed on
the missing ``factor`` (KeyError at model construction).
"""

from __future__ import annotations

import os
import sys

import torch
from transformers import DeepseekV3Config

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    RotaryEmbedding,
    get_rope,
)
from tokenspeed.runtime.models.deepseek_v3 import mla_rope_scaling  # noqa: E402


def _mla_rope(config):
    """The rotary embedding the MLA layer builds for ``config``."""
    rope_scaling = mla_rope_scaling(config)
    if rope_scaling:
        rope_scaling = dict(rope_scaling, rope_type="deepseek_yarn")
    return get_rope(
        64,
        rotary_dim=64,
        max_position=config.max_position_embeddings,
        base=50000.0,
        rope_scaling=rope_scaling,
        is_neox_style=False,
        dtype=torch.float32,
    )


def test_unscaled_checkpoint_gets_plain_rope():
    config = DeepseekV3Config(max_position_embeddings=8192, rope_theta=50000.0)
    # The transformers normalization this guards against.
    assert config.rope_scaling and config.rope_scaling.get("rope_type") == "default"
    assert mla_rope_scaling(config) is None
    assert type(_mla_rope(config)) is RotaryEmbedding


def test_yarn_checkpoint_keeps_its_scaling():
    yarn = {
        "type": "yarn",
        "factor": 40.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    config = DeepseekV3Config(max_position_embeddings=163840, rope_scaling=yarn)
    scaling = mla_rope_scaling(config)
    assert scaling is not None and scaling["factor"] == 40.0
    assert type(_mla_rope(config)) is not RotaryEmbedding
