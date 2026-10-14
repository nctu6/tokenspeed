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

"""SM90 head_dim 512 CuTe-DSL paged extend: selection and correctness."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_flashinfer_fa2_extend import (  # noqa: E402
    _build_extend_inputs,
    _torch_extend_ref,
)

sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 only",
)


def _select(head_dim: int, fa4_env: str) -> str:
    code = textwrap.dedent(f"""
        import torch
        from tokenspeed_kernel.signature import dense_tensor_format, format_signature
        from tokenspeed_kernel.selection import select_kernel
        import tokenspeed_kernel.ops.attention.mha  # noqa: F401
        sig = format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k_cache=dense_tensor_format(torch.bfloat16),
            v_cache=dense_tensor_format(torch.bfloat16),
        )
        traits = {{"head_dim": {head_dim}, "page_size": 64, "is_causal": True,
                   "sliding_window": False, "logit_cap": False, "sinks": False,
                   "return_lse": False}}
        print(select_kernel("attention", "mha_extend_with_kvcache", sig, traits=traits).name)
        """)
    env = os.environ.copy()
    env["TOKENSPEED_FA4_SM90_HD512"] = fa4_env
    env.pop("TOKENSPEED_FLASHINFER_FA2_EXTEND", None)
    env.pop("TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_EXTEND_WITH_KVCACHE", None)
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env
    )
    return out.stdout.strip().splitlines()[-1]


@sm90
def test_auto_selection_prefers_cute_hd512_and_keeps_fa2_fallback():
    assert _select(512, "on") == "fa4_sm90_hd512_mha_extend_with_kvcache"
    assert "flashinfer_fa2" in _select(512, "off")
    # head_dim 256 stays on FA3.
    assert "fa4_sm90_hd512" not in _select(256, "on")


@sm90
@pytest.mark.parametrize(
    "num_q,num_kv,prefix,query",
    [
        (32, 4, [0], [17]),
        (32, 4, [2048], [3]),
        (32, 4, [4096, 0, 100], [256, 64, 1]),
        (16, 2, [128, 64], [2, 1]),
    ],
)
def test_cute_hd512_matches_fa2_and_fp32(num_q, num_kv, prefix, query):
    from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

    inputs = _build_extend_inputs(
        device="cuda",
        dtype=torch.bfloat16,
        head_dim=512,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        page_size=64,
        prefix_lens=prefix,
        query_lens=query,
    )
    scale = 1.0 / math.sqrt(512)
    out = mha_extend_with_kvcache(
        **inputs, softmax_scale=scale, override="fa4_sm90_hd512_mha_extend_with_kvcache"
    )
    out_fa2 = mha_extend_with_kvcache(
        **inputs,
        softmax_scale=scale,
        override="flashinfer_fa2_mha_extend_with_kvcache",
    )
    ref = _torch_extend_ref(inputs, scale)
    torch.testing.assert_close(out, out_fa2, atol=2e-2, rtol=2e-2)
    err = (out.float() - ref).abs().max().item()
    err_fa2 = (out_fa2.float() - ref).abs().max().item()
    assert err <= err_fa2 * 1.5 + 1e-2, (err, err_fa2)
