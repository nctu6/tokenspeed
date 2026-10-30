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

"""Fused residual add + LayerNorm (weight and bias) against the eager pair."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.layernorm import add_layernorm
from tokenspeed_kernel.ops.layernorm import triton as ln_triton
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

platform = current_platform()
pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)


def _inputs(rows, width, dtype, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, width, generator=gen, device=device, dtype=dtype)
    residual = 4 * torch.randn(rows, width, generator=gen, device=device, dtype=dtype)
    weight = 1 + 0.2 * torch.randn(width, generator=gen, device=device, dtype=dtype)
    bias = 0.2 * torch.randn(width, generator=gen, device=device, dtype=dtype)
    return x, residual, weight, bias


def _assert_within_one_ulp(out: torch.Tensor, exact: torch.Tensor) -> None:
    """``out`` is less than one ulp from the float64 ``exact`` value: a
    correctly rounded result is within half an ulp; the FP32 row statistics
    may move a value that sits near a rounding midpoint to the other
    neighbour, never further. Outputs near zero (the bias cancelling the
    normalized value) also carry the FP32 rounding of the O(1) terms that
    cancel, a few 1e-7 absolute, which is more than an FP16 ulp there."""
    if out.dtype == torch.float32:
        torch.testing.assert_close(out.double(), exact, atol=1e-5, rtol=1e-5)
        return
    mantissa = {torch.bfloat16: 7, torch.float16: 10}[out.dtype]
    mag = exact.abs().clamp_min(2.0**-14)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - mantissa)
    diff = (out.double() - exact).abs()
    assert (diff <= ulp + 4e-6).all(), (diff / ulp).max().item()


def _exact_layer_norm(x, weight, bias, eps):
    return F.layer_norm(x.double(), (x.shape[-1],), weight.double(), bias.double(), eps)


@pytest.mark.parametrize("rows", [1, 3, 16, 1500])
@pytest.mark.parametrize("width", [1280, 384, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("seed", [0, 1])
def test_add_layernorm_matches_eager(rows, width, dtype, seed, device):
    x, residual, weight, bias = _inputs(rows, width, dtype, seed, device)
    eps = 1e-5
    summed = residual + x
    ref = F.layer_norm(summed, (width,), weight, bias, eps)

    out, new_residual = add_layernorm(x, residual, weight, bias, eps=eps)

    # The residual stream is the eager sum, bit for bit.
    assert torch.equal(new_residual, summed)
    # The norm rounds the exact LayerNorm of that sum once, as torch's does.
    exact = _exact_layer_norm(summed, weight, bias, eps)
    _assert_within_one_ulp(out, exact)
    _assert_within_one_ulp(ref, exact)


@pytest.mark.parametrize("rows", [1, 7])
def test_add_layernorm_without_residual_is_layer_norm(rows, device):
    x, _, weight, bias = _inputs(rows, 1280, torch.bfloat16, 3, device)
    out, new_residual = add_layernorm(x, None, weight, bias, eps=1e-5)
    assert new_residual is x
    _assert_within_one_ulp(out, _exact_layer_norm(x, weight, bias, 1e-5))


def test_add_layernorm_strided_rows(device):
    # Row-strided views (unit inner stride) read and write the right rows.
    x_wide, r_wide, weight, bias = _inputs(5, 2 * 1280, torch.bfloat16, 4, device)
    x, residual = x_wide[:, :1280], r_wide[:, 1280:]
    weight, bias = weight[:1280].contiguous(), bias[:1280].contiguous()
    out, new_residual = add_layernorm(x, residual, weight, bias, eps=1e-5)
    summed = residual + x
    assert torch.equal(new_residual, summed)
    _assert_within_one_ulp(out, _exact_layer_norm(summed, weight, bias, 1e-5))


def test_add_layernorm_compiles_nothing_new_across_row_counts(device):
    weight_bias = _inputs(1, 1280, torch.bfloat16, 5, device)[2:]
    for rows in (1, 16):  # warm-up: one launch per integer class
        x, residual, _, _ = _inputs(rows, 1280, torch.bfloat16, rows, device)
        add_layernorm(x, residual, *weight_bias, eps=1e-5)
    with assert_no_triton_compile(ln_triton._triton_add_layernorm_kernel):
        for rows in range(1, 17):
            x, residual, _, _ = _inputs(rows, 1280, torch.bfloat16, rows, device)
            add_layernorm(x, residual, *weight_bias, eps=1e-5)
        torch.cuda.synchronize()
