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

"""Skinny dense BF16 GEMM (M <= 16): values, selection and compile stability."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.gemm._triton import skinny
from tokenspeed_kernel.ops.gemm.triton import triton_mm_bf16_skinny
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

platform = current_platform()
pytestmark = pytest.mark.skipif(
    not platform.is_nvidia, reason="tl.dot BF16 skinny GEMM test runs on NVIDIA"
)

# whisper-large-v3 decoder projections (N, K) plus a non-square small one.
SHAPES = [(3840, 1280), (1280, 1280), (5120, 1280), (1280, 5120), (256, 384)]


def _operands(m, n, k, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, generator=gen, device=device, dtype=torch.bfloat16)
    b = 0.05 * torch.randn(n, k, generator=gen, device=device, dtype=torch.bfloat16)
    bias = torch.randn(n, generator=gen, device=device, dtype=torch.bfloat16)
    return a, b, bias


def _ulp(t: torch.Tensor) -> torch.Tensor:
    mag = t.float().abs().clamp_min(2.0**-14)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


@pytest.mark.parametrize("m", [1, 2, 5, 16])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_skinny_matches_fp32_reference(m, n, k, with_bias, device):
    a, b, bias = _operands(m, n, k, m * 7 + n, device)
    bias = bias if with_bias else None
    out = triton_mm_bf16_skinny(a, b, None, None, torch.bfloat16, bias=bias)
    ref = a.float() @ b.float().T
    if bias is not None:
        ref = ref + bias.float()
    # FP32 accumulation, one rounding: within one BF16 ulp of the FP32 value
    # plus FP32 summation-order noise.
    diff = (out.float() - ref).abs()
    assert (diff <= _ulp(ref) + 1e-3 * ref.abs().max()).all(), diff.max().item()


def test_skinny_writes_out_and_reads_strided_rows(device):
    a_wide, b, bias = _operands(4, 1280, 2 * 1280, 3, device)
    a = a_wide[:, 1280:]
    b = b[:, :1280].contiguous()
    out_wide = torch.full((4, 2 * 1280), 7.0, device=device, dtype=torch.bfloat16)
    out = out_wide[:, :1280]
    got = triton_mm_bf16_skinny(a, b, None, None, torch.bfloat16, bias=bias, out=out)
    assert got.data_ptr() == out.data_ptr()
    ref = a.float() @ b.float().T + bias.float()
    assert ((out.float() - ref).abs() <= _ulp(ref) + 1e-3 * ref.abs().max()).all()
    # Columns outside the view are untouched.
    assert (out_wide[:, 1280:] == 7.0).all()


def test_skinny_rejects_unsupported_shapes(device):
    a, b, _ = _operands(17, 1280, 1280, 0, device)
    with pytest.raises(ValueError):
        triton_mm_bf16_skinny(a, b, None, None, torch.bfloat16)
    a, b, _ = _operands(2, 1288, 1280, 0, device)
    with pytest.raises(ValueError):
        triton_mm_bf16_skinny(a, b, None, None, torch.bfloat16)


@pytest.mark.skipif(
    not (
        platform.is_nvidia
        and platform.arch_version.major == 12
        and platform.arch_version.minor == 0
    ),
    reason="registered for NVIDIA SM120 only",
)
@pytest.mark.parametrize("m,expected", [(1, True), (16, True), (17, False)])
@pytest.mark.parametrize("n", [1280, 3840])
def test_mm_selects_skinny_for_decode_rows_on_sm120(m, expected, n, device):
    a, b, bias = _operands(m, n, 1280, 1, device)
    traits = {
        "m": m,
        "n": n,
        "k": 1280,
        # mm's label for this B: a square [N, K] weight reads as "KN".
        "b_layout": "KN" if n == 1280 else "NK",
        "a_inner_stride_one": True,
        "a_ndim": 2,
        "a_scales_inner_stride_one": True,
        "b_inner_stride_one": True,
        "b_scales_inner_stride_one": True,
        "block_scale_layout": "canonical_blackwell",
        "out_dtype": torch.bfloat16,
        "out_inner_stride_one": True,
        "pdl_enabled": False,
    }
    kernel = select_kernel(
        "gemm",
        "mm",
        format_signature(
            a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
        ),
        traits=traits,
    )
    assert (kernel.name == "triton_mm_bf16_skinny") is expected
    # The public mm gives the same values either way.
    got = mm(a, b, bias=bias)
    ref = a.float() @ b.float().T + bias.float()
    assert ((got.float() - ref).abs() <= _ulp(ref) + 1e-3 * ref.abs().max()).all()


def test_mm_with_batched_rows_keeps_every_row(device):
    # A 3-D A's first dim is not its row count (m = A.shape[0] = 1 here), so
    # the bounded-m kernel must not take it on any architecture.
    a, b, bias = _operands(40, 1280, 1280, 2, device)
    got = mm(a.view(1, 40, 1280), b, bias=bias)
    ref = a.float() @ b.float().T + bias.float()
    assert got.shape == (1, 40, 1280)
    assert ((got[0].float() - ref).abs() <= _ulp(ref) + 1e-3 * ref.abs().max()).all()


def test_skinny_compiles_nothing_new_across_decode_rows(device):
    operands = {(n, k): _operands(16, n, k, n + k, device) for n, k in SHAPES[:4]}
    for (n, k), (a, b, bias) in operands.items():  # one warm-up per shape
        triton_mm_bf16_skinny(a[:3], b, None, None, torch.bfloat16, bias=bias)
    with assert_no_triton_compile(skinny._triton_mm_bf16_skinny_kernel):
        for m in range(1, 17):
            for (n, k), (a, b, bias) in operands.items():
                triton_mm_bf16_skinny(a[:m], b, None, None, torch.bfloat16, bias=bias)
        torch.cuda.synchronize()
