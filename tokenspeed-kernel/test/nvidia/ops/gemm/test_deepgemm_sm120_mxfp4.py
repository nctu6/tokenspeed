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

"""SM120 DeepGEMM MXFP4: packed-scale quantizer and the ``mxfp4_linear`` leaf."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mxfp4_linear, prepare_mxfp4_linear
from tokenspeed_kernel.ops.quantization.triton import (
    _mxfp4_quantize_mn_major_kernel,
    mxfp4_quantize,
    mxfp4_quantize_mn_major_scales,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.deepgemm import (
    deepgemm_supports_sm120,
    pack_ue8m0_scales,
)
from utils import assert_no_triton_compile

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
requires_sm120_deepgemm = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform().arch_version.major == 12
        and deepgemm_supports_sm120()
    ),
    reason="requires SM120 and a DeepGEMM build with SM120 kernels",
)


def _random_activations(m: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    # Wide dynamic range so groups land on many different scales; one all-zero group.
    x = torch.randn(m, k, device="cuda") * torch.rand(m, k, device="cuda") ** 4 * 50
    x[0, :32] = 0
    return x.to(dtype)


@requires_cuda
@pytest.mark.parametrize("m", [1, 3, 4, 5, 9, 64, 130])
@pytest.mark.parametrize("k", [128, 512, 7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_scale_quantizer_matches_linear_quantizer(
    m: int, k: int, dtype: torch.dtype
) -> None:
    x = _random_activations(m, k, dtype)
    packed, scale = mxfp4_quantize(x)
    packed_mn_major, scale_mn_major = mxfp4_quantize_mn_major_scales(x)

    torch.testing.assert_close(packed_mn_major, packed, rtol=0, atol=0)
    expected = pack_ue8m0_scales(scale)
    torch.testing.assert_close(scale_mn_major, expected, rtol=0, atol=0)
    assert scale_mn_major.stride() == (1, (m + 3) // 4 * 4)


@requires_cuda
def test_packed_scale_quantizer_does_not_recompile_across_batch_sizes() -> None:
    k = 512
    for m in (1, 17):  # warm both integer-specialization classes of M
        mxfp4_quantize_mn_major_scales(_random_activations(m, k, torch.bfloat16))
    with assert_no_triton_compile(_mxfp4_quantize_mn_major_kernel):
        # Small grids launch with PDL, large ones as a plain kernel.
        for m in (1, 2, 7, 16, 33, 96, 257, 1000, 4096, 5000):
            mxfp4_quantize_mn_major_scales(_random_activations(m, k, torch.bfloat16))


def _dequantize(packed: torch.Tensor, scale: torch.Tensor, k: int) -> torch.Tensor:
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
        dtype=torch.float32,
    )
    values = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1)
    values = values.reshape(packed.shape[0], k)
    exponent = (scale.to(torch.int32) << 23).view(torch.float32)
    return values * exponent.repeat_interleave(32, 1)


@requires_sm120_deepgemm
@pytest.mark.parametrize(
    ("m", "n", "k"), [(1, 2112, 7168), (64, 4096, 512), (130, 3072, 1536)]
)
def test_deepgemm_mxfp4_linear_matches_dequantized_reference(
    m: int, n: int, k: int
) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight, weight_scale = mxfp4_quantize(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    )
    prepared = prepare_mxfp4_linear(
        weight, weight_scale, swiglu=False, solution="deepgemm"
    )

    out = mxfp4_linear(x, prepared)

    x_packed, x_scale = mxfp4_quantize(x)
    reference = (
        _dequantize(x_packed, x_scale, k) @ _dequantize(weight, weight_scale, k).T
    )
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
