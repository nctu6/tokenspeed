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
"""The dense NVFP4 linear path: activation quantization plus ``mm(quant="nvfp4")``.

This is what ``Nvfp4LinearMethod.apply`` runs. On SM120 the FP4 quantizer was
imported only for SM100-family parts, so the first dense NVFP4 layer of
Qwen3.8-27B-NVFP4 raised "Kernel implementation not found".
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_nvfp4_blocks
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell_plus,
    reason="NVFP4 GEMM needs NVIDIA Blackwell or newer",
)


def _quantize(x: torch.Tensor, global_scale: torch.Tensor):
    """Swizzled operands for the GEMM and their FP32 values for the reference."""
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    packed, swizzled = fp4_quantize(x, global_scale)
    linear_packed, linear = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
    assert torch.equal(packed, linear_packed)
    values = dequantize_nvfp4_blocks(
        packed, linear.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
    )
    return packed, swizzled, values / global_scale


@pytest.mark.parametrize("m", [1, 7, 64])
def test_quantized_linear_matches_dequantized_reference(m: int) -> None:
    n, k = 256, 512
    generator = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    w = torch.randn(n, k, device="cuda", generator=generator).bfloat16()
    x_global = ((448 * 6) / x.abs().max().float()).reshape(1)
    w_global = ((448 * 6) / w.abs().max().float()).reshape(1)
    x_fp4, x_scales, x_values = _quantize(x, x_global)
    w_fp4, w_scales, w_values = _quantize(w, w_global)

    out = mm(
        x_fp4,
        w_fp4.T,
        A_scales=x_scales,
        B_scales=w_scales.T,
        out_dtype=torch.bfloat16,
        alpha=(1.0 / (x_global * w_global)).float(),
        quant="nvfp4",
    )

    # The operands are exact FP4 values, so only the FP32 sum order and the
    # BF16 output rounding separate the kernel from this reference.
    expected = (x_values @ w_values.T).bfloat16()
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
