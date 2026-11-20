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

"""Prepacked FP8 1x128 activation quantization on every supported GPU."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm._flashinfer.fp8_quant import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.mark.parametrize("m", [1, 3, 4, 8, 128, 130, 256])
@pytest.mark.parametrize("k", [128, 1024])
def test_prepacked_quantization_reconstructs_the_input(m: int, k: int) -> None:
    """Rows divisible by four must not take a path the GPU has no image for.

    The TRT-LLM quantizer used for those rows has no SM120 image. Its failed
    launch is not raised, so a wrong result is the only symptom.
    """
    torch.manual_seed(m * k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[:, :128] = 0

    q, scales = flashinfer_fp8_blockscale_quantize_prepacked(x, 128)

    padded = (m + 3) // 4 * 4
    assert q.shape == (padded, k) and scales.shape == (k // 128, padded)
    per_group = scales[:, :m].t().repeat_interleave(128, dim=1)
    reconstructed = q[:m].float() * per_group
    # E4M3 keeps three mantissa bits: a relative error of at most 1/16.
    torch.testing.assert_close(reconstructed, x.float(), rtol=0.07, atol=1e-4)
    if padded != m:
        assert torch.count_nonzero(q[m:]).item() == 0
        assert torch.equal(scales[:, m:], torch.ones_like(scales[:, m:]))
