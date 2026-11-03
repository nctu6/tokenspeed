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

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel.platform import prepare_cuda_toolkit_env

# Compute capability major of workstation Blackwell (RTX PRO 6000 / RTX 50).
SM120_ARCH_MAJOR = 12


@functools.cache
def deepgemm_supports_sm120() -> bool:
    """Whether the installed DeepGEMM has SM120 FP8 GEMM and MQA-logits kernels.

    Builds with SM120 kernels report it through ``get_supported_arch_majors``;
    older builds have no such API and fail SM120 calls at runtime with
    "Unsupported architecture", so they must not be selected there.
    """
    prepare_cuda_toolkit_env()
    try:
        import deep_gemm
    except ImportError:
        return False
    try:
        supported = deep_gemm.get_supported_arch_majors()
    except AttributeError:
        return False
    return SM120_ARCH_MAJOR in supported


def block_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Adapt canonical FP32 scales without rounding them to UE8M0."""
    import deep_gemm

    m, k = a.shape
    n = b.shape[0]
    sa = deep_gemm.transform_sf_into_required_layout(
        a_scale, m, k, (1, 128), disable_ue8m0_cast=True
    )
    sb = deep_gemm.transform_sf_into_required_layout(
        b_scale, n, k, (128, 128), disable_ue8m0_cast=True
    )
    deep_gemm.fp8_gemm_nt(
        (a, sa),
        (b, sb),
        out,
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
    )
    return out
