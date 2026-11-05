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

import torch
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)
from tokenspeed_kernel.thirdparty.deepgemm import (
    SM120_ARCH_MAJOR,
    block_fp8_mm,
    block_mxfp4_mm,
    deepgemm_supports_sm120,
)

_SIGNATURES = format_signatures(
    ("a", "b"),
    "mxfp8",
    {torch.float8_e4m3fn},
    scale=ScaleFormat(
        storage_dtype=torch.float32, granularity="block", block_shape=(128, 128)
    ),
)
_TRAITS = {
    "n_align": frozenset({64}),
    "k_align": frozenset({128}),
    "a_inner_stride_one": frozenset({True}),
    "b_inner_stride_one": frozenset({True}),
    "block_scale_layout": frozenset({"canonical"}),
    "out_dtype": frozenset({torch.bfloat16}),
    "out_inner_stride_one": frozenset({True}),
}


def _run_deepgemm_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int] | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    platform = Platform.get()
    if not (
        platform.is_hopper
        or (
            platform.is_nvidia
            and platform.arch_version.major == SM120_ARCH_MAJOR
            and deepgemm_supports_sm120()
        )
    ):
        raise ValueError(
            "DeepGEMM block FP8 requires Hopper, or SM120 with a DeepGEMM build "
            "that has SM120 kernels"
        )
    if alpha is not None or block_size is None or tuple(block_size) != (128, 128):
        raise ValueError("DeepGEMM requires 128x128 block scales and alpha=None")
    if (
        out_dtype != torch.bfloat16
        or A.dtype != torch.float8_e4m3fn
        or B.dtype != A.dtype
    ):
        raise ValueError("DeepGEMM requires E4M3 inputs and BF16 output")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous 2D A[M,K] and B[N,K]")
    m, k = A.shape
    n, kb = B.shape
    if m < 1 or n < 1 or k < 1 or k != kb or n % 64 or k % 128:
        raise ValueError(
            "DeepGEMM requires positive dimensions, matching K, N%64=0 and K%128=0"
        )
    if not A.is_cuda or B.device != A.device:
        raise ValueError("DeepGEMM operands must share a CUDA device")
    for scale, shape in (
        (A_scales, (m, k // 128)),
        (B_scales, ((n + 127) // 128, k // 128)),
    ):
        if (
            scale is None
            or scale.dtype != torch.float32
            or tuple(scale.shape) != shape
            or scale.device != A.device
        ):
            raise ValueError(
                "DeepGEMM requires canonical FP32 scales on the operand device"
            )
    if not B_scales.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous weight scales")
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "DeepGEMM out must be contiguous BF16 [M,N] on the operand device"
        )
    return block_fp8_mm(A, B, A_scales, B_scales, out)


@register_kernel(
    "gemm",
    "mm",
    name="deepgemm_mm_fp8_blockscale",
    solution="deepgemm",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=_SIGNATURES,
    traits=_TRAITS,
    # Below the existing Triton block-FP8 leaf: auto keeps its current choice.
    priority=Priority.PERFORMANT,
)
def deepgemm_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit Hopper E4M3 GEMM: [M,K] @ [N,K].T with FP32 block scales.

    Activations use 1x128 scales and weights use 128x128 scales. Output is
    BF16; a supplied contiguous output buffer is written in place. Layout
    conversion is per call and does not retain a second weight copy.
    Unsupported inputs are rejected even for a forced kernel override.
    """
    return _run_deepgemm_mm_fp8_blockscale(
        A, B, A_scales, B_scales, out_dtype, alpha=alpha, block_size=block_size, out=out
    )


# MXFP4: packed E2M1 a[M,K/2] / b[N,K/2] (uint8) with raw UE8M0 per-32 scales.
_MXFP4_SIGNATURES = format_signatures(
    ("a", "b"),
    "mxfp4",
    {torch.uint8},
    scale=ScaleFormat(
        storage_dtype=torch.uint8, granularity="block", block_shape=(32,)
    ),
)
_MXFP4_TRAITS = {
    "n_align": frozenset({64}),
    "k_align": frozenset({128}),
    "a_inner_stride_one": frozenset({True}),
    "b_inner_stride_one": frozenset({True}),
    "out_dtype": frozenset({torch.bfloat16}),
    "out_inner_stride_one": frozenset({True}),
}


def _run_deepgemm_mm_mxfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int] | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """SM120 MXFP4 GEMM: packed-E2M1 [M,K/2] @ [N,K/2].T with UE8M0 per-32 scales."""
    platform = Platform.get()
    if not (
        platform.is_nvidia
        and platform.arch_version.major == SM120_ARCH_MAJOR
        and deepgemm_supports_sm120()
    ):
        raise ValueError("DeepGEMM MXFP4 GEMM requires SM120 with SM120 kernels")
    if alpha is not None:
        raise ValueError("DeepGEMM MXFP4 GEMM does not take alpha")
    if A.dtype != torch.uint8 or B.dtype != torch.uint8:
        raise ValueError("DeepGEMM MXFP4 GEMM requires packed uint8 operands")
    if A_scales is None or B_scales is None:
        raise ValueError("DeepGEMM MXFP4 GEMM requires A and B scales")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError(
            "DeepGEMM MXFP4 GEMM requires contiguous 2D A[M,K/2], B[N,K/2]"
        )
    m, k_half = A.shape
    n, kb_half = B.shape
    k = k_half * 2
    if m < 1 or n < 1 or k_half != kb_half or n % 64 or k % 128:
        raise ValueError("DeepGEMM MXFP4 GEMM requires matching K, N%64=0 and K%128=0")
    if out_dtype != torch.bfloat16:
        raise ValueError("DeepGEMM MXFP4 GEMM produces BF16 output")
    for scale, mn in ((A_scales, m), (B_scales, n)):
        if (
            scale.dtype != torch.uint8
            or tuple(scale.shape) != (mn, k // 32)
            or scale.device != A.device
        ):
            raise ValueError(
                "DeepGEMM MXFP4 GEMM requires uint8 UE8M0 [MN, K/32] scales"
            )
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "DeepGEMM MXFP4 out must be contiguous BF16 [M,N] on the device"
        )
    return block_mxfp4_mm(A, B, A_scales, B_scales, out)


# The SM120 build reads the canonical FP32 scales directly (no transpose), and
# beats both the FlashInfer and Triton block-FP8 leaves on decode shapes while
# matching them on prefill (see the DeepGEMM fork's `deepgemm-analysis.md`).
# Its shape contract covers FlashInfer's, so FlashInfer steps back on SM120
# whenever this leaf is registered (only by DeepGEMM builds with SM120 kernels).
if current_platform().is_hopper_plus and deepgemm_supports_sm120():

    @register_kernel(
        "gemm",
        "mm",
        name="deepgemm_sm120_mm_fp8_blockscale",
        solution="deepgemm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(SM120_ARCH_MAJOR, 0),
            max_arch_version=ArchVersion(SM120_ARCH_MAJOR, 9),
        ),
        signatures=_SIGNATURES,
        # The facade labels canonical scales ``canonical_blackwell`` on SM100+
        traits=_TRAITS | {"block_scale_layout": frozenset({"canonical_blackwell"})},
        priority=Priority.SPECIALIZED,
    )
    def deepgemm_sm120_mm_fp8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SM120 E4M3 GEMM: [M,K] @ [N,K].T with FP32 1x128 / 128x128 scales.

        Same contract as ``deepgemm_mm_fp8_blockscale``.
        """
        return _run_deepgemm_mm_fp8_blockscale(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
        )

    @register_kernel(
        "gemm",
        "mm",
        name="deepgemm_sm120_mm_mxfp4",
        solution="deepgemm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(SM120_ARCH_MAJOR, 0),
            max_arch_version=ArchVersion(SM120_ARCH_MAJOR, 9),
        ),
        signatures=_MXFP4_SIGNATURES,
        traits=_MXFP4_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def deepgemm_sm120_mm_mxfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SM120 MXFP4 GEMM: [M,K] @ [N,K].T, packed E2M1 with UE8M0 per-32 scales."""
        return _run_deepgemm_mm_mxfp4(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
        )

    # Dense MXFP4 linear (dynamic MXFP4 activations), the op the MXFP4 dense layer
    # calls. Weights stay in checkpoint order ("linear" layout); the activation is
    # quantized per call, then the SM120 block-scaled GEMM runs.
    _MXFP4_LINEAR_SCALE = ScaleFormat(
        storage_dtype=torch.uint8, granularity="block", block_shape=(32,)
    )
    _MXFP4_LINEAR_SIGNATURE = format_signature(
        x=dense_tensor_format(torch.bfloat16),
        weight=tensor_format("mxfp4", torch.uint8, scale=_MXFP4_LINEAR_SCALE),
    )

    @register_kernel(
        "gemm",
        "mxfp4_linear",
        name="deepgemm_sm120_mxfp4_linear",
        solution="deepgemm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(SM120_ARCH_MAJOR, 0),
            max_arch_version=ArchVersion(SM120_ARCH_MAJOR, 9),
        ),
        signatures=frozenset({_MXFP4_LINEAR_SIGNATURE}),
        traits={"weight_layout": frozenset({"linear"})},
        priority=Priority.SPECIALIZED,
    )
    def deepgemm_sm120_mxfp4_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``mxfp4(x) @ W.T`` on SM120: quantize x to MXFP4, then the GEMM."""
        x_fp4, x_scale = mxfp4_quantize(x)
        return _run_deepgemm_mm_mxfp4(
            x_fp4,
            weight,
            x_scale,
            weight_scale,
            x.dtype,
            alpha=None,
            block_size=None,
            out=out,
        )
