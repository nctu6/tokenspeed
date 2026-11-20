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
from tokenspeed_kernel.platform import pdl_enabled, prepare_cuda_toolkit_env

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


@functools.cache
def deepgemm_has_deterministic_query() -> bool:
    """Whether the installed DeepGEMM reports its deterministic-algorithms mode.

    Builds that have it also keep SM120 block-scaled GEMMs batch invariant in
    that mode (no stream-K split and no K-warp split, whose summation order
    follows the batch).
    """
    if not deepgemm_supports_sm120():
        return False
    import deep_gemm

    return "get_deterministic_algorithms" in dir(deep_gemm)


@functools.cache
def deepgemm_sm120_mega_moe_max_ranks() -> int:
    """Most ranks the installed DeepGEMM's SM120 ``fp8_fp4_mega_moe`` serves (0: none).

    SM120 has no NVLink and no TMEM, so DeepGEMM runs MegaMoE there as a sequence of
    kernels whose ranks write into each other's symmetric buffers with CUDA IPC peer
    stores over PCIe. Builds with only the SM100 kernel, or with the one-rank SM120
    version, have no ``get_mega_moe_max_ranks`` and report 0.
    """
    if not deepgemm_supports_sm120():
        return 0
    import deep_gemm

    try:
        return int(deep_gemm.get_mega_moe_max_ranks(SM120_ARCH_MAJOR))
    except AttributeError:
        return 0


def prepare_block_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize E4M3/FP32 block weights and pack UE8M0 scales once.

    Args:
        weight: E4M3 [N,K] weights, with K divisible by 128.
        scales: Canonical FP32 [ceil(N/128),K/128] dequantization scales.

    Returns:
        Requantized E4M3 weights and DeepGEMM int32-packed per-row scales.
        Superseded weight/scale allocations are not retained.
    """
    import deep_gemm

    n, k = weight.shape
    padded_n = (n + 127) // 128 * 128
    dequantized = torch.zeros((padded_n, k), device=weight.device, dtype=torch.float32)
    dequantized[:n] = weight.float() * scales.repeat_interleave(128, 0)[
        :n
    ].repeat_interleave(128, 1)
    blocks = dequantized.view(padded_n // 128, 128, k // 128, 128)
    raw_scales = blocks.abs().amax((1, 3)).clamp_min(1e-10) / 448.0
    bits = raw_scales.view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).clamp(1, 254)
    ue8m0_scales = (exponent << 23).view(torch.float32)
    quantized = (blocks / ue8m0_scales[:, None, :, None]).clamp(-448, 448)
    quantized = quantized.to(torch.float8_e4m3fn).view(padded_n, k)[:n].contiguous()
    packed = deep_gemm.transform_sf_into_required_layout(ue8m0_scales, n, k, (128, 128))
    return quantized, packed


def block_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    enable_pdl: bool,
) -> torch.Tensor:
    """Run Hopper FP32 scales or already-packed Blackwell UE8M0 scales."""
    import deep_gemm

    # DeepGEMM's launch mode is process-wide. Keep it aligned with the caller's
    # PDL setting (the engine's, as the DSA / MoE leaves do), so dense GEMMs
    # overlap their prologue with the previous kernel instead of breaking the
    # PDL chain. Hopper, SM100/SM103 and SM120 all go through this.
    if deep_gemm.get_pdl() != enable_pdl:
        deep_gemm.set_pdl(enable_pdl)

    if a_scale.dtype == torch.int32:
        deep_gemm.fp8_gemm_nt(
            (a, a_scale),
            (b, b_scale),
            out,
            recipe_a=(1, 128),
            recipe_b=(1, 128),
        )
        return out

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


def pack_ue8m0_scales(sf: torch.Tensor) -> torch.Tensor:
    """Pack raw UE8M0 scale bytes ``[MN, K/32]`` for DeepGEMM's MXFP4 GEMM.

    DeepGEMM reads four consecutive K-group bytes as one int32 word, MN-major,
    with MN aligned up to a multiple of 4. The bytes are already UE8M0, so this
    is a transposing copy.

    Returns:
        int32 ``[MN, K/128]`` with strides ``(1, align4(MN))``.
    """
    mn, groups = sf.shape
    if groups % 4 != 0:
        raise ValueError(f"UE8M0 scales need K divisible by 128, got {groups} groups")
    aligned_mn = (mn + 3) // 4 * 4
    packed = torch.zeros((groups // 4, aligned_mn), dtype=torch.int32, device=sf.device)
    packed[:, :mn] = sf.contiguous().view(torch.int32).T
    return packed.T[:mn]


def block_mxfp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """MXFP4 GEMM d = a @ b.T on SM120.

    Args:
        a: Packed E2M1 uint8 ``[M, K/2]``.
        b: Packed E2M1 uint8 ``[N, K/2]``.
        a_scale: UE8M0 scales ``[M, K/128]`` int32 packed as ``pack_ue8m0_scales``.
        b_scale: UE8M0 scales ``[N, K/128]`` int32 packed as ``pack_ue8m0_scales``.
        out: BF16 ``[M, N]`` destination.
    """
    import deep_gemm

    if deep_gemm.get_pdl() != pdl_enabled():
        deep_gemm.set_pdl(pdl_enabled())

    # DeepGEMM tags packed FP4 operands as int8 (kPackedFP4 == torch.int8); the
    # MXFP4 quantizer emits uint8.
    deep_gemm.fp8_fp4_gemm_nt(
        (a.view(torch.int8), a_scale),
        (b.view(torch.int8), b_scale),
        out,
        recipe=(1, 1, 32),
    )
    return out


def block_mxfp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """MXFP8 GEMM d = a @ b.T on SM120 (E4M3 operands, UE8M0 scale per 32 K).

    Batch invariant: a row of ``out`` depends only on its row of ``a``.

    Args:
        a: E4M3 ``[M, K]``.
        b: E4M3 ``[N, K]``.
        a_scale: UE8M0 scales ``[M, K/128]`` int32 packed as ``pack_ue8m0_scales``.
        b_scale: UE8M0 scales ``[N, K/128]`` int32 packed as ``pack_ue8m0_scales``.
        out: BF16 or FP32 ``[M, N]`` destination.
    """
    import deep_gemm

    if deep_gemm.get_pdl() != pdl_enabled():
        deep_gemm.set_pdl(pdl_enabled())

    # Run with deterministic algorithms, which keep each output a single
    # in-order K loop: a row's bits then do not depend on the batch. The mode
    # is process-wide, so restore the caller's choice afterwards.
    deterministic = deep_gemm.get_deterministic_algorithms()
    if not deterministic:
        deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.fp8_fp4_gemm_nt((a, a_scale), (b, b_scale), out, recipe=(1, 1, 32))
    finally:
        if not deterministic:
            deep_gemm.use_deterministic_algorithms(False)
    return out


def grouped_fp8_weight_scales(scales: torch.Tensor, groups: int) -> torch.Tensor:
    """Pack 1x32 UE8M0 weight scales ``[groups * n, k / 32]`` for ``grouped_fp8_mm``."""
    import deep_gemm

    rows, blocks = scales.shape
    exponents = scales.to(torch.int32) << 23
    return deep_gemm.transform_sf_into_required_layout(
        exponents.view(torch.float32).view(groups, rows // groups, blocks),
        rows // groups,
        blocks * 32,
        (1, 1, 32),
        num_groups=groups,
        is_sfa=False,
    )


def grouped_fp8_mm(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor | None,
) -> torch.Tensor:
    """``einsum('tgk,gnk->tgn')`` of E4M3 operands with 1x32 UE8M0 scales.

    With ``out_scales``, ``out`` receives E4M3 codes of the BF16-rounded result
    and ``out_scales`` their packed 1x32 UE8M0 scales over ``out.flatten(1)``.
    """
    import deep_gemm

    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (x, x_scales),
        (weight, weight_scales),
        out if out_scales is None else (out, out_scales),
        recipe=(1, 1, 32),
    )
    return out


def fp8_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack 1x32 UE8M0 weight scales ``[n, k / 32]`` for ``fp8_mm``."""
    import deep_gemm

    rows, blocks = scales.shape
    exponents = scales.to(torch.int32) << 23
    return deep_gemm.transform_sf_into_required_layout(
        exponents.view(torch.float32), rows, blocks * 32, (1, 1, 32), is_sfa=False
    )


def fp8_mm(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``x @ weight.T`` of E4M3 operands with 1x32 UE8M0 scales into BF16 ``out``."""
    import deep_gemm

    deep_gemm.fp8_gemm_nt(
        (x, x_scales), (weight, weight_scales), out, recipe=(1, 1, 32)
    )
    return out
