# Copyright (c) 2026 LightSeek Foundation
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Kernels adapted from vLLM
# (vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py, Apache-2.0,
# Copyright contributors to the vLLM project), whose packed kernel and
# in-launch split-K reduction come from ROCm/aiter#5750 (MIT, AMD).
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
"""Scaled-MFMA MXFP8 GEMM for decode-to-medium M on gfx950 (``tl.dot_scaled``).

``C = A @ B.T`` for E4M3 ``A [M, K]`` / ``B [N, K]`` with canonical uint8
UE8M0 scales ``[M, K/32]`` / ``[N, K/32]`` (one per row and 32 K). Unlike
``decode.py`` (exact BF16 MFMA per scale group), this runs the native scaled
E4M3 MFMA, like the large-tile ``mm.py`` kernel, so it is several times
cheaper in ALU work at 32-512 rows.

Two kernels, picked per (N, K, M) from a table swept on MI350P (128 CUs):

* a tiled kernel (one output tile per program);
* a packed kernel for small M, which fills the MFMA's rows with K panels
  instead of tokens and keeps only the block-diagonal products.

Either can split K; partials are summed in the same launch by the last
program of each tile, in split order, so results are deterministic. All
splits of a tile are mapped to one XCD (program ids congruent mod 8), so the
reduction needs only CTA-scope ordering through that XCD's L2.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import tl, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

MXFP8_SCALE_GROUP = 32
# Workgroups go to XCDs round-robin by program id; 8 is a multiple of the XCD
# count of every gfx950 part/partition mode. _split_tile hardcodes it.
_XCD_STRIDE = 8
_SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}


def _block32_launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": (m + n) * k
        + (m + n) * (k // MXFP8_SCALE_GROUP)
        + m * n * args["out_ptr"].element_size(),
    }


@triton.jit
def _split_tile(N, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    pid = tl.program_id(0)
    tile = pid // (8 * SPLITS) * 8 + pid % 8
    grid_n = tl.cdiv(N, BLOCK_N)
    return tile // grid_n, tile % grid_n, tile, pid // 8 % SPLITS


@triton.jit
def _sum_splits(
    acc, out_ptrs, out_mask, slot_ptr, count_ptr, split, SPLITS: tl.constexpr
):
    """Publish this split's partial; the last arrival sums them in split order
    and re-arms the counter."""
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]
    local = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(slot_ptr + split * BM * BN + local, acc)
    # The splits share an XCD: the partial only has to reach its L2.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(count_ptr, 1, sem="acq_rel", scope="cta") == SPLITS - 1:
        total = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            total += tl.load(slot_ptr + s * BM * BN + local, cache_modifier=".cv")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=out_mask)
        tl.store(count_ptr, 0)


@triton.jit(launch_metadata=_block32_launch_metadata, do_not_specialize=["M"])
def triton_mm_mxfp8_block32_gfx950(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_xm,
    stride_xsm,
    stride_wn,
    stride_wsn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    N_FIRST: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
    else:
        tile = 0
        if N_FIRST:
            pid_n, pid_m = tl.program_id(0), tl.program_id(1)
        else:
            pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        pid_k = 0
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    # Clamp weight rows instead of masking: the store drops them.
    offs_wn = tl.minimum(offs_n, N - 1)
    k0 = pid_k * K_PER_SPLIT
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_sk = k0 // 32 + tl.arange(0, BLOCK_K // 32)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :]
    w_ptrs = w_ptr + offs_wn[:, None] * stride_wn + offs_k[None, :]
    ws_ptrs = ws_ptr + offs_wn[:, None] * stride_wsn + offs_sk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
            w = tl.load(w_ptrs)
            ws = tl.load(ws_ptrs)
        else:
            k_ok = (offs_k + kk) < K
            s_ok = (offs_sk + kk // 32) < K // 32
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_ok[None, :], other=0)
            w = tl.load(w_ptrs, mask=k_ok[None, :], other=0.0)
            ws = tl.load(ws_ptrs, mask=s_ok[None, :], other=0)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        ws_ptrs += BLOCK_K // 32
    out_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if FUSED_SPLITS > 1:
        _sum_splits(
            acc,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit(launch_metadata=_block32_launch_metadata, do_not_specialize=["M"])
def triton_mm_mxfp8_block32_packed_gfx950(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_xm,
    stride_xsm,
    stride_wn,
    stride_wsn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
        k_start = pid_k * K_PER_SPLIT
        k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        tile = 0
        pid_k = 0
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        k_start = 0
        k_stop = K
    # Tile row r is token r // K_PACK reading K panel r % K_PACK (likewise for
    # weight rows), so one dot covers K_PACK consecutive BLOCK_K slices.
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M * K_PACK) // K_PACK
    x_panel = tl.arange(0, BLOCK_M * K_PACK) % K_PACK
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * K_PACK) // K_PACK
    cols = tl.minimum(cols, N - 1)
    w_panel = tl.arange(0, BLOCK_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 32)
    row_ok = rows < M
    acc = tl.zeros((BLOCK_M * K_PACK, BLOCK_N * K_PACK), dtype=tl.float32)
    for base in range(k_start, k_stop, BLOCK_K * K_PACK):
        xk = base + x_panel[:, None] * BLOCK_K + ks[None, :]
        wk = base + w_panel[:, None] * BLOCK_K + ks[None, :]
        xg = base // 32 + x_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        wg = base // 32 + w_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        if EVEN_K:
            x = tl.load(x_ptr + rows[:, None] * stride_xm + xk, mask=row_ok[:, None])
            xs = tl.load(xs_ptr + rows[:, None] * stride_xsm + xg, mask=row_ok[:, None])
            w = tl.load(w_ptr + cols[:, None] * stride_wn + wk)
            ws = tl.load(ws_ptr + cols[:, None] * stride_wsn + wg)
        else:
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + xk,
                mask=row_ok[:, None] & (xk < K),
                other=0.0,
            )
            xs = tl.load(
                xs_ptr + rows[:, None] * stride_xsm + xg,
                mask=row_ok[:, None] & (xg < K // 32),
                other=0,
            )
            w = tl.load(w_ptr + cols[:, None] * stride_wn + wk, mask=wk < K, other=0.0)
            ws = tl.load(
                ws_ptr + cols[:, None] * stride_wsn + wg, mask=wg < K // 32, other=0
            )
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
    # Keep only products of matching K panels: the block diagonal.
    blocks = acc.reshape(BLOCK_M, K_PACK, BLOCK_N, K_PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], blocks, 0.0
    )
    y = tl.sum(tl.sum(diagonal, 3), 2)
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + out_rows[:, None] * stride_om + out_cols[None, :]
    out_mask = (out_rows[:, None] < M) & (out_cols[None, :] < N)
    if FUSED_SPLITS > 1:
        _sum_splits(
            y,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=out_mask)


class Block32Config(NamedTuple):
    packed: bool
    block_m: int
    block_n: int
    block_k: int
    # K panels per MFMA tile (packed only; 1 for tiled).
    k_pack: int
    n_first: bool
    num_warps: int
    num_stages: int
    matrix_instr_nonkdim: int
    waves_per_eu: int
    # K partitions summed within the launch.
    splits: int = 1


def _tiled(bm, bn, bk, n_first, warps, stages, nonkdim, waves, splits=1):
    return Block32Config(
        False, bm, bn, bk, 1, bool(n_first), warps, stages, nonkdim, waves, splits
    )


def _packed(bm, bn, bk, k_pack, warps, stages, nonkdim, waves, splits=1):
    return Block32Config(
        True, bm, bn, bk, k_pack, False, warps, stages, nonkdim, waves, splits
    )


# (N, K) -> [(max M, config), ...] for the DeepSeek V4.1 TP4 dense MXFP8
# projections, swept on MI350P (128 CUs, 4 XCDs) with graph replay and weights
# rotated through more than the MALL. A tier serves every M up to its bound.
# Only shapes and M ranges where this kernel beats decode.py/mm.py are listed;
# mm() keeps the existing kernels elsewhere.
# Fields: packed, BLOCK_M, BLOCK_N, BLOCK_K, K_PACK, N_FIRST, warps, stages,
# MFMA non-K dim, waves_per_eu, K splits.
BLOCK32_CONFIGS: dict[tuple[int, int], list[tuple[int, Block32Config]]] = {
    # wq_a|wkv
    (1792, 5120): [
        (16, Block32Config(True, 16, 32, 256, 2, False, 4, 2, 16, 0, 2)),
        (48, Block32Config(False, 64, 32, 512, 1, True, 8, 2, 16, 1, 2)),
        (96, Block32Config(False, 64, 64, 512, 1, True, 8, 2, 16, 1, 2)),
        (192, Block32Config(False, 64, 64, 512, 1, True, 4, 2, 32, 2, 1)),
        (384, Block32Config(False, 64, 64, 256, 1, True, 4, 2, 32, 2, 1)),
        (768, Block32Config(False, 128, 128, 256, 1, True, 8, 2, 32, 1, 1)),
    ],
    # wq_b
    (8192, 1280): [
        (16, Block32Config(True, 16, 64, 128, 2, False, 4, 2, 16, 1, 1)),
        (48, Block32Config(False, 64, 64, 256, 1, True, 8, 2, 16, 2, 1)),
        (96, Block32Config(False, 128, 64, 256, 1, True, 8, 3, 32, 0, 1)),
        (192, Block32Config(False, 128, 128, 256, 1, True, 8, 2, 32, 1, 1)),
        (384, Block32Config(False, 128, 128, 128, 1, True, 8, 2, 32, 0, 1)),
    ],
    # wo_b
    (5120, 2048): [
        (16, Block32Config(True, 16, 16, 128, 4, False, 4, 2, 32, 0, 1)),
        (48, Block32Config(False, 64, 64, 512, 1, True, 8, 2, 16, 0, 1)),
        (96, Block32Config(False, 64, 64, 256, 1, True, 4, 2, 32, 2, 1)),
        (192, Block32Config(False, 64, 64, 256, 1, True, 4, 2, 32, 2, 1)),
        (384, Block32Config(False, 128, 128, 256, 1, True, 8, 2, 32, 1, 1)),
        (768, Block32Config(False, 128, 128, 128, 1, True, 8, 2, 32, 0, 1)),
    ],
    # shared expert gate_up
    (1152, 5120): [
        (16, Block32Config(True, 16, 16, 256, 4, False, 4, 2, 16, 0, 4)),
        (48, Block32Config(False, 64, 64, 512, 1, True, 8, 2, 16, 0, 8)),
        (96, Block32Config(False, 32, 32, 512, 1, True, 4, 2, 16, 0, 2)),
        (192, Block32Config(False, 64, 64, 512, 1, True, 4, 2, 16, 2, 2)),
        (384, Block32Config(False, 64, 64, 512, 1, True, 4, 2, 32, 2, 1)),
        (768, Block32Config(False, 64, 64, 256, 1, True, 4, 2, 32, 0, 1)),
    ],
    # shared expert down
    (5120, 576): [
        (48, Block32Config(False, 16, 64, 256, 1, True, 2, 3, 16, 2, 1)),
        (96, Block32Config(False, 32, 64, 256, 1, True, 4, 2, 16, 0, 1)),
        (192, Block32Config(False, 64, 128, 128, 1, True, 4, 3, 32, 2, 1)),
        (384, Block32Config(False, 128, 128, 128, 1, True, 8, 2, 32, 2, 1)),
        (768, Block32Config(False, 128, 128, 128, 1, True, 8, 2, 32, 1, 1)),
    ],
    # indexer wq_b
    (4096, 1280): [
        (16, Block32Config(True, 16, 32, 256, 2, False, 4, 3, 16, 1, 1)),
        (48, Block32Config(False, 32, 64, 256, 1, True, 4, 2, 16, 2, 1)),
        (96, Block32Config(False, 128, 32, 256, 1, True, 8, 3, 16, 1, 1)),
        (192, Block32Config(False, 128, 64, 256, 1, True, 8, 3, 32, 1, 1)),
        (384, Block32Config(False, 128, 128, 256, 1, True, 8, 2, 32, 0, 1)),
        (768, Block32Config(False, 128, 128, 128, 1, True, 8, 2, 32, 0, 1)),
    ],
}


def _lookup_config(m: int, n: int, k: int) -> Block32Config | None:
    for max_m, cfg in BLOCK32_CONFIGS.get((n, k), ()):
        if m <= max_m:
            return cfg
    return None


def _k_partition(k: int, splits: int, step: int) -> tuple[int, int]:
    size = triton.cdiv(triton.cdiv(k, splits), step) * step
    return size, triton.cdiv(k, size)


def _plan(m: int, n: int, k: int, cfg: Block32Config) -> tuple[int, int, int]:
    """(tiles, k_per_split, splits) of one launch."""
    tiles = triton.cdiv(m, cfg.block_m) * triton.cdiv(n, cfg.block_n)
    step = cfg.block_k * cfg.k_pack
    if cfg.splits > 1:
        k_per_split, splits = _k_partition(k, cfg.splits, step)
    else:
        k_per_split, splits = k, 1
    return tiles, k_per_split, splits


def supports_triton_mm_mxfp8_block32_gfx950(m: int, n: int, k: int) -> bool:
    """Whether a tuned block32 config covers an ``[m, k] x [n, k]`` problem."""
    return m >= 1 and k % MXFP8_SCALE_GROUP == 0 and _lookup_config(m, n, k) is not None


def triton_mm_mxfp8_block32_num_counters_gfx950(
    m: int, n: int, k: int, config: Block32Config | None = None
) -> int:
    """Split-K arrival counters a launch needs; 0 without split-K."""
    cfg = config or _lookup_config(m, n, k)
    if cfg is None:
        return 0
    tiles, _, splits = _plan(m, n, k, cfg)
    return tiles if splits > 1 else 0


def _launch(A, B, A_scales, B_scales, C, cfg: Block32Config, counters):
    M, K = A.shape
    N = B.shape[0]
    tiles, k_per_split, splits = _plan(M, N, K, cfg)
    opts = dict(
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        matrix_instr_nonkdim=cfg.matrix_instr_nonkdim,
        waves_per_eu=cfg.waves_per_eu,
    )
    if splits > 1:
        check_splitk_counters(counters, tiles, A.device, "gfx950 MXFP8 block32 GEMM")
        slots = torch.empty(
            tiles * splits * cfg.block_m * cfg.block_n,
            dtype=torch.float32,
            device=A.device,
        )
        grid = (triton.cdiv(tiles, _XCD_STRIDE) * _XCD_STRIDE * splits,)
    else:
        slots = counters = C
        grid_m, grid_n = triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n)
        grid = (
            (grid_n, grid_m) if (cfg.n_first and not cfg.packed) else (grid_m, grid_n)
        )
    args = (
        A,
        A_scales,
        B,
        B_scales,
        C,
        slots,
        counters,
        M,
        N,
        K,
        A.stride(0),
        A_scales.stride(0),
        B.stride(0),
        B_scales.stride(0),
        C.stride(0),
    )
    if cfg.packed:
        triton_mm_mxfp8_block32_packed_gfx950[grid](
            *args,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            K_PACK=cfg.k_pack,
            K_PER_SPLIT=k_per_split,
            EVEN_K=K % (cfg.block_k * cfg.k_pack) == 0,
            FUSED_SPLITS=splits,
            **opts,
        )
    else:
        triton_mm_mxfp8_block32_gfx950[grid](
            *args,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            K_PER_SPLIT=k_per_split,
            N_FIRST=cfg.n_first,
            EVEN_K=k_per_split % cfg.block_k == 0 and K % k_per_split == 0,
            FUSED_SPLITS=splits,
            **opts,
        )
    return C


def _validate_scale(name: str, scale: torch.Tensor, rows: int, k: int) -> None:
    expected = (rows, k // MXFP8_SCALE_GROUP)
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
        raise ValueError(
            f"{name} must be uint8 E8M0 with shape {expected}, "
            f"got dtype={scale.dtype}, shape={tuple(scale.shape)}"
        )
    if scale.stride(1) != 1:
        raise ValueError(f"{name} must have unit inner stride")


def launch_triton_mm_mxfp8_block32_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int],
    counters: torch.Tensor,
    out: torch.Tensor | None,
    config: Block32Config | None = None,
) -> torch.Tensor:
    """Compute MXFP8 ``A @ B.T`` with the scaled-MFMA block32 kernels.

    Args:
        A: E4M3 activation ``[M, K]`` with unit inner stride.
        B: E4M3 weight ``[N, K]`` with unit inner stride; ``K % 32 == 0``.
        A_scales: uint8 UE8M0 scales ``[M, K/32]``, unit inner stride.
        B_scales: uint8 UE8M0 scales ``[N, K/32]``, unit inner stride.
        out_dtype: BF16 or FP16.
        alpha: Optional output multiplier applied after the GEMM.
        block_size: Must be ``[1, 32]``.
        counters: Zeroed int32 split-K counters, at least
            ``triton_mm_mxfp8_block32_num_counters_gfx950(M, N, K)``; left
            zeroed.
        out: Optional ``[M, N]`` output with unit inner stride.
        config: Explicit tile config (tuning); default is the tuned table.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("gfx950 MXFP8 block32 GEMM requires rank-2 A and B")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("gfx950 MXFP8 block32 GEMM requires E4M3 A and B")
    if list(block_size) != [1, MXFP8_SCALE_GROUP]:
        raise ValueError("gfx950 MXFP8 block32 GEMM requires block_size=[1, 32]")
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError(f"gfx950 MXFP8 block32 GEMM does not support {out_dtype}")
    M, K = A.shape
    N = B.shape[0]
    if B.shape[1] != K or K % MXFP8_SCALE_GROUP:
        raise ValueError(f"gfx950 MXFP8 block32 GEMM bad K: A={K}, B={B.shape[1]}")
    if A.stride(1) != 1 or B.stride(1) != 1:
        raise ValueError("gfx950 MXFP8 block32 GEMM requires unit inner strides")
    _validate_scale("A_scales", A_scales, M, K)
    _validate_scale("B_scales", B_scales, N, K)
    cfg = config or _lookup_config(M, N, K)
    if cfg is None:
        raise ValueError(f"gfx950 MXFP8 block32 GEMM has no config for {M}x{N}x{K}")
    if out is None:
        out = torch.empty((M, N), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (M, N)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(1) != 1
    ):
        raise ValueError("gfx950 MXFP8 block32 out must match [M, N]/dtype/device")
    if M:
        _launch(A, B, A_scales, B_scales, out, cfg, counters)
    if alpha is not None:
        out.mul_(alpha.to(device=out.device, dtype=out.dtype))
    return out


__all__ = [
    "BLOCK32_CONFIGS",
    "Block32Config",
    "triton_mm_mxfp8_block32_num_counters_gfx950",
    "launch_triton_mm_mxfp8_block32_gfx950",
    "supports_triton_mm_mxfp8_block32_gfx950",
]
