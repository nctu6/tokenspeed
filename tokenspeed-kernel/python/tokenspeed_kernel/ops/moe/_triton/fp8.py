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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton.common import (
    _combine,
    _num_programs,
    _prepare_routed_output,
    _swiglu,
    _swiglu_params,
    _validate_launch,
    _validate_topk,
)

_FP8_BLOCK = 128
# Up to this many routed rows (tokens x top-k) the route-parallel kernels run;
# above it the grouped kernels, which share a weight read between the rows of
# an expert. Measured crossover: 64 routes (8 decode tokens at top-8) is 1.08x,
# 128 routes ties (ops/moe/README.md, "Triton FP8 MoE for few tokens").
_ROUTE_PARALLEL_MAX_ROUTES = 64
_ROUTE_STAGE1_BLOCK_N = 32
_ROUTE_STAGE2_BLOCK_N = 64


def _validate_fp8_moe(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_launch(w, topk_weights, topk_ids, do_finalize)
    if any(
        getattr(w, name, None) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Triton MoE does not support expert bias")

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Triton FP8 MoE does not support activation {activation!r}")
    limit = getattr(getattr(w, "swiglu_arg", None), "limit", None)
    if limit is not None and limit <= 0:
        raise ValueError("SwiGLU limit must be positive")
    if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
        raise ValueError("Triton MoE requires concatenated gate/up weights")

    w13 = w.w13_weight
    w2 = w.w2_weight
    w13_scale = w.w13_weight_scale_inv
    w2_scale = w.w2_weight_scale_inv
    weights = (w13, w2, w13_scale, w2_scale)
    if x.ndim != 2 or any(t.ndim != 3 for t in weights):
        raise ValueError("x and block-FP8 MoE weights must be rank-2/rank-3")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use torch.bfloat16")
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError("w13_weight and w2_weight must use torch.float8_e4m3fn")
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("block-FP8 inverse scales must use torch.float32")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, *weights)):
        raise ValueError("x, weights, and scales must be contiguous GPU tensors")
    _validate_topk(x, topk_weights, topk_ids)
    if any(t.device != x.device for t in weights):
        raise ValueError("x and weights must be on the same device")

    hidden_size = x.shape[1]
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    if num_experts == 0:
        raise ValueError("block-FP8 MoE requires at least one expert")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    if hidden_size % _FP8_BLOCK or intermediate_size % _FP8_BLOCK:
        raise ValueError(
            f"hidden and intermediate sizes must be multiples of {_FP8_BLOCK}"
        )
    hidden_blocks = hidden_size // _FP8_BLOCK
    intermediate_blocks = intermediate_size // _FP8_BLOCK
    if w13_scale.shape != (num_experts, 2 * intermediate_blocks, hidden_blocks):
        raise ValueError("w13_weight_scale_inv has an incompatible shape")
    if w2_scale.shape != (num_experts, hidden_blocks, intermediate_blocks):
        raise ValueError("w2_weight_scale_inv has an incompatible shape")
    return topk_weights, topk_ids


@triton.jit
def _stage1_kernel(
    x_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route_count = num_tokens * top_k
    scale_k = hidden_size // SCALE_BLOCK
    up_scale_offset = intermediate_size // SCALE_BLOCK * scale_k
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(intermediate_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w13_ptr + expert_id.to(tl.int64) * (
            2 * intermediate_size * hidden_size
        )
        expert_scale = w13_scale_ptr + expert_id * 2 * up_scale_offset

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            token_ids = tl.where(row_mask, route_ids // top_k, 0).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            gate_rows = n_offset + tl.arange(0, BLOCK_N)
            up_rows = intermediate_size + gate_rows
            gate_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
            up_scale = gate_scale + up_scale_offset
            gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, hidden_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + token_ids[:, None] * hidden_size + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                gate = tl.load(
                    expert_weight + gate_rows[:, None] * hidden_size + k_cols[None, :]
                ).to(tl.bfloat16)
                up = tl.load(
                    expert_weight + up_rows[:, None] * hidden_size + k_cols[None, :]
                ).to(tl.bfloat16)
                scale_col = k_offset // SCALE_BLOCK
                gate_acc += tl.dot(x, gate.T) * tl.load(gate_scale + scale_col)
                up_acc += tl.dot(x, up.T) * tl.load(up_scale + scale_col)

            activated = _swiglu(
                gate_acc,
                up_acc,
                swiglu_alpha,
                swiglu_limit,
                swiglu_beta,
                HAS_LIMIT,
            ).to(tl.bfloat16)
            inter_offsets = route_ids[:, None] * intermediate_size + gate_rows[None, :]
            tl.store(inter_ptr + inter_offsets, activated, mask=row_mask[:, None])
            tile_idx += num_programs

        problem_start += problem_tiles


@triton.jit
def _stage2_kernel(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route_count = num_tokens * top_k
    scale_k = intermediate_size // SCALE_BLOCK
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(hidden_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w2_ptr + expert_id.to(tl.int64) * (
            hidden_size * intermediate_size
        )
        expert_scale = w2_scale_ptr + expert_id * (hidden_size // SCALE_BLOCK * scale_k)

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            route_ids = tl.where(row_mask, route_ids, -1).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            weight_rows = n_offset + tl.arange(0, BLOCK_N)
            weight_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, intermediate_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                intermediate = tl.load(
                    inter_ptr
                    + route_ids[:, None] * intermediate_size
                    + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                weight = tl.load(
                    expert_weight
                    + weight_rows[:, None] * intermediate_size
                    + k_cols[None, :]
                ).to(tl.bfloat16)
                acc += tl.dot(intermediate, weight.T) * tl.load(
                    weight_scale + k_offset // SCALE_BLOCK
                )

            output_offsets = route_ids[:, None] * hidden_size + weight_rows[None, :]
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += num_programs

        problem_start += problem_tiles


@triton.jit
def _stage1_route_kernel(
    x_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_ptr,
    topk_ids_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 1 of one routed row: program (route, column tile), no expert scan.

    The math is `_stage1_kernel` with one valid row in a BLOCK_M tile, so the
    result is bit-identical."""
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route = tl.program_id(0)
    tile_n = tl.program_id(1)
    expert_id = tl.load(topk_ids_ptr + route).to(tl.int32)
    # Ids outside [0, num_experts) are not routed: their rows stay untouched.
    if (expert_id >= 0) & (expert_id < num_experts):
        token = route // top_k
        scale_k = hidden_size // SCALE_BLOCK
        up_scale_offset = intermediate_size // SCALE_BLOCK * scale_k
        expert_weight = w13_ptr + expert_id.to(tl.int64) * (
            2 * intermediate_size * hidden_size
        )
        expert_scale = w13_scale_ptr + expert_id * 2 * up_scale_offset
        row_mask = tl.arange(0, BLOCK_M) < 1
        n_offset = tile_n * BLOCK_N
        gate_rows = n_offset + tl.arange(0, BLOCK_N)
        up_rows = intermediate_size + gate_rows
        gate_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
        up_scale = gate_scale + up_scale_offset
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_offset in range(0, hidden_size, BLOCK_K):
            k_cols = k_offset + tl.arange(0, BLOCK_K)
            x = tl.load(
                tl.broadcast_to(
                    x_ptr + token.to(tl.int64) * hidden_size + k_cols[None, :],
                    (BLOCK_M, BLOCK_K),
                ),
                mask=row_mask[:, None],
                other=0.0,
            )
            gate = tl.load(
                expert_weight + gate_rows[:, None] * hidden_size + k_cols[None, :]
            ).to(tl.bfloat16)
            up = tl.load(
                expert_weight + up_rows[:, None] * hidden_size + k_cols[None, :]
            ).to(tl.bfloat16)
            scale_col = k_offset // SCALE_BLOCK
            gate_acc += tl.dot(x, gate.T) * tl.load(gate_scale + scale_col)
            up_acc += tl.dot(x, up.T) * tl.load(up_scale + scale_col)
        activated = _swiglu(
            gate_acc, up_acc, swiglu_alpha, swiglu_limit, swiglu_beta, HAS_LIMIT
        ).to(tl.bfloat16)
        inter_offsets = tl.broadcast_to(
            route.to(tl.int64) * intermediate_size + gate_rows[None, :],
            (BLOCK_M, BLOCK_N),
        )
        tl.store(inter_ptr + inter_offsets, activated, mask=row_mask[:, None])


@triton.jit
def _stage2_route_kernel(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    route_output_ptr,
    topk_ids_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 2 of one routed row (program per route and column tile); zero rows for unrouted ids."""
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route = tl.program_id(0)
    tile_n = tl.program_id(1)
    expert_id = tl.load(topk_ids_ptr + route).to(tl.int32)
    row_mask = tl.arange(0, BLOCK_M) < 1
    n_offset = tile_n * BLOCK_N
    weight_rows = n_offset + tl.arange(0, BLOCK_N)
    output_offsets = tl.broadcast_to(
        route.to(tl.int64) * hidden_size + weight_rows[None, :], (BLOCK_M, BLOCK_N)
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if (expert_id >= 0) & (expert_id < num_experts):
        scale_k = intermediate_size // SCALE_BLOCK
        expert_weight = w2_ptr + expert_id.to(tl.int64) * (
            hidden_size * intermediate_size
        )
        expert_scale = w2_scale_ptr + expert_id * (hidden_size // SCALE_BLOCK * scale_k)
        weight_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
        for k_offset in range(0, intermediate_size, BLOCK_K):
            k_cols = k_offset + tl.arange(0, BLOCK_K)
            intermediate = tl.load(
                tl.broadcast_to(
                    inter_ptr
                    + route.to(tl.int64) * intermediate_size
                    + k_cols[None, :],
                    (BLOCK_M, BLOCK_K),
                ),
                mask=row_mask[:, None],
                other=0.0,
            )
            weight = tl.load(
                expert_weight
                + weight_rows[:, None] * intermediate_size
                + k_cols[None, :]
            ).to(tl.bfloat16)
            acc += tl.dot(intermediate, weight.T) * tl.load(
                weight_scale + k_offset // SCALE_BLOCK
            )
    tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])


def _route_parallel_fp8_moe(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Block-FP8 MoE for few routed rows: one program per (route, column tile).

    The grouped kernels give each program a stripe of experts and make it scan
    all `num_experts` counts; with a handful of routed rows that scan and the
    few active programs (a stage has `routes * columns / BLOCK_N` tiles) leave
    the GPU idle. Here every route reads its own expert's weight columns, so
    one decode token spreads over 128 + 48 programs and needs no routing
    pass. Same math as `_triton_fp8_moe_impl`, bit-identical output."""
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w.w13_weight.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    route_count = num_tokens * top_k
    swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_params(w)
    intermediate = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=x.dtype
    )
    route_output = torch.empty(
        (route_count, hidden_size), device=x.device, dtype=x.dtype
    )
    output = torch.empty_like(x)
    _stage1_route_kernel[(route_count, intermediate_size // _ROUTE_STAGE1_BLOCK_N)](
        x,
        w.w13_weight,
        w.w13_weight_scale_inv,
        intermediate,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=1.0 if swiglu_limit is None else swiglu_limit,
        swiglu_beta=swiglu_beta,
        HAS_LIMIT=swiglu_limit is not None,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=16,
        BLOCK_N=_ROUTE_STAGE1_BLOCK_N,
        BLOCK_K=_FP8_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _stage2_route_kernel[(route_count, hidden_size // _ROUTE_STAGE2_BLOCK_N)](
        intermediate,
        w.w2_weight,
        w.w2_weight_scale_inv,
        route_output,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=16,
        BLOCK_N=_ROUTE_STAGE2_BLOCK_N,
        BLOCK_K=_FP8_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _combine(route_output, topk_weights, output, None, 0)
    return output


def _triton_fp8_moe_impl(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if x.shape[0] * topk_ids.shape[1] <= _ROUTE_PARALLEL_MAX_ROUTES:
        return _route_parallel_fp8_moe(x, w, topk_weights, topk_ids)
    return _grouped_fp8_moe(x, w, topk_weights, topk_ids)


def _grouped_fp8_moe(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Block-FP8 MoE with the routed rows grouped by expert (two persistent kernels)."""
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w.w13_weight.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_params(w)
    expert_route_ids, expert_counts, route_output, output = _prepare_routed_output(
        x, topk_ids, num_experts
    )
    route_count = num_tokens * top_k
    intermediate = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=x.dtype
    )
    # Measured tiles (ops/moe/README.md, "Triton FP8 MoE tiles"): 128 weight
    # columns with 8 warps beat 32 columns from 1 to 8192 tokens. The row tile
    # follows the mean routed rows per expert, the rows one tile of an expert
    # covers: 16 rows up to 16 rows per expert, else 32.
    block_m = 16 if route_count <= 16 * num_experts else 32
    block_n = 128
    num_warps = 8
    stage1_programs = _num_programs(x.device, route_count, intermediate_size, block_n)
    stage2_programs = _num_programs(x.device, route_count, hidden_size, block_n)

    _stage1_kernel[(stage1_programs,)](
        x,
        w.w13_weight,
        w.w13_weight_scale_inv,
        intermediate,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage1_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=1.0 if swiglu_limit is None else swiglu_limit,
        swiglu_beta=swiglu_beta,
        HAS_LIMIT=swiglu_limit is not None,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_FP8_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _stage2_kernel[(stage2_programs,)](
        intermediate,
        w.w2_weight,
        w.w2_weight_scale_inv,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage2_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_FP8_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _combine(route_output, topk_weights, output, None, 0)
    return output


# ===-----------------------------------------------------------------------===#
# FP8 activations: both GEMMs run on the FP8 tensor cores
# ===-----------------------------------------------------------------------===#
# The activations are quantized to E4M3 with one FP32 scale per row and 128
# columns, the layout of the weights' scale blocks, so each K step is one FP8
# `tl.dot` whose FP32 result is multiplied by the activation scale times the
# weight block scale. `x` is quantized once by `_quantize_rows_kernel`; stage 1
# quantizes its own output in the epilogue (a stage-1 tile is exactly 128
# intermediate columns), so stage 2 reads FP8 rows too.
@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    x = tl.load(x_ptr + row * hidden_size + cols).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-10) / 448.0
    tl.store(q_ptr + row * hidden_size + cols, (x / scale).to(tl.float8e4nv))
    tl.store(scale_ptr + row * (hidden_size // GROUP) + group, scale)


@triton.jit
def _stage1_fp8_act_kernel(
    xq_ptr,
    xs_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_q_ptr,
    inter_s_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_N == SCALE_BLOCK)
    tl.static_assert(BLOCK_K == SCALE_BLOCK)
    route_count = num_tokens * top_k
    scale_k = hidden_size // SCALE_BLOCK
    inter_scale_k = intermediate_size // SCALE_BLOCK
    up_scale_offset = inter_scale_k * scale_k
    tile_idx = tl.program_id(0)
    problem_start = 0
    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = inter_scale_k
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w13_ptr + expert_id.to(tl.int64) * (
            2 * intermediate_size * hidden_size
        )
        expert_scale = w13_scale_ptr + expert_id * 2 * up_scale_offset
        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            token_ids = tl.where(row_mask, route_ids // top_k, 0).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            gate_rows = n_offset + tl.arange(0, BLOCK_N)
            up_rows = intermediate_size + gate_rows
            gate_scale = expert_scale + tile_n * scale_k
            up_scale = gate_scale + up_scale_offset
            gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_offset in range(0, hidden_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                xq = tl.load(
                    xq_ptr + token_ids[:, None] * hidden_size + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                xs = tl.load(
                    xs_ptr + token_ids * scale_k + k_offset // SCALE_BLOCK,
                    mask=row_mask,
                    other=0.0,
                )
                gate = tl.load(
                    expert_weight + gate_rows[:, None] * hidden_size + k_cols[None, :]
                )
                up = tl.load(
                    expert_weight + up_rows[:, None] * hidden_size + k_cols[None, :]
                )
                scale_col = k_offset // SCALE_BLOCK
                gate_acc += tl.dot(xq, gate.T) * (
                    xs[:, None] * tl.load(gate_scale + scale_col)
                )
                up_acc += tl.dot(xq, up.T) * (
                    xs[:, None] * tl.load(up_scale + scale_col)
                )
            activated = _swiglu(
                gate_acc,
                up_acc,
                swiglu_alpha,
                swiglu_limit,
                swiglu_beta,
                HAS_LIMIT,
            )
            # Round to BF16 first, as the BF16 path stores it, then quantize the
            # tile's 128 columns per row.
            activated = activated.to(tl.bfloat16).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(activated), axis=1), 1e-10) / 448.0
            inter_offsets = route_ids[:, None] * intermediate_size + gate_rows[None, :]
            tl.store(
                inter_q_ptr + inter_offsets,
                (activated / scale[:, None]).to(tl.float8e4nv),
                mask=row_mask[:, None],
            )
            tl.store(
                inter_s_ptr + route_ids * inter_scale_k + tile_n,
                scale,
                mask=row_mask,
            )
            tile_idx += num_programs
        problem_start += problem_tiles


@triton.jit
def _stage2_fp8_act_kernel(
    inter_q_ptr,
    inter_s_ptr,
    w2_ptr,
    w2_scale_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(BLOCK_K == SCALE_BLOCK)
    route_count = num_tokens * top_k
    scale_k = intermediate_size // SCALE_BLOCK
    tile_idx = tl.program_id(0)
    problem_start = 0
    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(hidden_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w2_ptr + expert_id.to(tl.int64) * (
            hidden_size * intermediate_size
        )
        expert_scale = w2_scale_ptr + expert_id * (hidden_size // SCALE_BLOCK * scale_k)
        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            route_ids = tl.where(row_mask, route_ids, -1).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            weight_rows = n_offset + tl.arange(0, BLOCK_N)
            weight_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_offset in range(0, intermediate_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                aq = tl.load(
                    inter_q_ptr
                    + route_ids[:, None] * intermediate_size
                    + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                a_scale = tl.load(
                    inter_s_ptr + route_ids * scale_k + k_offset // SCALE_BLOCK,
                    mask=row_mask,
                    other=0.0,
                )
                weight = tl.load(
                    expert_weight
                    + weight_rows[:, None] * intermediate_size
                    + k_cols[None, :]
                )
                acc += tl.dot(aq, weight.T) * (
                    a_scale[:, None] * tl.load(weight_scale + k_offset // SCALE_BLOCK)
                )
            output_offsets = route_ids[:, None] * hidden_size + weight_rows[None, :]
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += num_programs
        problem_start += problem_tiles


def _fp8_act_is_compute_bound(route_count: int, num_experts: int) -> bool:
    """Whether the FP8-activation kernels pay off: from 64 routed rows per expert
    on average the GEMMs are compute bound; below, the expert weight reads
    dominate and the extra quantization launch only costs (ops/moe/README.md)."""
    return route_count >= 64 * num_experts


def _fp8_act_row_tile(route_count: int, num_experts: int) -> int:
    """Rows per tile from the mean routed rows per expert (64 up to 255, else 128)."""
    return 64 if route_count < 256 * num_experts else 128


def _triton_fp8_act_moe_impl(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w.w13_weight.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty_like(x)
    if not _fp8_act_is_compute_bound(num_tokens * top_k, num_experts):
        return _triton_fp8_moe_impl(x, w, topk_weights, topk_ids)
    swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_params(w)
    expert_route_ids, expert_counts, route_output, output = _prepare_routed_output(
        x, topk_ids, num_experts
    )
    route_count = num_tokens * top_k
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(
        (num_tokens, hidden_size // _FP8_BLOCK), device=x.device, dtype=torch.float32
    )
    _quantize_rows_kernel[(num_tokens, hidden_size // _FP8_BLOCK)](
        x, x_q, x_scale, hidden_size=hidden_size, GROUP=_FP8_BLOCK, num_warps=4
    )
    inter_q = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=torch.float8_e4m3fn
    )
    inter_scale = torch.empty(
        (route_count, intermediate_size // _FP8_BLOCK),
        device=x.device,
        dtype=torch.float32,
    )
    block_m = _fp8_act_row_tile(route_count, num_experts)
    stage1_programs = _num_programs(
        x.device, route_count, intermediate_size, _FP8_BLOCK
    )
    stage2_programs = _num_programs(x.device, route_count, hidden_size, _FP8_BLOCK)
    _stage1_fp8_act_kernel[(stage1_programs,)](
        x_q,
        x_scale,
        w.w13_weight,
        w.w13_weight_scale_inv,
        inter_q,
        inter_scale,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage1_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=1.0 if swiglu_limit is None else swiglu_limit,
        swiglu_beta=swiglu_beta,
        HAS_LIMIT=swiglu_limit is not None,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=_FP8_BLOCK,
        BLOCK_K=_FP8_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _stage2_fp8_act_kernel[(stage2_programs,)](
        inter_q,
        inter_scale,
        w.w2_weight,
        w.w2_weight_scale_inv,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage2_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=_FP8_BLOCK,
        BLOCK_K=_FP8_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _combine(route_output, topk_weights, output, None, 0)
    return output
