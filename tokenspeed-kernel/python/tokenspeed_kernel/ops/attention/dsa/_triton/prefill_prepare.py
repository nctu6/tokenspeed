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

"""One launch that prepares everything a DSA prefill top-k leaf scores with.

The DeepGEMM prefill leaf scores a chunk of ``T`` query tokens against ``R``
workspace key rows. Before the scoring kernel it needs the E4M3 queries and the
weights with their scales folded in, each key row (and its scale) gathered from
the page-planar index-K cache, and every token's candidate count. Done with
framework ops that is about twenty small launches around a ~4 us GEMM, so tiny
chunks are bound by the host. Here a token program quantizes the token's
heads (the canonical ``token_group`` FP8 scheme: ``scale = max(amax, 1e-10) /
448``, ``code = clamp(x / scale, -448, 448)``) and folds the scale into the
weights in the same FP32 operation order as ``combine_topk_weights``; a gather
program copies a block of key rows. Both run in the one grid.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_EPS = 1e-10
_FP8_MAX = 448.0
_GATHER_ROWS = 32


# Token and row counts follow the batch: one binary for every chunk size.
@triton.jit(do_not_specialize=["topk", "num_tokens", "num_rows"])
def _dsa_prefill_prepare_kernel(
    q_ptr,
    q_row_stride,
    weights_ptr,
    weights_row_stride,
    row_starts_ptr,
    row_ends_ptr,
    slots_ptr,
    cache_ptr,
    q_fp8_ptr,
    scaled_weights_ptr,
    candidate_lens_ptr,
    lens_out_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    softmax_scale,
    topk,
    num_tokens,
    num_rows,
    HEADS: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GATHER: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_tokens:
        token = pid.to(tl.int64)
        heads = tl.arange(0, OUT_HEADS)
        dims = tl.arange(0, HEAD_DIM)
        # Heads past HEADS are the zero heads that adapt a 16-head indexer to
        # DeepGEMM's 32-head ABI; they carry zero weights.
        x = tl.load(
            q_ptr + token * q_row_stride + heads[:, None] * HEAD_DIM + dims[None, :],
            mask=heads[:, None] < HEADS,
            other=0.0,
        ).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10) / 448.0
        codes = tl.clamp(x / scale[:, None], -448.0, 448.0).to(
            q_fp8_ptr.dtype.element_ty
        )
        tl.store(
            q_fp8_ptr + (token * OUT_HEADS + heads[:, None]) * HEAD_DIM + dims[None, :],
            codes,
        )
        w = tl.load(
            weights_ptr + token * weights_row_stride + heads,
            mask=heads < HEADS,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            scaled_weights_ptr + token * OUT_HEADS + heads, w * scale * softmax_scale
        )
        candidates = tl.maximum(
            tl.load(row_ends_ptr + token) - tl.load(row_starts_ptr + token), 0
        )
        tl.store(candidate_lens_ptr + token, candidates)
        tl.store(lens_out_ptr + token, tl.minimum(candidates, topk))
    elif GATHER:
        rows = (pid - num_tokens) * GATHER_ROWS + tl.arange(0, GATHER_ROWS)
        in_range = rows < num_rows
        slot = tl.load(slots_ptr + rows, mask=in_range, other=0)
        page = slot // PAGE_SIZE
        in_page = slot - page * PAGE_SIZE
        # Page-planar rows: PAGE_SIZE keys of HEAD_DIM bytes, then PAGE_SIZE
        # FP32 scales.
        page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
        dims = tl.arange(0, HEAD_DIM)
        keys = tl.load(
            cache_ptr
            + (page * page_bytes + in_page * HEAD_DIM)[:, None]
            + dims[None, :],
            mask=in_range[:, None],
            other=0,
        )
        tl.store(
            k_fp8_ptr + rows[:, None].to(tl.int64) * HEAD_DIM + dims[None, :],
            keys.to(k_fp8_ptr.dtype.element_ty, bitcast=True),
            mask=in_range[:, None],
        )
        scales = tl.load(
            (cache_ptr + page * page_bytes + PAGE_SIZE * HEAD_DIM + in_page * 4).to(
                tl.pointer_type(tl.float32)
            ),
            mask=in_range,
            other=0.0,
        )
        tl.store(k_scale_ptr + rows, scales, mask=in_range)


def dsa_prefill_prepare(
    q: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    softmax_scale: float,
    topk: int,
    lens_out: torch.Tensor,
    out_heads: int,
    index_k_cache: torch.Tensor | None = None,
    kv_workspace_slots: torch.Tensor | None = None,
    page_size: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Quantize the queries, fold the weights, size the rows and gather the keys.

    Args:
        q: BF16 ``[tokens, heads, 128]`` queries with contiguous rows.
        weights: ``[tokens, heads]`` BF16/FP16/FP32 weights with unit column
            stride (the row stride may be anything, e.g. a fused projection).
        row_starts, row_ends: int32 ``[tokens]`` workspace row bounds.
        softmax_scale: Score scale folded into the weights.
        topk: Selection width; bounds ``lens_out``.
        lens_out: int32 ``[tokens]`` destination of ``min(candidate_lens, topk)``.
        out_heads: ``heads``, or 32 for a 16-head indexer (zero heads added).
        index_k_cache: Page-planar FP8 index-K plane (``PAGE_SIZE`` keys of 128
            bytes, then ``PAGE_SIZE`` FP32 scales, per page). With
            ``kv_workspace_slots`` and ``page_size`` it is gathered into
            workspace-row order; without them no keys are gathered.
        kv_workspace_slots: int64 ``[rows]`` cache slot of each workspace row.
        page_size: Keys per page of ``index_k_cache``.

    Returns:
        ``(q_fp8, scaled_weights, candidate_lens, k_fp8, k_scale)``: E4M3
        ``[tokens, out_heads, 128]``; FP32 ``[tokens, out_heads]``
        (``weights * q_scale * softmax_scale``); int32 ``[tokens]``
        ``max(row_ends - row_starts, 0)``; and, when gathering, E4M3
        ``[rows, 128]`` and FP32 ``[rows]`` (else ``None``).
    """
    tokens, heads, head_dim = q.shape
    gather = kv_workspace_slots is not None
    rows = kv_workspace_slots.numel() if gather else 0
    device = q.device
    q_fp8 = torch.empty(
        (tokens, out_heads, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    scaled_weights = torch.empty(
        (tokens, out_heads), dtype=torch.float32, device=device
    )
    candidate_lens = torch.empty((tokens,), dtype=torch.int32, device=device)
    if gather:
        k_fp8 = torch.empty((rows, head_dim), dtype=torch.float8_e4m3fn, device=device)
        k_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    else:
        k_fp8 = k_scale = None
    num_programs = tokens + (triton.cdiv(rows, _GATHER_ROWS) if gather else 0)
    # Positional arguments and the default launch options: the Triton launcher
    # spends host time on every keyword.
    _dsa_prefill_prepare_kernel[(num_programs,)](
        q,
        q.stride(0),
        weights,
        weights.stride(0),
        row_starts,
        row_ends,
        kv_workspace_slots if gather else row_starts,
        index_k_cache if gather else q,
        q_fp8,
        scaled_weights,
        candidate_lens,
        lens_out,
        k_fp8 if gather else q_fp8,
        k_scale if gather else scaled_weights,
        float(softmax_scale),
        int(topk),
        tokens,
        rows,
        heads,
        out_heads,
        head_dim,
        int(page_size) if gather else 1,
        gather,
        _GATHER_ROWS,
    )
    return q_fp8, scaled_weights, candidate_lens, k_fp8, k_scale
