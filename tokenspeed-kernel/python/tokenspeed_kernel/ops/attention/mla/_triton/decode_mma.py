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

"""Tensor-core MLA decode over a paged latent cache (SM120).

One program owns a tile of ``(query, head)`` rows of one request. Every row
reads the same latent cache rows, so a loaded KV tile is multiplied by all rows
of the tile at once: ``S = Q K^T`` and ``O += P V`` run on ``mma.sync`` with an
online softmax in FP32, and the latent row serves both the score and the
value product. The request's visible range is split across ``num_splits``
programs when a batch has too few row tiles to fill the GPU; a second kernel
merges the partial softmax states.

Masks follow the Triton decode kernel's contract. Row ``i`` of a request with
``L`` cached tokens and ``Q`` query rows sees

* causal block (``block_size == 1``): the first ``L - (Q - 1 - i)`` tokens;
* non-causal block (``block_size > 1``): all ``L`` tokens;

and, with ``window_left >= 0``, only tokens at or after
``visible - block_size - window_left + position`` (``position`` is the row's
index in its non-causal block, 0 for a causal block).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_LOG2E = 1.4426950408889634
_FAR = tl.constexpr(1 << 30)


@triton.jit
def _load_kv_tile(
    kv,
    table_row,
    tile_start,
    hi,
    active,
    stride_page,
    stride_token,
    PAGE_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Load the latent and RoPE parts of ``BLOCK_N`` consecutive tokens."""
    pos = tile_start + tl.arange(0, BLOCK_N)
    in_range = (pos < hi) & active
    page = tl.load(table_row + pos // PAGE_SIZE, mask=in_range, other=0).to(tl.int64)
    rows = page * stride_page + (pos % PAGE_SIZE) * stride_token
    k = tl.load(
        kv + rows[:, None] + tl.arange(0, RANK)[None, :],
        mask=in_range[:, None],
        other=0.0,
    )
    k_rope = tl.load(
        kv + rows[:, None] + RANK + tl.arange(0, ROPE)[None, :],
        mask=in_range[:, None],
        other=0.0,
    )
    return k, k_rope


# Everything that follows the batch or the metadata is a runtime argument that
# Triton must not specialize on (== 1, divisible by 16), or each new batch
# would recompile.
@triton.jit(
    do_not_specialize=[
        "q_len",
        "num_heads",
        "num_splits",
        "block_size",
        "window_left",
        "stride_table",
    ]
)
def _mla_mma_decode_kernel(
    q,
    kv,
    page_table,
    cache_seqlens,
    out,
    lse,
    part_acc,
    part_ml,
    q_len,
    num_heads,
    num_splits,
    block_size,
    window_left,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_page,
    stride_token,
    stride_table,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lb,
    stride_lq,
    scale_log2,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    FP8_QK: tl.constexpr,
    FP8_KV: tl.constexpr,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    num_batch = tl.num_programs(1)

    rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < q_len * num_heads
    query = rows // num_heads
    heads = rows - query * num_heads
    rank_offsets = tl.arange(0, RANK)
    rope_offsets = tl.arange(0, ROPE)

    length = tl.load(cache_seqlens + batch).to(tl.int32)
    if CAUSAL:
        visible = length - (q_len - 1 - query)
    else:
        visible = length + 0 * query
    visible = tl.where(row_ok, tl.maximum(visible, 0), 0)
    if HAS_WINDOW:
        position = (batch * q_len + query) % block_size
        first = tl.maximum(visible - block_size - window_left + position, 0)
    else:
        first = 0 * query
    first = tl.where(row_ok, first, _FAR)

    # The tile walks the union of its rows' ranges, aligned down to a tile.
    lo = (tl.min(first, axis=0) // BLOCK_N) * BLOCK_N
    hi = tl.max(visible, axis=0)
    num_tiles = tl.cdiv(tl.maximum(hi - lo, 0), BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, num_splits)
    tile_begin = split * tiles_per_split
    tile_end = tl.minimum(tile_begin + tiles_per_split, num_tiles)

    q_rows = batch * stride_qb + query * stride_qq + heads * stride_qh
    q_nope = tl.load(
        q + q_rows[:, None] + rank_offsets[None, :], mask=row_ok[:, None], other=0.0
    )
    q_rope = tl.load(
        q + q_rows[:, None] + RANK + rope_offsets[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, RANK), tl.float32)

    table_row = page_table + batch.to(tl.int64) * stride_table
    k, k_rope = _load_kv_tile(
        kv,
        table_row,
        lo + tile_begin * BLOCK_N,
        hi,
        tile_begin < tile_end,
        stride_page,
        stride_token,
        PAGE_SIZE,
        RANK,
        ROPE,
        BLOCK_N,
    )
    for tile in range(tile_begin, tile_end):
        # Issue the next tile's load before this tile's math; a tile past the
        # end loads nothing.
        k_next, k_rope_next = _load_kv_tile(
            kv,
            table_row,
            lo + (tile + 1) * BLOCK_N,
            hi,
            tile + 1 < tile_end,
            stride_page,
            stride_token,
            PAGE_SIZE,
            RANK,
            ROPE,
            BLOCK_N,
        )
        pos = lo + tile * BLOCK_N + tl.arange(0, BLOCK_N)
        live = (pos[None, :] >= first[:, None]) & (pos[None, :] < visible[:, None])

        if FP8_QK:
            s = tl.dot(q_nope, tl.trans(k))
            s = tl.dot(q_rope, tl.trans(k_rope), s)
        else:
            s = tl.dot(q_nope, tl.trans(k.to(q_nope.dtype)))
            s = tl.dot(q_rope, tl.trans(k_rope.to(q_rope.dtype)), s)
        s = tl.where(live, s * scale_log2, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if FP8_KV:
            # E4M3 values are exact in FP16, which keeps 3 more bits of the
            # probability than BF16 does.
            acc = tl.dot(p.to(tl.float16), k.to(tl.float16), acc)
        else:
            acc = tl.dot(p.to(q_nope.dtype), k.to(q_nope.dtype), acc)
        m_i = m_new
        k = k_next
        k_rope = k_rope_next

    if num_splits == 1:
        result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
        out_rows = batch * stride_ob + query * stride_oq + heads * stride_oh
        tl.store(
            out + out_rows[:, None] + rank_offsets[None, :],
            result.to(out.dtype.element_ty),
            mask=row_ok[:, None],
        )
        if RETURN_LSE:
            lse_value = tl.where(
                l_i > 0.0, (m_i + tl.log2(l_i)) * 0.6931471805599453, -float("inf")
            )
            tl.store(
                lse + batch * stride_lb + query * stride_lq + heads,
                lse_value,
                mask=row_ok,
            )
    else:
        part = ((split * num_batch + batch) * q_len + query) * num_heads + heads
        tl.store(
            part_acc + part.to(tl.int64)[:, None] * RANK + rank_offsets[None, :],
            acc,
            mask=row_ok[:, None],
        )
        tl.store(part_ml + part.to(tl.int64) * 2, m_i, mask=row_ok)
        tl.store(part_ml + part.to(tl.int64) * 2 + 1, l_i, mask=row_ok)


@triton.jit(do_not_specialize=["q_len", "num_heads", "num_splits"])
def _mla_mma_decode_combine_kernel(
    part_acc,
    part_ml,
    out,
    lse,
    q_len,
    num_heads,
    num_splits,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lb,
    stride_lq,
    RETURN_LSE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    request_row = tl.program_id(0)  # batch * q_len + query
    head = tl.program_id(1)
    num_rows = tl.num_programs(0)
    batch = request_row // q_len
    query = request_row - batch * q_len
    splits = tl.arange(0, BLOCK_S)
    split_mask = splits < num_splits
    rank_offsets = tl.arange(0, RANK)

    part = ((splits * num_rows + request_row) * num_heads + head).to(tl.int64)
    m = tl.load(part_ml + part * 2, mask=split_mask, other=-float("inf"))
    l = tl.load(part_ml + part * 2 + 1, mask=split_mask, other=0.0)
    m_all = tl.max(m, axis=0)
    m_safe = tl.where(m_all == -float("inf"), 0.0, m_all)
    weight = tl.where(split_mask & (l > 0.0), tl.exp2(m - m_safe), 0.0)
    l_all = tl.sum(weight * l, axis=0)
    acc = tl.load(
        part_acc + part[:, None] * RANK + rank_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    total = tl.sum(acc * weight[:, None], axis=0)
    result = tl.where(l_all > 0.0, total / l_all, 0.0)
    tl.store(
        out + batch * stride_ob + query * stride_oq + head * stride_oh + rank_offsets,
        result.to(out.dtype.element_ty),
    )
    if RETURN_LSE:
        tl.store(
            lse + batch * stride_lb + query * stride_lq + head,
            tl.where(
                l_all > 0.0,
                (m_safe + tl.log2(l_all)) * 0.6931471805599453,
                -float("inf"),
            ),
        )


BLOCK_N = 32
NUM_WARPS = 8
NUM_STAGES = 2
MAX_SPLITS = 128


def plan_launch(batch: int, rows: int, num_sms: int) -> tuple[int, int]:
    """Pick ``(block_m, num_splits)`` for ``batch`` requests of ``rows`` rows.

    ``rows`` is ``q_len * num_heads``. A wider tile reuses each loaded KV tile
    for more rows, so it wins once there are enough programs to fill the GPU;
    below that a 16-row tile gives a second program per tile pair. The
    remaining parallelism comes from splitting the cached range; a split past
    a request's end exits at once. Both depend on the batch size only, never
    on the cached lengths, so a CUDA graph captured per batch size sees fixed
    launch parameters and the result does not depend on ``max_seqlen_k``.
    """
    wide = triton.cdiv(rows, 32) * batch * MAX_SPLITS >= num_sms * 2 // 3
    block_m = 32 if rows > 16 and wide else 16
    programs = triton.cdiv(rows, block_m) * batch
    target = num_sms * 2 // 3 if batch <= 8 else num_sms * 11 // 4
    num_splits = min(max(round(target / programs), 1), MAX_SPLITS)
    return block_m, num_splits


def mla_mma_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    lse: torch.Tensor | None,
    window_left: int,
    noncausal_block_size: int,
    block_m: int,
    num_splits: int,
) -> None:
    """Attend ``q`` over the latent cache rows of each request.

    Args:
        q: ``[batch, q_len, heads, kv_lora_rank + qk_rope_head_dim]``, unit
            inner stride; BF16, FP16 or E4M3.
        kv_cache: ``[pages, page_size, 1, dim]`` latent cache, unit inner
            stride, in the dtype of ``q``.
        out: ``[batch, q_len, heads, kv_lora_rank]`` result, unit inner stride.
        page_table: ``[batch, max_pages]`` int32 page ids.
        cache_seqlens: ``[batch]`` int32 visible lengths (the last query's, for
            a causal block).
        lse: Optional ``[batch, q_len, heads]`` FP32 natural-log normalizer.
        window_left: History window or -1; see the module docstring.
        noncausal_block_size: 1 for causal rows, else the proposal block size.
        block_m, num_splits: launch shape from :func:`plan_launch`.
    """
    batch, q_len, heads, _ = q.shape
    page_size = kv_cache.shape[1]
    device = q.device
    fp8 = q.dtype == torch.float8_e4m3fn
    if num_splits > 1:
        part_acc = torch.empty(
            (num_splits, batch * q_len, heads, kv_lora_rank),
            dtype=torch.float32,
            device=device,
        )
        part_ml = torch.empty(
            (num_splits, batch * q_len, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = out
    lse_arg = lse if lse is not None else out
    lse_strides = (lse.stride(0), lse.stride(1)) if lse is not None else (0, 0)
    grid = (triton.cdiv(q_len * heads, block_m), batch, num_splits)
    _mla_mma_decode_kernel[grid](
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        out,
        lse_arg,
        part_acc,
        part_ml,
        q_len,
        heads,
        num_splits,
        noncausal_block_size,
        window_left,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        page_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        *lse_strides,
        float(softmax_scale) * _LOG2E,
        CAUSAL=noncausal_block_size == 1,
        HAS_WINDOW=window_left >= 0,
        RETURN_LSE=lse is not None,
        FP8_QK=fp8,
        FP8_KV=fp8,
        RANK=kv_lora_rank,
        ROPE=qk_rope_head_dim,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    if num_splits > 1:
        _mla_mma_decode_combine_kernel[(batch * q_len, heads)](
            part_acc,
            part_ml,
            out,
            lse_arg,
            q_len,
            heads,
            num_splits,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            *lse_strides,
            RETURN_LSE=lse is not None,
            RANK=kv_lora_rank,
            BLOCK_S=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
