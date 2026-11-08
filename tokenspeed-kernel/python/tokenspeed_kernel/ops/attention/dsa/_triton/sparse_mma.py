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

"""Tensor-core sparse MLA attention over a dense latent cache (SM120).

One program owns a tile of query heads of one token. All heads of a token
share the token's selected slots, so a gathered KV tile is multiplied by every
head of the tile at once: ``S = Q K^T`` and ``O += P V`` run on ``mma.sync``
(FP8 ``m16n8k32`` when query and cache are both E4M3, BF16 ``m16n8k16``
otherwise) with an online softmax in FP32. The latent rows are gathered once
per tile and serve both the score and the value product.

A token's selected columns are cut into ``num_chunks`` chunks of whole tiles;
the count depends on the selection width only. Each chunk runs its own online
softmax and the chunk states are folded in order by :func:`_merge`. One program
folds them in registers. When a batch has too few (token, head tile) pairs to
fill the GPU, several programs share a token: each runs a run of chunks and
stores every chunk's state, and a second kernel folds all stored states with
the same function in the same order. The result is bitwise the same for every
batch size and launch shape, so a token's row does not depend on the batch it
runs in (``docs/design/numerics.md``).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_LOG2E = 1.4426950408889634


@triton.jit
def _load_tile(
    kv,
    topk_row,
    tile,
    tile_end,
    valid_len,
    page_stride,
    token_stride,
    channel_stride,
    PAGE_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather the latent and RoPE rows of the selected slots of one tile."""
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    in_range = (cols < valid_len) & (tile < tile_end)
    slot = tl.load(topk_row + cols, mask=in_range, other=-1).to(tl.int64)
    valid = slot >= 0
    slot = tl.where(valid, slot, 0)
    rows = (slot // PAGE_SIZE) * page_stride + (slot % PAGE_SIZE) * token_stride
    k = tl.load(
        kv + rows[:, None] + tl.arange(0, RANK)[None, :] * channel_stride,
        mask=valid[:, None],
        other=0.0,
    )
    if ROPE > 0:
        k_rope = tl.load(
            kv + rows[:, None] + (RANK + tl.arange(0, ROPE)[None, :]) * channel_stride,
            mask=valid[:, None],
            other=0.0,
        )
    else:
        # A zero RoPE width compiles the RoPE gather and score term out.
        k_rope = k
    return valid, k, k_rope


@triton.jit
def _merge(m_run, l_run, acc_run, m_part, l_part, acc_part):
    """Fold one chunk's softmax state into a running one.

    Explicit ``fma`` keeps the compiler from contracting the products
    differently in the two kernels that call this, which would break the
    bitwise match between split and unsplit launches.
    """
    m_new = tl.maximum(m_run, m_part)
    m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
    scale_run = tl.where(m_run == -float("inf"), 0.0, tl.exp2(m_run - m_safe))
    scale_part = tl.where(m_part == -float("inf"), 0.0, tl.exp2(m_part - m_safe))
    l_new = tl.fma(l_part, scale_part, l_run * scale_run)
    acc_new = tl.fma(acc_part, scale_part[:, None], acc_run * scale_run[:, None])
    return m_new, l_new, acc_new


# Values that change with the batch are runtime arguments that Triton must not
# specialize on (== 1, divisible by 16), or each new batch would recompile.
@triton.jit(
    do_not_specialize=[
        "num_heads",
        "topk",
        "chunk_tiles",
        "num_chunks",
        "chunks_per_split",
    ]
)
def _sparse_mma_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    lse,
    part_acc,
    part_ml,
    num_heads,
    topk,
    topk_stride,
    page_stride,
    token_stride,
    channel_stride,
    chunk_tiles,
    num_chunks,
    chunks_per_split,
    scale_log2,
    RETURN_LSE: tl.constexpr,
    SPLIT: tl.constexpr,
    FP8_QK: tl.constexpr,
    FP8_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Head tiles of one token are adjacent in launch order, so they gather the
    # same rows while those are resident in L2.
    head_block = tl.program_id(0)
    token = tl.program_id(1)
    split = tl.program_id(2)

    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    rank_offsets = tl.arange(0, RANK)
    n_offsets = tl.arange(0, BLOCK_N)

    q_rows = (token.to(tl.int64) * num_heads + heads) * HEAD_DIM
    valid_len = tl.minimum(tl.load(topk_lens + token).to(tl.int32), topk)
    num_tiles = tl.cdiv(valid_len, BLOCK_N)

    q_nope = tl.load(
        q + q_rows[:, None] + rank_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    if ROPE > 0:
        q_rope = tl.load(
            q + q_rows[:, None] + RANK + tl.arange(0, ROPE)[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
    else:
        q_rope = q_nope

    topk_row = topk_indices + token.to(tl.int64) * topk_stride
    if SPLIT:
        # A run of chunks per program; a chunk past the selection stores an
        # empty state.
        chunk_begin = split * chunks_per_split
        chunk_end = tl.minimum(chunk_begin + chunks_per_split, num_chunks)
    else:
        # Chunk 0 always runs; trailing empty chunks, which the combine kernel
        # does fold, leave a state unchanged.
        chunk_begin = 0
        chunk_end = tl.maximum(tl.cdiv(num_tiles, chunk_tiles), 1)
    m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_run = tl.zeros((BLOCK_H,), tl.float32)
    acc_run = tl.zeros((BLOCK_H, RANK), tl.float32)
    for chunk in range(chunk_begin, chunk_end):
        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, RANK), tl.float32)
        tile_begin = chunk * chunk_tiles
        tile_end = tl.minimum(tile_begin + chunk_tiles, num_tiles)
        valid, k, k_rope = _load_tile(
            kv,
            topk_row,
            tile_begin,
            tile_end,
            valid_len,
            page_stride,
            token_stride,
            channel_stride,
            PAGE_SIZE,
            RANK,
            ROPE,
            BLOCK_N,
        )
        for tile in range(tile_begin, tile_end):
            # Issue the next tile's gather before this tile's math; a tile
            # past the end loads nothing.
            valid_next, k_next, k_rope_next = _load_tile(
                kv,
                topk_row,
                tile + 1,
                tile_end,
                valid_len,
                page_stride,
                token_stride,
                channel_stride,
                PAGE_SIZE,
                RANK,
                ROPE,
                BLOCK_N,
            )

            if FP8_QK:
                s = tl.dot(q_nope, tl.trans(k))
                if ROPE > 0:
                    s = tl.dot(q_rope, tl.trans(k_rope), s)
            else:
                s = tl.dot(q_nope, tl.trans(k.to(tl.bfloat16)))
                if ROPE > 0:
                    s = tl.dot(q_rope, tl.trans(k_rope.to(tl.bfloat16)), s)
            s = tl.where(valid[None, :], s * scale_log2, -float("inf"))

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
                acc = tl.dot(p.to(tl.bfloat16), k.to(tl.bfloat16), acc)
            m_i = m_new
            valid = valid_next
            k = k_next
            k_rope = k_rope_next
        if SPLIT:
            part = (token.to(tl.int64) * num_chunks + chunk) * num_heads + heads
            tl.store(
                part_acc + part[:, None] * RANK + rank_offsets[None, :],
                acc,
                mask=head_mask[:, None],
            )
            tl.store(part_ml + part * 2, m_i, mask=head_mask)
            tl.store(part_ml + part * 2 + 1, l_i, mask=head_mask)
        else:
            m_run, l_run, acc_run = _merge(m_run, l_run, acc_run, m_i, l_i, acc)

    if not SPLIT:
        result = tl.where(l_run[:, None] > 0.0, acc_run / l_run[:, None], 0.0)
        out_rows = (token.to(tl.int64) * num_heads + heads) * RANK
        tl.store(
            out + out_rows[:, None] + rank_offsets[None, :],
            result,
            mask=head_mask[:, None],
        )
        if RETURN_LSE:
            lse_value = tl.where(
                l_run > 0.0,
                (m_run + tl.log2(l_run)) * 0.6931471805599453,
                -float("inf"),
            )
            tl.store(
                lse + token.to(tl.int64) * num_heads + heads, lse_value, mask=head_mask
            )


@triton.jit(do_not_specialize=["num_heads", "num_chunks"])
def _sparse_mma_combine_kernel(
    part_acc,
    part_ml,
    out,
    lse,
    num_heads,
    num_chunks,
    RETURN_LSE: tl.constexpr,
    RANK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    rank_offsets = tl.arange(0, RANK)
    one = tl.zeros((1,), tl.int64)
    # Same chunk states, same order and same _merge as the unsplit kernel.
    m_run = tl.full((1,), -float("inf"), tl.float32)
    l_run = tl.zeros((1,), tl.float32)
    acc_run = tl.zeros((1, RANK), tl.float32)
    for chunk in range(0, num_chunks):
        part = (token.to(tl.int64) * num_chunks + chunk) * num_heads + head + one
        m_part = tl.load(part_ml + part * 2)
        l_part = tl.load(part_ml + part * 2 + 1)
        acc_part = tl.load(part_acc + part[:, None] * RANK + rank_offsets[None, :])
        m_run, l_run, acc_run = _merge(m_run, l_run, acc_run, m_part, l_part, acc_part)
    result = tl.where(l_run[:, None] > 0.0, acc_run / l_run[:, None], 0.0)
    out_row = (token.to(tl.int64) * num_heads + head) * RANK
    tl.store(out + out_row + rank_offsets[None, :], result)
    if RETURN_LSE:
        lse_value = tl.where(
            l_run > 0.0, (m_run + tl.log2(l_run)) * 0.6931471805599453, -float("inf")
        )
        tl.store(lse + token.to(tl.int64) * num_heads + head + one, lse_value)


BLOCK_N = 32
NUM_WARPS = 8
NUM_STAGES = 2
MAX_SPLITS = 16
# Chunks per token selection; a split launch gives each its own program.
MAX_CHUNKS = 8


def plan_launch(tokens: int, heads: int, num_sms: int) -> tuple[int, int]:
    """Pick ``(block_h, num_splits)`` for a batch of ``tokens`` queries.

    ``num_splits`` is how many programs the launch rule wants per (token, head
    tile). It never changes the result.

    The tile shape is fixed (``BLOCK_N`` selected slots per step, ``NUM_WARPS``
    warps, ``NUM_STAGES`` stages). Only two things follow the batch: how many
    heads share one gathered tile, and how many programs split a token's
    selected slots. Measured on an RTX PRO 6000 (188 SMs):

    * Wide tiles (32 heads) reuse each gathered row for twice as many heads, so
      they win once there are enough programs to fill the GPU. Below that, 16
      heads give a second program per tile pair.
    * The best program count is about 2/3 of the SM count for up to 8 tokens
      and about 2.75 programs per SM above that; splitting further only adds
      partial-state traffic. ``num_splits`` is that count over the unsplit
      programs, rounded and clamped to ``[1, MAX_SPLITS]``.

    Both depend on the batch only through ``tokens``, so a CUDA graph captured
    per batch size sees fixed launch parameters.
    """
    few_tokens = tokens <= 8
    target = num_sms * 2 // 3 if few_tokens else num_sms * 11 // 4
    wide_tiles = triton.cdiv(heads, 32) * tokens * MAX_SPLITS >= num_sms * 2 // 3
    block_h = 32 if heads > 16 and wide_tiles else 16
    programs = triton.cdiv(heads, block_h) * tokens
    num_splits = min(max(round(target / programs), 1), MAX_SPLITS)
    return block_h, num_splits


def num_chunks_for(width: int, block_n: int = BLOCK_N) -> tuple[int, int]:
    """``(chunk_tiles, num_chunks)`` for a selection ``width`` columns wide.

    A function of the width alone, so every batch size cuts a token's
    selection at the same columns.
    """
    tiles = max(triton.cdiv(width, block_n), 1)
    chunk_tiles = triton.cdiv(tiles, MAX_CHUNKS)
    return chunk_tiles, triton.cdiv(tiles, chunk_tiles)


def sparse_mma_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    return_lse: bool,
    block_h: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Attend ``q`` over the cache rows named by ``topk_slots``.

    Args:
        q: ``[tokens, heads, kv_lora_rank + qk_rope_head_dim]`` contiguous,
            BF16 or E4M3.
        kv_cache: ``[pages, page_size, 1, dim]``-like dense cache, BF16 or
            E4M3, any strides; a channel stride other than one is read as is.
        topk_slots: ``[tokens, width]`` int32 global slots, ``-1`` for none.
        topk_lens: ``[tokens]`` int32 count of leading columns that are live.
        block_h, block_n, num_warps, num_stages: tile shape and launch shape.
        num_splits: programs sharing one (token, head tile), at most the
            number of chunks. One folds the chunks in registers; more store
            every chunk's state for a second kernel to fold. The result is
            bitwise the same for every value.

    Returns:
        ``(out, lse)``: ``out`` is ``[tokens, heads, kv_lora_rank]`` in BF16
        for E4M3 queries and the query dtype otherwise; ``lse`` is the FP32
        natural-log softmax normalizer ``[tokens, heads]`` or None.
    """
    tokens, heads, head_dim = q.shape
    fp8_qk = q.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.float8_e4m3fn
    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    if q.dtype == torch.float8_e4m3fn and not fp8_qk:
        q = q.to(kv_cache.dtype)
    if kv_cache.dim() == 4:
        kv_cache = (
            kv_cache.squeeze(1) if kv_cache.shape[1] == 1 else kv_cache.squeeze(2)
        )
    elif kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1] if kv_cache.dim() == 3 else 1
    device = q.device
    out = torch.empty((tokens, heads, kv_lora_rank), dtype=out_dtype, device=device)
    lse = (
        torch.empty((tokens, heads), dtype=torch.float32, device=device)
        if return_lse
        else None
    )
    width = topk_slots.shape[1]
    chunk_tiles, num_chunks = num_chunks_for(width, block_n)
    num_splits = max(min(num_splits, num_chunks), 1)
    split = num_splits > 1
    if split:
        part_acc = torch.empty(
            (tokens, num_chunks, heads, kv_lora_rank),
            dtype=torch.float32,
            device=device,
        )
        part_ml = torch.empty(
            (tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = out
    grid = (triton.cdiv(heads, block_h), tokens, num_splits)
    _sparse_mma_kernel[grid](
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        out,
        lse if return_lse else out,
        part_acc,
        part_ml,
        heads,
        width,
        topk_slots.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(-2),
        kv_cache.stride(-1),
        chunk_tiles,
        num_chunks,
        triton.cdiv(num_chunks, num_splits),
        float(softmax_scale) * _LOG2E,
        RETURN_LSE=return_lse,
        SPLIT=split,
        FP8_QK=fp8_qk,
        FP8_KV=kv_cache.dtype == torch.float8_e4m3fn,
        HEAD_DIM=head_dim,
        RANK=kv_lora_rank,
        ROPE=qk_rope_head_dim,
        PAGE_SIZE=page_size,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if split:
        _sparse_mma_combine_kernel[(tokens, heads)](
            part_acc,
            part_ml,
            out,
            lse if return_lse else out,
            heads,
            num_chunks,
            RETURN_LSE=return_lse,
            RANK=kv_lora_rank,
            num_warps=4,
        )
    return out, lse
