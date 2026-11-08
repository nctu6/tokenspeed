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

"""Tensor-core DeepSeek V4 selected attention over a dense workspace (SM120).

One program owns a tile of query heads of one token. All heads of a token read
the same selected workspace rows, so a gathered row tile is multiplied by every
head of the tile at once: ``S = Q K^T`` and ``O += P K`` run on ``mma.sync``
(BF16 ``m16n8k16``) with an online softmax in FP32. A V4 latent row is key and
value at once, so one gathered tile serves both products. An optional
per-head sink logit joins the denominator only.

A token's selection is cut into ``num_chunks`` chunks of whole tiles; the
count depends on the selection width only. Each chunk runs its own online
softmax and the chunk states are folded in order by :func:`_merge`. One
program folds them in registers. When a batch has too few (token, head tile)
pairs to fill the GPU, every chunk gets its own program and a second kernel
folds the stored chunk states with the same function in the same order. The
result is bitwise the same for every batch size, so a prefill chunk replayed
in a different batch reproduces its rows exactly, as FlashMLA's unsplit sparse
prefill does. The tile shape and head-tile rule are those of the SM120 DSA
kernel, whose gather is the same.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa._triton.sparse_mma import (
    BLOCK_N,
    NUM_STAGES,
    NUM_WARPS,
    plan_launch,
)

_LOG2E = 1.4426950408889634
HEAD_DIM = 512

__all__ = ["BLOCK_N", "MAX_CHUNKS", "plan_launch", "sparse_mma_attention"]

# Chunks per token selection; a split launch gives each its own program.
MAX_CHUNKS = 4


@triton.jit
def _load_rows(
    kv,
    index_row,
    tile,
    tile_end,
    valid_len,
    num_rows,
    kv_stride_row,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """Gather the workspace rows of one tile of the selection."""
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    in_range = (cols < valid_len) & (tile < tile_end)
    rows = tl.load(index_row + cols, mask=in_range, other=-1).to(tl.int64)
    valid = (rows >= 0) & (rows < num_rows)
    rows = tl.where(valid, rows, 0)
    k = tl.load(
        kv + rows[:, None] * kv_stride_row + tl.arange(0, DIM)[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    return valid, k


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
        "width",
        "num_rows",
        "indices_stride",
        "chunk_tiles",
        "num_chunks",
    ]
)
def _dsv4_sparse_mma_kernel(
    q,
    kv,
    indices,
    lens,
    sink,
    out,
    lse,
    part_acc,
    part_ml,
    num_heads,
    width,
    num_rows,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    indices_stride,
    out_stride_token,
    out_stride_head,
    chunk_tiles,
    num_chunks,
    scale_log2,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    SPLIT: tl.constexpr,
    DIM: tl.constexpr,
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
    dim_offsets = tl.arange(0, DIM)

    valid_len = tl.minimum(tl.maximum(tl.load(lens + token).to(tl.int32), 0), width)
    num_tiles = tl.cdiv(valid_len, BLOCK_N)
    sink_logit = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    if HAS_SINK:
        sink_logit = tl.load(sink + heads, mask=head_mask, other=0.0).to(tl.float32)
        sink_logit = sink_logit * 1.4426950408889634

    q_tile = tl.load(
        q
        + token.to(tl.int64) * q_stride_token
        + heads[:, None] * q_stride_head
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    index_row = indices + token.to(tl.int64) * indices_stride

    if SPLIT:
        # One chunk per program; a chunk past the selection stores an empty
        # state (or only the sink).
        chunk_begin = split
        chunk_end = split + 1
    else:
        # Chunk 0 always runs (it owns the sink); trailing empty chunks, which
        # the combine kernel does fold, leave a state unchanged.
        chunk_begin = 0
        chunk_end = tl.maximum(tl.cdiv(num_tiles, chunk_tiles), 1)
    m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_run = tl.zeros((BLOCK_H,), tl.float32)
    acc_run = tl.zeros((BLOCK_H, DIM), tl.float32)
    for chunk in range(chunk_begin, chunk_end):
        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        if HAS_SINK:
            # The sink is one extra key with a zero value, owned by chunk 0.
            if chunk == 0:
                m_i = sink_logit
                l_i = tl.where(head_mask, 1.0, 0.0)
        acc = tl.zeros((BLOCK_H, DIM), tl.float32)
        tile_begin = chunk * chunk_tiles
        tile_end = tl.minimum(tile_begin + chunk_tiles, num_tiles)
        valid, k = _load_rows(
            kv,
            index_row,
            tile_begin,
            tile_end,
            valid_len,
            num_rows,
            kv_stride_row,
            BLOCK_N,
            DIM,
        )
        for tile in range(tile_begin, tile_end):
            # Issue the next tile's gather before this tile's math; a tile
            # past the end loads nothing.
            valid_next, k_next = _load_rows(
                kv,
                index_row,
                tile + 1,
                tile_end,
                valid_len,
                num_rows,
                kv_stride_row,
                BLOCK_N,
                DIM,
            )
            s = tl.dot(q_tile, tl.trans(k))
            s = tl.where(valid[None, :], s * scale_log2, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(k.dtype), k, acc)
            m_i = m_new
            valid = valid_next
            k = k_next
        if SPLIT:
            part = (token.to(tl.int64) * num_chunks + chunk) * num_heads + heads
            tl.store(
                part_acc + part[:, None] * DIM + dim_offsets[None, :],
                acc,
                mask=head_mask[:, None],
            )
            tl.store(part_ml + part * 2, m_i, mask=head_mask)
            tl.store(part_ml + part * 2 + 1, l_i, mask=head_mask)
        else:
            m_run, l_run, acc_run = _merge(m_run, l_run, acc_run, m_i, l_i, acc)

    if not SPLIT:
        result = tl.where(l_run[:, None] > 0.0, acc_run / l_run[:, None], 0.0)
        out_rows = (
            token.to(tl.int64) * out_stride_token + heads[:, None] * out_stride_head
        )
        tl.store(
            out + out_rows + dim_offsets[None, :],
            result.to(out.dtype.element_ty),
            mask=head_mask[:, None],
        )
        if RETURN_LSE:
            lse_value = tl.where(
                l_run > 0.0,
                (m_run + tl.log2(l_run)) * 0.6931471805599453,
                -float("inf"),
            )
            tl.store(
                lse + token.to(tl.int64) * num_heads + heads,
                lse_value,
                mask=head_mask,
            )


@triton.jit(do_not_specialize=["num_heads", "num_chunks"])
def _dsv4_sparse_mma_combine_kernel(
    part_acc,
    part_ml,
    out,
    lse,
    num_heads,
    num_chunks,
    out_stride_token,
    out_stride_head,
    RETURN_LSE: tl.constexpr,
    DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim_offsets = tl.arange(0, DIM)
    one = tl.zeros((1,), tl.int64)
    # Same chunk states, same order and same _merge as the unsplit kernel.
    m_run = tl.full((1,), -float("inf"), tl.float32)
    l_run = tl.zeros((1,), tl.float32)
    acc_run = tl.zeros((1, DIM), tl.float32)
    for chunk in range(0, num_chunks):
        part = (token.to(tl.int64) * num_chunks + chunk) * num_heads + head + one
        m_part = tl.load(part_ml + part * 2)
        l_part = tl.load(part_ml + part * 2 + 1)
        acc_part = tl.load(part_acc + part[:, None] * DIM + dim_offsets[None, :])
        m_run, l_run, acc_run = _merge(m_run, l_run, acc_run, m_part, l_part, acc_part)
    result = tl.where(l_run[:, None] > 0.0, acc_run / l_run[:, None], 0.0)
    tl.store(
        out
        + token.to(tl.int64) * out_stride_token
        + head * out_stride_head
        + dim_offsets[None, :],
        result.to(out.dtype.element_ty),
    )
    if RETURN_LSE:
        lse_value = tl.where(
            l_run > 0.0, (m_run + tl.log2(l_run)) * 0.6931471805599453, -float("inf")
        )
        tl.store(lse + token.to(tl.int64) * num_heads + head + one, lse_value)


def num_chunks_for(width: int) -> tuple[int, int]:
    """``(chunk_tiles, num_chunks)`` for a selection ``width`` columns wide.

    A function of the width alone, so every batch size cuts a token's
    selection at the same columns.
    """
    tiles = max(triton.cdiv(width, BLOCK_N), 1)
    chunk_tiles = triton.cdiv(tiles, MAX_CHUNKS)
    return chunk_tiles, triton.cdiv(tiles, chunk_tiles)


def sparse_mma_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    *,
    softmax_scale: float,
    block_h: int,
    split: bool,
) -> None:
    """Attend ``q`` over the workspace rows named by ``indices``.

    Args:
        q: ``[tokens, heads, 512]`` BF16 with unit inner stride.
        kv: ``[rows, 512]`` BF16 workspace with unit inner stride.
        indices: ``[tokens, width]`` int32/int64 row ids; entries that are
            negative or past the last row are skipped.
        lens: ``[tokens]`` int32/int64 count of leading live columns.
        sink: ``[heads]`` FP32 sink logits or None.
        out: ``[tokens, heads, 512]`` BF16 result with unit inner stride.
        lse: ``[tokens, heads]`` contiguous FP32 natural-log normalizer or None.
        block_h: heads per program (16 or 32).
        split: give every chunk its own program and merge in a second kernel;
            the result is bitwise the same either way.
    """
    tokens, heads, dim = q.shape
    device = q.device
    chunk_tiles, num_chunks = num_chunks_for(indices.shape[1])
    if split:
        part_acc = torch.empty(
            (tokens, num_chunks, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = out
    grid = (triton.cdiv(heads, block_h), tokens, num_chunks if split else 1)
    _dsv4_sparse_mma_kernel[grid](
        q,
        kv,
        indices,
        lens,
        sink if sink is not None else out,
        out,
        lse if lse is not None else out,
        part_acc,
        part_ml,
        heads,
        indices.shape[1],
        kv.shape[0],
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        chunk_tiles,
        num_chunks,
        float(softmax_scale) * _LOG2E,
        HAS_SINK=sink is not None,
        RETURN_LSE=lse is not None,
        SPLIT=split,
        DIM=dim,
        BLOCK_H=block_h,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    if split:
        _dsv4_sparse_mma_combine_kernel[(tokens, heads)](
            part_acc,
            part_ml,
            out,
            lse if lse is not None else out,
            heads,
            num_chunks,
            out.stride(0),
            out.stride(1),
            RETURN_LSE=lse is not None,
            DIM=dim,
            num_warps=4,
        )
