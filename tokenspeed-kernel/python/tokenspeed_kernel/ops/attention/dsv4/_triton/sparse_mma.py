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

When a batch has too few (token, head tile) pairs to fill the GPU, the
selection is split across ``num_splits`` programs and a second kernel merges
the partial softmax states. The launch shape is the one tuned for the SM120
DSA kernel, whose tile and gather are the same.
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

__all__ = ["BLOCK_N", "plan_launch", "sparse_mma_attention"]


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


# Values that change with the batch are runtime arguments that Triton must not
# specialize on (== 1, divisible by 16), or each new batch would recompile.
@triton.jit(
    do_not_specialize=[
        "num_heads",
        "width",
        "num_rows",
        "indices_stride",
        "tiles_per_split",
        "num_splits",
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
    tiles_per_split,
    num_splits,
    scale_log2,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
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
    tile_begin = split * tiles_per_split
    tile_end = tl.minimum(tile_begin + tiles_per_split, num_tiles)

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    if HAS_SINK:
        # The sink is one extra key with a zero value; only split 0 owns it.
        sink_logit = tl.load(sink + heads, mask=head_mask, other=-float("inf"))
        if split == 0:
            m_i = sink_logit.to(tl.float32) * 1.4426950408889634
            l_i = tl.where(head_mask, 1.0, 0.0)
    acc = tl.zeros((BLOCK_H, DIM), tl.float32)

    q_tile = tl.load(
        q
        + token.to(tl.int64) * q_stride_token
        + heads[:, None] * q_stride_head
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    index_row = indices + token.to(tl.int64) * indices_stride
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
        # Issue the next tile's gather before this tile's math; a tile past
        # the end loads nothing.
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

    if num_splits == 1:
        result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
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
                l_i > 0.0, (m_i + tl.log2(l_i)) * 0.6931471805599453, -float("inf")
            )
            tl.store(
                lse + token.to(tl.int64) * num_heads + heads,
                lse_value,
                mask=head_mask,
            )
    else:
        part = (token.to(tl.int64) * num_splits + split) * num_heads + heads
        tl.store(
            part_acc + part[:, None] * DIM + dim_offsets[None, :],
            acc,
            mask=head_mask[:, None],
        )
        tl.store(part_ml + part * 2, m_i, mask=head_mask)
        tl.store(part_ml + part * 2 + 1, l_i, mask=head_mask)


@triton.jit(do_not_specialize=["num_heads", "num_splits"])
def _dsv4_sparse_mma_combine_kernel(
    part_acc,
    part_ml,
    out,
    lse,
    num_heads,
    num_splits,
    out_stride_token,
    out_stride_head,
    RETURN_LSE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_S)
    split_mask = splits < num_splits
    dim_offsets = tl.arange(0, DIM)

    part = (token.to(tl.int64) * num_splits + splits) * num_heads + head
    m = tl.load(part_ml + part * 2, mask=split_mask, other=-float("inf"))
    l = tl.load(part_ml + part * 2 + 1, mask=split_mask, other=0.0)
    m_all = tl.max(m, axis=0)
    m_safe = tl.where(m_all == -float("inf"), 0.0, m_all)
    weight = tl.where(split_mask & (l > 0.0), tl.exp2(m - m_safe), 0.0)
    l_all = tl.sum(weight * l, axis=0)
    acc = tl.load(
        part_acc + part[:, None] * DIM + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    total = tl.sum(acc * weight[:, None], axis=0)
    result = tl.where(l_all > 0.0, total / l_all, 0.0)
    tl.store(
        out
        + token.to(tl.int64) * out_stride_token
        + head * out_stride_head
        + dim_offsets,
        result.to(out.dtype.element_ty),
    )
    if RETURN_LSE:
        tl.store(
            lse + token.to(tl.int64) * num_heads + head,
            tl.where(
                l_all > 0.0,
                (m_safe + tl.log2(l_all)) * 0.6931471805599453,
                -float("inf"),
            ),
        )


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
    num_splits: int,
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
        block_h, num_splits: launch shape from :func:`plan_launch`.
    """
    tokens, heads, dim = q.shape
    device = q.device
    if num_splits > 1:
        part_acc = torch.empty(
            (tokens, num_splits, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tokens, num_splits, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = out
    width = indices.shape[1]
    tiles = triton.cdiv(width, BLOCK_N)
    grid = (triton.cdiv(heads, block_h), tokens, num_splits)
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
        width,
        kv.shape[0],
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        triton.cdiv(tiles, num_splits),
        num_splits,
        float(softmax_scale) * _LOG2E,
        HAS_SINK=sink is not None,
        RETURN_LSE=lse is not None,
        DIM=dim,
        BLOCK_H=block_h,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    if num_splits > 1:
        _dsv4_sparse_mma_combine_kernel[(tokens, heads)](
            part_acc,
            part_ml,
            out,
            lse if lse is not None else out,
            heads,
            num_splits,
            out.stride(0),
            out.stride(1),
            RETURN_LSE=lse is not None,
            DIM=dim,
            BLOCK_S=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
