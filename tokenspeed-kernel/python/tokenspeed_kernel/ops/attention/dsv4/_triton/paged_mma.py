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

"""Tensor-core DeepSeek V4 / V4.1 selected attention over paged packed caches (SM120).

The portable decode path gathers every selected cache row into a BF16
workspace (a launch per segment, a ``cat``, mask and index construction) and
then runs the dense kernel of ``sparse_mma.py``. That is 640 KB of workspace
per query at 640 selected rows, written and read back once. This kernel reads
the packed rows itself: a gathered tile of 32 selected rows is dequantized in
registers into the same BF16 tile the workspace would hold, and the rest of the
loop is ``sparse_mma._dsv4_sparse_mma_kernel`` operation for operation, so the
result is bitwise the workspace route's. The rows are never rounded: an E4M3
or E2M1 value times a power of two or an E4M3 scale has at most four
significant bits, which BF16 holds exactly.

Two segments are read, each from its own cache and format: the sliding window
(``slots`` of the SWA cache) and the compressed rows. The selection is their
concatenation, ``swa_width`` columns then ``extra_width``, cut into tiles of
32 columns and into at most 4 chunks by the total width alone, like the
dense kernel; ``swa_width`` must be a multiple of the tile so a tile belongs to
one segment.

Formats (page-planar: a page holds ``PAGE_ROWS`` data rows, then ``PAGE_ROWS``
scale rows; byte offsets honor strided views like the gather kernel's
``_planar_offset``):

* ``FP8_G32`` (V4.1 ``swa``): 512 E4M3 values, one UE8M0 scale per 32.
* ``FP8_V4`` (``swa_v4`` / ``global_v4`` and the ``dsv4_decode`` cache): 448 E4M3
  values with a UE8M0 scale per 64, then 64 BF16 RoPE values.
* ``FP4_G16`` (V4.1 ``global``): 512 E2M1 values in nibbles, one E4M3 scale per 16.
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
from tokenspeed_kernel.ops.attention.dsv4._triton.sparse_mma import (
    _dsv4_sparse_mma_combine_kernel,
    _merge,
    num_chunks_for,
)
from tokenspeed_kernel.platform import current_platform

_LOG2E = 1.4426950408889634
HEAD_DIM = 512

FP8_G32 = 0
FP8_V4 = 1
FP4_G16 = 2

# (quantization group, value bytes, quantized dims, scale bytes) per format;
# the row also carries the BF16 RoPE tail of the dims past the quantized ones.
_FORMATS = {
    "swa": (FP8_G32, 32, 512, 512, 16),
    "swa_v4": (FP8_V4, 64, 448, 448, 8),
    "global_v4": (FP8_V4, 64, 448, 448, 8),
    "global": (FP4_G16, 16, 256, 512, 32),
}
ROW_BYTES = {"swa": 528, "swa_v4": 584, "global_v4": 584, "global": 288}


@triton.jit
def _planar_offset(byte, CR: tl.constexpr, CB: tl.constexpr, ROW_BYTES: tl.constexpr):
    # Same mapping as the cache gather: the field keeps its [pages, rows,
    # row_bytes] shape, so a strided view must be honored.
    if CR == ROW_BYTES * CB:
        return byte * CB
    return (byte // ROW_BYTES) * CR + (byte % ROW_BYTES) * CB


@triton.jit
def _e2m1_decode(code):
    # The V4.1 codec of ``dsv41/triton.py``.
    a = code & 7
    value = tl.where(
        a < 4,
        a.to(tl.float32) * 0.5,
        tl.where(a == 4, 2.0, tl.where(a == 5, 3.0, tl.where(a == 6, 4.0, 6.0))),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _read_rows(
    cache,
    slot,
    valid,
    cache_page_stride,
    page_rows,
    FORMAT: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    QUANTIZED: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    CR: tl.constexpr,
    CB: tl.constexpr,
    DIM: tl.constexpr,
):
    """Dequantize the rows ``slot`` (``valid`` ones; the rest read as zeros).

    Returns a BF16 ``[len(slot), DIM]`` tile.
    """
    ROPE: tl.constexpr = DIM - QUANTIZED
    DATA_BYTES: tl.constexpr = VALUES + ROPE * 2
    ROW: tl.constexpr = DATA_BYTES + SCALE_BYTES
    safe = tl.where(valid, slot, 0)
    base = cache + (safe // page_rows) * cache_page_stride
    page_row = safe % page_rows
    d = tl.arange(0, DIM)
    base = base[:, None]
    page_row = page_row[:, None]
    live = valid[:, None]
    scale_offset = (
        page_rows * DATA_BYTES + page_row * SCALE_BYTES + (d // GROUP)[None, :]
    )
    if FORMAT == 2:
        byte = tl.load(
            base
            + _planar_offset(page_row * DATA_BYTES + (d // 2)[None, :], CR, CB, ROW),
            mask=live,
            other=0,
        ).to(tl.int32)
        scale_byte = tl.load(
            base + _planar_offset(scale_offset, CR, CB, ROW), mask=live, other=0
        )
        scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        value = _e2m1_decode((byte >> ((d % 2) * 4)[None, :]) & 15) * scale
    else:
        quantized = live & (d < QUANTIZED)[None, :]
        byte = tl.load(
            base + _planar_offset(page_row * DATA_BYTES + d[None, :], CR, CB, ROW),
            mask=quantized,
            other=0,
        )
        scale_byte = tl.load(
            base + _planar_offset(scale_offset, CR, CB, ROW), mask=quantized, other=0
        )
        # E8M0 byte 0 represents 2**-127 (a float32 subnormal).
        scale = tl.where(
            scale_byte == 0,
            2.0**-127,
            (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
        )
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
        if ROPE > 0:
            # BF16 halves, little-endian, right after the quantized values.
            rope = live & (d >= QUANTIZED)[None, :]
            rope_byte = (
                page_row * DATA_BYTES
                + VALUES
                + (tl.maximum(d - QUANTIZED, 0) * 2)[None, :]
            )
            lo = tl.load(
                base + _planar_offset(rope_byte, CR, CB, ROW), mask=rope, other=0
            )
            hi = tl.load(
                base + _planar_offset(rope_byte + 1, CR, CB, ROW), mask=rope, other=0
            )
            word = (hi.to(tl.uint16) << 8) | lo.to(tl.uint16)
            value = tl.where(
                (d < QUANTIZED)[None, :],
                value,
                word.to(tl.bfloat16, bitcast=True).to(tl.float32),
            )
    return tl.where(live, value, 0.0).to(tl.bfloat16)


@triton.jit
def _tile_slots(
    swa_slots_row,
    swa_lens_value,
    extra_slots_row,
    extra_lens_value,
    tile,
    tile_end,
    width,
    swa_width,
    swa_capacity,
    extra_capacity,
    BLOCK_N: tl.constexpr,
):
    """Cache slots of one tile of the selection, and which of them are live.

    A column is live when it is inside its segment's length, names a slot and
    that slot lies in the cache: the conditions of the portable gather.
    """
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    in_range = (cols < width) & (tile < tile_end)
    if tile * BLOCK_N < swa_width:
        slot = tl.load(swa_slots_row + cols, mask=in_range, other=-1).to(tl.int64)
        live = in_range & (cols < swa_lens_value) & (slot >= 0) & (slot < swa_capacity)
    else:
        local = cols - swa_width
        slot = tl.load(extra_slots_row + local, mask=in_range, other=-1).to(tl.int64)
        live = (
            in_range
            & (local < extra_lens_value)
            & (slot >= 0)
            & (slot < extra_capacity)
        )
    return slot, live


# Values that change with the batch are runtime arguments that Triton must not
# specialize on (== 1, divisible by 16), or each new batch would recompile.
@triton.jit(
    do_not_specialize=[
        "num_heads",
        "width",
        "swa_width",
        "swa_capacity",
        "extra_capacity",
        "swa_page_rows",
        "extra_page_rows",
        "swa_slots_stride",
        "extra_slots_stride",
        "chunk_tiles",
        "num_chunks",
    ]
)
def _paged_mma_kernel(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    extra_cache,
    extra_slots,
    extra_lens,
    sink,
    out,
    lse,
    part_acc,
    part_ml,
    num_heads,
    width,
    swa_width,
    swa_capacity,
    extra_capacity,
    swa_page_stride,
    extra_page_stride,
    swa_page_rows,
    extra_page_rows,
    q_stride_token,
    q_stride_head,
    swa_slots_stride,
    extra_slots_stride,
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
    SWA_FORMAT: tl.constexpr,
    SWA_GROUP: tl.constexpr,
    SWA_VALUES: tl.constexpr,
    SWA_QUANTIZED: tl.constexpr,
    SWA_SCALE_BYTES: tl.constexpr,
    SWA_CR: tl.constexpr,
    SWA_CB: tl.constexpr,
    EXTRA_FORMAT: tl.constexpr,
    EXTRA_GROUP: tl.constexpr,
    EXTRA_VALUES: tl.constexpr,
    EXTRA_QUANTIZED: tl.constexpr,
    EXTRA_SCALE_BYTES: tl.constexpr,
    EXTRA_CR: tl.constexpr,
    EXTRA_CB: tl.constexpr,
):
    head_block = tl.program_id(0)
    token = tl.program_id(1)
    split = tl.program_id(2)

    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    dim_offsets = tl.arange(0, DIM)

    num_tiles = tl.cdiv(width, BLOCK_N)
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
    swa_slots_row = swa_slots + token.to(tl.int64) * swa_slots_stride
    extra_slots_row = extra_slots + token.to(tl.int64) * extra_slots_stride
    swa_len = tl.load(swa_lens + token).to(tl.int32)
    extra_len = tl.load(extra_lens + token).to(tl.int32)

    if SPLIT:
        chunk_begin = split
        chunk_end = split + 1
    else:
        chunk_begin = 0
        chunk_end = tl.maximum(tl.cdiv(num_tiles, chunk_tiles), 1)
    m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_run = tl.zeros((BLOCK_H,), tl.float32)
    acc_run = tl.zeros((BLOCK_H, DIM), tl.float32)
    for chunk in range(chunk_begin, chunk_end):
        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        if HAS_SINK:
            if chunk == 0:
                m_i = sink_logit
                l_i = tl.where(head_mask, 1.0, 0.0)
        acc = tl.zeros((BLOCK_H, DIM), tl.float32)
        tile_begin = chunk * chunk_tiles
        tile_end = tl.minimum(tile_begin + chunk_tiles, num_tiles)
        slot, live = _tile_slots(
            swa_slots_row,
            swa_len,
            extra_slots_row,
            extra_len,
            tile_begin,
            tile_end,
            width,
            swa_width,
            swa_capacity,
            extra_capacity,
            BLOCK_N,
        )
        for tile in range(tile_begin, tile_end):
            if tile * BLOCK_N < swa_width:
                k = _read_rows(
                    swa_cache,
                    slot,
                    live,
                    swa_page_stride,
                    swa_page_rows,
                    SWA_FORMAT,
                    SWA_GROUP,
                    SWA_VALUES,
                    SWA_QUANTIZED,
                    SWA_SCALE_BYTES,
                    SWA_CR,
                    SWA_CB,
                    DIM,
                )
            else:
                k = _read_rows(
                    extra_cache,
                    slot,
                    live,
                    extra_page_stride,
                    extra_page_rows,
                    EXTRA_FORMAT,
                    EXTRA_GROUP,
                    EXTRA_VALUES,
                    EXTRA_QUANTIZED,
                    EXTRA_SCALE_BYTES,
                    EXTRA_CR,
                    EXTRA_CB,
                    DIM,
                )
            valid = live
            slot, live = _tile_slots(
                swa_slots_row,
                swa_len,
                extra_slots_row,
                extra_len,
                tile + 1,
                tile_end,
                width,
                swa_width,
                swa_capacity,
                extra_capacity,
                BLOCK_N,
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


def paged_mma_attention(
    q: torch.Tensor,
    swa: tuple[torch.Tensor, str, torch.Tensor, torch.Tensor, int],
    extra: tuple[torch.Tensor, str, torch.Tensor, torch.Tensor, int] | None,
    sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    *,
    softmax_scale: float,
    block_h: int,
    split: bool,
) -> None:
    """Attend ``q`` over the SWA rows and optional compressed rows of paged caches.

    Args:
        q: ``[tokens, heads, 512]`` BF16 with unit inner stride.
        swa, extra: ``(cache, format, slots, lens, page_rows)``: the uint8
            cache (``[pages, rows, row_bytes]`` or ``[pages, rows * row_bytes]``),
            its format name (``swa``, ``swa_v4``, ``global_v4``, ``global``),
            int32/int64 ``[tokens, width]`` slots, int32/int64 ``[tokens]``
            live lengths and rows per page. Negative slots, slots past the
            cache and columns past the length do not contribute. ``extra`` may
            be None.
        sink: ``[heads]`` FP32 sink logits or None.
        out: ``[tokens, heads, 512]`` BF16 result with unit inner stride.
        lse: ``[tokens, heads]`` contiguous FP32 natural-log normalizer or None.
        block_h, split: as ``sparse_mma.sparse_mma_attention``; neither changes
            the result.
    """
    tokens, heads, dim = q.shape
    device = q.device
    swa_cache, swa_fmt, swa_slots, swa_lens, swa_rows = swa
    if extra is None:
        extra_cache, extra_fmt, extra_slots, extra_lens, extra_rows = (
            swa_cache,
            swa_fmt,
            swa_slots[:, :0],
            swa_lens,
            swa_rows,
        )
    else:
        extra_cache, extra_fmt, extra_slots, extra_lens, extra_rows = extra
    swa_width = swa_slots.shape[1]
    width = swa_width + extra_slots.shape[1]
    chunk_tiles, num_chunks = num_chunks_for(width)
    if split:
        part_acc = torch.empty(
            (tokens, num_chunks, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = out

    def describe(cache, fmt, rows):
        kind, group, values, quantized, scale_bytes = _FORMATS[fmt]
        row_bytes = ROW_BYTES[fmt]
        if cache.ndim == 3:
            page_stride, row_stride, byte_stride = cache.stride()
        else:  # [pages, rows * row_bytes]: contiguous planar bytes
            page_stride, byte_stride = cache.stride()
            row_stride = row_bytes * byte_stride
        return (
            dict(
                GROUP=group,
                VALUES=values,
                QUANTIZED=quantized,
                SCALE_BYTES=scale_bytes,
                CR=row_stride,
                CB=byte_stride,
            ),
            kind,
            page_stride,
            cache.shape[0] * rows,
        )

    swa_const, swa_kind, swa_page_stride, swa_capacity = describe(
        swa_cache, swa_fmt, swa_rows
    )
    extra_const, extra_kind, extra_page_stride, extra_capacity = describe(
        extra_cache, extra_fmt, extra_rows
    )
    grid = (triton.cdiv(heads, block_h), tokens, num_chunks if split else 1)
    _paged_mma_kernel[grid](
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        extra_cache,
        extra_slots,
        extra_lens,
        sink if sink is not None else out,
        out,
        lse if lse is not None else out,
        part_acc,
        part_ml,
        heads,
        width,
        swa_width,
        swa_capacity,
        extra_capacity,
        swa_page_stride,
        extra_page_stride,
        swa_rows,
        extra_rows,
        q.stride(0),
        q.stride(1),
        swa_slots.stride(0),
        extra_slots.stride(0),
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
        SWA_FORMAT=swa_kind,
        SWA_GROUP=swa_const["GROUP"],
        SWA_VALUES=swa_const["VALUES"],
        SWA_QUANTIZED=swa_const["QUANTIZED"],
        SWA_SCALE_BYTES=swa_const["SCALE_BYTES"],
        SWA_CR=swa_const["CR"],
        SWA_CB=swa_const["CB"],
        EXTRA_FORMAT=extra_kind,
        EXTRA_GROUP=extra_const["GROUP"],
        EXTRA_VALUES=extra_const["VALUES"],
        EXTRA_QUANTIZED=extra_const["QUANTIZED"],
        EXTRA_SCALE_BYTES=extra_const["SCALE_BYTES"],
        EXTRA_CR=extra_const["CR"],
        EXTRA_CB=extra_const["CB"],
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


def paged_mma_decode(
    q: torch.Tensor,
    swa: tuple[torch.Tensor, str, torch.Tensor, torch.Tensor, int],
    extra: tuple[torch.Tensor, str, torch.Tensor, torch.Tensor, int] | None,
    sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    *,
    softmax_scale: float,
) -> None:
    """:func:`paged_mma_attention` with the launch rule of the dense kernel."""
    tokens, heads, _ = q.shape
    if sink is not None:
        if sink.numel() < heads:
            raise ValueError("attention sink must provide one value per query head")
        sink = sink.reshape(-1)[:heads].to(torch.float32).contiguous()
    width = swa[2].shape[1] + (0 if extra is None else extra[2].shape[1])
    block_h, num_splits = plan_launch(tokens, heads, current_platform().sm_count)
    paged_mma_attention(
        q,
        swa,
        extra,
        sink,
        out,
        lse,
        softmax_scale=softmax_scale,
        block_h=block_h,
        split=2 * num_splits >= num_chunks_for(width)[1],
    )
