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
"""CuTe DSL tensor-core DeepSeek V4 selected attention (SM120).

The kernel is the BF16 path of the DSA CuTe DSL core
(``dsa/_cute_dsl/sparse_mma.py``) in its V4 variant: a 512-wide query (rank 512,
no RoPE columns) over the rows of a flat BF16 workspace, where a V4 latent row
is key and value at once. What the variant changes:

* The workspace is a flat row table (a page of one row), so a row id addresses
  ``kv + id * row_stride`` with no divide. Row ids are int32 or int64 and are
  checked against the workspace in the kernel: a negative id or one past the
  last row is skipped, as in ``_triton/sparse_mma.py``.
* The query and output are addressed with their own token and head strides.
* The per-head sink is one extra zero-valued key and enters as the initial
  state of chunk 0 (``m`` = sink logit, ``l`` = 1), so it is the first thing the
  fold sees and rows with no live key still return the sink-only state. It
  joins the denominator only.

A 16-head CTA runs 8 warps (``cwarps``) and a 32-head CTA 8 warps as two head
groups of 4. The 16-head CTA splits the rows of the score tiles between two
warps per channel quarter and the output channels between all 8, so it adds
the same four channel quarters in the same order: its rows are bitwise those of
a 4-warp CTA.

Numerics contract and batch invariance are the Triton leaf's: BF16 scores and
probabilities accumulated in FP32, an FP32 online softmax per chunk, and a
selection cut into at most ``MAX_CHUNKS`` chunks of whole tiles whose number
depends on the width alone, folded in order with explicit ``fma`` either in the
CTA or, when the batch cannot fill the GPU, by the combine kernel from stored
chunk states. A row is bitwise the same for every batch, head tile and split.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel.ops.attention.dsa._cute_dsl import sparse_mma as _core

__all__ = [
    "MAX_CHUNKS",
    "num_chunks_for",
    "paged_attention",
    "plan_launch",
    "plan_paged",
    "sparse_mma_attention",
    "supports",
    "supports_paged",
]

HEAD_DIM = 512
ROWS = _core.rows_for(False)
# Chunks per token selection; a split launch gives each its own CTA.
MAX_CHUNKS = 4


def num_chunks_for(width: int) -> tuple[int, int]:
    """``(chunk_tiles, num_chunks)`` for a selection ``width`` columns wide.

    A function of the width alone, so every batch size cuts a token's
    selection at the same columns.
    """
    tiles = max(-(-width // ROWS), 1)
    chunk_tiles = -(-tiles // MAX_CHUNKS)
    return chunk_tiles, -(-tiles // chunk_tiles)


# Measured on an RTX PRO 6000 (188 SMs), CUDA graph, one token, 1-4 splits of
# widths 128 / 640 / 1152: a CTA takes ``_FIXED_US`` (query load, prologue,
# stores) plus ``_TILE_US`` per 32-row tile of its chunks, and a split CTA about
# a microsecond more in all (its stored states and the combine kernel). The
# dequantizing launches (:func:`paged_attention`) add the conversion of each
# tile to the tile time: "v4" is FP8 rows with a BF16 RoPE tail, "v41" the
# FP8 window and FP4 compressed rows (4 + 1 to 36 tiles, 1 / 2 / 4 splits,
# widths 128-1152).
_FIXED_US = {
    None: {16: 5.0, 32: 7.8},
    "v4": {16: 5.3, 32: 8.2},
    "v41": {16: 6.6, 32: 10.0},
}
_TILE_US = {
    None: {16: 1.44, 32: 1.9},
    "v4": {16: 1.45, 32: 2.2},
    "v41": {16: 1.9, 32: 2.63},
}
_SPLIT_CTA_US = 0.5
_COMBINE_US = 0.5
_PLAN_MARGIN = (
    1.05  # a plan replaces the base plan only if the model puts it this far ahead
)
_MAX_PLANNED_WAVES = 64  # beyond this the last partial wave is below 1.6%


def _base_plan(tokens: int, heads: int, num_sms: int, num_chunks: int):
    """``(block_h, num_splits)`` of the uniform rule: a launch is latency bound
    until about two thirds of the SMs hold a CTA, so the split aims at that
    many CTAs, one chunk each. A 16-head CTA has half the warps of a 32-head one
    on a 4-warp layout and loses to it once the batch has more than 32 of them.
    """
    narrow = -(-heads // 16) * tokens
    block_h = 16 if heads <= 16 or narrow <= 32 else 32
    programs = -(-heads // block_h) * tokens
    target = num_sms * 2 // 3
    return block_h, min(max(round(target / programs), 1), num_chunks)


@functools.lru_cache(maxsize=4096)
def plan_launch(
    tokens: int, heads: int, num_sms: int, width: int, packed: str | None = None
) -> tuple[int, int, int, int]:
    """``(block_h, cwarps, num_splits, full_tokens)`` for ``tokens`` queries.

    No value changes the result. The launch runs the programs (token, head
    tile) of the first ``full_tokens`` tokens in one CTA each and shares every
    later program among ``num_splits`` CTAs, each taking a run of chunks whose
    states a combine kernel folds (see ``dsa._cute_dsl.sparse_mma``). A CTA
    fills an SM, so a launch runs in waves of ``num_sms`` CTAs; a model of the
    waves (:func:`_core._launch_time`, the constants above) picks the head tile
    (16 heads in 8 warps, or 32 in two groups of 4) and the split, and a
    stream-K tail that fills the SMs the last wave of whole CTAs leaves idle.
    A plan replaces the uniform rule (:func:`_base_plan`) only if the model puts
    it ``_PLAN_MARGIN`` ahead of it. ``packed`` (None, "v4" or "v41"; see
    :func:`plan_paged`) picks the tile times of a launch that dequantizes packed
    rows.
    """
    chunk_tiles, num_chunks = num_chunks_for(width)
    tiles = chunk_tiles * num_chunks
    fixed_us, tile_us = _FIXED_US[packed], _TILE_US[packed]

    def modelled(block_h: int, splits: int, full_tokens: int) -> float:
        head_tiles = -(-heads // block_h)
        unsplit = fixed_us[block_h] + tiles * tile_us[block_h]
        return _core._launch_time(
            head_tiles * tokens,
            head_tiles * full_tokens,
            splits,
            num_sms,
            unsplit,
            fixed_us[block_h],
            _SPLIT_CTA_US,
            _COMBINE_US,
        )

    block_h, splits = _base_plan(tokens, heads, num_sms, num_chunks)
    best = (block_h, splits, 0 if splits > 1 else tokens)
    cwarps = lambda block_h: (
        2 * _core.CHANNEL_WARPS if block_h == 16 else _core.CHANNEL_WARPS
    )
    if tokens * -(-heads // 16) <= _MAX_PLANNED_WAVES * num_sms:
        base_us = best_us = modelled(*best)
        for tile_h in (16, 32) if heads > 16 else (16,):
            head_tiles = -(-heads // tile_h)
            programs = head_tiles * tokens
            tails = {programs % num_sms + num_sms * j for j in (0, 1, 2)}
            tails |= {num_sms * m // 8 for m in range(1, 17)}
            for splits in (1, 2, 4):
                if splits > num_chunks:
                    continue
                options = [tokens] if splits == 1 else [0]
                if splits > 1:
                    options += [
                        tokens - tail_tokens
                        for tail_tokens in sorted(
                            {-(-tail // head_tiles) for tail in tails}
                        )
                        if 0 < tail_tokens < tokens
                    ]
                for full_tokens in options:
                    t = modelled(tile_h, splits, full_tokens)
                    if t < best_us and t * _PLAN_MARGIN < base_us:
                        best_us = t
                        best = (tile_h, splits, full_tokens)
    return best[0], cwarps(best[0]), best[1], best[2]


def plan_paged(
    tokens: int, heads: int, num_sms: int, swa, extra
) -> tuple[int, int, int, int]:
    """:func:`plan_launch` for a :func:`paged_attention` call over ``swa`` and ``extra``.

    The tile times follow the cache formats alone (FP8 V4 rows, or the V4.1
    FP8 window and FP4 compressed rows), never the batch.
    """
    formats = {_PACKED[segment[1]][0] for segment in (swa, extra) if segment}
    width = swa[2].shape[1] + (0 if extra is None else extra[2].shape[1])
    return plan_launch(
        tokens, heads, num_sms, width, "v4" if formats == {"fp8_v4"} else "v41"
    )


def supports(
    q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor, indices: torch.Tensor
) -> bool:
    """Whether the kernel's loads and stores are aligned for these tensors.

    The query is read in 4-byte words, the output written in 8-byte words and
    the workspace rows gathered in 16-byte ``cp.async`` copies. The answer
    depends on the tensors' layout, never on the batch contents.
    """
    kv_row_bytes = kv.stride(0) * kv.element_size()
    return (
        q.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM
        and q.stride(-1) == 1
        and out.stride(-1) == 1
        and kv.stride(-1) == 1
        and q.data_ptr() % 4 == 0
        and q.stride(0) % 2 == 0
        and q.stride(1) % 2 == 0
        and out.data_ptr() % 8 == 0
        and out.stride(0) % 4 == 0
        and out.stride(1) % 4 == 0
        and kv.data_ptr() % 16 == 0
        and kv_row_bytes % 16 == 0
        and kv.shape[0] < 2**31
        and indices.stride(-1) == 1
    )


def _origin(tensor: torch.Tensor) -> torch.Tensor:
    """A one-element view at ``tensor``'s first element: the kernel only needs
    its address, and the tensor itself may be a strided view."""
    return tensor.as_strided((1,), (1,))


# Cache format names of ``dsv4/_triton/paged_mma.py`` -> packed row formats of
# the CuTe core, with the bytes of one packed row.
_PACKED = {
    "swa": ("fp8_g32", 528),
    "swa_v4": ("fp8_v4", 584),
    "global_v4": ("fp8_v4", 584),
    "global": ("fp4_g16", 288),
}


def _segment_layout(segment) -> tuple[int, int] | None:
    """``(page stride in bytes, rows in the cache)`` of a packed segment the
    kernel can read, or None.

    ``segment`` is ``(cache, format, slots, lens, page_rows)`` as in
    :func:`paged_attention`. The rows of a page must be contiguous bytes (a
    page-planar page with a unit byte stride), 16-byte aligned, and the slots a
    row-contiguous int32/int64 matrix; anything else is read by the Triton
    reader (the choice follows the layout alone).
    """
    cache, fmt, slots, lens, page_rows = segment
    row_bytes = _PACKED[fmt][1]
    if (
        cache.dtype != torch.uint8
        or cache.dim() not in (2, 3)
        or page_rows < 1
        or page_rows & (page_rows - 1)
    ):
        return None
    if cache.dim() == 3:
        if cache.shape[1:] != (page_rows, row_bytes):
            return None
        if cache.stride(2) != 1 or cache.stride(1) != row_bytes:
            return None
    elif cache.shape[1] != page_rows * row_bytes or cache.stride(1) != 1:
        return None
    page_stride = cache.stride(0)
    if (
        cache.data_ptr() % 16
        or page_stride % 16
        or cache.shape[0] * page_rows >= 2**31
        or slots.dtype not in (torch.int32, torch.int64)
        or slots.dim() != 2
        or slots.stride(-1) != 1
        or lens.dtype not in (torch.int32, torch.int64)
        or lens.dim() != 1
        or lens.stride(0) != 1
    ):
        return None
    return page_stride, cache.shape[0] * page_rows


def supports_paged(q: torch.Tensor, out: torch.Tensor, swa, extra) -> bool:
    """Whether :func:`paged_attention` can read these tensors.

    The query and output are addressed as in :func:`supports`; each cache
    segment must satisfy :func:`_segment_layout`, the first segment's width
    must be a whole number of tiles when a second follows (a tile belongs to
    one segment), and the selection must not be empty.
    """
    if (
        q.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or q.dim() != 3
        or q.shape[-1] != HEAD_DIM
        or q.stride(-1) != 1
        or out.stride(-1) != 1
        or q.data_ptr() % 4
        or q.stride(0) % 2
        or q.stride(1) % 2
        or out.data_ptr() % 8
        or out.stride(0) % 4
        or out.stride(1) % 4
    ):
        return False
    if _segment_layout(swa) is None:
        return False
    width = swa[2].shape[1]
    if extra is not None:
        if (
            _segment_layout(extra) is None
            or width % ROWS
            or extra[2].dtype != swa[2].dtype
        ):
            return False
        width += extra[2].shape[1]
    return width > 0 and swa[2].shape[0] == q.shape[0] == swa[3].shape[0]


def paged_attention(
    q: torch.Tensor,
    swa,
    extra,
    sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    *,
    softmax_scale: float,
    block_h: int,
    cwarps: int,
    num_splits: int,
    full_tokens: int,
) -> None:
    """Attend ``q`` over the rows of packed page-planar caches, dequantizing them
    while gathering (no BF16 workspace).

    The result is bitwise that of dequantizing the rows into a flat BF16
    workspace and calling :func:`sparse_mma_attention` (the rows the workspace
    holds are exactly the rows this kernel builds in shared memory, and the
    loop is the same).

    Args:
        q: ``[tokens, heads, 512]`` BF16 with unit inner stride.
        swa, extra: ``(cache, format, slots, lens, page_rows)`` for the first
            segment (the sliding window: ``swa``, ``swa_v4``) and an optional
            second one (``global``, ``global_v4``); ``cache`` is the uint8
            page-planar cache, ``format`` one of :data:`_PACKED`, ``slots`` an
            int32/int64 ``[tokens, width]`` matrix, ``lens`` the live length of
            each token's segment and ``page_rows`` the rows per page. A slot that
            is negative or past the cache, or a column past the length, is an
            empty row. The selection is the first segment's columns followed by
            the second's, cut into tiles of 32 columns.
        sink, out, lse, block_h, cwarps, num_splits, full_tokens: as
            :func:`sparse_mma_attention`.
    """
    tokens, heads, dim = q.shape
    device = q.device
    if tokens == 0 or heads == 0:
        return
    swa_cache, swa_format, swa_slots, swa_lens, swa_rows = swa
    swa_page_stride, swa_capacity = _segment_layout(swa)
    swa_width = swa_slots.shape[1]
    first_format = _PACKED[swa_format][0]
    width = swa_width
    second_format, extra_wide_ids, extra_wide_lens = None, False, False
    if extra is not None:
        extra_format, extra_slots = extra[1], extra[2]
        second_format = _PACKED[extra_format][0]
        extra_wide_ids = extra_slots.dtype == torch.int64
        extra_wide_lens = extra[3].dtype == torch.int64
        width += extra_slots.shape[1]
    chunk_tiles, num_chunks = num_chunks_for(width)
    num_splits = max(min(num_splits, num_chunks), 1)
    if not 0 <= full_tokens <= tokens:
        raise ValueError(f"full_tokens {full_tokens} outside [0, {tokens}]")
    if num_splits == 1:
        full_tokens = tokens
    tail_tokens = tokens - full_tokens
    if full_tokens + tail_tokens * num_splits > _core.MAX_GRID_ROWS:
        raise ValueError(
            f"{full_tokens} whole + {tail_tokens} split tokens x {num_splits} "
            f"splits exceed the {_core.MAX_GRID_ROWS} rows of a grid"
        )
    split = tail_tokens > 0
    head_tiles = -(-heads // block_h)
    if split:
        part_acc = torch.empty(
            (tail_tokens, num_chunks, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tail_tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = lse if lse is not None else torch.empty(4, device=device)
    lse_arg = lse if lse is not None else part_ml
    swa_lens = swa_lens.contiguous()
    if extra is None:
        # Unused second segment: int32 buffers of the compiled dtypes.
        unused = part_ml.reshape(-1).view(torch.int32)
        extra_args = (0, unused, unused, 0, 0, 1, 0)
    else:
        extra_page_stride, extra_capacity = _segment_layout(extra)
        extra_args = (
            extra[0].data_ptr(),
            _origin(extra_slots),
            extra[3].contiguous(),
            extra_slots.stride(0),
            extra_page_stride,
            extra[4].bit_length() - 1,
            extra_capacity,
        )
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    kernel = _core.compiled_attention(
        False,
        block_h,
        dim,
        0,
        device_index,
        (
            sink is not None,
            swa_slots.dtype == torch.int64,
            swa_lens.dtype == torch.int64,
        ),
        cwarps,
        (first_format, second_format, extra_wide_ids, extra_wide_lens),
    )
    out_stride_token = out.stride(0) * 2
    out_stride_head = out.stride(1) * 2
    kernel(
        q.data_ptr(),
        swa_cache.data_ptr(),
        _origin(swa_slots),
        swa_lens,
        out.data_ptr(),
        lse_arg.reshape(-1),
        part_acc.reshape(-1),
        part_ml.reshape(-1),
        heads,
        width,
        swa_slots.stride(0),
        swa_page_stride,
        0,
        swa_rows.bit_length() - 1,
        chunk_tiles,
        num_chunks,
        -(-num_chunks // num_splits),
        full_tokens,
        num_splits,
        _core.split_magic(num_splits),
        int(lse is not None),
        float(softmax_scale) * 1.4426950408889634,
        swa_capacity,
        sink if sink is not None else part_ml.reshape(-1),
        q.stride(0) * 2,
        q.stride(1) * 2,
        out_stride_token,
        out_stride_head,
        head_tiles,
        full_tokens + tail_tokens * num_splits,
        *extra_args,
        swa_width,
    )
    if split:
        _core.compiled_combine(dim, device_index, True)(
            part_acc.reshape(-1),
            part_ml.reshape(-1),
            out.data_ptr(),
            lse_arg.reshape(-1),
            heads,
            num_chunks,
            int(lse is not None),
            out_stride_token,
            out_stride_head,
            full_tokens,
            tail_tokens,
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
    cwarps: int,
    num_splits: int,
    full_tokens: int,
) -> None:
    """Attend ``q`` over the workspace rows named by ``indices``.

    Args:
        q: ``[tokens, heads, 512]`` BF16 with unit inner stride.
        kv: ``[rows, 512]`` BF16 workspace with unit inner stride
            (:func:`supports` holds).
        indices: ``[tokens, width]`` int32/int64 row ids with unit column
            stride; entries that are negative or past the last row are skipped.
        lens: ``[tokens]`` int32/int64 count of leading live columns.
        sink: ``[heads]`` contiguous FP32 sink logits or None.
        out: ``[tokens, heads, 512]`` BF16 result with unit inner stride.
        lse: ``[tokens, heads]`` contiguous FP32 natural-log normalizer or None.
        block_h: heads per CTA (16 or 32).
        cwarps: warps per head group (4, or 8 with ``block_h`` 16).
        num_splits, full_tokens: the first ``full_tokens`` tokens run each head
            tile in one CTA and fold its chunks in registers; every later token
            shares each head tile among ``num_splits`` CTAs, each taking a run
            of chunks whose stored states a combine kernel folds.
            ``full_tokens`` equal to the token count is an unsplit launch, 0
            splits every token. The result is bitwise the same for every
            value of the three.
    """
    tokens, heads, dim = q.shape
    device = q.device
    if tokens == 0 or heads == 0:
        return
    chunk_tiles, num_chunks = num_chunks_for(indices.shape[1])
    num_splits = max(min(num_splits, num_chunks), 1)
    if not 0 <= full_tokens <= tokens:
        raise ValueError(f"full_tokens {full_tokens} outside [0, {tokens}]")
    if num_splits == 1:
        full_tokens = tokens
    tail_tokens = tokens - full_tokens
    if full_tokens + tail_tokens * num_splits > _core.MAX_GRID_ROWS:
        raise ValueError(
            f"{full_tokens} whole + {tail_tokens} split tokens x {num_splits} "
            f"splits exceed the {_core.MAX_GRID_ROWS} rows of a grid"
        )
    split = tail_tokens > 0
    head_tiles = -(-heads // block_h)
    if split:
        # States are [tail token][chunk][head].
        part_acc = torch.empty(
            (tail_tokens, num_chunks, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tail_tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = lse if lse is not None else torch.empty(4, device=device)
    lse_arg = lse if lse is not None else part_ml
    lens = lens.contiguous()
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    kernel = _core.compiled_attention(
        False,
        block_h,
        dim,
        0,
        device_index,
        (sink is not None, indices.dtype == torch.int64, lens.dtype == torch.int64),
        cwarps,
    )
    # The kernel counts strides in bytes.
    out_stride_token = out.stride(0) * 2
    out_stride_head = out.stride(1) * 2
    kernel(
        q.data_ptr(),
        kv.data_ptr(),
        _origin(indices),
        lens,
        out.data_ptr(),
        lse_arg.reshape(-1),
        part_acc.reshape(-1),
        part_ml.reshape(-1),
        heads,
        indices.shape[1],
        indices.stride(0),
        kv.stride(0) * 2,
        0,
        1,
        chunk_tiles,
        num_chunks,
        -(-num_chunks // num_splits),
        full_tokens,
        num_splits,
        _core.split_magic(num_splits),
        int(lse is not None),
        float(softmax_scale) * 1.4426950408889634,
        kv.shape[0],
        sink if sink is not None else part_ml.reshape(-1),
        q.stride(0) * 2,
        q.stride(1) * 2,
        out_stride_token,
        out_stride_head,
        head_tiles,
        full_tokens + tail_tokens * num_splits,
        0,
        part_ml.reshape(-1).view(torch.int32),
        part_ml.reshape(-1).view(torch.int32),
        0,
        0,
        1,
        0,
        0,
    )
    if split:
        _core.compiled_combine(dim, device_index, True)(
            part_acc.reshape(-1),
            part_ml.reshape(-1),
            out.data_ptr(),
            lse_arg.reshape(-1),
            heads,
            num_chunks,
            int(lse is not None),
            out_stride_token,
            out_stride_head,
            full_tokens,
            tail_tokens,
        )
