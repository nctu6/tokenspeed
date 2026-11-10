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

"""Tensor-core sparse MLA attention over a dense latent cache, in CuTe DSL (SM120).

Same contract and chunked fold as the Triton kernel (``_triton/sparse_mma.py``);
what differs is how a CTA spends its tensor pipe and shared memory.

A CTA owns ``BLOCK_H`` (16 or 32) heads of one token. The selected rows of a
tile (64 slots for E4M3, 32 for BF16) are gathered with ``cp.async`` into a
two-stage shared-memory ring while the previous tile is multiplied. Each warp
of the CTA belongs to one of ``BLOCK_H / 16`` head groups and one of four
channel groups: it multiplies 16 heads by a quarter of the latent channels in
``O += P V``, so a warp's FP32 accumulator is ``16 x RANK / 4`` and there is one
of it (plus the running fold, below).

Scores, ``S = Q K^T``:

* E4M3 (``m16n8k32``): the query tile is staged in shared memory and each warp
  scores 16 of the 64 rows over all channels with ``ldmatrix`` fragments; the
  four row slices are exchanged through shared memory. Rows are unpadded and
  XOR-swizzled in 16-byte units, which keeps every ``ldmatrix`` and V load
  free of bank conflicts and the CTA within 99 KiB.
* BF16 (``m16n8k16``): a 64-row tile and the query tile do not both fit, so
  each warp keeps its quarter of the query channels in registers, scores the
  32 rows over them, and the four partial sums are added in a fixed order.
  Rows are padded by 16 bytes for conflict-free fragment loads. The V4 variant
  can run a 16-head CTA in 8 warps (``cwarps``): the two warps of a channel
  quarter split the 32 rows between them (the same MMA sequence per row, so the
  same partial sums) and the 8 warps split the output channels of ``O += P V``
  (64 each, half the accumulator). Two warps per scheduler hide the tile chain
  better; the bits are those of the 4-warp CTA.

The FP32 online softmax runs in registers (every warp of a head group computes
the same values) and the probabilities are reused as the ``m16n8k16`` A
fragments: FP16 for an E4M3 cache (whose values are exact in FP16; the V
fragments are converted with ``cvt.rn.f16x2.e4m3x2``), BF16 otherwise. V
fragments come from ``ldmatrix.trans``.

Batch invariance is the Triton kernel's scheme: a token's selection is cut into
``num_chunks`` chunks of whole tiles fixed by the width, each chunk runs its
own softmax state, and the states are folded in order by :func:`_fold` with
explicit ``fma``. A CTA that owns all the chunks of a (token, head tile)
folds them in registers; the others store every chunk state and
:class:`_Combine` folds them with the same helpers, so a row is bitwise the
same for any batch size, head tile or split.

The launch is a stream-K schedule: the grid has one row of blocks per whole
token (a CTA per head tile) followed by ``num_splits`` rows per split token.
The hardware hands CTAs out in grid order, so the short split CTAs fill the SMs
that the last wave of whole CTAs leaves idle (:func:`plan_launch`). Which
tokens are whole or split never changes a row's bits.
Batch-varying values (token count, head count, width, strides, chunk and split
counts) are runtime arguments; a binary is keyed by dtype, head tile, rank and
RoPE width only.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass.memory import SmemAllocator

CHANNEL_WARPS = 4  # warps that split the channels of one head group
_LN2 = 0.6931471805599453
_NEG_INF = -math.inf


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------


@cute.jit
def _ldg_u32(addr):
    return cute.arch.inline_ptx(
        "ld.global.nc.b32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Uint32],
        read_only_args=[addr],
    )


@cute.jit
def _ldg_s32(addr):
    return cute.arch.inline_ptx(
        "ld.global.nc.b32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Int32],
        read_only_args=[addr],
    )


@cute.jit
def _ldg_s64(addr):
    return cute.arch.inline_ptx(
        "ld.global.nc.b64 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Int64],
        read_only_args=[addr],
    )


@cute.jit
def _ldg_f32(addr):
    return cute.arch.inline_ptx(
        "ld.global.b32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )


@cute.jit
def _ldg_v4_f32(addr):
    return cute.arch.inline_ptx(
        "ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[addr],
    )


@cute.jit
def _stg_f32(addr, v):
    cute.arch.inline_ptx("st.global.f32 [{$r0}], {$r1};", read_only_args=[addr, v])


@cute.jit
def _stg_v4_f32(addr, a, b, c, d):
    cute.arch.inline_ptx(
        "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


@cute.jit
def _stg_u32(addr, a):
    cute.arch.inline_ptx("st.global.b32 [{$r0}], {$r1};", read_only_args=[addr, a])


@cute.jit
def _stg_v2_f32(addr, a, b):
    cute.arch.inline_ptx(
        "st.global.v2.f32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[addr, a, b]
    )


@cute.jit
def _stg_v2_u32(addr, a, b):
    cute.arch.inline_ptx(
        "st.global.v2.b32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[addr, a, b]
    )


@cute.jit
def _lds_u32(addr):
    return cute.arch.inline_ptx(
        "ld.shared.b32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Uint32],
        read_only_args=[addr],
    )


@cute.jit
def _lds_v2_s32(addr):
    return cute.arch.inline_ptx(
        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[cutlass.Int32] * 2,
        read_only_args=[addr],
    )


@cute.jit
def _sts_s32(addr, v):
    cute.arch.inline_ptx("st.shared.b32 [{$r0}], {$r1};", read_only_args=[addr, v])


@cute.jit
def _lds_v4_f32(addr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[addr],
    )


@cute.jit
def _sts_v4_f32(addr, a, b, c, d):
    cute.arch.inline_ptx(
        "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


@cute.jit
def _ldsm_x4(addr):
    """``ldmatrix.x4``: four 8x8 b16 matrices, rows addressed by lanes 0-7, 8-15, ..."""
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[cutlass.Uint32] * 4,
        read_only_args=[addr],
    )


@cute.jit
def _ldsm_x4_trans(addr):
    """``ldmatrix.x4.trans``: the transposed b16 fragments of four 8x8 matrices."""
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[cutlass.Uint32] * 4,
        read_only_args=[addr],
    )


@cute.jit
def _cp_async_16(dst, src, src_bytes):
    """16-byte global-to-shared copy; ``src_bytes`` 0 writes zeros."""
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[dst, src, src_bytes],
    )


@cute.jit
def _mma_e4m3_k32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def _mma_16bit_k16(fp16: cutlass.Constexpr, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    if cutlass.const_expr(fp16):
        ptx = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    else:
        ptx = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    return cute.arch.inline_ptx(
        ptx + "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def _pack_16bit(fp16: cutlass.Constexpr, lo, hi):
    """Two FP32 values rounded to one FP16x2 / BF16x2 word, ``lo`` in the low half."""
    if cutlass.const_expr(fp16):
        ptx = "cvt.rn.f16x2.f32 {$w0}, {$r1}, {$r0};"
    else:
        ptx = "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};"
    return cute.arch.inline_ptx(
        ptx, write_only_types=[cutlass.Uint32], read_only_args=[lo, hi]
    )


@cute.jit
def _e4m3x2_lo_to_f16x2(word):
    """The two E4M3 bytes of the low half of ``word`` as FP16x2 (exact)."""
    return cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
        "cvt.rn.f16x2.e4m3x2 {$w0}, lo; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[word],
    )


@cute.jit
def _e4m3x2_hi_to_f16x2(word):
    """The two E4M3 bytes of the high half of ``word`` as FP16x2 (exact)."""
    return cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
        "cvt.rn.f16x2.e4m3x2 {$w0}, hi; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[word],
    )


@cute.jit
def _prmt(a, b, selector: cutlass.Constexpr):
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r1}}, {selector};",
        write_only_types=[cutlass.Uint32],
        read_only_args=[a, b],
    )


@cute.jit
def _mulhi(x, magic):
    """``x // d`` for ``magic = ceil(2**32 / d)``, ``d >= 2`` and ``x < 2**20``."""
    return cute.arch.inline_ptx(
        "mul.hi.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[x, magic],
    )


@cute.jit
def _ex2(x):
    return cute.arch.inline_ptx(
        "ex2.approx.ftz.f32 {$w0}, {$r0};",
        write_only_types=[cutlass.Float32],
        read_only_args=[x],
    )


@cute.jit
def _lg2(x):
    return cute.arch.inline_ptx(
        "lg2.approx.ftz.f32 {$w0}, {$r0};",
        write_only_types=[cutlass.Float32],
        read_only_args=[x],
    )


@cute.jit
def _mul(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


@cute.jit
def _fma(a, b, c):
    return cute.arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b, c],
    )


@cute.jit
def _div(a, b):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


@cute.jit
def _shfl_xor(x, mask: cutlass.Constexpr):
    return cute.arch.inline_ptx(
        f"shfl.sync.bfly.b32 {{$w0}}, {{$r0}}, {mask}, 31, 0xffffffff;",
        write_only_types=[cutlass.Float32],
        read_only_args=[x],
    )


# ---------------------------------------------------------------------------
# Chunk-state fold, shared by the attention and the combine kernels
# ---------------------------------------------------------------------------


@cute.jit
def _fold_scales(m_run, l_run, m_part, l_part):
    """``(m, l, scale_run, scale_part)`` of folding one chunk state into a run.

    The accumulators fold as ``fma(acc_part, scale_part, acc_run * scale_run)``
    (:func:`_fold`); both kernels call these two helpers, so the rounding is
    the same wherever the fold happens.
    """
    neg_inf = cutlass.Float32(_NEG_INF)
    m_new = cute.arch.fmax(m_run, m_part)
    m_safe = m_new
    if m_new == neg_inf:
        m_safe = cutlass.Float32(0.0)
    scale_run = cutlass.Float32(0.0)
    if m_run != neg_inf:
        scale_run = _ex2(m_run - m_safe)
    scale_part = cutlass.Float32(0.0)
    if m_part != neg_inf:
        scale_part = _ex2(m_part - m_safe)
    l_new = _fma(l_part, scale_part, _mul(l_run, scale_run))
    return m_new, l_new, scale_run, scale_part


@cute.jit
def _fold(acc_run, acc_part, scale_run, scale_part):
    return _fma(acc_part, scale_part, _mul(acc_run, scale_run))


@cute.jit
def _finish(acc, l):
    """Normalized output; a row that attended nothing is zero."""
    result = cutlass.Float32(0.0)
    if l > cutlass.Float32(0.0):
        result = _div(acc, l)
    return result


@cute.jit
def _lse(m, l):
    """Natural-log softmax normalizer from the base-2 running state."""
    result = cutlass.Float32(_NEG_INF)
    if l > cutlass.Float32(0.0):
        result = _mul(m + _lg2(l), cutlass.Float32(_LN2))
    return result


# ---------------------------------------------------------------------------
# Attention kernel
# ---------------------------------------------------------------------------


class _SparseMma:
    """One CTA per (head tile, token, split); see the module docstring."""

    def __init__(
        self,
        fp8: bool,
        block_h: int,
        rank: int,
        rope: int,
        stages: int,
        v4: tuple[bool, bool] | None = None,
        cwarps: int = CHANNEL_WARPS,
    ):
        assert block_h in (16, 32) and rank in (128, 512) and rope in (0, 64)
        # ``cwarps`` warps of a head group split the output channels of
        # ``O += P V``: 4, or 8 for a 16-head BF16 CTA without RoPE (V4). The
        # scores always sum four channel quarters, so both give the same bits.
        assert cwarps == CHANNEL_WARPS or (
            cwarps == 2 * CHANNEL_WARPS and block_h == 16 and rope == 0 and v4
        )
        self.cwarps = cwarps
        # ``v4`` = (sink, wide_ids) selects the DeepSeek V4 variant: BF16 rows
        # of a flat workspace (a page of one row, no divide), strided query and
        # output, row ids checked against the workspace (int32 or int64), and
        # an optional per-head sink as the first chunk's initial state.
        assert v4 is None or (not fp8 and rope == 0)
        self.v4 = v4 is not None
        self.sink = bool(v4 and v4[0])
        self.wide_ids = bool(v4 and v4[1])
        self.id_bytes = 8 if self.wide_ids else 4
        self.fp8 = fp8
        self.block_h = block_h
        self.rank = rank
        self.rope = rope
        self.stages = stages
        self.esize = 1 if fp8 else 2
        self.head_dim = rank + rope
        self.row_bytes = self.head_dim * self.esize
        self.head_groups = block_h // 16
        self.num_warps = self.head_groups * cwarps
        self.num_threads = self.num_warps * 32
        self.warp_ch = rank // cwarps  # latent channels per warp (output)
        self.score_ch = rank // CHANNEL_WARPS  # channels per score quarter
        if fp8:
            # 64-row tiles, rows unpadded and XOR-swizzled in 16-byte units;
            # the query tile lives in shared memory and each warp scores 16
            # of the 64 rows over all channels (key split).
            self.rows = 64
            self.row_stride = self.row_bytes
            self.latent_chunks = rank // 16
        else:
            # 32-row tiles padded by 16 bytes (an odd number of 16-byte units
            # per row keeps the fragment loads conflict-free); each warp keeps
            # its quarter of the query channels in registers and scores all
            # 32 rows over them (channel split).
            self.rows = 32
            self.row_stride = self.row_bytes + 16
            self.latent_chunks = 0
        self.s_tiles = self.rows // 8  # 16 x 8 score tiles per head group
        self.pv_steps = self.rows // 16
        self.rope_ch = rope // CHANNEL_WARPS  # RoPE channels per warp (BF16)
        # V fragments come from ``ldmatrix.trans`` over 16-byte chunks of 8
        # rows. A group of GROUP_CH channels feeds TPG MMA column tiles and
        # each lane owns CPL contiguous channels of it: for E4M3 a chunk is
        # 16 channels whose even and odd ones go to two tiles, for BF16 a
        # chunk is one tile of 8 channels.
        self.group_ch = 16 if fp8 else 8
        self.tpg = 2 if fp8 else 1
        self.cpl = 4 if fp8 else 2
        self.warp_chunks = self.warp_ch * self.esize // 16
        self.groups = self.warp_ch // self.group_ch
        self.n_tiles = self.groups * self.tpg  # 16 x 8 output tiles per warp
        self.qk_steps = (self.head_dim // 32) if fp8 else (self.score_ch // 16)
        self.chunks_per_row = self.row_bytes // 16
        self.threads_per_row = self.num_threads // self.rows
        self.copies_per_thread = -(-self.chunks_per_row // self.threads_per_row)
        ring = stages * self.rows * self.row_stride
        if fp8:
            exchange = self.num_warps * 32 * 8 * 4
            q_tile = block_h * self.row_bytes
        else:
            exchange = self.head_groups * CHANNEL_WARPS * 32 * 16 * 4
            q_tile = 0
        self.q_offset = ring
        self.x_offset = ring + q_tile
        self.slot_offset = ring + q_tile + exchange
        self.smem_bytes = self.slot_offset + stages * self.rows * 4

    @cute.jit
    def __call__(
        self,
        q,  # cute.Tensor, or the address (Int64) in the V4 variant
        kv_addr: cutlass.Int64,
        slots: cute.Tensor,
        lens: cute.Tensor,
        out,  # cute.Tensor, or the address (Int64) in the V4 variant
        lse: cute.Tensor,
        part_acc: cute.Tensor,
        part_ml: cute.Tensor,
        num_heads: cutlass.Int32,
        topk: cutlass.Int32,
        slot_stride: cutlass.Int32,
        page_stride: cutlass.Int64,
        token_stride: cutlass.Int64,
        page_size: cutlass.Int32,
        chunk_tiles: cutlass.Int32,
        num_chunks: cutlass.Int32,
        chunks_per_split: cutlass.Int32,
        full_tokens: cutlass.Int32,
        tail_splits: cutlass.Int32,
        tail_magic: cutlass.Uint32,
        return_lse: cutlass.Int32,
        scale_log2: cutlass.Float32,
        num_rows: cutlass.Int32,
        sink: cute.Tensor,
        q_stride_token: cutlass.Int64,
        q_stride_head: cutlass.Int64,
        out_stride_token: cutlass.Int64,
        out_stride_head: cutlass.Int64,
        grid_x: cutlass.Int32,
        grid_y: cutlass.Int32,
        stream,
    ):
        self.kernel(
            q,
            kv_addr,
            slots,
            lens,
            out,
            lse,
            part_acc,
            part_ml,
            num_heads,
            topk,
            slot_stride,
            page_stride,
            token_stride,
            page_size,
            chunk_tiles,
            num_chunks,
            chunks_per_split,
            full_tokens,
            tail_splits,
            tail_magic,
            return_lse,
            scale_log2,
            num_rows,
            sink,
            q_stride_token,
            q_stride_head,
            out_stride_token,
            out_stride_head,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _load_slot(self, slot_row, col, num_rows):
        """The slot id at column ``col``; the V4 variant maps a row id that is
        negative or past the workspace to ``-1`` (int32 or int64 ids)."""
        if cutlass.const_expr(self.v4):
            slot = cutlass.Int32(-1)
            if cutlass.const_expr(self.wide_ids):
                wide = _ldg_s64(slot_row + cutlass.Int64(col) * 8)
                if (wide >= cutlass.Int64(0)) & (wide < cutlass.Int64(num_rows)):
                    slot = cutlass.Int32(wide)
            else:
                narrow = _ldg_s32(slot_row + cutlass.Int64(col) * 4)
                if (narrow >= 0) & (narrow < num_rows):
                    slot = narrow
            return slot
        else:
            return _ldg_s32(slot_row + cutlass.Int64(col) * 4)

    @cute.jit
    def _chunk_offset(self, row, chunk):
        """Byte offset of 16-byte ``chunk`` of a tile row in shared memory."""
        if cutlass.const_expr(self.fp8):
            phys = chunk ^ (row & 7)
            if cutlass.const_expr(self.rope > 0):
                if chunk >= self.latent_chunks:
                    phys = self.latent_chunks + (
                        (chunk - self.latent_chunks) ^ (row & 3)
                    )
            return row * self.row_stride + phys * 16
        else:
            return row * self.row_stride + chunk * 16

    @cute.jit
    def _issue_tile(
        self,
        tile,
        tile_hi,
        valid_len,
        slot,
        stage,
        kv_addr,
        page_stride,
        token_stride,
        page_size,
        ring_addr,
        slot_addr,
        tidx,
    ):
        """``cp.async`` one tile's rows into ``stage``; this thread's row is ``slot``."""
        rows = cutlass.const_expr(self.rows)
        row = tidx // self.threads_per_row
        part = tidx % self.threads_per_row
        col = tile * rows + row
        ok = (tile < tile_hi) & (col < valid_len) & (slot >= 0)
        live = cutlass.Int32(0)
        src = kv_addr
        if ok:
            live = cutlass.Int32(16)
            if cutlass.const_expr(self.v4):
                src = kv_addr + cutlass.Int64(slot) * page_stride
            else:
                src = (
                    kv_addr
                    + cutlass.Int64(slot // page_size) * page_stride
                    + cutlass.Int64(slot % page_size) * token_stride
                )
        dst = ring_addr + stage * (rows * self.row_stride)
        for j in cutlass.range_constexpr(self.copies_per_thread):
            chunk = part + j * self.threads_per_row
            if cutlass.const_expr(
                (j + 1) * self.threads_per_row <= self.chunks_per_row
            ):
                _cp_async_16(
                    dst + self._chunk_offset(row, chunk),
                    src + cutlass.Int64(chunk * 16),
                    live,
                )
            else:
                if chunk < self.chunks_per_row:
                    _cp_async_16(
                        dst + self._chunk_offset(row, chunk),
                        src + cutlass.Int64(chunk * 16),
                        live,
                    )
        if part == 0:
            marked = cutlass.Int32(-1)
            if ok:
                marked = slot
            _sts_s32(slot_addr + (stage * rows + row) * 4, marked)

    @cute.jit
    def _scores_key_split(self, s, base, q_off, k_off, x_addr, hg, cw, lane):
        """E4M3: this warp's 16 rows over all channels, exchanged in the head group."""
        sacc = cute.make_rmem_tensor((16,), cutlass.Float32)
        for e in cutlass.range_constexpr(16):
            sacc[e] = cutlass.Float32(0.0)
        for ks in cutlass.range_constexpr(self.qk_steps):
            # Two accumulators per tile (even and odd steps) halve the chain.
            par = ks % 2
            latent_steps = cutlass.const_expr(self.latent_chunks // 2)
            if cutlass.const_expr(ks < latent_steps):
                step = cutlass.const_expr(128 * (ks // 4))
                a0, a1, a2, a3 = _ldsm_x4(q_off[ks % 4] + step)
                b0, b1, b2, b3 = _ldsm_x4(base + k_off[ks % 4] + step)
            else:
                a0, a1, a2, a3 = _ldsm_x4(q_off[4 + ks - latent_steps])
                b0, b1, b2, b3 = _ldsm_x4(base + k_off[4 + ks - latent_steps])
            for jj in cutlass.range_constexpr(2):
                o = 8 * par + 4 * jj
                bl = b0
                bh = b1
                if cutlass.const_expr(jj == 1):
                    bl = b2
                    bh = b3
                d0, d1, d2, d3 = _mma_e4m3_k32(
                    a0,
                    a1,
                    a2,
                    a3,
                    bl,
                    bh,
                    sacc[o],
                    sacc[o + 1],
                    sacc[o + 2],
                    sacc[o + 3],
                )
                sacc[o] = d0
                sacc[o + 1] = d1
                sacc[o + 2] = d2
                sacc[o + 3] = d3
        warp_x = x_addr + ((hg * CHANNEL_WARPS + cw) * 2 * 32 + lane) * 16
        for jj in cutlass.range_constexpr(2):
            _sts_v4_f32(
                warp_x + jj * 512,
                sacc[4 * jj] + sacc[8 + 4 * jj],
                sacc[4 * jj + 1] + sacc[8 + 4 * jj + 1],
                sacc[4 * jj + 2] + sacc[8 + 4 * jj + 2],
                sacc[4 * jj + 3] + sacc[8 + 4 * jj + 3],
            )
        cute.arch.barrier(barrier_id=1 + hg, number_of_threads=CHANNEL_WARPS * 32)
        for c in cutlass.range_constexpr(CHANNEL_WARPS):
            for jj in cutlass.range_constexpr(2):
                j = 2 * c + jj
                p0, p1, p2, p3 = _lds_v4_f32(
                    x_addr + (((hg * CHANNEL_WARPS + c) * 2 + jj) * 32 + lane) * 16
                )
                s[4 * j] = p0
                s[4 * j + 1] = p1
                s[4 * j + 2] = p2
                s[4 * j + 3] = p3

    @cute.jit
    def _scores_channel_split(self, s, base, qa, qr, x_addr, hg, cw, lane, g, t):
        """BF16: all 32 rows over a channel quarter, partials summed in the group.

        With 4 warps a warp scores all 32 rows over its quarter. With 8 the two
        warps of a quarter split the rows (``jn`` of the 4 score tiles each);
        a row's partial is the same sequence of MMAs, so the sum is unchanged.
        """
        rank = cutlass.const_expr(self.rank)
        rs = cutlass.const_expr(self.row_stride)
        jn = cutlass.const_expr(4 * CHANNEL_WARPS // self.cwarps)
        qcw = cw
        rh = 0
        if cutlass.const_expr(self.cwarps != CHANNEL_WARPS):
            qcw = cw % CHANNEL_WARPS
            rh = cw // CHANNEL_WARPS
        for e in cutlass.range_constexpr(16):
            s[e] = cutlass.Float32(0.0)
        # ldmatrix: lanes 8 m .. 8 m + 7 address rows 8 j + lane % 8 of the
        # m-th 8-channel chunk of a pair of steps.
        k_lane = (lane % 8) * rs + (qcw * self.score_ch + 8 * (lane // 8)) * 2
        for jj in cutlass.range_constexpr(jn):
            j = rh * jn + jj
            row_addr = base + (8 * j + g) * rs
            for kp in cutlass.range_constexpr(self.qk_steps // 2):
                k0, k1, k2, k3 = _ldsm_x4(base + k_lane + (8 * j * rs + kp * 64))
                for half in cutlass.range_constexpr(2):
                    ks = 2 * kp + half
                    b0 = k0
                    b1 = k1
                    if cutlass.const_expr(half == 1):
                        b0 = k2
                        b1 = k3
                    d0, d1, d2, d3 = _mma_16bit_k16(
                        False,
                        qa[ks * 4 + 0],
                        qa[ks * 4 + 1],
                        qa[ks * 4 + 2],
                        qa[ks * 4 + 3],
                        b0,
                        b1,
                        s[4 * jj + 0],
                        s[4 * jj + 1],
                        s[4 * jj + 2],
                        s[4 * jj + 3],
                    )
                    s[4 * jj + 0] = d0
                    s[4 * jj + 1] = d1
                    s[4 * jj + 2] = d2
                    s[4 * jj + 3] = d3
            if cutlass.const_expr(self.rope > 0):
                ch = rank + cw * self.rope_ch
                b0 = _lds_u32(row_addr + (ch + 2 * t) * 2)
                b1 = _lds_u32(row_addr + (ch + 8 + 2 * t) * 2)
                d0, d1, d2, d3 = _mma_16bit_k16(
                    False,
                    qr[0],
                    qr[1],
                    qr[2],
                    qr[3],
                    b0,
                    b1,
                    s[4 * jj + 0],
                    s[4 * jj + 1],
                    s[4 * jj + 2],
                    s[4 * jj + 3],
                )
                s[4 * jj + 0] = d0
                s[4 * jj + 1] = d1
                s[4 * jj + 2] = d2
                s[4 * jj + 3] = d3
            _sts_v4_f32(
                x_addr + (((hg * self.cwarps + cw) * jn + jj) * 32 + lane) * 16,
                s[4 * jj + 0],
                s[4 * jj + 1],
                s[4 * jj + 2],
                s[4 * jj + 3],
            )
        cute.arch.barrier(barrier_id=1 + hg, number_of_threads=self.cwarps * 32)
        for j in cutlass.range_constexpr(4):
            for c in cutlass.range_constexpr(CHANNEL_WARPS):
                # Quarter c of tile j was scored by warp (j // jn) * 4 + c.
                owner = cutlass.const_expr((j // jn) * CHANNEL_WARPS + c)
                p0, p1, p2, p3 = _lds_v4_f32(
                    x_addr
                    + (((hg * self.cwarps + owner) * jn + j % jn) * 32 + lane) * 16
                )
                if cutlass.const_expr(c == 0):
                    s[4 * j + 0] = p0
                    s[4 * j + 1] = p1
                    s[4 * j + 2] = p2
                    s[4 * j + 3] = p3
                else:
                    s[4 * j + 0] = s[4 * j + 0] + p0
                    s[4 * j + 1] = s[4 * j + 1] + p1
                    s[4 * j + 2] = s[4 * j + 2] + p2
                    s[4 * j + 3] = s[4 * j + 3] + p3

    @cute.jit
    def _softmax_pv(
        self,
        s,
        acc,
        base,
        v_off,
        scale_log2,
        mx0,
        mx1,
        m_i0,
        m_i1,
        l_i0,
        l_i1,
        scale_acc,
    ):
        """Online softmax update of a masked score tile and ``O += P V``.

        ``mx0`` / ``mx1`` are the raw-score maxima of the thread's two rows
        (the scale is positive, so it is applied inside ``p = 2^(s c - m)``);
        ``s`` holds the masked scores, ``base`` the tile's shared-memory
        address, ``scale_acc`` whether the accumulator holds earlier tiles of
        the chunk. Returns the updated ``m`` and ``l`` of the two rows.
        """
        fp8 = cutlass.const_expr(self.fp8)
        rs = cutlass.const_expr(self.row_stride)
        n_tiles = cutlass.const_expr(self.n_tiles)
        s_tiles = cutlass.const_expr(self.s_tiles)
        neg_inf = cutlass.Float32(_NEG_INF)
        m_n0 = cute.arch.fmax(m_i0, _mul(mx0, scale_log2))
        m_n1 = cute.arch.fmax(m_i1, _mul(mx1, scale_log2))
        ms0 = m_n0
        ms1 = m_n1
        if m_n0 == neg_inf:
            ms0 = cutlass.Float32(0.0)
        if m_n1 == neg_inf:
            ms1 = cutlass.Float32(0.0)
        alpha0 = _ex2(m_i0 - ms0)
        alpha1 = _ex2(m_i1 - ms1)
        nms0 = -ms0
        nms1 = -ms1
        rs0 = cutlass.Float32(0.0)
        rs1 = cutlass.Float32(0.0)
        # The probabilities go straight into the PV A fragments: tile j
        # of the scores is half of the k16 step j // 2.
        pk = cute.make_rmem_tensor((s_tiles * 2,), cutlass.Uint32)
        for j in cutlass.range_constexpr(s_tiles):
            p00 = _ex2(_fma(s[4 * j], scale_log2, nms0))
            p01 = _ex2(_fma(s[4 * j + 1], scale_log2, nms0))
            p10 = _ex2(_fma(s[4 * j + 2], scale_log2, nms1))
            p11 = _ex2(_fma(s[4 * j + 3], scale_log2, nms1))
            rs0 = rs0 + p00
            rs0 = rs0 + p01
            rs1 = rs1 + p10
            rs1 = rs1 + p11
            pk[2 * j] = _pack_16bit(fp8, p00, p01)
            pk[2 * j + 1] = _pack_16bit(fp8, p10, p11)
        rs0 = rs0 + _shfl_xor(rs0, 1)
        rs0 = rs0 + _shfl_xor(rs0, 2)
        rs1 = rs1 + _shfl_xor(rs1, 1)
        rs1 = rs1 + _shfl_xor(rs1, 2)
        l_i0 = _fma(l_i0, alpha0, rs0)
        l_i1 = _fma(l_i1, alpha1, rs1)
        m_i0 = m_n0
        m_i1 = m_n1
        if scale_acc:
            for n in cutlass.range_constexpr(n_tiles):
                acc[4 * n + 0] = acc[4 * n + 0] * alpha0
                acc[4 * n + 1] = acc[4 * n + 1] * alpha0
                acc[4 * n + 2] = acc[4 * n + 2] * alpha1
                acc[4 * n + 3] = acc[4 * n + 3] * alpha1

        # ---- O += P V over this warp's channels ----
        for kk in cutlass.range_constexpr(self.pv_steps):
            pa0 = pk[4 * kk]
            pa1 = pk[4 * kk + 1]
            pa2 = pk[4 * kk + 2]
            pa3 = pk[4 * kk + 3]
            for cp in cutlass.range_constexpr(self.warp_chunks // 2):
                # Rows 16 kk .. 16 kk + 15 of chunks 2 cp and 2 cp + 1.
                if cutlass.const_expr(fp8):
                    v_addr = base + v_off[cp % 4] + (128 * (cp // 4) + kk * 16 * rs)
                else:
                    v_addr = base + v_off[0] + (32 * cp + kk * 16 * rs)
                r0, r1, r2, r3 = _ldsm_x4_trans(v_addr)
                for ci in cutlass.range_constexpr(2):
                    ra = r0
                    rb = r1
                    if cutlass.const_expr(ci == 1):
                        ra = r2
                        rb = r3
                    if cutlass.const_expr(fp8):
                        # Bytes (k, c), (k, c+1), (k+1, c), (k+1, c+1):
                        # pair the rows of channel c, then of c + 1.
                        xa = _prmt(ra, ra, 0x3120)
                        xb = _prmt(rb, rb, 0x3120)
                    for jj in cutlass.range_constexpr(self.tpg):
                        n = (2 * cp + ci) * self.tpg + jj
                        if cutlass.const_expr(fp8):
                            if cutlass.const_expr(jj == 0):
                                b0 = _e4m3x2_lo_to_f16x2(xa)
                                b1 = _e4m3x2_lo_to_f16x2(xb)
                            else:
                                b0 = _e4m3x2_hi_to_f16x2(xa)
                                b1 = _e4m3x2_hi_to_f16x2(xb)
                        else:
                            b0 = ra
                            b1 = rb
                        d0, d1, d2, d3 = _mma_16bit_k16(
                            fp8,
                            pa0,
                            pa1,
                            pa2,
                            pa3,
                            b0,
                            b1,
                            acc[4 * n + 0],
                            acc[4 * n + 1],
                            acc[4 * n + 2],
                            acc[4 * n + 3],
                        )
                        acc[4 * n + 0] = d0
                        acc[4 * n + 1] = d1
                        acc[4 * n + 2] = d2
                        acc[4 * n + 3] = d3
        return m_i0, m_i1, l_i0, l_i1

    @cute.kernel
    def kernel(
        self,
        q,  # cute.Tensor, or the address (Int64) in the V4 variant
        kv_addr: cutlass.Int64,
        slots: cute.Tensor,
        lens: cute.Tensor,
        out,  # cute.Tensor, or the address (Int64) in the V4 variant
        lse: cute.Tensor,
        part_acc: cute.Tensor,
        part_ml: cute.Tensor,
        num_heads: cutlass.Int32,
        topk: cutlass.Int32,
        slot_stride: cutlass.Int32,
        page_stride: cutlass.Int64,
        token_stride: cutlass.Int64,
        page_size: cutlass.Int32,
        chunk_tiles: cutlass.Int32,
        num_chunks: cutlass.Int32,
        chunks_per_split: cutlass.Int32,
        full_tokens: cutlass.Int32,
        tail_splits: cutlass.Int32,
        tail_magic: cutlass.Uint32,
        return_lse: cutlass.Int32,
        scale_log2: cutlass.Float32,
        num_rows: cutlass.Int32,
        sink: cute.Tensor,
        q_stride_token: cutlass.Int64,
        q_stride_head: cutlass.Int64,
        out_stride_token: cutlass.Int64,
        out_stride_head: cutlass.Int64,
    ):
        fp8 = cutlass.const_expr(self.fp8)
        rank = cutlass.const_expr(self.rank)
        rope = cutlass.const_expr(self.rope)
        stages = cutlass.const_expr(self.stages)
        rows = cutlass.const_expr(self.rows)
        esize = cutlass.const_expr(self.esize)
        rs = cutlass.const_expr(self.row_stride)
        n_tiles = cutlass.const_expr(self.n_tiles)
        s_tiles = cutlass.const_expr(self.s_tiles)
        neg_inf = cutlass.Float32(_NEG_INF)

        tidx, _, _ = cute.arch.thread_idx()
        head_tile, row, _ = cute.arch.block_idx()
        # Rows of blocks are tokens, head tiles run along a row. The first
        # ``full_tokens`` rows each fold their chunks in registers; every later
        # token takes ``tail_splits`` rows, each a run of chunks whose states a
        # combine kernel folds.
        token = row
        split_id = cutlass.Int32(0)
        if row >= full_tokens:
            tail_token = _mulhi(row - full_tokens, tail_magic)
            split_id = row - full_tokens - tail_token * tail_splits
            token = full_tokens + tail_token
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        g = lane // 4
        t = lane % 4
        hg = warp // self.cwarps  # head group
        cw = warp % self.cwarps  # output channel (and, for E4M3 scores, row) group
        qcw = cw  # score channel quarter
        if cutlass.const_expr(self.cwarps != CHANNEL_WARPS):
            qcw = cw % CHANNEL_WARPS

        smem = SmemAllocator()
        pool = smem.allocate_tensor(
            element_type=cutlass.Uint8,
            layout=cute.make_layout((self.smem_bytes,)),
            byte_alignment=128,
        )
        ring_addr = cutlass.Int32(pool.iterator.toint())
        q_addr = ring_addr + self.q_offset
        x_addr = ring_addr + self.x_offset
        slot_addr = ring_addr + self.slot_offset

        # ---- This token's selection and this CTA's chunks ----
        valid_len = cutlass.min(cutlass.Int32(lens[token]), topk)
        if cutlass.const_expr(self.v4):
            valid_len = cutlass.max(valid_len, cutlass.Int32(0))
        num_tiles = (valid_len + (rows - 1)) // rows
        chunk_begin = cutlass.Int32(0)
        chunk_end = cutlass.max(
            (num_tiles + chunk_tiles - 1) // chunk_tiles, cutlass.Int32(1)
        )
        if row >= full_tokens:
            chunk_begin = split_id * chunks_per_split
            chunk_end = cutlass.min(chunk_begin + chunks_per_split, num_chunks)
        tile_lo = chunk_begin * chunk_tiles
        tile_hi = cutlass.min(chunk_end * chunk_tiles, num_tiles)

        slot_row = (
            slots.iterator.toint()
            + cutlass.Int64(token) * cutlass.Int64(slot_stride) * self.id_bytes
        )
        my_row = tidx // self.threads_per_row

        head0 = head_tile * self.block_h + hg * 16 + g
        head1 = head0 + 8
        live0 = head0 < num_heads
        live1 = head1 < num_heads
        if cutlass.const_expr(self.v4):
            q_base = q  # the V4 variant takes the query's address
        else:
            q_base = q.iterator.toint()
        qa = cute.make_rmem_tensor((max(self.qk_steps, 1) * 4,), cutlass.Uint32)
        qr = cute.make_rmem_tensor((4,), cutlass.Uint32)
        if cutlass.const_expr(fp8):
            # ---- The query tile goes to shared memory with the first tiles ----
            q_tile = (
                q_base
                + (
                    cutlass.Int64(token) * cutlass.Int64(num_heads)
                    + cutlass.Int64(head_tile * self.block_h)
                )
                * self.row_bytes
            )
            per_tile = cutlass.const_expr(self.block_h * self.chunks_per_row)
            for k in cutlass.range_constexpr(-(-per_tile // self.num_threads)):
                idx = tidx + k * self.num_threads
                if idx < per_tile:
                    h = idx // self.chunks_per_row
                    c = idx % self.chunks_per_row
                    live_h = cutlass.Int32(0)
                    if head_tile * self.block_h + h < num_heads:
                        live_h = cutlass.Int32(16)
                    _cp_async_16(
                        q_addr + self._chunk_offset(h, c),
                        q_tile + cutlass.Int64(h * self.row_bytes + c * 16),
                        live_h,
                    )
        else:
            # ---- This warp's query channels, kept in registers ----
            if cutlass.const_expr(self.v4):
                q_row0 = (
                    q_base
                    + cutlass.Int64(token) * q_stride_token
                    + cutlass.Int64(head0) * q_stride_head
                )
                q_row1 = q_row0 + 8 * q_stride_head
            else:
                q_row0 = (
                    q_base
                    + (
                        cutlass.Int64(token) * cutlass.Int64(num_heads)
                        + cutlass.Int64(head0)
                    )
                    * self.row_bytes
                )
                q_row1 = q_row0 + 8 * self.row_bytes
            for i in cutlass.range_constexpr(self.qk_steps * 4):
                qa[i] = cutlass.Uint32(0)
            for i in cutlass.range_constexpr(4):
                qr[i] = cutlass.Uint32(0)
            for ks in cutlass.range_constexpr(self.qk_steps):
                ch = qcw * self.score_ch + ks * 16
                off0 = cutlass.Int64((ch + 2 * t) * 2)
                off1 = cutlass.Int64((ch + 8 + 2 * t) * 2)
                if live0:
                    qa[ks * 4 + 0] = _ldg_u32(q_row0 + off0)
                    qa[ks * 4 + 2] = _ldg_u32(q_row0 + off1)
                if live1:
                    qa[ks * 4 + 1] = _ldg_u32(q_row1 + off0)
                    qa[ks * 4 + 3] = _ldg_u32(q_row1 + off1)
            if cutlass.const_expr(rope > 0):
                ch = rank + cw * self.rope_ch
                off0 = cutlass.Int64((ch + 2 * t) * 2)
                off1 = cutlass.Int64((ch + 8 + 2 * t) * 2)
                if live0:
                    qr[0] = _ldg_u32(q_row0 + off0)
                    qr[2] = _ldg_u32(q_row0 + off1)
                if live1:
                    qr[1] = _ldg_u32(q_row1 + off0)
                    qr[3] = _ldg_u32(q_row1 + off1)

        # ---- Softmax state: rows g and g + 8 of the head group ----
        acc = cute.make_rmem_tensor((n_tiles * 4,), cutlass.Float32)
        acc_run = cute.make_rmem_tensor((n_tiles * 4,), cutlass.Float32)
        for i in cutlass.range_constexpr(n_tiles * 4):
            acc[i] = cutlass.Float32(0.0)
            acc_run[i] = cutlass.Float32(0.0)
        m_i0 = neg_inf
        m_i1 = neg_inf
        l_i0 = cutlass.Float32(0.0)
        l_i1 = cutlass.Float32(0.0)
        m_r0 = neg_inf
        m_r1 = neg_inf
        l_r0 = cutlass.Float32(0.0)
        l_r1 = cutlass.Float32(0.0)
        if cutlass.const_expr(self.sink):
            # The sink is one extra key with a zero value: the initial state of
            # chunk 0 (m = sink logit, l = 1), the first thing the fold sees.
            if chunk_begin == 0:
                sink_base = sink.iterator.toint()
                if live0:
                    m_i0 = _mul(
                        _ldg_f32(sink_base + cutlass.Int64(head0) * 4),
                        cutlass.Float32(1.4426950408889634),
                    )
                    l_i0 = cutlass.Float32(1.0)
                if live1:
                    m_i1 = _mul(
                        _ldg_f32(sink_base + cutlass.Int64(head1) * 4),
                        cutlass.Float32(1.4426950408889634),
                    )
                    l_i1 = cutlass.Float32(1.0)
        s = cute.make_rmem_tensor((s_tiles * 4,), cutlass.Float32)
        # Lane addresses of the E4M3 score fragments: lanes 8 m .. 8 m + 7
        # address the rows of matrix m (query rows (m % 2) * 8 .., chunk
        # 2 ks + m // 2; key rows (m // 2) * 8 .., chunk 2 ks + m % 2).
        # Within a block of 8 chunks the swizzle is an XOR of the low bits, so
        # step ks reads chunk (2 ks) ^ y: four lane offsets serve every block
        # of the latent channels and two the RoPE chunks.
        q_off = cute.make_rmem_tensor((6,), cutlass.Int32)
        k_off = cute.make_rmem_tensor((6,), cutlass.Int32)
        if cutlass.const_expr(fp8):
            mi = lane // 8
            q_row = hg * 16 + (lane % 8) + 8 * (mi % 2)
            k_row = cw * 16 + 8 * (mi // 2) + (lane % 8)
            yq = (mi // 2) ^ (q_row % 8)
            yk = (mi % 2) ^ (k_row % 8)
            for m in cutlass.range_constexpr(4):
                q_off[m] = q_addr + q_row * rs + ((2 * m) ^ yq) * 16
                k_off[m] = k_row * rs + ((2 * m) ^ yk) * 16
            if cutlass.const_expr(rope > 0):
                yq = (mi // 2) ^ (q_row % 4)
                yk = (mi % 2) ^ (k_row % 4)
                for m in cutlass.range_constexpr(2):
                    q_off[4 + m] = (
                        q_addr + q_row * rs + (self.latent_chunks + ((2 * m) ^ yq)) * 16
                    )
                    k_off[4 + m] = (
                        k_row * rs + (self.latent_chunks + ((2 * m) ^ yk)) * 16
                    )
        # Lane offsets of the ldmatrix.trans rows: lanes 8 m .. 8 m + 7 address
        # rows (m % 2) * 8 + lane % 8 of chunk 2 cp + m // 2.
        v_off = cute.make_rmem_tensor((4,), cutlass.Int32)
        v_row = (lane % 8) + 8 * ((lane // 8) % 2)
        if cutlass.const_expr(fp8):
            yv = (lane // 16) ^ (lane % 8)
            for m in cutlass.range_constexpr(min(4, self.warp_chunks // 2)):
                v_off[m] = v_row * rs + (cw * self.warp_chunks + ((2 * m) ^ yv)) * 16
        else:
            v_off[0] = v_row * rs + (cw * self.warp_chunks + lane // 16) * 16

        # ---- Prologue: STAGES - 1 tiles in flight ----
        for st in cutlass.range_constexpr(stages - 1):
            tile = tile_lo + st
            col = tile * rows + my_row
            slot = cutlass.Int32(-1)
            if (tile < tile_hi) & (col < valid_len):
                slot = self._load_slot(slot_row, col, num_rows)
            self._issue_tile(
                tile,
                tile_hi,
                valid_len,
                slot,
                cutlass.Int32(st),
                kv_addr,
                page_stride,
                token_stride,
                page_size,
                ring_addr,
                slot_addr,
                tidx,
            )
            cute.arch.cp_async_commit_group()
        # The slot of the next tile to issue, loaded one iteration early.
        next_tile = tile_lo + (stages - 1)
        next_col = next_tile * rows + my_row
        next_slot = cutlass.Int32(-1)
        if (next_tile < tile_hi) & (next_col < valid_len):
            next_slot = self._load_slot(slot_row, next_col, num_rows)

        for tile in range(tile_lo, tile_hi):
            i = tile - tile_lo
            stage = i % stages
            cute.arch.cp_async_wait_group(stages - 2)
            cute.arch.barrier()
            # Refill the stage tile - 1 used; every warp is past it.
            issue = tile + (stages - 1)
            self._issue_tile(
                issue,
                tile_hi,
                valid_len,
                next_slot,
                (i + stages - 1) % stages,
                kv_addr,
                page_stride,
                token_stride,
                page_size,
                ring_addr,
                slot_addr,
                tidx,
            )
            cute.arch.cp_async_commit_group()
            next_col = (issue + 1) * rows + my_row
            next_slot = cutlass.Int32(-1)
            if (issue + 1 < tile_hi) & (next_col < valid_len):
                next_slot = self._load_slot(slot_row, next_col, num_rows)

            base = ring_addr + stage * (rows * rs)
            # ---- Full scores of the head group ----
            if cutlass.const_expr(fp8):
                self._scores_key_split(s, base, q_off, k_off, x_addr, hg, cw, lane)
            else:
                self._scores_channel_split(s, base, qa, qr, x_addr, hg, cw, lane, g, t)
            # ---- Mask, scale and the online softmax ----
            # The row maximum is taken over the raw scores (the scale is
            # positive) and the scale is applied inside p = 2^(s c - m).
            mx0 = neg_inf
            mx1 = neg_inf
            for j in cutlass.range_constexpr(s_tiles):
                live_a, live_b = _lds_v2_s32(
                    slot_addr + (stage * rows + 8 * j + 2 * t) * 4
                )
                for e in cutlass.range_constexpr(2):
                    col_slot = live_a
                    if cutlass.const_expr(e == 1):
                        col_slot = live_b
                    v0 = neg_inf
                    v1 = neg_inf
                    if col_slot >= 0:
                        v0 = s[4 * j + e]
                        v1 = s[4 * j + 2 + e]
                    s[4 * j + e] = v0
                    s[4 * j + 2 + e] = v1
                    mx0 = cute.arch.fmax(mx0, v0)
                    mx1 = cute.arch.fmax(mx1, v1)
            mx0 = cute.arch.fmax(mx0, _shfl_xor(mx0, 1))
            mx0 = cute.arch.fmax(mx0, _shfl_xor(mx0, 2))
            mx1 = cute.arch.fmax(mx1, _shfl_xor(mx1, 1))
            mx1 = cute.arch.fmax(mx1, _shfl_xor(mx1, 2))
            m_i0, m_i1, l_i0, l_i1 = self._softmax_pv(
                s,
                acc,
                base,
                v_off,
                scale_log2,
                mx0,
                mx1,
                m_i0,
                m_i1,
                l_i0,
                l_i1,
                # A chunk's first tile starts from a zero accumulator.
                tile % chunk_tiles != 0,
            )

            # ---- Chunk end: fold (one CTA) or store (split) the state ----
            if ((tile + 1) % chunk_tiles == 0) | (tile + 1 == tile_hi):
                chunk = tile // chunk_tiles
                if token >= full_tokens:
                    self._store_state(
                        part_acc,
                        part_ml,
                        acc,
                        m_i0,
                        m_i1,
                        l_i0,
                        l_i1,
                        token - full_tokens,
                        chunk,
                        num_chunks,
                        num_heads,
                        head0,
                        live0,
                        live1,
                        cw,
                        g,
                        t,
                    )
                else:
                    m_r0, l_r0, sr0, sp0 = _fold_scales(m_r0, l_r0, m_i0, l_i0)
                    m_r1, l_r1, sr1, sp1 = _fold_scales(m_r1, l_r1, m_i1, l_i1)
                    for n in cutlass.range_constexpr(n_tiles):
                        acc_run[4 * n + 0] = _fold(
                            acc_run[4 * n + 0], acc[4 * n + 0], sr0, sp0
                        )
                        acc_run[4 * n + 1] = _fold(
                            acc_run[4 * n + 1], acc[4 * n + 1], sr0, sp0
                        )
                        acc_run[4 * n + 2] = _fold(
                            acc_run[4 * n + 2], acc[4 * n + 2], sr1, sp1
                        )
                        acc_run[4 * n + 3] = _fold(
                            acc_run[4 * n + 3], acc[4 * n + 3], sr1, sp1
                        )
                for e in cutlass.range_constexpr(n_tiles * 4):
                    acc[e] = cutlass.Float32(0.0)
                m_i0 = neg_inf
                m_i1 = neg_inf
                l_i0 = cutlass.Float32(0.0)
                l_i1 = cutlass.Float32(0.0)

        cute.arch.cp_async_wait_group(0)
        if token >= full_tokens:
            # Chunks of this run past the selection store an empty state.
            first_empty = cutlass.max(
                chunk_begin, (num_tiles + chunk_tiles - 1) // chunk_tiles
            )
            for chunk in range(first_empty, chunk_end):
                self._store_state(
                    part_acc,
                    part_ml,
                    acc,
                    m_i0,
                    m_i1,
                    l_i0,
                    l_i1,
                    token - full_tokens,
                    chunk,
                    num_chunks,
                    num_heads,
                    head0,
                    live0,
                    live1,
                    cw,
                    g,
                    t,
                )
                if cutlass.const_expr(self.sink):
                    # Only chunk 0 owns the sink.
                    m_i0 = neg_inf
                    m_i1 = neg_inf
                    l_i0 = cutlass.Float32(0.0)
                    l_i1 = cutlass.Float32(0.0)
        else:
            if cutlass.const_expr(self.sink):
                if tile_hi <= tile_lo:
                    # No tiles: chunk 0 holds the sink alone.
                    m_r0 = m_i0
                    m_r1 = m_i1
                    l_r0 = l_i0
                    l_r1 = l_i1
            self._store_out(
                out,
                lse,
                acc_run,
                m_r0,
                m_r1,
                l_r0,
                l_r1,
                token,
                num_heads,
                head0,
                live0,
                live1,
                return_lse,
                out_stride_token,
                out_stride_head,
                cw,
                g,
                t,
            )

    @cute.jit
    def _lane_channel(self, cw, grp, t):
        """First of the ``cpl`` contiguous channels this lane owns in ``grp``."""
        return cw * self.warp_ch + grp * self.group_ch + t * self.cpl

    @cute.jit
    def _store_state(
        self,
        part_acc,
        part_ml,
        acc,
        m0,
        m1,
        l0,
        l1,
        tail_token,
        chunk,
        num_chunks,
        num_heads,
        head0,
        live0,
        live1,
        cw,
        g,
        t,
    ):
        """Store a chunk state of rows ``head0`` (+ 8): ``[tail token][chunk][head]``."""
        rank = cutlass.const_expr(self.rank)
        tpg = cutlass.const_expr(self.tpg)
        part0 = (
            cutlass.Int64(tail_token) * cutlass.Int64(num_chunks) + cutlass.Int64(chunk)
        ) * cutlass.Int64(num_heads) + cutlass.Int64(head0)
        acc_base = part_acc.iterator.toint()
        ml_base = part_ml.iterator.toint()
        for half in cutlass.range_constexpr(2):
            live = live0
            if cutlass.const_expr(half == 1):
                live = live1
            if live:
                part = part0 + 8 * half
                row = acc_base + part * (rank * 4)
                for grp in cutlass.range_constexpr(self.groups):
                    ch = self._lane_channel(cw, grp, t)
                    # Channel ch + idx is tile idx % tpg, column
                    # 2 t + idx // tpg of the group.
                    vals = []
                    for idx in cutlass.range_constexpr(self.cpl):
                        n = grp * tpg + idx % tpg
                        vals.append(acc[4 * n + 2 * half + idx // tpg])
                    addr = row + cutlass.Int64(ch * 4)
                    if cutlass.const_expr(self.cpl == 4):
                        _stg_v4_f32(addr, vals[0], vals[1], vals[2], vals[3])
                    else:
                        _stg_v2_f32(addr, vals[0], vals[1])
                if (cw == 0) & (t == 0):
                    m = m0
                    l = l0
                    if cutlass.const_expr(half == 1):
                        m = m1
                        l = l1
                    _stg_f32(ml_base + part * 8, m)
                    _stg_f32(ml_base + part * 8 + 4, l)

    @cute.jit
    def _store_out(
        self,
        out,
        lse,
        acc_run,
        m0,
        m1,
        l0,
        l1,
        token,
        num_heads,
        head0,
        live0,
        live1,
        return_lse,
        out_stride_token,
        out_stride_head,
        cw,
        g,
        t,
    ):
        rank = cutlass.const_expr(self.rank)
        tpg = cutlass.const_expr(self.tpg)
        if cutlass.const_expr(self.v4):
            out_base = out  # the V4 variant takes the output's address
        else:
            out_base = out.iterator.toint()
        lse_base = lse.iterator.toint()
        for half in cutlass.range_constexpr(2):
            live = live0
            m = m0
            l = l0
            if cutlass.const_expr(half == 1):
                live = live1
                m = m1
                l = l1
            if live:
                row_id = cutlass.Int64(token) * cutlass.Int64(
                    num_heads
                ) + cutlass.Int64(head0 + 8 * half)
                if cutlass.const_expr(self.v4):
                    row = (
                        out_base
                        + cutlass.Int64(token) * out_stride_token
                        + cutlass.Int64(head0 + 8 * half) * out_stride_head
                    )
                else:
                    row = out_base + row_id * (rank * 2)
                for grp in cutlass.range_constexpr(self.groups):
                    ch = self._lane_channel(cw, grp, t)
                    words = []
                    for i in cutlass.range_constexpr(self.cpl // 2):
                        pair = []
                        for k in cutlass.range_constexpr(2):
                            idx = 2 * i + k
                            n = grp * tpg + idx % tpg
                            pair.append(
                                _finish(acc_run[4 * n + 2 * half + idx // tpg], l)
                            )
                        words.append(_pack_16bit(False, pair[0], pair[1]))
                    if cutlass.const_expr(self.cpl == 4):
                        _stg_v2_u32(row + cutlass.Int64(ch * 2), words[0], words[1])
                    else:
                        _stg_u32(row + cutlass.Int64(ch * 2), words[0])
                if (return_lse != 0) & (cw == 0) & (t == 0):
                    _stg_f32(lse_base + row_id * 4, _lse(m, l))


class _Combine:
    """Fold the stored chunk states of one (token, head) and normalize."""

    def __init__(self, rank: int, strided_out: bool = False):
        self.rank = rank
        self.num_threads = rank // 4
        # The V4 variant takes the output's address and strides.
        self.strided_out = strided_out

    @cute.jit
    def __call__(
        self,
        part_acc: cute.Tensor,
        part_ml: cute.Tensor,
        out,  # cute.Tensor, or the address (Int64) in the V4 variant
        lse: cute.Tensor,
        num_heads: cutlass.Int32,
        num_chunks: cutlass.Int32,
        return_lse: cutlass.Int32,
        out_stride_token: cutlass.Int64,
        out_stride_head: cutlass.Int64,
        full_tokens: cutlass.Int32,
        tail_tokens: cutlass.Int32,
        stream,
    ):
        self.kernel(
            part_acc,
            part_ml,
            out,
            lse,
            num_heads,
            num_chunks,
            return_lse,
            out_stride_token,
            out_stride_head,
            full_tokens,
        ).launch(
            grid=(tail_tokens, num_heads, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        part_acc: cute.Tensor,
        part_ml: cute.Tensor,
        out,  # cute.Tensor, or the address (Int64) in the V4 variant
        lse: cute.Tensor,
        num_heads: cutlass.Int32,
        num_chunks: cutlass.Int32,
        return_lse: cutlass.Int32,
        out_stride_token: cutlass.Int64,
        out_stride_head: cutlass.Int64,
        full_tokens: cutlass.Int32,
    ):
        rank = cutlass.const_expr(self.rank)
        tidx, _, _ = cute.arch.thread_idx()
        tail_token, head, _ = cute.arch.block_idx()
        token = full_tokens + tail_token
        acc_base = part_acc.iterator.toint()
        ml_base = part_ml.iterator.toint()
        m_run = cutlass.Float32(_NEG_INF)
        l_run = cutlass.Float32(0.0)
        a0 = cutlass.Float32(0.0)
        a1 = cutlass.Float32(0.0)
        a2 = cutlass.Float32(0.0)
        a3 = cutlass.Float32(0.0)
        for chunk in range(num_chunks):
            part = (
                cutlass.Int64(tail_token) * cutlass.Int64(num_chunks)
                + cutlass.Int64(chunk)
            ) * cutlass.Int64(num_heads) + cutlass.Int64(head)
            m_part = _ldg_f32(ml_base + part * 8)
            l_part = _ldg_f32(ml_base + part * 8 + 4)
            p0, p1, p2, p3 = _ldg_v4_f32(acc_base + part * (rank * 4) + tidx * 16)
            m_run, l_run, sr, sp = _fold_scales(m_run, l_run, m_part, l_part)
            a0 = _fold(a0, p0, sr, sp)
            a1 = _fold(a1, p1, sr, sp)
            a2 = _fold(a2, p2, sr, sp)
            a3 = _fold(a3, p3, sr, sp)
        row_id = cutlass.Int64(token) * cutlass.Int64(num_heads) + cutlass.Int64(head)
        w0 = _pack_16bit(False, _finish(a0, l_run), _finish(a1, l_run))
        w1 = _pack_16bit(False, _finish(a2, l_run), _finish(a3, l_run))
        if cutlass.const_expr(self.strided_out):
            out_row = (
                out
                + cutlass.Int64(token) * out_stride_token
                + cutlass.Int64(head) * out_stride_head
            )
        else:
            out_row = out.iterator.toint() + row_id * (rank * 2)
        _stg_v2_u32(out_row + tidx * 8, w0, w1)
        if (return_lse != 0) & (tidx == 0):
            _stg_f32(lse.iterator.toint() + row_id * 4, _lse(m_run, l_run))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


MAX_SPLITS = 16
MAX_GRID_ROWS = 65535  # CUDA's limit on a grid's y extent
MAX_CHUNKS = 8  # chunks per selection (the Triton core's count)


# Measured on an RTX PRO 6000 (188 SMs), CUDA graph, 2048 selected slots, one
# CTA per program: the time of a CTA that runs all 8 chunks of its program,
# microseconds, by (E4M3, block_h). A CTA that takes ``1 / k`` of the chunks
# adds about ``_SPLIT_CTA_US`` for its prologue and the stored states, and the
# combine kernel ``_COMBINE_US`` once per launch with split programs
# (BF16 32-head, k = 2 / 4 / 8: 79 / 43 / 25 us against 138 / k).
_UNSPLIT_US = {
    (False, 16): 115.0,
    (False, 32): 138.0,
    (True, 16): 62.0,
    (True, 32): 81.0,
}
_SPLIT_CTA_US = 4.0
_COMBINE_US = 4.0
_PLAN_MARGIN = (
    1.08  # a stream-K plan must beat the base plan by this factor in the model
)
_MAX_PLANNED_WAVES = 64  # beyond this the last partial wave is below 1.6%


def _tail_end(ready: list[tuple[float, int]], duration: float, count: int) -> float:
    """When the last of ``count`` CTAs of ``duration`` finishes.

    ``ready`` lists ``(time an SM frees up, SMs)``; the hardware hands the
    next CTA to whichever SM is free first, so the finish time is the least
    ``m`` for which the SMs can run ``count`` CTAs by ``m``.
    """

    def capacity(m: float) -> int:
        return sum(
            n * max(int((m - t) / duration + 1e-9), 0) for t, n in ready if m > t
        )

    low = min(t for t, _ in ready)
    high = low + duration * count
    for _ in range(60):
        mid = (low + high) / 2
        if capacity(mid) >= count:
            high = mid
        else:
            low = mid
    return high


def _launch_time(
    programs: int,
    full: int,
    splits: int,
    num_sms: int,
    unsplit_us: float,
    fixed_us: float = 0.0,
    split_cta_us: float = _SPLIT_CTA_US,
    combine_us: float = _COMBINE_US,
) -> float:
    """Modelled time of ``full`` whole programs, then the rest split ``splits`` ways.

    ``fixed_us`` of a whole program's ``unsplit_us`` is per-CTA setup that a
    split CTA pays in full; the rest divides among the splits.
    """
    waves, last = divmod(full, num_sms)
    ready = [(0.0, num_sms)]
    end = 0.0
    if full:
        waves += last > 0
        end = waves * unsplit_us
        busy = last or num_sms
        ready = [((waves - 1) * unsplit_us, num_sms - busy), (end, busy)]
        ready = [(t, n) for t, n in ready if n]
    tail = programs - full
    if tail == 0:
        return end
    cta_us = fixed_us + (unsplit_us - fixed_us) / splits + split_cta_us
    return max(end, _tail_end(ready, cta_us, tail * splits)) + combine_us


def _uniform_splits(programs: int, block_h: int, num_sms: int) -> int:
    """CTAs per program when every program is split alike (1: none are).

    The split with the fewest chunk times, ``waves * chunks per CTA`` (ties go
    to fewer splits, which store fewer chunk states), taken only if it needs at
    least 1.25x (16-head CTAs) or 1.5x (32-head CTAs) fewer chunk times than
    unsplit, because the stored states cost what that count does not (16-head
    BF16, 512 tokens: 4 splits are 0.76x of unsplit).
    """
    chunk_times = {
        splits: -(-programs * splits // num_sms) * (MAX_CHUNKS // splits)
        for splits in (1, 2, 4, MAX_CHUNKS)
    }
    best = min((times, splits) for splits, times in chunk_times.items())[1]
    margin = 1.25 if block_h == 16 else 1.5
    return best if chunk_times[best] * margin < chunk_times[1] else 1


@functools.lru_cache(maxsize=4096)
def plan_launch(
    tokens: int, heads: int, num_sms: int, fp8: bool
) -> tuple[int, int, int]:
    """``(block_h, num_splits, full_tokens)`` for a batch of ``tokens`` queries.

    No value changes the result. A program is one (token, head tile); the
    launch runs the programs of the first ``full_tokens`` tokens in one CTA
    each and shares every later program among ``num_splits`` CTAs, each taking
    a run of ``MAX_CHUNKS / num_splits`` chunks whose states a second kernel
    folds.

    32 heads share a gathered tile once the 32-head CTAs alone can cover the
    SMs, 16 below that (the Triton rule, ``_triton.sparse_mma.plan_launch``,
    switches at two thirds of the SMs; a 16-head CTA wins at that crossing by
    1.07-1.14x).

    A CTA fills an SM, so a launch runs in waves of ``num_sms`` CTAs, and one
    CTA more than SMs costs a whole second wave (BF16, 32-head programs: 188
    take 149 us, 189 take 287). The base plan splits every program alike
    (:func:`_uniform_splits`; none for a batch that fills the SMs). A stream-K
    plan runs whole programs first and splits the rest after them (the
    hardware hands CTAs out in block order), so the short CTAs fill the SMs
    that the last whole wave leaves idle; it replaces the base plan only if
    the model of :func:`_launch_time` puts it ``_PLAN_MARGIN`` ahead. Measured
    on an RTX PRO 6000 (188 SMs) over decode batches of 1-192 tokens at 16-128
    heads, E4M3 and BF16.
    """
    wide_tiles = -(-heads // 32) * tokens * MAX_SPLITS >= num_sms
    block_h = 32 if heads > 16 and wide_tiles else 16
    head_tiles = -(-heads // block_h)
    programs = head_tiles * tokens
    base_splits = _uniform_splits(programs, block_h, num_sms)
    best = (base_splits, 0 if base_splits > 1 else tokens)
    if programs > _MAX_PLANNED_WAVES * num_sms or programs <= num_sms:
        return block_h, best[0], best[1]
    unsplit_us = _UNSPLIT_US[(fp8, block_h)]
    best_us = _launch_time(programs, best[1] * head_tiles, best[0], num_sms, unsplit_us)
    base_us = best_us
    tails = {programs % num_sms + num_sms * j for j in (0, 1, 2)}
    tails |= {num_sms * m // 8 for m in range(1, 17)}
    for splits in (2, 4, MAX_CHUNKS):
        for tail in sorted(tails):
            tail_tokens = -(-tail // head_tiles)
            if not 0 < tail_tokens < tokens:
                continue
            full = (tokens - tail_tokens) * head_tiles
            t = _launch_time(programs, full, splits, num_sms, unsplit_us)
            if t < best_us and t * _PLAN_MARGIN < base_us:
                best_us = t
                best = (splits, tokens - tail_tokens)
    return block_h, best[0], best[1]


def split_magic(num_splits: int) -> int:
    """The multiplier the kernel divides block rows by ``num_splits`` with."""
    return -(-(1 << 32) // num_splits) if num_splits > 1 else 0


def rows_for(fp8: bool) -> int:
    """Selected slots per tile: 64 for E4M3, 32 for BF16."""
    return 64 if fp8 else 32


def stages_for(
    fp8: bool,
    block_h: int,
    rank: int,
    rope: int,
    v4: tuple[bool, bool] | None = None,
    cwarps: int = CHANNEL_WARPS,
) -> int:
    """Ring depth: the most tiles (at most 4) that fit 99 KiB of shared memory."""
    for stages in (4, 3, 2):
        core = _SparseMma(fp8, block_h, rank, rope, stages, v4, cwarps)
        if core.smem_bytes <= 99 * 1024:
            return stages
    raise ValueError("no ring depth fits the shared memory")


def _flat(tensor: torch.Tensor, dtype) -> torch.Tensor:
    return tensor.view(dtype).reshape(-1)


def _fake(dtype, align: int):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (cute.sym_int(),), stride_order=(0,), assumed_align=align
    )


@functools.cache
def compiled_attention(
    fp8: bool,
    block_h: int,
    rank: int,
    rope: int,
    device_index: int,
    v4: tuple[bool, bool, bool] | None = None,
    cwarps: int = CHANNEL_WARPS,
):
    """Compile (once per dtype, head tile, rank and RoPE width) the attention kernel.

    ``v4`` is None for the DSA kernel and ``(sink, wide_ids, wide_lens)`` for the
    DeepSeek V4 variant (``dsv4/_cute_dsl/sparse_mma.py``), whose metadata
    dtypes and sink are compile-time choices.
    """
    stages = stages_for(fp8, block_h, rank, rope, v4 and v4[:2], cwarps)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    wide_ids = bool(v4 and v4[1])
    wide_lens = bool(v4 and v4[2])
    with torch.cuda.device(device_index):
        return cute.compile(
            _SparseMma(fp8, block_h, rank, rope, stages, v4 and v4[:2], cwarps),
            # The V4 variant takes the query and output as addresses.
            cutlass.Int64(0) if v4 else _fake(cutlass.Uint8, 16),
            cutlass.Int64(0),
            _fake(cutlass.Int64 if wide_ids else cutlass.Int32, 8 if wide_ids else 4),
            _fake(cutlass.Int64 if wide_lens else cutlass.Int32, 8 if wide_lens else 4),
            cutlass.Int64(0) if v4 else _fake(cutlass.Uint8, 16),
            _fake(cutlass.Float32, 4),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 4),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Uint32(0),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            cutlass.Int32(0),
            _fake(cutlass.Float32, 4),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            stream=stream,
            options="--enable-tvm-ffi",
        )


@functools.cache
def compiled_combine(rank: int, device_index: int, strided_out: bool = False):
    """Compile (once per rank) the kernel that folds stored chunk states.

    ``strided_out`` takes the output as an address with token and head strides
    (the V4 variant).
    """
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            _Combine(rank, strided_out),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 4),
            cutlass.Int64(0) if strided_out else _fake(cutlass.Uint8, 16),
            _fake(cutlass.Float32, 4),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(0),
            cutlass.Int32(1),
            stream=stream,
            options="--enable-tvm-ffi",
        )


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
    chunk_tiles: int,
    num_chunks: int,
    num_splits: int,
    full_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Attend ``q`` over the cache rows named by ``topk_slots``.

    Args:
        q: ``[tokens, heads, kv_lora_rank + qk_rope_head_dim]`` contiguous,
            E4M3 with an E4M3 cache or BF16 with a BF16 cache.
        kv_cache: ``[pages, page_size, dim]`` (a unit dimension of size one
            may follow the pages or the page slots), unit channel stride and
            16-byte aligned rows.
        topk_slots: ``[tokens, width]`` int32 global slots, ``-1`` for none,
            unit column stride.
        topk_lens: ``[tokens]`` int32 count of leading columns that are live.
        block_h: 16 or 32 heads per CTA.
        chunk_tiles, num_chunks: the chunks of the selection (by its width
            alone; ``_triton.sparse_mma.num_chunks_for``).
        num_splits, full_tokens: the first ``full_tokens`` tokens run each head
            tile in one CTA and fold its chunks in registers; every later token
            shares each head tile among ``num_splits`` CTAs, each taking a run
            of chunks whose stored states a second kernel folds.
            ``full_tokens`` equal to the token count is an unsplit launch, 0
            splits every token. The result is bitwise the same for every
            value.

    Returns:
        ``(out, lse)``: BF16 ``[tokens, heads, kv_lora_rank]`` and the FP32
        natural-log normalizer ``[tokens, heads]`` (or None).
    """
    tokens, heads, head_dim = q.shape
    fp8 = q.dtype == torch.float8_e4m3fn
    if kv_cache.dtype != q.dtype or q.dtype not in (
        torch.float8_e4m3fn,
        torch.bfloat16,
    ):
        raise ValueError(
            f"query and cache must both be E4M3 or both BF16, got {q.dtype} and {kv_cache.dtype}"
        )
    if head_dim != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(f"query head dim {head_dim} != rank + RoPE width")
    if kv_cache.dim() == 4:
        kv_cache = (
            kv_cache.squeeze(1) if kv_cache.shape[1] == 1 else kv_cache.squeeze(2)
        )
    elif kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        kv_cache = kv_cache.squeeze(1)
    if kv_cache.dim() == 2:
        kv_cache = kv_cache.unsqueeze(1)
    esize = kv_cache.element_size()
    page_stride = kv_cache.stride(0) * esize
    token_stride = kv_cache.stride(1) * esize
    if (
        kv_cache.stride(-1) != 1
        or kv_cache.shape[-1] < head_dim
        or page_stride % 16
        or token_stride % 16
        or kv_cache.data_ptr() % 16
    ):
        raise ValueError("the cache needs unit channel stride and 16-byte aligned rows")
    if not (
        q.is_contiguous() and topk_slots.is_contiguous() and topk_lens.is_contiguous()
    ):
        raise ValueError("q, topk_slots and topk_lens must be contiguous")
    device = q.device
    out = torch.empty(
        (tokens, heads, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    lse = (
        torch.empty((tokens, heads), dtype=torch.float32, device=device)
        if return_lse
        else None
    )
    if tokens == 0 or heads == 0:
        return out, lse
    num_splits = max(min(num_splits, num_chunks), 1)
    head_tiles = -(-heads // block_h)
    if not 0 <= full_tokens <= tokens:
        raise ValueError(f"full_tokens {full_tokens} outside [0, {tokens}]")
    if num_splits == 1:
        full_tokens = tokens
    tail_tokens = tokens - full_tokens
    if full_tokens + tail_tokens * num_splits > MAX_GRID_ROWS:
        raise ValueError(
            f"{full_tokens} whole + {tail_tokens} split tokens x {num_splits} "
            f"splits exceed the {MAX_GRID_ROWS} rows of a grid"
        )
    split = tail_tokens > 0
    if split:
        # States are [tail token][chunk][head].
        part_acc = torch.empty(
            (tail_tokens, num_chunks, heads, kv_lora_rank),
            dtype=torch.float32,
            device=device,
        )
        part_ml = torch.empty(
            (tail_tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = lse if return_lse else torch.empty(4, device=device)
    lse_arg = lse if return_lse else part_ml
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    kernel = compiled_attention(
        fp8, block_h, kv_lora_rank, qk_rope_head_dim, device_index
    )
    kernel(
        _flat(q, torch.uint8),
        kv_cache.data_ptr(),
        topk_slots.reshape(-1),
        topk_lens,
        _flat(out, torch.uint8),
        lse_arg.reshape(-1),
        part_acc.reshape(-1),
        part_ml.reshape(-1),
        heads,
        topk_slots.shape[1],
        topk_slots.stride(0),
        page_stride,
        token_stride,
        kv_cache.shape[1],
        chunk_tiles,
        num_chunks,
        -(-num_chunks // num_splits),
        full_tokens,
        num_splits,
        split_magic(num_splits),
        int(return_lse),
        float(softmax_scale) * 1.4426950408889634,
        0,
        part_ml.reshape(-1),
        heads * head_dim * esize,
        head_dim * esize,
        heads * kv_lora_rank * 2,
        kv_lora_rank * 2,
        head_tiles,
        full_tokens + tail_tokens * num_splits,
    )
    if split:
        compiled_combine(kv_lora_rank, device_index)(
            part_acc.reshape(-1),
            part_ml.reshape(-1),
            _flat(out, torch.uint8),
            lse_arg.reshape(-1),
            heads,
            num_chunks,
            int(return_lse),
            0,
            0,
            full_tokens,
            tail_tokens,
        )
    return out, lse
