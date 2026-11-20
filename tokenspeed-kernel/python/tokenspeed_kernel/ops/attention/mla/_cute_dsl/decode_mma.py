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
"""CuTe DSL tensor-core MLA decode over a paged latent cache (SM120).

The kernel is the DSA CuTe DSL core (``dsa/_cute_dsl/sparse_mma.py``) with the
gather and the mask of ``_triton/decode_mma.py``: same contract, same masks,
same split-and-merge launch as the Triton kernel, same tile math as the sparse
core (``cp.async`` ring, ``ldmatrix`` fragments, ``mma.sync``, FP32 online
softmax).

A CTA owns ``block_m`` (16 or 32) rows of one request, a row being a
``(query, head)`` pair, so one gathered tile of cache rows is multiplied by all
of them. Instead of a list of selected slots, the CTA walks the contiguous range
``[lo, hi)`` that its rows can see, one tile of consecutive positions at a time,
and resolves each position through the request's page table. Row ``i`` of a
request with ``L`` cached tokens and ``Q`` query rows sees positions
``first <= p < visible``:

* causal block (``block_size == 1``): ``visible = L - (Q - 1 - i)``;
* non-causal block (``block_size > 1``): ``visible = L``;
* with ``window_left >= 0``: ``first = max(visible - block_size - window_left +
  position, 0)`` (``position`` is the row's index in its non-causal block, 0 for
  a causal block); otherwise ``first = 0``.

When a batch has too few CTAs to fill the GPU the range is split over
``num_splits`` CTAs, each storing a partial softmax state that a second kernel
merges. The launch shape follows the batch size only, never the cached lengths
(a CUDA graph captured per batch size replays with fixed launch parameters), and
``max_seqlen_k`` is not an input. Like the Triton kernel it is not invariant to
the batch size: the split changes the order the FP32 sums are taken in.

Everything that follows the batch or the metadata is a runtime argument; a
binary is keyed by the cache dtype and the row tile only.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from tokenspeed_kernel.ops.attention.dsa._cute_dsl import sparse_mma as _core
from tokenspeed_kernel.ops.attention.dsa._cute_dsl.sparse_mma import (
    CHANNEL_WARPS,
    _cp_async_16,
    _ex2,
    _finish,
    _fma,
    _ldg_f32,
    _ldg_s32,
    _ldg_u32,
    _ldg_v4_f32,
    _lse,
    _mul,
    _pack_16bit,
    _shfl_xor,
    _SparseMma,
    _stg_f32,
    _stg_u32,
    _stg_v2_f32,
    _stg_v2_u32,
    _stg_v4_f32,
)
from tokenspeed_kernel.ops.attention.mla._triton.decode_mma import MAX_SPLITS
from tokenspeed_kernel.platform import current_platform

__all__ = ["mla_mma_decode", "plan_launch", "supports"]

_NEG_INF = float("-inf")
_FAR = 1 << 30
# Partial states the combine kernel loads before it accumulates them.
_COMBINE_UNROLL = 8


@cute.jit
def _lds_f32(addr):
    return cute.arch.inline_ptx(
        "ld.shared.f32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )


@cute.jit
def _sts_f32(addr, v):
    cute.arch.inline_ptx("st.shared.f32 [{$r0}], {$r1};", read_only_args=[addr, v])


class _MlaDecode(_SparseMma):
    """One CTA per (row tile, request, split); see the module docstring."""

    def __init__(self, fp8: bool, block_m: int, rank: int, rope: int, stages: int):
        super().__init__(fp8, block_m, rank, rope, stages)
        self.block_m = block_m

    @cute.jit
    def __call__(
        self,
        q_addr: cutlass.Int64,
        kv_addr: cutlass.Int64,
        table_addr: cutlass.Int64,
        lens: cute.Tensor,
        out_addr: cutlass.Int64,
        lse_addr: cutlass.Int64,
        acc_addr: cutlass.Int64,
        ml_addr: cutlass.Int64,
        q_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        num_splits: cutlass.Int32,
        block_size: cutlass.Int32,
        window_left: cutlass.Int32,
        return_lse: cutlass.Int32,
        page_shift: cutlass.Int32,
        num_batch: cutlass.Int32,
        q_b: cutlass.Int64,
        q_q: cutlass.Int64,
        q_h: cutlass.Int64,
        page_stride: cutlass.Int64,
        token_stride: cutlass.Int64,
        table_stride: cutlass.Int64,
        out_b: cutlass.Int64,
        out_q: cutlass.Int64,
        out_h: cutlass.Int64,
        lse_b: cutlass.Int64,
        lse_q: cutlass.Int64,
        scale_log2: cutlass.Float32,
        grid_x: cutlass.Int32,
        grid_y: cutlass.Int32,
        grid_z: cutlass.Int32,
        stream,
    ):
        self.kernel(
            q_addr,
            kv_addr,
            table_addr,
            lens,
            out_addr,
            lse_addr,
            acc_addr,
            ml_addr,
            q_len,
            num_heads,
            num_splits,
            block_size,
            window_left,
            return_lse,
            page_shift,
            num_batch,
            q_b,
            q_q,
            q_h,
            page_stride,
            token_stride,
            table_stride,
            out_b,
            out_q,
            out_h,
            lse_b,
            lse_q,
            scale_log2,
        ).launch(
            grid=(grid_x, grid_y, grid_z),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _visible_range(self, query, length, q_len, batch, block_size, window_left):
        """``(visible, first)`` of one query row: it sees ``first <= p < visible``."""
        visible = length
        if block_size == 1:
            visible = length - (q_len - 1 - query)
        visible = cutlass.max(visible, cutlass.Int32(0))
        first = cutlass.Int32(0)
        if window_left >= 0:
            position = (batch * q_len + query) % block_size
            first = cutlass.max(
                visible - block_size - window_left + position, cutlass.Int32(0)
            )
        return visible, first

    @cute.jit
    def _issue_tile_mla(
        self,
        tile,
        tile_hi,
        lo,
        hi,
        page,
        stage,
        kv_addr,
        page_stride,
        token_stride,
        page_shift,
        ring_addr,
        tidx,
    ):
        """``cp.async`` one tile of consecutive positions into ``stage``.

        This thread's row is position ``lo + tile * rows + row``; ``page`` is
        that position's page id (``-1`` if it is past the range).
        """
        rows = cutlass.const_expr(self.rows)
        row = tidx // self.threads_per_row
        part = tidx % self.threads_per_row
        pos = lo + tile * rows + row
        ok = (tile < tile_hi) & (pos < hi) & (page >= 0)
        live = cutlass.Int32(0)
        src = kv_addr
        if ok:
            live = cutlass.Int32(16)
            slot = pos & ((cutlass.Int32(1) << page_shift) - 1)
            src = (
                kv_addr
                + cutlass.Int64(page) * page_stride
                + cutlass.Int64(slot) * token_stride
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

    @cute.kernel
    def kernel(
        self,
        q_addr: cutlass.Int64,
        kv_addr: cutlass.Int64,
        table_addr: cutlass.Int64,
        lens: cute.Tensor,
        out_addr: cutlass.Int64,
        lse_addr: cutlass.Int64,
        acc_addr: cutlass.Int64,
        ml_addr: cutlass.Int64,
        q_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        num_splits: cutlass.Int32,
        block_size: cutlass.Int32,
        window_left: cutlass.Int32,
        return_lse: cutlass.Int32,
        page_shift: cutlass.Int32,
        num_batch: cutlass.Int32,
        q_b: cutlass.Int64,
        q_q: cutlass.Int64,
        q_h: cutlass.Int64,
        page_stride: cutlass.Int64,
        token_stride: cutlass.Int64,
        table_stride: cutlass.Int64,
        out_b: cutlass.Int64,
        out_q: cutlass.Int64,
        out_h: cutlass.Int64,
        lse_b: cutlass.Int64,
        lse_q: cutlass.Int64,
        scale_log2: cutlass.Float32,
    ):
        fp8 = cutlass.const_expr(self.fp8)
        rank = cutlass.const_expr(self.rank)
        rope = cutlass.const_expr(self.rope)
        stages = cutlass.const_expr(self.stages)
        rows = cutlass.const_expr(self.rows)
        rs = cutlass.const_expr(self.row_stride)
        n_tiles = cutlass.const_expr(self.n_tiles)
        s_tiles = cutlass.const_expr(self.s_tiles)
        neg_inf = cutlass.Float32(_NEG_INF)
        far = cutlass.Int32(_FAR)

        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, split_id = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        g = lane // 4
        t = lane % 4
        hg = warp // CHANNEL_WARPS  # row group of 16
        cw = warp % CHANNEL_WARPS  # channel (and, for E4M3 scores, key) group

        smem = _core.SmemAllocator()
        pool = smem.allocate_tensor(
            element_type=cutlass.Uint8,
            layout=cute.make_layout((self.smem_bytes,)),
            byte_alignment=128,
        )
        ring_addr = cutlass.Int32(pool.iterator.toint())
        q_smem = ring_addr + self.q_offset
        x_addr = ring_addr + self.x_offset

        # ---- The positions this CTA's rows can see, and its share of them ----
        rows_total = q_len * num_heads
        length = _ldg_s32(lens.iterator.toint() + cutlass.Int64(batch) * 4)
        row_lo = m_block * self.block_m
        row_hi = cutlass.min(row_lo + self.block_m, rows_total)
        first_min = far
        visible_max = cutlass.Int32(0)
        for query in range(row_lo // num_heads, (row_hi - 1) // num_heads + 1):
            vis, fst = self._visible_range(
                query, length, q_len, batch, block_size, window_left
            )
            first_min = cutlass.min(first_min, fst)
            visible_max = cutlass.max(visible_max, vis)
        lo = (first_min // rows) * rows
        hi = visible_max
        num_tiles = (cutlass.max(hi - lo, cutlass.Int32(0)) + (rows - 1)) // rows
        tiles_per_split = (num_tiles + num_splits - 1) // num_splits
        tile_lo = split_id * tiles_per_split
        tile_hi = cutlass.min(tile_lo + tiles_per_split, num_tiles)

        table_row = table_addr + cutlass.Int64(batch) * table_stride
        my_row = tidx // self.threads_per_row

        # ---- This thread's two rows of the head group ----
        row0 = m_block * self.block_m + hg * 16 + g
        row1 = row0 + 8
        live0 = row0 < rows_total
        live1 = row1 < rows_total
        query0 = row0 // num_heads
        head0 = row0 - query0 * num_heads
        query1 = row1 // num_heads
        head1 = row1 - query1 * num_heads
        vis0, fst0 = self._visible_range(
            query0, length, q_len, batch, block_size, window_left
        )
        vis1, fst1 = self._visible_range(
            query1, length, q_len, batch, block_size, window_left
        )
        if row0 >= rows_total:
            vis0 = cutlass.Int32(0)
            fst0 = far
        if row1 >= rows_total:
            vis1 = cutlass.Int32(0)
            fst1 = far

        qa = cute.make_rmem_tensor((max(self.qk_steps, 1) * 4,), cutlass.Uint32)
        qr = cute.make_rmem_tensor((4,), cutlass.Uint32)
        q_batch = q_addr + cutlass.Int64(batch) * q_b
        if cutlass.const_expr(fp8):
            # ---- The query tile goes to shared memory with the first tiles ----
            per_tile = cutlass.const_expr(self.block_m * self.chunks_per_row)
            for k in cutlass.range_constexpr(-(-per_tile // self.num_threads)):
                idx = tidx + k * self.num_threads
                if idx < per_tile:
                    h = idx // self.chunks_per_row
                    c = idx % self.chunks_per_row
                    r = m_block * self.block_m + h
                    live_h = cutlass.Int32(0)
                    if r < rows_total:
                        live_h = cutlass.Int32(16)
                    qh = r // num_heads
                    hh = r - qh * num_heads
                    _cp_async_16(
                        q_smem + self._chunk_offset(h, c),
                        q_batch
                        + cutlass.Int64(qh) * q_q
                        + cutlass.Int64(hh) * q_h
                        + cutlass.Int64(c * 16),
                        live_h,
                    )
        else:
            # ---- This warp's query channels, kept in registers ----
            q_row0 = q_batch + cutlass.Int64(query0) * q_q + cutlass.Int64(head0) * q_h
            q_row1 = q_batch + cutlass.Int64(query1) * q_q + cutlass.Int64(head1) * q_h
            for i in cutlass.range_constexpr(self.qk_steps * 4):
                qa[i] = cutlass.Uint32(0)
            for i in cutlass.range_constexpr(4):
                qr[i] = cutlass.Uint32(0)
            for ks in cutlass.range_constexpr(self.qk_steps):
                ch = cw * self.warp_ch + ks * 16
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

        # ---- Softmax state: rows g and g + 8 of the row group ----
        acc = cute.make_rmem_tensor((n_tiles * 4,), cutlass.Float32)
        for i in cutlass.range_constexpr(n_tiles * 4):
            acc[i] = cutlass.Float32(0.0)
        m_i0 = neg_inf
        m_i1 = neg_inf
        l_i0 = cutlass.Float32(0.0)
        l_i1 = cutlass.Float32(0.0)
        s = cute.make_rmem_tensor((s_tiles * 4,), cutlass.Float32)
        q_off = cute.make_rmem_tensor((6,), cutlass.Int32)
        k_off = cute.make_rmem_tensor((6,), cutlass.Int32)
        if cutlass.const_expr(fp8):
            mi = lane // 8
            q_row = hg * 16 + (lane % 8) + 8 * (mi % 2)
            k_row = cw * 16 + 8 * (mi // 2) + (lane % 8)
            yq = (mi // 2) ^ (q_row % 8)
            yk = (mi % 2) ^ (k_row % 8)
            for m in cutlass.range_constexpr(4):
                q_off[m] = q_smem + q_row * rs + ((2 * m) ^ yq) * 16
                k_off[m] = k_row * rs + ((2 * m) ^ yk) * 16
            if cutlass.const_expr(rope > 0):
                yq = (mi // 2) ^ (q_row % 4)
                yk = (mi % 2) ^ (k_row % 4)
                for m in cutlass.range_constexpr(2):
                    q_off[4 + m] = (
                        q_smem + q_row * rs + (self.latent_chunks + ((2 * m) ^ yq)) * 16
                    )
                    k_off[4 + m] = (
                        k_row * rs + (self.latent_chunks + ((2 * m) ^ yk)) * 16
                    )
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
            pos = lo + tile * rows + my_row
            page = cutlass.Int32(-1)
            if (tile < tile_hi) & (pos < hi):
                page = _ldg_s32(table_row + cutlass.Int64(pos >> page_shift) * 4)
            self._issue_tile_mla(
                tile,
                tile_hi,
                lo,
                hi,
                page,
                cutlass.Int32(st),
                kv_addr,
                page_stride,
                token_stride,
                page_shift,
                ring_addr,
                tidx,
            )
            cute.arch.cp_async_commit_group()
        # The page of the next tile to issue, loaded one iteration early.
        next_tile = tile_lo + (stages - 1)
        next_pos = lo + next_tile * rows + my_row
        next_page = cutlass.Int32(-1)
        if (next_tile < tile_hi) & (next_pos < hi):
            next_page = _ldg_s32(table_row + cutlass.Int64(next_pos >> page_shift) * 4)

        for tile in range(tile_lo, tile_hi):
            i = tile - tile_lo
            stage = i % stages
            cute.arch.cp_async_wait_group(stages - 2)
            cute.arch.barrier()
            # Refill the stage tile - 1 used; every warp is past it.
            issue = tile + (stages - 1)
            self._issue_tile_mla(
                issue,
                tile_hi,
                lo,
                hi,
                next_page,
                (i + stages - 1) % stages,
                kv_addr,
                page_stride,
                token_stride,
                page_shift,
                ring_addr,
                tidx,
            )
            cute.arch.cp_async_commit_group()
            next_pos = lo + (issue + 1) * rows + my_row
            next_page = cutlass.Int32(-1)
            if (issue + 1 < tile_hi) & (next_pos < hi):
                next_page = _ldg_s32(
                    table_row + cutlass.Int64(next_pos >> page_shift) * 4
                )

            base = ring_addr + stage * (rows * rs)
            # ---- Full scores of the row group ----
            if cutlass.const_expr(fp8):
                self._scores_key_split(s, base, q_off, k_off, x_addr, hg, cw, lane)
            else:
                self._scores_channel_split(s, base, qa, qr, x_addr, hg, cw, lane, g, t)
            # ---- Mask by position: first <= p < visible, per row ----
            pos0 = lo + tile * rows
            mx0 = neg_inf
            mx1 = neg_inf
            for j in cutlass.range_constexpr(s_tiles):
                for e in cutlass.range_constexpr(2):
                    pos = pos0 + (8 * j + 2 * t + e)
                    v0 = neg_inf
                    v1 = neg_inf
                    if (pos >= fst0) & (pos < vis0):
                        v0 = s[4 * j + e]
                    if (pos >= fst1) & (pos < vis1):
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
                tile != tile_lo,
            )

        cute.arch.cp_async_wait_group(0)
        if num_splits > 1:
            # Part of [split][request row][head]; a split past the range
            # stores the empty state.
            part_row = cutlass.Int64(split_id * num_batch + batch) * cutlass.Int64(
                rows_total
            )
            self._store_partial(
                acc_addr,
                ml_addr,
                acc,
                m_i0,
                m_i1,
                l_i0,
                l_i1,
                part_row + cutlass.Int64(row0),
                live0,
                live1,
                tile_lo >= tile_hi,
                cw,
                t,
            )
        else:
            self._store_result(
                out_addr + cutlass.Int64(batch) * out_b,
                lse_addr + cutlass.Int64(batch) * lse_b,
                acc,
                m_i0,
                m_i1,
                l_i0,
                l_i1,
                query0,
                head0,
                query1,
                head1,
                live0,
                live1,
                return_lse,
                out_q,
                out_h,
                lse_q,
                cw,
                t,
            )

    @cute.jit
    def _store_partial(
        self, acc_addr, ml_addr, acc, m0, m1, l0, l1, part0, live0, live1, empty, cw, t
    ):
        """Store the CTA's softmax state; an empty range stores ``l = 0`` only,
        which the merge reads as a weight of zero without touching the sums."""
        rank = cutlass.const_expr(self.rank)
        tpg = cutlass.const_expr(self.tpg)
        for half in cutlass.range_constexpr(2):
            live = live0
            if cutlass.const_expr(half == 1):
                live = live1
            if live:
                part = part0 + 8 * half
                row = acc_addr + part * (rank * 4)
                if not empty:
                    for grp in cutlass.range_constexpr(self.groups):
                        ch = self._lane_channel(cw, grp, t)
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
                    _stg_f32(ml_addr + part * 8, m)
                    _stg_f32(ml_addr + part * 8 + 4, l)

    @cute.jit
    def _store_result(
        self,
        out_batch,
        lse_batch,
        acc,
        m0,
        m1,
        l0,
        l1,
        query0,
        head0,
        query1,
        head1,
        live0,
        live1,
        return_lse,
        out_q,
        out_h,
        lse_q,
        cw,
        t,
    ):
        tpg = cutlass.const_expr(self.tpg)
        for half in cutlass.range_constexpr(2):
            live = live0
            m = m0
            l = l0
            query = query0
            head = head0
            if cutlass.const_expr(half == 1):
                live = live1
                m = m1
                l = l1
                query = query1
                head = head1
            if live:
                row = (
                    out_batch
                    + cutlass.Int64(query) * out_q
                    + cutlass.Int64(head) * out_h
                )
                for grp in cutlass.range_constexpr(self.groups):
                    ch = self._lane_channel(cw, grp, t)
                    words = []
                    for i in cutlass.range_constexpr(self.cpl // 2):
                        pair = []
                        for k in cutlass.range_constexpr(2):
                            idx = 2 * i + k
                            n = grp * tpg + idx % tpg
                            pair.append(_finish(acc[4 * n + 2 * half + idx // tpg], l))
                        words.append(_pack_16bit(False, pair[0], pair[1]))
                    if cutlass.const_expr(self.cpl == 4):
                        _stg_v2_u32(row + cutlass.Int64(ch * 2), words[0], words[1])
                    else:
                        _stg_u32(row + cutlass.Int64(ch * 2), words[0])
                if (return_lse != 0) & (cw == 0) & (t == 0):
                    _stg_f32(
                        lse_batch
                        + cutlass.Int64(query) * lse_q
                        + cutlass.Int64(head) * 4,
                        _lse(m, l),
                    )


class _MlaCombine:
    """Merge the stored partial states of one (request row, head)."""

    def __init__(self, rank: int):
        self.rank = rank
        self.num_threads = rank // 4

    @cute.jit
    def __call__(
        self,
        acc: cute.Tensor,
        ml_addr: cutlass.Int64,
        out_addr: cutlass.Int64,
        lse_addr: cutlass.Int64,
        q_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        num_splits: cutlass.Int32,
        num_rows: cutlass.Int32,
        return_lse: cutlass.Int32,
        out_b: cutlass.Int64,
        out_q: cutlass.Int64,
        out_h: cutlass.Int64,
        lse_b: cutlass.Int64,
        lse_q: cutlass.Int64,
        stream,
    ):
        self.kernel(
            acc,
            ml_addr,
            out_addr,
            lse_addr,
            q_len,
            num_heads,
            num_splits,
            num_rows,
            return_lse,
            out_b,
            out_q,
            out_h,
            lse_b,
            lse_q,
        ).launch(
            grid=(num_rows, num_heads, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        acc: cute.Tensor,
        ml_addr: cutlass.Int64,
        out_addr: cutlass.Int64,
        lse_addr: cutlass.Int64,
        q_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        num_splits: cutlass.Int32,
        num_rows: cutlass.Int32,
        return_lse: cutlass.Int32,
        out_b: cutlass.Int64,
        out_q: cutlass.Int64,
        out_h: cutlass.Int64,
        lse_b: cutlass.Int64,
        lse_q: cutlass.Int64,
    ):
        rank = cutlass.const_expr(self.rank)
        acc_addr = acc.iterator.toint()
        tidx, _, _ = cute.arch.thread_idx()
        request_row, head, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        batch = request_row // q_len
        query = request_row - batch * q_len
        neg_inf = cutlass.Float32(_NEG_INF)
        smem = _core.SmemAllocator()
        pool = smem.allocate_tensor(
            element_type=cutlass.Uint8,
            layout=cute.make_layout((8 * 4 + 128 * 4,)),
            byte_alignment=16,
        )
        scratch = cutlass.Int32(pool.iterator.toint())
        weights = scratch + 32
        # One thread per split (at most 128): the state of split ``tidx``.
        m_part = neg_inf
        l_part = cutlass.Float32(0.0)
        if tidx < num_splits:
            part = (
                cutlass.Int64(tidx) * cutlass.Int64(num_rows)
                + cutlass.Int64(request_row)
            ) * cutlass.Int64(num_heads) + cutlass.Int64(head)
            m_part = _ldg_f32(ml_addr + part * 8)
            l_part = _ldg_f32(ml_addr + part * 8 + 4)
        # Row maximum over the splits, in a fixed order.
        m_max = m_part
        m_max = cute.arch.fmax(m_max, _shfl_xor(m_max, 16))
        m_max = cute.arch.fmax(m_max, _shfl_xor(m_max, 8))
        m_max = cute.arch.fmax(m_max, _shfl_xor(m_max, 4))
        m_max = cute.arch.fmax(m_max, _shfl_xor(m_max, 2))
        m_max = cute.arch.fmax(m_max, _shfl_xor(m_max, 1))
        if lane == 0:
            _sts_f32(scratch + warp * 4, m_max)
        cute.arch.barrier()
        m_all = _lds_f32(scratch)
        for w in cutlass.range_constexpr(1, 4):
            m_all = cute.arch.fmax(m_all, _lds_f32(scratch + w * 4))
        m_safe = m_all
        if m_all == neg_inf:
            m_safe = cutlass.Float32(0.0)
        weight = cutlass.Float32(0.0)
        if l_part > cutlass.Float32(0.0):
            weight = _ex2(m_part - m_safe)
        _sts_f32(weights + tidx * 4, weight)
        l_sum = _mul(weight, l_part)
        l_sum = l_sum + _shfl_xor(l_sum, 16)
        l_sum = l_sum + _shfl_xor(l_sum, 8)
        l_sum = l_sum + _shfl_xor(l_sum, 4)
        l_sum = l_sum + _shfl_xor(l_sum, 2)
        l_sum = l_sum + _shfl_xor(l_sum, 1)
        if lane == 0:
            _sts_f32(scratch + 16 + warp * 4, l_sum)
        cute.arch.barrier()
        l_all = _lds_f32(scratch + 16)
        for w in cutlass.range_constexpr(1, 4):
            l_all = l_all + _lds_f32(scratch + 16 + w * 4)
        # Weighted sum of the partial outputs; a split of weight zero is not read
        # (its row may be unwritten), and the loads of a group are issued first.
        a0 = cutlass.Float32(0.0)
        a1 = cutlass.Float32(0.0)
        a2 = cutlass.Float32(0.0)
        a3 = cutlass.Float32(0.0)
        loaded = cute.make_rmem_tensor((_COMBINE_UNROLL * 4,), cutlass.Float32)
        wts = cute.make_rmem_tensor((_COMBINE_UNROLL,), cutlass.Float32)
        for base in range(0, num_splits, _COMBINE_UNROLL):
            for u in cutlass.range_constexpr(_COMBINE_UNROLL):
                split = base + u
                w_u = cutlass.Float32(0.0)
                if split < num_splits:
                    w_u = _lds_f32(weights + split * 4)
                wts[u] = w_u
                for k in cutlass.range_constexpr(4):
                    loaded[4 * u + k] = cutlass.Float32(0.0)
                if w_u > cutlass.Float32(0.0):
                    part = (
                        cutlass.Int64(split) * cutlass.Int64(num_rows)
                        + cutlass.Int64(request_row)
                    ) * cutlass.Int64(num_heads) + cutlass.Int64(head)
                    p0, p1, p2, p3 = _ldg_v4_f32(
                        acc_addr + part * (rank * 4) + tidx * 16
                    )
                    loaded[4 * u] = p0
                    loaded[4 * u + 1] = p1
                    loaded[4 * u + 2] = p2
                    loaded[4 * u + 3] = p3
            for u in cutlass.range_constexpr(_COMBINE_UNROLL):
                a0 = _fma(loaded[4 * u], wts[u], a0)
                a1 = _fma(loaded[4 * u + 1], wts[u], a1)
                a2 = _fma(loaded[4 * u + 2], wts[u], a2)
                a3 = _fma(loaded[4 * u + 3], wts[u], a3)
        row = (
            out_addr
            + cutlass.Int64(batch) * out_b
            + cutlass.Int64(query) * out_q
            + cutlass.Int64(head) * out_h
        )
        w0 = _pack_16bit(False, _finish(a0, l_all), _finish(a1, l_all))
        w1 = _pack_16bit(False, _finish(a2, l_all), _finish(a3, l_all))
        _stg_v2_u32(row + tidx * 8, w0, w1)
        if (return_lse != 0) & (tidx == 0):
            _stg_f32(
                lse_addr
                + cutlass.Int64(batch) * lse_b
                + cutlass.Int64(query) * lse_q
                + cutlass.Int64(head) * 4,
                _lse(m_safe, l_all),
            )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def plan_launch(batch: int, rows: int, num_sms: int) -> tuple[int, int]:
    """``(block_m, num_splits)`` for ``batch`` requests of ``rows`` rows.

    ``rows`` is ``q_len * num_heads``. The launch shape depends on the batch
    size only, never on the cached lengths (see the module docstring).
    """
    wide = -(-rows // 32) * batch * MAX_SPLITS >= num_sms * 2 // 3
    block_m = 32 if rows > 16 and wide else 16
    programs = -(-rows // block_m) * batch
    target = num_sms * 2 // 3 if batch <= 8 else num_sms * 11 // 4
    num_splits = min(max(round(target / programs), 1), MAX_SPLITS)
    return block_m, num_splits


def supports(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    lse: torch.Tensor | None,
) -> bool:
    """Whether this kernel's vector loads and stores can address the tensors.

    The BF16 query is read in 4-byte words and the E4M3 query is copied in
    16-byte chunks; the cache rows are copied in 16-byte chunks; the output is
    written in 8-byte words. The answer depends on the tensors' layout and
    dtype, never on the batch contents.
    """
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    if kv_cache.dtype != q.dtype or out.dtype != torch.bfloat16:
        return False
    q_align = 16 if q.dtype == torch.float8_e4m3fn else 4
    esize = kv_cache.element_size()
    return (
        q.data_ptr() % q_align == 0
        and all(q.stride(i) * q.element_size() % q_align == 0 for i in range(3))
        and kv_cache.data_ptr() % 16 == 0
        and kv_cache.stride(0) * esize % 16 == 0
        and kv_cache.stride(1) * esize % 16 == 0
        and out.data_ptr() % 8 == 0
        and all(out.stride(i) % 4 == 0 for i in range(3))
        and page_table.dtype == torch.int32
        and page_table.stride(-1) == 1
        and cache_seqlens.dtype == torch.int32
        and cache_seqlens.is_contiguous()
        and (lse is None or lse.is_contiguous())
    )


@functools.cache
def compiled_decode(fp8: bool, block_m: int, rank: int, rope: int, device_index: int):
    """Compile (once per cache dtype and row tile) the decode kernel."""
    stages = _core.stages_for(fp8, block_m, rank, rope)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64 = cutlass.Int64(0)
    i32 = cutlass.Int32(1)
    with torch.cuda.device(device_index):
        return cute.compile(
            _MlaDecode(fp8, block_m, rank, rope, stages),
            *[i64] * 3,
            _core._fake(cutlass.Int32, 4),
            *[i64] * 4,
            *[i32] * 8,
            *[i64] * 11,
            cutlass.Float32(1.0),
            i32,
            i32,
            i32,
            stream=stream,
            options="--enable-tvm-ffi",
        )


@functools.cache
def compiled_combine(rank: int, device_index: int):
    """Compile (once per rank) the kernel that merges the partial states."""
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64 = cutlass.Int64(0)
    i32 = cutlass.Int32(1)
    with torch.cuda.device(device_index):
        return cute.compile(
            _MlaCombine(rank),
            _core._fake(cutlass.Float32, 16),
            *[i64] * 3,
            *[i32] * 5,
            *[i64] * 5,
            stream=stream,
            options="--enable-tvm-ffi",
        )


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

    Same arguments as ``_triton.decode_mma.mla_mma_decode`` (and the same masks);
    :func:`supports` must hold. ``q`` is E4M3 or BF16 with the cache in its
    dtype and ``out`` is BF16.
    """
    batch, q_len, heads, _ = q.shape
    page_size = kv_cache.shape[1]
    device = q.device
    if batch == 0 or heads == 0:
        return
    if num_splits > 1:
        part_acc = torch.empty(
            (num_splits, batch * q_len, heads, kv_lora_rank),
            dtype=torch.float32,
            device=device,
        )
        part_ml = torch.empty(
            (num_splits, batch * q_len, heads, 2), dtype=torch.float32, device=device
        )
        ml_addr = part_ml.data_ptr()
        acc_addr = part_acc.data_ptr()
    else:
        acc_addr = ml_addr = out.data_ptr()
    lse_strides = (lse.stride(0) * 4, lse.stride(1) * 4) if lse is not None else (0, 0)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    esize = q.element_size()
    compiled_decode(
        q.dtype == torch.float8_e4m3fn,
        block_m,
        kv_lora_rank,
        qk_rope_head_dim,
        device_index,
    )(
        q.data_ptr(),
        kv_cache.data_ptr(),
        page_table.data_ptr(),
        cache_seqlens,
        out.data_ptr(),
        lse.data_ptr() if lse is not None else out.data_ptr(),
        acc_addr,
        ml_addr,
        q_len,
        heads,
        num_splits,
        noncausal_block_size,
        window_left,
        int(lse is not None),
        page_size.bit_length() - 1,
        batch,
        q.stride(0) * esize,
        q.stride(1) * esize,
        q.stride(2) * esize,
        kv_cache.stride(0) * kv_cache.element_size(),
        kv_cache.stride(1) * kv_cache.element_size(),
        page_table.stride(0) * 4,
        out.stride(0) * 2,
        out.stride(1) * 2,
        out.stride(2) * 2,
        lse_strides[0],
        lse_strides[1],
        float(softmax_scale) * 1.4426950408889634,
        -(-q_len * heads // block_m),
        batch,
        num_splits,
    )
    if num_splits > 1:
        compiled_combine(kv_lora_rank, device_index)(
            part_acc.reshape(-1),
            ml_addr,
            out.data_ptr(),
            lse.data_ptr() if lse is not None else out.data_ptr(),
            q_len,
            heads,
            num_splits,
            batch * q_len,
            int(lse is not None),
            out.stride(0) * 2,
            out.stride(1) * 2,
            out.stride(2) * 2,
            *lse_strides,
        )
