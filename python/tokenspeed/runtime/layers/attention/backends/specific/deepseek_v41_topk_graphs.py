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

"""CUDA graphs of the prefill span selection (``dsv41.index_topk``).

A prefill span scores one request's history, so its ``index_topk`` call is a
dense, request-local selection whose launch sequence depends only on the span
length and the history width. Eager, that sequence costs several times its GPU
time in Python and launch overhead, on every index-source layer of every
prefill step. Spans are padded to power-of-two row and page buckets and each
bucket is captured once: a call then costs its input copies, one replay and
its output copies.

Padding is inert: padded rows have position -1, hence no visible history and an
empty selection, and padded pages are null, which the kernel never reads.
Selections over the live rows are those of the eager call (up to the
equal-score ties ``index_topk`` already leaves unspecified).
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_INDEX_SCORE_CHUNK,
)

_MIN_ROWS = 16
_MIN_PAGES = 8
_MIN_POOL = 64


def _bucket(n: int, floor: int) -> int:
    return max(floor, 1 << max(0, n - 1).bit_length())


@dataclass
class _Statics:
    """Inputs and outputs of every graph of one row bucket and output shape."""

    q: torch.Tensor
    weights: torch.Tensor
    positions: torch.Tensor
    candidates: torch.Tensor | None
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    statics: _Statics
    table: torch.Tensor


class V41PrefillTopkGraphs:
    """Bucketed CUDA graphs of dense prefill ``index_topk`` calls.

    Graphs share one memory pool and run strictly one at a time on the model
    stream, so their scratch may alias. Statics are shared by the graphs of a
    row bucket for the same reason; outputs are copied out after each replay.
    """

    def __init__(self, device: torch.device, max_graphs: int = 256) -> None:
        self.device = device
        self.max_graphs = max_graphs
        self._graphs: OrderedDict[tuple, _Graph | None] = OrderedDict()
        self._statics: dict[tuple, _Statics] = {}
        self._pool = None
        self._stream: torch.cuda.Stream | None = None

    def select(
        self,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        index_cache: torch.Tensor,
        table_row: torch.Tensor,
        positions: torch.Tensor,
        ratio: int,
        candidates: torch.Tensor | None,
        topk: int,
        candidate_limit: int,
        candidate_block_size: int,
        query_tile: int,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Run one span's selection from a graph; False leaves it to the caller.

        ``table_row`` is the request's raw ``[1, pages]`` block-table row; null
        entries are <= 0. ``positions`` are the span's query positions.
        ``candidate_limit`` is the source candidate capacity (0 when this pass
        sources no candidates); ``out[2]`` may be narrower, down to the blocks
        the caller's table can hold. The graph's pool is as wide as its page
        bucket holds, and every valid block fits in the narrower destination,
        so its leading columns are the eager pool.
        """
        if not index_q.is_cuda or torch.cuda.is_current_stream_capturing():
            return False
        n, pages = index_q.shape[0], table_row.shape[1]
        rows, width = _bucket(n, _MIN_ROWS), _bucket(pages, _MIN_PAGES)
        candidate_topk = min(candidate_limit, width * candidate_block_size)
        if out[2].shape[1] > candidate_topk:
            return False
        # Reindex candidate pools are as wide as their source's table allowed;
        # a wider pool padded with -1 holds the same blocks.
        pool = 0 if candidates is None else _bucket(candidates.shape[1], _MIN_POOL)
        shape = (
            rows,
            pool,
            None if candidates is None else candidates.dtype,
            candidate_topk,
            topk,
            index_q.dtype,
            weights.dtype,
            positions.dtype,
        )
        key = (
            shape,
            width,
            index_cache.data_ptr(),
            ratio,
            candidate_block_size,
            query_tile,
        )
        entry = self._graphs.get(key)
        if key in self._graphs and entry is None:
            return False
        statics = (
            entry.statics
            if entry is not None
            else self._statics_for(shape, index_q, weights, positions, candidates)
        )
        statics.q[:n].copy_(index_q)
        statics.weights[:n].copy_(weights)
        statics.positions.fill_(-1)
        statics.positions[:n].copy_(positions)
        if statics.candidates is not None:
            used = candidates.shape[1]
            statics.candidates[:n, :used].copy_(candidates)
            if used < pool:
                statics.candidates[:n, used:].fill_(-1)
        if entry is None:
            table = torch.zeros((1, width), dtype=table_row.dtype, device=self.device)
            table[:, :pages].copy_(table_row)

            def body():
                from tokenspeed_kernel.ops.attention import dsv41

                visible = (
                    ((statics.positions + 1) // ratio).clamp_min(0).to(torch.int32)
                )
                dsv41.index_topk(
                    statics.q,
                    statics.weights,
                    index_cache,
                    table.masked_fill(table <= 0, -1).expand(rows, -1),
                    visible,
                    statics.candidates,
                    topk,
                    candidate_topk,
                    candidate_block_size,
                    query_tile,
                    V41_INDEX_SCORE_CHUNK,
                    None,
                    statics.out,
                    None,
                )

            # The eager run serves this call and compiles every kernel the
            # capture records.
            body()
            entry = self._capture(key, body, statics, table)
        else:
            entry.table.zero_()
            entry.table[:, :pages].copy_(table_row)
            self._graphs.move_to_end(key)
            entry.graph.replay()
        for destination, source in zip(out, statics.out):
            destination.copy_(
                source[:n, : destination.shape[-1]] if source.ndim == 2 else source[:n]
            )
        return True

    def _statics_for(self, shape, index_q, weights, positions, candidates) -> _Statics:
        statics = self._statics.get(shape)
        if statics is not None:
            return statics
        rows, pool, _, candidate_topk, topk = shape[:5]

        def empty(*size, dtype=torch.int32):
            return torch.empty(size, dtype=dtype, device=self.device)

        statics = _Statics(
            q=torch.zeros(
                (rows, *index_q.shape[1:]), dtype=index_q.dtype, device=self.device
            ),
            weights=torch.zeros(
                (rows, *weights.shape[1:]), dtype=weights.dtype, device=self.device
            ),
            positions=empty(rows, dtype=positions.dtype),
            candidates=(
                torch.full(
                    (rows, pool),
                    -1,
                    dtype=candidates.dtype,
                    device=self.device,
                )
                if candidates is not None
                else None
            ),
            out=(
                empty(rows, topk),
                empty(rows),
                empty(rows, candidate_topk),
                empty(rows),
            ),
        )
        self._statics[shape] = statics
        return statics

    def _capture(self, key, body, statics: _Statics, table) -> _Graph | None:
        if len(self._graphs) >= self.max_graphs:
            self._graphs.popitem(last=False)
        if self._stream is None:
            self._stream = torch.cuda.Stream(self.device)
        graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream):
                graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    body()
                finally:
                    graph.capture_end()
        except RuntimeError:
            # An uncapturable configuration stays eager for good.
            self._graphs[key] = None
            current.wait_stream(self._stream)
            return None
        current.wait_stream(self._stream)
        if self._pool is None:
            self._pool = graph.pool()
        entry = _Graph(graph, statics, table)
        self._graphs[key] = entry
        return entry
