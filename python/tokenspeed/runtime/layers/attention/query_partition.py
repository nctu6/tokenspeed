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

"""Interleaved QCP query ownership, independent of cache storage and head TP."""

from dataclasses import dataclass

import torch

from tokenspeed.runtime.distributed.comm_backend import Group
from tokenspeed.runtime.distributed.comm_ops import all_gather_single


@dataclass(frozen=True)
class InterleavedQueryPlan:
    """Own packed query rows r, r + size, ... without changing positions."""

    total_rows: int
    rank: int
    size: int

    def __post_init__(self):
        if self.total_rows < 0 or self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError("Invalid interleaved query layout")

    @property
    def local_slice(self) -> slice:
        return slice(self.rank, self.total_rows, self.size)

    @property
    def local_rows(self) -> int:
        return max(0, (self.total_rows - self.rank + self.size - 1) // self.size)

    def local_bounds(self, start: int, stop: int) -> tuple[int, int]:
        """Map a global request span to its contiguous local query interval."""
        return (
            max(0, (start - self.rank + self.size - 1) // self.size),
            max(0, (stop - self.rank + self.size - 1) // self.size),
        )

    def gather_selection(
        self, rows: torch.Tensor, lengths: torch.Tensor, group: Group
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather logical IDs/lengths and restore packed query order.

        All ranks participate, including empty local shards. Physical KV
        slots are resolved by each consumer after this operation.
        """
        if len(group) != self.size or rows.shape[0] != self.local_rows:
            raise ValueError("Query shard and communication group disagree")
        if rows.dtype != torch.int32 or lengths.dtype != torch.int32:
            raise ValueError("Selection transport requires int32 IDs and lengths")
        if lengths.shape != (self.local_rows,):
            raise ValueError("Selection lengths disagree with the local query rows")
        if self.size == 1 or self.total_rows == 0:
            return rows, lengths
        capacity = (self.total_rows + self.size - 1) // self.size
        columns = rows.shape[1]
        # Keep each row 16-byte aligned for the runtime communication backends.
        wire_columns = ((columns + 1 + 3) // 4) * 4
        packed = torch.full(
            (capacity, wire_columns), -1, device=rows.device, dtype=torch.int32
        )
        packed[: self.local_rows, :columns].copy_(rows)
        packed[: self.local_rows, columns].copy_(lengths)
        gathered = torch.empty(
            (self.size * capacity, wire_columns),
            device=rows.device,
            dtype=torch.int32,
        )
        all_gather_single(gathered, packed, group, backend=None)
        ordered = (
            gathered.view(self.size, capacity, wire_columns)
            .transpose(0, 1)
            .reshape(-1, wire_columns)[: self.total_rows]
        )
        return ordered[:, :columns].contiguous(), ordered[:, columns].contiguous()


@dataclass(frozen=True)
class ContiguousQueryPlan:
    """Own one contiguous block of packed query rows per rank.

    Decode packs each request's verify rows together; a block that is a
    multiple of the verify width keeps a request's rows on one rank, so the
    indexer still shares their key loads. Rank r owns rows
    [r * block, min((r + 1) * block, total_rows)).
    """

    total_rows: int
    rank: int
    size: int
    block: int

    def __post_init__(self):
        if (
            self.total_rows < 0
            or self.size < 1
            or not 0 <= self.rank < self.size
            or self.block < 1
            or self.block * self.size < self.total_rows
        ):
            raise ValueError("Invalid contiguous query layout")

    @property
    def start(self) -> int:
        return min(self.rank * self.block, self.total_rows)

    @property
    def stop(self) -> int:
        return min((self.rank + 1) * self.block, self.total_rows)

    @property
    def local_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def local_rows(self) -> int:
        return self.stop - self.start

    def local_bounds(self, start: int, stop: int) -> tuple[int, int]:
        """Clip a global row interval to this rank's block, in local rows."""
        lo, hi = self.start, self.stop
        return (
            min(max(start, lo), hi) - lo,
            min(max(stop, lo), hi) - lo,
        )

    def gather_selection(
        self, rows: torch.Tensor, lengths: torch.Tensor, group: Group
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather logical IDs/lengths once; packed order is rank order."""
        return self.gather_rows((rows, lengths), group)

    def gather_rows(
        self, tensors: tuple[torch.Tensor, ...], group: Group
    ) -> tuple[torch.Tensor, ...]:
        """All-gather int32 per-row tensors ([local_rows] or [local_rows, C])
        in one collective, returning each at [total_rows, ...].

        Fixed shapes for a given total_rows, so decode graphs capture it.
        """
        if len(group) != self.size:
            raise ValueError("Query shard and communication group disagree")
        widths = []
        for t in tensors:
            if t.dtype != torch.int32 or t.shape[0] != self.local_rows or t.ndim > 2:
                raise ValueError("Row transport requires int32 [local_rows, ...]")
            widths.append(t.shape[1] if t.ndim == 2 else 1)
        if self.size == 1 or self.total_rows == 0:
            return tuple(tensors)
        # Keep each row 16-byte aligned for the runtime communication backends.
        wire_columns = ((sum(widths) + 3) // 4) * 4
        device = tensors[0].device
        packed = torch.empty(
            (self.block, wire_columns), device=device, dtype=torch.int32
        )
        column = 0
        for t, w in zip(tensors, widths):
            packed[: self.local_rows, column : column + w].copy_(t.view(-1, w))
            column += w
        if self.local_rows < self.block:
            packed[self.local_rows :].fill_(-1)
        gathered = torch.empty(
            (self.size * self.block, wire_columns), device=device, dtype=torch.int32
        )
        all_gather_single(gathered, packed, group, backend=None)
        ordered = gathered[: self.total_rows]
        out, column = [], 0
        for t, w in zip(tensors, widths):
            part = ordered[:, column : column + w].contiguous()
            out.append(part if t.ndim == 2 else part.view(-1))
            column += w
        return tuple(out)
