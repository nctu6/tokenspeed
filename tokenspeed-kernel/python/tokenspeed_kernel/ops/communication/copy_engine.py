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
"""Two-rank all-reduce whose data moves on the copy engines.

Each rank owns half of the tensor. A rank copies the peer's half into the
peer's staging buffer, waits at a barrier, adds its own half to the staged
half the peer sent, copies that sum into the peer's result buffer, waits at
a second barrier, and copies the peer's sum back into its tensor. Every
transfer is a device-to-device ``cudaMemcpyAsync`` into a peer-mapped
symmetric buffer, which the copy engines carry; only the add and the two
barriers run on SMs.

A sum of two values rounds once, so the result is bitwise equal to NCCL's
all-reduce of the same two tensors. Over PCIe the copy engines move a
direction at about 49 GB/s while the other direction runs, against about
29 GB/s of NCCL algorithm bandwidth: 0.98 against 1.72 ms for 50 MB of
BF16 on two RTX PRO 6000s (see README.md).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Each rank's half starts on a 16-byte boundary.
_ALIGN_BYTES = 16
# Sums of two values in these types round once (an FP32 add of two BF16 or
# FP16 values is exact before the final rounding), as NCCL's do.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class CopyEngineAllReduceState:
    """Symmetric staging and result buffers of one two-rank group.

    Creation rendezvouses the group: every rank must build it, in the same
    order, outside CUDA-graph capture. Calls and their consumers run on one
    stream per rank; the barriers order a call's buffer reuse against the
    peer's previous call.
    """

    def __init__(
        self, group: dist.ProcessGroup, max_bytes: int, device: torch.device
    ) -> None:
        """Allocate the buffers for chunks of up to ``max_bytes`` bytes.

        Args:
            group: The two-rank process group.
            max_bytes: Bytes one chunk reduces; longer tensors run in chunks
                of this size.
            device: This rank's device.
        """
        if group.size() != 2:
            raise ValueError(
                f"the copy-engine all-reduce serves two ranks, not {group.size()}"
            )
        if max_bytes < 2 * _ALIGN_BYTES:
            raise ValueError(f"max_bytes {max_bytes} below two aligned halves")
        self.rank = group.rank()
        self.max_bytes = max_bytes // _ALIGN_BYTES * _ALIGN_BYTES
        # The larger half of a chunk: m - floor_align(m / 2) <= m / 2 + align.
        half = self.max_bytes // 2 + _ALIGN_BYTES
        with torch.inference_mode(False), torch.no_grad():
            self._buffer = symm_mem.empty((2, half), dtype=torch.uint8, device=device)
        self._handle = symm_mem.rendezvous(self._buffer, group=group)
        peer = self._handle.get_buffer(1 - self.rank, (2, half), torch.uint8)
        self._stage, self._result = self._buffer[0], self._buffer[1]
        self._peer_stage, self._peer_result = peer[0], peer[1]

    def all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum ``tensor`` over both ranks in place and return it.

        Args:
            tensor: Contiguous tensor of a ``SUPPORTED_DTYPES`` type, same
                shape and dtype on both ranks.

        Returns:
            ``tensor``, holding the sum.
        """
        if tensor.dtype not in SUPPORTED_DTYPES or not tensor.is_contiguous():
            raise ValueError(
                "copy-engine all-reduce takes a contiguous tensor of "
                f"{SUPPORTED_DTYPES}, got {tensor.dtype} "
                f"contiguous={tensor.is_contiguous()}"
            )
        dtype = tensor.dtype
        align = _ALIGN_BYTES // tensor.element_size()
        chunk_numel = self.max_bytes // tensor.element_size()
        stage, result = self._stage.view(dtype), self._result.view(dtype)
        peer_stage = self._peer_stage.view(dtype)
        peer_result = self._peer_result.view(dtype)
        flat = tensor.view(-1)
        total = flat.numel()
        for start in range(0, total, chunk_numel):
            chunk = flat[start : min(total, start + chunk_numel)]
            # Rank 0 owns [0, split), rank 1 owns [split, numel).
            split = chunk.numel() // 2 // align * align
            own = chunk[:split] if self.rank == 0 else chunk[split:]
            other = chunk[split:] if self.rank == 0 else chunk[:split]
            peer_stage[: other.numel()].copy_(other)
            self._handle.barrier(channel=0)
            own.add_(stage[: own.numel()])
            peer_result[: own.numel()].copy_(own)
            self._handle.barrier(channel=0)
            other.copy_(result[: other.numel()])
        return tensor
