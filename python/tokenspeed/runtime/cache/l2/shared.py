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

"""One Host L2 buffer shared by the ranks of an attention-TP group.

When a cache recipe declares ``kv_identical_across_attn_tp`` (every rank of
the attention-TP group holds the same cache bytes), the per-rank L2 buffers
hold identical copies. ``--kvstore-share-host-across-attn-tp`` replaces them
with one POSIX shared-memory segment per group, ``tp_size`` times one rank's
size, so the node pins the same Host memory and the L2 tier holds
``tp_size`` times the pages:

* attention-TP rank 0 (the owner) creates the segment in ``/dev/shm`` and
  reserves it; every rank maps it and registers the mapping with the device
  runtime (``hipHostRegister`` / ``cudaHostRegister``, mapped), then the owner
  unlinks the name, so the memory goes away with the last process;
* the mirrored schedulers address the shared pages with the same block ids,
  so only the owner copies Device to Host (and moves L3 bytes in and out of
  them); every rank loads Host to Device from the segment. The completion
  intersection of ``CacheOpHooks`` advances an op only once every rank,
  the owner included, has acknowledged it, which orders every load after
  the owner's write and every reuse of a page after every rank's load.
"""

from __future__ import annotations

import mmap
import os
import socket
import uuid
from dataclasses import dataclass

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_SHM_DIR = "/dev/shm"
# cudaHostRegisterPortable | cudaHostRegisterMapped (same values in HIP).
_HOST_REGISTER_FLAGS = 0x1 | 0x2


def _gather(payload, cpu_group, size: int) -> list:
    if cpu_group is None or size == 1:
        return [payload]
    gathered = [None] * size
    torch.distributed.all_gather_object(gathered, payload, group=cpu_group)
    return gathered


def _barrier(cpu_group, size: int) -> None:
    if cpu_group is not None and size > 1:
        torch.distributed.barrier(group=cpu_group)


def same_host(cpu_group, size: int) -> bool:
    """Whether every rank of ``cpu_group`` runs on this node; a collective."""
    hosts = _gather(socket.gethostname(), cpu_group, size)
    return len(set(hosts)) == 1


def _register(ptr: int, nbytes: int) -> None:
    status = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, _HOST_REGISTER_FLAGS)
    code = int(getattr(status, "value", status))
    if code != 0:
        raise RuntimeError(
            f"host register of the shared L2 segment failed ({status!s}, "
            f"{nbytes} bytes)"
        )


def _unregister(ptr: int) -> None:
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # pragma: no cover - teardown best effort
        pass


class SharedHostBuffer:
    """A mapped, device-registered shared-memory segment as a uint8 tensor.

    Attributes:
        tensor: The ``[nbytes]`` uint8 CPU tensor over the mapping; pinned
            from the device runtime's point of view once registered.
        nbytes: Size of the segment.
        owner: Whether this rank created the segment (attention-TP rank 0).
    """

    def __init__(self, path: str, nbytes: int, *, owner: bool, register: bool):
        self.nbytes = int(nbytes)
        self.owner = owner
        self.path = path
        fd = os.open(path, os.O_RDWR)
        try:
            self._mmap = mmap.mmap(fd, self.nbytes, flags=mmap.MAP_SHARED)
        finally:
            os.close(fd)
        self.tensor = torch.frombuffer(self._mmap, dtype=torch.uint8)
        self._registered = False
        if register:
            _register(self.tensor.data_ptr(), self.nbytes)
            self._registered = True

    def close(self) -> None:
        if self._registered:
            _unregister(self.tensor.data_ptr())
            self._registered = False

    @staticmethod
    def create_segment(nbytes: int) -> str:
        """Create and reserve one segment; return its path."""
        path = os.path.join(_SHM_DIR, f"tokenspeed-l2-{os.getpid()}-{uuid.uuid4().hex}")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # Reserve every page now: a tmpfs that cannot hold the segment
            # fails here with ENOSPC instead of a SIGBUS on first touch.
            os.posix_fallocate(fd, 0, int(nbytes))
        except BaseException:
            os.close(fd)
            os.unlink(path)
            raise
        os.close(fd)
        return path


def allocate_shared_host_buffer(
    nbytes: int,
    *,
    rank: int,
    size: int,
    cpu_group,
    register: bool = True,
) -> SharedHostBuffer:
    """One segment for the attention-TP group; every rank must call it.

    Rank 0 creates the segment and shares its path; every rank maps and
    registers it; after a barrier rank 0 unlinks the name. A rank that
    fails to map raises after the barrier ranks entered, so a failure does
    not leave peers blocked (they raise from the gathered status).

    Args:
        nbytes: Segment size, the group's whole L2 tier.
        rank: This rank in the attention-TP group.
        size: The group size.
        cpu_group: The group's gloo process group (None with one rank).
        register: Register the mapping with the device runtime (tests on
            CPU pass False).
    """
    path = None
    error = None
    if rank == 0:
        try:
            path = SharedHostBuffer.create_segment(nbytes)
        except Exception as exc:  # shared below so every rank raises
            error = repr(exc)
    path, error = _gather((path, error), cpu_group, size)[0]
    if error is not None:
        raise RuntimeError(f"creating the shared Host L2 segment failed: {error}")
    buffer = None
    local_error = None
    try:
        buffer = SharedHostBuffer(path, nbytes, owner=rank == 0, register=register)
    except Exception as exc:
        local_error = repr(exc)
    errors = _gather(local_error, cpu_group, size)
    if rank == 0:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    # No rank returns before the name is gone.
    _barrier(cpu_group, size)
    failed = [(index, err) for index, err in enumerate(errors) if err is not None]
    if failed:
        if buffer is not None:
            buffer.close()
        raise RuntimeError(f"mapping the shared Host L2 segment failed: {failed}")
    return buffer


@dataclass(frozen=True)
class SharedL2Group:
    """The attention-TP group sharing one L2 buffer.

    Attributes:
        rank: This rank in the group; rank 0 owns the writes.
        size: The group size.
        cpu_group: The group's gloo process group.
    """

    rank: int
    size: int
    cpu_group: object
