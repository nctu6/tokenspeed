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
"""Copy-engine all-reduce tier for two PCIe-connected ranks.

Large plain SUM all-reduces of a two-rank group whose GPUs talk over PCIe
move their data on the copy engines (``tokenspeed_kernel.ops.communication.
copy_engine``), which reach about 1.75x NCCL's bandwidth there. A prefill
chunk's per-layer all-reduces are the payloads this serves; payloads up to
the one-shot window keep the latency-bound tiers.
"""

import torch
from tokenspeed_kernel.ops.communication.copy_engine import (
    SUPPORTED_DTYPES,
    CopyEngineAllReduceState,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

# Bytes one copy-engine chunk reduces at most: the copies reach their full
# rate from about 12 MB (100 MB runs at the 50 MB rate), so larger tensors
# run in chunks of this size instead of growing the symmetric buffers.
COPY_ENGINE_MAX_CHUNK_BYTES = 64 << 20


def copy_engine_all_reduce_applies(group: Group) -> bool:
    """Whether ``group`` is two ranks of one node joined only by PCIe.

    Measured on two RTX PRO 6000s over PCIe (BF16, 3-100 MB: 0.70-0.58x
    NCCL's time). NVLink groups keep NCCL and the trtllm kernels, whose
    bandwidth the copy engines do not beat there; other group sizes need a
    different split and are not implemented.
    """
    platform = current_platform()
    return (
        len(group) == 2
        and platform.is_nvidia
        and platform.interconnect.topology == "pcie"
    )


class CopyEngineAllReduceBackend:
    """Per-group copy-engine all-reduce states, armed before serving."""

    def __init__(self, min_bytes: int):
        # Payloads at or below this many bytes keep the latency-bound tiers.
        self._min_bytes = min_bytes
        self._states: dict[Group, CopyEngineAllReduceState] = {}

    def configure_group(self, group: Group, max_bytes: int) -> None:
        """Arm ``group`` (collective over its ranks, before graph capture)
        with chunks of ``max_bytes``."""
        if group in self._states:
            return
        self._states[group] = CopyEngineAllReduceState(
            group=pg_manager.get_process_group("nccl", group),
            max_bytes=max_bytes,
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    def has_group(self, group: Group) -> bool:
        return group in self._states

    def serves(self, tensor: torch.Tensor, group: Group, op) -> bool:
        """Whether this tier reduces ``tensor``: an armed group, a SUM, a
        supported dtype, contiguous, and past the one-shot window."""
        return (
            group in self._states
            and (op is None or op == torch.distributed.ReduceOp.SUM)
            and tensor.dtype in SUPPORTED_DTYPES
            and tensor.is_contiguous()
            and tensor.numel() * tensor.element_size() > self._min_bytes
        )

    def all_reduce(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._states[group].all_reduce_(tensor)
