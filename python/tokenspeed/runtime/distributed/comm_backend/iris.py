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

"""Iris collectives with persistent, per-group symmetric states."""

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.amd_fabric import local_gpus_pcie_only
from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris.all_gather import IrisAllGather
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        ProducerAllReduce,
    )
    from tokenspeed_kernel.ops.communication._iris.broadcast import IrisBroadcast
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )
    from tokenspeed_kernel.ops.communication.iris import IrisAllReduce

_DEFAULT_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
_DEFAULT_ALL_REDUCE_MAX_BYTES = 512 * 1024
# On PCIe-only hosts Iris also beats RCCL on prefill activations (see
# amd_fabric), so both paths cover a whole chunked-prefill all-reduce:
# 16384 tokens x 7168 hidden in BF16 is 235 MB.
_PCIE_ALL_REDUCE_MAX_BYTES = 256 * 1024 * 1024
# Overlap lanes reserved when TOKENSPEED_PREFILL_TBO=1: one per micro-batch.
_OVERLAP_LANE_COUNT = 2


def _overlap_lane_numel(producer_direct_max_numel: int) -> int:
    """A lane carries one micro-batch: half the group's producer capacity."""
    return producer_direct_max_numel // 2


class IrisOverlapLane:
    """One micro-batch's producer-direct all-reduce, for comm/compute overlap.

    The producer writes its partial into ``acquire``'s view; ``reduce`` then
    sums it over the group into ``out`` on the current stream. The pull kernel
    is capped at ``max_programs`` workgroups: on PCIe-only MI350P, 16-32
    programs move as many bytes as 84 while leaving the CUs to the compute
    stream. Reusing a lane's buffer is safe once its previous ``reduce`` has
    completed locally: the two-shot reduction finishes only after every peer
    has read this rank's partition.
    """

    def __init__(self, state: "IrisAllReduce", max_programs: int):
        self.state = state
        self.max_programs = max_programs

    def acquire(self, shape: tuple[int, ...]) -> torch.Tensor:
        return self.state.acquire_outputs((shape,))[0]

    def can_reduce(self, shape: tuple[int, ...]) -> bool:
        return self.state.producer_pull.can_run((shape,))

    def reduce(self, out: torch.Tensor) -> None:
        import tokenspeed_kernel.ops.communication._iris.all_reduce as pull_impl

        # The pull kernels read their program cap at launch.
        saved = pull_impl._PULL_MAX_PROGRAMS
        pull_impl._PULL_MAX_PROGRAMS = min(saved, self.max_programs)
        try:
            self.state.producer_pull.run(out)
        finally:
            pull_impl._PULL_MAX_PROGRAMS = saved


class IrisBackend(CommBackend):
    """Keep direct handles to the Iris states prepared for each group/dtype."""

    def __init__(
        self,
        fallback: CommBackend,
        producer_direct_max_bytes: int | None = None,
    ):
        """Args:
        fallback: Backend for collectives Iris does not serve.
        producer_direct_max_bytes: Producer-direct all-reduce capacity, or
            None for the host default (larger on PCIe-only AMD hosts).
        """
        pcie_only = current_platform().is_amd and local_gpus_pcie_only()
        all_reduce_max_bytes = (
            _PCIE_ALL_REDUCE_MAX_BYTES if pcie_only else _DEFAULT_ALL_REDUCE_MAX_BYTES
        )
        if producer_direct_max_bytes is None:
            producer_direct_max_bytes = (
                _PCIE_ALL_REDUCE_MAX_BYTES
                if pcie_only
                else _DEFAULT_PRODUCER_DIRECT_MAX_BYTES
            )
        self._fallback = fallback
        self._instances: dict[tuple[Group, torch.dtype], IrisAllReduce] = {}
        self._gathers: dict[tuple[Group, torch.dtype], IrisAllGather] = {}
        self._broadcasts: dict[Group, IrisBroadcast] = {}
        self._ordinary_all_reduces: dict[Group, ProducerAllReduce] = {}
        self._pull_exchanges: dict[Group, IrisPullExchange] = {}
        self._producer_direct_max_bytes = producer_direct_max_bytes
        self._max_numel = (
            min(producer_direct_max_bytes, all_reduce_max_bytes)
            // torch.bfloat16.itemsize
        )
        # Overlap lanes (prefill two-micro-batch overlap) share the Iris heap,
        # which is sized when the first state is created: reserve them then.
        self._overlap_lane_reserve = (
            _OVERLAP_LANE_COUNT
            if os.environ.get("TOKENSPEED_PREFILL_TBO", "0") == "1"
            else 0
        )
        self._overlap_lanes: dict[
            tuple[Group, torch.dtype], tuple[IrisOverlapLane, ...] | None
        ] = {}

    @property
    def producer_direct_max_bytes(self) -> int:
        return self._producer_direct_max_bytes

    def state(self, group: Group, dtype: torch.dtype) -> "IrisAllReduce | None":
        """Return this group's prepared all-reduce state without allocating."""
        return self._instances.get((group, dtype))

    def install(self, group: Group, state: "IrisAllReduce") -> None:
        """Bind a state reserved alongside a model's fused operations."""
        previous = self.state(group, state.dtype)
        if previous is not None and previous is not state:
            raise RuntimeError("cannot replace a prepared Iris all-reduce state")
        self._instances[group, state.dtype] = state

    def install_all_gather(self, group: Group, state: "IrisAllGather") -> None:
        """Bind a prepared gather without replacing captured storage."""
        key = (group, state.producer.data.dtype)
        previous = self._gathers.get(key)
        if previous is not None and previous is not state:
            raise RuntimeError("cannot replace a prepared Iris all-gather")
        self._gathers[key] = state

    def install_broadcast(self, group: Group, state: "IrisBroadcast") -> None:
        """Bind a prepared broadcast without replacing captured mailboxes."""
        previous = self._broadcasts.get(group)
        if previous is not None and previous is not state:
            raise RuntimeError("cannot replace a prepared Iris broadcast")
        self._broadcasts[group] = state

    def install_ordinary_all_reduce(
        self, group: Group, reduction: "ProducerAllReduce"
    ) -> None:
        """Serve the group's ordinary BF16 all-reduces with a model's reduction.

        The reduction then reduces every caller-owned tensor it admits, before
        the generic staged path, sharing its mailboxes and producer buffer
        with the model's own uses in stream order.
        """
        previous = self._ordinary_all_reduces.get(group)
        if previous is not None and previous is not reduction:
            raise RuntimeError("cannot replace a prepared Iris reduction")
        self._ordinary_all_reduces[group] = reduction

    def _prepared_reduction(
        self, tensor: torch.Tensor, group: Group, op
    ) -> "ProducerAllReduce | None":
        reduction = self._ordinary_all_reduces.get(group)
        if (
            reduction is None
            or op not in (None, torch.distributed.ReduceOp.SUM)
            or not reduction.can_reduce_ordinary(tensor, rccl_fallback=True)
        ):
            return None
        return reduction

    def install_pull_exchange(self, group: Group, state: "IrisPullExchange") -> None:
        """Bind a prepared pull exchange without replacing captured storage."""
        previous = self._pull_exchanges.get(group)
        if previous is not None and previous is not state:
            raise RuntimeError("cannot replace a prepared Iris pull exchange")
        self._pull_exchanges[group] = state

    def acquire_exchange_input(
        self, nbytes: int, like: torch.Tensor, group: Group
    ) -> torch.Tensor:
        """Borrow the group's pull exchange input when it holds ``nbytes``."""
        state = self._pull_exchanges.get(group)
        borrowed = None if state is None else state.input(nbytes)
        if borrowed is None:
            return super().acquire_exchange_input(nbytes, like, group)
        return borrowed

    def try_combine_records(
        self,
        records: torch.Tensor,
        group: Group,
        *,
        head_dim: int,
        partial_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Pull and combine records borrowed from the group's pull exchange."""
        state = self._pull_exchanges.get(group)
        if state is None or not state.can_combine_records(
            records,
            head_dim=head_dim,
            partial_dtype=partial_dtype,
            output_dtype=output_dtype,
        ):
            return None
        return state.combine_records(records, head_dim=head_dim)

    def get_prepared_all_gather(
        self, group: Group, dtype: torch.dtype
    ) -> "IrisAllGather | None":
        return self._gathers.get((group, dtype))

    def broadcast_from_first_rank(self, tensor: torch.Tensor, group: Group) -> bool:
        state = self._broadcasts.get(group)
        if state is None or not state.can_run(tensor):
            return False
        state.run(tensor)
        return True

    def _create(
        self,
        group: Group,
        dtype: torch.dtype,
        staged_max_numel: int,
        producer_direct_max_numel: int,
    ) -> "IrisAllReduce":
        from tokenspeed_kernel.ops.communication._iris import (
            HEAP_HEADROOM_BYTES,
            MIN_HEAP_BYTES,
        )
        from tokenspeed_kernel.ops.communication.iris import (
            IrisAllReduce,
            create_iris_state,
        )

        heap_size = None
        if self._overlap_lane_reserve:
            lane_bytes = IrisAllReduce.heap_bytes(
                0, _overlap_lane_numel(producer_direct_max_numel), len(group), dtype
            )
            heap_size = max(
                MIN_HEAP_BYTES,
                IrisAllReduce.heap_bytes(
                    staged_max_numel, producer_direct_max_numel, len(group), dtype
                )
                + self._overlap_lane_reserve * lane_bytes
                + HEAP_HEADROOM_BYTES,
            )
        state = create_iris_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            staged_max_numel=staged_max_numel,
            producer_direct_max_numel=producer_direct_max_numel,
            dtype=dtype,
            heap_size=heap_size,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        self.install(group, state)
        return state

    def overlap_all_reduce_lanes(
        self, group: Group, dtype: torch.dtype, count: int, max_programs: int
    ) -> "tuple[IrisOverlapLane, ...] | None":
        """Lanes reserved in the shared heap (see CommBackend); None if absent."""
        key = (group, dtype)
        if key in self._overlap_lanes:
            lanes = self._overlap_lanes[key]
            return None if lanes is None or len(lanes) < count else lanes[:count]
        if (
            len(group) <= 1
            or not current_platform().is_cdna4
            or dtype != torch.bfloat16
            or count > self._overlap_lane_reserve
        ):
            return None
        from tokenspeed_kernel.ops.communication.iris import create_iris_state

        main = self._get_or_create(group, dtype)
        lanes = tuple(
            IrisOverlapLane(
                create_iris_state(
                    group=pg_manager.get_process_group("nccl", group),
                    rank_in_group=group.index(dist.get_rank()),
                    staged_max_numel=0,
                    producer_direct_max_numel=_overlap_lane_numel(
                        main.producer_direct_max_numel
                    ),
                    dtype=dtype,
                    heap_size=None,
                    device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                ),
                max_programs,
            )
            for _ in range(self._overlap_lane_reserve)
        )
        self._overlap_lanes[key] = lanes
        return lanes[:count]

    def _get_or_create(self, group: Group, dtype: torch.dtype) -> "IrisAllReduce":
        state = self.state(group, dtype)
        if state is None:
            state = self._create(
                group,
                dtype,
                self._max_numel,
                self._producer_direct_max_bytes // dtype.itemsize,
            )
        return state

    def prepare_all_reduce_buffers(
        self,
        group: Group,
        *,
        staged_max_numel: int,
        producer_direct_max_numel: int,
        dtype: torch.dtype,
    ) -> bool:
        """Allocate fixed capacities before cache planning and graph capture."""
        if len(group) <= 1 or not current_platform().is_amd:
            return False
        if dtype != torch.bfloat16:
            return False
        staged_max_numel = min(staged_max_numel, self._max_numel)
        requested = (staged_max_numel, producer_direct_max_numel)
        if min(requested) < 0 or not any(requested):
            raise ValueError(f"invalid all-reduce buffer capacities: {requested}")
        state = self.state(group, dtype)
        if state is None:
            self._create(group, dtype, *requested)
        else:
            available = (state.staged_max_numel, state.producer_direct_max_numel)
            if any(have < need for have, need in zip(available, requested)):
                raise RuntimeError(
                    "all-reduce buffers were initialized below the requested "
                    f"capacities: available={available}, requested={requested}"
                )
        return True

    def can_run(self, tensor: torch.Tensor, group: Group, op=None) -> bool:
        """Check ordinary eligibility without allocating collective state."""
        if len(group) <= 1 or not current_platform().is_amd:
            return False
        if self._prepared_reduction(tensor, group, op) is not None:
            return True
        state = self.state(group, tensor.dtype)
        max_numel = self._max_numel
        if state is not None:
            max_numel = min(max_numel, state.staged_max_numel)
        return (
            op in (None, torch.distributed.ReduceOp.SUM)
            and tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.dtype == torch.bfloat16
            and 0 < tensor.numel() <= max_numel
        )

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            if self.can_reduce_outputs(tensor, group, op=op):
                from tokenspeed_kernel.ops.communication.iris import (
                    iris_all_reduce_symmetric,
                )

                return iris_all_reduce_symmetric(
                    self.state(group, tensor[0].dtype), tensor
                )
            return super().all_reduce(tensor, group, op=op)
        reduction = self._prepared_reduction(tensor, group, op)
        if reduction is not None:
            return reduction.reduce_ordinary(tensor)
        if self.can_run(tensor, group, op=op):
            from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

            return iris_all_reduce(
                self._get_or_create(group, tensor.dtype), tensor, op=op, safe=False
            )
        return self._fallback.all_reduce(tensor, group, op=op)

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Borrow consecutive symmetric views for a producer and its reduction."""
        if not self.can_acquire_outputs(shapes, like, group, op=op):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)
        from tokenspeed_kernel.ops.communication.iris import iris_acquire_outputs

        return iris_acquire_outputs(self._get_or_create(group, like.dtype), shapes)

    def can_acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        return self.can_acquire_outputs(shapes, like, group, op=op)

    def can_acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Check producer eligibility without initializing Iris or its heap."""
        if not current_platform().is_cdna4 or not like.is_cuda:
            return False
        numels = tuple(math.prod(shape) for shape in shapes)
        if (
            op not in (None, torch.distributed.ReduceOp.SUM)
            or not numels
            or any(numel <= 0 for numel in numels)
        ):
            return False
        state = self.state(group, like.dtype)
        max_bytes = (
            state.producer_direct_max_numel * state.dtype.itemsize
            if state is not None
            else self._producer_direct_max_bytes
        )
        if sum(numels) * like.dtype.itemsize > max_bytes:
            return False
        from tokenspeed_kernel.ops.communication.iris import (
            producer_direct_all_reduce_can_run,
        )

        return producer_direct_all_reduce_can_run(
            world_size=len(group),
            total_numel=sum(numels),
            dtype=like.dtype,
            max_bytes=max_bytes,
        )

    def all_reduce_mhc_post(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        group: Group,
    ) -> torch.Tensor | None:
        """Fuse CDNA4 all-reduce and mHC post when prepared Iris state permits.

        Return None on other platforms, without state, or for ineligible operands.
        """
        if not current_platform().is_cdna4:
            return None
        state = self.state(group, tensor.dtype)
        if state is None or not state.can_reduce_mhc_post(tensor, residual, post, comb):
            return None
        return state.all_reduce_symmetric_mhc_post(tensor, residual, post, comb)

    def can_reduce_outputs(
        self,
        tensors: tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> bool:
        """Recognize producer views by ownership, without searching states."""
        if (
            not current_platform().is_cdna4
            or op not in (None, torch.distributed.ReduceOp.SUM)
            or not tensors
        ):
            return False
        state = self.state(group, tensors[0].dtype)
        return state is not None and state.owns_outputs(tensors)

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if (
            current_platform().is_cdna4
            and len(group) in (4, 8)
            and tensor.ndim == 2
            and dim in (-1, tensor.ndim - 1)
            and tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.is_contiguous()
        ):
            gather = self.get_prepared_all_gather(group, tensor.dtype)
            if gather is None:
                state = self.state(group, tensor.dtype)
                max_bytes = (
                    state.producer_direct_max_numel * tensor.element_size()
                    if state is not None
                    else self._producer_direct_max_bytes
                )
                from tokenspeed_kernel.ops.communication._iris.all_gather import (
                    IrisAllGather,
                    all_gather_columns_can_run,
                )

                if all_gather_columns_can_run(
                    len(group), 1, tensor.shape[1], tensor.dtype, max_bytes
                ):
                    state = self._get_or_create(group, tensor.dtype)
                    gather = IrisAllGather(state.producer)
                    self.install_all_gather(group, gather)
                    state.peers.finish_prepare()
            if gather is not None and gather.can_run(tensor):
                return gather.run(tensor)
        return self._fallback.all_gather(tensor, group, dim)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        # An input borrowed from the group's pull exchange is pulled in place.
        state = self._pull_exchanges.get(group)
        if state is not None and state.can_all_gather(output, input):
            return state.all_gather(output, input)
        return self._fallback.all_gather_single(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._fallback.reduce_scatter(tensor, group)

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: Group,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> None:
        state = self._pull_exchanges.get(group)
        if (
            output_split_sizes is None
            and input_split_sizes is None
            and state is not None
            and state.can_all_to_all(output, input)
        ):
            return state.all_to_all(output, input)
        return self._fallback.all_to_all_single(
            output,
            input,
            group,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")
