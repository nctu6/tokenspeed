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

"""One Host L2 buffer shared by an attention-TP group (two real processes).

The CPU test maps the segment in two gloo ranks; the GPU test runs two
``HostCacheExecutor`` ranks on one device over one shared buffer: the
follower's write-back copies nothing, the owner's bytes are what both ranks
load, the follower makes no L3 transfer, and the scheduler sees the group's
pages.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

torch = pytest.importorskip("torch")

_WORLD = 2


def _init_gloo(rank: int, init_file: str):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD
    )
    return dist.group.WORLD


def _run_rank(target, rank: int, init_file: str, results) -> None:
    try:
        results.put((rank, target(rank, init_file)))
    except BaseException:  # reported to the parent, which fails the test
        results.put((rank, traceback.format_exc()))


def _spawn(target) -> dict:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "gloo-init")
        procs = [
            ctx.Process(target=_run_rank, args=(target, rank, init_file, results))
            for rank in range(_WORLD)
        ]
        for proc in procs:
            proc.start()
        outcome = dict(results.get(timeout=240) for _ in procs)
        for proc in procs:
            proc.join(timeout=60)
    for rank, value in outcome.items():
        if isinstance(value, str):
            pytest.fail(f"rank {rank} failed:\n{value}")
    return outcome


# ----------------------------------------------------------------------
# Segment mapping (CPU)
# ----------------------------------------------------------------------


def _segment_rank(rank: int, init_file: str) -> dict:
    import torch.distributed as dist

    from tokenspeed.runtime.cache.l2.shared import allocate_shared_host_buffer

    group = _init_gloo(rank, init_file)
    buffer = allocate_shared_host_buffer(
        1 << 20, rank=rank, size=_WORLD, cpu_group=group, register=False
    )
    unlinked = not os.path.exists(buffer.path)
    if rank == 0:
        buffer.tensor[123:456].fill_(0x5A)
    dist.barrier(group=group)
    seen = buffer.tensor[123:456].eq(0x5A).all().item()
    untouched = not buffer.tensor[:123].any().item()
    dist.barrier(group=group)
    dist.destroy_process_group()
    return {
        "owner": buffer.owner,
        "nbytes": buffer.nbytes,
        "seen": bool(seen),
        "untouched": bool(untouched),
        "unlinked": unlinked,
    }


def test_segment_is_one_mapping_for_the_group_and_unlinked():
    outcome = _spawn(_segment_rank)
    assert [outcome[rank]["owner"] for rank in range(_WORLD)] == [True, False]
    for rank in range(_WORLD):
        assert outcome[rank]["nbytes"] == 1 << 20
        # The follower reads the owner's write: one mapping, not two copies.
        assert outcome[rank]["seen"]
        assert outcome[rank]["untouched"]
        # The name is gone once every rank mapped it; the memory dies with
        # the last process.
        assert outcome[rank]["unlinked"]


# ----------------------------------------------------------------------
# Two executor ranks over one buffer (GPU)
# ----------------------------------------------------------------------


def _layout(torch_module):
    from tokenspeed.runtime.cache.transfer.layout import (
        CacheField,
        CacheGroupLayout,
        CacheTransferLayout,
    )

    first = torch_module.zeros((128,), dtype=torch_module.uint8, device="cuda")
    second = torch_module.zeros((128,), dtype=torch_module.uint8, device="cuda")
    return CacheTransferLayout(
        num_lcm_blocks=4,
        groups=(
            CacheGroupLayout(
                group_id="full",
                cache_blocks_per_lcm_block=2,
                fields=(
                    CacheField("layer.0.k", 0, 8, 8, 4),
                    CacheField("layer.0.v", 1, 16, 12, 6),
                ),
            ),
        ),
        buffers=(first, second),
        consumers=(("layer.0.k", "layer.0.v"),),
    )


class _Pool:
    def __init__(self, layout):
        self.layout = layout
        self.load_tracker = None
        specs = tuple(
            SimpleNamespace(group_id=group.group_id, shard_count=1)
            for group in layout.groups
        )
        self.arena = SimpleNamespace(
            cache_group_specs=specs,
            runtime_contract=SimpleNamespace(
                group_specs=specs,
                virtual_block_counts={
                    group.group_id: 1
                    + layout.num_lcm_blocks * group.cache_blocks_per_lcm_block
                    for group in layout.groups
                },
            ),
        )

    def cache_transfer_layout(self):
        return self.layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


class _WriteBackOp:
    """The scheduler's L2 write-back wire op: one pinned op of two blocks."""

    def __init__(self):
        self.op_ids = [7]
        self.group_ids = [[0, 0]]
        self.src_pages = [[1, 4]]
        self.dst_pages = [[3, 6]]
        self.source_pinned = [True]
        self.content_hashes = [["h0", "h1"]]
        self.page_offsets = [[0, 1]]


def _poll(executor, *, want: int, deadline_s: float = 30.0) -> list:
    results: list = []
    deadline = time.monotonic() + deadline_s
    while len(results) < want and time.monotonic() < deadline:
        results.extend(executor.poll_results())
        time.sleep(0.001)
    return results


def _executor_rank(rank: int, init_file: str) -> dict:
    import torch
    import torch.distributed as dist

    import tokenspeed.runtime.cache.l2.executor as executor_module
    from tokenspeed.runtime.cache.l2.shared import SharedL2Group
    from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
    from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
    from tokenspeed.runtime.cache.transfer.ops import PrefetchOp, PrefetchRow

    torch.cuda.set_device(0)
    group = _init_gloo(rank, init_file)
    layout = _layout(torch)
    pool = _Pool(layout)
    no_pool = RetractionPoolRequest(
        host_gb=0.0, ratio=0.0, max_retracted_requests=0, tail_lcm_blocks_per_request=0
    )
    with patch.object(executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
        executor = executor_module.HostCacheExecutor(
            pool,
            l2_tier=True,
            host_ratio=1.0,
            host_size_gb=0,
            snapshot_pool=no_pool,
            slot_state_exporters=None,
            io_backend="direct",
            attn_tp_rank=rank,
            kvp_rank=0,
            nprocs_per_node=_WORLD,
            world_cpu_group=group,
            shared_l2=SharedL2Group(rank=rank, size=_WORLD, cpu_group=group),
        )
    store = MemoryKvStore()
    executor.attach_l3_storage(
        store,
        key_prefix="shared",
        rank=0,
        prefix_for_weight_version=lambda version: f"shared-{version}",
        prefetch_timeout_base_s=10.0,
        prefetch_timeout_per_page_s=0.0,
        prefetch_batch_pages=128,
    )
    first, second = layout.buffers
    # Each rank's Device bytes differ, so the Host bytes name their writer.
    pattern = 0x30 + rank
    first.fill_(pattern)
    second.fill_(pattern)
    torch.cuda.synchronize()

    with patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True):
        executor.submit_write_backs(
            [_WriteBackOp()],
            prerequisite_stream=torch.cuda.current_stream(),
            fence_stream=torch.cuda.current_stream(),
        )
    write_ids = [int(event.op_id) for event in _poll(executor, want=1)]
    # The hooks' completion intersection: no rank reads before every rank
    # (the owner included) acknowledged the write.
    dist.barrier(group=group)
    host = executor.host_storage.host_buffer
    written = [
        host[offset : offset + size].clone()
        for offset, size in (
            executor.host_storage.host_block_range(0, 3),
            executor.host_storage.host_block_range(0, 6),
        )
    ]
    l3_objects = len(store._objects)  # pylint: disable=protected-access
    exists = executor.l3_exists([(0, 3, "h0", 0), (0, 3, "absent", 0)])

    # Every rank loads the shared blocks into its zeroed Device pages.
    first.zero_()
    second.zero_()
    torch.cuda.synchronize()
    load_index = executor._start_loading(  # pylint: disable=protected-access
        [9],
        [(0, 2, 3), (0, 5, 6)],
        prerequisite_stream=torch.cuda.current_stream(),
    )
    pool.load_tracker.set_consumers(load_index)
    pool.load_tracker.wait_for_layer(0)
    torch.cuda.synchronize()
    load_ids = [int(event.op_id) for event in _poll(executor, want=1)]
    loaded = (
        # Device blocks 2 and 5 (field k at 8 + 8 * block, field v at
        # 16 + 12 * block).
        first[24:28].cpu().unique().tolist(),
        second[40:46].cpu().unique().tolist(),
        first[48:52].cpu().unique().tolist(),
    )

    # An L3 prefetch: the follower fetches nothing and lands every page.
    executor.submit_prefetches(
        [
            PrefetchOp(
                op_id=21,
                request_id="r",
                first_page=0,
                num_pages=1,
                rows=(PrefetchRow(0, 8, "missing", 0, 0),),
            )
        ]
    )
    deadline = time.monotonic() + 30
    progress = executor.l3_prefetch_progress()
    while not progress[21][0] and time.monotonic() < deadline:
        time.sleep(0.001)
        progress = executor.l3_prefetch_progress()
    executor.complete_l3_prefetch(21, progress[21][1])
    executor.poll_results()
    dist.barrier(group=group)
    executor.shutdown()
    dist.destroy_process_group()
    return {
        "host_pages": executor.num_host_pages,
        "follower": executor.l2_follower,
        "write_ids": write_ids,
        "written": [block.unique().tolist() for block in written],
        "l3_objects": l3_objects,
        "exists": exists,
        "load_ids": load_ids,
        "loaded": loaded,
        "prefetch_landed": progress[21][1],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_owner_writes_and_every_rank_loads_one_shared_buffer():
    outcome = _spawn(_executor_rank)
    owner, follower = outcome[0], outcome[1]
    # host_ratio 1 of 4 Device LCM blocks per rank: the group's L2 holds 8
    # (plus the null block), the same count on both mirrored schedulers.
    assert owner["host_pages"] == follower["host_pages"] == 1 + 4 * _WORLD
    assert (owner["follower"], follower["follower"]) == (False, True)
    assert owner["write_ids"] == follower["write_ids"] == [7]
    # Only the owner copied: the shared Host blocks hold its bytes.
    for rank in range(_WORLD):
        assert outcome[rank]["written"] == [[0x30], [0x30]]
        assert outcome[rank]["load_ids"] == [9]
        assert outcome[rank]["loaded"] == ([0x30], [0x30], [0x30])
    # Only the owner backs up to L3; the follower defers its probes to it.
    assert owner["l3_objects"] == 2
    assert follower["l3_objects"] == 0
    assert owner["exists"] == [True, False]
    assert follower["exists"] == [True, True]
    # A key L3 lacks: the owner lands nothing, the follower trivially all,
    # so the replica MIN is the owner's 0.
    assert owner["prefetch_landed"] == 0
    assert follower["prefetch_landed"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
