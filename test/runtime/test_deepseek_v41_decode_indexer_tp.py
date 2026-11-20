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

"""Decode indexer row sharding (TOKENSPEED_DSV41_DECODE_INDEXER_TP) selects
exactly the rows of the replicated indexer. Ranks are simulated in one process:
every rank's local block is scored, then the all-gather is fed the blocks the
other ranks produced."""

import os
import sys
from dataclasses import replace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention import query_partition
from tokenspeed.runtime.layers.attention.backends.specific import (
    deepseek_v41 as backend_module,
)
from tokenspeed.runtime.layers.attention.query_partition import ContiguousQueryPlan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

TP = 4


@pytest.mark.parametrize("total,block", [(0, 6), (6, 6), (96, 24), (90, 24)])
def test_contiguous_plan_covers_rows_once(total, block):
    owned = []
    for rank in range(TP):
        plan = ContiguousQueryPlan(total, rank, TP, block)
        rows = list(range(total))[plan.local_slice]
        assert len(rows) == plan.local_rows
        owned += rows
        lo, hi = plan.local_bounds(0, total)
        assert (lo, hi) == (0, plan.local_rows)
    assert owned == list(range(total))
    with pytest.raises(ValueError):
        ContiguousQueryPlan(total + TP * block + 1, 0, TP, block)


def _fake_world(stash):
    """all_gather_single that rebuilds the world from per-rank blocks."""
    state = {"rank": 0, "call": 0}

    def gather(output, value, group, backend):
        del backend
        assert len(group) == TP and output.shape[0] == TP * value.shape[0]
        key = state["call"]
        state["call"] += 1
        stash.setdefault(key, {})[state["rank"]] = value.clone()
        block = value.shape[0]
        for rank in range(TP):
            part = stash[key].get(rank)
            out = output[rank * block : (rank + 1) * block]
            if part is None:
                out.fill_(-7)
            else:
                out.copy_(part)

    return state, gather


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("deferred", [False, True])
def test_sharded_decode_topk_is_bit_identical(deferred, monkeypatch):
    import test_deepseek_v41_cache as fx
    from tokenspeed_kernel.ops.attention import dsv41
    from tokenspeed_kernel.platform import current_platform

    width, bs = 6, 16
    backend = fx._verify_backend("cuda", bs, width)
    backend.spec = replace(backend.spec, index_topk=64, candidate_topk=16)
    packing = {
        group.group_id: group.cache_blocks_per_lcm_block
        for group in backend.cache_pool.arena.plan.groups
    }
    tables = fx._tables_with_packing("cuda", packing)
    torch.manual_seed(66047)
    for owner, gid in ((2, fx.R2), (8, fx.R2), (14, fx.R2), (20, fx.R1)):
        cache = backend.cache_pool.index_k(owner)
        for req in range(2):
            pages = int((tables[gid][req] > 0).sum())
            logical = torch.arange(pages * 64, device="cuda")
            slots = tables[gid][req, logical // 64].long() * 64 + logical % 64
            dsv41.cache_scatter(
                torch.randn(logical.numel(), 128, device="cuda", dtype=torch.bfloat16),
                cache,
                slots,
                backend.index_format,
            )
    heads = 32 if current_platform().is_blackwell else 2
    q = torch.randn(bs * width, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(bs * width, heads, device="cuda", dtype=torch.bfloat16)
    req = (torch.arange(bs, device="cuda") % 2).long()
    seq = torch.randint(1, 400, (bs,), device="cuda", dtype=torch.int32)
    seq[-1] = 0  # one padded request
    incoming = {gid: t[req].contiguous() for gid, t in tables.items()}
    rope_cache = fx._global_rope_table()

    def refresh():
        backend.refresh_decode_metadata(
            bs,
            bs - 1,
            torch.arange(bs, device="cuda"),
            seq,
            forward_mode=ForwardMode.DECODE,
            block_tables=incoming,
            for_graph_replay=False,
        )

    layers = (2, 8, 14, 20, 24, 28, 32, 36)

    def run():
        refresh()
        backend.begin_indexer_stage()
        meta = backend.query_metadata(ForwardMode.DECODE)
        query = (
            dsv41.IndexQueryInput(q, meta.positions, rope_cache, 0.023456789)
            if deferred
            else q
        )
        out = []
        for layer in layers:
            backend.select_global(
                layer,
                query,
                weights,
                meta.positions,
                meta.request_indices,
                ForwardMode.DECODE,
                None,
            )
            record = backend.sparse_topk.decode
            out.append((record.logical_rows.clone(), record.lengths.clone()))
            if layer == 20:
                out.append(
                    (
                        record.candidates.block_ids.clone(),
                        record.candidates.lengths.clone(),
                    )
                )
        return out

    monkeypatch.setattr(backend_module, "_DECODE_INDEXER_TP", False)
    reference = run()
    assert any((rows >= 0).sum() > 0 for rows, _ in reference)

    monkeypatch.setattr(backend_module, "_DECODE_INDEXER_TP", True)
    monkeypatch.setattr(backend_module, "_DECODE_INDEXER_TP_MIN_ROWS", 64)
    stash = {}
    state, gather = _fake_world(stash)
    monkeypatch.setattr(query_partition, "all_gather_single", gather)
    for rank in (3, 2, 1, 0):
        state["rank"], state["call"] = rank, 0
        backend.spec = replace(
            backend.spec,
            indexer_decode_group=tuple(range(TP)),
            indexer_decode_rank=rank,
        )
        sharded = run()
    # Every source layer shards; the candidate source gathers its blocks too.
    assert state["call"] == len(layers)
    names = (2, 8, 14, 20, "20-candidates", 24, 28, 32, 36)
    for name, want, got in zip(names, reference, sharded, strict=True):
        for a, b in zip(want, got):
            assert torch.equal(a, b), f"layer {name} differs"

    # Below the row floor every rank keeps the replicated path.
    monkeypatch.setattr(backend_module, "_DECODE_INDEXER_TP_MIN_ROWS", 97)
    state["call"] = 0
    for (a0, b0), (a1, b1) in zip(reference, run()):
        assert torch.equal(a0, a1) and torch.equal(b0, b1)
    assert state["call"] == 0
