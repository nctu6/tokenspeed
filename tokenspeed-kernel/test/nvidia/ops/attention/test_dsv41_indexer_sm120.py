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

"""The V4.1 native indexer on SM120 (FP8 and packed FP4 index rows), on real hardware."""

import os
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.thirdparty.deepgemm import (
    SM120_ARCH_MAJOR,
    deepgemm_supports_sm120,
)

pytestmark = pytest.mark.skipif(
    not (
        Platform.get().is_nvidia
        and Platform.get().arch_version.major == SM120_ARCH_MAJOR
        and deepgemm_supports_sm120()
    ),
    reason="SM120 with a DeepGEMM build that has SM120 kernels only",
)


def test_fp8_indexer_declines_dense_reindex_where_the_pool_scorer_wins():
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import fp8_indexer_wins

    # Full passes always win
    assert fp8_indexer_wins(4096, 32768, False)
    # Reindex scores the whole history here: short histories and small
    # (rows x history) products win, long histories of many rows lose
    assert fp8_indexer_wins(4096, 16384, True)
    assert fp8_indexer_wins(8, 65536, True)
    assert not fp8_indexer_wins(32, 32768, True)
    assert not fp8_indexer_wins(128, 32768, True)


def _index_case(rows, history, shared_table):
    """FP8 ``index_v4`` rows, 32 heads, a paged history and per-row visible lengths."""
    torch.manual_seed(5)
    pages = -(-history // 64) + 1
    copies = 1 if shared_table else rows
    keys = torch.randn(pages * 64 * copies, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(pages * copies, 64, 132, dtype=torch.uint8, device="cuda")
    dsv41.cache_scatter(keys, cache, torch.arange(len(keys), device="cuda"), "index_v4")
    q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, 32, dtype=torch.bfloat16, device="cuda")
    if shared_table:
        table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)
        table = table[None, :].expand(rows, -1)
        visible = history - rows + 1 + torch.arange(rows, device="cuda").int()
    else:
        table = torch.randperm(pages * rows, device="cuda").int().view(rows, pages)
        visible = torch.full((rows,), history, dtype=torch.int32, device="cuda")
    return q, weights, cache, table, visible


@pytest.mark.parametrize(("rows", "shared_table"), [(8, False), (96, True)])
def test_native_index_topk_selects_what_the_portable_scorer_selects(rows, shared_table):
    q, weights, cache, table, visible = _index_case(rows, 4096, shared_table)

    def run(solution, candidate_blocks=None, candidate_topk=64):
        return dsv41.index_topk(
            q,
            weights,
            cache,
            table,
            visible,
            candidate_blocks,
            512,
            candidate_topk,
            8,
            64,
            4096,
            None,
            None,
            solution,
        )

    def agreement(native, portable, native_lengths, portable_lengths):
        assert native_lengths.tolist() == portable_lengths.tolist()
        shared = [
            torch.isin(x[x >= 0], y[y >= 0]).float().mean().item()
            for x, y in zip(native, portable)
        ]
        return min(shared)

    native, native_lengths, blocks, block_lengths = run(None)
    portable, portable_lengths, _, _ = run("triton")
    # Both score the same FP8 rows; only scores tied at the boundary may swap
    assert agreement(native, portable, native_lengths, portable_lengths) > 0.97
    # A Reindex pass over the pool the Full pass produced
    native, native_lengths, _, _ = run(None, blocks, 0)
    portable, portable_lengths, _, _ = run("triton", blocks, 0)
    assert agreement(native, portable, native_lengths, portable_lengths) > 0.97


def _packed_case(rows, history, shared_table, seed=7):
    """Packed 68-byte V4.1 index rows, 32 heads, as the runtime stores them on SM120."""
    torch.manual_seed(seed)
    pages = -(-history // 64) + 1
    copies = 1 if shared_table else rows
    keys = torch.randn(pages * 64 * copies, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(pages * copies, 64, 68, dtype=torch.uint8, device="cuda")
    dsv41.cache_scatter(keys, cache, torch.arange(len(keys), device="cuda"), "index")
    q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, 32, dtype=torch.bfloat16, device="cuda")
    if shared_table:
        table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)
        table = table[None, :].expand(rows, -1)
        visible = history - rows + 1 + torch.arange(rows, device="cuda").int()
    else:
        table = torch.randperm(pages * rows, device="cuda").int().view(rows, pages)
        # Ragged lengths, including a partial last page
        visible = history - torch.randint(0, 100, (rows,), device="cuda").int()
    return q, weights, cache, table, visible


def _reference_logits(q, weights, cache, table, visible):
    """Exact (float64) scores of the dequantized queries against the dequantized rows."""
    capacity = table.shape[1] * 64
    q = dsv41.index_q_quantize(q, "index", None).double()
    logical = torch.arange(capacity, device="cuda")
    slots = table.long()[:, logical // 64] * 64 + logical % 64
    keys = dsv41.cache_gather(
        cache, slots, "index", torch.empty((*slots.shape, 128), device="cuda")
    ).double()
    scores = torch.einsum("thd,tsd->ths", q, keys).relu()
    logits = (scores * weights.double()[:, :, None]).sum(1)
    return logits.masked_fill(logical[None, :] >= visible[:, None].long(), -torch.inf)


def _assert_exact_topk(selected, lengths, logits, pool=None, k=512, rtol=1e-9):
    """Selected rows are a top-k set of the reference logits (ties may swap).

    ``rtol`` bounds how far below the k-th reference score a pick may be: FP32
    rounding for the dense scorers, BF16 rounding for the sparse one.
    """
    for row in range(logits.shape[0]):
        ids = selected[row, : lengths[row]].long()
        assert (selected[row, lengths[row] :] == -1).all()
        assert ids.unique().numel() == ids.numel()
        values = logits[row] if pool is None else logits[row, pool[row]]
        live = values[values > -torch.inf]
        want = min(k, live.numel())
        assert lengths[row].item() == want
        if want == 0:
            continue
        kth = live.topk(want).values[-1]
        picked = logits[row, ids]
        if pool is not None:
            assert torch.isin(ids, pool[row]).all()
        # Every pick scores at least the k-th reference score, so the set is a
        # top-k set; only rows within float rounding of the boundary may swap
        assert (picked >= kth - rtol * kth.abs().clamp_min(1)).all(), row


def _pool_rows(blocks, block_lengths, visible):
    """Logical rows of each query's candidate pool."""
    rows = []
    for b, n, v in zip(blocks, block_lengths, visible):
        ids = (b[:n].long()[:, None] * 8 + torch.arange(8, device="cuda")).flatten()
        rows.append(ids[ids < v])
    return rows


def _run(q, weights, cache, table, visible, solution, candidates, candidate_topk):
    return dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        512,
        candidate_topk,
        8,
        64,
        4096,
        None,
        None,
        solution,
    )


@pytest.mark.parametrize(
    ("rows", "history", "shared_table"),
    # The last two cases' Reindex passes read only the pool (see the rule test)
    [
        (1, 700, False),
        (8, 4096, False),
        (33, 20000, False),
        (96, 4096, True),
        (64, 33000, False),
        (96, 33000, True),
    ],
)
def test_packed_rows_select_the_exact_topk(rows, history, shared_table):
    q, weights, cache, table, visible = _packed_case(rows, history, shared_table)
    reference = _reference_logits(q, weights, cache, table, visible)
    # Full pass that also sources a candidate pool, as the candidate-source layer does
    selected, lengths, blocks, block_lengths = _run(
        q, weights, cache, table, visible, None, None, 2048
    )
    _assert_exact_topk(selected, lengths, reference)
    # The pool holds the best blocks by their maximum row, plus the newest block
    maxima = reference.view(rows, -1, 8).amax(-1)
    for row in range(rows):
        pool = blocks[row, : block_lengths[row]].long()
        live = maxima[row][maxima[row] > -torch.inf]
        want = min(2048, live.numel())
        assert block_lengths[row].item() == want
        kth = live.topk(want).values[-1]
        newest = (visible[row].item() - 1) // 8
        others = pool[pool != newest]
        assert (maxima[row, others] >= kth - 1e-9 * kth.abs().clamp_min(1)).all()
    # Ordinary Full pass
    full, full_lengths, _, empty = _run(
        q, weights, cache, table, visible, None, None, 0
    )
    _assert_exact_topk(full, full_lengths, reference)
    assert (empty == 0).all()
    # Reindex over that pool: picks come from the pool only. The sparse scorer
    # writes BF16 logits (DeepGEMM's sparse contract), so a pick and the k-th
    # score may each round by half an ulp: at most 2^-7 apart, relatively
    # (measured worst 5.6e-3, with 0.5-0.8% of the picks below the k-th score)
    pool = _pool_rows(blocks, block_lengths, visible)
    selected, lengths, _, _ = _run(q, weights, cache, table, visible, None, blocks, 0)
    for row in range(rows):
        masked = torch.full_like(reference[row], -torch.inf)
        masked[pool[row]] = reference[row, pool[row]]
        _assert_exact_topk(
            selected[row : row + 1],
            lengths[row : row + 1],
            masked[None],
            k=512,
            rtol=2**-7,
        )


def test_packed_reindex_scores_only_the_pool_from_twice_its_size():
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import _sparse_reindex_wins

    pool = 2048 * 8
    # A history within the pool is taken whole, and below twice the pool the
    # dense scorers are as fast or faster
    assert not _sparse_reindex_wins(16384, pool)
    assert not _sparse_reindex_wins(24576, pool)
    # From twice the pool the pool-only scorer wins, for any number of rows
    assert _sparse_reindex_wins(32768, pool)
    assert _sparse_reindex_wins(33088, pool)
    assert _sparse_reindex_wins(131072, pool)


def test_packed_reindex_takes_a_small_pool_whole():
    """A small pool is taken whole, whatever the history holds."""
    q, weights, cache, table, visible = _packed_case(4, 30000, False)
    blocks = torch.full((4, 2048), -1, dtype=torch.int32, device="cuda")
    blocks[:, :40] = torch.arange(0, 400, 10, device="cuda", dtype=torch.int32)
    selected, lengths, candidate_blocks, candidate_lengths = _run(
        q, weights, cache, table, visible, None, blocks, 0
    )
    expected = (
        blocks[0, :40].long()[:, None] * 8 + torch.arange(8, device="cuda")
    ).flatten()
    for row in range(4):
        assert lengths[row].item() == 320
        assert torch.equal(
            selected[row, :320].long().sort().values, expected.sort().values
        )
    assert (candidate_lengths == 0).all()


def test_packed_rows_are_selected_natively():
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_packed_sm120_indexer_available,
    )
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    assert is_packed_sm120_indexer_available()
    kernel = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=None,
        platform=None,
        traits={
            "index_heads": 32,
            "index_k_format": "mxfp4",
            "index_shards": 1,
            "native_indexer": True,
        },
        solution=None,
        override=None,
    )
    assert KernelRegistry.get().get_by_name(kernel.name).name == (
        "deep_gemm_sm120_dsv41_index_topk"
    )


def _triton_kernels(*modules):
    from tokenspeed_kernel._triton import triton

    return [
        value
        for module in modules
        for value in vars(module).values()
        if isinstance(value, triton.runtime.JITFunction)
    ]


def _deep_gemm_kernels():
    """Compiled DeepGEMM kernels in the JIT cache (None without a known cache)."""
    root = os.environ.get("DG_JIT_CACHE_DIR")
    cache = Path(root) / "cache" if root else None
    return None if cache is None or not cache.is_dir() else set(os.listdir(cache))


def test_packed_rows_reuse_the_compiled_kernels_across_shapes():
    """Request counts and history lengths are runtime values of every kernel."""
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from tokenspeed_kernel.ops.attention.dsv41 import triton as portable
    from tokenspeed_kernel.ops.attention.dsv41._deep_gemm import indexer
    from utils import assert_no_triton_compile, warm_specialization_classes

    def run(shape):
        rows, history, shared = shape
        q, weights, cache, table, visible = _packed_case(rows, history, shared)
        _, _, blocks, _ = _run(q, weights, cache, table, visible, None, None, 2048)
        _run(q, weights, cache, table, visible, None, None, 0)
        _run(q, weights, cache, table, visible, None, blocks, 0)

    def key(shape):
        # Triton keys integers on 1 / divisible by 16 / other; query packing
        # buckets its tile at 64 queries (and skips weight conversion below 16);
        # the pool layout (paged or one request's chunk) and the Reindex scorer
        # pick the launches
        rows, history, shared = shape
        cls = 1 if rows == 1 else (16 if rows % 16 == 0 else 0)
        sparse = history >= 32768
        return cls, rows < 16, rows > 64, shared, sparse

    sweep = [
        (2, 900, False),
        (3, 5000, False),
        (9, 2000, False),
        (48, 1500, False),
        (100, 3000, False),
        (40, 34000, False),
        (5, 2000, True),
        (33, 4096, True),
        (21, 36000, True),
    ]
    pool = [
        (1, 700, False),
        (16, 1024, False),
        (17, 1024, False),
        (64, 33000, False),
        (80, 1024, False),
        (16, 1024, True),
        (17, 2048, True),
        (17, 33000, True),
        (4, 1024, False),
        (4, 1024, True),
    ]
    warm_specialization_classes(run, key, sweep, pool)
    compiled = _deep_gemm_kernels()
    with assert_no_triton_compile(*_triton_kernels(portable, indexer)):
        for shape in sweep:
            run(shape)
    if compiled is not None:
        assert _deep_gemm_kernels() == compiled
