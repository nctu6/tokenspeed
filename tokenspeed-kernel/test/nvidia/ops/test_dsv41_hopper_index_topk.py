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

"""The Hopper index Top-K glue selects exactly what its unfused form selected.

``_reference_index_topk`` keeps the eager PyTorch glue the fused launches
replaced (slot arithmetic, advanced-index key gathers, ``torch.where`` masks,
per-tile length clamps, separate length fills). The scoring and selection
kernels are shared, so every output -- row ids, row counts, candidate blocks,
block counts -- and the gathered FP8 keys must match bit for bit.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as impl
from tokenspeed_kernel.ops.attention.dsv41 import triton as kernels
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not impl.is_hopper_indexer_available(),
    reason="requires the Hopper FP8 indexer",
)

DEVICE = "cuda"


def _reference_gather(cache, slots):
    values, scales = impl._index_planes(cache)
    live = (slots >= 0) & (slots < cache.shape[0] * 64)
    safe = slots.clamp_min(0)
    pages, rows = safe // 64, safe % 64
    gathered = values[pages, rows]
    gathered = torch.where(live[:, None], gathered, torch.zeros_like(gathered))
    row_scales = torch.where(live, scales[pages, rows], torch.zeros_like(live).float())
    return gathered.view(torch.float8_e4m3fn).contiguous(), row_scales.contiguous()


def _reference_slots(page_table, capacity, pages_in_cache):
    logical = torch.arange(capacity, dtype=torch.int64, device=page_table.device)
    pages = page_table[0, logical // 64]
    return (pages.to(torch.int64) * 64 + logical % 64).masked_fill(
        (pages < 0) | (pages >= pages_in_cache), -1
    )


def _reference_select(scores, k, capacity, destination, lengths, candidates, ends):
    width = min(k, scores.shape[1])
    if width == 0:
        destination.fill_(-1)
        lengths.zero_()
        return
    if ends is None:
        ends = torch.full(
            (scores.shape[0],), scores.shape[1], dtype=torch.int32, device=scores.device
        )
    rs = impl.row_select
    cluster = rs.choose_cluster_size(
        scores.shape[0], scores.shape[1], capacity, impl.platform.sm_count
    )
    indices, values = rs.deepselect_topk(
        scores, ends, width, capacity=capacity, cluster_size=cluster
    )
    kernels._finish_topk(values, indices, destination, lengths, candidates)


def _reference_native_select(
    logits, visible, candidates, topk, candidate_topk, out, scores, block_stats
):
    rows, lengths, blocks, block_lengths = out
    if candidates is None:
        _reference_select(logits, topk, 512, rows, lengths, None, visible)
    else:
        if scores is None:
            scores = kernels.candidate_scores(logits, candidates)
        _reference_select(scores, topk, 512, rows, lengths, candidates, None)
    if candidate_topk:
        maxima, block_ends = block_stats
        _reference_select(
            maxima, candidate_topk, 2048, blocks, block_lengths, None, block_ends
        )
    else:
        blocks.fill_(-1)
        block_lengths.zero_()


def _reference_paged_scores(queries, cache, weights, table, visible, capacity, clean):
    api = impl._hopper_api(queries)
    safe, lengths = kernels.safe_metadata(table, visible, cache.shape[0], capacity, 64)
    context = lengths.clamp_min(1)[:, None]
    schedule = api.get_paged_mqa_logits_metadata(context, 64, api.get_num_sms())
    flat = impl._index_pages(cache)
    view = flat.as_strided((cache.shape[0], 64, 1, 132), (flat.stride(0), 132, 132, 1))
    logits = api.fp8_paged_mqa_logits(
        queries.unsqueeze(1),
        view,
        weights,
        context,
        safe,
        schedule,
        capacity,
        clean_logits=False,
    )
    logits = logits.reshape(queries.shape[0], -1)[:, :capacity]
    if not clean:
        return logits
    return kernels.clean_logits(logits, lengths, table, cache.shape[0], capacity, 64)


def _reference_index_topk(index_q, weights, cache, page_table, visible_lens, candidate_blocks, topk, candidate_topk, query_chunk_size,):  # fmt: skip
    n = index_q.shape[0]
    shapes = ((n, topk), (n,), (n, candidate_topk), (n,))
    out = tuple(torch.empty(s, dtype=torch.int32, device=DEVICE) for s in shapes)
    capacity = page_table.shape[1] * 64
    if capacity == 0:
        out[0].fill_(-1)
        out[1].zero_()
        out[2].fill_(-1)
        out[3].zero_()
        return out, None
    dense = page_table.stride(0) == 0 and n > 1
    budget = (128 if dense else 32) << 20
    tile = min(query_chunk_size, max(1, budget // (capacity * 4)))
    queries, folded = kernels.quantize_index_queries(index_q, weights)
    keys = None
    if dense:
        keys = _reference_gather(
            cache, _reference_slots(page_table, capacity, cache.shape[0])
        )
    sparse = (
        candidate_blocks is not None
        and not candidate_topk
        and not dense
        and impl.sparse_index_scores_supported(
            queries, folded, page_table, candidate_blocks
        )
    )
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        visible = visible_lens[begin:end].clamp(0, capacity).to(torch.int32)
        tile_out = tuple(t[begin:end] for t in out)
        cands = None if candidate_blocks is None else candidate_blocks[begin:end]
        if sparse:
            values, scales = impl._index_planes(cache)
            scores = impl.sparse_index_scores(
                queries[begin:end],
                folded[begin:end],
                values.view(torch.float8_e4m3fn),
                scales,
                page_table[begin:end],
                visible,
                cands,
                impl.pdl_enabled(),
            )
            _reference_native_select(
                None, visible, cands, topk, 0, tile_out, scores, None
            )
            continue
        bounded = bool(candidate_topk) and candidate_blocks is None
        if dense:
            starts, ends = kernels.dense_ranges(visible, capacity)
            logits = impl._hopper_api(queries).fp8_mqa_logits(
                q=queries[begin:end],
                kv=keys,
                weights=folded[begin:end],
                cu_seq_len_k_start=starts,
                cu_seq_len_k_end=ends,
                clean_logits=False,
                max_seqlen_k=capacity,
            )
            if not bounded:
                logits = kernels.clean_logits(
                    logits, ends, page_table[begin:end], cache.shape[0], capacity, 64
                )
        else:
            logits = _reference_paged_scores(
                queries[begin:end],
                cache,
                folded[begin:end],
                page_table[begin:end],
                visible,
                capacity,
                not bounded,
            )
        stats = None
        if bounded:
            logits, maxima, block_ends = kernels.clean_block_maxima(
                logits, visible, page_table[begin:end], cache.shape[0]
            )
            stats = (maxima, block_ends)
        _reference_native_select(
            logits, visible, cands, topk, candidate_topk, tile_out, None, stats
        )
    return out, keys


def _index_cache(pages, seed, padded=False):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (pages, 2, 64, 132) if padded else (pages, 64, 132)
    cache = torch.zeros(shape, dtype=torch.uint8, device=DEVICE)
    if padded:
        cache = cache[:, 1]
    rows = pages * 64
    dsv41.cache_scatter(
        torch.randn(
            (rows, 128), dtype=torch.bfloat16, device=DEVICE, generator=generator
        ),
        cache,
        torch.arange(rows, device=DEVICE),
        "index_v4",
    )
    return cache, generator


def _case(n, pages, width, seed, *, dense, dtype=torch.int32, padded=False):
    cache, generator = _index_cache(pages, seed, padded)
    q = torch.randn(
        (n, 32, 128), dtype=torch.bfloat16, device=DEVICE, generator=generator
    )
    w = torch.rand((n, 32), dtype=torch.float32, device=DEVICE, generator=generator)
    rows = 1 if dense else n
    table = torch.stack(
        [
            torch.randperm(pages, device=DEVICE, generator=generator)[:width]
            for _ in range(rows)
        ]
    ).to(dtype)
    if width > 3:
        table[:, 1] = -1  # null page
        table[:, 2] = pages + 3  # out-of-range page
    if dense:
        table = table.expand(n, -1)
    capacity = width * 64
    visible = torch.randint(
        0, capacity + 1, (n,), device=DEVICE, generator=generator
    ).to(dtype)
    visible[0] = 0
    if n > 1:
        visible[1] = capacity + 37  # past the table
    if n > 2:
        visible[2] = -5
    if n > 3:
        visible[3] = capacity - 3  # inside a partial block
    return cache, q, w, table, visible, generator


def _candidates(n, blocks, width_blocks, generator, rows_per_request=None):
    out = torch.stack(
        [
            torch.randperm(width_blocks + 16, device=DEVICE, generator=generator)[
                :blocks
            ]
            for _ in range(n)
        ]
    ).to(torch.int32)
    out[:, ::7] = -1
    out[0] = -1
    return out


def _run_both(q, w, cache, table, visible, cands, topk, candidate_topk, chunk):
    want, _ = _reference_index_topk(
        q, w, cache, table, visible, cands, topk, candidate_topk, chunk
    )
    got = dsv41.index_topk(
        q, w, cache, table, visible, cands, topk, candidate_topk, 8, chunk, 256,
        None, None, "deep_gemm",
    )  # fmt: skip
    for name, actual, expected in zip(
        ("rows", "lengths", "blocks", "block lengths"), got, want, strict=True
    ):
        assert torch.equal(actual, expected), name
    return got


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("mode", ["full", "source", "reindex"])
@pytest.mark.parametrize("chunk", [1024, 5])
def test_index_topk_matches_unfused_glue(dense, mode, chunk):
    n = 37
    pages, width = 96, 80
    cache, q, w, table, visible, generator = _case(n, pages, width, 11, dense=dense)
    cands = None
    candidate_topk = 0
    if mode == "source":
        candidate_topk = 2048
    elif mode == "reindex":
        cands = _candidates(n, 128, width * 8, generator)
    got = _run_both(q, w, cache, table, visible, cands, 512, candidate_topk, chunk)
    assert int(got[1].max()) > 0
    if mode == "source":
        assert int(got[3].max()) > 0


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_index_topk_matches_with_int64_metadata_and_padded_pages(dtype):
    cache, q, w, table, visible, generator = _case(
        9, 40, 24, 3, dense=False, dtype=dtype, padded=True
    )
    for candidate_topk in (0, 100):
        _run_both(q, w, cache, table, visible, None, 65, candidate_topk, 4)
    cands = _candidates(9, 48, 24 * 8, generator).to(dtype)
    _run_both(q, w, cache, table, visible, cands, 65, 0, 4)
    # A table striding along its row is not served by the pool scorer.
    strided = table.repeat_interleave(2, dim=1)[:, ::2]
    _run_both(q, w, cache, strided, visible, cands, 65, 0, 4)
    dense = table[:1].expand(9, -1)
    _run_both(q, w, cache, dense, visible, cands, 65, 0, 4)
    _run_both(q, w, cache, dense, visible, None, 65, 100, 4)


def test_index_topk_matches_on_empty_inputs():
    cache, q, w, table, visible, generator = _case(4, 16, 8, 5, dense=False)
    visible.zero_()
    for candidate_topk in (0, 64):
        _run_both(q, w, cache, table, visible, None, 512, candidate_topk, 1024)
    # A single query never takes the dense path, even with a broadcast table.
    _run_both(
        q[:1], w[:1], cache, table[:1].expand(1, -1), visible[:1], None, 512, 0, 1
    )
    # No history at all, and no candidates in the pool.
    _run_both(q, w, cache, table[:, :0], visible, None, 512, 0, 1024)
    empty_pool = torch.full((4, 16), -1, dtype=torch.int32, device=DEVICE)
    visible.fill_(300)
    _run_both(q, w, cache, table, visible, empty_pool, 512, 0, 1024)
    _run_both(q, w, cache, table[:1].expand(4, -1), visible, empty_pool, 512, 0, 1024)


@pytest.mark.parametrize("width", [1, 3, 17, 80])
def test_gather_index_fp8_matches_advanced_indexing(width):
    pages = 40
    # Random bytes, including NaN and denormal scale patterns: the gather is a
    # byte copy, never a float conversion.
    generator = torch.Generator(device=DEVICE).manual_seed(width)
    cache = torch.randint(
        0, 256, (pages, 2, 64, 132), dtype=torch.uint8, device=DEVICE,
        generator=generator,
    )[:, 0]  # fmt: skip
    table = torch.randint(
        -3, pages + 3, (2, 2 * width), dtype=torch.int64, device=DEVICE,
        generator=generator,
    )[:1, ::2]  # fmt: skip
    capacity = width * 64
    values, scales = kernels.gather_index_fp8(cache, table[0], capacity)
    want_values, want_scales = _reference_gather(
        cache, _reference_slots(table, capacity, pages)
    )
    assert values.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert torch.equal(values.view(torch.uint8), want_values.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), want_scales.view(torch.int32))


def test_fused_length_outputs():
    lengths = torch.tensor([-4, 0, 1, 63, 64, 900], dtype=torch.int64, device=DEVICE)
    widths, visible = kernels.pool_lengths(lengths, 640, 1024)
    assert widths.tolist() == [1024] * 6
    assert visible.tolist() == [0, 0, 1, 63, 64, 640]
    table = torch.tensor([[0, -1, 9, 3]] * 6, dtype=torch.int32, device=DEVICE)
    context = torch.empty(6, dtype=torch.int32, device=DEVICE)
    safe, clamped = kernels.safe_metadata(table, lengths, 4, 256, 64, context=context)
    assert safe.tolist() == [[0, 0, 0, 3]] * 6
    assert clamped.tolist() == [0, 0, 1, 63, 64, 256]
    assert context.tolist() == [1, 1, 1, 63, 64, 256]
    logits = torch.randn(6, 40, device=DEVICE)
    cands = torch.tensor([[0, -1, 4]] * 6, dtype=torch.int32, device=DEVICE)
    ends = torch.full((6,), -1, dtype=torch.int32, device=DEVICE)
    scores = kernels.candidate_scores(logits, cands, ends)
    assert torch.equal(scores, kernels.candidate_scores(logits, cands))
    assert ends.tolist() == [24] * 6


def test_fused_glue_kernels_do_not_recompile_per_batch():
    """Row counts, widths and strides are runtime values for the new launches."""

    def run(n, width, pages, stride, pad):
        cache = torch.zeros((pages, 2, 64, 132), dtype=torch.uint8, device=DEVICE)
        cache = cache[:, 1] if pad else cache[:, 0]
        table = torch.arange(width * stride, dtype=torch.int32, device=DEVICE)
        kernels.gather_index_fp8(cache, table[::stride] % pages, width * 64)
        lengths = torch.arange(n + pad, dtype=torch.int32, device=DEVICE)[pad:]
        kernels.pool_lengths(lengths, width * 64, width * 8 + pad)
        rows = torch.randint(0, pages, (n, width * stride), device=DEVICE)
        context = torch.empty(n, dtype=torch.int32, device=DEVICE)
        kernels.safe_metadata(
            rows[:, ::stride], lengths, pages, width * 64, 64, context
        )
        logits = torch.randn(n, width * 64 + pad, device=DEVICE)[:, pad:]
        cands = torch.randint(-1, width * 8, (n, width + pad), device=DEVICE)
        kernels.candidate_scores(logits, cands[:, pad:].to(torch.int32), context)

    run(16, 16, 32, 1, 0)
    run(17, 17, 33, 2, 1)
    run(1, 1, 1, 1, 0)
    with assert_no_triton_compile(
        kernels._gather_index_fp8_kernel,
        kernels._dense_ranges,
        kernels._safe_metadata,
        kernels._candidate_scores_kernel,
    ):
        for case in ((5, 3, 7, 1, 1), (96, 160, 300, 2, 0), (1408, 81, 90, 3, 3)):
            run(*case)


@pytest.mark.parametrize("mode", ["full", "source", "reindex"])
def test_index_topk_graph_replay_follows_device_lengths(mode):
    """The fused glue keeps every length on the device: replays track them."""
    n = 12
    cache, q, w, table, visible, generator = _case(n, 64, 48, 23, dense=False)
    cands = _candidates(n, 128, 48 * 8, generator) if mode == "reindex" else None
    candidate_topk = 2048 if mode == "source" else 0
    shapes = ((n, 512), (n,), (n, candidate_topk), (n,))
    out = tuple(torch.empty(s, dtype=torch.int32, device=DEVICE) for s in shapes)

    def run():
        return dsv41.index_topk(
            q, w, cache, table, visible, cands, 512, candidate_topk, 8, 256, 256,
            None, out, "deep_gemm",
        )  # fmt: skip

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths in (0, 1, 700, 48 * 64 + 9):
        visible.fill_(lengths)
        visible[::3] = 300
        for tensor in out:
            tensor.fill_(12345)  # stale contents must be overwritten
        graph.replay()
        want, _ = _reference_index_topk(
            q, w, cache, table, visible, cands, 512, candidate_topk, 256
        )
        for actual, expected in zip(out, want, strict=True):
            assert torch.equal(actual, expected)
