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

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41_topk_graphs import (
    V41PrefillTopkGraphs,
)

PAGES = 512


def _cache(generator):
    from tokenspeed_kernel.ops.attention import dsv41

    cache = torch.zeros((PAGES, 64, 132), dtype=torch.uint8, device="cuda")
    dsv41.cache_scatter(
        torch.randn(
            (PAGES * 64, 128), dtype=torch.bfloat16, device="cuda", generator=generator
        ),
        cache,
        torch.arange(PAGES * 64, device="cuda"),
        "index_v4",
    )
    return cache


def _eager(q, w, cache, row, positions, ratio, candidates, ctopk):
    from tokenspeed_kernel.ops.attention import dsv41

    n = q.shape[0]
    visible = ((positions + 1) // ratio).clamp_min(0).to(torch.int32)
    out = (
        torch.empty((n, 512), dtype=torch.int32, device="cuda"),
        torch.empty((n,), dtype=torch.int32, device="cuda"),
        torch.empty((n, ctopk), dtype=torch.int32, device="cuda"),
        torch.empty((n,), dtype=torch.int32, device="cuda"),
    )
    dsv41.index_topk(
        q,
        w,
        cache,
        row.masked_fill(row <= 0, -1).expand(n, -1),
        visible,
        candidates,
        512,
        ctopk,
        8,
        1024,
        256,
        None,
        out,
        None,
    )
    return out


def _same_selection(graph_out, eager_out):
    rows, lens, blocks, block_lens = graph_out
    erows, elens, eblocks, eblock_lens = eager_out
    assert torch.equal(lens, elens)
    assert torch.equal(block_lens, eblock_lens)
    # Equal-score ties may pick another equivalent subset; on random scores
    # they are vanishingly rare, so the sorted selections must agree.
    assert torch.equal(rows.sort(1).values, erows.sort(1).values)
    assert torch.equal(blocks.sort(1).values, eblocks.sort(1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["full", "source", "reindex"])
def test_prefill_topk_graph_matches_eager(mode):
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_hopper_indexer_available,
    )

    if not is_hopper_indexer_available():
        pytest.skip("requires the Hopper DeepGEMM indexer")
    g = torch.Generator(device="cuda").manual_seed(0)
    cache = _cache(g)
    graphs = V41PrefillTopkGraphs(torch.device("cuda"))
    ratio = 2
    # Two spans per bucket: the first captures, the second replays with new
    # inputs, a narrower table and fewer rows.
    for n, pages, prefix in ((100, 90, 5000), (70, 70, 4300), (300, 200, 12000)):
        q = torch.randn((n, 32, 128), dtype=torch.bfloat16, device="cuda", generator=g)
        w = torch.rand((n, 32), dtype=torch.float32, device="cuda", generator=g)
        row = torch.randperm(PAGES, device="cuda", generator=g)[None, :pages]
        row = row.to(torch.int32)
        row[0, 3] = 0  # a null page
        positions = torch.arange(prefix, prefix + n, device="cuda")
        candidates, limit, ctopk = None, 0, 0
        if mode == "source":
            limit, ctopk = 2048, min(2048, pages * 8)
        elif mode == "reindex":
            # The replayed span's pool is narrower than its bucket.
            pool = 200 if n == 70 else 256
            candidates = torch.stack(
                [
                    torch.randperm(pages * 8, device="cuda", generator=g)[:pool]
                    for _ in range(n)
                ]
            ).to(torch.int32)
        expected = _eager(q, w, cache, row, positions, ratio, candidates, ctopk)
        out = tuple(torch.full_like(t, -7) for t in expected)
        assert graphs.select(
            q, w, cache, row, positions, ratio, candidates, 512, limit, 8, 1024, out
        )
        torch.cuda.synchronize()
        _same_selection(out, expected)
    assert all(entry is not None for entry in graphs._graphs.values())
