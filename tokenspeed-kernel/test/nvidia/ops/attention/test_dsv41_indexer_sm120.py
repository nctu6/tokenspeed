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

"""The V4.1 FP8 index-row indexer on SM120, on real hardware."""

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
