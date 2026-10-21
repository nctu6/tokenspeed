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

"""Deferred indexer preprocessing through native and portable selection."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_blackwell
    or current_platform().arch_version.minor > 3,
    reason="requires SM100-SM103 native indexer",
)


def _sparse_inputs():
    tokens, count = 4, 128
    torch.manual_seed(917 + tokens + count)
    capacity = 65536
    pages = tokens * (capacity // 64) + 1
    storage = torch.empty((pages, 9216), dtype=torch.uint8, device="cuda")
    cache = storage[:, :4352]
    cache[:, :4096].random_(0, 256)
    cache[:, 4096:].random_(123, 127)
    weights = torch.randn((tokens, 32), device="cuda", dtype=torch.bfloat16)
    table = torch.arange(1, pages, device="cuda", dtype=torch.int32).reshape(tokens, -1)
    lengths = torch.full((tokens,), capacity - 3, device="cuda", dtype=torch.int32)
    candidates = torch.randint(
        -5, capacity // 8 + 5, (tokens, count), device="cuda", dtype=torch.int32
    )
    lengths[0] = 0
    lengths[1] = 81
    weights[-1] = -weights[-1].abs()
    table[:, 1] = -1
    table[:, 3] = pages + 7
    candidates[:, :5] = torch.tensor([0, 7, 7, -1, 8192], device="cuda")
    candidates[:, -1] = (lengths - 1).clamp_min(0) // 8

    def expand(tensor):
        out = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            device="cuda",
            dtype=tensor.dtype,
        )[..., ::2]
        out.copy_(tensor)
        return out

    table, lengths, candidates, weights = map(
        expand, (table, lengths, candidates, weights)
    )
    return weights, cache, table, lengths, candidates


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("candidate_pool", [False, True])
def test_deferred_query_preprocessing_matches_explicit_rope(broadcast, candidate_pool):
    from tokenspeed_kernel.ops.attention.dsv41.triton import rope_inplace

    weights, cache, table, lengths, candidates = _sparse_inputs()
    q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([-1, 1, 127, 255], device="cuda")
    angles = (
        torch.arange(256, device="cuda")[:, None]
        * torch.linspace(0.001, 1, 32, device="cuda")[None, :]
    )
    rotary = torch.cat((angles.cos(), angles.sin()), -1)
    if broadcast:
        table = table[-1:].expand(4, -1)
    paged_cache = cache.as_strided((cache.shape[0], 64, 68), (cache.stride(0), 68, 1))
    deferred = dsv41.IndexQueryInput(q, positions, rotary, 0.023456789)
    original = q.clone()
    chosen = candidates if candidate_pool else None

    def select(query, head_weights, solution):
        return dsv41.index_topk(
            query,
            head_weights,
            paged_cache,
            table,
            lengths,
            chosen,
            64,
            0,
            8,
            32,
            256,
            None,
            None,
            solution,
        )

    for solution in ("deep_gemm", "triton"):
        expected = select(
            rope_inplace(q.clone(), positions, rotary),
            weights * deferred.weight_scale,
            solution,
        )
        actual = select(deferred, weights, solution)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(q, original, rtol=0, atol=0)
