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

"""DeepGEMM indexers with DeepSelect top-k on SM120, on real hardware.

V3.2 DSA leaves (decode and prefill) select with the in-tree CuTe DSL
DeepSelect kernel where it beats the TRT-LLM and Triton selectors; the V4.1
FP8 index-row leaf scores with DeepGEMM and selects with the same kernel.
"""

import math

import pytest
import torch
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

PAGE, HEADS, DIM = 64, 64, 128


def _index_cache(num_pages):
    """Page-planar FP8 index-K cache: 64 E4M3 rows then 64 FP32 scales per page."""
    keys = torch.randn(num_pages, PAGE, DIM, device="cuda", dtype=torch.bfloat16)
    scales = keys.abs().amax(-1).float().clamp_min(1e-4) / 448.0
    cache = torch.empty(num_pages, PAGE * (DIM + 4), device="cuda", dtype=torch.uint8)
    cache[:, : PAGE * DIM] = (
        (keys / scales[..., None])
        .to(torch.float8_e4m3fn)
        .view(num_pages, -1)
        .view(torch.uint8)
    )
    cache[:, PAGE * DIM :] = scales.view(torch.uint8).view(num_pages, -1)
    return cache.view(num_pages * PAGE, DIM + 4)


@pytest.fixture
def spy(monkeypatch):
    """Record the logits and lengths the leaf hands to DeepSelect."""
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    calls = []
    real = dsa_deep_gemm.deepselect_local_topk

    def record(logits, lengths, topk, out):
        calls.append((logits.clone(), lengths.clone()))
        return real(logits, lengths, topk, out)

    monkeypatch.setattr(dsa_deep_gemm, "deepselect_local_topk", record)
    return calls


def _assert_exact_topk(offsets, logits, lengths, topk):
    """Each row holds the float32 top-k of its first ``lengths`` columns."""
    for row in range(offsets.shape[0]):
        end = int(lengths[row])
        chosen = offsets[row][offsets[row] >= 0].long()
        count = min(topk, end)
        assert chosen.numel() == count, (row, end, chosen.numel())
        assert chosen.unique().numel() == count
        assert chosen.max() < end if count else True
        reference = logits[row, :end].topk(count).values
        picked = logits[row, chosen].sort(descending=True).values
        torch.testing.assert_close(picked, reference, atol=0, rtol=0)


def test_prefill_topk_selects_the_exact_top_k(spy):
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    torch.manual_seed(0)
    topk, num_tokens, prefix = 2048, 640, 5000
    num_keys = prefix + num_tokens
    cache = _index_cache(math.ceil(num_keys / PAGE))
    slots = torch.arange(num_keys, device="cuda", dtype=torch.int64)
    row_starts = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    row_ends = prefix + 1 + torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    q = torch.randn(num_tokens, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(num_tokens, HEADS, device="cuda", dtype=torch.float32)
    out, lens = dsa_deep_gemm.deep_gemm_dsa_prefill_topk(
        q,
        weights,
        slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=DIM**-0.5,
        index_k_cache=cache,
        page_size=PAGE,
        max_seqlen_k=num_keys,
    )
    assert len(spy) == 1, "prefill must select with DeepSelect"
    logits, lengths = spy[0]
    assert lens.tolist() == lengths.clamp_max(topk).tolist()
    _assert_exact_topk(out, logits, lengths, topk)


@pytest.mark.parametrize(
    ("batch", "context", "uses_deepselect"),
    [(96, 32768, True), (48, 65536, True), (4, 32768, False), (96, 16384, False)],
)
def test_decode_topk_selects_the_exact_top_k(batch, context, uses_deepselect, spy):
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    torch.manual_seed(1)
    topk = 2048
    seq_lens = torch.randint(
        context * 3 // 4, context + 1, (batch,), device="cuda", dtype=torch.int32
    )
    max_pages = math.ceil(context / PAGE)
    cache = _index_cache(batch * max_pages)
    block_table = torch.randperm(
        batch * max_pages, device="cuda", dtype=torch.int32
    ).view(batch, max_pages)
    q = torch.randn(batch, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, HEADS, device="cuda", dtype=torch.float32)
    seq_lens_2d = seq_lens.view(-1, 1).contiguous()
    plan = dsa_deep_gemm.deep_gemm_dsa_plan(page_size=PAGE, seq_lens_2d=seq_lens_2d)
    slots, lens = dsa_deep_gemm.deep_gemm_dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=PAGE,
        topk=topk,
        softmax_scale=DIM**-0.5,
        index_k_cache=cache,
        seq_lens_2d=seq_lens_2d,
        plan=plan,
    )
    assert (len(spy) == 1) is uses_deepselect
    assert lens.tolist() == seq_lens.clamp_max(topk).tolist()
    if uses_deepselect:
        logits, lengths = spy[0]
        assert lengths.tolist() == seq_lens.tolist()
        # The leaf maps local offsets to physical slots; invert that to compare
        pages = block_table.long()
        for row in range(batch):
            chosen = slots[row][slots[row] >= 0].long()
            local = torch.empty_like(chosen)
            page_of = {int(p): i for i, p in enumerate(pages[row].tolist())}
            for i, slot in enumerate(chosen.tolist()):
                local[i] = page_of[slot // PAGE] * PAGE + slot % PAGE
            end = int(lengths[row])
            reference = logits[row, :end].topk(topk).values
            picked = logits[row, local].sort(descending=True).values
            torch.testing.assert_close(picked, reference, atol=0, rtol=0)


def test_deepselect_rejects_what_it_cannot_read():
    from tokenspeed_kernel.ops.attention.dsa.cute_dsl import deepselect_supports

    good = torch.randn(4, 4096, device="cuda")
    assert deepselect_supports(good, 2048)
    assert not deepselect_supports(good, 4096)  # beyond the largest capacity
    assert not deepselect_supports(good.double(), 2048)
    assert not deepselect_supports(good.t(), 2048)
    assert not deepselect_supports(torch.randn(4, 4098, device="cuda")[:, :4097], 2048)
    assert not deepselect_supports(torch.randn(4, 4097, device="cuda")[:, 1:], 2048)


def test_deepselect_local_topk_splits_more_rows_than_one_grid_holds():
    from tokenspeed_kernel.ops.attention.dsa.cute_dsl import deepselect_local_topk

    torch.manual_seed(3)
    rows, width, topk = 70000, 3000, 512
    logits = torch.randn(rows, width, device="cuda")
    lengths = torch.randint(1, width + 1, (rows,), device="cuda", dtype=torch.int32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    deepselect_local_topk(logits, lengths, topk, out)
    sample = torch.cat([torch.arange(0, 40), torch.arange(65500, 65600)])
    _assert_exact_topk(out[sample], logits[sample], lengths[sample], topk)
