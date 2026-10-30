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

"""Paged MHA decode/extend as a fixed-length cross-attention read.

The layout is Whisper's cross KV: every slot owns ``PAGES_PER_SLOT`` whole
pages of one per-layer page pool, ``ENCODER_LEN = 1500`` rows are valid and
the padding rows ``1500..1535`` of each slot are never written. Here they are
poisoned, so a kernel that reads past ``cache_seqlens`` fails the comparison.
The page table points each request at its own slot's pages, the way the
runtime derives it from the request-pool index.

* Decode (one query row per request) and non-causal extend (1-8 query rows
  per request, every row sees all 1500 keys) match an fp32 reference over the
  request's own rows, for random slots including the first and last.
* After one warm-up sweep over bs 1..16 (what decode-graph capture does at
  startup), a second sweep compiles nothing new: no per-batch value reaches a
  Triton compile-time parameter. The decode stage-2 split count is a
  power-of-two bucket of the batch size, which the warm-up covers.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
)
from tokenspeed_kernel.ops.attention.mha._triton import decode as triton_decode
from tokenspeed_kernel.ops.attention.mha._triton import prefill as triton_prefill
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# whisper-large-v3 decoder cross-attention.
HEADS = 20
HEAD_DIM = 64
ENCODER_LEN = 1500
PAGE_SIZE = 64
PAGES_PER_SLOT = -(-ENCODER_LEN // PAGE_SIZE)
PADDED_LEN = PAGES_PER_SLOT * PAGE_SIZE
# max_num_seqs 16 -> 17 request-pool slots.
NUM_SLOTS = 17
SCALE = HEAD_DIM**-0.5
POISON = 100.0


def _cross_kv(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer's page pool ``[slots * pages_per_slot, page, heads, dim]``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (NUM_SLOTS, PADDED_LEN, HEADS, HEAD_DIM)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    k[:, ENCODER_LEN:] = POISON
    v[:, ENCODER_LEN:] = POISON
    pages = (NUM_SLOTS * PAGES_PER_SLOT, PAGE_SIZE, HEADS, HEAD_DIM)
    return k.view(pages), v.view(pages)


def _page_table(slots: list[int]) -> torch.Tensor:
    slot_t = torch.tensor(slots, dtype=torch.int32, device="cuda")
    offsets = torch.arange(PAGES_PER_SLOT, dtype=torch.int32, device="cuda")
    return slot_t[:, None] * PAGES_PER_SLOT + offsets


def _slots(bs: int, seed: int) -> list[int]:
    """``bs`` distinct slots; the first and last slot are always in the batch."""
    gen = torch.Generator().manual_seed(seed)
    rest = (torch.randperm(NUM_SLOTS - 2, generator=gen) + 1).tolist()
    slots = ([0, NUM_SLOTS - 1] + rest)[:bs]
    order = torch.randperm(bs, generator=gen).tolist()
    return [slots[i] for i in order]


def _reference(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    slots: list[int],
    q_lens: list[int],
) -> torch.Tensor:
    """fp32 non-causal attention of each request's rows over its own 1500 keys."""
    k_slots = k_pages.view(NUM_SLOTS, PADDED_LEN, HEADS, HEAD_DIM)
    v_slots = v_pages.view(NUM_SLOTS, PADDED_LEN, HEADS, HEAD_DIM)
    outs, start = [], 0
    for slot, q_len in zip(slots, q_lens):
        qb = q[start : start + q_len].float().transpose(0, 1)
        kb = k_slots[slot, :ENCODER_LEN].float().transpose(0, 1)
        vb = v_slots[slot, :ENCODER_LEN].float().transpose(0, 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, is_causal=False, scale=SCALE
        )
        outs.append(out.transpose(0, 1))
        start += q_len
    return torch.cat(outs)


def _queries(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Unit scale: logits ~N(0, 1) after 1/sqrt(dim). A peaked softmax makes the
    # kernels' bf16 rounding of P (a few ULPs of |V|) exceed a tight tolerance
    # without saying anything about which rows were read.
    return torch.randn(
        rows, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
    )


def _decode(q, k_pages, v_pages, page_table, solution=None):
    bs = page_table.shape[0]
    return mha_decode_with_kvcache(
        q,
        k_pages,
        v_pages,
        page_table=page_table,
        cache_seqlens=torch.full((bs,), ENCODER_LEN, dtype=torch.int32, device="cuda"),
        max_seqlen_k=PADDED_LEN,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        solution=solution,
    )


def _extend(q, k_pages, v_pages, page_table, q_lens, solution=None):
    bs = page_table.shape[0]
    cu_q = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    cu_q[1:] = torch.cumsum(torch.tensor(q_lens, device="cuda"), 0)
    cu_kv = torch.arange(
        0, (bs + 1) * ENCODER_LEN, ENCODER_LEN, dtype=torch.int32, device="cuda"
    )
    return mha_extend_with_kvcache(
        q,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        k_cache=k_pages,
        v_cache=v_pages,
        page_table=page_table,
        cache_seqlens=torch.full((bs,), ENCODER_LEN, dtype=torch.int32, device="cuda"),
        # The runtime passes the forward's row count, an upper bound.
        max_seqlen_q=int(q.shape[0]),
        max_seqlen_k=PADDED_LEN,
        is_causal=False,
        softmax_scale=SCALE,
        solution=solution,
    )


def _extend_lens(bs: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, 9, (bs,), generator=gen).tolist()
    # Both ends of the 1-8 range appear in every batch of two or more.
    lens[0] = 1
    if bs > 1:
        lens[-1] = 8
    return lens


@pytest.mark.parametrize("bs", [1, 2, 5, 8, 16])
@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_paged_decode_reads_each_requests_own_slot(bs, seed):
    k_pages, v_pages = _cross_kv(seed)
    slots = _slots(bs, seed)
    q = _queries(bs, seed + 100)
    out = _decode(q, k_pages, v_pages, _page_table(slots))
    ref = _reference(q, k_pages, v_pages, slots, [1] * bs)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 3, 8, 16])
@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_paged_noncausal_extend_reads_each_requests_own_slot(bs, seed):
    k_pages, v_pages = _cross_kv(seed)
    slots = _slots(bs, seed)
    q_lens = _extend_lens(bs, seed)
    q = _queries(sum(q_lens), seed + 100)
    out = _extend(q, k_pages, v_pages, _page_table(slots), q_lens)
    ref = _reference(q, k_pages, v_pages, slots, q_lens)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_cross_read_compiles_nothing_new_across_batch_sizes():
    """The registered Triton solution, pinned: it is what sm120 selects, and it
    is the portable one every vendor can run."""
    k_pages, v_pages = _cross_kv(0)

    def sweep(seed: int) -> None:
        for bs in range(1, 17):
            slots = _slots(bs, seed + bs)
            table = _page_table(slots)
            _decode(_queries(bs, bs), k_pages, v_pages, table, solution="triton")
            q_lens = _extend_lens(bs, seed + bs)
            _extend(
                _queries(sum(q_lens), bs),
                k_pages,
                v_pages,
                table,
                q_lens,
                solution="triton",
            )
        torch.cuda.synchronize()

    sweep(0)  # warm-up: every split-count bucket and integer class is reached
    with assert_no_triton_compile(
        triton_decode._fwd_kernel_stage1,
        triton_decode._fwd_kernel_stage2,
        triton_prefill._fwd_kernel,
    ):
        sweep(1000)
