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

"""A MIXED round of the paged MHA backend with speculative verify rows.

The decode requests of a MIXED round carry their whole verify window
(``spec_num_tokens`` rows each), behind the prefill rows' chunks. Through the
real backend, layer and kernels, the mixed forward must equal the prefill
chunk run as an EXTEND forward and the verify windows run as a DECODE
forward, for the target and for a chain drafter's first step."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

KV_HEADS, Q_HEADS, HEAD_DIM, PAGE_SIZE = 2, 8, 64, 64
MAX_BS, CONTEXT = 4, 1 << 13
WIDTH = 3  # speculative_num_draft_tokens: the anchor and two drafts
# One prefill request (cached prefix + chunk), then the decode requests'
# cache lengths, verify window included.
PREFIX, CHUNK = 2 * PAGE_SIZE + 7, 150
DECODE_LENGTHS = [5 * PAGE_SIZE + 3, 40, 2 * PAGE_SIZE]


@pytest.fixture(scope="module", autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("the paged MHA kernels need a GPU")


class _Cache:
    """KV storage only; the backend and the kernels stay real."""

    def __init__(self, generator: torch.Generator):
        pages = MAX_BS * CONTEXT // PAGE_SIZE
        self.key, self.value = (
            torch.randn(
                pages * PAGE_SIZE,
                KV_HEADS,
                HEAD_DIM,
                generator=generator,
                device="cuda",
            ).bfloat16()
            for _ in range(2)
        )

    def get_key_buffer(self, layer_id):
        return self.key

    def get_value_buffer(self, layer_id):
        return self.value


def _backend(is_draft: bool):
    from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=CONTEXT,
        max_bs=MAX_BS,
        max_resident_requests=MAX_BS,
        speculative_num_draft_tokens=WIDTH,
        is_draft=is_draft,
        draft_block_decode=False,
        components=(spec,),
    )
    backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE_SIZE)
    backend.init_cuda_graph_state(MAX_BS)
    return backend


def _init_extend(backend, mode, seq_lens, page_table, num_extends):
    bs = seq_lens.shape[0]
    extend_lens = torch.tensor([CHUNK], dtype=torch.int32)
    prefix_lens = torch.tensor([PREFIX], dtype=torch.int32)
    backend.init_forward_metadata(
        bs,
        num_extends,
        seq_lens,
        page_table[:bs],
        mode,
        extend_seq_lens=extend_lens.cuda(),
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens=prefix_lens.cuda(),
        extend_prefix_lens_cpu=prefix_lens,
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )


@pytest.mark.parametrize("is_draft", [False, True], ids=["target", "draft"])
def test_mixed_round_equals_extend_and_verify_rounds(is_draft):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    generator = torch.Generator(device="cuda").manual_seed(3)
    cache = _Cache(generator)
    layer = PagedAttention(
        Q_HEADS,
        HEAD_DIM,
        HEAD_DIM**-0.5,
        KV_HEADS,
        0,
        sliding_window_size=-1,
        rotary_emb=None,
        qk_norm=None,
    )
    pages = CONTEXT // PAGE_SIZE
    page_table = (
        torch.randperm(MAX_BS * pages, generator=generator, device="cuda")
        .to(torch.int32)
        .view(MAX_BS, pages)
    )
    num_decodes = len(DECODE_LENGTHS)
    rows = CHUNK + num_decodes * WIDTH
    q, k, v = (
        torch.randn(
            rows, heads * HEAD_DIM, generator=generator, device="cuda"
        ).bfloat16()
        for heads in (Q_HEADS, KV_HEADS, KV_HEADS)
    )
    seq_lens = torch.tensor(
        [PREFIX + CHUNK, *DECODE_LENGTHS], dtype=torch.int32, device="cuda"
    )

    # Reference: the prefill chunk as an EXTEND, the verify windows as a DECODE.
    backend = _backend(is_draft)
    _init_extend(backend, ForwardMode.EXTEND, seq_lens[:1], page_table, 1)
    extend_out = backend.forward_extend(
        q[:CHUNK], k[:CHUNK], v[:CHUNK], layer, None, cache, 1
    )
    backend.refresh_decode_metadata(
        num_decodes, num_decodes, seq_lens[1:], page_table[1 : 1 + num_decodes]
    )
    verify_out = backend.forward_decode(
        q[CHUNK:], None, None, layer, None, cache, num_decodes
    )

    # Candidate: both in one MIXED round, the extend request first.
    backend = _backend(is_draft)
    _init_extend(backend, ForwardMode.MIXED, seq_lens, page_table, 1)
    mixed = backend.forward_extend(q, k, v, layer, None, cache, 1 + num_decodes)

    assert tuple(mixed.shape) == (rows, Q_HEADS * HEAD_DIM)
    torch.testing.assert_close(mixed[:CHUNK], extend_out, atol=2**-8, rtol=2**-7)
    torch.testing.assert_close(mixed[CHUNK:], verify_out, atol=0, rtol=0)
