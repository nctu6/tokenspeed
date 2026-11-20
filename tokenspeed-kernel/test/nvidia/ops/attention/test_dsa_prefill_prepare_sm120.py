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

"""The fused input preparation of the DeepGEMM DSA prefill leaf on SM120.

A tiny prefill chunk is bound by the host: the leaf used to issue about twenty
framework launches around a ~4 us scoring kernel. ``dsa_prefill_prepare`` does
the query quantization, the weight folding, the candidate counts and the key
gather in one launch, and the selector reports workspace rows itself. These
tests hold it to the framework ops it replaces, bit for bit, and to the
launch count.
"""

import math

import pytest
import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.thirdparty.deepgemm import (
    SM120_ARCH_MAJOR,
    deepgemm_supports_sm120,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not (
        Platform.get().is_nvidia
        and Platform.get().arch_version.major == SM120_ARCH_MAJOR
        and deepgemm_supports_sm120()
    ),
    reason="SM120 with a DeepGEMM build that has SM120 kernels only",
)

PAGE, DIM, TOPK = 64, 128, 2048


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


def _queries(tokens, heads, generator):
    q = torch.randn(tokens, heads, DIM, device="cuda", dtype=torch.bfloat16)
    # Rows the quantizer treats specially: all zero (scale floor), huge, tiny.
    q[0, 0] = 0
    if tokens > 1:
        q[1, heads - 1] = q[1, heads - 1] * 3e4
        q[tokens - 1, 0] = q[tokens - 1, 0] * 1e-6
    return q


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 7, 128, 300])
def test_prepare_equals_the_framework_ops_bit_for_bit(heads, weights_dtype, tokens):
    from tokenspeed_kernel.ops.attention.dsa._triton.prefill_prepare import (
        dsa_prefill_prepare,
    )
    from tokenspeed_kernel.ops.attention.dsa.deep_gemm import (
        _gather_index_rows,
        _pad_index_heads,
    )
    from tokenspeed_kernel.ops.attention.dsa.triton import combine_topk_weights
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    torch.manual_seed(heads * 1000 + tokens)
    num_keys = tokens + 3000
    cache = _index_cache(math.ceil(num_keys / PAGE))
    # Workspace rows are arbitrary cache slots, not an ascending run.
    slots = torch.randperm(num_keys, device="cuda")[: tokens + 500]
    q = _queries(tokens, heads, None)
    # A column split of a wider projection: a row stride other than `heads`.
    wide = torch.randn(tokens, heads + 40, device="cuda").to(weights_dtype)
    weights = wide[:, :heads]
    row_starts = torch.randint(0, 400, (tokens,), device="cuda", dtype=torch.int32)
    row_ends = row_starts + torch.randint(
        -3, 500, (tokens,), device="cuda", dtype=torch.int32
    )
    lens_out = torch.empty(tokens, device="cuda", dtype=torch.int32)
    scale = DIM**-0.5

    q_fp8, scaled_weights, candidate_lens, k_fp8, k_scale = dsa_prefill_prepare(
        q,
        weights,
        row_starts,
        row_ends,
        softmax_scale=scale,
        topk=TOPK,
        lens_out=lens_out,
        out_heads=32 if heads == 16 else heads,
        index_k_cache=cache,
        kv_workspace_slots=slots,
        page_size=PAGE,
    )

    padded_q, padded_weights = _pad_index_heads(q, weights)
    ref_fp8, ref_scale = quantize_fp8(
        padded_q.view(-1, DIM),
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
    )
    assert torch.equal(
        q_fp8.view(torch.uint8), ref_fp8.view_as(padded_q).view(torch.uint8)
    )
    assert torch.equal(
        scaled_weights, combine_topk_weights(padded_weights, ref_scale, scale)
    )
    reference_lens = (row_ends - row_starts).clamp_min(0)
    assert torch.equal(candidate_lens, reference_lens)
    assert torch.equal(lens_out, reference_lens.clamp_max(TOPK))
    ref_k, ref_k_scale = _gather_index_rows(cache, slots, head_dim=DIM, page_size=PAGE)
    assert torch.equal(k_fp8.view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(k_scale, ref_k_scale.squeeze(-1))


def _prefill(leaf, tokens, prefixes, *, heads, cache, seed):
    """One prefill call over requests holding `prefixes` tokens, `tokens` fresh each."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    starts, ends, lens = [], [], []
    base = 0
    for prefix in prefixes:
        starts += [base] * tokens
        ends += [base + prefix + 1 + i for i in range(tokens)]
        base += prefix + tokens
    total = len(prefixes) * tokens
    row_starts = torch.tensor(starts, device="cuda", dtype=torch.int32)
    row_ends = torch.tensor(ends, device="cuda", dtype=torch.int32)
    slots = torch.randperm(base, device="cuda", generator=generator)
    q = torch.randn(
        total, heads, DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights = torch.randn(total, heads, device="cuda", generator=generator)
    return leaf(
        q,
        weights,
        slots,
        row_starts,
        row_ends,
        topk=TOPK,
        softmax_scale=DIM**-0.5,
        index_k_cache=cache,
        page_size=PAGE,
        candidate_lens_cpu=(row_ends - row_starts).cpu(),
    )


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize(
    ("tokens", "prefixes"),
    [(128, [0]), (64, [0, 700]), (50, [3000, 0, 100]), (300, [2500])],
)
def test_leaf_selects_what_the_framework_ops_select(
    monkeypatch, heads, tokens, prefixes
):
    """Requests starting at nonzero workspace rows get absolute row ids."""
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    cache = _index_cache(math.ceil((sum(prefixes) + tokens * len(prefixes)) / PAGE))
    leaf = dsa_deep_gemm.deep_gemm_dsa_prefill_topk
    fused, fused_lens = _prefill(
        leaf, tokens, prefixes, heads=heads, cache=cache, seed=tokens
    )
    monkeypatch.setattr(dsa_deep_gemm, "_FUSED_PREFILL_PREPARE", False)
    plain, plain_lens = _prefill(
        leaf, tokens, prefixes, heads=heads, cache=cache, seed=tokens
    )
    assert torch.equal(fused_lens, plain_lens)
    assert torch.equal(fused, plain)
    # Row ids stay inside the request's own rows.
    assert (fused[fused >= 0] < sum(prefixes) + tokens * len(prefixes)).all()


@pytest.mark.parametrize("heads", [32, 64])
def test_logits_budget_tiles_select_what_one_call_selects(heads):
    """A budget that splits the chunk into several launches changes no row.

    A tile that covers the whole chunk takes the tensors as they are, a split
    one slices them; both must score and select every row the same way.
    """
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    tokens, prefix = 200, 900
    cache = _index_cache(math.ceil((prefix + tokens) / PAGE))
    leaf = dsa_deep_gemm.deep_gemm_dsa_prefill_topk
    whole = _prefill(leaf, tokens, [prefix], heads=heads, cache=cache, seed=5)

    def tiled(*args, **kwargs):
        # Tiles of 64, 64, 64 and 8 rows: 64 rows of fp32 logits per key.
        return leaf(*args, max_logits_bytes=(prefix + tokens) * 4 * 64, **kwargs)

    split = _prefill(tiled, tokens, [prefix], heads=heads, cache=cache, seed=5)
    assert torch.equal(whole[0], split[0])
    assert torch.equal(whole[1], split[1])


def test_tiny_chunk_launches_a_handful_of_kernels():
    """The host-bound regime: the leaf is a few launches, not a dozen small ones."""
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm

    cache = _index_cache(4)
    leaf = dsa_deep_gemm.deep_gemm_dsa_prefill_topk
    args = dict(tokens=128, prefixes=[0], heads=64, cache=cache, seed=1)
    _prefill(leaf, **args)  # compile
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        _prefill(leaf, **args)
        torch.cuda.synchronize()
    names = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    # Inputs of the call itself (tensor creation) are not the leaf's launches.
    leaf_kernels = [
        n
        for n in names
        if "prefill_prepare" in n
        or "mqa_logits" in n
        or "DeepSelect" in n
        or "nan_to_num" in n
    ]
    assert len(leaf_kernels) == 4, names


def test_prepare_does_not_recompile_for_new_chunk_sizes():
    from tokenspeed_kernel.ops.attention.dsa._triton import prefill_prepare

    cache = _index_cache(40)

    def run(tokens, rows):
        q = torch.randn(tokens, 64, DIM, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(tokens, 64, device="cuda")
        row_starts = torch.zeros(tokens, device="cuda", dtype=torch.int32)
        row_ends = torch.full((tokens,), rows, device="cuda", dtype=torch.int32)
        slots = torch.arange(rows, device="cuda")
        prefill_prepare.dsa_prefill_prepare(
            q,
            weights,
            row_starts,
            row_ends,
            softmax_scale=0.1,
            topk=TOPK,
            lens_out=torch.empty(tokens, device="cuda", dtype=torch.int32),
            out_heads=64,
            index_k_cache=cache,
            kv_workspace_slots=slots,
            page_size=PAGE,
        )

    run(128, 1024)
    torch.cuda.synchronize()
    with assert_no_triton_compile(prefill_prepare._dsa_prefill_prepare_kernel):
        for tokens, rows in [(1, 33), (16, 16), (17, 1025), (333, 2000), (2, 1)]:
            run(tokens, rows)
