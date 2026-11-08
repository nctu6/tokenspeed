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

"""Tensor-core sparse DSA attention (``triton_dsa_mma_*``) on SM120.

The leaves attend a dense BF16 or E4M3 latent cache with ``mma.sync``; the
reference is float32 attention over the same stored values.
"""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsa import torch_dsa_sparse_attention
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton
from tokenspeed_kernel.ops.attention.dsa._triton import sparse_mma
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.selection import KernelPlan
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, warm_specialization_classes

PAGE, RANK, ROPE = 64, 512, 64
DIM = RANK + ROPE
SCALE = 0.1352

# Output and LSE are checked against float32 attention over the same stored
# values, itself rounded to BF16. The tensor-core scores are exact products
# accumulated in FP32, and the LSE is exact FP32. An E4M3 cache multiplies the
# probabilities in FP16 (the cache values are exact there), so the output is
# off by at most one BF16 ulp (2**-7 relative). A BF16 cache multiplies BF16
# probabilities, which adds at most one more ulp and a floor near zero.
OUT_TOLERANCE = {
    torch.float8_e4m3fn: {"atol": 2.0e-3, "rtol": 2.0**-7},
    torch.bfloat16: {"atol": 4.0e-3, "rtol": 2.0**-6},
}
LSE_ATOL = 2.0e-5

on_sm120 = pytest.mark.skipif(
    not (Platform.get().is_nvidia and Platform.get().arch_version.major == 12),
    reason="SM120 required",
)


def _select(mode, platform, q_dtype, **traits):
    from tokenspeed_kernel.selection import select_kernel

    request = {
        "q_len": 1,
        "qk_nope_head_dim": 128,
        "kv_lora_rank": RANK,
        "qk_rope_head_dim": ROPE,
        "page_size": PAGE,
        "topk": 2048,
        "has_kv_cache": True,
        "has_sparse_kv_cache": False,
        "logit_cap": False,
        "return_lse": False,
        "topk_layout": "global_slots",
        "slot_order": "selection",
    }
    request.update(traits)
    return select_kernel(
        "attention",
        mode,
        format_signature(q=dense_tensor_format(q_dtype)),
        platform=platform,
        traits=request,
    ).name


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_selects_the_mma_leaves_for_dense_caches(
    q_dtype, sm120_platform: PlatformInfo
):
    assert _select("dsa_decode", sm120_platform, q_dtype) == "triton_dsa_mma_decode"
    assert (
        _select("dsa_decode", sm120_platform, q_dtype, q_len=4, return_lse=True)
        == "triton_dsa_mma_decode"
    )
    assert _select("dsa_prefill", sm120_platform, q_dtype) == "triton_dsa_mma_prefill"
    assert (
        _select("dsa_prefill", sm120_platform, q_dtype, topk=2051)
        == "triton_dsa_mma_prefill"
    )


def test_packed_caches_and_other_archs_keep_the_portable_leaves(
    sm120_platform: PlatformInfo,
    b200_platform: PlatformInfo,
    h100_platform: PlatformInfo,
):
    # The packed FP8 cache has no mma leaf; SM100 and SM90 never offer one.
    assert (
        _select(
            "dsa_decode",
            sm120_platform,
            torch.bfloat16,
            has_kv_cache=False,
            has_sparse_kv_cache=True,
        )
        == "triton_dsa_decode"
    )
    for platform in (b200_platform, h100_platform):
        assert (
            _select("dsa_prefill", platform, torch.bfloat16, has_kv_cache=True)
            != "triton_dsa_mma_prefill"
        )


def test_plan_follows_tokens_and_heads_only():
    sms = 188
    # One token of 128 heads: narrow tiles, split to the limit.
    assert sparse_mma.plan_launch(1, 128, sms) == (16, 16)
    # Enough tokens to fill the GPU: wide tiles, no split.
    assert sparse_mma.plan_launch(128, 128, sms) == (32, 1)
    # 16 heads never use the wide tile.
    assert sparse_mma.plan_launch(64, 16, sms)[0] == 16
    for tokens in (1, 3, 8, 9, 40, 512, 8192):
        for heads in (8, 16, 24, 32, 64, 128):
            block_h, splits = sparse_mma.plan_launch(tokens, heads, sms)
            assert block_h in (16, 32)
            assert 1 <= splits <= sparse_mma.MAX_SPLITS


def _plan(mode, platform, dtype, slot_order):
    """The plan DSABackend builds under a bitwise envelope."""
    traits = {
        "qk_nope_head_dim": 128,
        "kv_lora_rank": RANK,
        "qk_rope_head_dim": ROPE,
        "page_size": PAGE,
        "has_kv_cache": True,
        "has_sparse_kv_cache": False,
        "logit_cap": False,
        "topk_layout": "global_slots",
        "slot_order": slot_order,
        "return_lse": frozenset({False}),
        "q_len": frozenset({1, 2, 3}) if mode == "dsa_decode" else 1,
    }
    return KernelPlan.select(
        "attention",
        mode,
        format_signature(q=dense_tensor_format(dtype)),
        features=frozenset({"batch_invariant"}),
        static_traits=traits,
        variable_traits=frozenset({"batch_size", "tokens", "topk", "max_seqlen_k"}),
        solution=None,
        declared_traits=(
            frozenset({"slot_order"}) if slot_order == "sorted" else frozenset()
        ),
        ignore_layout=False,
        platform=platform,
    )


@pytest.mark.parametrize("slot_order", ["selection", "sorted"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_bitwise_envelope_plans_the_mma_leaves(
    dtype, slot_order, sm120_platform: PlatformInfo
):
    for mode, name in (
        ("dsa_decode", "triton_dsa_mma_decode"),
        ("dsa_prefill", "triton_dsa_mma_prefill"),
    ):
        plan = _plan(mode, sm120_platform, dtype, slot_order)
        assert plan.kernel.name == name
        assert {"batch_invariant", "cuda_graph"} <= plan.spec.features


def _make_cache(num_slots, dtype, generator):
    kv = torch.randn(num_slots, DIM, device="cuda", generator=generator) * 0.7
    return kv.to(dtype)


def _make_selection(lengths, width, num_slots, generator, holes=False):
    slots = torch.full((len(lengths), width), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths):
        chosen = torch.randperm(num_slots, device="cuda", generator=generator)[:length]
        slots[row, :length] = chosen.to(torch.int32)
        if holes and length > 4:
            # Slots another context-parallel rank owns come back as -1.
            slots[row, 1] = -1
            slots[row, length // 2] = -1
    return slots, torch.tensor(lengths, dtype=torch.int32, device="cuda")


def _reference(q, kv, slots, lens):
    """Float32 attention of each row over the stored values; returns (out, lse)."""
    flat = kv.float()
    out = torch_dsa_sparse_attention(
        q.float(), flat[:, :RANK], flat[:, RANK:], slots, lens, SCALE
    ).float()
    lse = torch.full(q.shape[:2], -float("inf"), device="cuda")
    for row in range(q.shape[0]):
        live = slots[row, : int(lens[row])].long()
        live = live[live >= 0]
        if live.numel():
            scores = (q[row].float() @ flat[live].T) * SCALE
            lse[row] = torch.logsumexp(scores, dim=-1)
    return out, lse


def _run(mode, q, kv, slots, lens, *, return_lse):
    call = dsa_decode if mode == "decode" else dsa_prefill
    extra = {"q_len_per_req": 1} if mode == "decode" else {}
    return call(
        q=q,
        kv_cache=kv.view(-1, PAGE, 1, DIM),
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=int(kv.shape[0]),
        qk_nope_head_dim=128,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        softmax_scale=SCALE,
        page_size=PAGE,
        return_lse=return_lse,
        solution="triton",
        slot_order="selection",
        **extra,
    )


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    ("heads", "lengths", "width"),
    [
        (128, [2048], 2048),  # one token: split to fill the GPU
        (16, [2048, 700, 31, 1, 0], 2048),  # short, single and empty rows
        (24, [2048] * 9, 2051),  # head tail tile, padded width
        (32, [1500] * 40, 2048),  # wide tiles, no split
        (64, [512, 100, 33], 512),
    ],
)
def test_matches_float32_reference(mode, dtype, heads, lengths, width):
    generator = torch.Generator(device="cuda").manual_seed(heads + len(lengths))
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, dtype, generator)
    q = (
        torch.randn(len(lengths), heads, DIM, device="cuda", generator=generator) * 1.5
    ).to(dtype)
    slots, lens = _make_selection(lengths, width, num_slots, generator, holes=True)
    out, lse = _run(mode, q, kv, slots, lens, return_lse=True)
    expected, expected_lse = _reference(q, kv, slots, lens)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, **OUT_TOLERANCE[dtype])
    live = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(lse), live)
    torch.testing.assert_close(lse[live], expected_lse[live], atol=LSE_ATOL, rtol=1e-6)
    # Rows with nothing live attend nothing.
    assert not out[~live.any(dim=1)].any()


def _attend(q, kv, slots, lens, *, num_splits, block_h=32, return_lse=True):
    return sparse_mma.sparse_mma_attention(
        q,
        kv.view(-1, PAGE, 1, DIM),
        slots,
        lens,
        softmax_scale=SCALE,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        return_lse=return_lse,
        block_h=block_h,
        block_n=sparse_mma.BLOCK_N,
        num_warps=sparse_mma.NUM_WARPS,
        num_stages=sparse_mma.NUM_STAGES,
        num_splits=num_splits,
    )


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("block_h", [16, 32])
def test_split_count_does_not_change_the_result(dtype, block_h):
    """Folding the chunks in one program or across several is bitwise equal."""
    generator = torch.Generator(device="cuda").manual_seed(5)
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, dtype, generator)
    q = torch.randn(4, 40, DIM, device="cuda", generator=generator).to(dtype)
    # Full, partial, one-tile and empty selections.
    slots, lens = _make_selection([2048, 1000, 77, 0], 2048, num_slots, generator)
    out, lse = _attend(q, kv, slots, lens, num_splits=1, block_h=block_h)
    for splits in (2, 3, 5, sparse_mma.MAX_CHUNKS, 99):
        other, other_lse = _attend(
            q, kv, slots, lens, num_splits=splits, block_h=block_h
        )
        assert torch.equal(out, other), splits
        assert torch.equal(lse, other_lse), splits


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("heads", [16, 40, 128])
def test_rows_do_not_depend_on_the_batch(mode, dtype, heads):
    """A token's row is bitwise the same in any batch, split or not.

    Replaying a chunk after a prefix-cache hit, or decoding next to other
    requests, must reproduce the rows exactly; the launch shape follows the
    token count.
    """
    generator = torch.Generator(device="cuda").manual_seed(3 + heads)
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, dtype, generator)
    count = 600
    q = torch.randn(count, heads, DIM, device="cuda", generator=generator).to(dtype)
    lengths = [2048, 1999, 1024, 513, 200, 33, 1, 0] * (count // 8) + [2048] * (
        count % 8
    )
    slots, lens = _make_selection(lengths, 2048, num_slots, generator, holes=True)
    full, full_lse = _run(mode, q, kv, slots, lens, return_lse=True)
    for start, size in ((0, 1), (5, 3), (17, 9), (100, 40), (0, count)):
        rows = slice(start, start + size)
        part, part_lse = _run(
            mode, q[rows], kv, slots[rows], lens[rows], return_lse=True
        )
        assert torch.equal(part, full[rows]), (start, size)
        assert torch.equal(part_lse, full_lse[rows]), (start, size)
    sm_count = Platform.get().sm_count
    plans = {
        sparse_mma.plan_launch(n, heads, sm_count)[1] > 1 for n in (1, 3, 9, count)
    }
    assert plans == {True, False}  # both launch shapes were compared


@on_sm120
def test_out_buffer_and_flat_cache_layouts():
    generator = torch.Generator(device="cuda").manual_seed(9)
    num_slots = 4 * PAGE * 16
    kv = _make_cache(num_slots, torch.bfloat16, generator)
    q = torch.randn(5, 16, DIM, device="cuda", generator=generator).bfloat16()
    slots, lens = _make_selection([300] * 5, 512, num_slots, generator)
    expected, _ = _reference(q, kv, slots, lens)
    for cache in (kv, kv.view(-1, PAGE, DIM), kv.view(-1, PAGE, 1, DIM)):
        buffer = torch.empty(5, 16, RANK, dtype=torch.bfloat16, device="cuda")
        returned = dsa_triton.triton_dsa_mma_decode(
            q,
            cache,
            None,
            slots,
            lens,
            num_slots,
            128,
            RANK,
            ROPE,
            SCALE,
            PAGE,
            out=buffer,
            slot_order="selection",
        )
        assert returned is buffer
        torch.testing.assert_close(
            buffer.float(), expected, **OUT_TOLERANCE[torch.bfloat16]
        )


@on_sm120
def test_batch_size_does_not_recompile():
    """Token count, head count, top-k width and chunk count are runtime values."""
    generator = torch.Generator(device="cuda").manual_seed(11)
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, torch.float8_e4m3fn, generator)
    pool = torch.randn(130, 32, DIM, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    slots, lens = _make_selection([1900] * 130, 2048, num_slots, generator)

    def run(tokens):
        block_h, splits = sparse_mma.plan_launch(tokens, 32, 188)
        return _attend(
            pool[:tokens],
            kv,
            slots[:tokens],
            lens[:tokens],
            num_splits=splits,
            block_h=block_h,
            return_lse=False,
        )

    def key(tokens):
        # The binaries are keyed by the head-tile width and split-or-not;
        # tokens and the split count only change the launch grid.
        block_h, splits = sparse_mma.plan_launch(tokens, 32, 188)
        return block_h, splits > 1

    sweep = (1, 2, 3, 5, 8, 9, 17, 33, 64, 100, 130)
    warm_specialization_classes(run, key, sweep, pool=(1, 4, 8, 16, 40, 130))
    with assert_no_triton_compile(
        sparse_mma._sparse_mma_kernel, sparse_mma._sparse_mma_combine_kernel
    ):
        for tokens in sweep:
            run(tokens)


def test_sm120_selects_the_mma_leaves_for_the_zero_rope_kpool_layout(
    sm120_platform: PlatformInfo,
):
    # GLM-5.3-Flash: nope 256, no RoPE rows, 2048 selected slots plus a tail.
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        decode = dict(qk_nope_head_dim=256, qk_rope_head_dim=0, topk=2051)
        assert (
            _select("dsa_decode", sm120_platform, dtype, q_len=4, **decode)
            == "triton_dsa_mma_decode"
        )
        assert (
            _select("dsa_prefill", sm120_platform, dtype, **decode)
            == "triton_dsa_mma_prefill"
        )


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    ("heads", "lengths", "width"),
    [(16, [2051, 2048, 700, 0], 2051), (16, [2048] * 40, 2048), (64, [2051], 2051)],
)
def test_zero_rope_kpool_layout_matches_float32_reference(
    mode, dtype, heads, lengths, width
):
    """Every cache row is the 512-wide latent: key and value are the same row."""
    generator = torch.Generator(device="cuda").manual_seed(heads + len(lengths))
    num_slots = 8 * PAGE * 16
    kv = (torch.randn(num_slots, RANK, device="cuda", generator=generator) * 0.7).to(
        dtype
    )
    q = (
        torch.randn(len(lengths), heads, RANK, device="cuda", generator=generator) * 1.5
    ).to(dtype)
    slots, lens = _make_selection(lengths, width, num_slots, generator, holes=True)
    call = dsa_decode if mode == "decode" else dsa_prefill
    out, lse = call(
        q=q,
        kv_cache=kv.view(-1, PAGE, 1, RANK),
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=256,
        kv_lora_rank=RANK,
        qk_rope_head_dim=0,
        softmax_scale=SCALE,
        page_size=PAGE,
        return_lse=True,
        solution="triton",
        slot_order="selection",
        **({"q_len_per_req": 1} if mode == "decode" else {}),
    )
    flat = kv.float()
    expected = torch.zeros(len(lengths), heads, RANK, device="cuda")
    expected_lse = torch.full((len(lengths), heads), -float("inf"), device="cuda")
    for row in range(len(lengths)):
        live = slots[row, : int(lens[row])].long()
        live = live[live >= 0]
        if live.numel():
            scores = (q[row].float() @ flat[live].T) * SCALE
            expected[row] = scores.softmax(-1) @ flat[live]
            expected_lse[row] = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out.float(), expected, **OUT_TOLERANCE[dtype])
    live_rows = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(lse), live_rows)
    torch.testing.assert_close(
        lse[live_rows], expected_lse[live_rows], atol=LSE_ATOL, rtol=1e-6
    )


@on_sm120
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_planned_call_matches_the_unplanned_call(mode):
    """A plan fixes the leaf; the call through it computes the same bits."""
    generator = torch.Generator(device="cuda").manual_seed(21)
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, torch.bfloat16, generator)
    q = torch.randn(7, 24, DIM, device="cuda", generator=generator).bfloat16()
    slots, lens = _make_selection(
        [2048, 900, 64, 5, 0, 1500, 2048], 2048, num_slots, generator
    )
    plan = _plan(f"dsa_{mode}", Platform.get(), torch.bfloat16, "sorted")
    call = dsa_decode if mode == "decode" else dsa_prefill
    extra = {"q_len_per_req": 1} if mode == "decode" else {}
    planned = call(
        q=q,
        kv_cache=kv.view(-1, PAGE, 1, DIM),
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=128,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        softmax_scale=SCALE,
        page_size=PAGE,
        slot_order="sorted",
        kernel_plan=plan,
        **extra,
    )
    unplanned = _run(mode, q, kv, slots, lens, return_lse=False)
    assert torch.equal(planned, unplanned)


@on_sm120
@pytest.mark.parametrize("width", [1, 33, 64, 100, 512, 4096])
def test_any_selection_width_matches_the_float32_reference(width):
    """The leaves carry no width trait, so any width must be correct."""
    generator = torch.Generator(device="cuda").manual_seed(width)
    num_slots = 8 * PAGE * 16
    kv = _make_cache(num_slots, torch.bfloat16, generator)
    q = torch.randn(3, 16, DIM, device="cuda", generator=generator).bfloat16()
    lengths = [width, max(width // 2, 1), 1]
    slots, lens = _make_selection(lengths, width, num_slots, generator)
    out, lse = _run("decode", q, kv, slots, lens, return_lse=True)
    expected, expected_lse = _reference(q, kv, slots, lens)
    torch.testing.assert_close(out.float(), expected, **OUT_TOLERANCE[torch.bfloat16])
    torch.testing.assert_close(lse, expected_lse, atol=LSE_ATOL, rtol=1e-6)
