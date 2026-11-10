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
"""Tensor-core DeepSeek V4 selected attention on SM120.

Two leaves attend a BF16 workspace of 512-wide latent rows, with an optional
per-head sink: the Triton ``triton_dsv4_mma_prefill`` and the CuTe DSL
``cute_dsl_dsv4_mma_prefill``, which the registry prefers where it is built.
The reference is float32 attention over the same stored values, rounded to
BF16.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill
from tokenspeed_kernel.ops.attention.dsv4._triton import sparse_mma
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, warm_specialization_classes

DIM = 512
SCALE = 0.0442
TRITON_LEAF = "triton_dsv4_mma_prefill"
CUTE_LEAF = "cute_dsl_dsv4_mma_prefill"
# The CuTe DSL leaf registers only where the CuTe DSL imports.
HAS_CUTE = KernelRegistry.get().get_by_name(CUTE_LEAF) is not None
LEAVES = [TRITON_LEAF, CUTE_LEAF] if HAS_CUTE else [TRITON_LEAF]
LEAF = CUTE_LEAF if HAS_CUTE else TRITON_LEAF
if HAS_CUTE:
    from tokenspeed_kernel.ops.attention.dsv4._cute_dsl import (
        sparse_mma as cute_sparse_mma,
    )
on_cute = pytest.mark.skipif(not HAS_CUTE, reason="CuTe DSL leaf not built")

on_sm120 = pytest.mark.skipif(
    not (Platform.get().is_nvidia and Platform.get().arch_version.major == 12),
    reason="SM120 required",
)

# Scores are exact BF16 products accumulated in FP32 and the LSE is exact FP32.
# The output multiplies BF16-rounded probabilities, so it is within one BF16
# ulp (2**-7 relative) of the float32 result, plus a floor near zero.
OUT_TOLERANCE = {"atol": 4.0e-3, "rtol": 2.0**-6}
LSE_ATOL = 2.0e-5


def _inputs(tokens, heads, rows, width, index_dtype, *, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(tokens, heads, DIM, generator=generator, device="cuda") * 0.8
    kv = torch.randn(rows, 1, DIM, generator=generator, device="cuda") * 0.8
    indices = torch.randint(
        0, rows, (tokens, width), generator=generator, device="cuda"
    )
    # Dead columns, an out-of-range row and a short prefix are all skipped.
    indices[torch.rand(tokens, width, generator=generator, device="cuda") < 0.15] = -1
    indices[0, 0] = rows + 5
    lens = torch.randint(0, width + 1, (tokens,), generator=generator, device="cuda")
    lens[-1] = width
    return (
        q.to(torch.bfloat16),
        kv.to(torch.bfloat16),
        indices.to(index_dtype),
        lens.to(index_dtype),
        torch.randn(heads, generator=generator, device="cuda"),
    )


def _reference(q, kv, indices, lens, sink):
    """Float32 ``(out, lse)``; the LSE includes the sink when there is one."""
    tokens, heads, _ = q.shape
    rows = kv.reshape(-1, DIM).float()
    out = torch.zeros(tokens, heads, DIM, device="cuda")
    lse = torch.full((tokens, heads), float("-inf"), device="cuda")
    for token in range(tokens):
        chosen = indices[token, : int(lens[token])].long()
        chosen = chosen[(chosen >= 0) & (chosen < rows.shape[0])]
        keys = rows[chosen]
        logits = (q[token].float() @ keys.T) * SCALE
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=1)
        if logits.shape[1] == 0:
            continue
        top = logits.max(1, keepdim=True).values
        weights = torch.exp(logits - top)
        total = weights.sum(1, keepdim=True)
        out[token] = (weights[:, : keys.shape[0]] @ keys) / total
        lse[token] = (top + total.log()).squeeze(1)
    return out, lse


def _select(platform, **traits):
    request = {
        "num_q_heads": 64,
        "head_dim": DIM,
        "selected_width": 640,
        "cache_layout": "dense_workspace",
        "metadata_dtypes": frozenset({torch.int32}),
        "sinks": True,
    }
    request.update(traits)
    return select_kernel(
        "attention",
        "dsv4_prefill",
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv=dense_tensor_format(torch.bfloat16),
        ),
        platform=platform,
        traits=request,
    ).name


def test_sm120_selects_the_mma_leaf(sm120_platform: PlatformInfo):
    assert _select(sm120_platform) == LEAF
    assert _select(sm120_platform, num_q_heads=32, selected_width=128) == LEAF
    assert _select(sm120_platform, sinks=False, return_lse=True) == LEAF
    assert _select(sm120_platform, metadata_dtypes=frozenset({torch.int64})) == LEAF
    # Up to 16 heads the 4-warp CuTe CTA loses to the Triton leaf; the rule
    # follows the head count, so one model never mixes the two.
    assert _select(sm120_platform, num_q_heads=16, selected_width=128) == TRITON_LEAF
    assert _select(sm120_platform, num_q_heads=8) == TRITON_LEAF


def test_other_archs_keep_flashmla(
    b200_platform: PlatformInfo, h100_platform: PlatformInfo
):
    for platform in (b200_platform, h100_platform):
        assert _select(platform) == "flashmla_dsv4_prefill"


@on_sm120
@pytest.mark.parametrize("leaf", LEAVES)
@pytest.mark.parametrize(
    ("tokens", "heads", "rows", "width", "sink", "with_lse", "index_dtype"),
    [
        (1, 64, 700, 640, True, False, torch.int32),
        (5, 64, 3000, 640, True, False, torch.int32),
        (33, 32, 2000, 128, True, False, torch.int64),
        (9, 16, 500, 640, False, True, torch.int32),
        (130, 64, 4000, 576, True, False, torch.int32),
        (2, 128, 5000, 1280, True, False, torch.int32),
        (3, 12, 300, 77, False, True, torch.int32),
        (40, 24, 900, 1100, True, False, torch.int64),  # sink, wide ids, odd width
    ],
    ids=lambda v: str(v),
)
def test_matches_the_float32_reference(
    leaf, tokens, heads, rows, width, sink, with_lse, index_dtype
):
    q, kv, indices, lens, sink_logits = _inputs(tokens, heads, rows, width, index_dtype)
    sink_logits = sink_logits if sink else None
    lse = torch.empty(tokens, heads, device="cuda") if with_lse else None
    out = dsv4_prefill(q, kv, indices, lens, sink_logits, SCALE, lse=lse, override=leaf)
    ref_out, ref_lse = _reference(q, kv, indices, lens, sink_logits)
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    torch.testing.assert_close(out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE)
    if with_lse:
        assert torch.equal(torch.isinf(lse), torch.isinf(ref_lse))
        finite = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=LSE_ATOL, rtol=0)


@on_sm120
@pytest.mark.parametrize("leaf", LEAVES)
def test_strided_query_and_output(leaf):
    q, kv, indices, lens, sink = _inputs(6, 16, 400, 200, torch.int32)
    wide_q = torch.zeros(6, 16, DIM + 64, dtype=torch.bfloat16, device="cuda")
    wide_q[..., :DIM] = q
    wide_out = torch.zeros_like(wide_q)
    out = dsv4_prefill(
        wide_q[..., :DIM], kv, indices, lens, sink, SCALE, out=wide_out[..., :DIM],
        override=leaf,
    )  # fmt: skip
    assert out.data_ptr() == wide_out.data_ptr()
    ref_out, _ = _reference(q, kv, indices, lens, sink)
    torch.testing.assert_close(out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE)
    assert not wide_out[..., DIM:].any()


@on_sm120
def test_token_count_and_width_do_not_recompile():
    sm_count = Platform.get().sm_count
    heads = 64

    def run(tokens, width=640):
        q, kv, indices, lens, sink = _inputs(
            tokens, heads, 1000, width, torch.int32, seed=tokens
        )
        out = dsv4_prefill(q, kv, indices, lens, sink, SCALE, override=LEAF)
        assert out.shape == q.shape

    def key(tokens):
        # Binaries are keyed by the head-tile width and split-or-not only.
        block_h, splits = sparse_mma.plan_launch(tokens, heads, sm_count)
        return block_h, 2 * splits >= sparse_mma.num_chunks_for(640)[1]

    sweep = (1, 2, 3, 5, 8, 9, 17, 33, 64, 100, 130, 515)
    warm_specialization_classes(run, key, sweep, pool=(1, 4, 8, 16, 40, 130, 515))
    # The selection width follows the layer (window only, or window + top-k).
    run(7, width=128)
    run(7, width=592)
    with assert_no_triton_compile(
        sparse_mma._dsv4_sparse_mma_kernel, sparse_mma._dsv4_sparse_mma_combine_kernel
    ):
        for tokens in sweep:
            run(tokens)
        run(11, width=384)
        run(3, width=1408)
        run(5, width=577)  # a row stride that is not a multiple of 16


@on_sm120
@on_cute
@pytest.mark.parametrize("tokens", [1, 3, 600])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cute_sink_with_lse_and_rows_that_attend_nothing(tokens, index_dtype):
    """The sink is chunk 0's initial state: a row with no live key returns the
    sink-only state (zero output, LSE = sink logit), also when the launch is
    split, and the LSE with a sink includes it. The facade does not combine a
    sink with an LSE, so this calls the core directly."""
    heads, width = 40, 640
    q, kv, indices, lens, sink = _inputs(tokens, heads, 900, width, index_dtype)
    lens[::3] = 0
    indices[1::4] = -1
    out = torch.empty_like(q)
    lse = torch.empty(tokens, heads, device="cuda")
    block_h, cwarps, splits, full_tokens = cute_sparse_mma.plan_launch(
        tokens, heads, Platform.get().sm_count, width
    )
    cute_sparse_mma.sparse_mma_attention(
        q, kv.reshape(-1, DIM), indices, lens, sink.contiguous(), out, lse,
        softmax_scale=SCALE, block_h=block_h, cwarps=cwarps, num_splits=splits,
        full_tokens=full_tokens,
    )  # fmt: skip
    ref_out, ref_lse = _reference(q, kv, indices, lens, sink)
    torch.testing.assert_close(out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE)
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=0)
    empty = lens == 0
    assert not out[empty].any()
    torch.testing.assert_close(
        lse[empty], sink.expand_as(lse[empty]), atol=LSE_ATOL, rtol=0
    )


@on_sm120
@on_cute
def test_cute_binaries_are_keyed_by_head_tile_and_metadata_only():
    """Token count, width, strides, rows and split count are runtime values:
    a sweep over them compiles nothing once each (head tile, metadata dtype,
    sink) binary exists."""
    heads = 64
    sweep = (1, 2, 3, 5, 8, 9, 17, 33, 64, 100, 130, 515)

    def run(tokens, width=640, index_dtype=torch.int32, rows=1000):
        q, kv, indices, lens, sink = _inputs(
            tokens, heads, rows, width, index_dtype, seed=tokens
        )
        out = dsv4_prefill(q, kv, indices, lens, sink, SCALE, override=CUTE_LEAF)
        assert out.shape == q.shape

    for tokens in sweep:  # every head tile and split class
        run(tokens)
    run(7, width=128)
    run(7, width=592)
    for index_dtype in (torch.int32, torch.int64):
        run(5, index_dtype=index_dtype)
    compiled = cute_sparse_mma._core.compiled_attention
    combine = cute_sparse_mma._core.compiled_combine
    before = compiled.cache_info().misses, combine.cache_info().misses
    for tokens in sweep:
        run(tokens)
    run(11, width=384, rows=7000)
    run(3, width=1408)
    run(5, width=577)  # a row stride that is not a multiple of 16
    run(40, index_dtype=torch.int64, rows=300)
    assert (compiled.cache_info().misses, combine.cache_info().misses) == before


@on_sm120
@pytest.mark.parametrize("leaf", LEAVES)
@pytest.mark.parametrize("heads", [2, 16, 64])
def test_rows_do_not_depend_on_the_batch(leaf, heads):
    """A token's row is bitwise the same in any batch, split or not.

    A prefill chunk replayed after a prefix-cache hit must reproduce the rows
    of the original chunk exactly; the launch shape follows the token count.
    """
    q, kv, indices, lens, sink = _inputs(600, heads, 5000, 640, torch.int32, seed=3)
    full = dsv4_prefill(q, kv, indices, lens, sink, SCALE, override=leaf)
    for start, count in ((0, 1), (5, 4), (17, 72), (100, 200), (0, 600)):
        rows = slice(start, start + count)
        part = dsv4_prefill(
            q[rows], kv, indices[rows], lens[rows], sink, SCALE, override=leaf
        )
        assert torch.equal(part, full[rows]), (start, count)
    sm_count = Platform.get().sm_count
    chunks = sparse_mma.num_chunks_for(640)[1]
    if leaf == CUTE_LEAF:
        # One token splits every program; 600 tokens run whole CTAs (with a
        # stream-K tail of split ones for some head counts).
        plans = {
            cute_sparse_mma.plan_launch(n, heads, sm_count, 640)[3] == 0
            for n in (1, 600)
        }
    else:
        plans = {
            2 * sparse_mma.plan_launch(n, heads, sm_count)[1] >= chunks
            for n in (1, 600)
        }
    assert plans == {True, False}  # both launch shapes were compared


def _cute_rows(q, kv, indices, lens, sink, *, block_h, cwarps, splits, full_tokens):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], device="cuda")
    cute_sparse_mma.sparse_mma_attention(
        q, kv.reshape(-1, DIM), indices, lens, sink, out, lse,
        softmax_scale=SCALE, block_h=block_h, cwarps=cwarps, num_splits=splits,
        full_tokens=full_tokens,
    )  # fmt: skip
    return out, lse


@on_sm120
@on_cute
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cute_rows_do_not_depend_on_the_launch_shape(index_dtype):
    """Rows and LSE are bitwise the same for every head tile, warp layout
    (a 16-head CTA in 4 or 8 warps, or two 4-warp groups), split count and
    stream-K mix of whole and split tokens."""
    tokens, heads = 37, 40
    q, kv, indices, lens, sink = _inputs(tokens, heads, 3000, 640, index_dtype, seed=11)
    sink = sink.contiguous()
    shapes = [(16, 4), (16, 8), (32, 4)]
    reference = _cute_rows(
        q, kv, indices, lens, sink, block_h=32, cwarps=4, splits=1, full_tokens=tokens
    )
    for block_h, cwarps in shapes:
        for splits in (1, 2, 4):
            for full_tokens in (tokens, 0, 1, 20, tokens - 1):
                out, lse = _cute_rows(
                    q, kv, indices, lens, sink,
                    block_h=block_h, cwarps=cwarps, splits=splits, full_tokens=full_tokens,
                )  # fmt: skip
                assert torch.equal(out, reference[0]), (
                    block_h,
                    cwarps,
                    splits,
                    full_tokens,
                )
                assert torch.equal(lse, reference[1]), (
                    block_h,
                    cwarps,
                    splits,
                    full_tokens,
                )


@on_sm120
@on_cute
@pytest.mark.parametrize("heads", [1, 8, 16])
def test_cute_eight_warp_cta_with_partial_head_tiles(heads):
    """A 16-head CTA of 8 warps with fewer live heads than its tile still
    matches the 4-warp CTA bit for bit and the float32 reference."""
    q, kv, indices, lens, sink = _inputs(9, heads, 700, 300, torch.int32, seed=5)
    four = _cute_rows(
        q, kv, indices, lens, sink, block_h=16, cwarps=4, splits=1, full_tokens=9
    )
    eight = _cute_rows(
        q, kv, indices, lens, sink, block_h=16, cwarps=8, splits=2, full_tokens=3
    )
    assert torch.equal(four[0], eight[0]) and torch.equal(four[1], eight[1])
    ref_out, _ = _reference(q, kv, indices, lens, sink)
    torch.testing.assert_close(eight[0], ref_out.to(torch.bfloat16), **OUT_TOLERANCE)


@on_cute
@pytest.mark.parametrize("width", [128, 640, 1152])
@pytest.mark.parametrize("heads", [16, 17, 40, 128])
def test_cute_plan_is_a_valid_launch(heads, width):
    chunks = cute_sparse_mma.num_chunks_for(width)[1]
    for tokens in range(1, 700):
        block_h, cwarps, splits, full_tokens = cute_sparse_mma.plan_launch(
            tokens, heads, 188, width
        )
        assert block_h in (16, 32) and (heads > 16 or block_h == 16)
        assert cwarps == (8 if block_h == 16 else 4)
        assert 1 <= splits <= chunks and 0 <= full_tokens <= tokens
        assert splits > 1 or full_tokens == tokens
        # The grid keeps one row per whole token and ``splits`` per split token.
        assert full_tokens + (tokens - full_tokens) * splits <= 65535


@on_cute
def test_cute_plan_fills_the_last_wave_with_split_programs():
    """192 programs on 188 SMs: 188 whole CTAs and the other 4 split four ways
    run in one wave and a short tail, not in two whole-CTA waves."""
    assert cute_sparse_mma.plan_launch(192, 32, 188, 640) == (32, 4, 4, 188)
    # A batch that fills the SMs with whole CTAs, and one far below, are unsplit
    # and split by the uniform rule.
    assert cute_sparse_mma.plan_launch(188, 32, 188, 640)[2:] == (1, 188)
    assert cute_sparse_mma.plan_launch(1, 64, 188, 640) == (16, 8, 4, 0)
