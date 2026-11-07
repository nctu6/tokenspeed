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
"""Tensor-core DeepSeek V4 selected attention (``triton_dsv4_mma_prefill``).

The leaf attends a BF16 workspace of 512-wide latent rows, with an optional
per-head sink. The reference is float32 attention over the same stored values,
rounded to BF16.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill
from tokenspeed_kernel.ops.attention.dsv4._triton import sparse_mma
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, warm_specialization_classes

DIM = 512
SCALE = 0.0442
LEAF = "triton_dsv4_mma_prefill"

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
    assert _select(sm120_platform, num_q_heads=16, selected_width=128) == LEAF
    assert _select(sm120_platform, sinks=False, return_lse=True) == LEAF
    assert _select(sm120_platform, metadata_dtypes=frozenset({torch.int64})) == LEAF


def test_other_archs_keep_flashmla(
    b200_platform: PlatformInfo, h100_platform: PlatformInfo
):
    for platform in (b200_platform, h100_platform):
        assert _select(platform) == "flashmla_dsv4_prefill"


@on_sm120
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
    ],
    ids=lambda v: str(v),
)
def test_matches_the_float32_reference(
    tokens, heads, rows, width, sink, with_lse, index_dtype
):
    q, kv, indices, lens, sink_logits = _inputs(tokens, heads, rows, width, index_dtype)
    sink_logits = sink_logits if sink else None
    lse = torch.empty(tokens, heads, device="cuda") if with_lse else None
    out = dsv4_prefill(q, kv, indices, lens, sink_logits, SCALE, lse=lse, override=LEAF)
    ref_out, ref_lse = _reference(q, kv, indices, lens, sink_logits)
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    torch.testing.assert_close(out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE)
    if with_lse:
        assert torch.equal(torch.isinf(lse), torch.isinf(ref_lse))
        finite = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=LSE_ATOL, rtol=0)


@on_sm120
def test_strided_query_and_output():
    q, kv, indices, lens, sink = _inputs(6, 16, 400, 200, torch.int32)
    wide_q = torch.zeros(6, 16, DIM + 64, dtype=torch.bfloat16, device="cuda")
    wide_q[..., :DIM] = q
    wide_out = torch.zeros_like(wide_q)
    out = dsv4_prefill(
        wide_q[..., :DIM], kv, indices, lens, sink, SCALE, out=wide_out[..., :DIM],
        override=LEAF,
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
        block_h, splits = sparse_mma.plan_launch(tokens, heads, sm_count)
        return block_h, 1 << (splits - 1).bit_length(), splits == 1

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
