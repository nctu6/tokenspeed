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
"""Tensor-core MLA decode (``triton_mla_mma_decode_with_kvcache``) on SM120.

The leaf attends a paged BF16/FP16/E4M3 latent cache with ``mma.sync``. The
reference is float32 attention over the same stored values, rounded to the
output dtype.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache
from tokenspeed_kernel.ops.attention.mla import triton as mla_triton
from tokenspeed_kernel.ops.attention.mla._triton import decode_mma
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, warm_specialization_classes

RANK, ROPE = 512, 64
DIM = RANK + ROPE
SCALE = 0.1352
LEAF = "triton_mla_mma_decode_with_kvcache"

on_sm120 = pytest.mark.skipif(
    not (Platform.get().is_nvidia and Platform.get().arch_version.major == 12),
    reason="SM120 required",
)

# Scores are exact products accumulated in FP32 and the LSE is exact FP32. The
# output is the FP32 weighted sum of cache values with the probabilities
# rounded to the cache dtype's MMA operand (BF16, or FP16 for E4M3, which keeps
# 3 more bits), then rounded to the output dtype: within one ulp of it.
OUT_TOLERANCE = {
    torch.bfloat16: {"atol": 4.0e-3, "rtol": 2.0**-6},
    torch.float16: {"atol": 1.0e-3, "rtol": 2.0**-9},
    torch.float8_e4m3fn: {"atol": 2.0e-3, "rtol": 2.0**-7},
}
LSE_ATOL = 2.0e-5


def _inputs(batch, heads, length, q_len, dtype, *, page_size=64, varlen=True):
    generator = torch.Generator(device="cuda").manual_seed(batch * 131 + length)
    max_pages = -(-length // page_size)
    num_pages = batch * max_pages + 3
    table = torch.randperm(num_pages, generator=generator, device="cuda")
    table = table[: batch * max_pages].reshape(batch, max_pages).to(torch.int32)
    kv = torch.randn(num_pages, page_size, 1, DIM, generator=generator, device="cuda")
    q = torch.randn(batch, q_len, heads, DIM, generator=generator, device="cuda")
    kv, q = (kv * 0.8).to(torch.bfloat16), (q * 0.8).to(torch.bfloat16)
    if dtype != torch.bfloat16:
        kv, q = kv.to(dtype), q.to(dtype)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    if varlen and batch > 1:
        lengths = torch.randint(
            max(q_len, length // 3),
            length + 1,
            (batch,),
            generator=generator,
            device="cuda",
            dtype=torch.int32,
        )
        lengths[0] = length
    return q, kv, table, lengths


def _reference(q, kv, table, lengths, block, window):
    """Float32 attention; returns ``(out, lse)`` in FP32."""
    batch, q_len, heads, _ = q.shape
    page_size = kv.shape[1]
    out = torch.zeros(batch, q_len, heads, RANK, device="cuda")
    lse = torch.full((batch, q_len, heads), float("-inf"), device="cuda")
    for b in range(batch):
        length = int(lengths[b])
        pages = -(-length // page_size)
        rows = kv[table[b, :pages].long()].reshape(-1, DIM)[:length].float()
        for i in range(q_len):
            visible = length - (q_len - 1 - i) if block == 1 else length
            first = 0
            if window >= 0:
                position = (b * q_len + i) % block
                first = max(visible - block - window + position, 0)
            keys = rows[first:visible]
            if keys.shape[0] == 0:
                continue
            scores = (q[b, i].float() @ keys.T) * SCALE
            top = scores.max(-1, keepdim=True).values
            weights = torch.exp(scores - top)
            total = weights.sum(-1, keepdim=True)
            out[b, i] = (weights / total) @ keys[:, :RANK]
            lse[b, i] = (top + total.log()).squeeze(-1)
    return out, lse


def _select(platform, **traits):
    request = {
        "q_len": 1,
        "num_q_heads": 128,
        "kv_lora_rank": RANK,
        "qk_rope_head_dim": ROPE,
        "page_size": 64,
        "noncausal_block_size": 1,
        "block_on_query_axis": False,
        "logit_cap": False,
        "return_lse": False,
        "sliding_window": False,
    }
    request.update(traits)
    dtype = request.pop("dtype", torch.bfloat16)
    return select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        format_signature(
            q=dense_tensor_format(dtype), kv_cache=dense_tensor_format(dtype)
        ),
        platform=platform,
        traits=request,
    ).name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_selects_the_mma_leaf(dtype, sm120_platform: PlatformInfo):
    assert _select(sm120_platform, dtype=dtype) == LEAF
    assert _select(sm120_platform, dtype=dtype, return_lse=True) == LEAF
    # A causal verify block and a non-causal proposal block on the query axis.
    causal = {"q_len": 4, "block_on_query_axis": True}
    assert _select(sm120_platform, dtype=dtype, **causal) == LEAF
    proposal = {**causal, "noncausal_block_size": 4, "sliding_window": True}
    assert _select(sm120_platform, dtype=dtype, **proposal) == LEAF
    # A proposal block flattened to one row per position.
    flat = {"noncausal_block_size": 2, "sliding_window": True}
    assert _select(sm120_platform, dtype=dtype, **flat) == LEAF


def test_logit_cap_and_other_archs_keep_their_leaves(
    sm120_platform: PlatformInfo,
    b200_platform: PlatformInfo,
    h100_platform: PlatformInfo,
):
    assert _select(sm120_platform, logit_cap=True) == "triton_mla_decode_with_kvcache"
    for platform in (b200_platform, h100_platform):
        assert _select(platform) == "triton_mla_decode_with_kvcache"
        assert _select(platform, return_lse=True) != LEAF


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("batch", "heads", "length", "q_len", "block", "window"),
    [
        (3, 16, 700, 1, 1, -1),
        (2, 128, 5000, 1, 1, -1),
        (1, 128, 33000, 1, 1, -1),
        (2, 12, 1000, 4, 1, -1),
        (2, 128, 2000, 8, 1, 300),
        (2, 16, 1000, 4, 4, -1),
        (2, 16, 1500, 4, 4, 200),
        (4, 12, 900, 1, 2, 100),
        (5, 16, 17, 3, 1, -1),
    ],
    ids=lambda v: str(v),
)
def test_matches_the_float32_reference(
    dtype, batch, heads, length, q_len, block, window
):
    q, kv, table, lengths = _inputs(batch, heads, length, q_len, dtype)
    out, lse = mla_decode_with_kvcache(
        q,
        kv,
        table,
        lengths,
        length,
        128,
        RANK,
        ROPE,
        SCALE,
        return_lse=True,
        window_left=window,
        noncausal_block_size=block,
        override=LEAF,
    )
    out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    assert out.dtype == out_dtype and out.shape == (batch, q_len, heads, RANK)
    ref_out, ref_lse = _reference(q, kv, table, lengths, block, window)
    torch.testing.assert_close(out, ref_out.to(out_dtype), **OUT_TOLERANCE[dtype])
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=0)


@on_sm120
def test_fp16_and_empty_rows():
    q, kv, table, lengths = _inputs(2, 16, 600, 1, torch.float16)
    # A request with nothing cached sees nothing: zero output, -inf LSE.
    lengths[1] = 0
    out, lse = mla_decode_with_kvcache(
        q, kv, table, lengths, 600, 128, RANK, ROPE, SCALE, return_lse=True,
        window_left=1, noncausal_block_size=2, override=LEAF,
    )  # fmt: skip
    ref_out, ref_lse = _reference(q, kv, table, lengths, 2, 1)
    torch.testing.assert_close(out, ref_out.half(), **OUT_TOLERANCE[torch.float16])
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=0)


@on_sm120
def test_writes_into_a_strided_output():
    q, kv, table, lengths = _inputs(3, 16, 500, 2, torch.bfloat16)
    wide = torch.zeros(3, 2, 16, RANK + 64, dtype=torch.bfloat16, device="cuda")
    out = wide[..., :RANK]
    result = mla_decode_with_kvcache(
        q, kv, table, lengths, 500, 128, RANK, ROPE, SCALE, out=out, override=LEAF
    )
    assert result.data_ptr() == out.data_ptr()
    ref_out, _ = _reference(q, kv, table, lengths, 1, -1)
    torch.testing.assert_close(
        out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE[torch.bfloat16]
    )
    assert not wide[..., RANK:].any()


@on_sm120
def test_graph_replay_follows_the_metadata():
    q, kv, table, lengths = _inputs(4, 128, 3000, 1, torch.bfloat16)
    run = lambda: mla_decode_with_kvcache(  # noqa: E731
        q, kv, table, lengths, 3000, 128, RANK, ROPE, SCALE, override=LEAF
    )
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    lengths.copy_(torch.tensor([3000, 17, 1500, 64], dtype=torch.int32))
    graph.replay()
    ref_out, _ = _reference(q, kv, table, lengths, 1, -1)
    torch.testing.assert_close(
        out, ref_out.to(torch.bfloat16), **OUT_TOLERANCE[torch.bfloat16]
    )


@on_sm120
def test_batch_shape_does_not_recompile():
    sm_count = Platform.get().sm_count
    heads, length = 128, 2048

    def run(batch):
        q, kv, table, lengths = _inputs(batch, heads, length, 1, torch.bfloat16)
        # The table width follows the longest request, not the batch.
        table = table[:, : 1 + batch % 7].contiguous()
        lengths.clamp_(max=table.shape[1] * 64)
        out = mla_decode_with_kvcache(
            q, kv, table, lengths, 1 << 17, 128, RANK, ROPE, SCALE, override=LEAF
        )
        assert out.shape == (batch, 1, heads, RANK)

    def key(batch):
        block_m, splits = decode_mma.plan_launch(batch, heads, sm_count)
        return block_m, triton_bucket(splits), splits == 1

    def triton_bucket(splits):
        return 1 << (splits - 1).bit_length()

    sweep = (1, 2, 3, 5, 8, 9, 17, 33, 64, 100, 130)
    warm_specialization_classes(run, key, sweep, pool=(1, 2, 4, 8, 16, 40, 130))
    with assert_no_triton_compile(
        decode_mma._mla_mma_decode_kernel, decode_mma._mla_mma_decode_combine_kernel
    ):
        for batch in sweep:
            run(batch)
    assert mla_triton.triton_mla_mma_decode_with_kvcache is not None
