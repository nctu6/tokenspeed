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

"""SM120 DeepGEMM block-FP8 dense GEMM and DSA prefill top-k, on real hardware."""

import math

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _gemm_format_signature, mm
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import select_kernel
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


def operands(m, n, k):
    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(m, (k + 127) // 128, device="cuda") * 0.03 + 0.01
    sb = torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda") * 0.03 + 0.01
    return a, b, sa, sb


def reference(a, b, sa, sb):
    m, k = a.shape
    n = b.shape[0]
    return (a.float() * sa.repeat_interleave(128, 1)[:, :k]) @ (
        b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    ).T


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 2112, 7168),  # decode, N not a multiple of 128
        (17, 512, 512),
        (64, 3072, 1536),
        (129, 256, 256),
        (512, 7168, 2048),
        (2048, 4096, 7168),  # prefill, 8 math warps per CTA
    ],
)
def test_auto_selection_values_out_and_graph(m, n, k):
    a, b, sa, sb = operands(m, n, k)
    sig = _gemm_format_signature(a, b, sa, sb, torch.bfloat16, "mxfp8", [128, 128])
    selected = select_kernel(
        "gemm",
        "mm",
        sig,
        traits={
            "m": m,
            "n": n,
            "k": k,
            "a_inner_stride_one": True,
            "b_inner_stride_one": True,
            # The label the ``mm`` facade uses on SM100+
            "block_scale_layout": "canonical_blackwell",
            "out_dtype": torch.bfloat16,
            "out_inner_stride_one": True,
        },
    )
    assert selected.name == "deepgemm_sm120_mm_fp8_blockscale"

    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def run():
        return mm(
            a,
            b,
            A_scales=sa,
            B_scales=sb,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            out=out,
        )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    def check():
        torch.testing.assert_close(
            out.float(), reference(a, b, sa, sb), atol=2e-4, rtol=5e-3
        )

    try:
        assert run().data_ptr() == out.data_ptr()
        check()
        # Bitwise deterministic across launches
        first = out.clone()
        for _ in range(5):
            run()
            assert torch.equal(out, first)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        for factor in [0.0, 0.5, 2.0]:
            a.copy_((torch.randn(m, k, device="cuda") * factor).to(a.dtype))
            g.replay()
            check()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_dsa_prefill_topk_matches_triton():
    """The DeepGEMM DSA prefill leaf runs on SM120 (Triton radix top-k) and
    selects the same keys as the Triton leaf on well-separated scores."""
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm
    from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

    torch.manual_seed(0)
    page, heads, dim, topk = 64, 64, 128, 512
    num_tokens, prefix = 96, 1500
    num_keys = prefix + num_tokens
    num_pages = math.ceil(num_keys / page)
    keys = torch.randn(num_pages, page, dim, device="cuda", dtype=torch.bfloat16)
    scales = keys.abs().amax(-1).float().clamp_min(1e-4) / 448.0
    cache = torch.empty(num_pages, page * (dim + 4), device="cuda", dtype=torch.uint8)
    cache[:, : page * dim] = (
        (keys / scales[..., None])
        .to(torch.float8_e4m3fn)
        .view(num_pages, -1)
        .view(torch.uint8)
    )
    cache[:, page * dim :] = scales.view(torch.uint8).view(num_pages, -1)
    cache = cache.view(num_pages * page, dim + 4)
    slots = torch.arange(num_keys, device="cuda", dtype=torch.int64)
    row_starts = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    row_ends = (
        prefix + 1 + torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    ).contiguous()
    q = torch.randn(num_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(num_tokens, heads, device="cuda", dtype=torch.float32)
    kwargs = dict(
        topk=topk, softmax_scale=dim**-0.5, index_k_cache=cache, page_size=page
    )

    dg_out, dg_lens = dsa_deep_gemm.deep_gemm_dsa_prefill_topk(
        q, weights, slots, row_starts, row_ends, max_seqlen_k=num_keys, **kwargs
    )
    tr_out, tr_lens = dsa_triton.triton_dsa_prefill_topk_fp8(
        q, weights, slots, row_starts, row_ends, **kwargs
    )
    torch.testing.assert_close(dg_lens, tr_lens)
    overlap = [
        torch.isin(x[x >= 0], y[y >= 0]).float().mean().item()
        for x, y in zip(dg_out, tr_out)
    ]
    # Both score FP8 keys; only near-tied scores may swap at the top-k boundary
    assert min(overlap) > 0.97, min(overlap)
    assert bool(((dg_out >= -1) & (dg_out < num_keys)).all())
