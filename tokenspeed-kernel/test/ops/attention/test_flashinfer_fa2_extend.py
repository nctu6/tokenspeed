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

"""F3: FlashInfer FA2 paged-extend (head_dim 512) selection + correctness.

Env ``TOKENSPEED_FLASHINFER_FA2_EXTEND`` is read at import time, so selection
checks always run in a fresh subprocess. Correctness tests require CUDA and a
FlashInfer build that can JIT FA2 for the requested head_dim.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _arch_major() -> int | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


def _fa2_arch_ok() -> bool:
    major = _arch_major()
    return major in (9, 12)


def _subprocess_select(env_mode: str, head_dim: int) -> str:
    """Return selected kernel name for mha_extend at ``head_dim`` under env."""
    code = textwrap.dedent(
        f"""
        import os
        os.environ["TOKENSPEED_FLASHINFER_FA2_EXTEND"] = {env_mode!r}
        # Fresh process: registry import reads the env once.
        import torch
        from tokenspeed_kernel.signature import dense_tensor_format, format_signature
        from tokenspeed_kernel.selection import select_kernel
        import tokenspeed_kernel.ops.attention  # noqa: F401 — register kernels

        sig = format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k_cache=dense_tensor_format(torch.bfloat16),
            v_cache=dense_tensor_format(torch.bfloat16),
        )
        traits = {{
            "head_dim": {head_dim},
            "page_size": 64,
            "is_causal": True,
            "sliding_window": False,
            "support_logit_cap": False,
            "support_sinks": False,
            "return_lse": False,
        }}
        kernel = select_kernel(
            "attention",
            "mha_extend_with_kvcache",
            sig,
            traits=traits,
        )
        print(kernel.name)
        """
    )
    env = os.environ.copy()
    env["TOKENSPEED_FLASHINFER_FA2_EXTEND"] = env_mode
    # Avoid inheriting an override that would pin the solution.
    env.pop("TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_EXTEND_WITH_KVCACHE", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.stdout.strip().splitlines()[-1]


def _subprocess_fa3_head_dims() -> str:
    code = textwrap.dedent(
        """
        import tokenspeed_kernel.ops.attention.flash_attn as fa
        print(sorted(fa._FA3_HOPPER_HEAD_DIMS)[:3], max(fa._FA3_HOPPER_HEAD_DIMS))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return proc.stdout.strip()


@pytest.mark.parametrize(
    "mode,head_dim,expect_substr",
    [
        ("off", 512, "triton"),
        ("512", 512, "flashinfer_fa2"),
    ],
)
@requires_cuda
def test_selection_env_gates_fa2(mode, head_dim, expect_substr):
    if not _fa2_arch_ok():
        pytest.skip("FA2 extend registration only on sm90 / sm12x")
    name = _subprocess_select(mode, head_dim)
    assert expect_substr in name, (mode, head_dim, name)


@requires_cuda
def test_selection_sm90_keeps_fa3_for_256():
    if _arch_major() != 9:
        pytest.skip("FA3 gate check is Hopper-only")
    # Default off: 256 must still be FA3.
    name_off = _subprocess_select("off", 256)
    assert name_off.startswith("fa3_"), name_off
    # Env on for 512 only: 256 still FA3 (SPECIALIZED > PERFORMANT).
    name_512 = _subprocess_select("512", 256)
    assert name_512.startswith("fa3_"), name_512


@requires_cuda
def test_selection_all_mode_256_on_sm120():
    if _arch_major() != 12:
        pytest.skip("all-mode 256→FA2 check is sm12x-only")
    name = _subprocess_select("all", 256)
    assert "flashinfer_fa2" in name, name


def test_fa3_hopper_head_dim_gate_locked():
    """Regression lock for d50d297a: FA3 traits stay ≤256."""
    if not torch.cuda.is_available() or _arch_major() != 9:
        # Importing flash_attn FA3 block is hopper-gated; still check constant
        # via subprocess on hopper, else parse the source string.
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "python",
            "tokenspeed_kernel",
            "ops",
            "attention",
            "flash_attn",
            "__init__.py",
        )
        path = os.path.abspath(path)
        src = open(path, encoding="utf-8").read()
        assert "_FA3_HOPPER_HEAD_DIMS = frozenset(range(8, 257, 8))" in src
        return
    out = _subprocess_fa3_head_dims()
    assert "256" in out
    assert "512" not in out


def _build_extend_inputs(
    *,
    device,
    dtype,
    head_dim,
    num_q_heads,
    num_kv_heads,
    page_size,
    prefix_lens,
    query_lens,
):
    batch = len(prefix_lens)
    cache_lens = [p + q for p, q in zip(prefix_lens, query_lens)]
    total_q = sum(query_lens)
    max_pages = max((c + page_size - 1) // page_size for c in cache_lens)
    total_pages = sum((c + page_size - 1) // page_size for c in cache_lens)

    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=dtype)
    cu_q = [0]
    for n in query_lens:
        cu_q.append(cu_q[-1] + n)
    cu_seqlens_q = torch.tensor(cu_q, device=device, dtype=torch.int32)
    cu_kv = [0]
    for n in cache_lens:
        cu_kv.append(cu_kv[-1] + n)
    cu_seqlens_kv = torch.tensor(cu_kv, device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor(cache_lens, device=device, dtype=torch.int32)

    page_table = torch.zeros(batch, max_pages, device=device, dtype=torch.int32)
    next_page = 0
    for b, c in enumerate(cache_lens):
        n = (c + page_size - 1) // page_size
        page_table[b, :n] = torch.arange(
            next_page, next_page + n, device=device, dtype=torch.int32
        )
        next_page += n

    k_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    host_meta = {
        "cu_seqlens_q_cpu": cu_q,
        "cache_seqlens_cpu": cache_lens,
        "plan_cache": {},
    }
    return dict(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(cache_lens),
        is_causal=True,
        host_meta=host_meta,
    )


def _torch_extend_ref(inputs, softmax_scale: float) -> torch.Tensor:
    """Naive fp32 reference for causal paged extend."""
    q = inputs["q"].float()
    k_cache = inputs["k_cache"].float()
    v_cache = inputs["v_cache"].float()
    page_table = inputs["page_table"]
    cache_seqlens = inputs["cache_seqlens"].tolist()
    cu_q = inputs["cu_seqlens_q"].tolist()
    page_size = k_cache.shape[1]
    num_q = q.shape[1]
    num_kv = k_cache.shape[2]
    head_dim = q.shape[2]
    assert num_q % num_kv == 0
    group = num_q // num_kv
    outs = []
    for b, kv_len in enumerate(cache_seqlens):
        q_start, q_end = cu_q[b], cu_q[b + 1]
        q_len = q_end - q_start
        qb = q[q_start:q_end]  # [Lq, Hq, D]
        pages = (kv_len + page_size - 1) // page_size
        tokens = []
        for p in range(pages):
            phys = int(page_table[b, p].item())
            take = min(page_size, kv_len - p * page_size)
            tokens.append(k_cache[phys, :take])
        k = torch.cat(tokens, dim=0)  # [Lkv, Hkv, D]
        tokens = []
        for p in range(pages):
            phys = int(page_table[b, p].item())
            take = min(page_size, kv_len - p * page_size)
            tokens.append(v_cache[phys, :take])
        v = torch.cat(tokens, dim=0)
        # Expand KV heads for GQA.
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        # Causal: query i attends keys [0, kv_len - q_len + i]
        ob = torch.empty(q_len, num_q, head_dim, dtype=torch.float32, device=q.device)
        for i in range(q_len):
            end = kv_len - q_len + i + 1
            scores = torch.einsum("hd,lhd->hl", qb[i], k[:end]) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            ob[i] = torch.einsum("hl,lhd->hd", probs, v[:end])
        outs.append(ob)
    return torch.cat(outs, dim=0)


@requires_cuda
@pytest.mark.parametrize(
    "head_dim,num_q,num_kv,prefix,query",
    [
        (512, 32, 4, [0], [17]),
        (512, 32, 4, [2048], [3]),
        (512, 16, 2, [128, 64], [2, 1]),
    ],
)
def test_fa2_extend_matches_triton_and_fp32(
    head_dim, num_q, num_kv, prefix, query, monkeypatch
):
    if not _fa2_arch_ok():
        pytest.skip("FA2 extend only on sm90 / sm12x")
    monkeypatch.setenv("TOKENSPEED_FLASHINFER_FA2_EXTEND", "512")
    # Re-import is not enough for registry; call kernel by override name after
    # ensuring the module registered under this env via subprocess-less path:
    # the parent process may have imported with off. Force override.
    from tokenspeed_kernel.ops.attention import mha_extend_with_kvcache
    from tokenspeed_kernel.ops.attention.flashinfer import paged_extend as pe

    if pe.fa2_extend_mode() == "off":
        pytest.skip(
            "Parent process imported with FA2 off; run under "
            "TOKENSPEED_FLASHINFER_FA2_EXTEND=512"
        )

    device = "cuda"
    dtype = torch.bfloat16
    inputs = _build_extend_inputs(
        device=device,
        dtype=dtype,
        head_dim=head_dim,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        page_size=64,
        prefix_lens=prefix,
        query_lens=query,
    )
    scale = 1.0 / math.sqrt(head_dim)

    try:
        out_fa2 = mha_extend_with_kvcache(
            **inputs,
            softmax_scale=scale,
            override="flashinfer_fa2_mha_extend_with_kvcache",
        )
    except Exception as exc:  # JIT / arch support
        pytest.skip(f"FA2 kernel unavailable: {exc}")

    out_triton = mha_extend_with_kvcache(
        **{k: v for k, v in inputs.items() if k != "host_meta"},
        softmax_scale=scale,
        solution="triton",
    )
    ref = _torch_extend_ref(inputs, scale).to(dtype)

    # FA2 vs Triton: allow a few bf16 ULPs of reduction-order noise.
    torch.testing.assert_close(out_fa2, out_triton, atol=2e-2, rtol=2e-2)
    # Each should stay within Triton's distance to the fp32 reference.
    err_triton = (out_triton.float() - ref.float()).abs().max().item()
    err_fa2 = (out_fa2.float() - ref.float()).abs().max().item()
    assert err_fa2 <= err_triton * 1.5 + 2e-2, (err_fa2, err_triton)


@requires_cuda
def test_mha_plan_per_layer_head_dim_prewrite_on_sm90():
    """F3c: head_dim 512 has no FA3 prefill → prewrite; 256 → postwrite on Hopper."""
    if _arch_major() != 9:
        pytest.skip("prewrite/postwrite split is Hopper-specific")
    from tokenspeed_kernel.ops.attention import mha_plan

    plan_256 = mha_plan(dtype=torch.bfloat16, head_dim=256)
    plan_512 = mha_plan(dtype=torch.bfloat16, head_dim=512)
    assert plan_256["extend_mode"] == "postwrite", plan_256
    assert plan_512["extend_mode"] == "prewrite", plan_512
