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

"""Triton MHA prefill/extend: SKIP_OOR_BLOCKS is bitwise.

SKIP_OOR_BLOCKS returns from query blocks that lie entirely past a sequence's
extend length (their stores are fully masked), so it only drops work whose
contribution is provably nothing and must reproduce the legacy kernel with
``torch.equal`` -- not ``allclose``. The flag is a module-level constant read
from the environment at import; tests flip the module attribute directly and
check the env parsing in a subprocess.

The sliding-window KV-range trim is main's always-on ``begin_n``/``end_n``
(#1559, covered by ``test_attention.py::test_mha_prefill_triton_window_bounds``);
these tests run on top of it.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys

import pytest
import torch

_MOD_NAME = "tokenspeed_kernel.ops.attention.mha._triton.prefill"
mha_prefill_mod = importlib.import_module(_MOD_NAME)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# skip_oor values; the first entry is the legacy kernel.
_LEGACY = False
_VARIANTS = [True]


def _run_with_flags(monkeypatch, skip_oor: bool, fn):
    with monkeypatch.context() as m:
        m.setattr(mha_prefill_mod, "_TRITON_PREFILL_SKIP_OOR", skip_oor)
        out = fn()
    torch.cuda.synchronize()
    return out if isinstance(out, tuple) else (out,)


def _assert_bitwise_across_flags(monkeypatch, fn) -> None:
    ref = _run_with_flags(monkeypatch, _LEGACY, fn)
    for skip_oor in _VARIANTS:
        got = _run_with_flags(monkeypatch, skip_oor, fn)
        for idx, (a, b) in enumerate(zip(ref, got)):
            same_nan = torch.isnan(a) == torch.isnan(b)
            assert bool(same_nan.all()), (skip_oor, idx, "nan pattern")
            a_ = torch.nan_to_num(a.float(), nan=0.0)
            b_ = torch.nan_to_num(b.float(), nan=0.0)
            if not torch.equal(a_, b_):
                diff = (a_ - b_).abs()
                pytest.fail(
                    f"skip_oor={skip_oor} output[{idx}] "
                    f"not bitwise: max|diff|={diff.max().item():.3e}, "
                    f"mismatches={(diff > 0).sum().item()}"
                )


def _cu(lens: list[int], device) -> tuple[list[int], torch.Tensor]:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu, torch.tensor(cu, device=device, dtype=torch.int32)


# --------------------------------------------------------------------------
# Env parsing (no GPU work).
# --------------------------------------------------------------------------


def test_env_flag_parsing(monkeypatch) -> None:
    name = "TOKENSPEED_TEST_ENV_FLAG_F1"
    monkeypatch.delenv(name, raising=False)
    assert mha_prefill_mod._env_flag(name, True) is True
    assert mha_prefill_mod._env_flag(name, False) is False
    for raw in ("1", "true", "YES", " on "):
        monkeypatch.setenv(name, raw)
        assert mha_prefill_mod._env_flag(name, False) is True
    for raw in ("0", "false", "No", "OFF"):
        monkeypatch.setenv(name, raw)
        assert mha_prefill_mod._env_flag(name, True) is False
    monkeypatch.setenv(name, "")
    assert mha_prefill_mod._env_flag(name, True) is True
    monkeypatch.setenv(name, "2")
    with pytest.raises(ValueError):
        mha_prefill_mod._env_flag(name, True)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, True),
        ({"TOKENSPEED_TRITON_PREFILL_SKIP_OOR": "0"}, False),
        ({"TOKENSPEED_TRITON_PREFILL_SKIP_OOR": "1"}, True),
    ],
    ids=["defaults", "skip-off", "skip-on"],
)
def test_env_defaults_read_at_import(env, expected) -> None:
    child_env = {
        k: v for k, v in os.environ.items() if k != "TOKENSPEED_TRITON_PREFILL_SKIP_OOR"
    }
    child_env.update(env)
    code = (
        f"import importlib; m = importlib.import_module({_MOD_NAME!r}); "
        "print(int(m._TRITON_PREFILL_SKIP_OOR))"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, res.stderr
    got = bool(int(res.stdout.strip().split()[-1]))
    assert got == expected


# --------------------------------------------------------------------------
# mha_prefill (HAS_KV_CACHE=False): ragged / uniform, window, lse, cap, sinks.
# --------------------------------------------------------------------------

_PREFILL_SHAPES = {
    "ragged": [851, 914, 1053],
    "uniform": [1024, 1024, 1024, 1024],
    "ragged-short": [1053, 7, 1, 130],
}

_FEATURES = {
    "plain": dict(return_lse=False, logit_cap=0.0, sinks=False),
    "lse": dict(return_lse=True, logit_cap=0.0, sinks=False),
    "cap-sinks-lse": dict(return_lse=True, logit_cap=30.0, sinks=True),
}


@requires_cuda
@pytest.mark.parametrize("feature", list(_FEATURES))
@pytest.mark.parametrize("window_left", [-1, 255, 1024], ids=["full", "w255", "w1024"])
@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("shape", list(_PREFILL_SHAPES))
def test_prefill_bitwise(monkeypatch, shape, head_dim, window_left, feature) -> None:
    device = "cuda"
    torch.manual_seed(0)
    lens = _PREFILL_SHAPES[shape]
    cu_cpu, cu = _cu(lens, device)
    total = cu_cpu[-1]
    nq, nkv = 8, 2
    dtype = torch.bfloat16
    q = torch.randn(total, nq, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, nkv, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, nkv, head_dim, device=device, dtype=dtype)
    feat = _FEATURES[feature]
    sinks = torch.randn(nq, device=device, dtype=dtype) if feat["sinks"] else None

    def fn():
        return mha_prefill_mod._triton_mha_prefill_impl(
            q,
            k,
            v,
            cu,
            cu_cpu,
            max(lens),
            window_left=window_left,
            logit_cap=feat["logit_cap"],
            sinks=sinks,
            return_lse=feat["return_lse"],
        )

    _assert_bitwise_across_flags(monkeypatch, fn)


@requires_cuda
@pytest.mark.parametrize("window_left", [-1, 255], ids=["full", "w255"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
def test_prefill_custom_mask_bitwise(monkeypatch, is_causal, window_left) -> None:
    """Custom-mask path keeps lo=0/hi=len; OOR blocks (padded grid) still skip."""
    device = "cuda"
    torch.manual_seed(1)
    seq_len, nq, nkv, head_dim = 700, 4, 2, 128
    _, cu = _cu([seq_len], device)
    dtype = torch.bfloat16
    q = torch.randn(seq_len, nq, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, nkv, head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, nkv, head_dim, device=device, dtype=dtype)
    custom_mask = (torch.rand(seq_len, seq_len, device=device) > 0.3).to(torch.uint8)
    custom_mask.fill_diagonal_(1)
    custom_mask = custom_mask.flatten()
    empty_k = torch.empty((0, nkv, head_dim), dtype=dtype, device=device)
    empty_v = torch.empty((0, nkv, head_dim), dtype=dtype, device=device)
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=device)

    def fn():
        out = torch.empty_like(q)
        lse = torch.empty((seq_len, nq), dtype=torch.float32, device=device)
        mha_prefill_mod.prefill_attention_fwd(
            q,
            k,
            v,
            out,
            empty_k,
            empty_v,
            cu,
            cache_seqlens,
            custom_mask,
            is_causal,
            # Over-sized grid: trailing query blocks are out of range.
            seq_len + 300,
            sm_scale=1.0 / math.sqrt(head_dim),
            sliding_window_size=window_left,
            has_kv_cache=False,
            lse_extend=lse,
        )
        return out, lse

    _assert_bitwise_across_flags(monkeypatch, fn)


# --------------------------------------------------------------------------
# mha_extend_with_kvcache (HAS_KV_CACHE=True): ragged extend over prefix cache.
# --------------------------------------------------------------------------


def _build_paged_kv(
    cache_lens: list[int],
    nkv: int,
    head_dim: int,
    page_size: int,
    kv_dtype: torch.dtype,
    device: str,
    seed: int,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [(n + page_size - 1) // page_size for n in cache_lens]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    # Shuffled physical pages so the page-table indirection is exercised.
    perm = torch.randperm(total_pages, device=device, generator=gen).to(torch.int32)
    page_table = torch.zeros(
        len(cache_lens), max_pages, dtype=torch.int32, device=device
    )
    off = 0
    for b, n in enumerate(pages_per_seq):
        page_table[b, :n] = perm[off : off + n]
        off += n
    shape = (total_pages, page_size, nkv, head_dim)
    k_cache = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    v_cache = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    return k_cache.to(kv_dtype), v_cache.to(kv_dtype), page_table


_EXTEND_SHAPES = {
    # (prefix_lens, extend_lens)
    "small-ragged": ([63, 48, 17, 80], [3, 1, 2, 4]),
    "ragged-prefix": ([0, 2000, 517, 64], [851, 914, 1053, 1]),
    "uniform": ([0, 0, 0, 0], [1024, 1024, 1024, 1024]),
    # One long fresh chunk + many short multi-turn requests on a long prefix:
    # the case SKIP_OOR targets (most of their grid-M blocks are OOR).
    "long-chunk-plus-short-30k": ([0] + [30000] * 15, [8192] + [16] * 15),
}


def _extend_case_ids():
    cases = []
    for shape in ("small-ragged", "ragged-prefix", "uniform"):
        for head_dim in (128, 256, 512):
            for kv in ("bf16", "fp8"):
                cases.append(
                    pytest.param(shape, head_dim, kv, id=f"{shape}-d{head_dim}-{kv}")
                )
    for head_dim in (256, 512):
        cases.append(
            pytest.param(
                "long-chunk-plus-short-30k",
                head_dim,
                "bf16",
                id=f"long-chunk-plus-short-30k-d{head_dim}-bf16",
            )
        )
    cases.append(
        pytest.param(
            "long-chunk-plus-short-30k",
            512,
            "fp8",
            id="long-chunk-plus-short-30k-d512-fp8",
        )
    )
    return cases


@requires_cuda
@pytest.mark.parametrize("feature", ["plain", "cap-sinks-lse"])
@pytest.mark.parametrize("window_left", [-1, 1024], ids=["full", "w1024"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("shape,head_dim,kv", _extend_case_ids())
def test_extend_with_kvcache_bitwise(
    monkeypatch, shape, head_dim, kv, is_causal, window_left, feature
) -> None:
    if shape == "long-chunk-plus-short-30k" and (not is_causal or feature != "plain"):
        pytest.skip("long shape: causal/plain only to bound runtime")
    device = "cuda"
    torch.manual_seed(2)
    prefix_lens, extend_lens = _EXTEND_SHAPES[shape]
    cache_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    nq, nkv = (32, 4) if head_dim == 512 and shape == "small-ragged" else (8, 2)
    kv_dtype = torch.float8_e4m3fn if kv == "fp8" else torch.bfloat16
    page_size = 64
    k_cache, v_cache, page_table = _build_paged_kv(
        cache_lens, nkv, head_dim, page_size, kv_dtype, device, seed=3
    )
    _, cu_q = _cu(extend_lens, device)
    _, cu_kv = _cu(cache_lens, device)
    # Contract: cache_seqlens is the total length *including* this extend.
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    q = torch.randn(sum(extend_lens), nq, head_dim, device=device, dtype=torch.bfloat16)
    feat = _FEATURES[feature]
    sinks = (
        torch.randn(nq, device=device, dtype=torch.bfloat16) if feat["sinks"] else None
    )

    def fn():
        return mha_prefill_mod._triton_mha_extend_with_kvcache_impl(
            q,
            cu_q,
            cu_kv,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            max(extend_lens),
            max(cache_lens),
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=feat["logit_cap"],
            sinks=sinks,
            return_lse=feat["return_lse"],
        )

    _assert_bitwise_across_flags(monkeypatch, fn)


# --------------------------------------------------------------------------
# Contract lock: cache_seqlens includes the extend tokens (queries are the
# suffix of the KV sequence). The window/causal KV bounds are derived from
# this, so check the kernel against an fp32 reference built on that contract.
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("window_left", [-1, 100], ids=["full", "w100"])
@pytest.mark.parametrize("head_dim", [128, 512])
def test_extend_contract_matches_reference(monkeypatch, head_dim, window_left) -> None:
    device = "cuda"
    torch.manual_seed(4)
    prefix_lens, extend_lens = [0, 300, 37, 129], [70, 5, 200, 1]
    cache_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    nq, nkv, page_size = 8, 2, 16
    k_cache, v_cache, page_table = _build_paged_kv(
        cache_lens, nkv, head_dim, page_size, torch.bfloat16, device, seed=5
    )
    cu_q_cpu, cu_q = _cu(extend_lens, device)
    _, cu_kv = _cu(cache_lens, device)
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    q = torch.randn(sum(extend_lens), nq, head_dim, device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(head_dim)

    with monkeypatch.context() as m:
        m.setattr(mha_prefill_mod, "_TRITON_PREFILL_SKIP_OOR", True)
        out = mha_prefill_mod._triton_mha_extend_with_kvcache_impl(
            q,
            cu_q,
            cu_kv,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            max(extend_lens),
            max(cache_lens),
            is_causal=True,
            window_left=window_left,
            softmax_scale=scale,
        )

    group = nq // nkv
    for b, (plen, elen) in enumerate(zip(prefix_lens, extend_lens)):
        total = plen + elen
        pages = page_table[b, : (total + page_size - 1) // page_size].long()
        k_seq = k_cache[pages].reshape(-1, nkv, head_dim)[:total].float()
        v_seq = v_cache[pages].reshape(-1, nkv, head_dim)[:total].float()
        k_seq = k_seq.repeat_interleave(group, dim=1)
        v_seq = v_seq.repeat_interleave(group, dim=1)
        q_seq = q[cu_q_cpu[b] : cu_q_cpu[b + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * scale
        q_pos = torch.arange(plen, total, device=device)[:, None]
        k_pos = torch.arange(total, device=device)[None, :]
        mask = q_pos >= k_pos
        if window_left > 0:
            mask &= q_pos <= k_pos + window_left
        scores = scores.masked_fill(~mask[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v_seq)
        torch.testing.assert_close(
            out[cu_q_cpu[b] : cu_q_cpu[b + 1]].float(), ref, rtol=2e-2, atol=2e-2
        )


# --------------------------------------------------------------------------
# Regression: windowed KV-range trim + DECODE_KV_SPLITS=auto under
# concurrent-like ragged prefill. A dynamic KV range previously
# illegal-memory'd on sm120 when overlapped with auto-split decode (fixed by
# the ``page_in_range`` gather bound). This stress hammers that pairing.
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("window_left", [-1, 1024], ids=["full", "w1024"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_window_trim_auto_concurrent_ragged_stress(
    monkeypatch, window_left, head_dim
) -> None:
    device = "cuda"
    torch.manual_seed(6)
    # One long fresh chunk + many short multi-turn extends on long prefixes
    # (the E2E shape that crashed once on raichu gemma3 with a KV clamp+auto).
    # Sized to finish in CI while still mixing a long chunk, short extends on
    # multi-k prefixes, and a tight page-table width (no spare columns).
    prefix_lens = [0, 512, 2000] + [8192] * 6 + [1600] * 4
    extend_lens = [2048, 16, 32] + [16] * 6 + [24, 18, 20, 22]
    assert len(prefix_lens) == len(extend_lens)
    cache_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    nq, nkv, page_size = 8, 2, 64
    k_cache, v_cache, page_table = _build_paged_kv(
        cache_lens, nkv, head_dim, page_size, torch.bfloat16, device, seed=7
    )
    # Intentionally tight page-table width (= max pages of this batch) so
    # last-tile masked lanes can compute page_indices == stride without a
    # spare column — the sm120 fault mode under memory pressure.
    _, cu_q = _cu(extend_lens, device)
    _, cu_kv = _cu(cache_lens, device)
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    q_ext = torch.randn(
        sum(extend_lens), nq, head_dim, device=device, dtype=torch.bfloat16
    )

    mha_decode_mod = importlib.import_module(
        "tokenspeed_kernel.ops.attention.mha._triton.decode"
    )

    def run_extend(skip_oor: bool):
        with monkeypatch.context() as m:
            m.setattr(mha_prefill_mod, "_TRITON_PREFILL_SKIP_OOR", skip_oor)
            return mha_prefill_mod._triton_mha_extend_with_kvcache_impl(
                q_ext,
                cu_q,
                cu_kv,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                max(extend_lens),
                max(cache_lens),
                is_causal=True,
                window_left=window_left,
            )

    # Bitwise: SKIP_OOR on/off must still match on this stress shape.
    out_off = run_extend(False)
    out_on = run_extend(True)
    assert torch.equal(out_off, out_on), "SKIP_OOR must stay bitwise on ragged stress"

    # Interleave auto-split decode with windowed extend (no stream sync
    # between) to approximate overlap_schedule_depth>=1 under auto splits.
    bs_dec = min(32, len(cache_lens))
    q_dec = torch.randn(bs_dec, nq, head_dim, device=device, dtype=torch.bfloat16)
    cache_dec = cache_seqlens[:bs_dec].contiguous()
    # Decode page table may be wider than needed; reuse the extend table rows.
    pt_dec = page_table[:bs_dec].contiguous()

    with monkeypatch.context() as m:
        m.setattr(mha_prefill_mod, "_TRITON_PREFILL_SKIP_OOR", True)
        m.setattr(mha_decode_mod, "_TRITON_DECODE_KV_SPLITS", "auto")
        m.setattr(mha_decode_mod, "_TRITON_DECODE_MAX_KV_SPLITS", 16)
        m.setattr(mha_decode_mod, "_TRITON_DECODE_SPLIT_MIN_TOKENS", 256)
        m.setattr(mha_decode_mod, "_BATCH_INVARIANT", False)
        for _ in range(8):
            _ = mha_prefill_mod._triton_mha_extend_with_kvcache_impl(
                q_ext,
                cu_q,
                cu_kv,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                max(extend_lens),
                max(cache_lens),
                is_causal=True,
                window_left=window_left,
            )
            _ = mha_decode_mod._triton_mha_decode_with_kvcache_impl(
                q_dec,
                k_cache,
                v_cache,
                pt_dec,
                cache_dec,
                max(cache_lens),
                max_seqlen_q=1,
                window_left=window_left,
            )
        torch.cuda.synchronize()
