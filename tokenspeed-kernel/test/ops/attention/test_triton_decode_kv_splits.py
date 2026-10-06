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

"""Triton MHA decode: KV-split sizing policy (TOKENSPEED_TRITON_DECODE_KV_SPLITS).

* ``legacy`` (default) keeps the old 4-wide grid with one split per request.
* ``auto`` sizes MAX_KV_SPLITS on the host from static shapes + SM count and
  picks per-request splits on device from the actual KV length.
* an integer N forces N splits (sweeps).

Split counts change the reduction order, so non-legacy modes are checked
against an fp32 reference and against legacy with a small tolerance (not
bitwise). Mode/knobs are module constants read at import; tests patch the
attributes and check env parsing in a subprocess.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

_MOD_NAME = "tokenspeed_kernel.ops.attention.mha._triton.decode"
dec = importlib.import_module(_MOD_NAME)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_ENV_NAMES = (
    "TOKENSPEED_TRITON_DECODE_KV_SPLITS",
    "TOKENSPEED_TRITON_DECODE_MAX_KV_SPLITS",
    "TOKENSPEED_TRITON_DECODE_SPLIT_MIN_TOKENS",
    "TOKENSPEED_TRITON_DECODE_TARGET_WAVES",
    "VLLM_BATCH_INVARIANT",
)


# --------------------------------------------------------------------------
# Host policy (pure Python).
# --------------------------------------------------------------------------


def _max(total_q, nq, nkv, sm, cap=16, waves=2, bi=False):
    return dec.decode_max_kv_splits(
        total_q, nq, nkv, sm, max_cap=cap, target_waves=waves, batch_invariant=bi
    )


@pytest.mark.parametrize("sm", [132, 188])
@pytest.mark.parametrize(
    "nq,nkv", [(32, 4), (32, 16), (16, 2), (8, 8), (32, 32), (64, 1)]
)
def test_max_splits_monotone_and_bounded(sm, nq, nkv) -> None:
    prev = None
    for bs in [1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 128, 256, 452, 1024]:
        m = _max(bs, nq, nkv, sm)
        assert 1 <= m <= 16
        assert m & (m - 1) == 0, "power of two"
        if prev is not None:
            assert m <= prev, (bs, m, prev)
        prev = m
    assert _max(4096, nq, nkv, sm) == 1


def test_max_splits_gemma4_geometry() -> None:
    # Global layer TP1: 32 q / 4 kv -> kv_group 8 -> 4 head blocks per token.
    assert _max(1, 32, 4, 188) == 16
    assert _max(1, 32, 4, 132) == 16
    # 2*188/(32*4)=2.9 -> 2 ; 2*132/128=2.06 -> 2
    assert _max(32, 32, 4, 188) == 2
    assert _max(32, 32, 4, 132) == 2
    assert _max(128, 32, 4, 188) == 1
    # Sliding layer: 32 q / 16 kv -> kv_group 2 -> 16 head blocks.
    assert _max(1, 32, 16, 188) == 16
    assert _max(8, 32, 16, 188) == 2
    # MHA kernel: one CTA per q head.
    assert _max(1, 8, 8, 188) == 16
    assert _max(4, 32, 32, 188) == 2


def test_max_splits_cap_and_waves() -> None:
    assert _max(1, 32, 4, 188, cap=8) == 8
    assert _max(1, 32, 4, 188, cap=12) == 12
    assert _max(1, 32, 4, 188, cap=1) == 1
    assert _max(8, 32, 4, 188, waves=1) <= _max(8, 32, 4, 188, waves=2)
    assert _max(0, 32, 4, 188) >= 1


@pytest.mark.parametrize("cap", [1, 4, 16])
def test_max_splits_batch_invariant_fixed(cap) -> None:
    for bs in [1, 8, 64, 452, 4096]:
        assert _max(bs, 32, 4, 188, cap=cap, bi=True) == cap
        assert _max(bs, 8, 8, 132, cap=cap, bi=True) == cap


def test_num_kv_splits_from_length() -> None:
    lens = torch.tensor([0, 1, 31, 32, 33, 255, 256, 257, 1025, 4096, 26000, 10**6])
    got = dec.decode_num_kv_splits(lens.int(), -1, 16, 256)
    assert got.dtype == torch.int32
    assert got.tolist() == [1, 1, 1, 1, 1, 1, 1, 2, 5, 16, 16, 16]
    assert dec.decode_num_kv_splits(lens.int(), -1, 4, 256).max().item() == 4
    assert dec.decode_num_kv_splits(lens.int(), -1, 1, 256).tolist() == [1] * 12
    assert dec.decode_num_kv_splits(lens.int(), -1, 16, 64)[8].item() == 16
    # Sliding window: eff_len = min(len, window_left + 1).
    got_w = dec.decode_num_kv_splits(lens.int(), 1023, 16, 256)
    assert got_w.tolist() == [1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4]
    assert dec.decode_num_kv_splits(lens.int(), 0, 16, 256).tolist() == [1] * 12


def _patch(monkeypatch, mode, *, cap=16, min_tokens=256, waves=2, bi=False):
    monkeypatch.setattr(dec, "_TRITON_DECODE_KV_SPLITS", mode)
    monkeypatch.setattr(dec, "_TRITON_DECODE_MAX_KV_SPLITS", cap)
    monkeypatch.setattr(dec, "_TRITON_DECODE_SPLIT_MIN_TOKENS", min_tokens)
    monkeypatch.setattr(dec, "_TRITON_DECODE_TARGET_WAVES", waves)
    monkeypatch.setattr(dec, "_BATCH_INVARIANT", bi)


def test_resolve_modes_cpu(monkeypatch) -> None:
    q = torch.empty(4, 32, 8)
    lens = torch.tensor([10, 300, 5000, 26000], dtype=torch.int32)
    monkeypatch.setattr(dec, "_sm_count", lambda device: 188)
    monkeypatch.setattr(dec, "current_platform", lambda: SimpleNamespace(is_amd=False))

    _patch(monkeypatch, "legacy")
    m, n = dec._resolve_kv_splits(q, 4, lens, -1)
    assert m == 4 and n.tolist() == [1, 1, 1, 1] and n.dtype == torch.int32

    _patch(monkeypatch, 6)
    m, n = dec._resolve_kv_splits(q, 4, lens, -1)
    assert m == 6 and n.tolist() == [6] * 4

    _patch(monkeypatch, "auto")
    m, n = dec._resolve_kv_splits(q, 4, lens, -1)
    assert m == _max(4, 32, 4, 188) == 16
    assert n.tolist() == [1, 2, 16, 16]
    m, n = dec._resolve_kv_splits(q, 4, lens, 1023)
    assert n.tolist() == [1, 2, 4, 4]

    # Batch-invariant: MAX pinned to the cap, splits by length only.
    _patch(monkeypatch, "auto", cap=8, bi=True)
    big_q = torch.empty(512, 32, 8)
    big_lens = torch.full((512,), 26000, dtype=torch.int32)
    m, n = dec._resolve_kv_splits(big_q, 4, big_lens, -1)
    assert m == 8 and n.unique().tolist() == [8]
    m1, n1 = dec._resolve_kv_splits(q[:1], 4, lens[3:], -1)
    assert m1 == 8 and n1.tolist() == [8]

    # AMD: auto falls back to legacy.
    monkeypatch.setattr(dec, "current_platform", lambda: SimpleNamespace(is_amd=True))
    _patch(monkeypatch, "auto")
    m, n = dec._resolve_kv_splits(q, 4, lens, -1)
    assert m == 4 and n.tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, ("legacy", 16, 256, 2, False)),
        ({"TOKENSPEED_TRITON_DECODE_KV_SPLITS": "AUTO"}, ("auto", 16, 256, 2, False)),
        ({"TOKENSPEED_TRITON_DECODE_KV_SPLITS": "8"}, (8, 16, 256, 2, False)),
        (
            {
                "TOKENSPEED_TRITON_DECODE_KV_SPLITS": "legacy",
                "TOKENSPEED_TRITON_DECODE_MAX_KV_SPLITS": "32",
                "TOKENSPEED_TRITON_DECODE_SPLIT_MIN_TOKENS": "512",
                "TOKENSPEED_TRITON_DECODE_TARGET_WAVES": "1",
                "VLLM_BATCH_INVARIANT": "1",
            },
            ("legacy", 32, 512, 1, True),
        ),
    ],
    ids=["defaults", "auto", "fixed8", "knobs"],
)
def test_env_read_at_import(env, expected) -> None:
    child_env = {k: v for k, v in os.environ.items() if k not in _ENV_NAMES}
    child_env.update(env)
    code = (
        f"import importlib; m = importlib.import_module({_MOD_NAME!r}); "
        "print(repr((m._TRITON_DECODE_KV_SPLITS, m._TRITON_DECODE_MAX_KV_SPLITS, "
        "m._TRITON_DECODE_SPLIT_MIN_TOKENS, m._TRITON_DECODE_TARGET_WAVES, "
        "m._BATCH_INVARIANT)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == repr(expected)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("TOKENSPEED_TRITON_DECODE_KV_SPLITS", "0"),
        ("TOKENSPEED_TRITON_DECODE_KV_SPLITS", "fast"),
        ("TOKENSPEED_TRITON_DECODE_MAX_KV_SPLITS", "0"),
        ("TOKENSPEED_TRITON_DECODE_SPLIT_MIN_TOKENS", "x"),
    ],
)
def test_env_invalid_raises(monkeypatch, name, raw) -> None:
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError):
        if name == "TOKENSPEED_TRITON_DECODE_KV_SPLITS":
            dec._parse_kv_splits_mode(name)
        else:
            dec._env_pos_int(name, 1)


# --------------------------------------------------------------------------
# GPU correctness.
# --------------------------------------------------------------------------


def _build(cache_lens, nq, nkv, head_dim, page_size, kv_dtype, seqlen_q, seed):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [(n + page_size - 1) // page_size for n in cache_lens]
    total_pages = sum(pages_per_seq)
    perm = torch.randperm(total_pages, device=device, generator=gen).to(torch.int32)
    page_table = torch.zeros(
        len(cache_lens), max(pages_per_seq), dtype=torch.int32, device=device
    )
    off = 0
    for b, n in enumerate(pages_per_seq):
        page_table[b, :n] = perm[off : off + n]
        off += n
    shape = (total_pages, page_size, nkv, head_dim)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    q = torch.randn(
        len(cache_lens) * seqlen_q,
        nq,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    return q, k.to(kv_dtype), v.to(kv_dtype), page_table, cache_seqlens


def _reference(
    q,
    k_cache,
    v_cache,
    page_table,
    cache_lens,
    seqlen_q,
    window_left,
    scale,
    sinks,
    logit_cap=0.0,
):
    nq = q.shape[1]
    nkv, head_dim = k_cache.shape[2], k_cache.shape[3]
    page_size = k_cache.shape[1]
    group = nq // nkv
    outs = []
    for b, total in enumerate(cache_lens):
        pages = page_table[b, : (total + page_size - 1) // page_size].long()
        k = k_cache[pages].reshape(-1, nkv, head_dim)[:total].float()
        v = v_cache[pages].reshape(-1, nkv, head_dim)[:total].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        qb = q[b * seqlen_q : (b + 1) * seqlen_q].float()
        s = torch.einsum("qhd,khd->hqk", qb, k) * scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)
        q_pos = torch.arange(total - seqlen_q, total, device=q.device)[:, None]
        k_pos = torch.arange(total, device=q.device)[None, :]
        mask = k_pos <= q_pos
        if window_left >= 0:
            mask &= k_pos >= q_pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        if sinks is not None:
            sink = sinks.float()[:, None, None].expand(nq, seqlen_q, 1)
            p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
        else:
            p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs, dim=0)


_LENS = {
    "tiny": [1, 31, 32, 33],
    "mixed": [1, 33, 1025, 4097],
    "long": [26000, 300],
}

_MODES = [
    pytest.param(("auto", False), id="auto"),
    pytest.param(("auto", True), id="auto-batchinv"),
    pytest.param((4, False), id="fixed4"),
    pytest.param((16, False), id="fixed16"),
]


def _case_params():
    cases = []
    for head_dim, nq, nkv in [(512, 32, 4), (256, 32, 16), (128, 8, 8)]:
        for lens in _LENS:
            for window_left in (-1, 1023):
                for seqlen_q in (1, 4):
                    features = ["plain"]
                    if lens == "mixed" and seqlen_q == 1:
                        # sinks + logit cap + fp8 KV on one shape (runtime).
                        features.append("sinks-cap-fp8")
                    for feature in features:
                        cases.append(
                            pytest.param(
                                head_dim,
                                nq,
                                nkv,
                                lens,
                                window_left,
                                seqlen_q,
                                feature,
                                id=f"d{head_dim}-{nq}x{nkv}-{lens}-w{window_left}"
                                f"-q{seqlen_q}-{feature}",
                            )
                        )
    return cases


def _run(monkeypatch, mode, bi, fn):
    with monkeypatch.context() as m:
        _patch(m, mode, bi=bi)
        out = fn()
    torch.cuda.synchronize()
    return out


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    # Spacing of bf16 values around |x| (8 mantissa bits incl. implicit).
    mag = x.abs().clamp_min(2.0**-20)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


@requires_cuda
@pytest.mark.parametrize("mode_bi", _MODES)
@pytest.mark.parametrize(
    "head_dim,nq,nkv,lens,window_left,seqlen_q,feature", _case_params()
)
def test_decode_splits_match_reference_and_legacy(
    monkeypatch, head_dim, nq, nkv, lens, window_left, seqlen_q, mode_bi, feature
) -> None:
    mode, bi = mode_bi
    # Every query row needs at least one visible key (len >= seqlen_q).
    cache_lens = [max(n, seqlen_q) for n in _LENS[lens]]
    kv_dtype = torch.float8_e4m3fn if feature != "plain" else torch.bfloat16
    q, k_cache, v_cache, page_table, cache_seqlens = _build(
        cache_lens, nq, nkv, head_dim, 64, kv_dtype, seqlen_q, seed=7
    )
    scale = 1.0 / math.sqrt(head_dim)
    sinks = (
        torch.randn(nq, device="cuda", dtype=torch.float32)
        if feature != "plain"
        else None
    )
    logit_cap = 30.0 if feature != "plain" else 0.0

    def fn():
        return dec._triton_mha_decode_with_kvcache_impl(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            max_seqlen_k=131072,
            max_seqlen_q=seqlen_q,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            softmax_scale=scale,
        )

    legacy = _run(monkeypatch, "legacy", False, fn).float()
    got = _run(monkeypatch, mode, bi, fn).float()
    assert not torch.isnan(got).any()

    ref = _reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        seqlen_q,
        window_left,
        scale,
        sinks,
        logit_cap,
    )
    err_legacy = (legacy - ref).abs().max().item()
    err_got = (got - ref).abs().max().item()
    # Splitting must not be meaningfully worse than the 1-split kernel.
    assert err_got <= 1.5 * err_legacy + 2e-3, (err_got, err_legacy)
    tol = 2e-2 if kv_dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(got, ref, atol=tol, rtol=tol)

    # vs legacy: a few bf16 ULPs (different fp32 reduction order; the grouped
    # kernel rounds P to the V dtype per split). With fp8 KV that rounding is
    # to fp8 itself, so allow a fixed 1e-2 (legacy is ~2e-2 off fp32 there).
    diff = (got - legacy).abs()
    budget = 4 * _bf16_ulp(legacy) + (2e-3 if kv_dtype == torch.bfloat16 else 1e-2)
    bad = diff > budget
    assert not bad.any(), f"max|diff|={diff.max().item():.3e}, n_bad={int(bad.sum())}"


@requires_cuda
def test_auto_actually_splits_long_context(monkeypatch) -> None:
    """Sanity: auto mode produces >1 split for bs=1 long context."""
    q, k_cache, _, _, cache_seqlens = _build(
        [26000], 32, 4, 512, 64, torch.bfloat16, 1, seed=1
    )
    with monkeypatch.context() as m:
        _patch(m, "auto")
        mx, n = dec._resolve_kv_splits(q, k_cache.shape[2], cache_seqlens, -1)
    assert mx == 16 and n.tolist() == [16]


@requires_cuda
@pytest.mark.parametrize("bs", [1, 8, 64, 452])
def test_cuda_graph_replay_matches_eager(monkeypatch, bs) -> None:
    """Capture with one set of lengths, replay with others: MAX is static,
    per-request splits are recomputed on device each replay."""
    nq, nkv, head_dim, page_size = 32, 4, 512, 64
    max_len = 4096 if bs <= 8 else 512
    gen = torch.Generator(device="cuda").manual_seed(bs)
    pages_per_seq = max_len // page_size
    total_pages = bs * pages_per_seq
    shape = (total_pages, page_size, nkv, head_dim)
    k_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    v_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    page_table = (
        torch.randperm(total_pages, device="cuda", generator=gen)
        .to(torch.int32)
        .view(bs, pages_per_seq)
    )
    q = torch.randn(
        bs, nq, head_dim, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    cache_seqlens = torch.full((bs,), 64, dtype=torch.int32, device="cuda")

    with monkeypatch.context() as m:
        _patch(m, "auto")

        def fn():
            return dec._triton_mha_decode_with_kvcache_impl(
                q,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                max_seqlen_k=max_len,
                max_seqlen_q=1,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn()

        for trial in range(3):
            new_lens = torch.randint(
                1,
                max_len + 1,
                (bs,),
                device="cuda",
                generator=gen,
                dtype=torch.int32,
            )
            if trial == 0:
                new_lens[0] = max_len
            cache_seqlens.copy_(new_lens)
            graph.replay()
            torch.cuda.synchronize()
            eager = fn()
            torch.cuda.synchronize()
            assert torch.equal(static_out, eager), trial
