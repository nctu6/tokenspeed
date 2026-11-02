# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""GPU: the batched AR engine reproduces the Diffusers AR loop, song by song.

The reference is Diffusers' ``MiniMaxMusic3AutoregressiveStep`` loop at batch 1
with HF's own KV cache and attention, sampling with the engine's seeded
sampler (the one deliberate difference from Diffusers' ``torch.multinomial``).
Tiny real Qwen3 + depth-decoder modules; the vocabulary keeps the checkpoint's
audio-code offsets.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "needs a CUDA GPU (tokenspeed-kernel decode attention)", allow_module_level=True
    )

pytest.importorskip("diffusers")

from diffusers import MiniMaxMusic3RVQDepthDecoder  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from tokenspeed.runtime.diffusion import music3_ar as ar  # noqa: E402

DEVICE = "cuda"
HIDDEN = 64


def _modules():
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=ar.AUDIO_CODE_OFFSET + ar.SEMANTIC_VOCAB_SIZE + 8,
            hidden_size=HIDDEN,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=512,
        )
    )
    depth = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=HIDDEN, num_layers=2, num_attention_heads=4, intermediate_size=128
    )
    with torch.no_grad():
        # EOS logit 0 never reaches the conditional top-50 of 16k random logits,
        # so the songs run to their frame budget.
        lm.lm_head.weight[ar.AUDIO_END_TOKEN_ID].zero_()
    return (
        lm.to(DEVICE, torch.bfloat16).eval(),
        depth.to(DEVICE, torch.bfloat16).eval(),
    )


def _prompt(length: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (1, length), generator=g)
    uncond = ids.clone()
    uncond[:, 1:-2] = ar.AUDIO_CFG_TOKEN_ID
    return torch.cat((ids, uncond))


def _sample(logits, seed, counter, width):
    seeds = torch.tensor([seed & 0xFFFFFFFF], dtype=torch.int64, device=DEVICE)
    counters = torch.tensor([counter], dtype=torch.int64, device=DEVICE)
    return ar._sample_top_k(logits, seeds, counters, ar.column_keys(width, DEVICE))


@torch.no_grad()
def _reference(lm, depth, prompt_ids, max_frames, seed):
    """Diffusers' AR loop (encoders.py), batch 1, sampler keyed like the engine."""
    nb = depth.config.num_codebooks
    ids = prompt_ids.to(DEVICE)
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    cache, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
    head = torch.cat(
        (
            lm.lm_head.weight[
                ar.AUDIO_CODE_OFFSET : ar.AUDIO_CODE_OFFSET + ar.SEMANTIC_VOCAB_SIZE
            ],
            lm.lm_head.weight[ar.AUDIO_END_TOKEN_ID : ar.AUDIO_END_TOKEN_ID + 1],
        )
    )
    frames = []
    for frame_index in range(max_frames + 1):
        draws = frame_index * (nb + 1)
        logits = torch.nn.functional.linear(last_hidden, head).float()
        c, u = logits[0:1], logits[1:2]
        guided = u + (c - u) * ar.AR_CFG_SCALE
        threshold = torch.topk(c, ar.AR_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(c < threshold, -float("inf"))
        sampled = _sample(guided, seed, draws, ar.SEMANTIC_VOCAB_SIZE + 1)
        if int(sampled) == ar.SEMANTIC_VOCAB_SIZE:
            break
        semantic = sampled.repeat(2)
        seq = [depth.projection(last_hidden).unsqueeze(1)]
        seq.append(
            depth.projection(
                lm.model.embed_tokens(semantic + ar.AUDIO_CODE_OFFSET)
            ).unsqueeze(1)
        )
        codes, parts = [semantic], []
        for index in range(1, nb):
            hidden = depth(torch.cat(seq, dim=1))[:, -1]
            parts.append(hidden[:1])
            hl = depth.audio_heads[index - 1](hidden)
            cc, uu = hl[:1].float(), hl[1:2].float()
            code = _sample(
                uu + (cc - uu) * ar.AR_CFG_SCALE, seed, draws + index, 1024
            ).repeat(2)
            codes.append(code)
            if index < nb - 1:
                emb = depth.audio_embeddings(code + (index - 1) * 1024)
                seq.append(depth.projection(emb).unsqueeze(1))
        if frame_index > 0:
            frames.append(torch.cat((last_hidden[:1], *parts), dim=-1))
            if len(frames) >= max_frames:
                break
        fc = torch.stack(codes, dim=1)
        emb = lm.model.embed_tokens(fc[:, :1] + ar.AUDIO_CODE_OFFSET)
        offs = (torch.arange(nb - 1, device=DEVICE) * 1024).unsqueeze(0)
        extra = depth.audio_embeddings(fc[:, 1:] + offs).sum(dim=1, keepdim=True)
        feedback = (emb + extra.to(emb.dtype)) * nb**-0.5
        out = lm.model(inputs_embeds=feedback, past_key_values=cache, use_cache=True)
        cache, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
    return torch.cat(frames)


def _run_engine(lm, depth, songs, *, graph):
    engine = ar.Music3AREngine(
        lm, depth, device=DEVICE, kv_cache_gib=0.05, max_pairs=4, use_cuda_graph=graph
    )
    results = {}
    for name, prompt, frames, seed in songs:
        got = []
        req = ar.Music3ARRequest(
            prompt_ids=prompt,
            max_frames=frames,
            seed=seed,
            on_frames=lambda f, i, got=got: got.append(f.clone()),
            on_done=lambda n, e, name=name, got=got: results.__setitem__(
                name, (n, e, got)
            ),
        )
        engine.admit(req)
    while engine.active:
        engine.step()
    assert (
        len(engine.pool.free) == engine.k_cache[0].shape[0] - 1
    )  # every page returned
    return {name: (n, e, torch.cat(g)) for name, (n, e, g) in results.items()}


SONGS = [
    ("a", _prompt(37, 1), 24, 101),
    ("b", _prompt(20, 2), 16, 202),
    ("c", _prompt(51, 3), 20, 303),  # crosses a 64-token KV page mid-song
]


def _assert_in_step(got: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    """The engine follows the reference song up to bf16 rounding.

    The engine's fused kernels / paged decode attention and HF's modules /
    SDPA round differently in bf16, so on a random tiny model (near-flat
    logits) a near-tie sample flips every few frames. A flip perturbs at most
    the next couple of frames, then the song is back in step: the sampler
    noise is keyed by frame, not by history. A wrong KV position, page, RoPE
    row or depth-cache entry instead breaks every later frame. So: most frames
    match, and no mismatch run is longer than a flip episode.
    """
    error = (got.float() - expected.float()).abs().amax(dim=-1)
    close = (error <= 3e-2 * expected.float().abs().amax(dim=-1)).tolist()
    longest = run = 0
    for ok in close:
        run = 0 if ok else run + 1
        longest = max(longest, run)
    assert sum(close) >= 0.5 * len(close), (name, close)
    assert longest <= 4, (name, close)
    assert close[-1], (name, close)  # and it ends in step


def test_engine_matches_reference_alone_and_batched():
    lm, depth = _modules()
    expected = {name: _reference(lm, depth, p, n, s) for name, p, n, s in SONGS}
    for songs in ([SONGS[0]], SONGS):  # alone, then three pairs in one batch
        got = _run_engine(lm, depth, songs, graph=False)
        for name, _p, frames, _s in songs:
            count, error, hidden = got[name]
            assert error is None and count == frames
            _assert_in_step(hidden, expected[name], name)


def test_graph_replay_equals_eager_bitwise():
    lm, depth = _modules()
    eager = _run_engine(lm, depth, SONGS, graph=False)
    graph = _run_engine(lm, depth, SONGS, graph=True)
    for name, *_ in SONGS:
        assert eager[name][0] == graph[name][0]
        assert torch.equal(eager[name][2], graph[name][2]), name


def test_prompt_ids_are_the_diffusers_cond_uncond_pair():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/models/MiniMaxAI/MiniMax-Music3/tokenizer")
    ids = ar.build_prompt_ids(tok, "warm jazz", "[verse]\nhello there")
    assert ids.shape[0] == 2
    assert torch.equal(ids[0, :1], ids[1, :1]) and torch.equal(ids[0, -2:], ids[1, -2:])
    assert bool((ids[1, 1:-2] == ar.AUDIO_CFG_TOKEN_ID).all())
