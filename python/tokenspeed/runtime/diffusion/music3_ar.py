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

"""Batched autoregressive stage of MiniMax-Music3.

The math is the Diffusers ``MiniMaxMusic3AutoregressiveStep`` (same modules,
same guidance, same top-k), executed by TokenSpeed for many songs at once:

* Every song is a classifier-free-guidance *pair* of rows (conditional +
  unconditional prompt). Pairs join after their prefill and leave on EOS or at
  their frame budget, between frame steps (continuous batching per frame).
* The Qwen3 global LM decodes from a paged KV cache through the
  ``tokenspeed-kernel`` decode-attention op; row lengths are runtime tensors.
* One *frame step* = guided LM sampling, the 7 depth-decoder codebook steps,
  the emitted frame hidden state, the frame feedback embedding and the LM
  decode of that feedback. It is captured as one CUDA graph per batch bucket.
* Sampling is Gumbel-max over the same masked top-k set as Diffusers, with
  noise from a counter hash of ``(seed, frame, draw, vocab index)``: a song is
  reproducible from its seed regardless of what it is batched with, but it is
  not the random stream of Diffusers' batch-1 ``torch.multinomial``.

The prompt prefill runs through the HF model (same code as Diffusers) and its
KV is copied into the paged cache.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = [
    "Music3AREngine",
    "Music3ARRequest",
    "build_prompt_ids",
    "hash_gumbel",
]

# Checkpoint contract (Diffusers ``minimax_music3/encoders.py``).
AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384
MAX_PROMPT_TOKENS = 5_000
MAX_AUDIO_FRAMES = 9_000
AR_CFG_SCALE = 1.5
AR_TOP_K = 50

# Sliced LM head: index i < 16384 is audio code i, index 16384 is EOS. The
# other 183k vocabulary entries are masked to -inf by the checkpoint contract,
# so dropping them changes no probability.
_EOS_SLICE_INDEX = SEMANTIC_VOCAB_SIZE

PAGE_SIZE = 64
# Batch buckets in CFG pairs; one CUDA graph each.
PAIR_BUCKETS = (1, 2, 4, 8, 16)

_M32 = 0xFFFFFFFF


def _fmix32(x: torch.Tensor) -> torch.Tensor:
    """murmur3 32-bit finalizer on int64 tensors holding uint32 values."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    return x ^ (x >> 16)


def column_keys(width: int, device) -> torch.Tensor:
    """Per-vocab-column hash salt, shape ``[width]`` int64."""
    return _fmix32(torch.arange(width, dtype=torch.int64, device=device) + 0x9E3779B9)


def hash_gumbel(
    seeds: torch.Tensor, counters: torch.Tensor, columns: torch.Tensor
) -> torch.Tensor:
    """Gumbel noise ``[rows, width]`` (float64) from ``(seed, counter, column)``.

    ``seeds`` / ``counters`` are ``[rows]`` int64; ``columns`` is
    ``column_keys(width)``. Pure function of its inputs: batching-invariant.
    """
    row = _fmix32(seeds ^ _fmix32(counters + 0x632BE5AB))
    bits = _fmix32(row[:, None] ^ columns[None, :])
    uniform = (bits.to(torch.float64) + 0.5) * (1.0 / 4294967296.0)
    return -torch.log(-torch.log(uniform))


def _sample_top_k(
    logits: torch.Tensor, seeds: torch.Tensor, counters: torch.Tensor, columns
) -> torch.Tensor:
    """Diffusers ``_sample_top_k`` (top-k 50, temperature 1) via Gumbel-max."""
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, AR_TOP_K, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    return torch.argmax(values.double() + hash_gumbel(seeds, counters, columns), dim=-1)


def build_prompt_ids(tokenizer, prompt: str, lyrics: str) -> torch.Tensor:
    """``[2, P]`` cond / uncond prompt ids, exactly as Diffusers' tokenize step."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_START,
        _CAPTION_END,
        _CAPTION_START,
        _IM_END,
        _IM_START,
        _LYRICS_END,
        _LYRICS_START,
        _clean_caption,
        _normalize_lyrics,
    )

    text = (
        f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
        f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"The assembled prompt has {ids.shape[1]} tokens; the maximum is {MAX_PROMPT_TOKENS}"
        )
    uncond = ids.clone()
    uncond[:, 1:-2] = AUDIO_CFG_TOKEN_ID
    return torch.cat((ids, uncond), dim=0)


@dataclass
class Music3ARRequest:
    """One song in the AR stage. Callbacks run on the AR thread."""

    prompt_ids: torch.Tensor  # [2, P] on CPU
    max_frames: int
    seed: int
    # (frames [n, 8 * hidden] on the AR device, first frame index) per step batch.
    on_frames: Callable[[torch.Tensor, int], None]
    on_done: Callable[[int, BaseException | None], None]
    # --- engine-owned state ---
    frames_emitted: int = 0
    frame_index: int = 0  # Diffusers' loop counter (frame 0 is not emitted)
    length: int = 0  # tokens per row in the KV cache
    pages: list[int] = field(default_factory=list)
    pages_per_row: int = 0
    # Prefill output for both rows, consumed by the pair's first frame step.
    last_hidden: torch.Tensor | None = None


class _Kernels:
    """The ``tokenspeed-kernel`` ops the frame step uses (bound once)."""

    def __init__(self):
        from tokenspeed_kernel.ops.activation import silu_and_mul
        from tokenspeed_kernel.ops.gemm import mm
        from tokenspeed_kernel.ops.layernorm import qk_rmsnorm_rope_neox, rmsnorm

        self.mm = mm
        self.rmsnorm = rmsnorm
        self.qk_rmsnorm_rope = qk_rmsnorm_rope_neox
        self.silu_and_mul = silu_and_mul


def _fuse_linears(*linears) -> torch.Tensor:
    """Concatenate bias-free ``nn.Linear`` weights ``[N_i, K]`` into one ``[sum N_i, K]``.

    Each module's weight is rebound to its row slice of the fused tensor.
    """
    for linear in linears:
        if linear.bias is not None:
            raise ValueError("music3 AR fuses bias-free projections only")
    fused = torch.cat([linear.weight.data for linear in linears]).contiguous()
    start = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        linear.weight.data = fused[start : start + rows]
        start += rows
    return fused


class _PagePool:
    def __init__(self, num_pages: int):
        self.free = list(range(num_pages - 1, 0, -1))  # page 0 = scratch

    def take(self, count: int) -> list[int] | None:
        if count > len(self.free):
            return None
        taken = [self.free.pop() for _ in range(count)]
        return taken

    def give(self, pages: list[int]) -> None:
        self.free.extend(pages)


class Music3AREngine:
    """Owns the LM + depth decoder on one device and runs batched frame steps.

    Not thread-safe: ``admit`` / ``step`` are called from one AR thread.
    """

    def __init__(
        self,
        language_model,
        rvq_depth_decoder,
        *,
        device: str,
        kv_cache_gib: float,
        max_pairs: int,
        use_cuda_graph: bool,
    ):
        if max_pairs not in PAIR_BUCKETS:
            raise ValueError(f"max_pairs={max_pairs} must be one of {PAIR_BUCKETS}")
        # tokenspeed-kernel probes the GPU platform at import; import it with
        # the engine so the sampler helpers stay importable on CPU hosts.
        from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

        self.decode_attention = mha_decode_with_kvcache
        self.kernels = _Kernels()
        self.lm = language_model
        self.depth = rvq_depth_decoder
        self.device = torch.device(device)
        self.max_pairs = max_pairs
        self.use_cuda_graph = use_cuda_graph
        cfg = language_model.config
        self.hidden = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_hidden_layers)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )
        self.max_len = int(cfg.max_position_embeddings)
        self.num_codebooks = int(rvq_depth_decoder.config.num_codebooks)
        self.audio_vocab = int(rvq_depth_decoder.config.audio_vocab_size)
        dtype = language_model.dtype
        self.q_heads = int(cfg.num_attention_heads)
        self.lm_eps = float(cfg.rms_norm_eps)
        self.depth_heads = int(rvq_depth_decoder.config.num_attention_heads)
        self.depth_eps = float(rvq_depth_decoder.norm.eps)

        # Sliced LM head (audio codes + EOS), same weights as the full head,
        # padded with zero rows to a multiple of 16 output columns (skinny GEMM
        # shape); the padding columns are sliced off before sampling.
        weight = language_model.lm_head.weight
        head = torch.cat(
            (
                weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE],
                weight[AUDIO_END_TOKEN_ID : AUDIO_END_TOKEN_ID + 1],
            )
        )
        padded = -(-head.shape[0] // 16) * 16
        self.head_weight = torch.cat(
            (head, head.new_zeros(padded - head.shape[0], head.shape[1]))
        ).contiguous()

        # Fused projection weights; the module weights become views into them
        # (no extra memory, and the HF prefill keeps working).
        self.lm_qkv, self.lm_gate_up = [], []
        for layer in language_model.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            self.lm_qkv.append(_fuse_linears(attn.q_proj, attn.k_proj, attn.v_proj))
            self.lm_gate_up.append(_fuse_linears(mlp.gate_proj, mlp.up_proj))
        self.depth_qkv, self.depth_gate_up = [], []
        for layer in rvq_depth_decoder.layers:
            self.depth_qkv.append(
                _fuse_linears(layer.attn.to_q, layer.attn.to_k, layer.attn.to_v)
            )
            self.depth_gate_up.append(_fuse_linears(layer.gate_proj, layer.up_proj))
        # RoPE cos|sin cache from the HF module's own inverse frequencies.
        inv_freq = language_model.model.rotary_emb.inv_freq.float().to(self.device)
        angles = (
            torch.arange(self.max_len, dtype=torch.float32, device=self.device)[:, None]
            * inv_freq[None]
        )
        self.cos_sin_cache = torch.cat(
            (angles.cos(), angles.sin()), dim=-1
        ).contiguous()
        self.lm_columns = column_keys(SEMANTIC_VOCAB_SIZE + 1, self.device)
        self.depth_columns = column_keys(self.audio_vocab, self.device)

        bytes_per_page = (
            2
            * self.num_layers
            * PAGE_SIZE
            * self.kv_heads
            * self.head_dim
            * dtype.itemsize
        )
        num_pages = max(2, int(kv_cache_gib * 2**30 // bytes_per_page))
        shape = (num_pages, PAGE_SIZE, self.kv_heads, self.head_dim)
        self.k_cache = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.pool = _PagePool(num_pages)
        self.pages_per_row = math.ceil(self.max_len / PAGE_SIZE)
        logger.info(
            f"music3 AR: kv pages={num_pages} x {PAGE_SIZE} tokens "
            f"({num_pages * bytes_per_page / 2**30:.1f} GiB), max_pairs={max_pairs}, "
            f"cuda_graph={use_cuda_graph}"
        )

        rows = 2 * max_pairs
        dev = self.device
        # Static step inputs (row r = 2 * pair + {0 cond, 1 uncond}).
        self.in_hidden = torch.zeros(rows, self.hidden, dtype=dtype, device=dev)
        self.in_positions = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.in_write = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.in_seqlens = torch.ones(rows, dtype=torch.int32, device=dev)
        self.in_page_table = torch.zeros(
            rows, self.pages_per_row, dtype=torch.int32, device=dev
        )
        self.in_seeds = torch.zeros(max_pairs, dtype=torch.int64, device=dev)
        self.in_frame = torch.zeros(max_pairs, dtype=torch.int64, device=dev)
        # Static step outputs.
        self.out_hidden = torch.zeros_like(self.in_hidden)
        self.out_frame = torch.zeros(
            max_pairs, self.num_codebooks * self.hidden, dtype=dtype, device=dev
        )
        self.out_eos = torch.zeros(max_pairs, dtype=torch.bool, device=dev)
        self.host_eos = torch.zeros(max_pairs, dtype=torch.bool).pin_memory()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Highest-priority stream: the AR stage is latency-critical and may
        # share its GPU with low-priority acoustic work.
        _, highest = torch.cuda.Stream.priority_range()
        self.stream = torch.cuda.Stream(self.device, priority=highest)

        self.active: list[Music3ARRequest] = []

    # ------------------------------------------------------------------ math
    def _decode_layers(self, rows: int) -> torch.Tensor:
        """LM decode of ``in_hidden[:rows]`` (frame feedback embeddings).

        The Qwen3 decoder layer with fused kernels: one QKV GEMM, per-head
        Q/K RMSNorm + neox RoPE in one launch, paged decode attention, fused
        residual-add RMSNorm, one gate/up GEMM + SiLU-and-mul.
        """
        kops = self.kernels
        model = self.lm.model
        layers = model.layers
        hd, nq, nkv = self.head_dim, self.q_heads, self.kv_heads
        positions = self.in_positions[:rows]
        write = self.in_write[:rows]
        residual = self.in_hidden[:rows].clone()
        h = kops.rmsnorm(residual, layers[0].input_layernorm.weight, self.lm_eps)
        for index, layer in enumerate(layers):
            attn = layer.self_attn
            qkv = kops.mm(h, self.lm_qkv[index])
            q = qkv[:, : nq * hd].view(rows, nq, hd)
            k = qkv[:, nq * hd : (nq + nkv) * hd].view(rows, nkv, hd)
            v = qkv[:, (nq + nkv) * hd :].view(rows, nkv, hd)
            q, k = kops.qk_rmsnorm_rope(
                q,
                k,
                attn.q_norm.weight,
                attn.k_norm.weight,
                self.lm_eps,
                positions,
                self.cos_sin_cache,
            )
            self.k_cache[index].view(-1, nkv, hd).index_copy_(
                0, write, k.view(rows, nkv, hd)
            )
            self.v_cache[index].view(-1, nkv, hd).index_copy_(0, write, v)
            out = self.decode_attention(
                q.view(rows, nq, hd),
                self.k_cache[index],
                self.v_cache[index],
                page_table=self.in_page_table[:rows],
                cache_seqlens=self.in_seqlens[:rows],
                # Bound only (lengths come from cache_seqlens); a constant so
                # the selected kernel never respecializes per step.
                max_seqlen_k=self.max_len,
                max_seqlen_q=1,
                softmax_scale=attn.scaling,
            )
            o = kops.mm(out.reshape(rows, nq * hd), attn.o_proj.weight)
            h, residual = kops.rmsnorm(
                o, layer.post_attention_layernorm.weight, self.lm_eps, residual=residual
            )
            mlp = layer.mlp
            m = kops.mm(
                kops.silu_and_mul(kops.mm(h, self.lm_gate_up[index])),
                mlp.down_proj.weight,
            )
            next_norm = (
                layers[index + 1].input_layernorm.weight
                if index + 1 < len(layers)
                else model.norm.weight
            )
            h, residual = kops.rmsnorm(m, next_norm, self.lm_eps, residual=residual)
        return h  # = model.norm(final residual stream)

    def _depth_pass(
        self, tokens: torch.Tensor, cache: list, first: int
    ) -> torch.Tensor:
        """Depth decoder over new ``tokens`` ``[rows, t, H]`` at positions ``first..``.

        Equivalent to Diffusers' full-sequence causal recompute: earlier
        tokens never attend to later ones, so their keys/values are cached
        across the 7 codebook passes. Returns the normed hidden of the last
        new token, ``[rows, H]``.
        """
        kops = self.kernels
        depth = self.depth
        rows, t, _ = tokens.shape
        heads, hd = self.depth_heads, self.hidden // self.depth_heads
        x = (tokens + depth.pos_embedding.weight[first : first + t]).reshape(
            rows * t, -1
        )
        h = kops.rmsnorm(x, depth.layers[0].input_layernorm.weight, self.depth_eps)
        residual = x
        for index, layer in enumerate(depth.layers):
            qkv = kops.mm(h, self.depth_qkv[index]).view(rows, t, 3, heads, hd)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            if cache[index] is not None:
                k = torch.cat((cache[index][0], k), dim=2)
                v = torch.cat((cache[index][1], v), dim=2)
            cache[index] = (k, v)
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=first == 0 and t > 1
            )
            o = kops.mm(
                attn.transpose(1, 2).reshape(rows * t, heads * hd),
                layer.attn.to_out.weight,
            )
            h, residual = kops.rmsnorm(
                o,
                layer.post_attention_layernorm.weight,
                self.depth_eps,
                residual=residual,
            )
            m = kops.mm(
                kops.silu_and_mul(kops.mm(h, self.depth_gate_up[index])),
                layer.down_proj.weight,
            )
            next_norm = (
                depth.layers[index + 1].input_layernorm.weight
                if index + 1 < len(depth.layers)
                else depth.norm.weight
            )
            h, residual = kops.rmsnorm(m, next_norm, self.depth_eps, residual=residual)
        return h.view(rows, t, -1)[:, -1]

    def _frame_step(self, pairs: int) -> None:
        """One frame for ``pairs`` CFG pairs, reading/writing the static buffers.

        Mirrors Diffusers' loop body: sample the semantic code from
        ``last_hidden``, run the depth decoder, emit the frame hidden, embed the
        frame and decode it. EOS rows still compute (static shapes); the host
        drops their results.
        """
        kops = self.kernels
        rows = 2 * pairs
        last_hidden = self.out_hidden[:rows]
        seeds = self.in_seeds[:pairs]
        frame = self.in_frame[:pairs]
        draws = frame * (self.num_codebooks + 1)

        logits = kops.mm(last_hidden, self.head_weight)[
            :, : SEMANTIC_VOCAB_SIZE + 1
        ].float()
        conditional, unconditional = logits[0::2], logits[1::2]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        threshold = torch.topk(conditional, AR_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        sampled = _sample_top_k(guided, seeds, draws, self.lm_columns)  # [pairs]
        self.out_eos[:pairs].copy_(sampled == _EOS_SLICE_INDEX)
        # EOS rows feed code 0 so the step stays well-defined; they are dropped.
        semantic = torch.where(sampled == _EOS_SLICE_INDEX, 0, sampled)
        semantic_rows = semantic.repeat_interleave(2)

        depth = self.depth
        embed_tokens = self.lm.model.embed_tokens
        projection = depth.projection.weight
        first_tokens = torch.stack(
            (last_hidden, embed_tokens(semantic_rows + AUDIO_CODE_OFFSET)), dim=1
        )
        tokens = kops.mm(first_tokens.reshape(2 * rows, -1), projection).view(
            rows, 2, -1
        )
        cache: list = [None] * len(depth.layers)
        position = 0
        codes = [semantic_rows]
        parts = []
        for index in range(1, self.num_codebooks):
            hidden = self._depth_pass(tokens, cache, position)
            position += tokens.shape[1]
            parts.append(hidden[0::2])
            head = kops.mm(hidden, depth.audio_heads[index - 1].weight)
            cond, uncond = head[0::2].float(), head[1::2].float()
            code = _sample_top_k(
                uncond + (cond - uncond) * AR_CFG_SCALE,
                seeds,
                draws + index,
                self.depth_columns,
            ).repeat_interleave(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embed = depth.audio_embeddings(code + (index - 1) * self.audio_vocab)
                tokens = kops.mm(embed, projection).view(rows, 1, -1)
        self.out_frame[:pairs].copy_(torch.cat([last_hidden[0::2], *parts], dim=-1))

        # _embed_audio_frame on [rows, num_codebooks] codes.
        frame_codes = torch.stack(codes, dim=1)
        embeds = embed_tokens(frame_codes[:, :1] + AUDIO_CODE_OFFSET)
        offsets = (
            torch.arange(self.num_codebooks - 1, device=self.device) * self.audio_vocab
        ).unsqueeze(0)
        extra = depth.audio_embeddings(frame_codes[:, 1:] + offsets).sum(
            dim=1, keepdim=True
        )
        feedback = (embeds + extra.to(embeds.dtype)) * self.num_codebooks**-0.5
        self.in_hidden[:rows].copy_(feedback[:, 0])
        self.out_hidden[:rows].copy_(self._decode_layers(rows))

    # -------------------------------------------------------------- control
    def can_admit(self, req: Music3ARRequest) -> bool:
        need = math.ceil(self._row_tokens(req) / PAGE_SIZE) * 2
        return len(self.active) < self.max_pairs and need <= len(self.pool.free)

    def _row_tokens(self, req: Music3ARRequest) -> int:
        # Prompt + one feedback token per frame + one spare step.
        return int(req.prompt_ids.shape[1]) + req.max_frames + 2

    @torch.no_grad()
    def admit(self, req: Music3ARRequest) -> None:
        """Prefill ``req`` (HF forward, as Diffusers) and add its pair to the batch."""
        with torch.cuda.stream(self.stream):
            self._admit(req)
        self.stream.synchronize()

    def _admit(self, req: Music3ARRequest) -> None:
        prompt_len = int(req.prompt_ids.shape[1])
        if self._row_tokens(req) > self.max_len:
            raise ValueError(
                f"prompt ({prompt_len} tokens) + {req.max_frames} frames exceeds the "
                f"checkpoint context of {self.max_len} tokens"
            )
        per_row = math.ceil(self._row_tokens(req) / PAGE_SIZE)
        pages = self.pool.take(2 * per_row)
        if pages is None:
            raise RuntimeError("music3 AR: KV cache full (admit after can_admit)")
        req.pages = pages
        try:
            ids = req.prompt_ids.to(self.device)
            output = self.lm.model(
                inputs_embeds=self.lm.model.embed_tokens(ids), use_cache=True
            )
            cache = output.past_key_values
            for row in range(2):
                row_pages = torch.tensor(
                    pages[row * per_row : (row + 1) * per_row], device=self.device
                )
                slots = (
                    row_pages[:, None] * PAGE_SIZE
                    + torch.arange(PAGE_SIZE, device=self.device)
                ).flatten()
                slots = slots[:prompt_len]
                for index in range(self.num_layers):
                    layer = cache.layers[index]
                    # [2, kv_heads, P, dim] -> [P, kv_heads, dim]
                    self.k_cache[index].view(
                        -1, self.kv_heads, self.head_dim
                    ).index_copy_(0, slots, layer.keys[row].transpose(0, 1))
                    self.v_cache[index].view(
                        -1, self.kv_heads, self.head_dim
                    ).index_copy_(0, slots, layer.values[row].transpose(0, 1))
            req.length = prompt_len
            req.frame_index = 0
            req.frames_emitted = 0
            req.last_hidden = output.last_hidden_state[:, -1].contiguous()
            req.pages_per_row = per_row
        except BaseException:
            self.pool.give(pages)
            req.pages = []
            raise
        self.active.append(req)

    def _load_inputs(self, bucket: int) -> None:
        """Write the active pairs (+ scratch padding) into the static buffers."""
        rows = 2 * bucket
        positions = torch.zeros(rows, dtype=torch.int64)
        write = torch.zeros(rows, dtype=torch.int64)  # scratch page 0 for padding
        seqlens = torch.ones(rows, dtype=torch.int32)
        page_table = torch.zeros(rows, self.pages_per_row, dtype=torch.int32)
        seeds = torch.zeros(bucket, dtype=torch.int64)
        frame = torch.zeros(bucket, dtype=torch.int64)
        for pair, req in enumerate(self.active):
            per_row = req.pages_per_row
            seeds[pair] = int(req.seed) & _M32
            frame[pair] = req.frame_index
            for row in range(2):
                r = 2 * pair + row
                row_pages = req.pages[row * per_row : (row + 1) * per_row]
                page_table[r, :per_row] = torch.tensor(row_pages, dtype=torch.int32)
                position = req.length  # the feedback token's position
                positions[r] = position
                write[r] = (
                    row_pages[position // PAGE_SIZE] * PAGE_SIZE + position % PAGE_SIZE
                )
                seqlens[r] = position + 1
        self.in_positions[:rows].copy_(positions, non_blocking=True)
        self.in_write[:rows].copy_(write, non_blocking=True)
        self.in_seqlens[:rows].copy_(seqlens, non_blocking=True)
        self.in_page_table[:rows].copy_(page_table, non_blocking=True)
        self.in_seeds[:bucket].copy_(seeds, non_blocking=True)
        self.in_frame[:bucket].copy_(frame, non_blocking=True)

    def _run(self, bucket: int) -> None:
        if not self.use_cuda_graph:
            self._frame_step(bucket)
            return
        graph = self.graphs.get(bucket)
        if graph is None:
            # Warm up (Triton JIT, cuBLAS handles) outside capture, then record.
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            saved = (self.out_hidden.clone(), self.in_hidden.clone())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._frame_step(bucket)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._frame_step(bucket)
            self.graphs[bucket] = graph
            # The warmup steps advanced the static buffers; restore them. Their
            # only cache writes went to this step's feedback slot per row, which
            # the replay below rewrites from the restored inputs.
            self.out_hidden.copy_(saved[0])
            self.in_hidden.copy_(saved[1])
            logger.info(f"music3 AR: captured frame-step graph for {bucket} pairs")
        graph.replay()

    @torch.no_grad()
    def step(self) -> None:
        """One frame for every active pair; emits frames and retires finished pairs."""
        if not self.active:
            return
        with torch.cuda.stream(self.stream):
            self._step()

    def _step(self) -> None:
        bucket = next(b for b in PAIR_BUCKETS if b >= len(self.active))
        for pair, req in enumerate(self.active):
            if req.last_hidden is not None:
                self.out_hidden[2 * pair : 2 * pair + 2].copy_(req.last_hidden)
                req.last_hidden = None
        self._load_inputs(bucket)
        self._run(bucket)
        # Snapshot before the next step reuses the static output buffer.
        frames = self.out_frame[: len(self.active)].clone()
        self.host_eos.copy_(self.out_eos, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        keep: list[Music3ARRequest] = []
        moved: list[tuple[int, int]] = []  # (old pair, new pair) for live state
        for pair, req in enumerate(self.active):
            finished = bool(self.host_eos[pair])
            if not finished:
                if req.frame_index > 0:
                    req.on_frames(frames[pair : pair + 1], req.frames_emitted)
                    req.frames_emitted += 1
                    finished = req.frames_emitted >= req.max_frames
                if not finished:
                    req.frame_index += 1
                    req.length += 1
            if finished:
                self._retire(req, None)
            else:
                moved.append((pair, len(keep)))
                keep.append(req)
        if len(keep) != len(self.active):
            # Compact the per-row decode state of surviving pairs.
            hidden = self.out_hidden.clone()
            for old, new in moved:
                if old != new:
                    self.out_hidden[2 * new : 2 * new + 2].copy_(
                        hidden[2 * old : 2 * old + 2]
                    )
        self.active = keep

    def _retire(self, req: Music3ARRequest, error: BaseException | None) -> None:
        self.pool.give(req.pages)
        req.pages = []
        if error is None and req.frames_emitted == 0:
            error = ValueError(
                "MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately"
            )
        req.on_done(req.frames_emitted, error)

    def fail_all(self, error: BaseException) -> None:
        for req in self.active:
            self._retire(req, error)
        self.active = []
