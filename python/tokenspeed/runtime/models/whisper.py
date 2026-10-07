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

"""Whisper (encoder-decoder ASR).

Structure, and why it differs from the other ASR model in this tree:

``qwen3_asr.py`` is decoder-only with an audio prefix -- the audio tower's
output is spliced into the LM's input embeddings and the rest is ordinary
causal decode. Whisper is a real encoder-decoder: the decoder cross-attends
to a fixed encoder output that is computed once and read by every decode
step. That fixed KV is the whole design question, and the four reference
engines disagree about it (``unieinfra/xref/15-encoder-embedding.md`` §3.3):
vLLM gives it its own KV spec, manager and cache group; TRT-LLM gives it a
separate resource manager with its own slice of the KV budget; SGLang folds
it into the ordinary paged pool by offsetting the layer id so cross-attention
looks like ``encoder_layers + decoder_layers + i``.

We are on vLLM/TRT-LLM's side -- cross KV lives outside the paged arena --
but implemented as a plain preallocated per-slot buffer rather than a fourth
cache family, because cross KV needs nothing the arena provides:

  * it never grows (one write of ``encoder_len`` rows per request),
  * it is never prefix-matched (encoder output is unique per audio, which is
    why vLLM's CrossAttentionManager raises on ``find_longest_cache_hit``),
  * it is never evicted independently of its request.

The cost is bounded and small: ``decoder_layers x encoder_len x d_model x 2``
per slot -- 30 MB per request on large-v3-turbo (4 decoder layers, 1500
encoder frames), against ~1.2 GB for the same request's share of a paged
group sized for it. Keeping it out of the arena also keeps this port off the
shared C++ scheduler, which several sessions share.

Cross-attention itself needs no new kernel: ``flash_attn_varlen_func``
already takes ``cu_seqlens_q`` and ``cu_seqlens_k`` separately (the encoder
wrappers in ``mm_encoder_attention.py`` merely pass the same tensor twice),
which is exactly the "q from the decoder, kv from the encoder" shape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.multimodal.inputs import Modality
from tokenspeed.runtime.utils import add_prefix


class WhisperEncoderLayer(nn.Module):
    """One encoder block: bidirectional attention, no KV cache.

    Reuses ``MultimodalEncoderAttention`` -- the same non-causal, pool-free
    layer the vision and audio towers use. Whisper's encoder is exactly that
    shape, so it needs no encoder-specific attention code.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = MultimodalEncoderAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            mapping=mapping,
            # Whisper ships no bias for k_proj. The fused QKV bias is
            # allocated anyway and the k slice is zero-filled at load time
            # (see load_weights) -- a zero bias is exactly "no bias".
            qkv_bias=True,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]

        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        attended = self.self_attn(
            normed.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ).squeeze(0)
        hidden_states = residual + attended

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    """Log-mel frames -> a fixed-length sequence of encoder states.

    Runs once per request. The output is what every decode step cross-attends
    to; nothing here touches the paged KV pool.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Whisper's encoder positions are fixed sinusoids shipped as weights,
        # not a learned table; registered as a buffer so they load like one
        # and never appear in the optimizer's eyes.
        self.register_buffer(
            "embed_positions",
            torch.zeros(config.max_source_positions, embed_dim),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """``input_features``: [batch, num_mel_bins, frames] -> [batch*T, d]."""
        features = input_features.to(device=self.device, dtype=self.dtype)
        hidden = F.gelu(self.conv1(features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)

        batch, seq_len, _ = hidden.shape
        if seq_len > self.embed_positions.shape[0]:
            raise ValueError(
                f"Whisper encoder got {seq_len} frames but the position table "
                f"holds {self.embed_positions.shape[0]}; the feature extractor "
                "must pad or trim to the model's window."
            )
        hidden = hidden + self.embed_positions[:seq_len].to(hidden.dtype)

        # One packed varlen sequence per batch row, all the same length: the
        # encoder window is fixed, so cu_seqlens is a plain arange.
        flat = hidden.reshape(batch * seq_len, -1)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=flat.device,
            dtype=torch.int32,
        )
        for layer in self.layers:
            flat = layer(flat, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
        return self.layer_norm(flat)


class CrossAttentionCache:
    """Per-request encoder K/V for the decoder's cross-attention.

    Deliberately not a paged-arena cache group. Cross KV is written once,
    never grows, is never prefix-matched and is never evicted apart from its
    request, so none of the arena's machinery applies to it (see the module
    docstring, and §3.3 of the xref).

    Layout is ``[layer, slot, encoder_len, kv_heads, head_dim]`` -- layer
    first so that ``k[layer]`` is contiguous and reshaping it to the packed
    ``[slot * encoder_len, heads, dim]`` that flash-attn wants is a view, not
    a copy. With slot first, that reshape silently materialises the whole
    buffer on every layer of every step.

    Slots are indexed by ``req_pool_index``, which the scheduler bounds at
    ``per_rank_max_batch + 1`` (``event_loop.py:369``); ModelConfig stamps
    that bound on the HF config as ``cross_attn_slots``.
    """

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        encoder_len: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.encoder_len = encoder_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        shape = (num_layers, num_slots, encoder_len, kv_heads, head_dim)
        # Allocated once, up front. Growing it lazily would mean reading
        # ``slots.max()`` on the read path, and that is a device-to-host sync
        # on every decode step -- illegal under CUDA graph capture and a
        # per-step stall otherwise.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def bytes_total(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def write(
        self,
        slot: int,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Store one request's encoder K/V for one decoder layer."""
        if not 0 <= slot < self.num_slots:
            raise IndexError(
                f"cross-attention slot {slot} outside the pool of "
                f"{self.num_slots}; the buffer is sized from max_num_seqs"
            )
        if k.shape[0] != self.encoder_len:
            raise ValueError(
                f"cross-attention KV for slot {slot} has {k.shape[0]} rows but "
                f"the cache was built for {self.encoder_len}; Whisper's encoder "
                "window is fixed, so a different length means the feature "
                "extractor did not pad or trim to it."
            )
        self.k[layer_id, slot].copy_(k)
        self.v[layer_id, slot].copy_(v)

    def whole_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Every slot's K/V for one layer, packed, as a view.

        The decode fast path: rather than gathering the batch's slots into a
        contiguous block, attend over all of them and let the query scatter
        pick out the rows that matter. Nothing is copied, and the shape is
        constant, which is also what lets a CUDA graph capture it.

        Two measurements, and the difference between them is the point.
        Isolating the kernels (gather + attention only) says:

            large-v3-turbo  bs=16  1.864 ms -> 0.247 ms   7.55x

        Measuring the whole cross-attention block -- layer norm, the query
        and output projections, the scatter -- says:

            large-v3-turbo  bs=1   0.671 ms -> 0.766 ms   0.88x
                            bs=8   0.746 ms -> 0.774 ms   0.96x
                            bs=16  1.348 ms -> 0.755 ms   1.79x
            large-v3        bs=1   5.006 ms -> 5.777 ms   0.87x
                            bs=8   5.905 ms -> 5.951 ms   0.99x
                            bs=16 10.920 ms -> 5.936 ms   1.84x

        The block number is the one that matters. At small batch the block is
        launch-bound (an empty kernel launch costs 7.7-13.9 us on this box),
        so removing a bandwidth cost buys nothing there, and attending over
        every slot to serve one request costs ~13%. The gather grows with
        batch size and this does not, which is where the win comes from --
        and why the caller picks between the two by occupancy rather than
        taking this unconditionally.
        """
        flat = (self.num_slots * self.encoder_len, self.kv_heads, self.head_dim)
        return self.k[layer_id].view(flat), self.v[layer_id].view(flat)

    def gather(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The batch's slots only, as ``[layers, bs, ...]``.

        Used on steps that carry a prefill. Those are not CUDA-graph captured
        and happen once per request, so the copy is affordable there; it is
        the per-decode-step copy that :meth:`whole_buffer` exists to avoid.
        """
        index = slots.to(device=self.k.device, dtype=torch.long)
        return self.k.index_select(1, index), self.v.index_select(1, index)


class WhisperDecoderSelfAttention(nn.Module):
    """Causal self-attention over the decoded tokens; ordinary paged KV."""

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_heads,
            total_heads,
            # Whisper's k_proj carries no bias; the fused bias exists and its
            # k slice is zero-filled at load time.
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # The reference implementations pre-multiply q by the scale and build
        # the attention with scaling=1.0. Passing the scale instead is the
        # same arithmetic with one fewer elementwise pass.
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            rotary_emb=None,
            qk_norm=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Whisper has no RoPE on decoder self-attn; positions still required by
        # the nctu6 PagedAttention prologue signature.
        attended = self.attn(q, k, v, positions, ctx)
        output, _ = self.out_proj(attended)
        return output


class WhisperCrossAttention(nn.Module):
    """Decoder attends to the fixed encoder output.

    Two shapes, one kernel. At prefill the encoder states are in hand, so
    K/V are projected, stored, and used directly. At decode the states are
    long gone and K/V come back from :class:`CrossAttentionCache`. Both go
    through ``flash_attn_varlen_func`` with *separate* q and k cu_seqlens --
    the API already supports that; the encoder wrappers in
    ``mm_encoder_attention.py`` simply pass one tensor twice because
    self-attention has q_len == kv_len.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.layer_id = layer_id
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # Whisper ships k_proj without a bias. Kept as a separate projection
        # rather than fused with v: the two are only ever applied together at
        # prefill, so fusing would buy nothing and would need the same
        # zero-bias patch as the self-attention path.
        self.k_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.v_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def project_encoder_kv(
        self, encoder_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one request's encoder output into this layer's K/V."""
        k, _ = self.k_proj(encoder_states)
        v, _ = self.v_proj(encoder_states)
        return (
            k.view(-1, self.num_heads, self.head_dim),
            v.view(-1, self.num_heads, self.head_dim),
        )

    def project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        return q.view(-1, self.num_heads, self.head_dim)

    def attend(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
            cross_attn_varlen,
        )

        return cross_attn_varlen(
            q,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scaling,
        )

    def project_out(self, attended: torch.Tensor) -> torch.Tensor:
        output, _ = self.out_proj(attended.reshape(-1, self.num_heads * self.head_dim))
        return output


@dataclasses.dataclass
class CrossAttentionPlan:
    """Everything the decoder layers need to run cross-attention this step.

    Built once per forward rather than per layer: the cumulative-length
    tensors and the query scatter buffer are identical across layers, and
    rebuilding them 4 (or 32) times a step is pure launch overhead.

    ``padded`` selects the decode fast path -- attend over every slot with
    the queries scattered by pool index -- against the gather path used on
    steps that carry a prefill. See ``CrossAttentionCache.whole_buffer``.
    """

    cache: "CrossAttentionCache"
    padded: bool
    slots: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    encoder_len: int
    max_seqlen_q: int = 1
    query_buffer: torch.Tensor | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.layer_id = layer_id
        self.self_attn = WhisperDecoderSelfAttention(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn = WhisperCrossAttention(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("encoder_attn", prefix),
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.decoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            reduce_results=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        cross: "CrossAttentionPlan",
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, ctx)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        query = self.encoder_attn.project_query(hidden_states)
        if cross.padded:
            # Padded path: q row j is slot j, so the batch's queries are
            # scattered by pool index into a full-width buffer, attended over
            # the whole KV buffer, then read back out by the same index.
            # Both touch only ``bs`` rows -- tens of KB against the hundreds
            # of MB the gather it replaces would move.
            # Not zeroed per layer: rows outside the batch keep whatever a
            # previous step left, their outputs are discarded, and a zero_()
            # here is one more kernel launch per layer on a path that is
            # launch-bound at small batch. The buffer is zeroed once at
            # allocation so the first steps read zeros, not garbage.
            cross.query_buffer.index_copy_(0, cross.slots, query)
            attended = self.encoder_attn.attend(
                cross.query_buffer,
                *cross.cache.whole_buffer(self.layer_id),
                cu_seqlens_q=cross.cu_seqlens_q,
                cu_seqlens_k=cross.cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=cross.encoder_len,
            )
            attended = attended.index_select(0, cross.slots)
        else:
            attended = self.encoder_attn.attend(
                query,
                cross.key[self.layer_id].reshape(-1, *cross.key.shape[3:]),
                cross.value[self.layer_id].reshape(-1, *cross.value.shape[3:]),
                cu_seqlens_q=cross.cu_seqlens_q,
                cu_seqlens_k=cross.cu_seqlens_k,
                max_seqlen_q=cross.max_seqlen_q,
                max_seqlen_k=cross.encoder_len,
            )
        hidden_states = residual + self.encoder_attn.project_out(attended)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperDecoder(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            embed_dim,
            prefix=add_prefix("embed_tokens", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # Learned, unlike the encoder's fixed sinusoids.
        self.embed_positions = nn.Embedding(config.max_target_positions, embed_dim)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    config,
                    mapping,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        cross: CrossAttentionPlan,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, ctx, cross)
        return self.layer_norm(hidden_states)


class WhisperForConditionalGeneration(nn.Module):
    """Whisper as this engine serves it: one encoder pass, then ordinary decode.

    The request carries its audio the same way a multimodal request carries an
    image. What is different from ``qwen3_asr.py`` is where the encoder output
    goes: not into the decoder's input embeddings, but into every decoder
    layer's cross-attention K/V, written once at prefill and read by every
    step after.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        if not is_multimodal_active:
            # --language-model-only strips the tower from a prefix-style
            # multimodal model and still leaves a usable LM. Whisper's decoder
            # has nothing to attend to without its encoder, so there is no
            # meaningful text-only mode to fall back to.
            raise ValueError(
                "--language-model-only is not meaningful for Whisper: the "
                "decoder cross-attends to the encoder on every step, so "
                "dropping the encoder leaves no model to run."
            )
        self.config = config
        self.mapping = mapping
        self.encoder = WhisperEncoder(
            config,
            mapping,
            quant_config=None,  # the tower stays in bf16/fp16
            prefix=add_prefix("model.encoder", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.decoder = WhisperDecoder(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("model.decoder", prefix),
        )
        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            bias=False,
            prefix=add_prefix("proj_out", prefix),
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
        )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=mapping.attn.has_dp,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
            dp_lm_head_tp=mapping.attn.has_dp and mapping.lm_head.has_tp,
        )
        self._cross_attn_path = str(getattr(config, "cross_attn_path", "auto"))
        if self._cross_attn_path not in ("auto", "padded", "gather"):
            raise ValueError(
                f"--cross-attn-path must be auto|padded|gather, got "
                f"{self._cross_attn_path!r}"
            )
        self._pad_cu_seqlens_q: torch.Tensor | None = None
        self._pad_cu_seqlens_k: torch.Tensor | None = None
        self._pad_query_buffer: torch.Tensor | None = None
        self.cross_cache = CrossAttentionCache(
            # Bounded by the request pool, which the scheduler sizes at
            # per_rank_max_batch + 1 (event_loop.py:369). ModelConfig stamps
            # the bound on the HF config because the model is constructed
            # from that config alone.
            num_slots=int(getattr(config, "cross_attn_slots", 0) or 1),
            num_layers=config.decoder_layers,
            encoder_len=config.max_source_positions,
            kv_heads=config.decoder_attention_heads // mapping.attn.tp_size,
            head_dim=config.d_model // config.decoder_attention_heads,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

    def encode_audio(self, input_features: torch.Tensor, slots: list[int]) -> None:
        """Run the encoder once and fill each request's cross-attention K/V."""
        encoder_states = self.encoder(input_features)
        encoder_len = self.config.max_source_positions
        if encoder_states.shape[0] != len(slots) * encoder_len:
            raise ValueError(
                f"encoder produced {encoder_states.shape[0]} rows for "
                f"{len(slots)} request(s); expected {len(slots) * encoder_len}"
            )
        for position, slot in enumerate(slots):
            span = encoder_states[position * encoder_len : (position + 1) * encoder_len]
            for layer_id, layer in enumerate(self.decoder.layers):
                key, value = layer.encoder_attn.project_encoder_kv(span)
                self.cross_cache.write(slot, layer_id, key, value)

    # ------------------------------------------------------------------
    # Multimodal hooks
    # ------------------------------------------------------------------

    def get_multimodal_encoder_specs(self) -> dict:
        """No registered encoder: Whisper's encoder output is not scattered.

        The ``MultimodalEmbedder`` pipeline exists to run a tower and splice
        its output into the LM's input embeddings at placeholder offsets --
        that is what ``qwen3_asr.py`` needs. Whisper's encoder output goes to
        every decoder layer's cross-attention instead, so the model runs the
        encoder itself in :meth:`encode_audio` and registers nothing here.
        """
        return {}

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any) -> list[int]:
        """Audio reserves no positions in the decoder's token sequence.

        A prefix-style audio model pads placeholder tokens so the tower's
        output has somewhere to land. Whisper's decoder sequence is only the
        transcript prompt and what it generates; the audio lives in the
        cross-attention cache, outside this sequence entirely.
        """
        return input_ids

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    #
    # Three things the official layout forces:
    #
    #  * ``k_proj`` carries no bias. The fused QKV bias is allocated for all
    #    three shards, so the k slice must be an explicit zero rather than
    #    uninitialised memory -- a zero bias *is* no bias.
    #  * ``proj_out`` is absent: Whisper ties it to the decoder token
    #    embedding. Tied here from the checkpoint's actual contents rather
    #    than from a config flag, because the checkpoint is the thing that
    #    lacks the tensor.
    #  * the encoder's position table is a fixed sinusoid buffer, not a
    #    learned ``nn.Embedding``.

    # ``model.encoder.`` / ``model.decoder.`` in the checkpoint against
    # ``encoder.`` / ``decoder.`` here: this class is the whole model and
    # does not repeat the HF wrapper level.
    _PREFIX_REWRITES = (
        ("model.encoder.", "encoder."),
        ("model.decoder.", "decoder."),
    )
    _STACKED_QKV = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )

    @classmethod
    def _rewrite(cls, name: str) -> str:
        for source, target in cls._PREFIX_REWRITES:
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        # The encoder tower calls its output projection ``proj``; both stacks
        # are ``out_proj`` in the checkpoint.
        if name.startswith("encoder.layers.") and ".self_attn.out_proj." in name:
            name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        for param_name, param in params_dict.items():
            if param_name.endswith("qkv_proj.bias"):
                param.data.zero_()

        loaded: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = self._rewrite(raw_name)

            if name == "encoder.embed_positions.weight":
                self.encoder.embed_positions.copy_(
                    loaded_weight.to(self.encoder.embed_positions.dtype)
                )
                loaded.add(name)
                continue

            fused_target = None
            if ".self_attn." in name:
                for target, source, shard_id in self._STACKED_QKV:
                    if source in name and name.replace(source, target) in params_dict:
                        fused_target = (name.replace(source, target), shard_id)
                        break
            if fused_target is not None:
                fused_name, shard_id = fused_target
                param = params_dict[fused_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                continue

            if name not in params_dict:
                raise KeyError(
                    f"Whisper checkpoint tensor {raw_name!r} has no destination "
                    f"parameter (looked for {name!r}); the checkpoint layout "
                    "does not match this implementation"
                )
            param = params_dict[name]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(loaded_weight)
            else:
                loader(param, loaded_weight)
            loaded.add(name)

        # Tied output projection: the checkpoint ships no proj_out.
        self.proj_out.weight = self.decoder.embed_tokens.weight
        return loaded

    def _cross_attention_plan(
        self,
        ctx: ForwardContext,
        req_pool_indices: torch.Tensor,
        device: torch.device,
    ) -> CrossAttentionPlan:
        """Choose how this step reads the cross-attention KV.

        A pure decode step takes the padded path: attend over every slot with
        the queries scattered by pool index. A step carrying a prefill takes
        the gather path, because its query segments have different lengths
        per request and the padded layout assumes one row per slot.
        """
        encoder_len = self.config.max_source_positions
        bs = int(req_pool_indices.shape[0])
        slots = req_pool_indices.to(device=device, dtype=torch.long)

        # Occupancy decides, not the mode alone. The padded path always reads
        # every slot's KV, so it wins once the batch fills a good fraction of
        # the pool and loses when it does not. Measured on large-v3-turbo,
        # 17 slots, whole cross-attention block per decode step:
        #
        #     bs= 1  gather 0.671 ms  padded 0.766 ms  0.88x
        #     bs= 8  gather 0.746 ms  padded 0.774 ms  0.96x
        #     bs=16  gather 1.348 ms  padded 0.755 ms  1.79x
        #
        # large-v3 (32 decoder layers) tracks it: 0.87x / 0.99x / 1.84x.
        # Parity sits at about half occupancy, which is where this switches.
        # The branch depends only on bs and a constant, and the decode graph
        # captures one graph per batch-size bucket, so it is stable within
        # any captured graph.
        if ctx.num_extends == 0 and self._use_padded_cross_attention(bs):
            self._ensure_padded_buffers(device)
            return CrossAttentionPlan(
                cache=self.cross_cache,
                padded=True,
                slots=slots,
                cu_seqlens_q=self._pad_cu_seqlens_q,
                cu_seqlens_k=self._pad_cu_seqlens_k,
                encoder_len=encoder_len,
                query_buffer=self._pad_query_buffer,
            )

        key, value = self.cross_cache.gather(req_pool_indices)
        query_lengths = _query_lengths(ctx, bs, device)
        cu_seqlens_q = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.cumsum(query_lengths, dim=0, dtype=torch.int32),
            ]
        )
        cu_seqlens_k = torch.arange(
            0, (bs + 1) * encoder_len, encoder_len, device=device, dtype=torch.int32
        )
        return CrossAttentionPlan(
            cache=self.cross_cache,
            padded=False,
            slots=slots,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            encoder_len=encoder_len,
            # A python int, never ``.max().item()``: reading a device tensor
            # here is a host sync, and outright illegal while a CUDA graph is
            # capturing. flash-attn uses max_seqlen_q only to size its
            # schedule, so an upper bound is as good as the exact value.
            max_seqlen_q=int(ctx.input_num_tokens),
            key=key,
            value=value,
        )

    def _use_padded_cross_attention(self, bs: int) -> bool:
        """Which read path a decode step takes.

        ``auto`` is the measured default; ``padded`` and ``gather`` force one
        so the two can be A/B'd against each other on the same build.
        """
        choice = self._cross_attn_path
        if choice == "padded":
            return True
        if choice == "gather":
            return False
        return bs * 2 >= self.cross_cache.num_slots

    def _ensure_padded_buffers(self, device: torch.device) -> None:
        """Constant tensors for the padded path, built once.

        Every slot contributes exactly one query row and one ``encoder_len``
        key segment, so both cumulative-length tensors are fixed for the life
        of the server -- and being fixed is what lets a CUDA graph capture
        this path.
        """
        if getattr(self, "_pad_cu_seqlens_q", None) is not None:
            return
        slots = self.cross_cache.num_slots
        encoder_len = self.cross_cache.encoder_len
        self._pad_cu_seqlens_q = torch.arange(
            slots + 1, device=device, dtype=torch.int32
        )
        self._pad_cu_seqlens_k = torch.arange(
            0, (slots + 1) * encoder_len, encoder_len, device=device, dtype=torch.int32
        )
        self._pad_query_buffer = torch.zeros(
            slots,
            self.cross_cache.kv_heads,
            self.cross_cache.head_dim,
            device=device,
            dtype=self.cross_cache.k.dtype,
        )

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ):
        req_pool_indices = kwargs.get("req_pool_indices")
        if req_pool_indices is None:
            raise RuntimeError(
                "Whisper needs req_pool_indices to find each request's "
                "cross-attention KV slot"
            )

        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            multimodal_context is not None
            and multimodal_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        ):
            features, slots = _audio_features_for_extend(
                multimodal_context, req_pool_indices
            )
            if features is not None:
                self.encode_audio(features, slots)

        hidden_states = self.decoder(
            input_ids,
            positions,
            ctx,
            self._cross_attention_plan(ctx, req_pool_indices, input_ids.device),
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.proj_out,
            LogitsMetadata.from_forward_context(ctx),
            None,
        )


def _query_lengths(ctx: ForwardContext, bs: int, device: torch.device) -> torch.Tensor:
    """Per-request query-row count for this forward.

    Derived from ``gather_ids`` on an extend (it is ``cumsum(lens) - 1``, so
    differencing recovers the lengths) and fixed at one row per request on a
    decode. Kept here rather than threaded through the layers because
    cross-attention is the only consumer.
    """
    if ctx.num_extends == 0 or ctx.gather_ids is None:
        return torch.ones(bs, device=device, dtype=torch.int32)
    ends = ctx.gather_ids + 1
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    return (ends - starts).to(device=device, dtype=torch.int32)


def _materialize_feature(item: Any) -> torch.Tensor:
    """Get an item's log-mel tensor, pulling it out of shared memory if needed.

    Features cross the process boundary as SHM handles (``publish_shm_features``
    nulls ``feature`` on the way out). Models that go through
    ``MultimodalEmbedder`` get them materialized by its feature transport;
    Whisper does not use that path -- its encoder output feeds
    cross-attention, not input embeddings -- so it consumes the handle here.

    Single-rank only for now: the transport also decides which rank owns which
    item under TP, and doing that wrong would give one rank a silently empty
    tensor. Refused rather than guessed.
    """
    if isinstance(item.feature, torch.Tensor):
        return item.feature
    handle = item.feature_shm
    if handle is None:
        raise ValueError(
            "Whisper audio item carries neither a feature tensor nor a shared "
            "memory handle"
        )
    handle.attach()
    tensor = handle.consume()
    item.feature = tensor
    item.feature_shm = None
    return tensor


def _audio_features_for_extend(
    multimodal_context: Any, req_pool_indices: torch.Tensor
) -> tuple[torch.Tensor | None, list[int]]:
    """Collect the log-mel features of the requests being prefilled.

    ``mm_inputs`` is per request and aligned with the batch, so its index is
    the batch position and ``req_pool_indices[position]`` is that request's
    cross-attention slot.

    Only a *fresh* prefill carries audio to encode: a chunked continuation
    (``extend_prefix_lens > 0``) has already had its encoder run and its
    cross KV written, and re-encoding would redo the work and rewrite the
    same slot. Returns ``(None, [])`` when this extend has no fresh audio.
    """
    mm_inputs = getattr(multimodal_context, "mm_inputs", None)
    if not mm_inputs:
        return None, []
    prefix_lens = getattr(multimodal_context, "extend_prefix_lens", []) or []

    features: list[torch.Tensor] = []
    slots: list[int] = []
    for position, entry in enumerate(mm_inputs):
        if entry is None or position >= req_pool_indices.shape[0]:
            continue
        if position < len(prefix_lens) and prefix_lens[position] > 0:
            continue
        for item in entry.mm_items:
            if item is None or item.modality != Modality.AUDIO:
                continue
            feature = _materialize_feature(item)
            if feature.ndim == 2:
                feature = feature.unsqueeze(0)
            features.append(feature)
            slots.append(int(req_pool_indices[position].item()))
            # One audio window per request: Whisper's encoder consumes a
            # fixed 30s window, and a second item would overwrite the first
            # request's cross KV rather than extend it.
            break
    if not features:
        return None, []
    return torch.cat(features, dim=0), slots


EntryClass = WhisperForConditionalGeneration
