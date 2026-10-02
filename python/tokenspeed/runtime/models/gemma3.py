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

"""Inference-only Gemma 3 text model compatible with HuggingFace weights.

This implements the *text* decoder of ``Gemma3ForConditionalGeneration`` (the
multimodal 4B/12B/27B checkpoints) and the equivalent ``Gemma3ForCausalLM``
(the text-only release). The vision tower of the multimodal checkpoint is
skipped -- text-only inputs are supported, which covers the LLM benchmark.

Gemma 3 differs from both a naive GQA port AND from the Gemma 4 module in this
package. The source of truth is HF ``transformers.models.gemma3`` plus the
checkpoint ``config.json``:

* RMSNorm uses the Gemma ``x_normed * (1 + weight)`` form with zero-centred
  weights (``GemmaRMSNorm``). This is the OPPOSITE of Gemma 4, which switched
  to the standard ``x_normed * weight`` form. Every norm here -- the four
  sandwich norms, the final norm, and the per-head q/k norms -- is a
  ``GemmaRMSNorm``.
* Uniform ``head_dim`` (128 on 27B) across all layers. Unlike Gemma 4 there is
  no per-layer head_dim/kv-head split and no ``attention_k_eq_v``; every layer
  has a real ``v_proj``.
* Per-head QK-norm: a ``GemmaRMSNorm`` over ``head_dim`` applied to q and k
  (per head) BEFORE RoPE. There is NO value norm (Gemma 4 only).
* Attention softmax scaling is ``query_pre_attn_scalar ** -0.5`` (168 on 27B),
  NOT ``head_dim ** -0.5``. This differs from Gemma 4 (whose scaling is 1.0).
* Alternating attention: 5 local sliding-window layers to every 1 global
  full-attention layer (``sliding_window_pattern`` = 6). Local layers use a
  sliding window (1024 on 27B) and the ``rope_local_base_freq`` (10000) RoPE
  base with NO scaling; global layers are full attention and use ``rope_theta``
  (1000000) with the config's ``rope_scaling`` (linear factor 8 on 27B). The
  27B ``config.json`` omits an explicit ``layer_types`` list; it is synthesised
  onto the text config from ``sliding_window_pattern`` by ``ModelConfig`` so
  that the model here and the KV pool (``MHAConfig``) read the same labels.
* GeGLU MLP: ``down(gelu_tanh(gate) * up)``.
* Embedding scaling by ``sqrt(hidden_size)`` in the compute dtype.
* Tied embeddings: ``lm_head`` shares ``embed_tokens``.
* Softcapping: Gemma 3 removed the Gemma 2 logit softcaps; both
  ``final_logit_softcapping`` and ``attn_logit_softcapping`` default to
  ``None``. Whatever the config declares is honoured -- the final cap by the
  shared ``LogitsProcessor``, the attention cap by ``PagedAttention.logit_cap``.

The decoder layer shape (sandwich norms with the residual added AFTER the
post-norm) is identical to Gemma 4, so the layer is hand-written for the same
reason: it does not fit ``BaseDecoderLayer``'s pre-norm shell. Gemma 3 has no
per-layer output scalar (that is a Gemma 4 addition).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import GeluTanhAndMul
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger, make_layers
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# HF Gemma3TextConfig defaults for fields the 27B config.json omits.
_DEFAULT_ROPE_THETA = 1_000_000.0
_DEFAULT_ROPE_LOCAL_BASE_FREQ = 10_000.0
_DEFAULT_SLIDING_WINDOW_PATTERN = 6
_DEFAULT_QUERY_PRE_ATTN_SCALAR = 256


def _text_config(config):
    """Return the Gemma3 *text* sub-config for either the multimodal
    ``Gemma3Config`` (has ``.text_config``) or a bare text config."""
    return getattr(config, "text_config", None) or config


def gemma3_layer_types(config) -> list[str]:
    """Per-layer attention labels for a Gemma 3 text config.

    Prefers an explicit ``layer_types`` list (newer checkpoints ship one).
    Otherwise derives the 5:1 local:global pattern from
    ``sliding_window_pattern`` exactly as HF ``Gemma3TextConfig.__post_init__``
    does: layer ``i`` is ``full_attention`` when ``(i + 1) % pattern == 0`` and
    ``sliding_attention`` otherwise.

    Args:
        config: The Gemma 3 text config.

    Returns:
        A list of ``"sliding_attention"`` / ``"full_attention"`` labels, one
        per hidden layer.
    """
    explicit = getattr(config, "layer_types", None)
    if explicit:
        return list(explicit)
    pattern = int(
        getattr(config, "sliding_window_pattern", _DEFAULT_SLIDING_WINDOW_PATTERN)
        or _DEFAULT_SLIDING_WINDOW_PATTERN
    )
    num_layers = int(config.num_hidden_layers)
    return [
        FULL_ATTENTION if (i + 1) % pattern == 0 else SLIDING_ATTENTION
        for i in range(num_layers)
    ]


def _rope_cache_positions(config) -> int:
    """Rows to precompute in the RoPE cache.

    ``max_position_embeddings`` is 131072 on gemma-3-27b-it; the cache is
    ``[rows, head_dim]`` float32. Nothing can attend past the served context,
    so clamp to it when the server declares a shorter ``max_model_len`` (same
    reasoning as the Gemma 4 module and deepseek_v4).
    """
    max_position = int(getattr(config, "max_position_embeddings", 0) or 0)
    served = global_server_args_dict.get("max_model_len")
    if served:
        clamped = min(max_position, int(served)) if max_position else int(served)
        if max_position and clamped < max_position:
            logger.info(
                "Gemma 3 RoPE cache clamped to %d positions (config declares %d).",
                clamped,
                max_position,
            )
        return clamped
    return max_position


def _build_rope(config, layer_type: str, max_position: int, dtype: torch.dtype):
    """One shared RoPE instance for a Gemma 3 layer type.

    * ``full_attention`` (global) layers use ``rope_theta`` and the config's
      ``rope_scaling`` (e.g. linear factor 8 for the long-context 27B).
    * ``sliding_attention`` (local) layers use ``rope_local_base_freq`` with
      NO scaling -- the local window never needs the extrapolation the global
      layers do.

    ``get_rope`` is cached by (head_size, rotary_dim, max_position, base,
    is_neox, rope_scaling, dtype), so two calls with the same arguments return
    one shared cache; there is no per-layer duplication.
    """
    head_dim = int(config.head_dim)
    if layer_type == FULL_ATTENTION:
        base = float(getattr(config, "rope_theta", _DEFAULT_ROPE_THETA))
        rope_scaling = _normalize_rope_scaling(getattr(config, "rope_scaling", None))
    else:
        base = float(
            getattr(config, "rope_local_base_freq", _DEFAULT_ROPE_LOCAL_BASE_FREQ)
        )
        rope_scaling = None
    return get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=base,
        rope_scaling=rope_scaling,
        dtype=dtype,
    )


def _normalize_rope_scaling(rope_scaling) -> dict | None:
    """Reduce a config's ``rope_scaling`` to the flat dict ``get_rope`` expects.

    ``get_rope`` builds a cache key from ``rope_scaling`` and only tuple-ifies
    LIST values; a value that is itself a dict makes the key unhashable
    (``TypeError: unhashable type: 'dict'``). Newer transformers configs can
    carry exactly that -- a nested ``rope_parameters``-style mapping, or a
    default/None ``rope_type`` -- so flatten to only what the RoPE variants
    here read: ``rope_type`` plus its scalar params (e.g. ``factor``).

    Returns ``None`` when there is no scaling to apply (missing, or an explicit
    ``rope_type == "default"``), so the caller builds a plain RoPE.
    """
    if not rope_scaling or not isinstance(rope_scaling, dict):
        return None
    # Some configs nest the real scaling under a single attention-type key
    # (e.g. {"full_attention": {"rope_type": "linear", ...}}). Unwrap one level
    # when every value is a dict and none of the RoPE keys are present at top.
    if "rope_type" not in rope_scaling and "type" not in rope_scaling:
        nested = [v for v in rope_scaling.values() if isinstance(v, dict)]
        if len(nested) == 1:
            rope_scaling = nested[0]
    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
    if rope_type == "default":
        return None
    flat: dict[str, object] = {"rope_type": rope_type}
    # Copy only scalar params through; drop any nested/non-scalar entries that
    # would make the get_rope cache key unhashable. This covers linear/dynamic
    # ("factor", "alpha") and the extra llama3/yarn/longrope scalars.
    for key, value in rope_scaling.items():
        if key in ("rope_type", "type"):
            continue
        if isinstance(value, (int, float, str, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = list(value)
    return flat


class Gemma3MLP(nn.Module):
    """GeGLU MLP: ``down(gelu_tanh(gate) * up)``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Unsupported activation {hidden_activation!r}; Gemma 3 uses "
                "gelu_pytorch_tanh."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.act_fn = GeluTanhAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Gemma3Attention(nn.Module):
    """Gemma 3 attention: GQA with per-head qk-norm, per-layer-type RoPE and a
    per-layer sliding window. Uniform ``head_dim`` and kv-head count across
    layers (unlike Gemma 4)."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        layer_type: str,
        rotary_emb,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_type = layer_type
        self.layer_id = layer_id
        # Stock TokenSpeed binds cache groups at executor startup from
        # ModelConfig/MHAConfig layer_types (bind_cache_groups). Models do not
        # pass group_id into PagedAttention.
        self.is_sliding = layer_type == SLIDING_ATTENTION
        self.rotary_emb = rotary_emb

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group

        hidden_size = config.hidden_size
        self.head_dim = int(config.head_dim)
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        validate_attention_partition(total_num_heads, total_num_kv_heads, tp_size)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 3 scales the attention scores by query_pre_attn_scalar ** -0.5
        # (168 on 27B), NOT head_dim ** -0.5.
        query_pre_attn_scalar = float(
            getattr(config, "query_pre_attn_scalar", _DEFAULT_QUERY_PRE_ATTN_SCALAR)
        )
        self.scaling = query_pre_attn_scalar**-0.5

        attention_bias = bool(getattr(config, "attention_bias", False))
        # Fused QKV: one GEMM over the shared input instead of three separate
        # q/k/v projections. The three cuBLAS calls read the same activation
        # and under-use the weight-read pipeline on the narrow k/v shapes
        # (N = num_kv_heads * head_dim); one [q|k|v] GEMM amortizes that. The
        # per-head q/k norms still apply after the split (a view), so this is
        # transparent to the norm + RoPE path.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("o_proj", prefix),
        )

        # Per-head qk-norm in the Gemma (1 + weight) form.
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Attention logit softcapping (None on Gemma 3 -> 0.0 = disabled).
        logit_cap = float(getattr(config, "attn_logit_softcapping", None) or 0.0)

        # HF's ``sliding_window`` counts the current token (inclusive); the
        # engine stores ``window_left``, which it treats as EXCLUSIVE of the
        # current token, so a local layer that admits ``sliding_window`` HF
        # positions must pass ``sliding_window - 1`` here. gpt_oss, dflash and
        # inkling all subtract one for the same reason; matching HF exactly is
        # what keeps the local-layer attention span token-for-token correct.
        sliding_window_size = (
            int(config.sliding_window) - 1 if self.is_sliding else -1
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(
                (0, self.num_heads * self.head_dim), dtype=hidden_states.dtype
            )
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # qk-norm (per head) then RoPE. The norms reduce over head_dim, so the
        # head axis is folded into the row axis rather than materialised.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(num_tokens, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)

        # Stock TokenSpeed: PagedAttention reads write locations from ctx
        # (cf. llama_ts), not an explicit out_cache_loc argument.
        attn_output = self.attn(q, k, v, ctx=ctx)
        if attn_output.dim() == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma3DecoderLayer(nn.Module):
    """Gemma 3 decoder layer: sandwich norms around attention + MLP.

    The sandwich shape (residual added *after* the post-norm) does not fit
    ``BaseDecoderLayer``'s pre-norm shell, so this layer is hand-written and
    therefore does not get ``CommManager``'s fused allreduce+norm. With TP > 1
    it pays two bare all-reduces per layer instead.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        rotary_emb,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        layer_types = gemma3_layer_types(config)
        layer_type = layer_types[layer_id]
        self.self_attn = Gemma3Attention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            layer_type=layer_type,
            rotary_emb=rotary_emb,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            prefix=add_prefix("mlp", prefix),
        )
        eps = config.rms_norm_eps
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Sandwich norms with the residual stream carried out of band.
        #
        # Gemma's sandwich adds the residual AFTER the post-sublayer norm:
        #   h = r + post_norm(sublayer(pre_norm(r)))
        # and the next block immediately pre-norms that sum. Folding the add
        # into the following norm is exact (add-then-norm) and removes one
        # standalone residual-add kernel per sandwich -- i.e. one redundant
        # read+write of the hidden state over HBM at every block boundary.
        #
        # ``GemmaRMSNorm.forward(x, residual)`` runs ``gemma_fused_add_rmsnorm``:
        # it updates ``residual <- residual + x`` in place and returns
        # ``gemma_rmsnorm(residual)``. The first layer has no incoming residual,
        # so the block entry is a plain norm and the residual starts as the
        # (scaled) embedding.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, ctx)
        # o_proj / down_proj are built with reduce_results=False, so the shard
        # sums are still partial here.
        if self.mapping.attn.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
        # post_attn_norm(attn_out) folded with the (residual + .) add into the
        # pre_feedforward norm: residual becomes r + post_attn_norm(attn_out),
        # the real hidden going forward; hidden becomes pre_ff_norm(residual).
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )

        hidden_states = self.mlp(hidden_states)
        if self.mapping.dense.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        # post_ff_norm(mlp_out); the (residual + .) add folds into the NEXT
        # layer's input_layernorm (or the final norm), so return both streams.
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


class Gemma3Model(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # One RoPE cache per layer type (local / global), shared by every layer
        # of that type.
        max_position = _rope_cache_positions(config)
        dtype = getattr(config, "dtype", None) or torch.get_default_dtype()
        layer_types = gemma3_layer_types(config)
        self.rotary_emb = nn.ModuleDict(
            {
                layer_type: _build_rope(config, layer_type, max_position, dtype)
                for layer_type in sorted(set(layer_types))
            }
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma3DecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
                rotary_emb=self.rotary_emb[layer_types[idx]],
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size
        self._normalizer_by_dtype: dict[torch.dtype, float] = {}

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        # Gemma normalizer: sqrt(hidden_size), rounded into the embedding
        # compute dtype to match HF's bf16 ``embed_scale``. Built once on CPU
        # and cached as a Python float so the multiply folds into the kernel
        # and never issues a CPU->GPU copy during CUDA graph capture (the same
        # trap the Gemma 4 module documents).
        scale = self._normalizer_by_dtype.get(embeds.dtype)
        if scale is None:
            scale = float(torch.tensor(self.hidden_size**0.5, dtype=embeds.dtype))
            self._normalizer_by_dtype[embeds.dtype] = scale
        return embeds * scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        # ``out_cache_loc`` is accepted for BaseCausalLM call-site compatibility
        # but unused: stock PagedAttention obtains write locations from ``ctx``.
        del out_cache_loc, kwargs
        if input_embeds is None:
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = input_embeds
        # Residual stream carried out of band so each block boundary's
        # ``residual + .`` folds into the next norm (see Gemma3DecoderLayer).
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, ctx)
        # Final block's post_feedforward_layernorm output still owes its
        # residual add; fold it into the final norm.
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, None


class Gemma3ForConditionalGeneration(BaseCausalLM):
    """Text decoder of ``Gemma3ForConditionalGeneration``.

    The vision tower is skipped: text-only inputs are supported (covers the LLM
    benchmark). ``final_logit_softcapping`` (``None`` on Gemma 3) and the tied
    ``lm_head`` are handled by the shared ``BaseCausalLM`` / ``LogitsProcessor``,
    both reading from the text config passed to ``super().__init__``.
    """

    model_cls = Gemma3Model

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        # The sandwich-norm layer keeps attention and MLP on one residual
        # stream, so a split attn/dense TP would need a shard exchange this
        # layer does not perform. Refuse rather than return wrong numbers.
        if mapping.attn.tp_size != mapping.dense.tp_size:
            raise ValueError(
                "Gemma 3 requires attn.tp_size == dense.tp_size, got "
                f"{mapping.attn.tp_size} != {mapping.dense.tp_size}."
            )
        # Drive the LM off the text sub-config so BaseCausalLM sees hidden_size,
        # vocab_size, tie_word_embeddings and final_logit_softcapping.
        super().__init__(
            config=_text_config(config),
            mapping=mapping,
            quant_config=quant_config,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):
        # Gemma stores gate/up separately; fuse into gate_up_proj. q/k/v are
        # NOT fused in the checkpoint (separate q_norm/k_norm live between the
        # projections and RoPE), so they load directly.
        # (fused param, checkpoint shard name, shard id). q/k/v fuse into the
        # single qkv_proj GEMM; the checkpoint stores them separately (the
        # per-head q_norm/k_norm live between the projections and RoPE, so the
        # projection weights themselves are plain). gate/up fuse into gate_up.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped: list[str] = []
        for name, loaded_weight in weights:
            # Skip the vision tower and multimodal projector of the
            # *ForConditionalGeneration checkpoint.
            if (
                name.startswith("model.vision_tower")
                or name.startswith("vision_tower")
                or name.startswith("model.multi_modal_projector")
                or name.startswith("multi_modal_projector")
                or "rotary_emb.inv_freq" in name
            ):
                continue
            # Text weights live under (model.)language_model.model.* in the
            # multimodal checkpoint; map onto our model.* namespace.
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
            elif name.startswith("language_model.model."):
                name = "model." + name[len("language_model.model.") :]
            elif name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]
            # Tied lm_head is not stored separately.
            if "lm_head" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # gate/up live in mlp; q/k/v in self_attn. Keep the two fused
                # groups from matching each other's shard names.
                is_mlp_shard = param_name == "gate_up_proj"
                if is_mlp_shard != ("mlp" in name):
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    skipped.append(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # A checkpoint that renames a tensor would otherwise load
        # "successfully" and generate wrong text, which is the expensive way to
        # find out.
        missing = sorted(set(params_dict) - loaded)
        if missing or skipped:
            raise ValueError(
                "Gemma 3 checkpoint did not match the model: "
                f"{len(missing)} parameter(s) never written "
                f"(e.g. {missing[:5]}), "
                f"{len(skipped)} checkpoint tensor(s) unclaimed "
                f"(e.g. {skipped[:5]})."
            )


# ``Gemma3ForCausalLM`` is the text-only architecture string; it maps to the
# same implementation (a bare text config has no ``.text_config``).
class Gemma3ForCausalLM(Gemma3ForConditionalGeneration):
    pass


EntryClass = [Gemma3ForConditionalGeneration, Gemma3ForCausalLM]
