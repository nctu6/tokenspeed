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

"""Inference-only Gemma 4 text model compatible with HuggingFace weights.

This implements the *text* decoder of ``Gemma4ForConditionalGeneration`` (the
multimodal checkpoint) and the equivalent ``Gemma4ForCausalLM`` (the text-only
release). The vision tower, multimodal projector and audio paths of the
multimodal checkpoint are skipped -- text-only inputs are supported, which
covers the LLM benchmark.

Gemma 4 differs from the Gemma 3 module in this package in ways that matter for
token-for-token parity. The source of truth is the local gemma-4-31B-it
``config.json`` plus the vllm-unieai reference port:

* Per-layer-type geometry. Unlike Gemma 3's uniform ``head_dim``, Gemma 4
  splits head_dim and kv-head count by attention type: ``full_attention``
  layers use ``global_head_dim`` (512 on 31B) with
  ``num_global_key_value_heads`` (4), while ``sliding_attention`` layers use
  ``head_dim`` (256) with ``num_key_value_heads`` (16). ``gemma4_layer_config``
  resolves the geometry for one layer; the rest of the module and the KV pool
  read that resolved value rather than the flat config attributes.
* ``attention_k_eq_v`` -- K and V are shared on the full-attention layers (the
  checkpoint ships ``k_proj`` but no ``v_proj`` there); handled at weight load
  in a later task.
* RMSNorm uses the standard ``x_normed * weight`` form (Gemma 3 used the
  zero-centred ``x_normed * (1 + weight)`` form), and Gemma 4 adds a weightless
  per-head value norm on top of the q/k norms.
* Attention softmax scaling is ``1.0`` (the learnable q/k norms carry the
  scaling); Gemma 3 scaled by ``query_pre_attn_scalar ** -0.5``.
* ``final_logit_softcapping`` is reinstated (30.0 on 31B); Gemma 3 dropped it.

The decoder layer shape (sandwich norms with the residual added AFTER the
post-norm) matches Gemma 3, with the addition of a per-layer output scalar.

RoPE is per-layer-type: sliding layers use a plain RoPE (full rotary over
head_dim 256, theta 1e4); full layers use the proportional/partial-rotary RoPE
(theta 1e6, partial_rotary_factor 0.25 over head_dim 512). Both read their
parameters from ``rope_parameters[layer_type]``; ``_build_rope`` maps the raw
``"proportional"`` type -- which the TokenSpeed ``get_rope`` rejects -- onto the
base-plus-partial-rotary ``Gemma4RotaryEmbedding`` so the inv_freq/cos/sin match
the vllm-unieai oracle.

The config helpers and per-layer-type RoPE builder live here; the attention
module, decoder layer, model, weight loading and cache recipe are added by
later tasks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import GeluTanhAndMul
from tokenspeed.runtime.layers.layernorm import RMSNorm, RMSNormNoWeight
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import (
    Gemma4RotaryEmbedding,
    get_rope,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger, make_layers
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# HF Gemma4 defaults for the gemma-4-31B-it checkpoint, used only when the
# nested ``rope_parameters[layer_type]`` omits a field. Full-attention layers
# use the proportional/partial-rotary RoPE (theta 1e6, factor 0.25); sliding
# layers use a plain RoPE (theta 1e4, full rotary).
_DEFAULT_FULL_ROPE_THETA = 1_000_000.0
_DEFAULT_SLIDING_ROPE_THETA = 10_000.0
_DEFAULT_FULL_PARTIAL_ROTARY_FACTOR = 0.25
_DEFAULT_SLIDING_PARTIAL_ROTARY_FACTOR = 1.0


def _text_config(config):
    """Return the Gemma 4 *text* sub-config for either the multimodal
    ``Gemma4Config`` (has ``.text_config``) or a bare text config."""
    return getattr(config, "text_config", None) or config


def _reject_unsupported_features(text_config) -> None:
    """Refuse a Gemma 4 text config that enables a feature this port dropped.

    This is a *text-only, dense* port of Gemma 4: it serves the
    gemma-4-31B-it checkpoint, which has MoE, the double-wide MLP, per-layer
    input embeddings (PLE) and KV-sharing all DISABLED. The vllm-unieai
    reference ``Gemma4DecoderLayer`` / ``Gemma4Attention`` carry branches for
    each of those, but this port deliberately omitted them to keep a single
    dense path (AGENTS.md: make the general path cover the case; refuse rather
    than return wrong numbers).

    So if a *different* Gemma 4 config arrives with any of those features
    enabled, running the dense path anyway would silently produce wrong output
    (the extra expert/PLE/shared-KV contributions would just be dropped).
    Fail loudly at construction instead, with a message naming the exact
    unsupported feature, so the mismatch is obvious rather than mis-served.

    One check per feature, each raising a specific ``ValueError`` (gemma3 uses
    ``ValueError`` for its config refusals, so this matches). The gemma-4-31B-it
    checkpoint trips none of these, so the guard is a clean no-op there and the
    real model still constructs.

    Args:
        text_config: The Gemma 4 *text* sub-config (``_text_config(config)``).

    Raises:
        ValueError: If MoE / the double-wide MLP / per-layer input embeddings /
            KV-sharing is enabled on the config.
    """
    # MoE / second MLP block. The reference enables the block when EITHER flag
    # is truthy; ``num_experts`` / ``top_k_experts`` carry the expert count.
    # Any of them being set means the checkpoint expects a routed expert
    # contribution this dense port never computes.
    enable_moe_block = getattr(text_config, "enable_moe_block", False)
    use_second_mlp_block = getattr(text_config, "use_second_mlp_block", False)
    num_experts = getattr(text_config, "num_experts", None) or 0
    top_k_experts = getattr(text_config, "top_k_experts", None) or 0
    if (
        enable_moe_block
        or use_second_mlp_block
        or int(num_experts) > 0
        or int(top_k_experts) > 0
    ):
        raise ValueError(
            "Gemma 4 text port does not support MoE (enable_moe_block / "
            "use_second_mlp_block / num_experts / top_k_experts); this build "
            "serves the dense gemma-4-31B-it checkpoint only."
        )

    # Double-wide MLP: in the reference this only widens the KV-shared layers'
    # intermediate size, but KV-sharing is itself unsupported here, so guard
    # the flag whenever it is set rather than silently build a narrow MLP.
    if getattr(text_config, "use_double_wide_mlp", False):
        raise ValueError(
            "Gemma 4 text port does not support the double-wide MLP "
            "(use_double_wide_mlp); this build serves the dense "
            "gemma-4-31B-it checkpoint only."
        )

    # Per-layer input embeddings (PLE): a positive per-layer embedding width
    # means the decoder layers expect a gated per-layer embedding contribution
    # this port does not compute.
    hidden_size_per_layer_input = (
        getattr(text_config, "hidden_size_per_layer_input", None) or 0
    )
    if int(hidden_size_per_layer_input) > 0:
        raise ValueError(
            "Gemma 4 text port does not support per-layer input embeddings "
            "(hidden_size_per_layer_input > 0); this build serves the dense "
            "gemma-4-31B-it checkpoint only."
        )

    # KV-sharing: a positive shared-layer count means the last N layers reuse
    # an earlier layer's KV cache (and ship only q_proj); this port builds a
    # full fused QKV on every layer and has no shared-KV path.
    num_kv_shared_layers = getattr(text_config, "num_kv_shared_layers", None) or 0
    if int(num_kv_shared_layers) > 0:
        raise ValueError(
            "Gemma 4 text port does not support KV-sharing "
            "(num_kv_shared_layers > 0); this build serves the dense "
            "gemma-4-31B-it checkpoint only."
        )


def gemma4_layer_types(config) -> list[str]:
    """Per-layer attention labels for a Gemma 4 text config.

    Gemma 4 checkpoints always ship an explicit ``layer_types`` list, so unlike
    Gemma 3 there is no ``sliding_window_pattern`` fallback: if the list is
    absent we raise rather than guess the 5:1 local:global layout.

    Args:
        config: The Gemma 4 config (multimodal or bare text).

    Returns:
        A list of ``"sliding_attention"`` / ``"full_attention"`` labels, one
        per hidden layer.

    Raises:
        ValueError: If the text config carries no explicit ``layer_types``.
    """
    text_config = _text_config(config)
    explicit = getattr(text_config, "layer_types", None)
    if not explicit:
        raise ValueError(
            "Gemma 4 config is missing an explicit 'layer_types' list; the "
            "gemma-4 checkpoint always ships one and this port does not guess "
            "the sliding/full layout from a window pattern."
        )
    return list(explicit)


@dataclass(frozen=True)
class Gemma4LayerConfig:
    """Resolved per-layer attention geometry for a Gemma 4 layer.

    Gemma 4 stores head_dim and kv-head counts as flat config attributes that
    ``layer_types`` picks between; this is the homogeneous view for one layer,
    so callers read ``head_dim`` / ``num_key_value_heads`` without caring which
    attention type the layer is.
    """

    head_dim: int
    num_key_value_heads: int


def gemma4_layer_config(config, layer_idx: int) -> Gemma4LayerConfig:
    """The Gemma 4 text config as it applies to one layer.

    Gemma 4 uses a larger head dimension on its full-attention layers than on
    its sliding ones, and with ``attention_k_eq_v`` it uses fewer KV heads
    there too. The flat config attributes are resolved against the layer's type
    so the result is homogeneous either way (mirrors vllm-unieai's
    ``gemma4_layer_config``).

    Args:
        config: The Gemma 4 config (multimodal or bare text).
        layer_idx: Index of the layer whose geometry to resolve.

    Returns:
        A ``Gemma4LayerConfig`` carrying the layer's ``head_dim`` and
        ``num_key_value_heads``.
    """
    text_config = _text_config(config)
    layer_types = gemma4_layer_types(text_config)
    head_dim = int(text_config.head_dim)
    num_key_value_heads = int(text_config.num_key_value_heads)
    if layer_types[layer_idx] == FULL_ATTENTION:
        global_head_dim = getattr(text_config, "global_head_dim", None)
        head_dim = int(global_head_dim or head_dim)
        global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)
        num_key_value_heads = int(global_kv_heads or num_key_value_heads)
    return Gemma4LayerConfig(
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
    )


def gemma4_layer_kv_geometry(config) -> tuple[tuple[int, int], ...]:
    """Per-layer ``(num_kv_heads, head_dim)`` for a Gemma 4 text config, PRE-TP.

    One ``(kv_heads, head_dim)`` pair per hidden layer, in layer order and
    before tensor-parallel sharding: full_attention layers resolve to the
    global split (``num_global_key_value_heads`` x ``global_head_dim``),
    sliding_attention layers to the flat split (``num_key_value_heads`` x
    ``head_dim``). This is the geometry the KV cache sizes its per-layer pages
    from; it is derived from the same ``gemma4_layer_config`` the attention
    module reads, so cache and compute cannot disagree on a layer's shape.

    Args:
        config: The Gemma 4 config (multimodal or bare text).

    Returns:
        A tuple with one ``(num_kv_heads, head_dim)`` pair per hidden layer.
    """
    text_config = _text_config(config)
    num_layers = len(gemma4_layer_types(text_config))
    return tuple(
        (
            gemma4_layer_config(text_config, layer_idx).num_key_value_heads,
            gemma4_layer_config(text_config, layer_idx).head_dim,
        )
        for layer_idx in range(num_layers)
    )


def _rope_cache_positions(config) -> int:
    """Rows to precompute in the RoPE cos/sin cache.

    ``max_position_embeddings`` is 262144 on gemma-4-31B-it; each layer type's
    cache is ``[rows, head_dim]`` float32 and the full-attention head_dim is
    512, so the cache is not free. Nothing attends past the served context, so
    clamp to ``max_model_len`` when the server declares a shorter window (same
    reasoning as the Gemma 3 module and deepseek_v4).

    Args:
        config: The Gemma 4 config (multimodal or bare text).

    Returns:
        The number of positions to precompute: the clamped minimum of the
        config's ``max_position_embeddings`` and the served ``max_model_len``.
    """
    text_config = _text_config(config)
    max_position = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    served = global_server_args_dict.get("max_model_len")
    if served:
        clamped = min(max_position, int(served)) if max_position else int(served)
        if max_position and clamped < max_position:
            logger.info(
                "Gemma 4 RoPE cache clamped to %d positions (config declares %d).",
                clamped,
                max_position,
            )
        return clamped
    return max_position


def _rope_parameters_for(config, layer_type: str) -> dict:
    """The ``rope_parameters`` sub-dict for one Gemma 4 layer type.

    Gemma 4 nests RoPE settings per attention type:
    ``rope_parameters = {"full_attention": {...}, "sliding_attention": {...}}``.
    Return the entry for ``layer_type`` as a plain dict. Raise if the config
    carries no per-layer-type mapping -- this port does not support the flat
    legacy ``rope_parameters`` layout.

    Args:
        config: The Gemma 4 config (multimodal or bare text).
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.

    Returns:
        The RoPE parameter dict for the layer type (possibly empty, in which
        case the documented gemma-4-31B defaults apply at the call site).

    Raises:
        ValueError: If ``rope_parameters`` is missing or not keyed by layer
            type.
    """
    text_config = _text_config(config)
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict) or layer_type not in rope_parameters:
        raise ValueError(
            "Gemma 4 config is missing a per-layer-type 'rope_parameters' entry "
            f"for {layer_type!r}; this port requires the nested "
            "{'full_attention': {...}, 'sliding_attention': {...}} layout."
        )
    entry = rope_parameters[layer_type]
    if not isinstance(entry, dict):
        raise ValueError(
            f"Gemma 4 rope_parameters[{layer_type!r}] must be a mapping, got "
            f"{type(entry).__name__}."
        )
    return dict(entry)


def _build_rope(config, layer_type: str, max_position: int, dtype: torch.dtype):
    """One shared RoPE instance for a Gemma 4 layer type.

    Both layer types read their base ``theta`` and ``partial_rotary_factor``
    from ``rope_parameters[layer_type]`` (with the documented gemma-4-31B
    defaults as a fallback). The two differ in how the rotation is built:

    * ``sliding_attention``: ``rope_type="default"``, full rotary over the
      sliding head_dim (256) with ``theta`` 1e4. This is a plain ``get_rope``.
    * ``full_attention``: ``rope_type="proportional"``, ``theta`` 1e6, and a
      ``partial_rotary_factor`` of 0.25 over the global head_dim (512), giving a
      rotary_dim of 128. TokenSpeed's ``get_rope`` rejects the raw
      ``"proportional"`` type, so this is mapped to the base-plus-partial-rotary
      construction that reproduces the oracle: ``Gemma4RotaryEmbedding`` scales
      the inv_freq exponents by head_dim (not rotary_dim) and zero-pads the
      non-rotated dims, matching vllm-unieai's ``Gemma4RotaryEmbedding``.

    The head_dim comes from the per-layer geometry
    (``gemma4_layer_config``/``gemma4_layer_kv_geometry``), which is why the two
    types land on 256 vs 512. ``get_rope`` is cached by its arguments; the
    full-attention ``Gemma4RotaryEmbedding`` is a fresh instance per
    ``_build_rope`` call. The model builds one RoPE per unique layer type (as
    Gemma 3 does), so each layer type still shares a single cos/sin cache
    across all its layers.

    Args:
        config: The Gemma 4 config (multimodal or bare text).
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        max_position: Number of positions to precompute (see
            ``_rope_cache_positions``).
        dtype: Compute dtype for the RoPE instance.

    Returns:
        A shared ``RotaryEmbedding`` (sliding) or ``Gemma4RotaryEmbedding``
        (full) for the layer type.

    Raises:
        ValueError: If ``layer_type`` is neither sliding nor full attention.
    """
    text_config = _text_config(config)
    rope_parameters = _rope_parameters_for(text_config, layer_type)
    if layer_type == SLIDING_ATTENTION:
        head_dim = int(text_config.head_dim)
        base = float(rope_parameters.get("rope_theta", _DEFAULT_SLIDING_ROPE_THETA))
        partial_rotary_factor = float(
            rope_parameters.get(
                "partial_rotary_factor", _DEFAULT_SLIDING_PARTIAL_ROTARY_FACTOR
            )
        )
        # Sliding uses full rotary over the whole head (factor 1.0); route
        # through the shared get_rope cache as a plain default RoPE.
        return get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=base,
            is_neox_style=True,
            rope_scaling=None,
            dtype=dtype,
            partial_rotary_factor=partial_rotary_factor,
        )
    if layer_type == FULL_ATTENTION:
        global_head_dim = getattr(text_config, "global_head_dim", None)
        head_dim = int(global_head_dim or text_config.head_dim)
        base = float(rope_parameters.get("rope_theta", _DEFAULT_FULL_ROPE_THETA))
        partial_rotary_factor = float(
            rope_parameters.get(
                "partial_rotary_factor", _DEFAULT_FULL_PARTIAL_ROTARY_FACTOR
            )
        )
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError(
                "Gemma 4 full_attention partial_rotary_factor must be in "
                f"(0.0, 1.0], got {partial_rotary_factor}."
            )
        # proportional -> base + partial-rotary over head_dim. rotary_dim is the
        # reduced rotated span (head_dim * factor = 128 on the 31B); the
        # Gemma4RotaryEmbedding builds the proportional inv_freq (head_dim
        # denominator) and zero-pads the remaining dims to identity.
        rotary_dim = int(head_dim * partial_rotary_factor)
        return Gemma4RotaryEmbedding(
            head_dim,
            rotary_dim,
            max_position,
            base,
            True,
            dtype,
        )
    raise ValueError(
        f"Unknown Gemma 4 layer_type {layer_type!r}; expected "
        f"{SLIDING_ATTENTION!r} or {FULL_ATTENTION!r}."
    )


class Gemma4Attention(nn.Module):
    """Gemma 4 attention for one decoder layer.

    Gemma 4 differs from Gemma 3 (see :class:`Gemma3Attention`) in several ways
    that this ``__init__`` wires up:

    * Per-layer-type geometry. The layer's ``head_dim`` and KV-head count come
      from :func:`gemma4_layer_config` resolved against ``layer_type``, NOT
      from the flat config fields: full-attention layers are head_dim 512 / 4
      KV heads, sliding layers head_dim 256 / 16 KV heads on the 31B. The one
      fused ``QKVParallelLinear`` is built at that resolved geometry, so the Q
      projection follows ``num_attention_heads`` while K/V follow the layer's
      KV-head count.
    * Three per-head norms over the layer's ``head_dim``, in the standard
      ``x_normed * weight`` form (``RMSNorm``, not Gemma 3's ``1 + weight``
      ``GemmaRMSNorm``): ``q_norm`` and ``k_norm`` carry a learnable scale,
      while ``v_norm`` is weightless (:class:`RMSNormNoWeight`) -- pure RMS
      normalization matching vllm-unieai's ``RMSNorm(head_dim, eps,
      has_weight=False)``.
    * Softmax scaling is ``1.0`` (the learnable q/k norms carry the scaling);
      Gemma 3 scaled by ``query_pre_attn_scalar ** -0.5``.
    * Attention logit softcapping is OFF (``attn_logit_softcapping`` is None ->
      ``logit_cap`` 0.0); Gemma 3 honoured whatever the config declared.

    The checkpoint served here has KV-sharing, MoE, double-wide MLP and
    per-layer input embeddings all disabled, so this port is the non-shared
    path only: there is no ``is_kv_shared_layer`` branch and every layer builds
    a full fused QKV plus the three norms. ``attention_k_eq_v`` on the full
    layers is handled entirely at weight load (K duplicated into the V shard),
    not here.

    RoPE is NOT built here: the model builds one shared RoPE per layer type
    (via :func:`_build_rope`) and passes the shared instance in as
    ``rotary_emb``, mirroring the Gemma 3 idiom.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        layer_type: str,
        rotary_emb,
        quant_config: QuantizationConfig | None,
        prefix: str,
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

        text_config = _text_config(config)
        hidden_size = int(text_config.hidden_size)

        # Per-layer-type geometry: full layers resolve to global_head_dim (512)
        # / num_global_key_value_heads (4), sliding to head_dim (256) /
        # num_key_value_heads (16). This is the LAYER's geometry, not the flat
        # config fields.
        layer_config = gemma4_layer_config(text_config, layer_id)
        self.head_dim = int(layer_config.head_dim)
        total_num_heads = int(text_config.num_attention_heads)
        total_num_kv_heads = int(layer_config.num_key_value_heads)
        validate_attention_partition(total_num_heads, total_num_kv_heads, tp_size)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 4 scales by 1.0: query_pre_attn_scalar is NOT used; the
        # learnable q/k norms carry the scaling implicitly.
        self.scaling = 1.0

        attention_bias = bool(getattr(text_config, "attention_bias", False))
        # Fused QKV at the layer's geometry: one GEMM over the shared input.
        # On the full (k_eq_v) layers the checkpoint ships only k_proj; weight
        # load duplicates it into the V shard, so V == K with no forward
        # branch. QKVParallelLinear's K/V shards are sized from
        # total_num_kv_heads, which already reflects the layer's KV-head count.
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

        rms_norm_eps = float(text_config.rms_norm_eps)
        # Standard x*weight RMSNorm over head_dim (NOT Gemma 3's 1+weight). q/k
        # carry a learnable scale; v is weightless (pure normalization).
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.v_norm = RMSNormNoWeight(self.head_dim, eps=rms_norm_eps)

        # Gemma 4 has no attention softcapping (attn_logit_softcapping is None).
        logit_cap = float(getattr(text_config, "attn_logit_softcapping", None) or 0.0)

        # HF ``sliding_window`` counts the current token (inclusive); the engine
        # stores ``window_left`` (earlier tokens still visible), so a sliding
        # layer admitting ``sliding_window`` HF positions passes
        # ``sliding_window - 1`` here. Full layers get -1 (full attention).
        sliding_window_size = (
            hf_sliding_window_to_window_left(int(text_config.sliding_window))
            if self.is_sliding
            else -1
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            # Norms and RoPE stay in this module for now (weightless v_norm,
            # proportional RoPE at head_dim 512 and k_eq_v are not validated
            # in the attention prologue yet); the prologue only writes KV.
            rotary_emb=None,
            qk_norm=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """One attention block: qkv split, three per-head norms, RoPE, attend.

        The ordering is fixed by Gemma 4 (see vllm-unieai's ``Gemma4Attention``
        and this module's class docstring): ``q_norm`` and ``k_norm`` apply per
        head BEFORE RoPE, RoPE rotates only ``q`` and ``k`` (never ``v``), then
        ``v_norm`` applies to ``v``. All three norms reduce over ``head_dim``,
        so the head axis is folded into the row axis (``reshape(-1, head_dim)``
        then ``view`` back) rather than materialised per head.

        The RoPE call keeps TokenSpeed's FLATTENED convention: ``q``/``k`` are
        passed as ``[num_tokens, q_size]`` / ``[num_tokens, kv_size]`` and
        ``RotaryEmbedding``/``Gemma4RotaryEmbedding`` infer the head count from
        the last dim (``x.shape[-1] // head_size``). This differs from vllm's
        ``unflatten``/``flatten`` around the norm+RoPE; the norms are folded
        into the row axis instead.

        On the full (``attention_k_eq_v``) layers V equals K because weight
        load duplicated ``k_proj`` into the V shard of ``qkv_proj`` -- there is
        no forward branch for that; ``v`` is simply the V split of the fused
        projection and gets its own ``v_norm`` like any other layer.

        Args:
            positions: Per-token position ids, shape ``[num_tokens]``.
            hidden_states: Layer input, shape ``[num_tokens, hidden_size]``.
            ctx: Forward context carrying the KV write locations and attention
                metadata (stock PagedAttention reads these from ``ctx``).

        Returns:
            The attention output projected back to ``hidden_size``, shape
            ``[num_tokens, hidden_size]``. Empty (zero-row) inputs return a
            zero tensor of shape ``[0, num_heads * head_dim]`` without touching
            the projections.
        """
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(
                (0, self.num_heads * self.head_dim), dtype=hidden_states.dtype
            )
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # q/k norm (per head) then RoPE(q, k). The norms reduce over head_dim,
        # so the head axis is folded into the row axis rather than materialised.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(num_tokens, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)

        # v norm comes AFTER RoPE and applies only to v (v is never rotated).
        v = self.v_norm(v.reshape(-1, self.head_dim)).view(num_tokens, self.kv_size)

        # Stock TokenSpeed: PagedAttention reads write locations from ctx
        # (cf. gemma3), not an explicit out_cache_loc argument.
        attn_output = self.attn(q, k, v, positions, ctx=ctx)
        if attn_output.dim() == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MLP(nn.Module):
    """Gemma 4 GeGLU MLP: ``down(gelu_tanh(gate) * up)``.

    Structurally identical to :class:`Gemma3MLP`: a single fused
    ``gate_up_proj`` (``MergedColumnParallelLinear`` splitting the output into
    the gate and up halves), the ``gelu_pytorch_tanh`` gated activation
    (:class:`GeluTanhAndMul`, matching vllm-unieai's
    ``get_act_and_mul_fn("gelu_pytorch_tanh")``), and a ``down_proj`` row-
    parallel projection back to ``hidden_size``. ``reduce_results=False`` on
    ``down_proj`` leaves the tensor-parallel all-reduce to the decoder layer,
    as in Gemma 3. Gemma 4 does not use the double-wide MLP, so the fused output
    is exactly ``[intermediate_size, intermediate_size]`` (gate | up).

    Per the TokenSpeed guidelines every argument is passed explicitly at the
    call site; this signature therefore declares no defaults.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None,
        tp_rank: int | None,
        tp_size: int | None,
        tp_group: tuple[int, ...] | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Unsupported activation {hidden_activation!r}; Gemma 4 uses "
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
        """Apply the GeGLU MLP.

        Keeps the zero-row guard from Gemma 3: an empty (zero-token) input is
        returned untouched so the fused projections are never invoked on a
        degenerate shape.

        Args:
            x: Layer input, shape ``[num_tokens, hidden_size]``.

        Returns:
            The MLP output, shape ``[num_tokens, hidden_size]`` (the same empty
            tensor when ``num_tokens`` is 0).
        """
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Gemma4DecoderLayer(nn.Module):
    """Gemma 4 decoder layer: sandwich norms with EXPLICIT residual adds.

    The block shape differs from :class:`Gemma3DecoderLayer` in two ways that
    matter for token-for-token parity (see vllm-unieai's
    ``Gemma4DecoderLayer.forward``):

    * Explicit residual adds. Gemma 3 folds the sandwich residual add into the
      *next* norm (``gemma_fused_add_rmsnorm`` updates ``residual`` in place and
      the residual stream is carried out of band across layers). Gemma 4 does
      the two adds inside the layer with plain ``+`` operators and keeps no
      cross-layer residual stream, so ``forward`` returns ``(hidden, None)``.
      The incoming ``residual`` argument is accepted to stay call-compatible
      with the model loop but is NOT used -- the layer computes its own.
    * Per-layer output scalar. A ``[1]`` ``layer_scalar`` buffer (loaded from
      the checkpoint by the weight-load task) multiplies the layer output after
      the feed-forward residual add. Gemma 3 has no such scalar.

    The four sandwich norms are standard ``x_normed * weight`` ``RMSNorm`` over
    ``hidden_size`` (NOT Gemma 3's ``1 + weight`` ``GemmaRMSNorm``), matching
    the standalone q/k norms in :class:`Gemma4Attention`.

    Like Gemma 3 this layer is hand-written (the sandwich shape does not fit
    ``BaseDecoderLayer``'s pre-norm shell) and therefore does not get
    ``CommManager``'s fused allreduce+norm: ``o_proj`` and ``down_proj`` are
    built with ``reduce_results=False``, so with TP > 1 the layer pays two bare
    all-reduces per block -- one on the attention sublayer output (before its
    post-norm and residual add) and one on the MLP sublayer output (likewise).

    Per the TokenSpeed guidelines every argument is passed explicitly at the
    call site; this signature therefore declares no defaults.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        rotary_emb,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        layer_types = gemma4_layer_types(config)
        layer_type = layer_types[layer_id]
        self.self_attn = Gemma4Attention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            layer_type=layer_type,
            rotary_emb=rotary_emb,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        text_config = _text_config(config)
        self.mlp = Gemma4MLP(
            hidden_size=int(text_config.hidden_size),
            intermediate_size=int(text_config.intermediate_size),
            hidden_activation=text_config.hidden_activation,
            quant_config=quant_config,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            prefix=add_prefix("mlp", prefix),
        )
        eps = float(text_config.rms_norm_eps)
        # Standard x*weight RMSNorm (NOT Gemma 3's 1+weight GemmaRMSNorm).
        self.input_layernorm = RMSNorm(int(text_config.hidden_size), eps=eps)
        self.post_attention_layernorm = RMSNorm(int(text_config.hidden_size), eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(int(text_config.hidden_size), eps=eps)
        self.post_feedforward_layernorm = RMSNorm(
            int(text_config.hidden_size), eps=eps
        )
        # Per-layer output scalar: a [1] buffer loaded from the checkpoint
        # (filled by the weight-load task) and applied to the layer output.
        # Initialised to ones so an unloaded layer is the identity scaling.
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, None]:
        """One sandwich-norm block with explicit residual adds.

        The pattern mirrors vllm-unieai's ``Gemma4DecoderLayer.forward`` exactly
        (NOT Gemma 3's folded-add scheme)::

            residual = hidden
            hidden = input_layernorm(hidden)
            hidden = self_attn(hidden)          # + all-reduce if TP > 1
            hidden = post_attention_layernorm(hidden)
            hidden = hidden + residual          # explicit add #1
            residual = hidden
            hidden = pre_feedforward_layernorm(hidden)
            hidden = mlp(hidden)                # + all-reduce if TP > 1
            hidden = post_feedforward_layernorm(hidden)
            hidden = hidden + residual          # explicit add #2
            hidden = hidden * layer_scalar      # per-layer output scalar
            return hidden, None

        The two all-reduces land on the attention / MLP sublayer output BEFORE
        its post-norm and residual add, because ``o_proj`` / ``down_proj`` are
        built ``reduce_results=False`` and still hold partial shard sums at that
        point (same placement as :class:`Gemma3DecoderLayer`).

        Args:
            positions: Per-token position ids, shape ``[num_tokens]``.
            hidden_states: Layer input, shape ``[num_tokens, hidden_size]``.
            residual: Accepted for call-compatibility with the model loop; NOT
                used -- Gemma 4 keeps no cross-layer residual stream and
                computes its residuals internally.
            ctx: Forward context carrying the KV write locations and attention
                metadata.

        Returns:
            A ``(hidden_states, None)`` tuple: the scaled layer output and
            ``None`` for the residual stream (Gemma 4 carries none out of band).
        """
        del residual  # Gemma 4 computes its own residuals; see docstring.

        # Attention sublayer with explicit residual add #1.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, ctx)
        # o_proj is built reduce_results=False, so the shard sums are partial.
        if self.mapping.attn.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # Feed-forward sublayer with explicit residual add #2.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # down_proj is built reduce_results=False, so the shard sums are partial.
        if self.mapping.dense.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # Per-layer output scalar, applied at the very end of the layer.
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4Model(nn.Module):
    """Gemma 4 stacked text decoder: scaled embed -> N layers -> final norm.

    Mirrors :class:`Gemma3Model`'s wiring (shared VocabParallelEmbedding, one
    RoPE per layer type in a ``ModuleDict`` shared across every layer of that
    type, the ``make_layers`` decoder stack, and the tensor-returning scaled
    ``get_input_embeddings``), but adapts to Gemma 4's explicit-residual
    decoder layer:

    * The embedding scale is Gemma's ``sqrt(hidden_size)`` normalizer. It is
      exposed through :meth:`get_input_embeddings` as a TENSOR (not an
      ``nn.Module``). This matters: the gemma-3 "fluent garbage" bug was
      root-caused to the prefill CUDA graph feeding RAW embeddings and
      bypassing the normalizer, and the landed ``prefill_graph.py`` fix prefers
      a tensor-returning ``get_input_embeddings``. The normalizer is cached per
      dtype as a Python float so the multiply folds into the kernel and issues
      no CPU->GPU copy during CUDA-graph capture.
    * The forward loop uses explicit residual adds. Unlike Gemma 3 -- which
      carries a ``residual`` stream out of band and folds the final residual
      add into the final norm -- each :class:`Gemma4DecoderLayer` closes its own
      residual with plain ``+`` adds and returns ``(hidden, None)``. So this
      model threads ``residual = None`` through the loop (the layers ignore the
      incoming value) and applies a PLAIN final norm: ``self.norm(hidden)``,
      with NO residual folded in.
    * The final norm is a standard ``x_normed * weight`` ``RMSNorm`` over
      ``hidden_size`` (NOT Gemma 3's ``1 + weight`` ``GemmaRMSNorm``), matching
      the sandwich and q/k norms elsewhere in this module.

    Per the TokenSpeed guidelines every argument is passed explicitly at the
    call site; this signature therefore declares no defaults.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.config = config
        text_config = _text_config(config)
        self.vocab_size = int(text_config.vocab_size)
        self.hidden_size = int(text_config.hidden_size)
        # Shared token embedding, sharded over the attention TP group (the
        # lm_head is tied to this by the LM-head task).
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # One RoPE cache per unique layer type (sliding / full), shared by every
        # layer of that type. ``_build_rope`` returns a plain get_rope result
        # for sliding and a fresh Gemma4RotaryEmbedding for full -- both are
        # nn.Modules, so a ModuleDict holds exactly one per layer type.
        max_position = _rope_cache_positions(config)
        dtype = getattr(config, "dtype", None) or torch.get_default_dtype()
        layer_types = gemma4_layer_types(config)
        self.rotary_emb = nn.ModuleDict(
            {
                layer_type: _build_rope(config, layer_type, max_position, dtype)
                for layer_type in sorted(set(layer_types))
            }
        )
        self.layers = make_layers(
            int(text_config.num_hidden_layers),
            lambda idx, prefix: Gemma4DecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
                rotary_emb=self.rotary_emb[layer_types[idx]],
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        # STANDARD x*weight RMSNorm (NOT Gemma 3's 1+weight GemmaRMSNorm); no
        # residual is folded into it (see forward / class docstring).
        self.norm = RMSNorm(self.hidden_size, eps=float(text_config.rms_norm_eps))
        # Normalizer cache keyed by compute dtype; see get_input_embeddings.
        self._normalizer_by_dtype: dict[torch.dtype, float] = {}

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Scaled token embeddings as a TENSOR: ``embed_tokens(ids) *
        sqrt(hidden_size)``.

        Gemma scales its token embeddings by ``sqrt(hidden_size)`` (HF's
        ``embed_scale``). This MUST return a tensor (not an ``nn.Module``): the
        landed ``prefill_graph.py`` fix prefers a tensor-returning
        ``get_input_embeddings`` so the prefill CUDA graph consumes the SCALED
        embedding rather than the raw one (the gemma-3 garbage bug). The
        normalizer is built once per dtype on CPU and cached as a Python float
        so the multiply folds into the embedding kernel and issues no CPU->GPU
        copy during CUDA-graph capture.

        Args:
            input_ids: Token ids, shape ``[num_tokens]``.

        Returns:
            The scaled embeddings, shape ``[num_tokens, hidden_size]`` in the
            embedding compute dtype.
        """
        embeds = self.embed_tokens(input_ids)
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
        """Run the stacked decoder.

        Matches :class:`Gemma3Model.forward`'s call signature for
        ``BaseCausalLM`` compatibility, but drives Gemma 4's explicit-residual
        layers: ``residual`` is threaded as ``None`` (each layer computes and
        closes its own residual and returns ``(hidden, None)``), and the final
        norm is PLAIN -- ``self.norm(hidden)`` with no residual folded in,
        because every decoder layer already closed its residual with explicit
        adds.

        Args:
            input_ids: Token ids, shape ``[num_tokens]``. Ignored when
                ``input_embeds`` is provided.
            positions: Per-token position ids, shape ``[num_tokens]``.
            ctx: Forward context carrying KV write locations and attention
                metadata (stock PagedAttention reads these from ``ctx``).
            out_cache_loc: Accepted for ``BaseCausalLM`` call-site compatibility
                but unused -- PagedAttention obtains write locations from
                ``ctx``.
            input_embeds: Pre-computed input embeddings; when provided they are
                used as-is (already scaled by the caller) instead of
                ``get_input_embeddings``.
            **kwargs: Accepted and ignored for call-site compatibility.

        Returns:
            A ``(hidden_states, None)`` tuple: the final-normed decoder output
            and ``None`` for the residual stream (Gemma 4 carries none out of
            band).
        """
        # ``out_cache_loc`` / ``kwargs`` are accepted for BaseCausalLM call-site
        # compatibility but unused: PagedAttention reads write locations from
        # ``ctx`` and Gemma 4 needs no extra forward arguments.
        del out_cache_loc, kwargs
        if input_embeds is None:
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = input_embeds
        # Gemma 4 keeps no cross-layer residual stream: each layer closes its
        # own residual with explicit adds and returns ``(hidden, None)``. The
        # loop threads ``None`` so the layer's unused ``residual`` argument is
        # satisfied without carrying state across layers.
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, ctx)
        # PLAIN final norm: no residual folded in, because the final layer
        # already closed its residual with an explicit add (unlike Gemma 3).
        hidden_states = self.norm(hidden_states)
        return hidden_states, None


class Gemma4ForConditionalGeneration(BaseCausalLM):
    """Text decoder of ``Gemma4ForConditionalGeneration`` (text-only path).

    The vision tower, multimodal projector and audio paths of the multimodal
    checkpoint are skipped: only text inputs are supported, which covers the
    LLM benchmark. Everything LM-head-shaped is inherited from
    :class:`BaseCausalLM` by driving it off the Gemma 4 *text* sub-config
    (:func:`_text_config`) rather than the outer multimodal config:

    * Final logit softcapping. The text config carries
      ``final_logit_softcapping=30.0`` (Gemma 4 reinstated the cap Gemma 3
      dropped). ``BaseCausalLM.resolve_logits_processor`` builds a
      ``LogitsProcessor`` from this config, and the processor reads
      ``final_logit_softcapping`` off it and applies the softcap to the output
      logits. Nothing is plumbed here beyond passing the text config up, so the
      softcap is automatic.
    * Tied lm_head. The text config carries ``tie_word_embeddings=True``, so
      ``BaseCausalLM.resolve_lm_head`` ties the lm_head to ``embed_tokens``
      (``self.lm_head is self.model.embed_tokens``) instead of allocating a
      separate head. Again automatic from the text config.

    The embedding scale (``sqrt(hidden_size)``) is applied inside
    :class:`Gemma4Model`; this class only forwards ``get_input_embeddings`` to
    the model so the tensor-returning (prefill-graph-safe) form is preserved.

    Weight loading is NOT defined here on purpose. The checkpoint's text/vision
    split, the ``attention_k_eq_v`` K->V duplication and the ``layer_scalar``
    buffers are handled by the weight-load task (task 6); until then the
    inherited ``BaseCausalLM.load_weights`` applies. No stub is required to
    construct the module -- ``BaseCausalLM`` provides a concrete
    ``load_weights`` -- so there is no placeholder here.

    Per the TokenSpeed guidelines every argument is passed explicitly at the
    call site, so unlike Gemma 3's ``quant_config=None`` this signature declares
    ``quant_config`` WITHOUT a default. The model loader
    (``model_loader/loader.py:_initialize_model``) always passes
    ``quant_config=`` explicitly, so a required parameter matches the loader's
    contract and surfaces a missing argument instead of silently defaulting it.
    """

    model_cls = Gemma4Model

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
    ) -> None:
        # The hand-written sandwich / explicit-residual decoder layer keeps
        # attention and MLP on one residual stream, so a split attn/dense TP
        # would need a shard exchange this layer does not perform. Refuse rather
        # than return wrong numbers (same guard and reasoning as Gemma 3).
        if mapping.attn.tp_size != mapping.dense.tp_size:
            raise ValueError(
                "Gemma 4 requires attn.tp_size == dense.tp_size, got "
                f"{mapping.attn.tp_size} != {mapping.dense.tp_size}."
            )
        # This is a text-only DENSE port. Refuse a config that enables a
        # dropped feature (MoE / double-wide MLP / per-layer input embeddings /
        # KV-sharing) rather than silently running the dense path and producing
        # wrong numbers. The gemma-4-31B-it checkpoint has all of these
        # disabled, so this is a clean no-op there.
        _reject_unsupported_features(_text_config(config))
        # Drive the LM off the text sub-config so BaseCausalLM sees hidden_size,
        # vocab_size, tie_word_embeddings and final_logit_softcapping=30.0; the
        # softcap LogitsProcessor and the tied lm_head follow from this config
        # with nothing further to plumb here.
        super().__init__(
            config=_text_config(config),
            mapping=mapping,
            quant_config=quant_config,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate to the model's tensor-returning scaled embedding.

        Keeps the ``sqrt(hidden_size)`` normalizer in one place
        (:meth:`Gemma4Model.get_input_embeddings`) and preserves the
        tensor-returning form the prefill-graph embedding fix relies on.

        Args:
            input_ids: Token ids, shape ``[num_tokens]``.

        Returns:
            The scaled token embeddings, shape ``[num_tokens, hidden_size]``.
        """
        return self.model.get_input_embeddings(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):
        """Load the text decoder from the Gemma 4 multimodal checkpoint.

        The ``Gemma4ForConditionalGeneration`` checkpoint ships three top-level
        groups (confirmed from the gemma-4-31B-it
        ``model.safetensors.index.json``, every tensor prefixed ``model.``):

        * ``model.vision_tower.*`` (355 tensors) and
          ``model.embed_vision.*`` (``embedding_projection.weight``) -- the
          vision encoder and its projection into the text embedding space.
        * ``model.language_model.*`` (832 tensors) -- the text decoder:
          ``model.language_model.embed_tokens.weight``,
          ``model.language_model.norm.weight`` and
          ``model.language_model.layers.{i}.*``.

        This text port keeps only the ``model.language_model.*`` group and
        remaps it onto this module's ``model.*`` namespace. The checkpoint
        carries NO ``multi_modal_projector``, NO ``audio``/``audio_tower``, NO
        ``rotary_emb.inv_freq`` and NO ``lm_head`` tensors (the latter because
        ``tie_word_embeddings`` is true); the ``multi_modal_projector`` / audio
        skips are kept defensively so a sibling multimodal checkpoint variant
        that does ship them still loads text-only cleanly.

        Namespace facts that drive the remap and the later tasks (confirmed
        against the real index, enumerated by the task 6.1 dry run):

        * The text namespace is ``model.language_model.*`` directly -- there is
          NO inner ``model.`` (i.e. embeddings are
          ``model.language_model.embed_tokens.weight``, not
          ``model.language_model.model.embed_tokens.weight``), so the remap is
          ``model.language_model.* -> model.*``. The ``language_model.model.*``
          / ``language_model.*`` forms are also handled so a bare text-release
          layout loads too.
        * Full-attention layers (indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
          -- every 6th) ship ``self_attn.k_proj.weight`` but NO
          ``self_attn.v_proj.weight`` (``attention_k_eq_v``); sliding layers
          ship a real ``v_proj``. Duplicating K into the V shard is TASK 6.2,
          not done here.
        * Each layer ships a ``model.language_model.layers.{i}.layer_scalar``
          ([1]) buffer (60 total); loading those buffers is TASK 6.2.
        * Per-layer attention norms are ``q_norm``/``k_norm`` (learnable) only;
          there is no ``v_norm`` tensor (the module's ``v_norm`` is weightless),
          so nothing maps onto it.

        This 6.1 skeleton establishes the skip set + remap and loads every
        non-stacked parameter directly via its ``weight_loader`` /
        :func:`default_weight_loader`. The loader runs in three stages, all
        implemented here:

        * The stacked-fusing bodies (q/k/v -> ``qkv_proj``, gate/up ->
          ``gate_up_proj``), the full-layer K->V duplication, and the
          ``layer_scalar`` buffer load (TASK 6.2).
        * The strict missing/unclaimed coverage raise at the end: ``loaded`` /
          ``skipped`` are accumulated across the pass, then checked against the
          expected set so a renamed or dropped tensor fails loudly instead of
          mis-serving (TASK 6.3; see the comment at the raise for the exact
          composition of the expected set).

        Args:
            weights: Iterable of ``(name, tensor)`` pairs from the checkpoint.
            **kwargs: Accepted and ignored for loader call-site compatibility.

        Returns:
            None. Parameters are written in place through their weight loaders.
        """
        del kwargs  # Accepted for loader call-site compatibility; unused.
        # (fused param, checkpoint shard name, shard id). q/k/v fuse into the
        # single qkv_proj GEMM and gate/up into gate_up_proj; the fusing bodies
        # and the full-layer K->V duplication (TASK 6.2) live in the loop below.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Combined params + buffers lookup. The per-layer ``layer_scalar`` is a
        # registered buffer on Gemma4DecoderLayer (register_buffer, not an
        # nn.Parameter), so it is absent from named_parameters(); folding
        # named_buffers() in makes ``model.layers.{i}.layer_scalar`` resolvable
        # through the direct-load path below. named_parameters() wins on any
        # name collision (there is none here -- buffers and params are disjoint
        # by construction).
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        lookup: dict[str, torch.Tensor] = {**buffers_dict, **params_dict}
        # Full-attention layer indices (attention_k_eq_v). These layers ship a
        # k_proj but NO v_proj; their k weight is duplicated into the V shard of
        # qkv_proj at load (see below) so V == K. Sliding layers ship a real
        # v_proj and are loaded normally. Resolved from the per-layer type map
        # (self.config is the Gemma 4 TEXT sub-config, which carries
        # layer_types) so cache/compute/weight-load agree on which layers are
        # full.
        layer_types = gemma4_layer_types(self.config)
        full_layer_indices = frozenset(
            idx for idx, kind in enumerate(layer_types) if kind == FULL_ATTENTION
        )
        # Parse the ``layers.{i}.`` index out of a (post-remap) param name so a
        # k_proj can be classified full vs sliding without threading the index
        # through the loop.
        layer_index_re = re.compile(r"\blayers\.(\d+)\.")
        loaded: set[str] = set()
        skipped: list[str] = []
        for name, loaded_weight in weights:
            # Skip the vision/audio side of the *ForConditionalGeneration
            # checkpoint. The real gemma-4-31B-it index only has vision_tower +
            # embed_vision; multi_modal_projector / audio are kept defensively
            # for sibling multimodal variants. rotary_emb.inv_freq is a derived
            # buffer, never loaded.
            if (
                name.startswith("model.vision_tower")
                or name.startswith("vision_tower")
                or name.startswith("model.embed_vision")
                or name.startswith("embed_vision")
                or name.startswith("model.multi_modal_projector")
                or name.startswith("multi_modal_projector")
                or "audio_tower" in name
                or "audio" in name
                or "rotary_emb.inv_freq" in name
            ):
                continue
            # Text weights live under (model.)language_model.* in the
            # multimodal checkpoint; map onto this module's model.* namespace.
            # The gemma-4-31B-it layout is model.language_model.* directly (no
            # inner model.); the language_model.model.* / language_model.* forms
            # are handled too so a bare text-release layout loads as well.
            if name.startswith("model.language_model.model."):
                name = "model." + name[len("model.language_model.model.") :]
            elif name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
            elif name.startswith("language_model.model."):
                name = "model." + name[len("language_model.model.") :]
            elif name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]
            # Tied lm_head is not stored separately (tie_word_embeddings=true).
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
                # Fuse the checkpoint shard into its fused param: q/k/v ->
                # qkv_proj (shard_id "q"/"k"/"v") and gate/up -> gate_up_proj
                # (shard_id 0/1). The fused param's weight_loader narrows to the
                # shard and copies in place (mirrors gemma3.load_weights).
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                # attention_k_eq_v: full-attention layers ship k_proj but NO
                # v_proj, so when we load a full layer's k_proj into the K shard
                # we ALSO load the SAME weight into the V shard -- giving V == K
                # without any forward branch. The QKVParallelLinear V shard for
                # a full layer is sized num_global_key_value_heads *
                # global_head_dim, identical to its K shard, so the k weight
                # fits the v shard exactly. weight_loader copies the tensor into
                # the param slot (it does not mutate loaded_weight), so the same
                # tensor can be passed to both shards -- no clone needed.
                # Sliding layers are NOT touched here: they carry a real v_proj
                # that loads normally through this same loop, so they are never
                # double-loaded.
                if weight_name == "k_proj":
                    match = layer_index_re.search(name)
                    if match is not None and int(match.group(1)) in (
                        full_layer_indices
                    ):
                        param.weight_loader(param, loaded_weight, "v")
                        # The V shard target is the same fused qkv_proj param;
                        # it is already in ``loaded`` from the K shard above.
                break
            else:
                # Direct-load path for the non-stacked tensors: embeddings, the
                # four sandwich norms, the standalone q/k norms, o_proj,
                # down_proj, the final norm -- AND the per-layer ``layer_scalar``
                # [1] buffers (registered buffers, resolved via ``lookup`` which
                # folds in named_buffers). default_weight_loader broadcasts the
                # [1] scalar; buffers written here are counted in ``loaded`` so
                # task 6.3's coverage check sees them.
                if name not in lookup:
                    skipped.append(name)
                    continue
                param = lookup[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # Strict coverage. A checkpoint that renames or drops a tensor would
        # otherwise load "successfully" and generate wrong text -- the expensive
        # way to find out (same rationale as gemma3.load_weights). So after the
        # pass we require EXACT coverage: every expected name written exactly
        # once, and every checkpoint tensor claimed by some param/buffer.
        #
        # ``expected`` is the set of names the checkpoint is responsible for
        # filling, which is NOT simply ``set(params_dict)`` and NOT every
        # buffer either:
        #
        # * It MUST include the per-layer ``model.layers.{i}.layer_scalar``
        #   BUFFERS. Those are real checkpoint tensors (60 of them) loaded via
        #   the direct path above and added to ``loaded``; if they were absent
        #   from ``expected`` a checkpoint that silently dropped one would never
        #   be flagged. Hence ``layer_scalar`` buffers are folded in.
        # * It MUST NOT include the model's other buffers. ``named_buffers()``
        #   on the real gemma-4-31B-it geometry carries 122 entries: the 60
        #   ``layer_scalar`` buffers PLUS 60 weightless ``v_norm.weight``
        #   buffers (:class:`RMSNormNoWeight` registers a non-persistent ones
        #   buffer, never loaded) PLUS 2 RoPE ``cos_sin_cache`` buffers (derived
        #   at init, never loaded). Including those 62 would raise a false
        #   "never written" for tensors the checkpoint never ships. Selecting
        #   only names ending ``layer_scalar`` admits the loaded buffers and
        #   excludes the derived ones.
        # * The weightless ``v_norm`` has NO parameter at all (it is a buffer,
        #   excluded above), so ``set(params_dict)`` already omits it.
        # * The tied ``lm_head`` is skipped during iteration (never in
        #   ``loaded``). Because ``tie_word_embeddings`` makes
        #   ``self.lm_head.weight is self.model.embed_tokens.weight`` the SAME
        #   tensor object, ``named_parameters()`` dedups it by identity and
        #   lists it once under ``model.embed_tokens.weight`` -- there is NO
        #   separate ``lm_head.weight`` entry in ``params_dict`` (verified on
        #   the real geometry), so the tie needs no special exclusion.
        expected = set(params_dict) | {
            name for name in buffers_dict if name.endswith("layer_scalar")
        }
        missing = sorted(expected - loaded)
        if missing or skipped:
            raise ValueError(
                "Gemma 4 checkpoint did not match the model: "
                f"{len(missing)} parameter(s) never written "
                f"(e.g. {missing[:5]}), "
                f"{len(skipped)} checkpoint tensor(s) unclaimed "
                f"(e.g. {skipped[:5]})."
            )


# ``Gemma4ForCausalLM`` is the text-only architecture string; it maps to the
# same implementation (a bare text config has no ``.text_config``, so
# ``_text_config`` returns the config itself).
class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass


EntryClass = [Gemma4ForConditionalGeneration, Gemma4ForCausalLM]
