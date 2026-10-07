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

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    is_block_drafter,
    model_wide_kwargs,
    resolve_cache_layer_types,
    resolve_dtype,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(kw_only=True)
class MHAConfig(SoftmaxAttnConfig):
    # Mixed full+sliding models are ONE component carrying the inherited
    # per-layer cache_layer_types/window vectors; the per-layer kv-head counts
    # live on the cache-pool spec (``CachePoolSpec.layer_kv_head_counts``).

    # Per-layer ``(num_kv_heads, head_dim)`` PRE-TP, one pair per attention
    # layer, when a model's KV geometry differs BY LAYER (gemma-4: 4 x 512 on
    # full-attention layers, 16 x 256 on sliding ones). ``None`` is the uniform
    # default every other model takes: a single ``num_kv_heads`` / ``head_dim``
    # describes every layer and the cache path is byte-for-byte unchanged. The
    # field is only populated when ``generate`` sees >1 distinct geometry, so a
    # uniform model never leaves this None.
    layer_kv_geometry: tuple[tuple[int, int], ...] | None = None
    # BLASST skip-softmax sparsity, gluon MHA prefill only (gfx950); see
    # ServerArgs.skip_softmax_threshold.
    skip_softmax_threshold: float = 0.0

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = is_block_drafter(
            server_args.speculative_algorithm, is_draft
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"
        # "auto" means the model's own dtype (--kv-cache-dtype help). Whisper
        # fp16 checkpoints need a matching pool; bf16-hardcoded auto matched
        # no paged kernel when query was fp16.
        resolved_kv_cache_dtype = (
            model_config.dtype
            if kv_cache_dtype == "auto"
            else resolve_dtype(kv_cache_dtype)
        )

        hf_config = model_config.hf_config
        cache_layer_types = resolve_cache_layer_types(
            hf_config,
            num_layers=model_config.num_attention_layers,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
        )
        # The retention window belongs to the storage labels: a block drafter
        # has none of its own (its window is a compute mask on its layers).
        # Gemma3ForConditionalGeneration keeps ``sliding_window`` on text_config.
        if draft_block_decode:
            sliding_window_tokens = None
        else:
            sliding_window_tokens = getattr(hf_config, "sliding_window", None)
            if sliding_window_tokens is None:
                text_config = getattr(hf_config, "text_config", None)
                if text_config is not None:
                    sliding_window_tokens = getattr(text_config, "sliding_window", None)
        # Per-layer-type KV geometry, only when a model genuinely differs by
        # layer (gemma-4). A block drafter writes at the target's cache
        # locations, so it never carries its own per-layer geometry.
        layer_kv_geometry = (
            None if draft_block_decode else _per_layer_kv_geometry(model_config)
        )
        spec = cls(
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            cache_layer_types=cache_layer_types,
            sliding_window_tokens=sliding_window_tokens,
            layer_kv_geometry=layer_kv_geometry,
            skip_softmax_threshold=server_args.skip_softmax_threshold,
        )
        return AttnConfig(
            components=(spec,),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolved_kv_cache_dtype,
                kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
                draft_block_decode=draft_block_decode,
            ),
        )

    def _kv_cell_bytes(self, config: AttnConfig, kv_heads: int, head_dim: int) -> int:
        """Per-token K+V bytes of one layer at ``kv_heads`` x ``head_dim``.

        The one byte formula, so the uniform cell and the per-layer sum below
        charge an identical cell for an identical geometry.
        """
        cell = (
            max(kv_heads // self.attn_tp_size, 1)
            * head_dim
            * 2
            * torch._utils._element_size(config.kv_cache_dtype)
        )
        if config.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            cell += cell // 32
        return cell

    def cache_cell_size(self, config: AttnConfig) -> int:
        return self._kv_cell_bytes(config, self.num_kv_heads, self.head_dim)

    def total_cache_bytes_per_token(self, config: AttnConfig, num_layers: int) -> int:
        """Per-token KV bytes summed over this component's ``num_layers``.

        The uniform default (``layer_kv_geometry is None``) is exactly
        ``cache_cell_size`` times the layer count -- a byte-for-byte identity
        with the single-cell path every other model takes. Only a per-layer
        model (gemma-4) sums its actual, differing per-layer cells instead,
        because no single cell size describes a 4 x 512 full layer and a
        16 x 256 sliding layer at once.
        """
        geometry = self.layer_kv_geometry
        if geometry is None:
            return self.cache_cell_size(config) * num_layers
        if len(geometry) != num_layers:
            raise ValueError(
                f"layer_kv_geometry has {len(geometry)} entries but the MHA "
                f"component spans {num_layers} layers"
            )
        return sum(
            self._kv_cell_bytes(config, kv_heads, head_dim)
            for kv_heads, head_dim in geometry
        )


def _per_layer_kv_geometry(
    model_config: ModelConfig,
) -> tuple[tuple[int, int], ...] | None:
    """Per-layer ``(num_kv_heads, head_dim)`` PRE-TP, or None when uniform.

    Only gemma-4 splits its KV geometry by layer type today; every other MHA
    model has one geometry for every layer and takes the ``None`` uniform path
    (the cache layout and its byte budget are then identical to before). Even a
    gemma-4 config whose layers happen to share one geometry collapses to
    ``None`` so the per-layer code path is reserved for a genuine difference.
    """
    from tokenspeed.runtime.configs.model_config import is_gemma4

    if not is_gemma4(model_config.hf_config):
        return None
    from tokenspeed.runtime.models.gemma4 import gemma4_layer_kv_geometry

    geometry = gemma4_layer_kv_geometry(model_config.hf_config)
    if len(set(geometry)) < 2:
        return None
    return geometry
