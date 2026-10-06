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

"""The four ordinary cache families: MHA, MLA, DSA, MSA.

One recipe serves all four, and a heterogeneous draft too: what a layer costs
is dispatched on the attention config that owns it, so an MLA target with an
MHA draft is just layers with two different geometries in one plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import (
    DSAConfig,
    dsa_history_gather_workspace_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    MXFP8_KV_SCALE_TILE_TOKENS,
    CacheGroupDeclaration,
    hybrid_slab_group_size,
    layer_group_ids,
)


class OrdinaryRecipe(CacheRecipe):
    """MHA / MLA / DSA / MSA: one cache group per attention structure.

    Capacity comes from the profiled bytes-per-token rather than the parent
    size, and every group packs one CacheBlock per parent -- the identity
    grain is the block span.
    """

    def __init__(self, *, family, **kwargs) -> None:
        super().__init__(**kwargs)
        self.family = family

    # ---- layer vocabulary ----

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        ids = _config_group_ids(self.attn_config, self.num_target_layers)
        if self.draft_attn_config is None:
            return ids
        if self.draft_attn_config.prefix_granularity != self.prefix_granularity:
            raise ValueError("target and draft prefix granularities must match")
        return ids + _config_group_ids(self.draft_attn_config, self.num_draft_layers)

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        """Merged labels, target then draft, always one per layer.

        A side whose config labels cannot align per layer resolves to
        full-history rather than mislabeling a group: plain MLA/DSA configs
        declare no labels at all, and a NextN draft inherits the target
        hf_config's ``layer_types`` (one draft layer against 61 target
        labels).
        """
        target = tuple(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)
        if len(target) != self.num_target_layers:
            target = (FULL_ATTENTION,) * self.num_target_layers
        if self.draft_attn_config is None:
            return target
        draft = tuple(
            self.draft_attn_config.component(SoftmaxAttnConfig).cache_layer_types
        )
        if len(draft) != self.num_draft_layers:
            draft = (FULL_ATTENTION,) * self.num_draft_layers
        return target + draft

    @cached_property
    @override
    def layer_kv_head_counts(self) -> tuple[int, ...] | None:
        """Per-layer KV head count (PRE-TP) when a model differs by layer.

        None keeps the uniform default: every layer serves the pool's one
        head count and the cache spec carries no per-layer vector, exactly as
        before. Only a per-layer model (gemma-4) publishes its real per-layer
        counts so the pool and scheduler know the layers are not byte-uniform.
        """
        target = _config_kv_head_counts(self.attn_config, self.num_target_layers)
        draft = _config_kv_head_counts(self.draft_attn_config, self.num_draft_layers)
        if target is None and draft is None:
            return None
        target = target or (
            (self.attn_config.component(SoftmaxAttnConfig).num_kv_heads,)
            * self.num_target_layers
        )
        if self.draft_attn_config is None:
            return target
        draft = draft or (
            (self.draft_attn_config.component(SoftmaxAttnConfig).num_kv_heads,)
            * self.num_draft_layers
        )
        return target + draft

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        groups = super().groups()
        if self.family not in ("mla", "dsa") or self.attn_config.dcp_size == 1:
            return groups
        if self.draft_attn_config is not None:
            # The draft group would shard like the target's, but the draft's
            # DCP decode steps are unvalidated and AttnConfig already rejects
            # speculation under FlashMLA/GPU DSA DCP (docs/design/cache-concepts.md).
            raise ValueError(
                "Sharded MLA/DSA cache does not yet support a draft model; "
                "run DCP without speculative decoding"
            )
        return tuple(
            (replace(spec, shard_count=self.attn_config.dcp_size), fields)
            for spec, fields in groups
        )

    @override
    def workspace_bytes(self) -> int:
        """The query-context-parallel history gather workspace of GPU DSA:
        one whole history of latent rows plus index-K rows packed in the
        plane's format, reserved before the arena is sized
        (``AttnConfig.__post_init__`` has already pinned the family to GPU DSA
        for ``qcp_size > 1``). One workspace serves the target and the draft
        (the draft gathers into the target's buffers), so the two must pack
        index-K rows in the same format; a draft naming another is refused
        here, where both configs are visible."""
        if self.attn_config.qcp_size == 1:
            return 0
        target = self.attn_config.component(DSAConfig)
        if self.draft_attn_config is not None and target is not None:
            draft = self.draft_attn_config.component(DSAConfig)
            draft_format = None if draft is None else draft.index_k_format
            if draft_format != target.index_k_format:
                raise ValueError(
                    "query context parallelism shares one history gather "
                    "workspace between the target and the draft, so both must "
                    "store index keys in one format; the target's "
                    f"index_k_format is {target.index_k_format!r}, the draft's "
                    f"{draft_format!r}"
                )
        return dsa_history_gather_workspace_bytes(
            self.attn_config, max_model_len=self.attn_config.context_len
        )

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        capacity = super().token_capacity(layout, num_lcm_blocks)
        if self.token_limit is not None:
            capacity = min(
                capacity,
                self.token_limit // self.prefix_granularity * self.prefix_granularity,
            )
        return capacity

    # ---- geometry ----

    @property
    @override
    def alignment(self) -> int:
        return 1

    @property
    @override
    def max_padding_fraction(self) -> float:
        # One-block packing fixes each group's stride regardless of its layer
        # count or cache dtype. Small groups can therefore have arbitrary tail
        # padding; the profiled byte budget still bounds the arena capacity.
        return float("inf")

    @property
    def per_layer_geometry(self) -> bool:
        """Whether this model's layers are not byte-uniform across groups.

        The one predicate that separates the two capacity/packing regimes.
        It reuses the single signal a per-layer model already publishes --
        ``layer_kv_head_counts`` is ``None`` for every uniform model
        (gemma-3, llama, qwen, every MLA/DSA/MSA) and a per-layer vector only
        for a model (gemma-4) whose full and sliding layers differ in head_dim
        and therefore cannot share a byte-uniform slab. No ``if gemma4`` lives
        anywhere; both the packing and capacity seams branch on this alone.
        """
        return self.layer_kv_head_counts is not None

    @override
    def packing(
        self, groups: tuple[CacheGroupDeclaration, ...]
    ) -> Mapping[str, int] | None:
        """How many of each group's CacheBlocks share one physical parent.

        A per-layer-geometry model (gemma-4) returns ``None`` so :func:`pack`
        derives per-group packing from the groups' byte ratios plus the
        exact-page-stride constraints their distinct-width planes impose --
        the general path. Its two geometries are not byte-equal (full 4 x 512,
        sliding 16 x 256), so pinning every group to one CacheBlock per parent
        would size each group's stride at the whole LCM block while its payload
        fills only its own planes, blowing the padding fraction far past the
        limit. A uniform model keeps the shared-slab pin every current model
        relies on: one CacheBlock per parent, the block span as the identity
        grain.
        """
        if self.per_layer_geometry:
            return None
        return {spec.group_id: 1 for spec, _ in groups}

    # ---- fields ----

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id < self.num_target_layers:
            config, local_layer_id = self.attn_config, layer_id
        else:
            config = self.draft_attn_config
            local_layer_id = layer_id - self.num_target_layers
        return _config_layer_fields(
            config,
            layer_id=layer_id,
            local_layer_id=local_layer_id,
            group_id=group_id,
            occurrence=occurrence,
        )

    # ---- capacity: profiled bytes per token, not parent size ----

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        # A per-layer-geometry model derives per-group packing (its groups
        # pack at different CacheBlocks-per-parent, so a parent no longer spans
        # the identity grain), so the shared-slab shortcut below is wrong for
        # it: defer to the general base formula, which sizes a parent from the
        # tightest packing it actually carries (``_max_packing(layout) *
        # prefix_granularity``) and the budget against the packed LCM-block
        # bytes.
        if self.per_layer_geometry:
            return super().num_lcm_blocks(layout)
        bytes_per_token = _config_bytes_per_token(
            self.attn_config, self.num_target_layers
        )
        if self.draft_attn_config is not None:
            bytes_per_token += _config_bytes_per_token(
                self.draft_attn_config, self.num_draft_layers
            )
        if bytes_per_token <= 0:
            raise ValueError(
                f"KV cache cell size must be positive, got {bytes_per_token}"
            )
        # Every group packs one CacheBlock per parent, so a parent spans the
        # identity grain and profiled bytes/token size it directly.
        parent_tokens = self.prefix_granularity
        budgeted = self._budgeted_parents(
            self.cache_budget_bytes, bytes_per_token * parent_tokens
        )
        if self.token_limit is None:
            return budgeted
        # The DCP shard count, not layout packing: a layout that packs more
        # than one CacheBlock per parent without sharding must keep the
        # unsharded floor semantics of _capped_parents.
        shard_count = self._shard_counts[FULL_ATTENTION]
        if shard_count > 1:
            logical_pages = self.token_limit // parent_tokens
            if logical_pages < 1:
                raise ValueError(
                    "The configured token limit must hold at least one cache page"
                )
            return min(budgeted, (logical_pages + shard_count - 1) // shard_count)
        return self._capped_parents(budgeted, parent_tokens=parent_tokens)


def _config_kv_head_counts(config, num_layers: int) -> tuple[int, ...] | None:
    """Per-layer KV head count (PRE-TP) of one config, or None when uniform."""
    if config is None:
        return None
    spec = config.component(SoftmaxAttnConfig)
    geometry = getattr(spec, "layer_kv_geometry", None)
    if geometry is None:
        return None
    if len(geometry) != num_layers:
        raise ValueError("layer_kv_geometry must cover every layer")
    return tuple(kv_heads for kv_heads, _ in geometry)


def _config_bytes_per_token(config, num_layers: int) -> int:
    """Per-token KV bytes an ordinary config costs across its layers.

    The uniform default charges one cell size times the slab divisor -- the
    i-th layer of each group shares slab i, so the hybrid full+sliding model
    pays for the largest group's depth, not every layer. A per-layer model
    (gemma-4) whose layers differ in head_dim cannot share a slab (the rows
    are not byte-equal), so it charges each layer's own cell and skips the
    divisor; ``total_cache_bytes_per_token`` does the summation.
    """
    spec = config.component(SoftmaxAttnConfig)
    if getattr(spec, "layer_kv_geometry", None) is not None:
        return spec.total_cache_bytes_per_token(config, num_layers)
    return config.cache_cell_size() * _storage_layers(config, num_layers)


def _storage_layers(config, num_layers: int) -> int:
    spec = config.component(SoftmaxAttnConfig)
    group_size = hybrid_slab_group_size(
        spec.cache_layer_types,
        sliding_window_tokens=spec.sliding_window_tokens,
    )
    return group_size if group_size is not None else num_layers


def _config_group_ids(config, num_layers: int) -> tuple[str, ...]:
    """Per-layer group ids for one ordinary attention config."""
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, MHAConfig | MSAConfig):
        layer_types = tuple(spec.cache_layer_types)
        if layer_types:
            ids = tuple(
                layer_group_ids(
                    layer_types=layer_types,
                    sliding_window_tokens=spec.sliding_window_tokens,
                )
            )
            if len(ids) != num_layers:
                raise ValueError("cache group ids must cover every layer")
            return ids
    return (FULL_ATTENTION,) * num_layers


def _config_layer_fields(
    config, *, layer_id: int, local_layer_id: int, group_id: str, occurrence: int
) -> tuple[CacheFieldSpec, ...]:
    """What one layer costs, dispatched on the config that owns the layer."""
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, DSAConfig):
        return _mla_layer_fields(config, layer_id, occurrence) + (
            _index_k_field(config, layer_id),
        )
    if isinstance(spec, MSAConfig):
        fields = _mha_layer_fields(
            config, layer_id, local_layer_id, group_id, occurrence
        )
        if local_layer_id in spec.sparse_layer_ids:
            fields += (_index_k_field(config, layer_id),)
        return fields
    if isinstance(spec, MLAConfig):
        return _mla_layer_fields(config, layer_id, occurrence)
    if isinstance(spec, MHAConfig):
        return _mha_layer_fields(config, layer_id, local_layer_id, group_id, occurrence)
    raise TypeError(f"no ordinary cache recipe for {type(spec).__name__}")


def _mha_layer_geometry(spec, local_layer_id: int) -> tuple[int, int]:
    """This layer's ``(kv_heads, head_dim)`` AFTER TP sharding.

    The uniform default reads the one model-wide ``num_kv_heads`` / ``head_dim``
    every current model carries, so the resolved page is byte-identical to
    before. A per-layer model (gemma-4) reads this layer's own geometry out of
    ``layer_kv_geometry`` -- full-attention layers at 4 x 512, sliding at
    16 x 256 -- so a mixed-head_dim model pages each layer at its true width.
    """
    geometry = getattr(spec, "layer_kv_geometry", None)
    if geometry is None:
        return max(spec.num_kv_heads // spec.attn_tp_size, 1), spec.head_dim
    kv_heads, head_dim = geometry[local_layer_id]
    return max(kv_heads // spec.attn_tp_size, 1), head_dim


def _mha_plane_prefix(spec, group_id: str) -> str:
    """Plane namespace for an MHA layer's K/V planes.

    A uniform model keeps the bare ``unit`` prefix every current model uses,
    so its planes stay byte-for-byte ``unit.{occurrence}.k`` and the i-th
    layer of each cache group deliberately shares one plane (full and sliding
    layers are byte-equal, so the hybrid slab pairs them). A per-layer-geometry
    model (gemma-4) cannot share a plane across groups: a 4 x 512 full layer
    and a 16 x 256 sliding layer have different page strides, and
    ``pack``'s exact-page-stride check rejects two payloads on one plane. Such
    a model scopes the plane by its cache group so each geometry owns its own
    plane namespace.
    """
    if getattr(spec, "layer_kv_geometry", None) is None:
        return "unit"
    return f"unit.{group_id}"


def _mha_layer_fields(
    config, layer_id: int, local_layer_id: int, group_id: str, occurrence: int
):
    """One MHA layer's K/V pages, with mxfp8 scale planes when enabled."""
    spec = config.component(SoftmaxAttnConfig)
    mxfp8 = bool(config.kv_cache_mxfp8)
    if mxfp8 and config.prefix_granularity != MXFP8_KV_SCALE_TILE_TOKENS:
        raise AssertionError(
            "mxfp8 KV cache requires --prefix-granularity "
            f"{MXFP8_KV_SCALE_TILE_TOKENS} (the attention kernel consumes "
            "the interleaved paged scale layout)"
        )
    kv_heads, head_dim = _mha_layer_geometry(spec, local_layer_id)
    if config.prefix_granularity <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("MHA full-attention geometry must be positive")
    shape = (config.prefix_granularity, kv_heads, head_dim)
    kv_dtype = (
        # MXFP8 writes go through dtype-aware kernels, so the arena keeps the
        # fp8 view; the scatter-written paths fall back to uint8.
        cache_dtype_name(torch.float8_e4m3fn)
        if mxfp8
        else scatter_stored_dtype_name(config.kv_cache_dtype)
    )
    plane_prefix = _mha_plane_prefix(spec, group_id)
    fields = (
        CacheFieldSpec(
            f"layer.{layer_id}.k", f"{plane_prefix}.{occurrence}.k", shape, kv_dtype
        ),
        CacheFieldSpec(
            f"layer.{layer_id}.v", f"{plane_prefix}.{occurrence}.v", shape, kv_dtype
        ),
    )
    if not mxfp8:
        return fields
    return fields + mxfp8_kv_scale_fields(
        layer_id=layer_id,
        occurrence=occurrence,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_granularity=config.prefix_granularity,
    )


def _mla_layer_fields(config, layer_id: int, occurrence: int):
    """One MLA layer's latent page, split into planes when quantized."""
    spec = config.component(SoftmaxAttnConfig)
    if config.prefix_granularity <= 0:
        raise ValueError("MLA full-attention geometry must be positive")
    if config.kv_cache_quant_method != "per_token_head":
        latent_width = spec.kv_lora_rank + spec.qk_rope_head_dim
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.latent_kv",
                f"slot.{occurrence}",
                (config.prefix_granularity, 1, latent_width),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            ),
        )
    return tuple(
        CacheFieldSpec(
            f"layer.{layer_id}.{name}",
            f"layer.{layer_id}.{name}",
            shape,
            dtype,
        )
        for name, shape, dtype in (
            (
                "latent_kv",
                (config.prefix_granularity, 1, spec.kv_lora_rank),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            ),
            (
                "latent_scale",
                (config.prefix_granularity, 1, 1),
                cache_dtype_name(torch.float32),
            ),
            (
                "rope_k",
                (config.prefix_granularity, 1, spec.qk_rope_head_dim),
                cache_dtype_name(config.dtype),
            ),
        )
    )


def _index_k_field(config, layer_id: int) -> CacheFieldSpec:
    """The sparse indexer's key row for one layer (DSA bytes, MSA elements)."""
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        DSAConfig,
        dsa_index_k_row_bytes,
        index_k_plane_dtype,
    )

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, DSAConfig):
        # One plane layout per index_k_format: FP8 keys plus scales as uint8
        # bytes, or the bf16 keys unquantized (configs/dsa.py).
        if spec.index_k_format == "fp8_scaled":
            return CacheFieldSpec(
                f"layer.{layer_id}.index_k",
                f"layer.{layer_id}.index_k",
                (
                    config.prefix_granularity,
                    dsa_index_k_row_bytes(spec.index_head_dim),
                ),
                "uint8",
            )
        return CacheFieldSpec(
            f"layer.{layer_id}.index_k",
            f"layer.{layer_id}.index_k",
            (config.prefix_granularity, spec.index_head_dim),
            cache_dtype_name(index_k_plane_dtype(spec.index_k_format)),
        )
    return CacheFieldSpec(
        f"layer.{layer_id}.index_k",
        f"layer.{layer_id}.index_k",
        (config.prefix_granularity, spec.index_head_dim),
        cache_dtype_name(config.dtype),
    )
