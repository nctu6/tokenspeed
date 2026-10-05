"""Meta-device wiring tests for the Gemma 4 text decoder.

Covers the gemma-4-specific wiring that distinguishes it from gemma-3:
registration of both architecture strings; the per-layer-type head_dim / kv-head
split; the three per-head norms (learnable q/k, weightless v); the tied lm_head;
the reinstated final-logit softcap of 30; the per-layer ``layer_scalar``; and the
strict weight-load coverage against the real checkpoint index.
"""

import json
import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op). ``test/`` is three parents up
# from this file (test/runtime/models/), so "ci_system.ci_register" resolves.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.layernorm import (  # noqa: E402
    RMSNorm,
    RMSNormNoWeight,
)
from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    Gemma4RotaryEmbedding,
)
from tokenspeed.runtime.models.gemma4 import (  # noqa: E402
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4LayerConfig,
    Gemma4MLP,
    Gemma4Model,
    _build_rope,
    gemma4_layer_config,
    gemma4_layer_kv_geometry,
    gemma4_layer_types,
)


def _text_config(layer_types: list[str] | None) -> SimpleNamespace:
    """A tiny stand-in for the gemma-4-31B ``text_config``.

    Carries only the fields the config helpers read: the per-layer-type
    head_dim / kv-head split and the explicit ``layer_types`` list.
    """
    return SimpleNamespace(
        head_dim=256,
        global_head_dim=512,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        layer_types=layer_types,
    )


def _config(layer_types: list[str] | None) -> SimpleNamespace:
    """The multimodal gemma-4 config wrapping the text sub-config."""
    return SimpleNamespace(text_config=_text_config(layer_types))


def _five_to_one_layout(num_layers: int) -> list[str]:
    """gemma-4's 5 sliding : 1 full layout: a full layer every 6th position
    (indices 5, 11, ...), sliding everywhere else."""
    return [
        FULL_ATTENTION if (i + 1) % 6 == 0 else SLIDING_ATTENTION
        for i in range(num_layers)
    ]


# A short 12-layer slice of the layout for the per-index resolution tests.
_LAYER_TYPES = _five_to_one_layout(12)
# The real gemma-4-31B depth: 60 layers, 10 full (indices 5, 11, ..., 59).
_NUM_LAYERS_31B = 60
_LAYER_TYPES_31B = _five_to_one_layout(_NUM_LAYERS_31B)

# gemma-4-31B's per-layer-type geometry, PRE-TP (num_kv_heads, head_dim).
_FULL_GEOMETRY = (4, 512)
_SLIDING_GEOMETRY = (16, 256)


def test_layer_types_returns_the_explicit_list():
    config = _config(_LAYER_TYPES)
    assert gemma4_layer_types(config) == _LAYER_TYPES


def test_layer_types_reads_through_a_bare_text_config():
    text_config = _text_config(_LAYER_TYPES)
    assert gemma4_layer_types(text_config) == _LAYER_TYPES


def test_layer_types_raises_when_absent():
    config = _config(None)
    with pytest.raises(ValueError, match="layer_types"):
        gemma4_layer_types(config)


def test_full_layer_geometry_uses_the_global_split():
    """A full_attention layer resolves to global_head_dim / global kv heads."""
    config = _config(_LAYER_TYPES)
    full_idx = _LAYER_TYPES.index(FULL_ATTENTION)
    resolved = gemma4_layer_config(config, full_idx)
    assert resolved.head_dim == 512
    assert resolved.num_key_value_heads == 4


def test_sliding_layer_geometry_uses_the_flat_split():
    """A sliding_attention layer keeps the flat head_dim / kv heads."""
    config = _config(_LAYER_TYPES)
    sliding_idx = _LAYER_TYPES.index(SLIDING_ATTENTION)
    resolved = gemma4_layer_config(config, sliding_idx)
    assert resolved.head_dim == 256
    assert resolved.num_key_value_heads == 16


# ---------------------------------------------------------------------------
# Full 60-layer gemma-4-31B layer-type map
# ---------------------------------------------------------------------------


def test_layer_types_full_depth_is_five_to_one_sliding_to_full():
    """The real 60-layer config resolves to 10 full : 50 sliding, with the
    full layers at every 6th index (5, 11, ..., 59)."""
    config = _config(_LAYER_TYPES_31B)
    resolved = gemma4_layer_types(config)
    assert len(resolved) == _NUM_LAYERS_31B
    full_indices = [i for i, t in enumerate(resolved) if t == FULL_ATTENTION]
    assert full_indices == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    assert resolved.count(SLIDING_ATTENTION) == 50
    assert resolved.count(FULL_ATTENTION) == 10


def test_layer_types_returns_a_fresh_list_not_the_config_object():
    """``gemma4_layer_types`` copies the config list so a caller mutating the
    result cannot corrupt the shared config."""
    source = _LAYER_TYPES_31B
    config = _config(source)
    resolved = gemma4_layer_types(config)
    assert resolved == source
    resolved[0] = FULL_ATTENTION
    assert source[0] == SLIDING_ATTENTION


# ---------------------------------------------------------------------------
# Per-layer geometry resolution across the whole depth
# ---------------------------------------------------------------------------


def test_every_full_layer_resolves_to_the_global_geometry():
    """All 10 full-attention layers resolve to 512 head_dim / 4 kv heads."""
    config = _config(_LAYER_TYPES_31B)
    for layer_idx, layer_type in enumerate(_LAYER_TYPES_31B):
        if layer_type != FULL_ATTENTION:
            continue
        resolved = gemma4_layer_config(config, layer_idx)
        assert resolved == Gemma4LayerConfig(head_dim=512, num_key_value_heads=4)


def test_every_sliding_layer_resolves_to_the_flat_geometry():
    """All 50 sliding-attention layers resolve to 256 head_dim / 16 kv heads."""
    config = _config(_LAYER_TYPES_31B)
    for layer_idx, layer_type in enumerate(_LAYER_TYPES_31B):
        if layer_type != SLIDING_ATTENTION:
            continue
        resolved = gemma4_layer_config(config, layer_idx)
        assert resolved == Gemma4LayerConfig(head_dim=256, num_key_value_heads=16)


def test_layer_config_reads_through_a_bare_text_config():
    """The resolver works off a bare text config, not only the wrapper."""
    text_config = _text_config(_LAYER_TYPES_31B)
    assert gemma4_layer_config(text_config, 5) == Gemma4LayerConfig(
        head_dim=512, num_key_value_heads=4
    )
    assert gemma4_layer_config(text_config, 0) == Gemma4LayerConfig(
        head_dim=256, num_key_value_heads=16
    )


def test_full_layer_without_a_global_split_falls_back_to_the_flat_values():
    """When ``global_head_dim`` / ``num_global_key_value_heads`` are absent, a
    full layer resolves to the flat head_dim / kv heads rather than crashing."""
    text_config = SimpleNamespace(
        head_dim=256,
        global_head_dim=None,
        num_key_value_heads=16,
        num_global_key_value_heads=None,
        layer_types=_LAYER_TYPES_31B,
    )
    resolved = gemma4_layer_config(text_config, 5)
    assert resolved == Gemma4LayerConfig(head_dim=256, num_key_value_heads=16)


# ---------------------------------------------------------------------------
# gemma4_layer_kv_geometry: the two cache groups the KV pool sizes from
# ---------------------------------------------------------------------------


def test_kv_geometry_emits_one_pair_per_layer_in_order():
    """``gemma4_layer_kv_geometry`` yields one (kv_heads, head_dim) pair per
    layer, in layer order, matching each layer's resolved geometry."""
    config = _config(_LAYER_TYPES_31B)
    geometry = gemma4_layer_kv_geometry(config)
    assert len(geometry) == _NUM_LAYERS_31B
    expected = tuple(
        _FULL_GEOMETRY if lt == FULL_ATTENTION else _SLIDING_GEOMETRY
        for lt in _LAYER_TYPES_31B
    )
    assert geometry == expected


def test_kv_geometry_has_exactly_two_distinct_groups():
    """The 60 layers collapse to exactly the two gemma-4 cache groups: a full
    group (4 x 512) and a sliding group (16 x 256)."""
    config = _config(_LAYER_TYPES_31B)
    geometry = gemma4_layer_kv_geometry(config)
    groups = set(geometry)
    assert groups == {_FULL_GEOMETRY, _SLIDING_GEOMETRY}
    assert geometry.count(_FULL_GEOMETRY) == 10
    assert geometry.count(_SLIDING_GEOMETRY) == 50


def test_kv_geometry_full_group_widens_head_dim_and_shrinks_kv_heads():
    """The gemma-4 contrast: the full group has the WIDER head_dim (512 > 256)
    but FEWER kv heads (4 < 16) than the sliding group."""
    config = _config(_LAYER_TYPES_31B)
    geometry = gemma4_layer_kv_geometry(config)
    full_kv_heads, full_head_dim = geometry[5]
    sliding_kv_heads, sliding_head_dim = geometry[0]
    assert full_head_dim > sliding_head_dim
    assert full_kv_heads < sliding_kv_heads


def test_kv_geometry_reads_through_a_bare_text_config():
    text_config = _text_config(_LAYER_TYPES_31B)
    assert gemma4_layer_kv_geometry(text_config)[5] == _FULL_GEOMETRY
    assert gemma4_layer_kv_geometry(text_config)[0] == _SLIDING_GEOMETRY


def test_kv_geometry_raises_when_layer_types_absent():
    """Without an explicit layer_types list the geometry cannot be built."""
    config = _config(None)
    with pytest.raises(ValueError, match="layer_types"):
        gemma4_layer_kv_geometry(config)


# ---------------------------------------------------------------------------
# Per-layer-type RoPE: inv_freq + cos/sin vs an independent reference
# (Validates: Property 3)
# ---------------------------------------------------------------------------
#
# These tests check the two RoPE instances ``_build_rope`` produces against a
# hand computation of the HF/gemma-4 formula AND against the vllm-unieai
# ``gemma4_rope.py`` oracle. Importing the vllm-unieai oracle in this suite is
# impractical (the vllm package has heavy import side-effects and is not on the
# tokenspeed test path), so the oracle's exact ``_compute_inv_freq`` arithmetic
# is replicated inline below. tokenspeed's ``Gemma4RotaryEmbedding`` and the
# oracle share a byte-identical ``_compute_inv_freq``, so the hand computation
# and the oracle formula are identical by construction -- the replicated
# ``_oracle_proportional_inv_freq`` is that single shared formula.
#
# CPU-only: the RoPE instances are built directly with an explicit, small
# ``max_position`` (no server args, no GPU, no model construction).

# gemma-4-31B RoPE facts, per attention type.
_FULL_HEAD_DIM = 512
_FULL_ROPE_THETA = 1_000_000.0
_FULL_PARTIAL_ROTARY_FACTOR = 0.25
# rotary_dim = int(512 * 0.25) = 128 -> 64 rotated angle pairs.
_FULL_ROTARY_DIM = 128
_FULL_ROPE_ANGLES = 64
_FULL_NOPE_ANGLES = 192  # 512 // 2 - 64

_SLIDING_HEAD_DIM = 256
_SLIDING_ROPE_THETA = 10_000.0

# A small cache depth: large enough to sample the positions below, tiny enough
# to build on CPU in a unit test.
_ROPE_MAX_POSITION = 256
# Sample positions for the cos/sin spot-checks (within _ROPE_MAX_POSITION).
_SAMPLE_POSITIONS = [0, 1, 2, 5, 37, 128]


def _rope_config() -> SimpleNamespace:
    """A gemma-4-31B-like text config carrying only the RoPE-relevant fields.

    Wraps the per-layer-type head_dim split and the nested per-attention-type
    ``rope_parameters`` that ``_build_rope`` reads; ``layer_types`` is present
    so the shared config helpers resolve.
    """
    text_config = SimpleNamespace(
        head_dim=_SLIDING_HEAD_DIM,
        global_head_dim=_FULL_HEAD_DIM,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        max_position_embeddings=262144,
        layer_types=[SLIDING_ATTENTION, FULL_ATTENTION],
        rope_parameters={
            FULL_ATTENTION: {
                "rope_type": "proportional",
                "rope_theta": _FULL_ROPE_THETA,
                "partial_rotary_factor": _FULL_PARTIAL_ROTARY_FACTOR,
            },
            SLIDING_ATTENTION: {
                "rope_type": "default",
                "rope_theta": _SLIDING_ROPE_THETA,
            },
        },
    )
    return SimpleNamespace(text_config=text_config)


def _oracle_proportional_inv_freq(
    head_size: int, rotary_dim: int, base: float
) -> torch.Tensor:
    """The vllm-unieai ``Gemma4RotaryEmbedding._compute_inv_freq`` arithmetic.

    Replicated verbatim from ``vllm-unieai/.../gemma4_rope.py`` (see module
    note): the HF proportional formula divides the exponent by ``head_size``
    (not ``rotary_dim``), then zero-pads the non-rotated dimensions so their
    rotation collapses to the identity.
    """
    rope_angles = rotary_dim // 2
    nope_angles = (head_size // 2) - rope_angles
    freq_exponents = (
        torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_size
    )
    inv_freq = 1.0 / (base**freq_exponents)
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles, dtype=torch.float)])
    return inv_freq


def _cache_cos_sin(rope, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read (cos, sin) rows for ``positions`` out of a RoPE's cos/sin cache.

    The cache is ``[rows, head_size]`` with the cos half first and the sin half
    second, exactly as ``RotaryEmbedding._compute_cos_sin_cache`` builds it.
    """
    rows = rope.cos_sin_cache[torch.tensor(positions)]
    cos, sin = rows.chunk(2, dim=-1)
    return cos, sin


def test_full_rope_inv_freq_matches_the_hand_proportional_formula():
    """full_attention inv_freq is 64 proportional freqs + 192 identity zeros.

    The hand computation is the HF proportional formula directly: 64 nonzero
    frequencies ``1 / (1e6 ** (arange(0, 128, 2) / 512))`` followed by 192
    zeros, total 256 entries (one per head_dim // 2 pair... here the padded
    inv_freq spans head_size // 2 = 256).
    """
    config = _rope_config()
    rope = _build_rope(config, FULL_ATTENTION, _ROPE_MAX_POSITION, torch.float32)

    assert isinstance(rope, Gemma4RotaryEmbedding)
    assert rope.head_size == _FULL_HEAD_DIM
    # rotary_dim is widened to head_size so the kernel spans the full head; the
    # zero-padded inv_freq makes the nope span an identity rotation.
    assert rope.rotary_dim == _FULL_HEAD_DIM
    assert rope.rope_angles == _FULL_ROPE_ANGLES
    assert rope.nope_angles == _FULL_NOPE_ANGLES

    hand_nonzero = 1.0 / (
        _FULL_ROPE_THETA
        ** (torch.arange(0, _FULL_ROTARY_DIM, 2, dtype=torch.float) / _FULL_HEAD_DIM)
    )
    hand = torch.cat(
        [hand_nonzero, torch.zeros(_FULL_NOPE_ANGLES, dtype=torch.float)]
    )
    assert hand.shape == (_FULL_HEAD_DIM // 2,)
    assert hand_nonzero.shape == (_FULL_ROPE_ANGLES,)

    inv_freq = rope._compute_inv_freq(rope.base)
    torch.testing.assert_close(inv_freq, hand, rtol=0.0, atol=0.0)


def test_full_rope_inv_freq_and_cos_sin_match_the_oracle():
    """full_attention inv_freq AND sampled cos/sin match the vllm-unieai oracle.

    The oracle is built from the SAME args the port uses (head_size 512,
    rotary_dim 128, base 1e6) via the replicated oracle ``_compute_inv_freq``;
    cos/sin are then formed the same way ``RotaryEmbedding`` does
    (``outer(positions, inv_freq)``) and compared tightly in float32.
    """
    config = _rope_config()
    rope = _build_rope(config, FULL_ATTENTION, _ROPE_MAX_POSITION, torch.float32)

    oracle_inv_freq = _oracle_proportional_inv_freq(
        _FULL_HEAD_DIM, _FULL_ROTARY_DIM, _FULL_ROPE_THETA
    )
    inv_freq = rope._compute_inv_freq(rope.base)
    torch.testing.assert_close(inv_freq, oracle_inv_freq, rtol=0.0, atol=0.0)

    positions = torch.tensor(_SAMPLE_POSITIONS, dtype=torch.float32)
    oracle_freqs = torch.outer(positions, oracle_inv_freq)
    oracle_cos = oracle_freqs.cos()
    oracle_sin = oracle_freqs.sin()

    cos, sin = _cache_cos_sin(rope, _SAMPLE_POSITIONS)
    torch.testing.assert_close(cos, oracle_cos, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sin, oracle_sin, rtol=1e-6, atol=1e-6)


def test_sliding_rope_inv_freq_matches_the_plain_hand_formula():
    """sliding_attention inv_freq is the plain full-rotary formula, no padding.

    128 frequencies ``1 / (1e4 ** (arange(0, 256, 2) / 256))`` over the whole
    256-wide head; rotary_dim is the full head_dim and there is no zero-pad.
    """
    config = _rope_config()
    rope = _build_rope(config, SLIDING_ATTENTION, _ROPE_MAX_POSITION, torch.float32)

    assert rope.head_size == _SLIDING_HEAD_DIM
    assert rope.rotary_dim == _SLIDING_HEAD_DIM

    hand = 1.0 / (
        _SLIDING_ROPE_THETA
        ** (
            torch.arange(0, _SLIDING_HEAD_DIM, 2, dtype=torch.float)
            / _SLIDING_HEAD_DIM
        )
    )
    assert hand.shape == (_SLIDING_HEAD_DIM // 2,)

    inv_freq = rope._compute_inv_freq(rope.base)
    torch.testing.assert_close(inv_freq, hand, rtol=0.0, atol=0.0)


def test_sliding_rope_cos_sin_match_the_hand_computation():
    """sliding_attention cos/sin equal cos/sin of ``outer(pos, inv_freq)``."""
    config = _rope_config()
    rope = _build_rope(config, SLIDING_ATTENTION, _ROPE_MAX_POSITION, torch.float32)

    hand_inv_freq = 1.0 / (
        _SLIDING_ROPE_THETA
        ** (
            torch.arange(0, _SLIDING_HEAD_DIM, 2, dtype=torch.float)
            / _SLIDING_HEAD_DIM
        )
    )
    positions = torch.tensor(_SAMPLE_POSITIONS, dtype=torch.float32)
    hand_freqs = torch.outer(positions, hand_inv_freq)

    cos, sin = _cache_cos_sin(rope, _SAMPLE_POSITIONS)
    torch.testing.assert_close(cos, hand_freqs.cos(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sin, hand_freqs.sin(), rtol=1e-6, atol=1e-6)


def test_full_rope_has_exactly_the_nope_identity_span_and_sliding_has_none():
    """The nope identity span is a gemma-4 full-attention-only trait.

    The full-attention inv_freq ends in exactly 192 zeros (the non-rotated
    dims that collapse to cos=1 / sin=0); the sliding-attention inv_freq has no
    trailing zeros because it rotates the whole head.
    """
    config = _rope_config()
    full = _build_rope(config, FULL_ATTENTION, _ROPE_MAX_POSITION, torch.float32)
    sliding = _build_rope(config, SLIDING_ATTENTION, _ROPE_MAX_POSITION, torch.float32)

    full_inv_freq = full._compute_inv_freq(full.base)
    trailing_zeros = (full_inv_freq == 0.0).to(torch.int)
    # Count the contiguous run of trailing zeros.
    reversed_run = 0
    for value in reversed(trailing_zeros.tolist()):
        if value == 1:
            reversed_run += 1
        else:
            break
    assert reversed_run == _FULL_NOPE_ANGLES
    assert int(trailing_zeros.sum()) == _FULL_NOPE_ANGLES

    sliding_inv_freq = sliding._compute_inv_freq(sliding.base)
    assert int((sliding_inv_freq == 0.0).sum()) == 0
    # The full-attention cos/sin identity span: cos=1, sin=0 on the nope dims
    # for every position.
    cos, sin = _cache_cos_sin(full, _SAMPLE_POSITIONS)
    nope_cos = cos[:, _FULL_ROPE_ANGLES:]
    nope_sin = sin[:, _FULL_ROPE_ANGLES:]
    torch.testing.assert_close(nope_cos, torch.ones_like(nope_cos), rtol=0.0, atol=0.0)
    torch.testing.assert_close(nope_sin, torch.zeros_like(nope_sin), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Gemma4Attention: per-layer-type geometry, three norms, forward ordering
# (Validates: Property 1; design Components)
# ---------------------------------------------------------------------------
#
# Two tiers, matching the 4.1/4.2 smoke-check pattern:
#
# * Construction-only asserts (geometry, q_size/kv_size, scaling, the three
#   norms' Parameter counts, sliding window, logit_cap) are pure Python/CPU --
#   building QKVParallelLinear/RowParallelLinear and the norms allocates CPU
#   tensors, no kernel runs -- so they run unconditionally.
# * Forward-exercising asserts (pre-attn q/k/v shapes, o_proj output shape, and
#   that V is normalized-not-rotated) need the fused RMSNorm + RoPE kernels,
#   which are CUDA-only, so they are gated behind a CUDA skip. They run on a
#   CUDA host; a CPU-only laptop skips them.
#
# The real ``self.attn`` (PagedAttention) needs a ForwardContext and a bound KV
# pool from executor startup, which a unit test does not stand up. The forward
# tests therefore replace ``attn.forward`` with a capture stub that records the
# (q, k, v) it is handed and returns a correctly-shaped attention output; the
# assertions are made against what the module computed BEFORE the attention
# kernel, which is the part this task covers.

# gemma-4-31B attention facts used by the Gemma4Attention tests.
_NUM_ATTENTION_HEADS = 32
_RMS_NORM_EPS = 1e-6
_HIDDEN_SIZE = 5376
# gemma-4-31B feed-forward width (used by the decoder-layer / layer_scalar
# tests that build a Gemma4DecoderLayer, which constructs the MLP).
_INTERMEDIATE_SIZE = 21504
# The sliding window on gemma-4-31B; window_left = sliding_window - 1.
_SLIDING_WINDOW = 1024

_cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Gemma4Attention.forward uses the fused RMSNorm/RoPE CUDA kernels; "
    "run on a CUDA host.",
)


def _attention_text_config() -> SimpleNamespace:
    """A gemma-4-31B-like text config carrying the fields Gemma4Attention reads.

    The per-layer-type head_dim / kv-head split, ``num_attention_heads``,
    ``hidden_size``, the explicit ``layer_types`` list (so the shared helpers
    resolve), ``sliding_window``, ``rms_norm_eps`` and the (absent) attention
    softcapping / bias.
    """
    return SimpleNamespace(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        head_dim=_SLIDING_HEAD_DIM,
        global_head_dim=_FULL_HEAD_DIM,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        sliding_window=_SLIDING_WINDOW,
        rms_norm_eps=_RMS_NORM_EPS,
        attention_bias=False,
        attn_logit_softcapping=None,
        # The decoder layer (and the layer_scalar tests that build one) needs
        # the MLP fields; the attention-only tests ignore them.
        intermediate_size=_INTERMEDIATE_SIZE,
        hidden_activation="gelu_pytorch_tanh",
        layer_types=_LAYER_TYPES_31B,
        # ``_build_rope`` reads the nested per-attention-type RoPE params; the
        # two layer types this suite exercises must both be present.
        rope_parameters={
            FULL_ATTENTION: {
                "rope_type": "proportional",
                "rope_theta": _FULL_ROPE_THETA,
                "partial_rotary_factor": _FULL_PARTIAL_ROTARY_FACTOR,
            },
            SLIDING_ATTENTION: {
                "rope_type": "default",
                "rope_theta": _SLIDING_ROPE_THETA,
            },
        },
    )


def _attention_config() -> SimpleNamespace:
    """The multimodal gemma-4 config wrapping the attention text sub-config."""
    return SimpleNamespace(text_config=_attention_text_config())


def _tp1_mapping() -> Mapping:
    """A single-rank mapping (tp_size=1), as the gemma3/model suites build."""
    return Mapping(rank=0, world_size=1)


def _build_attention(layer_id: int, layer_type: str, device: str) -> Gemma4Attention:
    """Construct a Gemma4Attention for ``layer_id`` on ``device``.

    The RoPE instance is the real one ``_build_rope`` returns for the layer
    type (built with a small cache so CPU construction is cheap). Weights are
    allocated on ``device`` by building under its default-device context.
    """
    config = _attention_config()
    with torch.device(device):
        # Build the RoPE under the same device context so its cos/sin cache
        # buffer lands on ``device`` -- the fused RoPE kernel requires the
        # cache on the same device as q/k.
        rotary_emb = _build_rope(config, layer_type, _ROPE_MAX_POSITION, torch.float32)
        return Gemma4Attention(
            config=config,
            mapping=_tp1_mapping(),
            layer_id=layer_id,
            layer_type=layer_type,
            rotary_emb=rotary_emb,
            quant_config=None,
            prefix=f"model.layers.{layer_id}.self_attn",
        )


# A full layer (index 5) and a sliding layer (index 0) on the 31B layout.
_FULL_LAYER_ID = 5
_SLIDING_LAYER_ID = 0


# ---------------------------------------------------------------------------
# Construction-only asserts (CPU): geometry, scaling, norms, window, logit_cap
# ---------------------------------------------------------------------------


def test_full_attention_geometry_uses_the_global_split() -> None:
    """A full layer resolves to head_dim 512 / 4 kv-heads, with q_size and
    kv_size derived from that geometry and 32 query heads."""
    attn = _build_attention(_FULL_LAYER_ID, FULL_ATTENTION, "cpu")
    assert attn.head_dim == 512
    assert attn.num_heads == _NUM_ATTENTION_HEADS
    assert attn.num_kv_heads == 4
    assert attn.q_size == _NUM_ATTENTION_HEADS * 512
    assert attn.kv_size == 4 * 512


def test_sliding_attention_geometry_uses_the_flat_split() -> None:
    """A sliding layer resolves to head_dim 256 / 16 kv-heads, with q_size and
    kv_size derived from that geometry and 32 query heads."""
    attn = _build_attention(_SLIDING_LAYER_ID, SLIDING_ATTENTION, "cpu")
    assert attn.head_dim == 256
    assert attn.num_heads == _NUM_ATTENTION_HEADS
    assert attn.num_kv_heads == 16
    assert attn.q_size == _NUM_ATTENTION_HEADS * 256
    assert attn.kv_size == 16 * 256


def test_attention_scaling_is_one_on_both_layer_types() -> None:
    """Gemma 4 does not use query_pre_attn_scalar: softmax scaling is 1.0 (the
    learnable q/k norms carry the scaling), on both the module and the kernel.
    """
    full = _build_attention(_FULL_LAYER_ID, FULL_ATTENTION, "cpu")
    sliding = _build_attention(_SLIDING_LAYER_ID, SLIDING_ATTENTION, "cpu")
    assert full.scaling == 1.0
    assert sliding.scaling == 1.0
    assert full.attn.scaling == 1.0
    assert sliding.attn.scaling == 1.0


def test_v_norm_has_no_weight_while_qk_norms_each_have_one() -> None:
    """The gemma-4 norm contrast (Property 1 / design Components).

    ``q_norm`` and ``k_norm`` are learnable ``RMSNorm(head_dim)`` -- each
    exposes exactly one Parameter ``weight`` of shape ``(head_dim,)`` -- while
    ``v_norm`` is a weightless ``RMSNormNoWeight(head_dim)`` exposing ZERO
    Parameters. Checked on both layer types so the head_dim the norms span
    tracks the layer's geometry.
    """
    for layer_id, layer_type, head_dim in (
        (_FULL_LAYER_ID, FULL_ATTENTION, 512),
        (_SLIDING_LAYER_ID, SLIDING_ATTENTION, 256),
    ):
        attn = _build_attention(layer_id, layer_type, "cpu")

        assert isinstance(attn.q_norm, RMSNorm)
        assert isinstance(attn.k_norm, RMSNorm)
        assert isinstance(attn.v_norm, RMSNormNoWeight)

        q_params = list(attn.q_norm.parameters())
        k_params = list(attn.k_norm.parameters())
        assert len(q_params) == 1
        assert len(k_params) == 1
        assert q_params[0].shape == (head_dim,)
        assert k_params[0].shape == (head_dim,)

        # v_norm is weightless: no learnable Parameter at all.
        assert list(attn.v_norm.parameters()) == []
        assert not any(
            "v_norm" in name for name, _ in attn.named_parameters()
        )


def test_sliding_layer_window_is_hf_window_minus_one() -> None:
    """A sliding layer's kernel window_left is ``sliding_window - 1`` (HF counts
    the current token; the kernel takes earlier-tokens-visible), and its logit
    cap is off."""
    attn = _build_attention(_SLIDING_LAYER_ID, SLIDING_ATTENTION, "cpu")
    assert attn.is_sliding is True
    assert attn.attn.sliding_window_size == _SLIDING_WINDOW - 1
    assert attn.attn.logit_cap == 0.0


def test_full_layer_has_no_window_and_no_logit_cap() -> None:
    """A full layer attends the whole history (window_left -1) with no logit
    cap (gemma-4 attn_logit_softcapping is None)."""
    attn = _build_attention(_FULL_LAYER_ID, FULL_ATTENTION, "cpu")
    assert attn.is_sliding is False
    assert attn.attn.sliding_window_size == -1
    assert attn.attn.logit_cap == 0.0


# ---------------------------------------------------------------------------
# Forward-exercising asserts (CUDA): pre-attn shapes, V-not-rotated, o_proj
# ---------------------------------------------------------------------------


class _CaptureAttn(torch.nn.Module):
    """Stand-in for ``PagedAttention`` that records the (q, k, v) it is handed.

    A unit test cannot stand up the ForwardContext + bound KV pool a real
    PagedAttention needs, so the module's ``attn`` submodule is swapped for
    this. It is an ``nn.Module`` so it can be assigned over the registered
    ``attn`` child. It records the exact tensors the attention kernel would
    have seen (so the test can assert their shapes and that V is the ``v_norm``
    output, not a rotated tensor) and returns a correctly-shaped attention
    output ``[num_tokens, num_heads * head_dim]`` so ``o_proj`` runs normally.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = None
        self.k = None
        self.v = None

    def forward(self, q, k, v, ctx):
        self.q = q.detach().clone()
        self.k = k.detach().clone()
        self.v = v.detach().clone()
        num_tokens = q.shape[0]
        return q.new_zeros((num_tokens, self.num_heads * self.head_dim))


def _run_forward_with_capture(
    layer_id: int, layer_type: str, num_tokens: int
) -> tuple[Gemma4Attention, _CaptureAttn, torch.Tensor, torch.Tensor]:
    """Build a CUDA Gemma4Attention, capture its pre-attn (q, k, v), return the
    module, the capture, the layer input and the o_proj output.

    Construction is under bf16 default dtype on CUDA (the smoke-check pattern):
    the fused RMSNorm/RoPE kernels and the linear GEMMs all run on CUDA bf16.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = _build_attention(layer_id, layer_type, "cuda")
    finally:
        torch.set_default_dtype(old_dtype)

    # ``get_rope`` caches sliding RoPE instances globally by their build args
    # (device is not part of the key), so a sliding RoPE built on CPU by the
    # construction-only tests can be handed back here with a CPU cos/sin cache.
    # The fused RoPE kernel requires the cache on the q/k device, so move it to
    # CUDA defensively (the full-attention RoPE is a fresh per-call instance and
    # is already on CUDA from the device context, but this is harmless there).
    attn.rotary_emb.cos_sin_cache = attn.rotary_emb.cos_sin_cache.cuda()

    capture = _CaptureAttn(attn.num_heads, attn.head_dim)
    attn.attn = capture

    # The fused QKV / o_proj weights are zero-initialized and this unit test
    # does not load a checkpoint; fill them with random values so q/k/v are
    # non-trivial (a zero projection would make every shape and the
    # V-not-rotated check pass vacuously).
    torch.manual_seed(0)
    with torch.no_grad():
        attn.qkv_proj.weight.normal_(mean=0.0, std=0.02)
        attn.o_proj.weight.normal_(mean=0.0, std=0.02)

    hidden_states = torch.randn(
        num_tokens, _HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    output = attn.forward(positions, hidden_states, ctx=None)
    return attn, capture, hidden_states, output


@_cuda_only
def test_full_layer_pre_attn_shapes_and_output_shape() -> None:
    """Property 1 / per-layer-type output shapes, full layer.

    The q/k/v handed to attention are ``(num_tokens, q_size)`` /
    ``(num_tokens, kv_size)`` at the FULL geometry (head_dim 512, 4 kv-heads,
    32 q-heads), and o_proj projects back to ``(num_tokens, hidden_size)``.
    """
    num_tokens = 3
    attn, capture, _, output = _run_forward_with_capture(
        _FULL_LAYER_ID, FULL_ATTENTION, num_tokens
    )
    assert capture.q.shape == (num_tokens, _NUM_ATTENTION_HEADS * 512)
    assert capture.k.shape == (num_tokens, 4 * 512)
    assert capture.v.shape == (num_tokens, 4 * 512)
    assert output.shape == (num_tokens, _HIDDEN_SIZE)


@_cuda_only
def test_sliding_layer_pre_attn_shapes_and_output_shape() -> None:
    """Property 1 / per-layer-type output shapes, sliding layer.

    The q/k/v handed to attention are at the SLIDING geometry (head_dim 256, 16
    kv-heads, 32 q-heads); o_proj projects back to ``(num_tokens, hidden_size)``.
    """
    num_tokens = 4
    attn, capture, _, output = _run_forward_with_capture(
        _SLIDING_LAYER_ID, SLIDING_ATTENTION, num_tokens
    )
    assert capture.q.shape == (num_tokens, _NUM_ATTENTION_HEADS * 256)
    assert capture.k.shape == (num_tokens, 16 * 256)
    assert capture.v.shape == (num_tokens, 16 * 256)
    assert output.shape == (num_tokens, _HIDDEN_SIZE)


@_cuda_only
def test_v_is_normalized_not_rotated() -> None:
    """V reaches attention as ``v_norm(v_split)`` with NO RoPE applied.

    Recompute the expected V independently: pull the fused qkv projection, take
    the V split, apply the module's own ``v_norm`` -- and assert it equals the
    captured V. Then confirm V was NOT rotated by showing a RoPE-rotated V
    would differ from the captured V (RoPE on these random vectors changes
    them), so the equality above is meaningful rather than vacuous.
    """
    num_tokens = 5
    attn, capture, hidden_states, _ = _run_forward_with_capture(
        _SLIDING_LAYER_ID, SLIDING_ATTENTION, num_tokens
    )

    qkv, _ = attn.qkv_proj(hidden_states)
    _, _, v_split = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    expected_v = attn.v_norm(v_split.reshape(-1, attn.head_dim)).view(
        num_tokens, attn.kv_size
    )
    torch.testing.assert_close(capture.v, expected_v, rtol=0.0, atol=0.0)

    # A RoPE-rotated V would differ: rotate expected_v through the same RoPE
    # the layer uses (as a stand-in q, since RoPE rotates both its args) and
    # show it is not what attention received -- i.e. V skipped RoPE.
    positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    rotated_v, _ = attn.rotary_emb(positions, expected_v.clone(), expected_v.clone())
    assert not torch.allclose(rotated_v, capture.v, rtol=1e-3, atol=1e-3)

# ---------------------------------------------------------------------------
# Task-5 components: MLP guard, decoder layer_scalar, model embedding
# normalizer, LM-head softcap + tied head
# (Validates: Property 4 embedding scale; Property 6 final logit softcap)
# ---------------------------------------------------------------------------
#
# Same two-tier split as the attention section:
#
# * CPU-only asserts (the MLP activation guard, the layer_scalar buffer's type
#   and shape, the embedding normalizer value + per-dtype cache, the RoPE
#   ModuleDict keys, the softcap=30 plumbing and the tied lm_head identity) all
#   run unconditionally -- building the modules allocates CPU tensors and reads
#   back Python/attribute state, no kernel runs.
# * The forward-exercising layer_scalar check needs the fused RMSNorm/RoPE/GEMM
#   kernels inside Gemma4DecoderLayer.forward, which are CUDA-only, so it is
#   gated behind the shared ``_cuda_only`` skip (a CUDA host runs it; a
#   CPU-only laptop skips it).
#
# The full-model asserts build a TINY Gemma4ForConditionalGeneration /
# Gemma4Model (small vocab/hidden/intermediate, 2 layers) so BaseCausalLM's
# embedding + lm_head + LogitsProcessor are the real objects without paying the
# 60-layer / 262144-vocab cost. The geometry still carries both layer types so
# the RoPE ModuleDict and per-layer-type wiring are exercised.

# A tiny-but-complete gemma-4 text config: small enough to build a full model
# on CPU in a unit test, with both layer types present (so the RoPE ModuleDict
# and per-layer-type geometry are real). ``final_logit_softcapping`` is the
# gemma-4 value (30.0) and ``tie_word_embeddings`` is on.
_TINY_HIDDEN_SIZE = 64
_TINY_INTERMEDIATE_SIZE = 128
_TINY_VOCAB_SIZE = 320
_TINY_NUM_LAYERS = 2
_TINY_NUM_ATTENTION_HEADS = 4
_TINY_SLIDING_HEAD_DIM = 16
_TINY_FULL_HEAD_DIM = 32
_TINY_SLIDING_KV_HEADS = 4
_TINY_FULL_KV_HEADS = 2
_TINY_MAX_POSITION = 64
_FINAL_LOGIT_SOFTCAPPING = 30.0
# layer 0 sliding, layer 1 full -> both cache groups / both RoPE keys present.
_TINY_LAYER_TYPES = [SLIDING_ATTENTION, FULL_ATTENTION]


def _tiny_text_config(
    hidden_activation: str,
    final_logit_softcapping: float | None,
    tie_word_embeddings: bool,
) -> SimpleNamespace:
    """A small-but-complete gemma-4 *text* sub-config for full-model tests.

    Carries every field Gemma4Model / Gemma4ForConditionalGeneration (and the
    BaseCausalLM it drives) reads, at tiny widths. Every value is passed
    explicitly by the caller rather than defaulted so a test states exactly the
    activation / softcap / tie it exercises.
    """
    return SimpleNamespace(
        hidden_size=_TINY_HIDDEN_SIZE,
        intermediate_size=_TINY_INTERMEDIATE_SIZE,
        vocab_size=_TINY_VOCAB_SIZE,
        num_hidden_layers=_TINY_NUM_LAYERS,
        num_attention_heads=_TINY_NUM_ATTENTION_HEADS,
        head_dim=_TINY_SLIDING_HEAD_DIM,
        global_head_dim=_TINY_FULL_HEAD_DIM,
        num_key_value_heads=_TINY_SLIDING_KV_HEADS,
        num_global_key_value_heads=_TINY_FULL_KV_HEADS,
        hidden_activation=hidden_activation,
        rms_norm_eps=_RMS_NORM_EPS,
        max_position_embeddings=_TINY_MAX_POSITION,
        sliding_window=_SLIDING_WINDOW,
        attention_bias=False,
        attn_logit_softcapping=None,
        final_logit_softcapping=final_logit_softcapping,
        tie_word_embeddings=tie_word_embeddings,
        layer_types=_TINY_LAYER_TYPES,
        rope_parameters={
            FULL_ATTENTION: {
                "rope_type": "proportional",
                "rope_theta": _FULL_ROPE_THETA,
                "partial_rotary_factor": _FULL_PARTIAL_ROTARY_FACTOR,
            },
            SLIDING_ATTENTION: {
                "rope_type": "default",
                "rope_theta": _SLIDING_ROPE_THETA,
            },
        },
    )


def _tiny_config(
    hidden_activation: str,
    final_logit_softcapping: float | None,
    tie_word_embeddings: bool,
) -> SimpleNamespace:
    """The multimodal gemma-4 config wrapping the tiny text sub-config."""
    return SimpleNamespace(
        text_config=_tiny_text_config(
            hidden_activation=hidden_activation,
            final_logit_softcapping=final_logit_softcapping,
            tie_word_embeddings=tie_word_embeddings,
        )
    )


# ---------------------------------------------------------------------------
# Gemma4MLP: GeGLU activation guard (CPU)
# ---------------------------------------------------------------------------


def test_mlp_rejects_a_non_gelu_tanh_activation() -> None:
    """Gemma 4 only supports ``gelu_pytorch_tanh``; any other activation raises
    at construction (mirrors gemma3's guard), before any kernel runs."""
    with pytest.raises(ValueError, match="gelu_pytorch_tanh"):
        Gemma4MLP(
            hidden_size=_TINY_HIDDEN_SIZE,
            intermediate_size=_TINY_INTERMEDIATE_SIZE,
            hidden_activation="silu",
            quant_config=None,
            tp_rank=0,
            tp_size=1,
            tp_group=None,
            prefix="model.layers.0.mlp",
        )


def test_mlp_accepts_gelu_pytorch_tanh() -> None:
    """The supported activation builds a GeGLU MLP with the fused gate|up and
    down projections (construction-only; no forward kernel)."""
    mlp = Gemma4MLP(
        hidden_size=_TINY_HIDDEN_SIZE,
        intermediate_size=_TINY_INTERMEDIATE_SIZE,
        hidden_activation="gelu_pytorch_tanh",
        quant_config=None,
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        prefix="model.layers.0.mlp",
    )
    assert hasattr(mlp, "gate_up_proj")
    assert hasattr(mlp, "down_proj")


# ---------------------------------------------------------------------------
# Gemma4DecoderLayer: layer_scalar buffer (CPU) + applied at layer end (CUDA)
# ---------------------------------------------------------------------------


def _build_decoder_layer(layer_id: int, device: str) -> Gemma4DecoderLayer:
    """Construct a Gemma4DecoderLayer for ``layer_id`` on ``device``.

    The shared RoPE for the layer's type is built under the device context (so
    its cos/sin cache lands on ``device``) and handed in, exactly as
    Gemma4Model wires it.
    """
    config = _attention_config()
    layer_type = _LAYER_TYPES_31B[layer_id]
    with torch.device(device):
        rotary_emb = _build_rope(config, layer_type, _ROPE_MAX_POSITION, torch.float32)
        return Gemma4DecoderLayer(
            config=config,
            mapping=_tp1_mapping(),
            layer_id=layer_id,
            rotary_emb=rotary_emb,
            quant_config=None,
            prefix=f"model.layers.{layer_id}",
        )


def test_layer_scalar_is_a_ones_buffer_of_shape_one_not_a_parameter() -> None:
    """``layer_scalar`` is a registered ``[1]`` buffer initialised to ones and
    is NOT a learnable Parameter (the weight-load task fills it per layer).

    Checked on CPU: it is pure module state, no kernel involved.
    """
    layer = _build_decoder_layer(_SLIDING_LAYER_ID, "cpu")
    buffers = dict(layer.named_buffers())
    assert "layer_scalar" in buffers
    scalar = buffers["layer_scalar"]
    assert scalar.shape == (1,)
    torch.testing.assert_close(scalar, torch.ones(1), rtol=0.0, atol=0.0)
    # It must be a buffer, not a Parameter: no entry in named_parameters().
    assert not any("layer_scalar" in name for name, _ in layer.named_parameters())
    assert not isinstance(layer.layer_scalar, torch.nn.Parameter)


@_cuda_only
def test_layer_scalar_scales_the_whole_layer_output() -> None:
    """Property: the per-layer ``layer_scalar`` multiplies the ENTIRE layer
    output at the very end of the block.

    Build one decoder layer, stub its attention with ``_CaptureAttn`` (a real
    PagedAttention needs a ForwardContext + bound KV pool a unit test does not
    stand up), give the projections non-trivial weights, then run the SAME
    layer twice: once with ``layer_scalar = 1.0`` and once with
    ``layer_scalar = 0.5``. Everything before the final scalar multiply is
    identical across the two runs, so the output must satisfy
    ``output(0.5) == 0.5 * output(1.0)`` elementwise.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = _build_decoder_layer(_SLIDING_LAYER_ID, "cuda")
    finally:
        torch.set_default_dtype(old_dtype)

    # ``get_rope`` caches sliding RoPE by build args (device not in the key), so
    # the cos/sin cache may come back on CPU; the fused RoPE kernel needs it on
    # the q/k device.
    attn = layer.self_attn
    attn.rotary_emb.cos_sin_cache = attn.rotary_emb.cos_sin_cache.cuda()

    # Stub attention so the layer forward does not touch the KV pool; the stub
    # returns a correctly-shaped [num_tokens, num_heads * head_dim] output so
    # o_proj runs normally.
    attn.attn = _CaptureAttn(attn.num_heads, attn.head_dim)

    # Non-trivial projection weights so the layer output is non-zero (a zero
    # output would make the scaling assertion vacuous).
    torch.manual_seed(0)
    with torch.no_grad():
        attn.qkv_proj.weight.normal_(mean=0.0, std=0.02)
        attn.o_proj.weight.normal_(mean=0.0, std=0.02)
        layer.mlp.gate_up_proj.weight.normal_(mean=0.0, std=0.02)
        layer.mlp.down_proj.weight.normal_(mean=0.0, std=0.02)

    num_tokens = 4
    hidden_states = torch.randn(
        num_tokens, _HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

    with torch.no_grad():
        layer.layer_scalar.fill_(1.0)
        out_one, residual_one = layer.forward(
            positions, hidden_states.clone(), None, None
        )
        layer.layer_scalar.fill_(0.5)
        out_half, residual_half = layer.forward(
            positions, hidden_states.clone(), None, None
        )

    # Gemma 4 carries no cross-layer residual stream.
    assert residual_one is None
    assert residual_half is None
    # The whole output scales by the buffer: output(0.5) == 0.5 * output(1.0).
    torch.testing.assert_close(out_half, 0.5 * out_one, rtol=1e-3, atol=1e-3)
    # Non-vacuous: the unscaled output is actually non-zero.
    assert out_one.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Gemma4Model: embedding normalizer (tensor-returning, per-dtype cache) + the
# per-layer-type RoPE ModuleDict (CPU)
# (Validates: Property 4)
# ---------------------------------------------------------------------------


def _build_tiny_model(device: str) -> Gemma4Model:
    """Construct a tiny Gemma4Model on ``device`` with finite embedding weights.

    The embedding is created with uninitialised storage; fill it with finite
    random values so the independent recompute in the normalizer test does not
    trip over NaNs/infs.
    """
    config = _tiny_config(
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=_FINAL_LOGIT_SOFTCAPPING,
        tie_word_embeddings=True,
    )
    with torch.device(device):
        model = Gemma4Model(
            config=config,
            mapping=_tp1_mapping(),
            quant_config=None,
            prefix="model",
        )
    torch.manual_seed(0)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(mean=0.0, std=0.02)
    return model


def test_get_input_embeddings_returns_a_scaled_tensor_not_a_module() -> None:
    """Property 4: ``get_input_embeddings`` returns a TENSOR equal to
    ``embed_tokens(ids) * sqrt(hidden_size)``.

    It must be a ``torch.Tensor`` (NOT an ``nn.Module``) -- the prefill-graph
    embedding fix depends on the scaled embedding flowing through the graph.
    The expected value is recomputed independently from the same embedding
    weights and the ``sqrt(hidden_size)`` scale, in float32 on CPU.
    """
    model = _build_tiny_model("cpu")
    input_ids = torch.tensor([0, 1, 5, 7, _TINY_VOCAB_SIZE - 1], dtype=torch.int64)

    embeds = model.get_input_embeddings(input_ids)
    assert isinstance(embeds, torch.Tensor)
    assert not isinstance(embeds, torch.nn.Module)
    assert embeds.shape == (input_ids.shape[0], _TINY_HIDDEN_SIZE)

    scale = _TINY_HIDDEN_SIZE**0.5
    expected = model.embed_tokens(input_ids) * scale
    torch.testing.assert_close(embeds, expected, rtol=1e-6, atol=1e-6)


def test_embedding_normalizer_is_cached_per_dtype() -> None:
    """Property 4: the ``sqrt(hidden_size)`` normalizer is cached per compute
    dtype as a Python float ~= sqrt(hidden_size).

    After a call the model's ``_normalizer_by_dtype`` holds the embedding's
    dtype key with a value close to ``sqrt(hidden_size)``; a second call does
    not change the cache (same key, same value).
    """
    model = _build_tiny_model("cpu")
    input_ids = torch.tensor([0, 1, 2], dtype=torch.int64)

    first = model.get_input_embeddings(input_ids)
    dtype = first.dtype
    assert dtype in model._normalizer_by_dtype
    expected_scale = _TINY_HIDDEN_SIZE**0.5
    assert model._normalizer_by_dtype[dtype] == pytest.approx(expected_scale, rel=1e-6)

    cache_after_first = dict(model._normalizer_by_dtype)
    model.get_input_embeddings(input_ids)
    assert model._normalizer_by_dtype == cache_after_first


def test_model_rope_moduledict_has_exactly_the_present_layer_type_keys() -> None:
    """The model builds exactly one RoPE per unique layer type present.

    The tiny layout carries one sliding and one full layer, so the RoPE
    ModuleDict has exactly the ``{sliding_attention, full_attention}`` keys --
    one shared cos/sin cache per layer type, no duplicate-per-layer instances.
    """
    model = _build_tiny_model("cpu")
    assert set(model.rotary_emb.keys()) == {SLIDING_ATTENTION, FULL_ATTENTION}
    # Exactly the two present types, no extras.
    assert len(model.rotary_emb) == 2


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration: final logit softcap=30, tied lm_head, and
# the attn!=dense tp_size guard
# (Validates: Property 6)
# ---------------------------------------------------------------------------


def _build_tiny_cond_gen() -> Gemma4ForConditionalGeneration:
    """Construct a tiny Gemma4ForConditionalGeneration on CPU.

    Drives BaseCausalLM off the tiny text sub-config, so the real
    LogitsProcessor and the tied lm_head are built (single-rank mapping ->
    last pp rank -> both are allocated).
    """
    config = _tiny_config(
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=_FINAL_LOGIT_SOFTCAPPING,
        tie_word_embeddings=True,
    )
    with torch.device("cpu"):
        return Gemma4ForConditionalGeneration(
            config=config,
            mapping=_tp1_mapping(),
            quant_config=None,
        )


def test_final_logit_softcapping_is_thirty() -> None:
    """Property 6: the LogitsProcessor applies ``final_logit_softcapping=30.0``.

    Gemma 4 drives BaseCausalLM off its text sub-config, which carries
    ``final_logit_softcapping=30.0``; the LogitsProcessor reads it off that
    config, so the processor's ``final_logit_softcapping`` is exactly 30.0.
    """
    model = _build_tiny_cond_gen()
    assert model.logits_processor.final_logit_softcapping == _FINAL_LOGIT_SOFTCAPPING


def test_lm_head_is_tied_to_the_token_embedding() -> None:
    """The tied head: ``lm_head.weight is model.embed_tokens.weight``.

    ``tie_word_embeddings=True`` makes BaseCausalLM reuse the token embedding as
    the LM head rather than allocating a separate one, so the two share the
    exact same weight tensor (identity, not just equality).
    """
    model = _build_tiny_cond_gen()
    assert model.lm_head is model.model.embed_tokens
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_mismatched_attn_dense_tp_size_is_rejected() -> None:
    """The attn/dense TP guard: the hand-written sandwich layer keeps attention
    and MLP on one residual stream, so a split attn/dense TP would need a shard
    exchange this layer does not perform; construction raises instead.

    A single-process mapping with ``decode_dense_tp_size`` forced to differ from
    the attention TP size trips the guard without standing up real distributed
    process groups.
    """
    config = _tiny_config(
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=_FINAL_LOGIT_SOFTCAPPING,
        tie_word_embeddings=True,
    )
    mapping = _tp1_mapping()
    # Force the attn/dense TP sizes to disagree so the guard fires. The guard
    # reads mapping.attn.tp_size vs mapping.dense.tp_size directly.
    object.__setattr__(mapping.dense, "tp_size", mapping.attn.tp_size + 1)
    with pytest.raises(ValueError, match="attn.tp_size == dense.tp_size"):
        Gemma4ForConditionalGeneration(
            config=config,
            mapping=mapping,
            quant_config=None,
        )


# ---------------------------------------------------------------------------
# Weight-load coverage: name-mapping dry-run against the REAL gemma-4-31B-it
# checkpoint tensor-name index (Validates: Property 5)
# ---------------------------------------------------------------------------
#
# Task 6.4. ``Gemma4ForConditionalGeneration.load_weights`` applies a strict
# coverage check (task 6.3): every expected parameter/buffer written exactly
# once and every claimed checkpoint tensor consumed, else a ValueError. This
# section drives that check as a DRY RUN against the REAL checkpoint's
# tensor-name index -- so the skip set (vision_tower / embed_vision), the
# ``model.language_model.* -> model.*`` remap, the q/k/v -> qkv_proj and
# gate/up -> gate_up_proj fusing, the full-layer K->V duplication
# (attention_k_eq_v) and the per-layer ``layer_scalar`` buffer load together
# consume EXACTLY the checkpoint's text tensors and fill EXACTLY the model's
# expected params + layer_scalar buffers, with nothing missing and nothing
# unclaimed.
#
# Cost control:
#
# * The 60-layer / 262144-vocab model is built on the META device, so no GPU
#   and no real weight memory is allocated -- the parameters are symbolic
#   shapes only (the coverage check reads names/shapes, never values).
# * ``load_weights`` is driven with a stub iterator yielding ``(name, tensor)``
#   pairs whose NAMES are the real checkpoint ``weight_map`` keys and whose
#   tensors are correctly-shaped META tensors (shapes derived from the config
#   geometry via :func:`gemma4_layer_config`). The byte copy is irrelevant to
#   coverage; the real fused/direct weight loaders run (``narrow`` + ``copy_``
#   are meta-safe), so the shape asserts inside them are exercised for real.
# * The ONE exception is the ``[1]`` ``layer_scalar`` buffer: it loads through
#   ``default_weight_loader``, which "broadcasts" a scalar via
#   ``loaded_weight.item()`` -- and ``.item()`` is rejected on a meta tensor.
#   So the stub yields a tiny REAL CPU ``[1]`` tensor for each ``layer_scalar``
#   (matching how the 6.3 verification handled it); the param side
#   (``param.data.fill_(...)``) is meta-safe.
#
# The real index lives ONLY on a host that has the checkpoint; the test reads
# it from ``_GEMMA4_31B_INDEX`` and is skipped where the checkpoint is absent
# (so the suite still passes on a laptop) -- it WILL run where the weights
# exist.
#
# The checkpoint directory is taken from the ``GEMMA4_31B_DIR`` environment
# variable, defaulting to the Hugging Face repo id ``google/gemma-4-31B-it``;
# point it at a local snapshot to exercise the coverage tests.

# The real gemma-4-31B-it checkpoint index. Present only where the checkpoint
# is downloaded; the coverage tests below skip when it is absent.
_GEMMA4_31B_DIR = os.environ.get("GEMMA4_31B_DIR", "google/gemma-4-31B-it")
_GEMMA4_31B_INDEX = os.path.join(_GEMMA4_31B_DIR, "model.safetensors.index.json")

# gemma-4-31B model-level geometry needed to build the full 60-layer model (on
# top of the per-layer-type attention/MLP fields in ``_attention_text_config``).
_VOCAB_SIZE_31B = 262144
_MAX_POSITION_EMBEDDINGS_31B = 262144

# Expected checkpoint tensor counts (confirmed by reading the real index). The
# full layers (every 6th: indices 5, 11, ..., 59 -> 10 layers) ship a k_proj
# but NO v_proj (attention_k_eq_v); sliding layers (50) ship a real v_proj.
_EXPECTED_NUM_V_PROJ = 50  # sliding layers only
_EXPECTED_NUM_FULL_LAYERS = 10
_EXPECTED_NUM_SLIDING_LAYERS = 50
_EXPECTED_NUM_LAYER_SCALAR = 60

_requires_gemma4_31b_index = pytest.mark.skipif(
    not os.path.exists(_GEMMA4_31B_INDEX),
    reason=(
        "the real gemma-4-31B-it checkpoint index is required for the "
        f"weight-load coverage dry run ({_GEMMA4_31B_INDEX}); present where the "
        "checkpoint is downloaded, absent on a laptop."
    ),
)


def _full_text_config_31b() -> SimpleNamespace:
    """The real gemma-4-31B *text* sub-config, at full depth/width.

    Extends ``_attention_text_config`` (which already carries the per-layer-type
    head_dim / kv-head split, 32 query heads, hidden_size 5376,
    intermediate_size 21504, the 60-entry ``_LAYER_TYPES_31B`` and the nested
    RoPE params) with the model-level fields ``Gemma4Model`` /
    ``Gemma4ForConditionalGeneration`` / ``BaseCausalLM`` read: the 60-layer
    depth, the 262144 vocab / max position, the tied head and the reinstated
    ``final_logit_softcapping=30.0``.
    """
    text_config = _attention_text_config()
    text_config.num_hidden_layers = _NUM_LAYERS_31B
    text_config.vocab_size = _VOCAB_SIZE_31B
    text_config.max_position_embeddings = _MAX_POSITION_EMBEDDINGS_31B
    text_config.final_logit_softcapping = _FINAL_LOGIT_SOFTCAPPING
    text_config.tie_word_embeddings = True
    return text_config


def _full_config_31b() -> SimpleNamespace:
    """The multimodal gemma-4-31B config wrapping the full text sub-config."""
    return SimpleNamespace(text_config=_full_text_config_31b())


def _build_meta_cond_gen_31b() -> Gemma4ForConditionalGeneration:
    """Build the full 60-layer ``Gemma4ForConditionalGeneration`` on META.

    Meta construction allocates only symbolic parameter shapes -- no GPU, no
    weight memory, no 262144 x 5376 embedding materialised -- which is all the
    coverage check needs (it reads names and shapes). A single-rank mapping
    keeps TP out of the picture, so each fused shard is full-width and the stub
    tensor shapes below are the un-sharded checkpoint shapes.
    """
    config = _full_config_31b()
    with torch.device("meta"):
        return Gemma4ForConditionalGeneration(
            config=config,
            mapping=_tp1_mapping(),
            quant_config=None,
        )


def _load_index_weight_map(index_path: str) -> dict[str, str]:
    """Return the checkpoint index ``weight_map`` (tensor name -> shard file)."""
    with open(index_path) as handle:
        index = json.load(handle)
    return index["weight_map"]


def _checkpoint_text_shape_31b(name: str) -> tuple[int, ...]:
    """The shape of a ``model.language_model.*`` checkpoint tensor on 31B.

    Derived from the config geometry (NOT read from the safetensors headers),
    so the stub tensors match what ``load_weights`` fuses: q/k/v and gate/up are
    the per-shard checkpoint shapes the fused loaders narrow into, and the
    per-layer-type head_dim / kv-head split is resolved via
    :func:`gemma4_layer_config` exactly as the model does.

    Args:
        name: A ``model.language_model.*`` tensor name from the real index.

    Returns:
        The tensor's shape as a tuple.
    """
    text_config = _full_text_config_31b()
    hidden_size = int(text_config.hidden_size)
    intermediate_size = int(text_config.intermediate_size)
    num_heads = int(text_config.num_attention_heads)

    if name == "model.language_model.embed_tokens.weight":
        return (int(text_config.vocab_size), hidden_size)
    if name == "model.language_model.norm.weight":
        return (hidden_size,)

    match = re.search(r"\blayers\.(\d+)\.", name)
    assert match is not None, name
    layer_idx = int(match.group(1))
    layer_geometry = gemma4_layer_config(text_config, layer_idx)
    head_dim = int(layer_geometry.head_dim)
    kv_heads = int(layer_geometry.num_key_value_heads)

    if name.endswith("layer_scalar"):
        return (1,)
    if name.endswith("self_attn.q_proj.weight"):
        return (num_heads * head_dim, hidden_size)
    if name.endswith("self_attn.k_proj.weight") or name.endswith(
        "self_attn.v_proj.weight"
    ):
        return (kv_heads * head_dim, hidden_size)
    if name.endswith("self_attn.o_proj.weight"):
        return (hidden_size, num_heads * head_dim)
    if name.endswith("self_attn.q_norm.weight") or name.endswith(
        "self_attn.k_norm.weight"
    ):
        return (head_dim,)
    if name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
        return (intermediate_size, hidden_size)
    if name.endswith("mlp.down_proj.weight"):
        return (hidden_size, intermediate_size)
    # The four sandwich norms (input / post_attention / pre_feedforward /
    # post_feedforward layernorm) are all [hidden_size].
    if name.endswith("layernorm.weight"):
        return (hidden_size,)
    raise AssertionError(f"unhandled checkpoint tensor name: {name}")


def _stub_weight_for(name: str) -> torch.Tensor:
    """A correctly-shaped stub tensor for a checkpoint tensor name.

    Everything is a META tensor (zero real memory) EXCEPT the ``[1]``
    ``layer_scalar`` buffers, which load through ``default_weight_loader`` ->
    ``loaded_weight.item()`` -- and ``.item()`` is rejected on a meta tensor --
    so those get a tiny REAL CPU tensor (the only values that ever leave the
    symbolic world).
    """
    shape = _checkpoint_text_shape_31b(name)
    if name.endswith("layer_scalar"):
        return torch.ones(shape, dtype=torch.float32, device="cpu")
    return torch.empty(shape, dtype=torch.float32, device="meta")


def _stub_checkpoint_weights(
    weight_map: dict[str, str],
) -> list[tuple[str, torch.Tensor]]:
    """Stub ``(name, tensor)`` pairs for EVERY tensor in the checkpoint index.

    The names are the real ``weight_map`` keys -- including the vision_tower /
    embed_vision tensors that ``load_weights`` must SKIP -- so the dry run
    exercises the skip set as well as the text remap/fuse. Only the
    ``model.language_model.*`` tensors get a config-derived shape; the
    vision-side tensors are skipped by name before any shape is read, so a
    zero-element meta placeholder is enough for them.
    """
    stub: list[tuple[str, torch.Tensor]] = []
    for name in weight_map:
        if name.startswith("model.language_model."):
            stub.append((name, _stub_weight_for(name)))
        else:
            # Vision-side tensors are skipped by name; shape is never read.
            stub.append((name, torch.empty(0, dtype=torch.float32, device="meta")))
    return stub


@_requires_gemma4_31b_index
def test_load_weights_dry_run_covers_the_real_checkpoint_index() -> None:
    """Property 5: the real gemma-4-31B-it index maps with NO missing / NO
    unclaimed.

    Build the full 60-layer model on meta, then drive ``load_weights`` with
    stub ``(name, meta-tensor)`` pairs whose names are the real checkpoint
    ``weight_map`` keys. The strict coverage check (task 6.3) must pass: every
    model parameter + every ``layer_scalar`` buffer is written exactly once and
    every text checkpoint tensor is consumed, so ``load_weights`` returns
    WITHOUT raising.
    """
    weight_map = _load_index_weight_map(_GEMMA4_31B_INDEX)
    model = _build_meta_cond_gen_31b()
    weights = _stub_checkpoint_weights(weight_map)

    # The whole point of the dry run: strict coverage is clean, so this does
    # NOT raise. A missing or unclaimed tensor would raise ValueError here.
    model.load_weights(weights)


@_requires_gemma4_31b_index
def test_load_weights_dry_run_positive_coverage_counts() -> None:
    """Property 5 (diagnosable form): the real index decomposes into exactly the
    expected text tensors -- 50 sliding v_proj, 10 full layers without a v_proj,
    60 layer_scalar -- so a regression in the split is pinpointable, not just a
    pass/fail.

    These counts are asserted on the index itself (independent of the model) so
    the clean-coverage result above is explained by the checkpoint actually
    having the gemma-4 k_eq_v shape, not by a lucky cancellation.
    """
    weight_map = _load_index_weight_map(_GEMMA4_31B_INDEX)
    lm_names = [n for n in weight_map if n.startswith("model.language_model.")]

    def _count(suffix: str) -> int:
        return sum(1 for n in lm_names if n.endswith(suffix))

    # q/k present on every layer; v only on the 50 sliding layers.
    assert _count("self_attn.q_proj.weight") == _NUM_LAYERS_31B
    assert _count("self_attn.k_proj.weight") == _NUM_LAYERS_31B
    assert _count("self_attn.v_proj.weight") == _EXPECTED_NUM_V_PROJ
    # gate/up/down and o_proj on every layer.
    assert _count("mlp.gate_proj.weight") == _NUM_LAYERS_31B
    assert _count("mlp.up_proj.weight") == _NUM_LAYERS_31B
    assert _count("mlp.down_proj.weight") == _NUM_LAYERS_31B
    assert _count("self_attn.o_proj.weight") == _NUM_LAYERS_31B
    # q/k norms on every layer; the checkpoint ships NO v_norm (the module's is
    # weightless), matching the design.
    assert _count("self_attn.q_norm.weight") == _NUM_LAYERS_31B
    assert _count("self_attn.k_norm.weight") == _NUM_LAYERS_31B
    assert sum(1 for n in lm_names if "self_attn.v_norm" in n) == 0
    # One layer_scalar per layer.
    assert _count("layer_scalar") == _EXPECTED_NUM_LAYER_SCALAR

    # The full layers (k_eq_v) are exactly those WITHOUT a v_proj: 10 of them,
    # at the 5:1 indices. So the fused V shard is filled 50 (real) + 10
    # (duplicated from k_proj) = 60 times across the 60 layers.
    layers_with_v = {
        int(re.search(r"\blayers\.(\d+)\.", n).group(1))
        for n in lm_names
        if n.endswith("self_attn.v_proj.weight")
    }
    all_layers = set(range(_NUM_LAYERS_31B))
    full_layers = sorted(all_layers - layers_with_v)
    assert full_layers == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    assert len(full_layers) == _EXPECTED_NUM_FULL_LAYERS
    assert len(layers_with_v) == _EXPECTED_NUM_SLIDING_LAYERS


@_requires_gemma4_31b_index
def test_load_weights_dry_run_raises_on_a_dropped_layer_scalar() -> None:
    """Property 5 (negative): a checkpoint that DROPS a text tensor is caught.

    Remove one ``layer_scalar`` from the stubbed index. That buffer is in the
    model's ``expected`` set but is now never written, so the strict coverage
    check must raise a ValueError naming the mismatch -- the whole reason the
    check exists (a silently short checkpoint would otherwise mis-serve).
    """
    weight_map = _load_index_weight_map(_GEMMA4_31B_INDEX)
    model = _build_meta_cond_gen_31b()
    weights = [
        (name, tensor)
        for name, tensor in _stub_checkpoint_weights(weight_map)
        if name != "model.language_model.layers.0.layer_scalar"
    ]

    with pytest.raises(ValueError, match="did not match the model"):
        model.load_weights(weights)


@_requires_gemma4_31b_index
def test_load_weights_dry_run_raises_on_an_unclaimed_text_tensor() -> None:
    """Property 5 (negative): a RENAMED text tensor is caught as unclaimed.

    Rename one real text tensor to a name no parameter claims (while leaving
    the real one out). The renamed tensor is not skipped (it is under
    ``model.language_model.``) and resolves to no param, so it lands in the
    ``unclaimed`` set AND its target param is now never written -- either way
    the strict coverage check raises.
    """
    weight_map = _load_index_weight_map(_GEMMA4_31B_INDEX)
    model = _build_meta_cond_gen_31b()
    real_name = "model.language_model.layers.0.input_layernorm.weight"
    bogus_name = "model.language_model.layers.0.input_layernorm.renamed_weight"
    weights = []
    for name, tensor in _stub_checkpoint_weights(weight_map):
        if name == real_name:
            # Keep the same (meta) shape but under a name no param claims.
            weights.append((bogus_name, tensor))
        else:
            weights.append((name, tensor))

    with pytest.raises(ValueError, match="did not match the model"):
        model.load_weights(weights)


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration: unsupported-feature guard (Task 7.2)
# (Validates: design Error Handling)
# ---------------------------------------------------------------------------
#
# This text port implements ONLY the dense, non-shared, non-MoE, no-PLE path --
# the gemma-4-31B-it checkpoint has MoE, the double-wide MLP, per-layer input
# embeddings (PLE) and KV-sharing all disabled. The vllm-unieai reference
# carries branches for each; this port dropped them. So a DIFFERENT gemma-4
# config that ENABLES any of those must raise a clear, feature-naming error at
# construction rather than silently run the dense path and mis-serve.
#
# The real/default dense config must still build (the guard is a clean no-op).
# Each check is construction-only (CPU): the guard runs before any kernel.


def _tiny_cond_gen_config_with(**overrides: object) -> SimpleNamespace:
    """A tiny dense gemma-4 config with the given text-config fields overridden.

    Starts from the same tiny-but-complete config the other construction tests
    build (dense gelu MLP, softcap 30, tied head) and sets each override on the
    TEXT sub-config, so a test enables exactly one unsupported feature in
    isolation on top of an otherwise-valid dense config.
    """
    config = _tiny_config(
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=_FINAL_LOGIT_SOFTCAPPING,
        tie_word_embeddings=True,
    )
    for field, value in overrides.items():
        setattr(config.text_config, field, value)
    return config


def _build_tiny_cond_gen_with(**overrides: object) -> Gemma4ForConditionalGeneration:
    """Construct a tiny Gemma4ForConditionalGeneration with text-config overrides.

    CPU construction, single-rank mapping -- the same path
    :func:`_build_tiny_cond_gen` uses, so the feature guard (and the rest of
    __init__) runs exactly as in the real build.
    """
    config = _tiny_cond_gen_config_with(**overrides)
    with torch.device("cpu"):
        return Gemma4ForConditionalGeneration(
            config=config,
            mapping=_tp1_mapping(),
            quant_config=None,
        )


def test_dense_config_constructs_without_the_feature_guard_firing() -> None:
    """The real/default dense config builds: the guard is a clean no-op.

    The gemma-4-31B-it checkpoint has MoE / double-wide MLP / PLE / KV-sharing
    all disabled (absent / zero / false), so a plain dense config must
    construct without the guard raising -- otherwise the guard would block the
    very checkpoint this port serves.
    """
    model = _build_tiny_cond_gen()
    # Constructed successfully and is the dense port (no MoE / PLE state).
    assert isinstance(model, Gemma4ForConditionalGeneration)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_moe_block": True}, "MoE"),
        ({"use_second_mlp_block": True}, "MoE"),
        ({"num_experts": 8}, "MoE"),
        ({"top_k_experts": 2}, "MoE"),
        ({"use_double_wide_mlp": True}, "double-wide MLP"),
        ({"hidden_size_per_layer_input": 256}, "per-layer input embeddings"),
        ({"num_kv_shared_layers": 4}, "KV-sharing"),
    ],
)
def test_enabled_unsupported_feature_raises_naming_that_feature(
    overrides: dict[str, object], match: str
) -> None:
    """Each dropped feature, enabled in isolation, makes construction raise.

    The error message names the specific unsupported feature (MoE / double-wide
    MLP / per-layer input embeddings / KV-sharing) so the refusal points at
    exactly what is unsupported, rather than silently running the dense path on
    a config that expects more.
    """
    with pytest.raises(ValueError, match=match):
        _build_tiny_cond_gen_with(**overrides)


def test_zero_or_falsey_feature_fields_do_not_trip_the_guard() -> None:
    """Explicitly-disabled feature fields (0 / False) build cleanly.

    A config may carry the feature fields set to their disabled values (as the
    real checkpoint config does: enable_moe_block=False, num_experts=0,
    use_double_wide_mlp=False, hidden_size_per_layer_input=0,
    num_kv_shared_layers=0). Those falsey values must NOT trip the guard.
    """
    model = _build_tiny_cond_gen_with(
        enable_moe_block=False,
        use_second_mlp_block=False,
        num_experts=0,
        top_k_experts=0,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    assert isinstance(model, Gemma4ForConditionalGeneration)


# ---------------------------------------------------------------------------
# Registered architectures: the model registry maps BOTH gemma-4 arch strings
# to this implementation, the config layer knows them as gemma-4, and the real
# checkpoint's architecture resolves end-to-end.
# (Validates: design Architecture / registration; task 7.1 committed)
# ---------------------------------------------------------------------------
#
# gemma4.py exports ``EntryClass = [Gemma4ForConditionalGeneration,
# Gemma4ForCausalLM]``; ``import_model_classes()`` discovers it so both strings
# resolve through ``ModelRegistry``. Task 7.1 only confirmed this ad-hoc -- these
# are the committed regression. All CPU / no checkpoint except the last, which
# reads the real config.json and is skipped where the checkpoint is absent.

# The real gemma-4-31B-it checkpoint config. Present only where the checkpoint
# is downloaded; the end-to-end resolution test below skips when it is absent.
_GEMMA4_31B_CONFIG = os.path.join(_GEMMA4_31B_DIR, "config.json")

_requires_gemma4_31b_config = pytest.mark.skipif(
    not os.path.exists(_GEMMA4_31B_CONFIG),
    reason=(
        "the real gemma-4-31B-it checkpoint config is required to confirm the "
        f"shipped architecture resolves ({_GEMMA4_31B_CONFIG}); present where "
        "the checkpoint is downloaded, absent on a laptop."
    ),
)


def test_conditional_generation_arch_resolves_to_this_implementation() -> None:
    """``Gemma4ForConditionalGeneration`` (the multimodal checkpoint's arch)
    resolves through the registry to this module's class.

    This is the string the gemma-4-31B-it ``config.json`` ships
    (``architectures=["Gemma4ForConditionalGeneration"]``); the registry must
    hand back exactly this port's class so the loader builds the text decoder.
    """
    from tokenspeed.runtime.models.registry import ModelRegistry

    cls, arch = ModelRegistry.resolve_model_cls(["Gemma4ForConditionalGeneration"])
    assert cls is Gemma4ForConditionalGeneration
    assert arch == "Gemma4ForConditionalGeneration"


def test_causal_lm_arch_resolves_to_this_implementation() -> None:
    """``Gemma4ForCausalLM`` (the text-only release's arch) resolves through the
    registry to this module's text-only subclass.

    The second ``EntryClass`` entry: a bare text-only gemma-4 checkpoint names
    this architecture, and it maps to ``Gemma4ForCausalLM`` (the subclass of the
    conditional-generation port).
    """
    from tokenspeed.runtime.models.registry import ModelRegistry

    cls, arch = ModelRegistry.resolve_model_cls(["Gemma4ForCausalLM"])
    assert cls is Gemma4ForCausalLM
    assert arch == "Gemma4ForCausalLM"


def test_entry_class_exports_exactly_the_two_gemma4_archs() -> None:
    """The module's ``EntryClass`` is exactly the two gemma-4 architectures, in
    the order ``[conditional-generation, causal-LM]``.

    ``import_model_classes()`` iterates ``EntryClass`` to populate the registry,
    so this is the single source of truth for which strings gemma-4 claims.
    """
    import tokenspeed.runtime.models.gemma4 as gemma4_module

    assert gemma4_module.EntryClass == [
        Gemma4ForConditionalGeneration,
        Gemma4ForCausalLM,
    ]


def test_both_arch_strings_are_in_the_config_gemma4_architecture_set() -> None:
    """The config layer recognises both arch strings as gemma-4.

    ``_GEMMA4_ARCHITECTURES`` is what the cache-family / layer-type dispatch
    keys on to resolve the per-layer-type KV geometry (the gemma-4-only split);
    both EntryClass strings must be members or a checkpoint naming either would
    silently take the uniform-geometry path.
    """
    from tokenspeed.runtime.configs import model_config

    assert "Gemma4ForConditionalGeneration" in model_config._GEMMA4_ARCHITECTURES
    assert "Gemma4ForCausalLM" in model_config._GEMMA4_ARCHITECTURES


def test_is_gemma4_detects_by_architecture_string() -> None:
    """``is_gemma4`` is True for a config carrying a gemma-4 architecture
    string even when the model_type is unset.

    The architecture list is the primary signal the cache path reads to switch
    on the per-layer-type geometry.
    """
    from tokenspeed.runtime.configs import model_config

    # A bare text config (no ``text_config`` sub-attribute) carrying the arch
    # string. ``model_type`` is unset so detection must come from the arch list.
    config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="",
    )
    assert model_config.is_gemma4(config) is True


def test_is_gemma4_detects_by_text_model_type_when_arch_is_lost() -> None:
    """``is_gemma4`` is True by ``model_type`` alone when the arch list is lost.

    A multimodal wrapper can drop its ``architectures`` to None; the
    ``gemma4_text`` model_type on the text sub-config must still identify it, so
    the per-layer-type geometry is not silently skipped.
    """
    from tokenspeed.runtime.configs import model_config

    # A multimodal wrapper that lost its arch list; the text sub-config carries
    # the gemma4_text model_type. The text sub-config must expose
    # ``num_attention_heads`` because ``get_hf_text_config`` reads it when a
    # ``text_config`` is present.
    config = SimpleNamespace(
        architectures=None,
        model_type="gemma4",
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            num_attention_heads=_NUM_ATTENTION_HEADS,
        ),
    )
    assert model_config.is_gemma4(config) is True


def test_is_gemma4_is_false_for_a_non_gemma4_config() -> None:
    """``is_gemma4`` is False for an unrelated architecture / model_type, so the
    per-layer-type geometry switch does not fire for other models."""
    from tokenspeed.runtime.configs import model_config

    # A bare (non-gemma-4) text config: a llama arch string and model_type, no
    # ``text_config`` sub-attribute.
    config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
    )
    assert model_config.is_gemma4(config) is False


@_requires_gemma4_31b_config
def test_real_checkpoint_architecture_resolves_end_to_end() -> None:
    """The shipped gemma-4-31B-it ``config.json`` architecture resolves to this
    port AND is recognised as gemma-4 by the config layer.

    Reads the real ``architectures`` list off the checkpoint config (rather than
    hard-coding it), resolves it through the registry to this port's class, and
    confirms ``is_gemma4`` fires on a config carrying that same architecture --
    the end-to-end path from checkpoint string to text-decoder class.
    """
    from tokenspeed.runtime.configs import model_config
    from tokenspeed.runtime.models.registry import ModelRegistry

    with open(_GEMMA4_31B_CONFIG) as handle:
        shipped = json.load(handle)
    architectures = shipped["architectures"]
    assert architectures, "the checkpoint config must name an architecture"

    cls, arch = ModelRegistry.resolve_model_cls(architectures)
    assert cls in (Gemma4ForConditionalGeneration, Gemma4ForCausalLM)
    assert arch in architectures

    # Confirm the config layer also keys this arch as gemma-4. A bare config
    # carrying the shipped arch list (no ``text_config`` sub-attribute, so
    # ``get_hf_text_config`` does not demand a text sub-config) is enough: the
    # arch string alone must drive ``is_gemma4`` to the per-layer-type path.
    config = SimpleNamespace(
        architectures=architectures,
        model_type=shipped.get("model_type", ""),
    )
    assert model_config.is_gemma4(config) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
