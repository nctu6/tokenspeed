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

"""Model configuration helpers and derived runtime metadata."""

import copy
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum, auto

import torch
import yaml
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DEEPSEEK_V4_PAGE_SIZE,
)
from tokenspeed.runtime.layers.pooler import PoolingType
from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.plugins import ensure_loaded
from tokenspeed.runtime.plugins.registry import resolve_model_profile
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_generation_config,
    model_loader_architectures,
    resolve_architecture,
)
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.spec_block_geometry import (
    BLOCK_SPEC_ALGORITHMS,
    read_checkpoint_block_size,
    resolve_block_widths,
    validate_block_widths,
)

logger = get_colorful_logger(__name__)

_DEEPSEEK_V4_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForCausalLMDSpark",
        "DeepseekV4ForCausalLMNextN",
    }
)
_QWEN4_EXP_ARCHITECTURES = frozenset(
    {
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForCausalLMNextN",
    }
)
_MLA_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
        "Eagle3DeepseekV2ForCausalLM",
        "LongcatFlashForCausalLM",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "KimiK3ForConditionalGenerationNextN",
        # The K3 DSpark draft is MLA-native (DeepSeek-V3 layout, RoPE + YaRN),
        # so it must resolve to the MLA family rather than defaulting to MHA.
        "K3DSparkModel",
    }
)
_DSA_ARCHITECTURES = frozenset(
    {
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaForCausalLMNextN",
        # DeepSeek-V3.2: V3 MLA/MoE backbone + DSA sparse indexer. Same
        # attention family and indexer geometry as GLM-DSA.
        "DeepseekV32ForCausalLM",
        "DeepseekV32ForCausalLMNextN",
        "Glm53FlashForConditionalGeneration",
        "Glm53FlashForConditionalGenerationNextN",
    }
)
_MSA_ARCHITECTURES = frozenset(
    {
        "MiniMaxM3SparseForConditionalGeneration",
    }
)
_DOUBLE_ATTENTION_LAYER_ARCHITECTURES = frozenset(
    {
        "LongcatFlashForCausalLM",
    }
)
# Architectures whose config numbers only the cache-owning blocks as layers.
_CACHE_LAYER_VIEW_ARCHITECTURES = frozenset(
    {
        "NemotronHForCausalLM",
    }
)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()
    DSA = auto()
    MSA = auto()


@dataclass(frozen=True)
class _AttentionFamilySpec:
    name: str
    architectures: frozenset[str]
    configure: Callable[[object, ServerArgs], None]
    default_backend: str | None = None
    default_prefix_granularity: int | None = None


def override_model_config(model_config, ext_yaml):
    with open(ext_yaml, encoding="utf-8") as f:
        ext_config = yaml.safe_load(f)

    override_model_config: dict = ext_config.get("override_model_config", {})
    for k, v in override_model_config.items():
        if hasattr(model_config, k):
            old_v = model_config.__getattribute__(k)
            if isinstance(v, dict):
                new_v = copy.deepcopy(old_v)
                new_v.__dict__.update(v)
            else:
                new_v = v
            model_config.__setattr__(k, new_v)
            logger.info(f"Override model config: {k!s}={new_v!r}")


def is_deepseek_v4(config: PretrainedConfig) -> bool:
    return resolve_architecture(config) in _DEEPSEEK_V4_ARCHITECTURES


def is_qwen4_exp(config: PretrainedConfig) -> bool:
    return (
        getattr(config, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}
        or resolve_architecture(config) in _QWEN4_EXP_ARCHITECTURES
    )


def is_deepseek_v4_nextn(config: PretrainedConfig) -> bool:
    return resolve_architecture(config) == "DeepseekV4ForCausalLMNextN"


def configure_deepseek_v4_attention(model_config, server_args: ServerArgs) -> None:
    """Derive DeepSeek V4's MLA-like dimensions for runtime setup."""
    del server_args  # the geometry follows the checkpoint alone

    hf_config = model_config.hf_config
    model_config.head_dim = hf_config.head_dim
    model_config.attention_arch = AttentionArch.MLA
    model_config.kv_lora_rank = hf_config.head_dim
    model_config.qk_rope_head_dim = hf_config.qk_rope_head_dim
    model_config.qk_nope_head_dim = hf_config.head_dim - hf_config.qk_rope_head_dim
    model_config.v_head_dim = hf_config.head_dim
    model_config.index_head_dim = getattr(hf_config, "index_head_dim", None)
    model_config.scaling = 1 / math.sqrt(model_config.head_dim)
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


def configure_deepseek_v41_attention(model_config, server_args: ServerArgs) -> None:
    """V4.1 latent dimensions; YaRN changes RoPE, not the attention scale."""
    del server_args  # the geometry follows the checkpoint alone
    hf = model_config.hf_text_config
    model_config.head_dim = hf.head_dim
    model_config.attention_arch = AttentionArch.MLA
    model_config.kv_lora_rank = hf.head_dim
    model_config.qk_rope_head_dim = hf.qk_rope_head_dim
    model_config.qk_nope_head_dim = hf.head_dim - hf.qk_rope_head_dim
    model_config.v_head_dim = hf.head_dim
    model_config.index_head_dim = hf.index_head_dim
    model_config.scaling = hf.head_dim**-0.5


def configure_dsa_attention(model_config, server_args: ServerArgs) -> None:
    """Derive MLA latent plus DSA indexer geometry (GLM-DSA, DeepSeek-V3.2).

    Every attention hook takes the resolved launch so a hook can key a choice
    on it (``ModelProfile.configure_attention``); the in-tree ones plan the
    same geometry under every launch.
    """
    del server_args
    mla_config = (
        model_config.hf_text_config
        if hasattr(model_config.hf_text_config, "kv_lora_rank")
        else model_config.hf_config
    )
    required_fields = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_topk",
        "index_head_dim",
        "index_n_heads",
    )
    missing_fields = [
        field for field in required_fields if not hasattr(mla_config, field)
    ]
    if missing_fields:
        raise ValueError(
            "DSA attention config is missing required fields: "
            + ", ".join(missing_fields)
        )

    model_config.head_dim = getattr(mla_config, "qk_head_dim", None)
    if model_config.head_dim is None:
        model_config.head_dim = (
            mla_config.qk_nope_head_dim + mla_config.qk_rope_head_dim
        )
    model_config.attention_arch = AttentionArch.DSA
    model_config.kv_lora_rank = mla_config.kv_lora_rank
    model_config.qk_nope_head_dim = mla_config.qk_nope_head_dim
    model_config.qk_rope_head_dim = mla_config.qk_rope_head_dim
    model_config.v_head_dim = mla_config.v_head_dim
    model_config.index_topk = mla_config.index_topk
    model_config.index_head_dim = mla_config.index_head_dim
    model_config.index_n_heads = mla_config.index_n_heads
    model_config.index_kpool = getattr(mla_config, "index_kpool", None)
    model_config.index_topk_pattern = getattr(mla_config, "index_topk_pattern", None)
    # The indexer's key plane: the FP8-with-scale rows every in-tree scoring
    # leaf reads. A plugin hook that scores the checkpoint's bf16 keys sets
    # "bf16" after this (layers/attention/configs/dsa.py INDEX_K_FORMATS).
    model_config.index_k_format = "fp8_scaled"

    model_config.scaling = 1 / math.sqrt(
        model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    )
    rope_scaling = getattr(mla_config, "rope_scaling", None)
    if rope_scaling and "factor" in rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


def configure_mla_attention(model_config, server_args: ServerArgs) -> None:
    del server_args  # the geometry follows the checkpoint alone
    mla_config = (
        model_config.hf_text_config
        if hasattr(model_config.hf_text_config, "kv_lora_rank")
        else model_config.hf_config
    )
    model_config.head_dim = 256
    model_config.attention_arch = AttentionArch.MLA
    model_config.kv_lora_rank = mla_config.kv_lora_rank
    model_config.qk_nope_head_dim = mla_config.qk_nope_head_dim
    model_config.qk_rope_head_dim = mla_config.qk_rope_head_dim
    model_config.v_head_dim = mla_config.v_head_dim

    model_config.scaling = 1 / math.sqrt(
        model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    )
    rope_scaling = getattr(mla_config, "rope_scaling", None)
    if rope_scaling and "factor" in rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


def configure_minimax_m3_attention(model_config, server_args: ServerArgs) -> None:
    del server_args  # the geometry follows the checkpoint alone
    model_config.attention_arch = AttentionArch.MSA


_ATTENTION_FAMILY_SPECS = (
    _AttentionFamilySpec(
        name="DeepSeek V4.1",
        architectures=frozenset(
            {"DeepseekV41ForCausalLM", "DeepseekV41ForCausalLMDSpark"}
        ),
        configure=configure_deepseek_v41_attention,
        default_backend="deepseek_v41",
        default_prefix_granularity=256,
    ),
    _AttentionFamilySpec(
        name="DeepSeek V4",
        architectures=_DEEPSEEK_V4_ARCHITECTURES,
        configure=configure_deepseek_v4_attention,
        # V4 kernels need P to be a multiple of their fixed page; default to
        # exactly one page rather than restating the number.
        default_prefix_granularity=DEEPSEEK_V4_PAGE_SIZE,
    ),
    _AttentionFamilySpec(
        name="GLM",
        architectures=_DSA_ARCHITECTURES,
        configure=configure_dsa_attention,
        default_backend="dsa",
    ),
    _AttentionFamilySpec(
        name="MLA",
        architectures=_MLA_ARCHITECTURES,
        configure=configure_mla_attention,
    ),
    _AttentionFamilySpec(
        name="MiniMax MSA",
        architectures=_MSA_ARCHITECTURES,
        configure=configure_minimax_m3_attention,
        default_prefix_granularity=128,
    ),
)


def _model_architectures(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> list[str]:
    return (
        [resolve_architecture(hf_config)]
        + list(getattr(hf_config, "architectures", None) or [])
        + list(getattr(hf_text_config, "architectures", []) or [])
    )


def _resolve_attention_family(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> _AttentionFamilySpec | None:
    architectures = _model_architectures(hf_config, hf_text_config)
    for spec in _ATTENTION_FAMILY_SPECS:
        if any(arch in spec.architectures for arch in architectures):
            return spec
    return None


def _is_dflash2_mla(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> bool:
    architectures = _model_architectures(hf_config, hf_text_config)
    dflash_config = getattr(hf_text_config, "dflash_config", None) or getattr(
        hf_config, "dflash_config", None
    )
    return (
        "DFlash2DraftModel" in architectures
        and isinstance(dflash_config, dict)
        and dflash_config.get("attention_mode") == "mla"
    )


_GEMMA3_ARCHITECTURES = frozenset(
    {"Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"}
)
_GEMMA3_DEFAULT_SLIDING_WINDOW_PATTERN = 6

_GEMMA4_ARCHITECTURES = frozenset(
    {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"}
)


def is_gemma4(config: PretrainedConfig) -> bool:
    """True for a gemma-4 checkpoint (multimodal wrapper or bare text).

    Detected by architecture string OR by ``model_type``, the same way
    ``_maybe_synthesize_gemma3_layer_types`` keys gemma-3: the ``model_type``
    (``gemma4`` / ``gemma4_text``) is always on the config, whereas the
    ``architectures`` list can be lost to None on a multimodal wrapper. gemma-4
    is the one MHA model that splits its KV geometry by layer type, so the
    cache path reads this to decide whether to resolve head_dim / kv-heads per
    layer rather than once model-wide.
    """
    architectures = _model_architectures(config, get_hf_text_config(config))
    if any(arch in _GEMMA4_ARCHITECTURES for arch in architectures):
        return True
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and str(getattr(cfg, "model_type", "")).startswith("gemma4"):
            return True
    return False


def _maybe_synthesize_gemma3_layer_types(
    hf_text_config: PretrainedConfig,
    architectures: list[str],
) -> None:
    """Give a Gemma 3 text config an explicit ``layer_types`` list.

    Gemma 3 alternates 5 local sliding-window layers to every 1 global
    full-attention layer, but the 27B ``config.json`` declares only
    ``sliding_window`` (+ an implicit ``sliding_window_pattern`` of 6) and no
    per-layer ``layer_types``. The model (``models/gemma3.py``) and the KV pool
    (``MHAConfig``) both key their sliding-group placement off ``layer_types``;
    without it the pool collapses to a single full-history group and ignores
    the window. Materialise the labels HF's ``Gemma3TextConfig.__post_init__``
    would compute, so both sides agree.

    No-op when the config already carries ``layer_types`` (newer checkpoints),
    when the model is not Gemma 3, or when the layer count is unknown.
    """
    # Detect Gemma 3 by architecture string OR by model_type. The model_type
    # ("gemma3" / "gemma3_text") is always present on the config, whereas
    # ``architectures`` can be lost to None on a multimodal wrapper -- keying on
    # both means a config that lost its arch list still gets its window.
    is_gemma3 = any(arch in _GEMMA3_ARCHITECTURES for arch in architectures) or str(
        getattr(hf_text_config, "model_type", "")
    ).startswith("gemma3")
    if not is_gemma3:
        return
    if getattr(hf_text_config, "layer_types", None):
        return
    num_layers = getattr(hf_text_config, "num_hidden_layers", None)
    if not num_layers:
        return
    pattern = int(
        getattr(
            hf_text_config,
            "sliding_window_pattern",
            _GEMMA3_DEFAULT_SLIDING_WINDOW_PATTERN,
        )
        or _GEMMA3_DEFAULT_SLIDING_WINDOW_PATTERN
    )
    layer_types = [
        "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
        for i in range(int(num_layers))
    ]
    # Bypass __setattr__: gemma3's wrapper config forwards attribute access to
    # text_config, and we want this pinned on the object the rest of the
    # pipeline reads.
    hf_text_config.__dict__["layer_types"] = layer_types


def _apply_block_spec_widths(
    server_args: ServerArgs,
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> int | None:
    """Reconcile the block-drafter launch widths with the draft checkpoint.

    Args:
        server_args: Server args whose speculative widths are checked, or set
            when they were left at their defaults.
        hf_config: The draft checkpoint's config.
        hf_text_config: Its text config, searched first.

    Returns:
        The checkpoint's block size, or None when it declares none.
    """
    algorithm = getattr(server_args, "speculative_algorithm", None)
    if algorithm not in BLOCK_SPEC_ALGORITHMS:
        return None
    block_size = read_checkpoint_block_size(hf_text_config, hf_config)
    if block_size is None:
        return None
    if getattr(server_args, "_speculative_widths_explicit", True):
        validate_block_widths(
            algorithm,
            block_size,
            server_args.speculative_num_steps,
            server_args.speculative_num_draft_tokens,
        )
    num_steps, num_draft_tokens = resolve_block_widths(algorithm, block_size)
    server_args.speculative_num_steps = num_steps
    server_args.speculative_num_draft_tokens = num_draft_tokens
    return block_size


def _apply_attention_defaults(
    server_args: ServerArgs,
    *,
    name: str,
    default_backend: str | None,
    default_prefix_granularity: int | None,
    is_draft_worker: bool,
) -> None:
    """Fill launch arguments the user left at their defaults."""
    if default_prefix_granularity is not None:
        granularity_default = ServerArgs.__dataclass_fields__[
            "prefix_granularity"
        ].default
        if server_args.prefix_granularity == granularity_default:
            logger.info(
                f"{name!s} default prefix_granularity="
                f"{default_prefix_granularity:d}; pass --prefix-granularity "
                f"with a value other than {granularity_default:d} to keep that value.",
            )
            server_args.prefix_granularity = default_prefix_granularity
    if default_backend is None:
        return
    # A draft model's default belongs to the drafter's backend selection;
    # writing the target field would either be discarded (already set) or
    # hijack the target's own default.
    if is_draft_worker:
        if server_args.drafter_attention_backend is None:
            server_args.drafter_attention_backend = default_backend
        elif server_args.drafter_attention_backend != default_backend:
            # Server-args resolution mirrors --attention-backend into the
            # drafter before any model is known, so the two cannot be told
            # apart here; say which one won.
            logger.info(
                f"{name!s} draft default attention backend {default_backend!r} "
                f"is superseded by {server_args.drafter_attention_backend!r} "
                "(--drafter-attention-backend, or --attention-backend mirrored "
                "to the drafter); pass --drafter-attention-backend to choose."
            )
    elif server_args.attention_backend is None:
        server_args.attention_backend = default_backend


def _derive_num_attention_layers(
    hf_config: PretrainedConfig,
    num_hidden_layers: int,
    model_profile: ModelProfile | None = None,
) -> int:
    # A registered model declares its own attention-instance count; the
    # architecture-name tables below stay as the in-tree seed.
    if model_profile is not None:
        return num_hidden_layers * model_profile.attention_instances_per_layer
    architectures = getattr(hf_config, "architectures", None) or []
    num_attention_layers = num_hidden_layers
    if "WhisperForConditionalGeneration" in architectures:
        # Encoder keeps no paged KV; cross-KV lives outside the arena.
        # Size the pool for decoder self-attention only.
        return int(getattr(hf_config, "decoder_layers", num_hidden_layers))
    if is_deepseek_v4_nextn(hf_config):
        num_attention_layers = int(getattr(hf_config, "num_nextn_predict_layers", 1))
    if any(arch in _DOUBLE_ATTENTION_LAYER_ARCHITECTURES for arch in architectures):
        num_attention_layers = num_hidden_layers * 2
    if any(arch in _CACHE_LAYER_VIEW_ARCHITECTURES for arch in architectures):
        num_attention_layers = len(hf_config.cache_layer_types)
    return num_attention_layers


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: str | None = None,
        context_length: int | None = None,
        model_override_args: dict | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        override_config_file: str | None = None,
        is_draft_worker: bool | None = False,
        server_args: ServerArgs = None,
    ) -> None:
        # Plugins may register the architecture, its config class and its
        # profile; every resolution below must see them.
        ensure_loaded()
        if server_args is not None and server_args.speculative_algorithm is not None:
            # Post-discovery replacement for the CLI choices= this flag no
            # longer carries: plugins may have added algorithms.
            from tokenspeed.runtime.execution.drafter import (
                require_plugin_draft_checkpoint,
                validate_drafter_algorithm,
            )

            validate_drafter_algorithm(server_args.speculative_algorithm)
            require_plugin_draft_checkpoint(server_args)
        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization
        self.mapping = server_args.mapping

        # Parse args
        self.model_override_args = json.loads(model_override_args)
        kwargs = {}
        if override_config_file and override_config_file.strip():
            kwargs["_configuration_file"] = override_config_file.strip()

        self.hf_config = get_config(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            is_draft_worker=is_draft_worker,
            speculative_algorithm=(
                getattr(server_args, "speculative_algorithm", None)
                if is_draft_worker
                else None
            ),
            **kwargs,
        )
        self.hf_generation_config = get_generation_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        self.hf_text_config = get_hf_text_config(self.hf_config)
        # A registered model states its own family facts; in-tree models
        # without a profile still resolve through the architecture tables.
        # Candidates are exactly the list the model loader walks, so the
        # profile cannot come from a different architecture than the class
        # that is built (the loader checks the pairing).
        resolved_profile = resolve_model_profile(
            model_loader_architectures(self.hf_config), self.hf_config
        )
        self.model_profile_architecture: str | None = (
            resolved_profile[0] if resolved_profile is not None else None
        )
        self.model_profile: ModelProfile | None = (
            resolved_profile[1] if resolved_profile is not None else None
        )
        self.spec_block_size: int | None = None
        if is_draft_worker:
            self.spec_block_size = _apply_block_spec_widths(
                server_args, self.hf_config, self.hf_text_config
            )
        self.dspark_prefix_replay_tokens: int | None = None
        if (
            is_draft_worker
            and getattr(server_args, "speculative_algorithm", None) == "DSPARK"
            and resolve_architecture(self.hf_config)
            in ("DeepseekV4ForCausalLMDSpark", "DeepseekV41ForCausalLMDSpark")
        ):
            from tokenspeed.runtime.models.deepseek_v4_dspark import (
                DEFAULT_DSPARK_WINDOW_SIZE,
                count_dspark_stages,
            )

            if self.spec_block_size is None:
                raise ValueError(
                    "DSPARK same-checkpoint decoding requires the checkpoint to "
                    "declare dspark_block_size."
                )
            dspark_window_size = int(
                getattr(
                    self.hf_text_config,
                    "dspark_window_size",
                    DEFAULT_DSPARK_WINDOW_SIZE,
                )
            )
            if dspark_window_size <= 0:
                raise ValueError(
                    "DSPARK captured-context window size must be positive; "
                    f"got {dspark_window_size}."
                )
            # V4.1 keeps its windows in the SWA cache group, so a prefix hit
            # already carries them; V4 rebuilds a drafter-private ring instead.
            self.dspark_prefix_replay_tokens = (
                0
                if resolve_architecture(self.hf_config)
                == "DeepseekV41ForCausalLMDSpark"
                else dspark_window_size
            )
            dspark_num_stages = count_dspark_stages(
                model_path,
                revision=revision,
            )
            if dspark_num_stages is None:
                raise ValueError(
                    "DSPARK requires a safetensors index with mtp.<stage> weights."
                )
            self.hf_text_config.dspark_num_stages = dspark_num_stages
            if self.hf_config is not self.hf_text_config:
                self.hf_config.dspark_num_stages = dspark_num_stages
        if (
            is_draft_worker
            and resolve_architecture(self.hf_config)
            == "InklingForConditionalGenerationNextN"
        ):
            # The MTP head's depth blocks have their own local/full attention
            # pattern (mtp_config.local_layer_ids) and only depths
            # 0..steps-1 ever run; swap in the depth-specialized (and
            # steps-pruned) text config so layer construction, attention
            # metadata, and cache-group layout all derive from it.
            from tokenspeed.runtime.configs.inkling_config import (
                inkling_mtp_text_config,
            )

            self.hf_text_config = inkling_mtp_text_config(
                self.hf_text_config,
                num_steps=getattr(server_args, "speculative_num_steps", None),
            )
            if hasattr(self.hf_config, "text_config"):
                self.hf_config.text_config = self.hf_text_config

        # Check model type.
        # A pooling checkpoint is served as an embedding model: one prefill,
        # one vector, no decode. Draft/auxiliary checkpoints are never pooled.
        self.pooling_config = (
            None if is_draft_worker else resolve_pooling_config(model_path, server_args)
        )
        self.is_generation = self.pooling_config is None and is_generation_model(
            self.hf_config.architectures
        )
        if self.pooling_config is not None:
            # Model-facing gate, same idiom as ``hf_config.encoder_only``: the
            # model class is constructed from the HF config alone, so the
            # decision has to travel on it.
            self.hf_config.pooling_config = self.pooling_config
            if hasattr(self.hf_config, "text_config"):
                self.hf_config.text_config.pooling_config = self.pooling_config
        # Prefer the combined architecture list: multimodal wrappers can
        # lose ``hf_config.architectures`` to None after text_config attach
        # (gemma3/gemma4), while ``resolve_architecture`` / text_config still
        # carry ``*ForConditionalGeneration``.
        _mm_archs = _model_architectures(self.hf_config, self.hf_text_config)
        self.is_multimodal = is_multimodal_model(_mm_archs)
        self.is_multimodal_gen = is_multimodal_gen_model(_mm_archs)
        self.is_image_gen = is_image_gen_model(_mm_archs)
        self.is_audio_model = is_audio_model(_mm_archs)

        language_model_only = bool(getattr(server_args, "language_model_only", False))
        # Target-only flag; never apply to draft / auxiliary checkpoints.
        apply_language_model_only = language_model_only and not is_draft_worker
        if apply_language_model_only:
            if not self.is_multimodal:
                raise ValueError(
                    "--language-model-only requires a multimodal model checkpoint."
                )
            logger.info(
                "Running in language-model-only mode: vision/audio encoders will "
                "be skipped; requests with multimodal inputs will be rejected."
            )
        # ``is_multimodal`` is the architectural fact; this is the runtime gate.
        self.is_multimodal_active = self.is_multimodal and not apply_language_model_only
        if (
            not is_draft_worker
            and getattr(server_args, "mm_encoder_tp_mode", "weights") == "data"
        ):
            if not self.is_multimodal_active:
                raise ValueError("item-DP requires an active multimodal encoder")
        # Vision-only role (EPD encode): the inverse axis of language_model_only.
        # Build the vision tower (is_multimodal_active stays True) but SKIP LM
        # construction + LM weight load so a full ViT fits at encode TP=1.
        encoder_only = (
            getattr(server_args, "disaggregation_mode", None) == "encode"
            and not is_draft_worker
        )
        if encoder_only and not self.is_multimodal:
            raise ValueError(
                "disaggregation_mode=encode requires a multimodal checkpoint."
            )
        if encoder_only and apply_language_model_only:
            raise ValueError(
                "disaggregation_mode=encode (encoder-only) and language_model_only "
                "are mutually exclusive."
            )
        if encoder_only and self.is_audio_model:
            raise ValueError(
                "disaggregation_mode=encode does not support audio models; "
                "only image/video encoders are currently supported."
            )
        if encoder_only:
            # Single model-facing gate: Kimi reads hf_config.encoder_only directly;
            # Qwen3_5ForConditionalGeneration reads it to skip LM construction.
            self.hf_config.encoder_only = True
            logger.info(
                "Running in encoder-only mode: the language model will not "
                "be constructed or loaded (encode role)."
            )
        # Cap gpu_memory_utilization for VLMs in mm mode — the vision encoder
        # needs headroom that the global default doesn't account for.
        if (
            self.is_multimodal_active
            and getattr(server_args, "_gpu_memory_utilization_defaulted", False)
            and server_args.gpu_memory_utilization > 0.9
        ):
            logger.info(
                "Clamping gpu_memory_utilization "
                f"{server_args.gpu_memory_utilization:.2f} -> 0.9 to leave headroom "
                "for the vision encoder.",
            )
            server_args.gpu_memory_utilization = 0.9
        self.mm_attention_backend = getattr(server_args, "mm_attention_backend", None)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)

        # Derive context length
        derived_context_len = get_context_length(self.hf_text_config)
        if context_length is not None:
            if context_length > derived_context_len:
                if envs.TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.get():
                    logger.warning(
                        f"User-specified context_length ({context_length!s}) is greater"
                        " than the derived "
                        f"context_length ({derived_context_len!s}). This may lead to "
                        "incorrect model outputs or "
                        "CUDA errors.",
                    )
                    self.context_len = context_length
                else:
                    raise ValueError(
                        f"User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config. "
                        f"To allow overriding this maximum, set the env var TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                    )
            else:
                self.context_len = context_length
        else:
            self.context_len = derived_context_len

        self._clamp_prefill_budget_to_context(server_args)

        if getattr(self.hf_config, "is_encoder_decoder", False):
            # Cross-attention KV is a per-slot buffer outside the paged arena.
            # +1 mirrors the scheduler's padding slot.
            self.hf_config.cross_attn_slots = int(server_args.max_num_seqs) + 1

        if self.pooling_config is not None:
            # Deferred to here, not to where pooling_config is resolved: the
            # prefill-chunk guard below is stated against context_len, which
            # only settles above.
            self._resolve_embedding_server_args(server_args)

        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        # Storage of the DSA index-key plane, one of INDEX_K_FORMATS
        # (layers/attention/configs/dsa.py). A DSA model's configure-attention
        # hook names it (configure_dsa_attention: "fp8_scaled"); None for a
        # model without an indexer, and DSAConfig refuses None.
        self.index_k_format: str | None = None

        # MLA/DSA families carry per-head dimension metadata that does not
        # follow the standard hidden_size / num_attention_heads derivation above.
        attention_family = _resolve_attention_family(
            self.hf_config,
            self.hf_text_config,
        )
        model_architectures = _model_architectures(
            self.hf_config,
            self.hf_text_config,
        )
        if self.model_profile is not None:
            _apply_attention_defaults(
                server_args,
                name=resolve_architecture(self.hf_config),
                default_backend=self.model_profile.default_attention_backend,
                default_prefix_granularity=(
                    self.model_profile.default_prefix_granularity
                ),
                is_draft_worker=bool(is_draft_worker),
            )
            self.model_profile.configure_attention(self, server_args)
        elif attention_family is not None:
            _apply_attention_defaults(
                server_args,
                name=attention_family.name,
                default_backend=attention_family.default_backend,
                default_prefix_granularity=attention_family.default_prefix_granularity,
                is_draft_worker=bool(is_draft_worker),
            )
            attention_family.configure(self, server_args)
        elif _is_dflash2_mla(self.hf_config, self.hf_text_config):
            configure_mla_attention(self, server_args)
        elif "MiniCPM3ForCausalLM" in model_architectures:
            self.head_dim = 128
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        else:
            self.attention_arch = AttentionArch.MHA

        # Gemma 3 ships as *ForConditionalGeneration and its 27B config.json
        # omits an explicit ``layer_types`` list. Synthesise onto the text
        # config so the model and KV pool read the same 5:1 local:global labels.
        _maybe_synthesize_gemma3_layer_types(self.hf_text_config, model_architectures)

        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        # for Dbrx and MPT models
        if self.hf_config.model_type in {"dbrx", "mpt"}:
            self.num_key_value_heads = getattr(
                self.hf_config.attn_config, "kv_n_heads", None
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = getattr(self.hf_text_config, "num_hidden_layers", None)
        if self.num_hidden_layers is None:
            self.num_hidden_layers = self.hf_text_config.num_layers
        self.num_attention_layers = _derive_num_attention_layers(
            self.hf_config,
            self.num_hidden_layers,
            self.model_profile,
        )
        if is_draft_worker:
            dspark_layers = getattr(self.hf_text_config, "dspark_num_stages", None)
            mtp_layers = getattr(self.hf_text_config, "mtp_num_hidden_layers", None)
            if dspark_layers is not None:
                self.num_attention_layers = int(dspark_layers)
            elif mtp_layers is not None:
                self.num_attention_layers = mtp_layers
            else:
                nextn_layers = getattr(
                    self.hf_text_config, "num_nextn_predict_layers", None
                )
                if nextn_layers is not None and nextn_layers > 0:
                    self.num_attention_layers = nextn_layers
        self.vocab_size = self.hf_text_config.vocab_size

        # Verify quantization
        self._verify_quantization()
        if server_args is not None and not is_draft_worker:
            # The decode TP layouts need unquantized o_proj / down_proj; judge
            # the checkpoint's resolved method, not only --quantization.
            server_args.validate_tp_batch_invariant_weights(
                self.quantization,
                getattr(self.hf_text_config, "disable_quant_module", None) or (),
            )

        # Cache attributes
        self.hf_eos_token_id = self.get_hf_eos_token_id()
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)
        if self.image_token_id is None:
            # Gemma 3 uses image_token_index; Gemma 4 uses image_token_id.
            self.image_token_id = getattr(self.hf_config, "image_token_index", None)

        if server_args is not None and server_args.load_format == "extensible":
            override_model_config(self, server_args.ext_yaml)

    @property
    def tokenizer_kwargs(self) -> Mapping[str, object]:
        """Extra tokenizer keyword arguments the model's profile declares."""
        if self.model_profile is None:
            return {}
        return self.model_profile.tokenizer_kwargs

    @property
    def requires_request_token_history(self) -> bool:
        """Whether the model reads each request's committed token history."""
        return (
            self.model_profile is not None and self.model_profile.request_token_history
        )



    def _clamp_prefill_budget_to_context(self, server_args: ServerArgs) -> None:
        """Cap the per-step prefill budget at what the request pool can hold.

        Whisper's decoder holds 448 positions, so the default 8192-token budget
        implies more concurrent requests than max_num_seqs allows, and the
        autotuner's dummy batch indexes past the pool. Clamped rather than
        refused: operators took a default that predates short-context models.
        """
        if server_args is None:
            return
        reachable = int(self.context_len) * max(1, int(server_args.max_num_seqs))
        for field in ("chunked_prefill_size", "max_prefill_tokens"):
            current = getattr(server_args, field, None)
            if current is None or current <= 0 or current <= reachable:
                continue
            logger.info(
                "%s=%s exceeds what the request pool can hold (context_len %s "
                "x max_num_seqs %s = %s); clamping to %s.",
                field,
                current,
                self.context_len,
                server_args.max_num_seqs,
                reachable,
                reachable,
            )
            setattr(server_args, field, reachable)

    def _resolve_embedding_server_args(self, server_args: ServerArgs) -> None:
        """Force off the features a pooling forward cannot express.

        All four reference engines disable the same two things for pooling
        models: chunked prefill (a pooled vector reduces over the whole prompt)
        and prefix caching (a pooling request never reads back its own KV).
        Both are forced rather than validated.
        """
        if server_args.enable_prefix_caching:
            logger.info(
                "Embedding model: disabling prefix caching. A pooling request "
                "computes one forward and never reads its own KV back."
            )
            server_args.enable_prefix_caching = False
        # Pooling reduces over the whole prompt; a partial chunk would produce
        # a well-formed vector over the wrong tokens. Open both the admission
        # budget and the chunk size to the context length rather than refusing
        # a default --max-prefill-tokens the operator never set.
        budget = max(int(server_args.max_prefill_tokens), int(self.context_len))
        if int(server_args.max_prefill_tokens) < self.context_len:
            logger.info(
                "Embedding model: raising --max-prefill-tokens from %s to %s "
                "so a pooled prompt fits in one prefill.",
                server_args.max_prefill_tokens,
                budget,
            )
            server_args.max_prefill_tokens = budget
        if server_args.chunked_prefill_size != budget:
            logger.info(
                "Embedding model: opening the prefill chunk budget from %s to "
                "%s so a pooled prompt is never split across chunks.",
                server_args.chunked_prefill_size,
                budget,
            )
            server_args.chunked_prefill_size = budget
        if server_args.speculative_algorithm:
            raise ValueError(
                "Speculative decoding is meaningless for an embedding model: "
                "there are no tokens to draft. Drop --speculative-algorithm."
            )
        if not server_args.disable_overlap_schedule:
            logger.info(
                "Embedding model: disabling the overlap schedule. A pooling "
                "request finishes on its first forward, and overlap would "
                "schedule a decode step against it before that lands."
            )
            server_args.disable_overlap_schedule = True

        # Settle the tri-state: after this, `server_args.is_embedding` is the
        # resolved answer, not the request.
        server_args.is_embedding = True
        server_args.pooling_type = self.pooling_config.pooling_type.value
        server_args.pooling_normalize = self.pooling_config.normalize

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        if quant_cfg is None:
            # modelopt NVFP4 checkpoints store quant config in hf_quant_config.json
            # Resolve the local snapshot directory (model_path may be a HF hub ID)
            if os.path.isdir(self.model_path):
                model_dir = self.model_path
            else:
                try:
                    from huggingface_hub import snapshot_download

                    model_dir = snapshot_download(
                        self.model_path,
                        revision=self.revision,
                        allow_patterns=["*.json"],
                        local_files_only=True,
                    )
                except Exception as exc:
                    logger.debug(
                        "Unable to resolve local quantization config for "
                        f"{self.model_path!s}: {exc!s}",
                    )
                    model_dir = None
            if model_dir is not None:
                hf_quant_path = os.path.join(model_dir, "hf_quant_config.json")
                if os.path.isfile(hf_quant_path):
                    with open(hf_quant_path, encoding="utf-8") as f:
                        hf_quant = json.load(f)
                    quant_algo = hf_quant.get("quantization", {}).get("quant_algo", "")
                    if quant_algo:
                        quant_cfg = {
                            "quant_method": "modelopt",
                            "quant_algo": quant_algo,
                        }
                        quant_cfg.update(hf_quant.get("quantization", {}))
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = [*QUANTIZATION_METHODS]

        optimized_quantization_methods = [
            "fp8",
            "nvfp4",
            "mxfp4",
            "modelopt_mixed",
            "compressed_tensors",
            "compressed-tensors",
            "w8a8_fp8",
        ]
        compatible_quantization_methods = {
            "w8a8_fp8": ["compressed-tensors", "compressed_tensors"],
        }
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()
            # Detect which checkpoint is it
            for _, method in QUANTIZATION_METHODS.items():
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization
                )
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                if (
                    self.quantization not in compatible_quantization_methods
                    or quant_method
                    not in compatible_quantization_methods[self.quantization]
                ):
                    raise ValueError(
                        "Quantization method specified in the model config "
                        f"({quant_method}) does not match the quantization "
                        f"method specified in the `quantization` argument "
                        f"({self.quantization})."
                    )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )

            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    f"{self.quantization!s} quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.",
                )

    def get_hf_eos_token_id(self) -> set[int] | None:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        if eos_ids is None:
            eos_ids = set()
        if self.hf_generation_config:
            generation_eos_ids = getattr(
                self.hf_generation_config, "eos_token_id", None
            )
            if generation_eos_ids:
                generation_eos_ids = (
                    {generation_eos_ids}
                    if isinstance(generation_eos_ids, int)
                    else set(generation_eos_ids)
                )
                eos_ids = eos_ids | generation_eos_ids
        return eos_ids


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = resolve_architecture(config)
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # We set `dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in
        # `python/tokenspeed/runtime/models/llava.py`.
        config.dtype = torch.float16
        return config

    if hasattr(config, "thinker_config"):
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            return thinker_config.text_config
        return thinker_config
    if hasattr(config, "text_config"):
        if not hasattr(config.text_config, "num_attention_heads"):
            raise ValueError("text_config must define num_attention_heads")
        return config.text_config
    return config


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str | torch.dtype,
) -> torch.dtype:
    # config.dtype can be missing or None.
    config_dtype = getattr(config, "dtype", None)
    if config_dtype is None:
        config_dtype = torch.bfloat16

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                if config.model_type == "gemma2":
                    logger.info(
                        "For Gemma 2, we downcast float32 to bfloat16 instead "
                        "of float16 by default. Please specify `dtype` if you "
                        "want to use float16."
                    )
                    torch_dtype = torch.bfloat16
                else:
                    # Following the common practice, we use float16 for float32
                    # models.
                    torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info(f"Upcasting {config_dtype!s} to {torch_dtype!s}.")
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info(f"Downcasting {config_dtype!s} to {torch_dtype!s}.")
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype!s} to {torch_dtype!s}.")

    return torch_dtype



@dataclass(frozen=True)
class PoolingConfig:
    """How an embedding checkpoint reduces hidden states to one vector."""

    pooling_type: PoolingType
    normalize: bool


# sentence-transformers spells its pooling choice as a set of mutually
# exclusive booleans. Only the three we implement are listed; a checkpoint
# selecting any other mode is rejected rather than silently pooled the wrong
# way -- a wrong reduction returns a well-formed vector nothing downstream
# can catch.
_ST_POOLING_MODES = {
    "pooling_mode_lasttoken": PoolingType.LAST,
    "pooling_mode_cls_token": PoolingType.CLS,
    "pooling_mode_mean_tokens": PoolingType.MEAN,
}


def read_sentence_transformers_pooling(model_path: str) -> PoolingConfig | None:
    """Read ``1_Pooling/config.json`` + ``modules.json`` from a checkpoint.

    Returns None when the checkpoint ships no sentence-transformers pooling
    config, which is how a plain generative checkpoint looks.
    """
    pooling_file = os.path.join(model_path, "1_Pooling", "config.json")
    if not os.path.isfile(pooling_file):
        return None
    with open(pooling_file) as handle:
        raw = json.load(handle)

    selected = [
        pooling_type for key, pooling_type in _ST_POOLING_MODES.items() if raw.get(key)
    ]
    enabled_unsupported = [
        key
        for key, value in raw.items()
        if key.startswith("pooling_mode_") and value and key not in _ST_POOLING_MODES
    ]
    if enabled_unsupported:
        raise ValueError(
            f"{pooling_file} selects unsupported pooling mode(s) "
            f"{sorted(enabled_unsupported)}; supported: "
            f"{sorted(_ST_POOLING_MODES)}. Pass --pooling-type to override."
        )
    if len(selected) != 1:
        raise ValueError(
            f"{pooling_file} selects {len(selected)} pooling modes; expected "
            "exactly one. Pass --pooling-type to override."
        )

    # Normalization is a separate sentence-transformers module, not a field of
    # the pooling config: read it off modules.json rather than assuming it.
    normalize = False
    modules_file = os.path.join(model_path, "modules.json")
    if os.path.isfile(modules_file):
        with open(modules_file) as handle:
            modules = json.load(handle)
        normalize = any(
            str(module.get("type", "")).endswith("Normalize") for module in modules
        )

    return PoolingConfig(pooling_type=selected[0], normalize=normalize)


def resolve_pooling_config(
    model_path: str, server_args: ServerArgs | None
) -> PoolingConfig | None:
    """Decide whether this server pools, and how.

    Returns None for a generative server. Raises when the operator asked for
    embedding mode on a checkpoint that declares no pooling and gave no
    override -- an embedding server that guessed its reduction would answer
    every request with a plausible wrong vector.
    """
    declared = read_sentence_transformers_pooling(model_path)
    requested = getattr(server_args, "is_embedding", None)
    if requested is False:
        return None
    if requested is None and declared is None:
        return None

    override_type = getattr(server_args, "pooling_type", None)
    override_normalize = getattr(server_args, "pooling_normalize", None)

    if declared is None and override_type is None:
        raise ValueError(
            f"--is-embedding was requested but {model_path} ships no "
            "1_Pooling/config.json to say how to pool. Pass --pooling-type "
            "(last|cls|mean), and --pooling-normalize if the checkpoint "
            "expects L2-normalized output."
        )

    pooling_type = (
        PoolingType(override_type)
        if override_type is not None
        else declared.pooling_type
    )
    if override_normalize is not None:
        normalize = override_normalize
    else:
        normalize = declared.normalize if declared is not None else False
    return PoolingConfig(pooling_type=pooling_type, normalize=normalize)


def is_generation_model(model_architectures: list[str]):
    return True


def is_multimodal_model(model_architectures: list[str] | None):
    multimodal_architectures = {
        "DeepseekV41ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "Glm53FlashForConditionalGeneration",
        "InklingForConditionalGeneration",
        "MiniMaxM3SparseForConditionalGeneration",
        # Audio in, tokens out: same multimodal request shape as an image,
        # even though Whisper is encoder-decoder rather than audio-prefix.
        "WhisperForConditionalGeneration",
        # Gemma 3/4 multimodal (text+image). Vision tower is optional at
        # runtime via language_model_only / is_multimodal_active.
        "Gemma3ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    }
    return any(arch in multimodal_architectures for arch in model_architectures or [])


def is_multimodal_gen_model(model_architectures: list[str]):
    return False


def is_image_gen_model(model_architectures: list[str]):
    return False


def is_audio_model(model_architectures: list[str] | None):
    audio_architectures = {
        "WhisperForConditionalGeneration",
        "InklingForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
    }
    return any(arch in audio_architectures for arch in model_architectures or [])


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
