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


"""Utilities for selecting and loading models."""

import contextlib
from collections.abc import Generator

import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype) -> Generator[None]:
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


# Fallback map from a config's ``model_type`` to the registered architecture,
# for checkpoints whose config arrives with ``architectures = None`` (e.g. a
# Gemma 3 text sub-config). Keyed by ``model_type`` PREFIX so both the
# multimodal wrapper ("gemma3") and its text sub-config ("gemma3_text") resolve
# to the same entry class; the text-only decoder is handled by that class.
_MODEL_TYPE_ARCH_ALIASES: tuple[tuple[str, str], ...] = (
    ("gemma3", "Gemma3ForConditionalGeneration"),
)


def _architectures_from_model_type(hf_config) -> list[str]:
    model_type = str(getattr(hf_config, "model_type", "") or "")
    for prefix, arch in _MODEL_TYPE_ARCH_ALIASES:
        if model_type.startswith(prefix):
            return [arch]
    return []


def _architectures_from_config_class(hf_config) -> list[str]:
    """Last-resort architecture name from the HF config class.

    Used when ``resolve_architecture`` is unavailable in this tokenspeed tree.
    """
    name = type(hf_config).__name__
    # Strip a trailing "Config" so e.g. Gemma3TextConfig -> Gemma3Text, which is
    # still not a registered arch but is better than raising AttributeError on
    # a None architectures list.
    if name.endswith("Config"):
        name = name[: -len("Config")]
    return [name] if name else []


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from tokenspeed.runtime.models.registry import ModelRegistry

    # ``hf_config.architectures`` can be present-but-None on wrapper / text
    # configs (e.g. Gemma 3). ``getattr(..., [])`` does NOT help -- the
    # attribute exists and is None, so the default never fires. Resolve in
    # order: outer -> nested text_config -> model_type alias ->
    # resolve_architecture (if available) -> config class name.
    architectures = getattr(model_config.hf_config, "architectures", None)
    if not architectures:
        text_config = getattr(model_config.hf_config, "text_config", None)
        if text_config is not None:
            architectures = getattr(text_config, "architectures", None)
    if not architectures:
        architectures = _architectures_from_model_type(model_config.hf_config)
    if not architectures:
        try:
            from tokenspeed.runtime.utils.hf_transformers_utils import (
                resolve_architecture,
            )

            architectures = [resolve_architecture(model_config.hf_config)]
        except Exception:
            architectures = _architectures_from_config_class(model_config.hf_config)
    # Mixtral only supports the quantization backends listed here in the
    # current model registry and loader stack.
    mixtral_supported = ["fp8", "compressed-tensors"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    return ModelRegistry.resolve_model_cls(architectures)
