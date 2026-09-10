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

"""torchao quantization config.

Loads checkpoints whose ``config.json`` declares a torchao
``quantization_config`` (``quant_method == "torchao"``), e.g. the
``gemma-3-27b-it-FP8`` build. The heavy lifting -- reconstructing the quantized
weight tensor subclass and running the dequant/compute -- is delegated to the
``torchao`` library itself (``quantize_`` + ``AOBaseConfig``); this module is
only the glue that adapts torchao to the engine's ``QuantizationConfig`` /
``LinearMethodBase`` contract.

Ported from the vLLM implementation
(``vllm/model_executor/layers/quantization/torchao.py``) and adapted to the
tokenspeed interfaces. ``torchao`` is imported lazily inside the methods so the
engine still imports on a box without the package installed; it is only
required when a torchao checkpoint is actually served.
"""

from __future__ import annotations

import importlib
import json
from importlib.util import find_spec
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def torchao_version_at_least(torchao_version: str) -> bool:
    """True when an installed torchao is at least ``torchao_version``."""
    if find_spec("torchao"):
        try:
            from packaging import version

            installed = importlib.metadata.version("torchao")
            return version.parse(installed) >= version.parse(torchao_version)
        except Exception:
            return False
    return False


def should_skip(prefix: str, skip_modules: list[str]) -> bool:
    """Whether a module ``prefix`` is in the not-to-convert list.

    Matches vLLM's robust rule: an exact fqn match, or ``skip`` appearing as a
    whole dotted segment of ``prefix`` (so ``"o_proj"`` skips
    ``model.layers.10.o_proj`` but ``"layers.1"`` does not skip
    ``layers.11``).
    """
    for s in skip_modules:
        if prefix == s:
            return True
        if f".{s}." in f".{prefix}.":
            return True
    return False


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao-quantized checkpoints."""

    def __init__(
        self,
        torchao_config: Any,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
        super().__init__()
        self.torchao_config = torchao_config
        self.skip_modules = skip_modules or []
        self.is_checkpoint_torchao_serialized = is_checkpoint_torchao_serialized

    def __repr__(self) -> str:
        return (
            f"TorchAOConfig(torchao_config={self.torchao_config!r}, "
            f"skip_modules={self.skip_modules!r}, "
            f"is_checkpoint_torchao_serialized="
            f"{self.is_checkpoint_torchao_serialized!r})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "torchao"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # torchao's int/float weight-only paths run broadly; the specific
        # kernel a config selects enforces its own capability at compute time.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        # torchao reads its config from the HF ``config.json`` quantization
        # block, not a side file.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TorchAOConfig:
        """Build the config from the HF ``quantization_config`` mapping."""
        try:
            from torchao.core.config import config_from_dict
        except ImportError as err:
            raise ImportError(
                "Loading a torchao-quantized checkpoint requires the torchao "
                "package. Install it with `pip install \"torchao>=0.10.0\"` "
                "(install.sh does this for the venv)."
            ) from err

        quant_method = cls.get_from_keys_or(config, ["quant_method"], None)
        is_checkpoint_torchao_serialized = (
            quant_method is not None and "torchao" in quant_method
        )

        quant_type = cls.get_from_keys_or(config, ["quant_type"], None)
        assert quant_type is not None, "quant_type must be specified"
        assert len(quant_type) == 1 and "default" in quant_type, (
            "Expected exactly one key 'default' in the torchao quant_type"
        )
        ao_config = config_from_dict(quant_type["default"])

        # Modules the checkpoint marks as not-to-convert stay unquantized.
        skip_modules = config.get("modules_to_not_convert", []) or []

        # A per-module config map may mark some modules as None (skip).
        inner = quant_type["default"].get("_data", {})
        if not isinstance(inner, dict):
            inner = {}
        module_fqn = inner.get("module_fqn_to_config", {})
        if not isinstance(module_fqn, dict):
            module_fqn = {}
        for layer, layer_cfg in module_fqn.items():
            if layer_cfg is None:
                skip_modules.append(layer)

        return cls(ao_config, skip_modules, is_checkpoint_torchao_serialized)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Resolve the linear method for ``layer`` at ``prefix``.

        Mirrors vLLM: honour ``skip_modules`` (unquantized), support a
        per-module-fqn config map (exact match, then ``re:``-prefixed regex,
        then ``_default``), and otherwise apply the single top-level config.
        """
        from tokenspeed.runtime.layers.dense.torchao import (
            TorchAOLinearMethod,
            UnquantizedLinearMethod,
        )
        from tokenspeed.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        try:
            from torchao.quantization import ModuleFqnToConfig
        except ImportError:
            ModuleFqnToConfig = ()  # type: ignore[assignment]

        if ModuleFqnToConfig and isinstance(self.torchao_config, ModuleFqnToConfig):
            import regex as re

            module_map = self.torchao_config.module_fqn_to_config
            c = None
            if prefix in module_map:
                c = module_map[prefix]
            else:
                for pattern in module_map:
                    if pattern.startswith("re:") and re.fullmatch(pattern[3:], prefix):
                        c = module_map[pattern]
                        break
                else:
                    c = module_map.get("_default", None)
            if c is None:
                return UnquantizedLinearMethod()
            resolved = TorchAOConfig(
                c, self.skip_modules, self.is_checkpoint_torchao_serialized
            )
            return TorchAOLinearMethod(resolved)

        return TorchAOLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []

    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> TorchAOConfig:
        """Build from a JSON string of a torchao config dict (test/helper)."""
        config_dict = json.loads(config_dict_json)
        return cls.from_config({"quant_type": {"default": config_dict}})
