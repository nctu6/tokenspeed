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

"""torchao linear method.

The quantized weight is a torchao tensor subclass; ``F.linear`` dispatches to
torchao's own dequant/compute kernel, so ``apply`` is a plain functional
linear. Ported from vLLM's ``TorchAOLinearMethod`` and adapted to tokenspeed's
``LinearMethodBase``. See ``runtime/layers/quantization/torchao.py`` for the
config that selects this method.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.utils import set_weight_attrs

__all__ = ["TorchAOLinearMethod", "UnquantizedLinearMethod"]


def _bond_method_to_cls(func, obj):
    if hasattr(func, "__self__") or not callable(func):
        return func
    return types.MethodType(func, obj)


def _get_weight_attrs(param: torch.Tensor) -> dict[str, Any]:
    """Snapshot the non-callable / bound-method attrs attached to a weight.

    torchao's ``quantize_`` and the packed-tensor conversion replace the
    ``Parameter`` object, which drops the loader attributes (``weight_loader``,
    ``input_dim`` / ``output_dim``, shard metadata) the engine set on it. We
    record them here and restore them onto the new parameter so weight loading
    still finds what it expects.
    """
    recorded: dict[str, Any] = {}
    for key in param.__dict__:
        if not hasattr(param, key):
            continue
        attr = getattr(param, key)
        if not callable(attr):
            recorded[key] = attr
        elif hasattr(attr, "__self__") and param is attr.__self__:
            recorded[key] = attr.__func__
        else:
            recorded[key] = attr
    return recorded


def _restore_weight_attrs(param: torch.Tensor, recorded: dict[str, Any]) -> None:
    for name, attr in recorded.items():
        if not hasattr(param, name):
            setattr(param, name, _bond_method_to_cls(attr, param))


def _convert_packed(weight: torch.Tensor) -> torch.Tensor:
    """Convert a torchao weight to its hardware-packed form when available.

    torchao >= 0.15 packs some quantized layouts for the current hardware;
    older versions have nothing to do, so this is the identity there.
    """
    try:
        from torchao.prototype.tensor_conversion.api import (
            convert_to_packed_tensor_based_on_current_hardware,
        )
    except Exception:
        return weight
    return convert_to_packed_tensor_based_on_current_hardware(weight)


def torchao_quantize_param_data(param: torch.Tensor, torchao_config: Any) -> Parameter:
    """Quantize a weight tensor in place per ``torchao_config``.

    Builds a throwaway ``Linear`` on the meta device (no real allocation),
    assigns ``param`` as its weight, runs torchao's ``quantize_`` (which may
    swap the module), and returns the resulting quantized weight parameter.
    """
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_

    assert isinstance(torchao_config, AOBaseConfig), f"{torchao_config}"
    with torch.device("meta"):
        # Not a top-level module: quantize_ is in-place and some configs do a
        # module swap, which only non-top-level modules support.
        dummy_linear = torch.nn.Sequential(
            torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
        )
    dummy_linear[0].weight = param
    quantize_(dummy_linear, torchao_config)
    return dummy_linear[0].weight


class TorchAOLinearMethod(LinearMethodBase):
    """Linear method for torchao-quantized weights."""

    def __init__(self, quant_config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # A torchao-serialized checkpoint stores the quantized tensor subclass;
        # pre-quantize the empty parameter so the on-disk tensor loads into a
        # matching structure. (Online quantization of a bf16 checkpoint happens
        # later, in process_weights_after_loading.)
        if self.quant_config.is_checkpoint_torchao_serialized:
            weight = torchao_quantize_param_data(
                weight, self.quant_config.torchao_config
            )

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # layer.weight is a torchao quantized tensor subclass; F.linear
        # dispatches to torchao's dequant/compute kernel.
        return F.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "weight"):
            return

        if self.quant_config.is_checkpoint_torchao_serialized:
            # Already quantized on load; just apply the hardware packing (if
            # any) and preserve the loader attributes across the swap.
            recorded = _get_weight_attrs(layer.weight)
            layer.weight = Parameter(
                _convert_packed(layer.weight),
                requires_grad=layer.weight.requires_grad,
            )
            _restore_weight_attrs(layer.weight, recorded)
            return

        # Online path: the checkpoint is unquantized, so quantize the loaded
        # bf16 weight now per the torchao config.
        recorded = _get_weight_attrs(layer.weight)
        weight = torchao_quantize_param_data(
            layer.weight, self.quant_config.torchao_config
        )
        weight = Parameter(_convert_packed(weight), requires_grad=weight.requires_grad)
        _restore_weight_attrs(weight, recorded)
        layer.register_parameter("weight", weight)
