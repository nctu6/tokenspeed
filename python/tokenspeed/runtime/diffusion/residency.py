# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""Resident placement of the MiniMax-H3 conditioner across GPUs.

The Qwen3-VL conditioner (64 decoder layers, ~63 GB) and the DiT (~62 GB)
cannot share one 96 GB GPU, which is what forced per-request CPU<->GPU swaps.
Two exact reductions make everything resident:

* H3 conditions on ``hidden_states[text_encoder_layer]`` (50) and never on the
  final norm or the LM head, so only the first ``text_encoder_layer + 1``
  decoder layers matter. (A stack of exactly ``text_encoder_layer`` layers is
  not enough: its last ``hidden_states`` entry is the post-norm output.)
* The remaining layers are split in contiguous ranges across the GPUs with
  accelerate's ``dispatch_model``; activations hop devices once per range.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

__all__ = ["truncate_text_encoder", "split_text_encoder"]


def _language_model(text_encoder: nn.Module) -> nn.Module:
    return text_encoder.model.language_model


def truncate_text_encoder(text_encoder: nn.Module, hidden_state_index: int) -> int:
    """Keep the decoder layers ``hidden_states[hidden_state_index]`` depends on.

    Returns the number of layers kept. Exact: ``hidden_states[i]`` for
    ``i <= hidden_state_index`` is unchanged.
    """
    keep = int(hidden_state_index) + 1
    lm = _language_model(text_encoder)
    total = len(lm.layers)
    if keep > total:
        raise ValueError(
            f"hidden_states[{hidden_state_index}] needs {keep} decoder layers, "
            f"the text encoder has {total}"
        )
    if keep == total:
        return total
    lm.layers = nn.ModuleList(list(lm.layers)[:keep])
    text_cfg = text_encoder.config.text_config
    text_cfg.num_hidden_layers = keep
    layer_types = getattr(text_cfg, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) > keep:
        text_cfg.layer_types = layer_types[:keep]
    lm.config.num_hidden_layers = keep
    return keep


def text_encoder_device_map(
    text_encoder: nn.Module, devices: list[str]
) -> dict[str, Any]:
    """Contiguous decoder-layer ranges per device; embeddings/vision on the first."""
    if not devices:
        raise ValueError("devices must not be empty")
    lm = _language_model(text_encoder)
    num_layers = len(lm.layers)
    first, last = devices[0], devices[-1]
    device_map: dict[str, Any] = {}
    for name, _ in text_encoder.model.named_children():
        if name != "language_model":
            device_map[f"model.{name}"] = first
    for name, _ in lm.named_children():
        if name == "layers":
            continue
        # Final norm sits with the last layer; embeddings / rotary with the first.
        device_map[f"model.language_model.{name}"] = last if name == "norm" else first
    per = -(-num_layers // len(devices))
    for index in range(num_layers):
        device_map[f"model.language_model.layers.{index}"] = devices[index // per]
    for name, _ in text_encoder.named_children():
        if name != "model":
            device_map[name] = last
    # Every parameter / buffer must be covered by exactly one prefix.
    for pname, _ in list(text_encoder.named_parameters()) + list(
        text_encoder.named_buffers()
    ):
        if not any(pname == k or pname.startswith(k + ".") for k in device_map):
            raise RuntimeError(f"text encoder device map does not cover {pname}")
    return device_map


def split_text_encoder(text_encoder: nn.Module, devices: list[str]) -> dict[str, Any]:
    """Place the (host-resident) text encoder across ``devices``; return the map."""
    from accelerate import dispatch_model

    device_map = text_encoder_device_map(text_encoder, devices)
    dispatch_model(text_encoder, device_map=device_map, main_device=devices[0])
    return device_map
