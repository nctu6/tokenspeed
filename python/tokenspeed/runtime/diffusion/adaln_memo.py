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

"""Exact memoization of the MiniMax-H3 per-block AdaLN projections.

Every DiT block owns an AdaLN projection ``2688 -> 6 * 5376 * 3`` (260 M
params, 13 B for the 50 blocks: ~26 GB of the 62 GB DiT). Its input is the
timestep embedding, which depends only on the distinct timestep values of the
step -- fixed by the sampling schedule, not by the prompt or the canvas. So
the projection output is a pure function of those values and can be cached.

This module keeps the projection weights in pinned host memory and caches the
GPU output per ``(timestep values, block)``. A hit is a table read; a miss
(the first request on a new schedule) streams the block's weights to the GPU
and runs the very same ``F.linear`` on the same input shape, so the result is
the one the resident layer would produce. The cache is exact, not a cross-step
approximation: different timestep values never share an entry.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AdaLNMemoTable", "MemoizedAdaLNModulation", "install_adaln_memo"]


class AdaLNMemoTable:
    """Per-device cache of AdaLN projection outputs, LRU-bounded by bytes.

    ``set_step_key`` is called once per DiT forward (from a forward pre-hook on
    the transformer) with the step's distinct timestep values; every block then
    looks up ``(step key, block index)``.
    """

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.max_bytes = int(max_bytes)
        self.entries: OrderedDict[tuple[bytes, int], torch.Tensor] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.step_key: bytes | None = None

    @staticmethod
    def key_for_timestep(timestep: torch.Tensor) -> bytes:
        # The exact float32 bit pattern plus the shape: equal keys <=> equal input.
        values = timestep.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return repr(tuple(values.shape)).encode() + values.numpy().tobytes()

    def set_step_key(self, timestep: torch.Tensor) -> None:
        self.step_key = self.key_for_timestep(timestep)

    def lookup(self, block_index: int) -> torch.Tensor | None:
        if self.step_key is None:
            raise RuntimeError(
                "AdaLN memo used outside a transformer forward (no step key set)"
            )
        key = (self.step_key, block_index)
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return value

    def store(self, block_index: int, value: torch.Tensor) -> None:
        key = (self.step_key, block_index)
        size = value.numel() * value.element_size()
        self.entries[key] = value
        self.bytes += size
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()


class MemoizedAdaLNModulation(nn.Module):
    """Drop-in for ``MiniMaxH3AdaLayerNormModulation`` backed by an ``AdaLNMemoTable``.

    The weights are plain tensor attributes in pinned host memory (not
    parameters), so moving the DiT to a GPU leaves them on the host.
    """

    def __init__(self, source: nn.Module, block_index: int, table: AdaLNMemoTable):
        super().__init__()
        self.hidden_size: int = int(source.hidden_size)
        self.block_index = int(block_index)
        self.table = table
        weight = source.linear.weight.detach().to("cpu")
        bias = source.linear.bias.detach().to("cpu")
        self.weight_dtype: torch.dtype = weight.dtype
        pin = torch.cuda.is_available()
        self.weight_host: torch.Tensor = weight.pin_memory() if pin else weight
        self.bias_host: torch.Tensor = bias.pin_memory() if pin else bias

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.table.lookup(self.block_index)
        if out is None:
            weight = self.weight_host.to(temb.device, non_blocking=True)
            bias = self.bias_host.to(temb.device, non_blocking=True)
            # Same op, dtype handling, and shape as the Diffusers module:
            # activate at temb's precision, cast to the projection dtype after.
            out = F.linear(F.silu(temb).to(self.weight_dtype), weight, bias)
            self.table.store(self.block_index, out)
        # Views of the cached tensor; the block only reads them.
        return out.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def install_adaln_memo(transformer: nn.Module, table: AdaLNMemoTable) -> int:
    """Swap every block's ``adaln_proj`` for a memoized one; return the count.

    Call while the transformer is still on the host, before moving it to the
    GPU, so the 26 GB of projection weights never land in device memory.
    """
    blocks = transformer.transformer_blocks
    for index, block in enumerate(blocks):
        block.adaln_proj = MemoizedAdaLNModulation(block.adaln_proj, index, table)

    def _set_key(_module, _args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None:
            raise RuntimeError(
                "MiniMax-H3 transformer called without a `timestep` keyword; "
                "the AdaLN memo needs it to key the step"
            )
        table.set_step_key(timestep)

    transformer.register_forward_pre_hook(_set_key, with_kwargs=True)
    return len(blocks)
