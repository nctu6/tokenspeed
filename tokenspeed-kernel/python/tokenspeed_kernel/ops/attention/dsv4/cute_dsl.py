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
"""CuTe DSL leaf for the DeepSeek V4 selected attention on SM120."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

try:
    from tokenspeed_kernel.ops.attention.dsv4._cute_dsl import sparse_mma as _sparse_mma
except ImportError:
    _sparse_mma = None

if _sparse_mma is not None:
    from tokenspeed_kernel.ops.attention.dsv4 import triton as _dsv4_triton

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name="cute_dsl_dsv4_mma_prefill",
        solution="cute_dsl",
        capability=_dsv4_triton._MMA_CAPABILITY,
        signatures=_dsv4_triton._MMA_SIGNATURES,
        # A 16-head CTA (4 warps) is 0.7-0.8x of the Triton leaf from 64
        # tokens, so narrower models keep the Triton leaf. The rule follows the
        # model's head count, never the batch, so a row stays bitwise stable.
        traits={**_dsv4_triton._MMA_TRAITS, "num_q_heads_min": frozenset({17})},
        priority=Priority.PERFORMANT + 1,
    )
    def cute_dsl_dsv4_mma_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lens: torch.Tensor,
        attn_sink: torch.Tensor | None,
        softmax_scale: float,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Selected shared-KV attention on tensor cores for DeepSeek V4 geometry.

        Same contract as ``triton_dsv4_mma_prefill``. A tensor layout the
        kernel's vector loads cannot address runs the Triton leaf instead; the
        choice follows the layout alone, so a row's bits never depend on its
        batch.
        """
        if q.dim() != 3 or q.shape[-1] != _sparse_mma.HEAD_DIM:
            raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
        kv_2d = kv.reshape(-1, kv.shape[-1])
        indices_2d = indices.reshape(indices.shape[0], -1)
        output = out if out is not None else torch.empty_like(q)
        if (
            kv_2d.shape[-1] != _sparse_mma.HEAD_DIM
            or indices_2d.stride(-1) != 1
            or not _sparse_mma.supports(q, kv_2d, output, indices_2d)
            or lens.reshape(-1).shape[0] != q.shape[0]
            or indices_2d.shape[0] != q.shape[0]
            or (attn_sink is not None and attn_sink.numel() < q.shape[1])
        ):
            return _dsv4_triton.triton_dsv4_mma_prefill(
                q, kv, indices, lens, attn_sink, softmax_scale, out=out, lse=lse
            )
        sink = None
        if attn_sink is not None:
            sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
        block_h, num_splits = _sparse_mma.plan_launch(
            q.shape[0], q.shape[1], current_platform().sm_count
        )
        _sparse_mma.sparse_mma_attention(
            q,
            kv_2d,
            indices_2d,
            lens.reshape(-1),
            sink,
            output,
            lse,
            softmax_scale=softmax_scale,
            block_h=block_h,
            num_splits=num_splits,
        )
        return output
