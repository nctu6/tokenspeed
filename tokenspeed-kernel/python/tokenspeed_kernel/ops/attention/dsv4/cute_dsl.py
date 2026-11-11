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
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

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
        # At 16 heads a token's gather is the whole job (the 640 rows of a
        # token are read for 16 heads only) and the Triton leaf already reads
        # them at the L2 rate: this leaf is 0.82-1.3x of it by token count, so
        # models of 16 heads or fewer keep the Triton leaf (docs: analysis
        # 9.35). The rule follows the model's head count, never the batch, so a
        # row stays bitwise stable.
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
        block_h, cwarps, num_splits, full_tokens = _sparse_mma.plan_launch(
            q.shape[0], q.shape[1], current_platform().sm_count, indices_2d.shape[1]
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
            cwarps=cwarps,
            num_splits=num_splits,
            full_tokens=full_tokens,
        )
        return output

    @register_kernel(
        "attention",
        "dsv4_decode",
        name="cute_dsl_dsv4_mma_decode",
        solution="cute_dsl",
        capability=_dsv4_triton._MMA_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    swa_kv_cache=dense_tensor_format(torch.uint8),
                )
            }
        ),
        # Every head count: the kernel dequantizes the FP8 page-planar rows
        # while it gathers them (584 bytes read per row instead of a 1 KiB BF16
        # workspace row written and read back), which is 1.05-3.3x faster than
        # the workspace route and 1.13-2.3x faster than the Triton direct
        # reader over 16-128 heads, 1-2048 tokens and widths 128 / 640 / 1152
        # (docs: analysis 9.40). The leaf never depends on the batch.
        traits={
            "head_dim": frozenset({_dsv4_triton.DEEPSEEK_V4_HEAD_DIM}),
            # A tile of the selection belongs to one segment.
            "swa_selected_width_align": frozenset({_sparse_mma.ROWS}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "has_extra_segment": frozenset({False, True}),
            "metadata_dtypes": frozenset({torch.int32, torch.int64}),
            "sinks": frozenset({True}),
            "topk_layout": frozenset({"global_slots"}),
        },
        priority=Priority.PERFORMANT + 1,
    )
    def cute_dsl_dsv4_mma_decode(
        q: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_slots: torch.Tensor,
        swa_lens: torch.Tensor,
        swa_page_size: int,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        extra_kv_cache: torch.Tensor | None = None,
        extra_slots: torch.Tensor | None = None,
        extra_lens: torch.Tensor | None = None,
        extra_page_size: int | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Selected attention that reads the FP8 page-planar rows itself (SM120).

        The result is bitwise that of ``triton_dsv4_decode`` followed by
        ``cute_dsl_dsv4_mma_prefill``, without the BF16 workspace between them.
        A cache layout the kernel's vector loads cannot address runs the
        portable workspace route instead; the choice follows the layout alone,
        so a row's bits never depend on its batch.
        """
        if q.dim() != 3 or q.shape[-1] != _sparse_mma.HEAD_DIM:
            raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
        if q.dtype != torch.bfloat16 or q.stride(-1) != 1:
            raise ValueError("q must be BF16 with a contiguous last dimension")
        tokens = q.shape[0]
        heads = q.shape[1]
        swa = (
            swa_kv_cache,
            "swa_v4",
            _dsv4_triton._selection_rows(swa_slots, tokens),
            swa_lens.reshape(-1),
            int(swa_page_size),
        )
        extra = None
        if extra_kv_cache is not None:
            assert extra_slots is not None
            assert extra_lens is not None
            assert extra_page_size is not None
            extra = (
                extra_kv_cache,
                "swa_v4",
                _dsv4_triton._selection_rows(extra_slots, tokens),
                extra_lens.reshape(-1),
                int(extra_page_size),
            )
        output = out if out is not None else torch.empty_like(q)
        if output.stride(-1) != 1:
            raise ValueError("out must have a contiguous last dimension")
        if attn_sink.numel() < heads:
            raise ValueError("attention sink must provide one value per query head")
        if not _sparse_mma.supports_paged(q, output, swa, extra):
            return _dsv4_triton.triton_dsv4_decode(
                q,
                swa_kv_cache,
                swa_slots,
                swa_lens,
                swa_page_size,
                attn_sink,
                softmax_scale,
                extra_kv_cache,
                extra_slots,
                extra_lens,
                extra_page_size,
                out,
            )
        sink = attn_sink.reshape(-1)[:heads].to(torch.float32).contiguous()
        block_h, cwarps, num_splits, full_tokens = _sparse_mma.plan_paged(
            tokens, heads, current_platform().sm_count, swa, extra
        )
        _sparse_mma.paged_attention(
            q,
            swa,
            extra,
            sink,
            output,
            None,
            softmax_scale=softmax_scale,
            block_h=block_h,
            cwarps=cwarps,
            num_splits=num_splits,
            full_tokens=full_tokens,
        )
        return output
