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
"""CuTe DSL leaf for the tensor-core MLA decode on SM120."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel.ops.attention.mla._cute_dsl import decode_mma as _decode_mma
except ImportError:
    _decode_mma = None

if _decode_mma is not None:
    from tokenspeed_kernel.ops.attention.mla import triton as _mla_triton

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="cute_dsl_mla_mma_decode_with_kvcache",
        solution="cute_dsl",
        capability=_mla_triton._MMA_CAPABILITY,
        # FP16 caches keep the Triton leaf.
        signatures=format_signatures(
            ("q", "kv_cache"), "dense", {torch.bfloat16, torch.float8_e4m3fn}
        ),
        priority=Priority.PERFORMANT + 1,
        traits=_mla_triton._MMA_TRAITS,
    )
    def cute_dsl_mla_mma_decode_with_kvcache(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        *,
        logit_cap: float = 0.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
        window_left: int = -1,
        noncausal_block_size: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Tensor-core MLA decode; same contract and masks as the Triton leaf.

        A tensor layout the kernel's vector loads cannot address runs the
        Triton launch instead; the choice follows layouts and dtypes alone.
        """

        def launch(q, kv_cache, out, lse):
            block_m, num_splits = _decode_mma.plan_launch(
                q.shape[0], q.shape[1] * q.shape[2], current_platform().sm_count
            )
            if _decode_mma.supports(q, kv_cache, out, page_table, cache_seqlens, lse):
                _decode_mma.mla_mma_decode(
                    q,
                    kv_cache,
                    out,
                    page_table,
                    cache_seqlens,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    softmax_scale,
                    lse=lse,
                    window_left=window_left,
                    noncausal_block_size=noncausal_block_size,
                    block_m=block_m,
                    num_splits=num_splits,
                )
                return
            block_m, num_splits = _mla_triton.plan_launch(
                q.shape[0], q.shape[1] * q.shape[2], current_platform().sm_count
            )
            _mla_triton.mla_mma_decode(
                q,
                kv_cache,
                out,
                page_table,
                cache_seqlens,
                kv_lora_rank,
                qk_rope_head_dim,
                softmax_scale,
                lse=lse,
                window_left=window_left,
                noncausal_block_size=noncausal_block_size,
                block_m=block_m,
                num_splits=num_splits,
            )

        return _mla_triton._mma_decode_checked(
            q,
            kv_cache,
            page_table,
            cache_seqlens,
            kv_lora_rank,
            qk_rope_head_dim,
            leaf="cute_dsl_mla_mma_decode_with_kvcache",
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
            window_left=window_left,
            noncausal_block_size=noncausal_block_size,
            launch=launch,
        )
