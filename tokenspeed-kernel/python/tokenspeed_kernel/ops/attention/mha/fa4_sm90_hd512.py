# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SM90 paged extend for head_dim 512 (gemma-4 global layers).

FA3 and the upstream CuTe-DSL SM90 forward stop at head_dim 256, which left
these layers on the FlashInfer FA2 template (~170 TFLOP/s). The vendored
CuTe-DSL SM90 forward in ``tokenspeed_kernel.thirdparty.fa4_sm90`` covers 512
with wgmma/TMA and runs ~2x faster. It is registered above FA2, so automatic
selection prefers it and FA2 remains the fallback.
``TOKENSPEED_FA4_SM90_HD512=off`` disables it; ``override=`` / an explicit
attention backend still pick any kernel by name or solution.
"""

from __future__ import annotations

import logging
import math
import os

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    raw = (os.environ.get("TOKENSPEED_FA4_SM90_HD512", "on") or "on").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _available() -> bool:
    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version.major == 9):
        return False
    try:
        import tokenspeed_kernel.thirdparty.fa4_sm90.interface  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — missing cutlass-dsl / quack
        logger.info("fa4_sm90 hd512 extend unavailable: %s", exc)
        return False
    return True


def fa4_sm90_hd512_extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    *,
    is_causal: bool,
    window_left: int,
    softmax_scale: float,
) -> torch.Tensor:
    from tokenspeed_kernel.thirdparty.fa4_sm90.interface import flash_attn_varlen_func

    window_size = (window_left, 0) if window_left >= 0 else (None, None)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
    )
    return out[0] if isinstance(out, tuple) else out


def warmup_fa4_sm90_hd512(
    device, *, num_qo_heads: int = 32, num_kv_heads: int = 4, page_size: int = 64
) -> None:
    """JIT-compile the head_dim 512 causal kernel before serving."""
    if not (_enabled() and _available()):
        return
    device = torch.device(device)
    q = torch.zeros((1, num_qo_heads, 512), dtype=torch.bfloat16, device=device)
    k = torch.zeros(
        (1, page_size, num_kv_heads, 512), dtype=torch.bfloat16, device=device
    )
    fa4_sm90_hd512_extend(
        q,
        k,
        torch.zeros_like(k),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.ones((1,), dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        1,
        is_causal=True,
        window_left=-1,
        softmax_scale=1.0 / math.sqrt(512),
    )
    torch.cuda.synchronize(device)


def _register() -> None:
    if not (_enabled() and _available()):
        return

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="fa4_sm90_hd512_mha_extend_with_kvcache",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        # Above the FA2 paged extend (PERFORMANT) it replaces for head_dim 512.
        priority=Priority.PERFORMANT + 2,
        traits={
            "head_dim": frozenset({512}),
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "logit_cap": frozenset({False}),
            "sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "page_size": frozenset({16, 32, 64, 128}),
        },
    )
    def fa4_sm90_hd512_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_causal: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
        host_meta=None,
    ) -> torch.Tensor:
        del (
            cu_seqlens_kv,
            max_seqlen_k,
            q_scale,
            k_scale,
            v_scale,
            enable_pdl,
            host_meta,
        )
        if return_lse or sinks is not None or (logit_cap and logit_cap != 0.0):
            raise NotImplementedError(
                "fa4_sm90_hd512 extend supports neither lse, sinks nor logit_cap"
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        return fa4_sm90_hd512_extend(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            max_seqlen_q,
            is_causal=is_causal,
            window_left=window_left,
            softmax_scale=float(softmax_scale),
        )


_register()

__all__ = ["fa4_sm90_hd512_extend", "warmup_fa4_sm90_hd512"]
