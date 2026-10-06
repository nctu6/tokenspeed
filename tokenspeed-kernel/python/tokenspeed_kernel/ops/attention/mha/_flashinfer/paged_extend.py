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

"""FlashInfer FA2 paged-extend for large head_dim (Gemma-4 global layers).

Enabled by ``TOKENSPEED_FLASHINFER_FA2_EXTEND`` (default ``off``). Uses only the
public FlashInfer ``BatchPrefillWithPagedKVCacheWrapper`` API with
``backend="fa2"`` forced so Hopper never silently selects FA3 templates that
reject head_dim 512.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)

# Workspace sized for FA2 float split-KV scratch. Allocated once per device so
# it is visible to memory profiling / KV-pool sizing when warmup runs early.
_FLOAT_WORKSPACE_BYTES = 256 * 1024 * 1024

_float_workspace: dict[torch.device, torch.Tensor] = {}
# One wrapper per (device, geometry) so each keeps its own int workspace;
# float workspace is shared (calls are serialized on one stream).
_wrapper_pool: dict[tuple[Any, ...], Any] = {}
_warned_host_meta_fallback = False


def _parse_fa2_extend_mode() -> str:
    raw = os.environ.get("TOKENSPEED_FLASHINFER_FA2_EXTEND", "off")
    mode = (raw or "off").strip().lower()
    if mode in ("0", "false", "no", "off", ""):
        return "off"
    if mode in ("512", "all"):
        return mode
    warnings.warn(
        f"Unknown TOKENSPEED_FLASHINFER_FA2_EXTEND={raw!r}; treating as off "
        "(expected off|512|all).",
        stacklevel=2,
    )
    return "off"


_FA2_EXTEND_MODE = _parse_fa2_extend_mode()


def fa2_extend_mode() -> str:
    """Return the import-time FA2 extend mode (``off`` / ``512`` / ``all``)."""
    return _FA2_EXTEND_MODE


def _head_dims_for_mode(mode: str) -> frozenset[int] | None:
    if mode == "512":
        return frozenset({512})
    if mode == "all":
        return frozenset({64, 128, 256, 512})
    return None


def _arch_supports_fa2_extend(platform) -> bool:
    """Hopper (9.x) and sm12x only; leave sm100 on trtllm/FA4."""
    if not platform.is_nvidia:
        return False
    major = platform.arch_version.major
    return major == 9 or major == 12


@dataclass(frozen=True)
class PagedExtendHostMeta:
    """Host-side mirrors for sync-free FlashInfer ``plan()``.

    ``plan_cache`` lives on ``MHAExtendMetadata`` (one forward) and is shared
    across layers of the same geometry so a step plans at most twice on Gemma-4.
    """

    cu_seqlens_q_cpu: list[int]
    cache_seqlens_cpu: list[int]
    plan_cache: dict


def ensure_fa2_extend_workspace(device: torch.device | str | int) -> torch.Tensor:
    """Pre-allocate the shared float workspace on ``device`` (idempotent)."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"FA2 extend workspace requires a CUDA device, got {device}")
    ws = _float_workspace.get(device)
    if ws is None:
        ws = torch.zeros(_FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _float_workspace[device] = ws
        logger.info(
            "FlashInfer FA2 extend: allocated %.0f MiB float workspace on %s",
            _FLOAT_WORKSPACE_BYTES / (1024 * 1024),
            device,
        )
    return ws


def _geometry_key(
    *,
    device: torch.device,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    causal: bool,
    window_left: int,
    logit_cap: float,
    scale: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> tuple[Any, ...]:
    return (
        device,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal,
        window_left,
        float(logit_cap),
        float(scale),
        q_dtype,
        kv_dtype,
    )


def _get_wrapper(device: torch.device, geometry_key: tuple[Any, ...]):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    wrapper = _wrapper_pool.get(geometry_key)
    if wrapper is None:
        float_ws = ensure_fa2_extend_workspace(device)
        # Force fa2: on Hopper ``auto`` would pick FA3 templates that reject 512.
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_ws, kv_layout="NHD", backend="fa2"
        )
        _wrapper_pool[geometry_key] = wrapper
    return wrapper


def _build_paged_plan_tensors(
    page_table: torch.Tensor,
    cache_seqlens_cpu: list[int],
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build FlashInfer KV page metadata without host←device sync on lengths.

    ``paged_kv_indices`` is gathered on-device from ``page_table``; indptr and
    last_page_len are built from the CPU length list.
    """
    batch = len(cache_seqlens_cpu)
    num_pages = [(int(s) + page_size - 1) // page_size for s in cache_seqlens_cpu]
    last_page_len = [
        (((int(s) - 1) % page_size) + 1) if int(s) > 0 else 0
        for s in cache_seqlens_cpu
    ]
    indptr = [0]
    for n in num_pages:
        indptr.append(indptr[-1] + n)

    pieces: list[torch.Tensor] = []
    for b, n in enumerate(num_pages):
        if n > 0:
            pieces.append(page_table[b, :n].to(dtype=torch.int32))
    if pieces:
        indices = torch.cat(pieces, dim=0)
    else:
        indices = torch.empty(0, dtype=torch.int32, device=page_table.device)

    kv_indptr = torch.tensor(indptr, dtype=torch.int32, device="cpu")
    last_page = torch.tensor(last_page_len, dtype=torch.int32, device="cpu")
    if batch != page_table.shape[0]:
        raise ValueError(
            f"cache_seqlens_cpu batch {batch} != page_table batch {page_table.shape[0]}"
        )
    return kv_indptr, indices, last_page


def _host_meta_or_fallback(
    host_meta: PagedExtendHostMeta | dict | None,
    cu_seqlens_q: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> PagedExtendHostMeta:
    global _warned_host_meta_fallback
    if host_meta is not None:
        if isinstance(host_meta, dict):
            return PagedExtendHostMeta(
                cu_seqlens_q_cpu=list(host_meta["cu_seqlens_q_cpu"]),
                cache_seqlens_cpu=list(host_meta["cache_seqlens_cpu"]),
                plan_cache=host_meta["plan_cache"],
            )
        return host_meta

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "flashinfer_fa2_mha_extend_with_kvcache requires host_meta during "
            "CUDA graph capture (D2H fallback is not capture-safe)."
        )
    if not _warned_host_meta_fallback:
        logger.warning(
            "flashinfer_fa2_mha_extend_with_kvcache: host_meta missing; "
            "falling back to a one-shot D2H of cu_seqlens_q/cache_seqlens "
            "(pass MHAExtendMetadata host mirrors to avoid sync)."
        )
        _warned_host_meta_fallback = True
    return PagedExtendHostMeta(
        cu_seqlens_q_cpu=[int(x) for x in cu_seqlens_q.detach().cpu().tolist()],
        cache_seqlens_cpu=[int(x) for x in cache_seqlens.detach().cpu().tolist()],
        plan_cache={},
    )


def _plan_and_run(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    host_meta: PagedExtendHostMeta,
    is_causal: bool,
    window_left: int,
    logit_cap: float,
    softmax_scale: float,
) -> torch.Tensor:
    page_size = int(k_cache.shape[1])
    num_qo_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    head_dim = int(q.shape[-1])
    geometry = _geometry_key(
        device=q.device,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        causal=is_causal,
        window_left=window_left,
        logit_cap=logit_cap,
        scale=softmax_scale,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
    )

    planned = host_meta.plan_cache.get(geometry)
    if planned is None:
        wrapper = _get_wrapper(q.device, geometry)
        qo_indptr = torch.tensor(
            host_meta.cu_seqlens_q_cpu, dtype=torch.int32, device="cpu"
        )
        kv_indptr, indices, last_page = _build_paged_plan_tensors(
            page_table, host_meta.cache_seqlens_cpu, page_size
        )
        soft_cap = float(logit_cap) if logit_cap and logit_cap > 0 else None
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            indices,
            last_page,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=is_causal,
            sm_scale=softmax_scale,
            window_left=window_left,
            logits_soft_cap=soft_cap,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            non_blocking=True,
        )
        host_meta.plan_cache[geometry] = wrapper
        planned = wrapper
    return planned.run(q, (k_cache, v_cache))


def warmup_fa2_extend(
    device: torch.device | str | int,
    *,
    head_dims: tuple[int, ...] | None = None,
    page_size: int = 64,
    num_qo_heads: int = 32,
    num_kv_heads: int = 4,
) -> None:
    """Allocate workspace and JIT-compile FA2 kernels used at runtime.

    Call during server warmup (before KV-pool sizing / graph capture). No-op
    when the feature is off or the arch is unsupported.
    """
    if _FA2_EXTEND_MODE == "off":
        return
    platform = current_platform()
    if not _arch_supports_fa2_extend(platform):
        return
    device = torch.device(device)
    ensure_fa2_extend_workspace(device)
    dims = head_dims
    if dims is None:
        declared = _head_dims_for_mode(_FA2_EXTEND_MODE)
        dims = tuple(sorted(declared)) if declared else (512,)
    for head_dim in dims:
        # Tiny causal extend to force JIT of the FA2 template.
        total_kv = page_size
        q = torch.zeros(
            (1, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        k_cache = torch.zeros(
            (1, page_size, num_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        page_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        host_meta = PagedExtendHostMeta(
            cu_seqlens_q_cpu=[0, 1],
            cache_seqlens_cpu=[total_kv],
            plan_cache={},
        )
        _plan_and_run(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            host_meta=host_meta,
            is_causal=True,
            window_left=-1,
            logit_cap=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
        )
    torch.cuda.synchronize(device)


def _register_fa2_paged_extend() -> None:
    head_dims = _head_dims_for_mode(_FA2_EXTEND_MODE)
    if head_dims is None:
        return
    platform = current_platform()
    if not _arch_supports_fa2_extend(platform):
        return

    arch = platform.arch_version
    if arch.major == 9:
        capability = CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        )
    else:
        capability = CapabilityRequirement(
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 9),
            vendors=frozenset({"nvidia"}),
        )

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_fa2_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=capability,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.PERFORMANT,
        traits={
            "head_dim": head_dims,
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            # First version: keep the trait set narrow (promised combinations).
            "logit_cap": frozenset({False}),
            "sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "page_size": frozenset({1, 16, 32, 64, 128}),
        },
    )
    def flashinfer_fa2_mha_extend_with_kvcache(
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
        host_meta: PagedExtendHostMeta | dict | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
        del q_scale, k_scale, v_scale
        if return_lse:
            raise NotImplementedError(
                "flashinfer_fa2_mha_extend_with_kvcache does not support return_lse"
            )
        if sinks is not None:
            raise NotImplementedError(
                "flashinfer_fa2_mha_extend_with_kvcache does not support sinks"
            )
        if logit_cap and logit_cap != 0.0:
            raise NotImplementedError(
                "flashinfer_fa2_mha_extend_with_kvcache does not support logit_cap"
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        meta = _host_meta_or_fallback(host_meta, cu_seqlens_q, cache_seqlens)
        return _plan_and_run(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            host_meta=meta,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=0.0,
            softmax_scale=float(softmax_scale),
        )


_register_fa2_paged_extend()

__all__ = [
    "PagedExtendHostMeta",
    "ensure_fa2_extend_workspace",
    "fa2_extend_mode",
    "warmup_fa2_extend",
]
