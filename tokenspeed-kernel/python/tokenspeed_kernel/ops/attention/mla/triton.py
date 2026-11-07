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


import torch
from tokenspeed_kernel.ops.attention.mla._triton.decode import (
    _normalize_kv_cache,
    _triton_mla_decode_with_kvcache_impl,
)
from tokenspeed_kernel.ops.attention.mla._triton.decode_mma import (
    mla_mma_decode,
    plan_launch,
)
from tokenspeed_kernel.ops.attention.mla._triton.page_table import *  # noqa: F403
from tokenspeed_kernel.ops.attention.mla._triton.prefill import (
    _triton_mla_prefill_impl,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_PORTABLE_DTYPES = frozenset({torch.float16, torch.bfloat16}) | _FP8_DTYPES
_PORTABLE_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))


@register_kernel(
    "attention",
    "mla_prefill",
    name="triton_mla_prefill",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(("q", "k", "v"), "dense", _PORTABLE_DTYPES),
    priority=Priority.PORTABLE,
    features={"sliding_window", "query_start_offsets"},
    traits={
        "is_causal": frozenset({False, True}),
        "logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
)
def triton_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
    window_left: int = -1,
    query_start_offsets: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _triton_mla_prefill_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
        seq_lens_kv=seq_lens_kv,
        window_left=window_left,
        query_start_offsets=query_start_offsets,
    )


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="triton_mla_decode_with_kvcache",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(("q", "kv_cache"), "dense", _PORTABLE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "q_len": frozenset({1}),
        "logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
    },
)
def triton_mla_decode_with_kvcache(
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
    return _triton_mla_decode_with_kvcache_impl(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
        window_left=window_left,
        noncausal_block_size=noncausal_block_size,
    )


# Query rows per request the tensor-core kernel takes on the query axis.
_MMA_Q_LENS = frozenset(range(1, 9))
_MMA_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float8_e4m3fn})


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="triton_mla_mma_decode_with_kvcache",
    solution="triton",
    # Measured on SM120, where the SM100-only tokenspeed_mla decode is not
    # offered and the portable kernel runs every head on CUDA cores.
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 9),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(("q", "kv_cache"), "dense", _MMA_DTYPES),
    priority=Priority.PERFORMANT,
    traits={
        # One query row per request is ordinary decode (or one flattened row of
        # a proposal block); 2 to 8 are a causal verify block or a non-causal
        # proposal block on the query axis.
        "q_len": _MMA_Q_LENS,
        "noncausal_block_size": _MMA_Q_LENS,
        "block_on_query_axis": frozenset({False, True}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "kv_lora_rank": frozenset({512}),
        "qk_rope_head_dim": frozenset({64}),
        "page_size": frozenset({16, 32, 64, 128}),
    },
)
def triton_mla_mma_decode_with_kvcache(
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
    """Tensor-core MLA decode; see ``_triton/decode_mma.py`` for the masks."""
    if logit_cap != 0.0:
        raise ValueError("triton_mla_mma_decode_with_kvcache has no logit cap")
    if q.dim() != 4 or q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"q must have shape [B, q_len, H, {kv_lora_rank + qk_rope_head_dim}], "
            f"got {tuple(q.shape)}"
        )
    batch, q_len, heads, _ = q.shape
    if q_len not in _MMA_Q_LENS:
        raise ValueError(f"q_len must be in 1..8, got {q_len}")
    if q_len > 1 and noncausal_block_size not in (1, q_len):
        raise ValueError(
            "query rows of one request are either a causal block "
            f"(noncausal_block_size=1) or one proposal block; got q_len={q_len}, "
            f"noncausal_block_size={noncausal_block_size}"
        )
    if window_left < -1 or noncausal_block_size <= 0:
        raise ValueError(
            f"window_left={window_left} and noncausal_block_size="
            f"{noncausal_block_size} must be >= -1 and positive"
        )
    if 0 <= window_left < noncausal_block_size - 1:
        raise ValueError(
            "window_left must cover the complete non-causal block; got "
            f"window_left={window_left}, block_size={noncausal_block_size}"
        )
    if q.stride(-1) != 1:
        raise ValueError("q must have a contiguous last dimension")
    kv_cache = _normalize_kv_cache(kv_cache)
    if kv_cache.shape[2] != 1 or kv_cache.shape[-1] != q.shape[-1]:
        raise ValueError(f"unsupported MLA kv_cache shape {tuple(kv_cache.shape)}")
    if kv_cache.stride(-1) != 1:
        raise ValueError("kv_cache must have a contiguous last dimension")
    if kv_cache.dtype != q.dtype:
        raise TypeError(f"q and kv_cache dtypes differ: {q.dtype} vs {kv_cache.dtype}")
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        out = torch.empty(
            (batch, q_len, heads, kv_lora_rank), dtype=out_dtype, device=q.device
        )
    elif out.shape != (batch, q_len, heads, kv_lora_rank) or out.stride(-1) != 1:
        raise ValueError(f"unsupported out {tuple(out.shape)} {out.stride()}")
    lse = (
        torch.empty((batch, q_len, heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    block_m, num_splits = plan_launch(batch, q_len * heads, current_platform().sm_count)
    mla_mma_decode(
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
    return (out, lse) if return_lse else out
