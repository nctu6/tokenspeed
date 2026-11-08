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

"""CuTe DSL row top-k kernels for the DSA indexer.

Two selectors live here. The first wraps the vendored TensorRT-LLM single-pass
multi-CTA (cluster) radix runners; the second (:func:`deepselect_local_topk`)
wraps the in-tree DeepSelect kernel, which fits workstation Blackwell's
99 KiB of shared memory. The notes below describe the first one.

Wraps the vendored TensorRT-LLM CuTe DSL radix runners as a length-aware per-row
top-k over a ``[num_rows, num_cols]`` indexer-logits matrix, matching the
``deterministic_decode_topk(..., lengths=seq_lens, q_len_per_req=next_n)`` contract
it replaces (returned int32 column indices are the "local offsets" that
``local_topk_to_global_slots`` maps to KV slots). Cluster-first: falls back to the
non-cluster runner when the problem exceeds cluster capacity. NVIDIA Blackwell
(sm_100+) only; gate on :func:`has_cute_dsl_decode_topk`.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "cute_dsl_decode_topk",
    "deepselect_local_topk",
    "deepselect_supports",
    "has_cute_dsl_decode_topk",
]


def _ts_supported_arch() -> bool:
    """Gate: the CuTe DSL multi-CTA / cluster radix top-k needs NVIDIA sm_100+.

    Returns False if platform detection raises (e.g. CPU-only host) so callers
    fall back transparently.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    sm = p.arch_version.major * 10 + p.arch_version.minor
    return sm >= 100


_CUTE_DSL_TOPK_AVAILABLE = False
_ClusterRunner = None
_BaseRunner = None

if _ts_supported_arch():
    try:
        import cutlass  # noqa: F401  (import probe: kernels JIT via cutlass-dsl)
        import cutlass.cute  # noqa: F401
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner as _ClusterRunner,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTARunner as _BaseRunner,
        )

        _CUTE_DSL_TOPK_AVAILABLE = True
    except ImportError:
        _CUTE_DSL_TOPK_AVAILABLE = False


def has_cute_dsl_decode_topk() -> bool:
    """Whether the CuTe DSL decode top-k kernel can run on this platform."""
    return _CUTE_DSL_TOPK_AVAILABLE


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cute_dsl_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    *,
    next_n: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Length-aware per-row top-k over a dense indexer-logits matrix.

    Args:
        logits: 2-D row-major contiguous CUDA tensor ``[num_rows, num_cols]``
            (``num_rows = num_reqs * next_n``), fp16/bf16/fp32. No pre-masking
            needed; the causal window is derived in-kernel from seq_lens/next_n.
        seq_lens: Per-request candidate length, int32 ``[num_reqs]``. Row ``r``'s
            window is ``seq_lens[r // next_n] - next_n + (r % next_n) + 1`` cols.
        topk: Candidates to select per row (1..2048).
        next_n: Query rows per request (speculative ``q_len_per_req``); 1 for decode.
        out: Optional int32 ``[num_rows, topk]`` buffer, written in place and returned.

    Returns:
        int32 ``[num_rows, topk]`` of per-row selected column indices (unsorted).
        Rows with window < topk have surplus slots filled by the radix collection;
        track valid count separately (``min(window, topk)``), as with
        ``deterministic_decode_topk``.

    Raises:
        RuntimeError: if the kernel is unavailable on this platform.
    """
    if not _CUTE_DSL_TOPK_AVAILABLE:
        raise RuntimeError(
            "cute_dsl_decode_topk is unavailable on this platform "
            "(requires NVIDIA Blackwell sm_100+ with nvidia-cutlass-dsl)."
        )
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2-D [num_rows, num_cols], got {tuple(logits.shape)}"
        )
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"logits dtype must be one of {_SUPPORTED_DTYPES}, got {logits.dtype}"
        )
    num_rows = logits.shape[0]
    topk = int(topk)
    next_n = int(next_n)

    logits = logits.contiguous()
    seq_lens = seq_lens.to(device=logits.device, dtype=torch.int32).contiguous()

    if out is not None:
        if out.shape != (num_rows, topk):
            raise ValueError(
                f"out must have shape {(num_rows, topk)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out dtype must be int32, got {out.dtype}")
        out = out.contiguous()

    # Cluster-first. Re-zero the shared ``row_states`` scratch every call: the
    # kernels self-clean it, but that reuse races under tight CUDA-graph scheduling.
    _ClusterRunner._row_states_initialized = False
    indices, _ = _ClusterRunner.forward(
        logits,
        seq_lens,
        topk,
        next_n,
        return_val=False,
        output_indices=out,
    )
    if indices is None:
        # Exceeds cluster capacity: fall back to the non-cluster runner.
        _BaseRunner._row_states_initialized = False
        indices, _ = _BaseRunner.forward(
            logits,
            seq_lens,
            topk,
            next_n,
            return_val=False,
            output_indices=out,
        )
    return indices


# ---------------------------------------------------------------------------
# In-tree DeepSelect row top-k
# ---------------------------------------------------------------------------

# Where it replaces a slower selector: SM120 has no TRT-LLM prefill top-k image
# (the Triton radix select runs there) and the TRT-LLM decode runners fall back
# to their non-cluster kernel on large batches of long rows.
_DEEPSELECT_AVAILABLE = False
_MAX_GRID_ROWS = 65535
_row_select = None
try:
    _platform = current_platform()
    _deepselect_arch = _platform.is_nvidia and _platform.arch_version.major == 12
except Exception:
    _deepselect_arch = False
if _deepselect_arch:
    try:
        from tokenspeed_kernel.ops.attention.dsa._cute_dsl import (
            deep_select as _row_select,
        )

        _DEEPSELECT_AVAILABLE = True
    except ImportError:
        _DEEPSELECT_AVAILABLE = False


def deepselect_supports(logits: torch.Tensor, topk: int) -> bool:
    """Whether DeepSelect can select ``topk`` entries per row of ``logits``."""
    return (
        _DEEPSELECT_AVAILABLE
        and 0 < topk <= _row_select.MAX_TOPK
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.is_cuda
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.stride(1) == 1
        and logits.stride(0) % _row_select.SCORE_ALIGNMENT == 0
        and logits.stride(0) >= logits.shape[1]
        and logits.data_ptr() % (_row_select.SCORE_ALIGNMENT * 4) == 0
    )


def deepselect_local_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Per-row top-k column offsets over the first ``lengths[r]`` columns.

    Args:
        logits: FP32 ``[rows, cols]`` scores with unit column stride, a row
            stride that is a multiple of 4 elements and a 16-byte aligned base
            (the DeepGEMM logits allocation satisfies this).
        lengths: integer ``[rows]`` valid column count of each row; columns past
            it are never read.
        topk: Offsets to select per row, at most 2048.
        out: Contiguous int32 ``[rows, topk]`` destination.

    Returns:
        ``out``: unsorted local column offsets. A row shorter than ``topk``
        holds ``0 .. length - 1`` followed by ``-1``. Ties at the k-th score
        resolve arbitrarily but deterministically.
    """
    if not _DEEPSELECT_AVAILABLE:
        raise RuntimeError("DeepSelect row top-k is unavailable on this platform")
    # Compiles on first use, so run it once eagerly before CUDA-graph capture.
    capacity = _row_select.capacity_for(topk)
    lengths = lengths.to(torch.int32).contiguous()
    # One CUDA grid row per score row, at most 65535 of them per launch.
    for begin in range(0, logits.shape[0], _MAX_GRID_ROWS):
        end = min(begin + _MAX_GRID_ROWS, logits.shape[0])
        cluster_size = _row_select.choose_cluster_size(
            end - begin, logits.shape[1], capacity, current_platform().sm_count
        )
        _row_select.deepselect_topk(
            logits[begin:end],
            lengths[begin:end],
            topk,
            capacity=capacity,
            cluster_size=cluster_size,
            indices=out[begin:end],
        )
    return out


# ---------------------------------------------------------------------------
# Tensor-core sparse attention over the dense latent cache (SM120)
# ---------------------------------------------------------------------------

# The CuTe DSL cores take the contract, traits and features of the Triton mma
# leaves (``triton_dsa_mma_*``) and rank above them on SM120.
try:
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl import sparse_mma as _sparse_mma
except ImportError:
    _sparse_mma = None

if _sparse_mma is not None:
    from tokenspeed_kernel.ops.attention.dsa import triton as _dsa_triton
    from tokenspeed_kernel.ops.attention.dsa._triton import (
        sparse_mma as _triton_sparse_mma,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel

    def _run_cute_dsa_mma(
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        k_scale: float,
        out: torch.Tensor | None,
        return_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """CuTe DSL sparse attention; mixed query/cache dtypes run the Triton core.

        The kernel is compiled for an E4M3 query over an E4M3 cache and a BF16
        query over a BF16 cache, the pairs the DSA backend sends. Any other
        pair keeps the Triton mma core, which handles it under the same
        contract; the choice depends on the dtypes alone, so a row's bits still
        do not depend on its batch.
        """
        if kv_cache.dtype != q.dtype:
            return _dsa_triton._run_dsa_mma(
                q=q,
                kv_cache=kv_cache,
                topk_slots=topk_slots,
                topk_lens=topk_lens,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                softmax_scale=softmax_scale,
                k_scale=k_scale,
                out=out,
                return_lse=return_lse,
            )
        if q.dim() == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
        q = q.contiguous()
        topk_slots = topk_slots.contiguous()
        if topk_lens is None:
            topk_lens = torch.full(
                (q.shape[0],),
                topk_slots.shape[-1],
                dtype=torch.int32,
                device=q.device,
            )
        block_h, num_splits = _sparse_mma.plan_launch(
            q.shape[0], q.shape[1], current_platform().sm_count
        )
        chunk_tiles, num_chunks = _triton_sparse_mma.num_chunks_for(
            topk_slots.shape[-1],
            _sparse_mma.rows_for(q.dtype == torch.float8_e4m3fn),
        )
        result, lse = _sparse_mma.sparse_mma_attention(
            q,
            kv_cache,
            topk_slots,
            topk_lens.to(torch.int32).contiguous(),
            softmax_scale=float(softmax_scale) * float(k_scale),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=return_lse,
            block_h=block_h,
            chunk_tiles=chunk_tiles,
            num_chunks=num_chunks,
            num_splits=num_splits,
        )
        if out is not None:
            out.reshape_as(result).copy_(result)
            result = out
        return (result, lse) if return_lse else result

    @register_kernel(
        "attention",
        "dsa_decode",
        name="cute_dsl_dsa_mma_decode",
        solution="cute_dsl",
        capability=_dsa_triton._SM120_MMA_CAPABILITY,
        signatures=_dsa_triton._MMA_SIGNATURES,
        features=_dsa_triton._MMA_FEATURES,
        traits={**_dsa_triton._MMA_TRAITS, "q_len": frozenset({1, 2, 3, 4, 5, 6})},
        priority=Priority.PERFORMANT + 1,
    )
    def cute_dsl_dsa_mma_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        kv_seq_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
        *,
        slot_order: str,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Both orders reduce the slots as they arrive; the top-k leaf chose it.
        del kv_seq_lens, slot_order
        return _run_cute_dsa_mma(
            q=q,
            kv_cache=kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            out=out,
            return_lse=return_lse,
        )

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="cute_dsl_dsa_mma_prefill",
        solution="cute_dsl",
        capability=_dsa_triton._SM120_MMA_CAPABILITY,
        signatures=_dsa_triton._MMA_SIGNATURES,
        features=_dsa_triton._MMA_FEATURES,
        traits={**_dsa_triton._MMA_TRAITS, "q_len": frozenset({1})},
        priority=Priority.PERFORMANT + 1,
    )
    def cute_dsl_dsa_mma_prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        kv_seq_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
        *,
        slot_order: str,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Both orders reduce the slots as they arrive; the top-k leaf chose it.
        del kv_seq_lens, slot_order
        return _run_cute_dsa_mma(
            q=q,
            kv_cache=kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            out=out,
            return_lse=return_lse,
        )
