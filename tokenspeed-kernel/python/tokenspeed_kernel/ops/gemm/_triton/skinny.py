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

"""Skinny dense BF16 GEMM for decode-sized row counts (M <= 16)."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

# Rows of one tile: every M up to this pads to it, so one binary serves them.
SKINNY_BLOCK_M = 16
# Weight rows per CTA: small, so even an N=1280 projection spreads over most
# SMs (each CTA streams BN x K of the weight once).
SKINNY_BLOCK_N = 16


@triton.jit(do_not_specialize=["M"])
def _triton_mm_bf16_skinny_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M,
    stride_am,
    stride_bn,
    stride_om,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``out[:M] = a[:M] @ b.T (+ bias)`` for one ``BLOCK_N`` slice of ``b``.

    The ``M`` rows pad to ``BLOCK_M`` (masked loads), so a decode step of any
    size up to 16 rows reads the weight exactly once at full width; products
    accumulate in FP32 and round once to the output dtype. ``M`` is the only
    per-batch value and is not specialized; the row strides are the static
    widths of the operands, so they keep Triton's alignment specialization
    (vectorized loads and stores).
    """
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    # 64-bit tile base (large weights), 32-bit offsets inside the tile.
    b_tile = b_ptr + (pid * BLOCK_N).to(tl.int64) * stride_bn
    tn = tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm[:, None] < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + (k + rk)[None, :],
            mask=row_mask,
            other=0.0,
        )
        # [BLOCK_K, BLOCK_N] view of the [N, K] weight: contiguous along K.
        b = tl.load(b_tile + tn[None, :] * stride_bn + (k + rk)[:, None])
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn).to(tl.float32)[None, :]
    tl.store(
        out_ptr + rm[:, None] * stride_om + rn[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


def skinny_block_k(k: int) -> int:
    """K tile: 256 for long rows (fc2-like K >= 4096), else 128.

    Both measured best on sm120 at whisper-large-v3's projections; K must be a
    multiple of the returned tile (the registry admits K % 128 == 0 only).
    """
    return 256 if k >= 4096 and k % 256 == 0 else 128


def mm_bf16_skinny(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """``A @ B.T (+ bias)`` for ``1 <= M <= 16`` BF16 rows.

    Args:
        A: ``[M, K]`` BF16 activations, unit inner stride, ``M <= 16``.
        B: ``[N, K]`` BF16 weight with unit inner stride, ``N % 16 == 0``,
            ``K % 128 == 0``.
        bias: optional ``[N]`` bias added before the single rounding.
        out: optional ``[M, N]`` BF16 destination with unit inner stride.

    Returns:
        ``[M, N]`` BF16 output (``out`` when given).
    """
    m, k = A.shape
    n = B.shape[0]
    if not (
        1 <= m <= SKINNY_BLOCK_M
        and A.stride(-1) == 1
        and B.stride(-1) == 1
        and B.shape[1] == k
        and n % SKINNY_BLOCK_N == 0
        and k % 128 == 0
    ):
        raise ValueError(
            f"mm_bf16_skinny cannot run A {tuple(A.shape)} x B {tuple(B.shape)}"
        )
    if out is None:
        out = torch.empty((m, n), dtype=A.dtype, device=A.device)
    _triton_mm_bf16_skinny_kernel[(n // SKINNY_BLOCK_N,)](
        A,
        B,
        bias if bias is not None else B,
        out,
        m,
        A.stride(0),
        B.stride(0),
        out.stride(0),
        N=n,
        K=k,
        HAS_BIAS=bias is not None,
        BLOCK_M=SKINNY_BLOCK_M,
        BLOCK_N=SKINNY_BLOCK_N,
        BLOCK_K=skinny_block_k(k),
        num_warps=4,
        num_stages=4,
    )
    return out
