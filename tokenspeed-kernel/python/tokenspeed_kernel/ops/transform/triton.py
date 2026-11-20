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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _hadamard_128_kernel(
    x,
    out,
    # Per-call row count; runtime so every batch shape shares one binary.
    n_rows,
    scale: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    out_block = tl.program_id(1)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    in_offsets = tl.arange(0, 128)

    vals = tl.load(x + row * 128 + in_offsets).to(tl.float32)
    bits = out_offsets[:, None] & in_offsets[None, :]
    parity = bits ^ (bits >> 1)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 4)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.sum(signs * vals[None, :], axis=1) * scale

    tl.store(
        out + row * 128 + out_offsets,
        acc,
        mask=(row < n_rows) & (out_offsets < 128),
    )


@register_kernel(
    "transform",
    "hadamard_transform",
    name="triton_hadamard_transform_128",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.bfloat16, torch.float16, torch.float32},
    ),
    traits={
        "last_dim": frozenset({128}),
        "contiguous": frozenset({False, True}),
    },
    priority=Priority.PORTABLE,
)
def triton_hadamard_transform_128(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a length-128 Sylvester Hadamard transform along the last dim."""
    if x.shape[-1] != 128:
        raise ValueError(
            f"triton_hadamard_transform_128 requires last dim 128, got {x.shape[-1]}"
        )
    if not x.is_cuda:
        raise RuntimeError("triton_hadamard_transform_128 requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            f"triton_hadamard_transform_128 does not support dtype {x.dtype}"
        )

    shape = x.shape
    x_2d = x.reshape(-1, 128).contiguous()
    out = torch.empty_like(x_2d)
    if x_2d.shape[0] == 0:
        return out.reshape(shape)
    _hadamard_128_kernel[(x_2d.shape[0], 8)](
        x_2d,
        out,
        n_rows=x_2d.shape[0],
        scale=float(scale),
        BLOCK_OUT=16,
        num_warps=8,
        num_stages=1,
    )
    return out.reshape(shape)


@triton.jit
def _hadamard_128_mma_kernel(
    x,
    out,
    # Per-call row count and scale; runtime so every batch shape shares one binary.
    n_rows,
    scale,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    # H[i, j] = (-1) ** popcount(i & j) is exact in BF16/FP16, so a tensor-core
    # product with FP32 accumulation is the exact transform of the stored input.
    bits = cols[:, None] & cols[None, :]
    bits = bits ^ (bits >> 1)
    bits = bits ^ (bits >> 2)
    bits = bits ^ (bits >> 4)
    hadamard = (1 - 2 * (bits & 1)).to(x.dtype.element_ty)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows[:, None] < n_rows
    offsets = rows[:, None] * 128 + cols[None, :]
    vals = tl.load(x + offsets, mask=mask, other=0.0)
    acc = tl.dot(vals, hadamard, out_dtype=tl.float32) * scale
    tl.store(out + offsets, acc.to(out.dtype.element_ty), mask=mask)


@register_kernel(
    "transform",
    "hadamard_transform",
    name="triton_hadamard_transform_128_mma",
    solution="triton",
    # Measured on SM120 only: there the fast-hadamard-transform wheel has no
    # cubin, and this kernel runs at the DRAM copy bandwidth.
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 9),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "last_dim": frozenset({128}),
        "contiguous": frozenset({False, True}),
    },
    priority=Priority.PERFORMANT,
)
def triton_hadamard_transform_128_mma(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Length-128 Sylvester Hadamard transform as one tensor-core matmul.

    Each program multiplies a ``[BLOCK_M, 128]`` row tile by the ``[128, 128]``
    sign matrix on ``mma.sync``. The result is bitwise equal to the
    butterfly transform in the CUDA ``fast_hadamard_transform`` kernel.
    """
    if x.shape[-1] != 128:
        raise ValueError(
            f"triton_hadamard_transform_128_mma requires last dim 128, got {x.shape[-1]}"
        )
    if not x.is_cuda:
        raise RuntimeError("triton_hadamard_transform_128_mma requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"triton_hadamard_transform_128_mma does not support dtype {x.dtype}"
        )
    shape = x.shape
    x_2d = x.reshape(-1, 128).contiguous()
    out = torch.empty_like(x_2d)
    if x_2d.shape[0] == 0:
        return out.reshape(shape)
    # Small batches are latency bound (decode); large ones stream at DRAM speed.
    block_m, num_warps, num_stages = (16, 4, 1) if x_2d.shape[0] <= 2048 else (64, 8, 2)
    _hadamard_128_mma_kernel[(triton.cdiv(x_2d.shape[0], block_m),)](
        x_2d,
        out,
        n_rows=x_2d.shape[0],
        scale=float(scale),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(shape)


__all__ = ["triton_hadamard_transform_128", "triton_hadamard_transform_128_mma"]
