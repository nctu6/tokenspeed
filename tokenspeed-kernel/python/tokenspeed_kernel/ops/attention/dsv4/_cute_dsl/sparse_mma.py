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
"""CuTe DSL tensor-core DeepSeek V4 selected attention (SM120).

The kernel is the BF16 path of the DSA CuTe DSL core
(``dsa/_cute_dsl/sparse_mma.py``) in its V4 variant: a 512-wide query (rank 512,
no RoPE columns) over the rows of a flat BF16 workspace, where a V4 latent row
is key and value at once. What the variant changes:

* The workspace is a flat row table (a page of one row), so a row id addresses
  ``kv + id * row_stride`` with no divide. Row ids are int32 or int64 and are
  checked against the workspace in the kernel: a negative id or one past the
  last row is skipped, as in ``_triton/sparse_mma.py``.
* The query and output are addressed with their own token and head strides.
* The per-head sink is one extra zero-valued key and enters as the initial
  state of chunk 0 (``m`` = sink logit, ``l`` = 1), so it is the first thing the
  fold sees and rows with no live key still return the sink-only state. It
  joins the denominator only.

Numerics contract and batch invariance are the Triton leaf's: BF16 scores and
probabilities accumulated in FP32, an FP32 online softmax per chunk, and a
selection cut into at most ``MAX_CHUNKS`` chunks of whole tiles whose number
depends on the width alone, folded in order with explicit ``fma`` either in the
CTA or, when the batch cannot fill the GPU, by the combine kernel from stored
chunk states. A row is bitwise the same for every batch, head tile and split.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsa._cute_dsl import sparse_mma as _core
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "MAX_CHUNKS",
    "num_chunks_for",
    "plan_launch",
    "sparse_mma_attention",
    "supports",
]

HEAD_DIM = 512
ROWS = _core.rows_for(False)
# Chunks per token selection; a split launch gives each its own CTA.
MAX_CHUNKS = 4


def num_chunks_for(width: int) -> tuple[int, int]:
    """``(chunk_tiles, num_chunks)`` for a selection ``width`` columns wide.

    A function of the width alone, so every batch size cuts a token's
    selection at the same columns.
    """
    tiles = max(-(-width // ROWS), 1)
    chunk_tiles = -(-tiles // MAX_CHUNKS)
    return chunk_tiles, -(-tiles // chunk_tiles)


def plan_launch(tokens: int, heads: int, num_sms: int) -> tuple[int, int]:
    """``(block_h, num_splits)`` for ``tokens`` queries of ``heads`` heads.

    Neither value changes the result. Measured on an RTX PRO 6000 (188 SMs)
    with 640 selected rows: a launch is latency bound until about two thirds
    of the SMs hold a CTA, so the split aims at that many CTAs, one chunk
    each. A 16-head CTA has half the warps of a 32-head one and loses to it as
    soon as the batch has more than 32 of them, which is also the point where
    splitting stops paying (128 or more 32-head CTAs fill the GPU unsplit).
    """
    narrow = -(-heads // 16) * tokens
    block_h = 16 if heads <= 16 or narrow <= 32 else 32
    programs = -(-heads // block_h) * tokens
    target = num_sms * 2 // 3
    return block_h, min(max(round(target / programs), 1), MAX_CHUNKS)


def supports(
    q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor, indices: torch.Tensor
) -> bool:
    """Whether the kernel's loads and stores are aligned for these tensors.

    The query is read in 4-byte words, the output written in 8-byte words and
    the workspace rows gathered in 16-byte ``cp.async`` copies. The answer
    depends on the tensors' layout, never on the batch contents.
    """
    kv_row_bytes = kv.stride(0) * kv.element_size()
    return (
        q.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM
        and q.stride(-1) == 1
        and out.stride(-1) == 1
        and kv.stride(-1) == 1
        and q.data_ptr() % 4 == 0
        and q.stride(0) % 2 == 0
        and q.stride(1) % 2 == 0
        and out.data_ptr() % 8 == 0
        and out.stride(0) % 4 == 0
        and out.stride(1) % 4 == 0
        and kv.data_ptr() % 16 == 0
        and kv_row_bytes % 16 == 0
        and kv.shape[0] < 2**31
        and indices.stride(-1) == 1
    )


def _origin(tensor: torch.Tensor) -> torch.Tensor:
    """A one-element view at ``tensor``'s first element: the kernel only needs
    its address, and the tensor itself may be a strided view."""
    return tensor.as_strided((1,), (1,))


def sparse_mma_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    *,
    softmax_scale: float,
    block_h: int,
    num_splits: int,
) -> None:
    """Attend ``q`` over the workspace rows named by ``indices``.

    Args:
        q: ``[tokens, heads, 512]`` BF16 with unit inner stride.
        kv: ``[rows, 512]`` BF16 workspace with unit inner stride
            (:func:`supports` holds).
        indices: ``[tokens, width]`` int32/int64 row ids with unit column
            stride; entries that are negative or past the last row are skipped.
        lens: ``[tokens]`` int32/int64 count of leading live columns.
        sink: ``[heads]`` contiguous FP32 sink logits or None.
        out: ``[tokens, heads, 512]`` BF16 result with unit inner stride.
        lse: ``[tokens, heads]`` contiguous FP32 natural-log normalizer or None.
        block_h: heads per CTA (16 or 32).
        num_splits: CTAs sharing one (token, head tile), each taking a run of
            chunks; 1 folds in registers. The result is bitwise the same for
            every value.
    """
    tokens, heads, dim = q.shape
    device = q.device
    if tokens == 0 or heads == 0:
        return
    chunk_tiles, num_chunks = num_chunks_for(indices.shape[1])
    num_splits = max(min(num_splits, num_chunks), 1)
    split = num_splits > 1
    # Every token is split or none is (the core also takes a mix).
    head_tiles = -(-heads // block_h)
    full_tokens = 0 if split else tokens
    if split:
        part_acc = torch.empty(
            (tokens, num_chunks, heads, dim), dtype=torch.float32, device=device
        )
        part_ml = torch.empty(
            (tokens, num_chunks, heads, 2), dtype=torch.float32, device=device
        )
    else:
        part_acc = part_ml = lse if lse is not None else torch.empty(4, device=device)
    lse_arg = lse if lse is not None else part_ml
    lens = lens.contiguous()
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    kernel = _core.compiled_attention(
        False,
        block_h,
        dim,
        0,
        device_index,
        (sink is not None, indices.dtype == torch.int64, lens.dtype == torch.int64),
    )
    # The kernel counts strides in bytes.
    out_stride_token = out.stride(0) * 2
    out_stride_head = out.stride(1) * 2
    kernel(
        q.data_ptr(),
        kv.data_ptr(),
        _origin(indices),
        lens,
        out.data_ptr(),
        lse_arg.reshape(-1),
        part_acc.reshape(-1),
        part_ml.reshape(-1),
        heads,
        indices.shape[1],
        indices.stride(0),
        kv.stride(0) * 2,
        0,
        1,
        chunk_tiles,
        num_chunks,
        -(-num_chunks // num_splits),
        full_tokens,
        num_splits,
        _core.split_magic(num_splits),
        int(lse is not None),
        float(softmax_scale) * 1.4426950408889634,
        kv.shape[0],
        sink if sink is not None else part_ml.reshape(-1),
        q.stride(0) * 2,
        q.stride(1) * 2,
        out_stride_token,
        out_stride_head,
        head_tiles,
        full_tokens + (tokens - full_tokens) * num_splits,
    )
    if split:
        _core.compiled_combine(dim, device_index, True)(
            part_acc.reshape(-1),
            part_ml.reshape(-1),
            out.data_ptr(),
            lse_arg.reshape(-1),
            heads,
            num_chunks,
            int(lse is not None),
            out_stride_token,
            out_stride_head,
            full_tokens,
            tokens - full_tokens,
        )
