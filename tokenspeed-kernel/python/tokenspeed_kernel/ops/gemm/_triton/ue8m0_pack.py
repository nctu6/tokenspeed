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

"""Pack canonical UE8M0 scales into the MN-major int32 layout DeepGEMM reads."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_BLOCK_ROWS = 32
_BLOCK_WORDS = 8


@triton.jit(do_not_specialize=["rows", "aligned_rows"])
def _pack_ue8m0_mn_major_kernel(
    src_ptr,
    dst_ptr,
    rows,
    aligned_rows,
    words,
    src_row_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    # src is [rows, words] int32 (four consecutive K-group bytes per word);
    # dst is [words, aligned_rows]. Padding rows are written as zero.
    r = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    w = tl.program_id(1) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    word_mask = w < words
    values = tl.load(
        src_ptr + r[:, None].to(tl.int64) * src_row_stride + w[None, :],
        mask=(r[:, None] < rows) & word_mask[None, :],
        other=0,
    )
    tl.store(
        dst_ptr + w[None, :].to(tl.int64) * aligned_rows + r[:, None],
        values,
        mask=(r[:, None] < aligned_rows) & word_mask[None, :],
    )


def pack_ue8m0_mn_major(scales: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 scale bytes ``[MN, K/32]`` for DeepGEMM's block-scaled GEMMs.

    DeepGEMM reads four consecutive K-group bytes as one int32 word, MN-major,
    with MN aligned up to a multiple of 4. One launch, no host work, and the
    same values as ``thirdparty.deepgemm.pack_ue8m0_scales``.

    Args:
        scales: uint8 ``[MN, K/32]`` with K divisible by 128 and unit inner
            stride.

    Returns:
        int32 ``[MN, K/128]`` with strides ``(1, align4(MN))``.
    """
    if scales.dtype != torch.uint8 or scales.ndim != 2 or scales.stride(1) != 1:
        raise ValueError("UE8M0 scales must be uint8 [MN, K/32] with unit inner stride")
    rows, groups = scales.shape
    if groups % 4 != 0:
        raise ValueError(f"UE8M0 scales need K divisible by 128, got {groups} groups")
    if scales.stride(0) % 4 != 0 or scales.data_ptr() % 4 != 0:
        scales = scales.contiguous()
        if scales.data_ptr() % 4 != 0:
            scales = scales.clone()
    words = groups // 4
    aligned_rows = (rows + 3) // 4 * 4
    packed = torch.empty((words, aligned_rows), dtype=torch.int32, device=scales.device)
    if aligned_rows:
        _pack_ue8m0_mn_major_kernel[
            (triton.cdiv(aligned_rows, _BLOCK_ROWS), triton.cdiv(words, _BLOCK_WORDS))
        ](
            scales.view(torch.int32),
            packed,
            rows,
            aligned_rows,
            words,
            scales.stride(0) // 4,
            BLOCK_ROWS=_BLOCK_ROWS,
            BLOCK_WORDS=_BLOCK_WORDS,
            num_warps=2,
            num_stages=1,
        )
    return packed.T[:rows]
