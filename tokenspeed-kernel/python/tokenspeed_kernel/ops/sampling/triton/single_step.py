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

"""Single-step (N == 1) decode sampling by inverse CDF.

Non-speculative decode is verify's one-column case: nothing to accept, one
token drawn from the target distribution with the row's final coin. This
kernel does exactly that draw and writes the verify triple
(``predict[b]``, ``accept_index[b] = b``, ``accept_length[b] = 0``), so it can
stand in for the chain verify kernel at N == 1.

Rows whose filter is the identity (no top-k, ``top_p >= 1``) draw straight
from ``probs``; every other row draws from ``renormed`` (the top-k/top-p
renormalized distribution). Pairing this with
``fused_topk_topp_renorm(..., skip_identity_rows=True)`` removes the renorm
read+write for identity rows. The choice is per row on device, so one CUDA
graph serves any mix of rows.

The draw: first index ``i`` with ``cumsum(src)[i] > coin * sum(src)``, the same
rule as the chain kernel's final sampling; exact zeros are never selected.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _single_step_sample_kernel(
    probs_ptr,
    renormed_ptr,
    top_ks_ptr,
    top_ps_ptr,
    coins_ptr,
    predict_ptr,
    accept_index_ptr,
    accept_length_ptr,
    vocab_size,
    max_topk,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.load(top_ks_ptr + row)
    p = tl.load(top_ps_ptr + row)
    identity = (k > max_topk) & (p >= 1.0)
    base = row.to(tl.int64) * vocab_size
    src = tl.where(identity, probs_ptr, renormed_ptr) + base
    offs = tl.arange(0, BLOCK)

    total = tl.zeros((), dtype=tl.float32)
    last_nz = tl.zeros((), dtype=tl.int32)
    for start in range(0, vocab_size, BLOCK):
        idx = start + offs
        m = idx < vocab_size
        x = tl.load(src + idx, mask=m, other=0.0)
        total += tl.sum(x, axis=0)
        last_nz = tl.maximum(last_nz, tl.max(tl.where(x > 0.0, idx, 0), axis=0))

    u = tl.load(coins_ptr + row) * total
    run = tl.zeros((), dtype=tl.float32)
    found = tl.full((), vocab_size, dtype=tl.int32)
    for start in range(0, vocab_size, BLOCK):
        if found == vocab_size:
            idx = start + offs
            m = idx < vocab_size
            x = tl.load(src + idx, mask=m, other=0.0)
            cum = run + tl.cumsum(x, axis=0)
            hit = (cum > u) & (x > 0.0) & m
            first = tl.min(tl.where(hit, idx, vocab_size), axis=0)
            found = tl.minimum(found, first)
            run += tl.sum(x, axis=0)
    token = tl.where(found < vocab_size, found, last_nz)
    tl.store(predict_ptr + row, token)
    tl.store(accept_index_ptr + row, row)
    tl.store(accept_length_ptr + row, 0)


def single_step_sample(
    probs: torch.Tensor,
    renormed: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    coins: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    max_topk: int,
) -> None:
    """Draw one token per row; write the N == 1 verify triple in place.

    Args:
        probs: ``[bs, V]`` float32 softmax probabilities.
        renormed: ``[bs, V]`` float32 top-k/top-p renormalized rows (identity
            rows may be left unwritten).
        top_ks: ``[bs]`` int32; ``k > max_topk`` means top-k is off.
        top_ps: ``[bs]`` float32.
        coins: ``[bs]`` uniform coins in ``(0, 1)``.
        predict, accept_length: ``[bs]`` int32 outputs.
        accept_index: ``[bs, 1]`` (or ``[bs]``) int32 output.
        max_topk: largest K served by the top-k path.
    """
    bs, vocab = probs.shape
    if bs == 0:
        return
    _single_step_sample_kernel[(bs,)](
        probs,
        renormed,
        top_ks,
        top_ps,
        coins,
        predict,
        accept_index,
        accept_length,
        vocab,
        max_topk,
        BLOCK=4096,
        num_warps=8,
    )
