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
"""CuTe DSL leaf for the DeepSeek V4.1 selected attention on SM120.

The CuTe DSL core of ``dsv4/_cute_dsl/sparse_mma.py`` reads the packed
page-planar rows itself: a CTA gathers the FP8 ``swa`` rows and the FP4
``global`` rows of a tile into registers and dequantizes them into its BF16
shared-memory ring, so no BF16 workspace is written. The rows it builds are
the rows the portable route (gather into a BF16 workspace, then the dense
core) holds, and the loop is the same, so the result is bitwise that of
``cute_dsl_dsv4_mma_prefill`` over the gathered workspace.
"""

from __future__ import annotations

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

try:
    from tokenspeed_kernel.ops.attention.dsv4._cute_dsl import sparse_mma as _sparse_mma
except ImportError:
    _sparse_mma = None

if _sparse_mma is not None:
    from tokenspeed_kernel.ops.attention.dsv41 import triton as _dsv41_triton

    @register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="cute_dsl_dsv41_mma_selected_attention",
        solution="cute_dsl",
        capability=_dsv41_triton._MMA_CAPABILITY,
        signatures=_dsv41_triton._QUERY_SIGNATURES,
        priority=Priority.PERFORMANT + 1,
    )
    def cute_dsl_dsv41_mma_selected_attention(
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        out,
        query_chunk_size,
        schedule,
        prefill_kv,
        prefill_indices,
        *,
        lse=None,
    ):
        """Selected attention over the packed ``swa`` and ``global`` cache rows.

        Layers with a compressed (FP4 ``global``) cache read both caches in the
        CuTe core: 1.5-7.9x over the workspace route and 1.4-3.0x over the
        Triton direct reader (docs: analysis 9.40). A window-only layer (FP8
        rows alone) is 1.4-7.6x over the workspace route and 1.16x geomean over
        the Triton direct reader (0.94-1.77x: ties of 0.94-1.06x at 32-128 heads for
        1-4 tokens and from 256 or 1024 tokens, a few tenths of a microsecond
        when launch-bound).
        Workspace prefill and any layout the kernel's vector loads cannot
        address run ``triton_dsv41_mma_selected_attention``; every choice
        follows the layer's cache layout alone, never the batch, so a row stays
        bitwise stable.
        """
        args = (
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            softmax_scale,
            out,
            query_chunk_size,
            schedule,
            prefill_kv,
            prefill_indices,
        )
        if prefill_kv is not None:
            return _dsv41_triton.mma_selected_attention(*args, lse=lse)
        segments = _dsv41_triton.direct_segments(*args[:8], out, lse, prefill_indices)
        if segments is None:
            return _dsv41_triton.mma_selected_attention(*args, lse=lse)
        swa, extra = segments
        output = _dsv41_triton._output(out, q.shape, q.dtype, q.device)
        if not _sparse_mma.supports_paged(q, output, swa, extra):
            return _dsv41_triton.mma_selected_attention(*args, lse=lse)
        sink = None
        if attn_sink is not None:
            sink = attn_sink.reshape(-1).contiguous()
        block_h, cwarps, num_splits, full_tokens = _sparse_mma.plan_paged(
            q.shape[0], q.shape[1], current_platform().sm_count, swa, extra
        )
        _sparse_mma.paged_attention(
            q,
            swa,
            extra,
            sink,
            output,
            lse,
            softmax_scale=softmax_scale,
            block_h=block_h,
            cwarps=cwarps,
            num_splits=num_splits,
            full_tokens=full_tokens,
        )
        return output
