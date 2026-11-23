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
"""FA4 prefill on SM120: selection by head dim, and results against an FP32 reference."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import mha_prefill
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_cute_compile


def _reference(q, k, v, lengths, window_left, scale):
    """Per-sequence causal softmax attention in FP32, GQA by repeating the KV heads."""
    outputs = []
    start = 0
    group = q.shape[1] // k.shape[1]
    for length in lengths:
        qs = q[start : start + length].float().transpose(0, 1)
        ks = k[start : start + length].float().repeat_interleave(group, dim=1)
        vs = v[start : start + length].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hqd,khd->hqk", qs, ks) * scale
        rows = torch.arange(length, device=q.device)[:, None]
        cols = torch.arange(length, device=q.device)[None, :]
        keep = cols <= rows
        if window_left >= 0:
            keep &= cols >= rows - window_left
        scores = scores.masked_fill(~keep, float("-inf"))
        outputs.append(
            torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs).to(q.dtype)
        )
        start += length
    return torch.cat(outputs)


def _batch(lengths, hq, hk, d, device):
    total = sum(lengths)
    q = torch.randn(total, hq, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, hk, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, hk, d, device=device, dtype=torch.bfloat16)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return q, k, v, torch.tensor(cu, device=device, dtype=torch.int32), cu


def _prefill(q, k, v, cu_seqlens, cu_cpu, window_left):
    return mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_cpu,
        max_seqlen=max(b - a for a, b in zip(cu_cpu, cu_cpu[1:])),
        window_left=window_left,
        solution="fa4",
    )


@pytest.mark.parametrize("head_dim", [64, 80, 96, 112, 128])
@pytest.mark.parametrize("heads", [(8, 2), (32, 8), (4, 4)], ids=["g4", "g4w", "mha"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
def test_fa4_prefill_matches_fp32_reference(
    device: str, require, head_dim: int, heads: tuple[int, int], window_left: int
) -> None:
    require(
        "attention",
        "mha_prefill",
        "fa4",
        torch.bfloat16,
        "q",
        traits={"head_dim": head_dim},
    )
    torch.manual_seed(0)
    lengths = [851, 64, 1053, 1]
    q, k, v, cu_seqlens, cu_cpu = _batch(lengths, *heads, head_dim, device)
    out = _prefill(q, k, v, cu_seqlens, cu_cpu, window_left)
    ref = _reference(q, k, v, lengths, window_left, 1 / math.sqrt(head_dim))
    # One bfloat16 rounding of the output and FP32 accumulation on both sides.
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_fa4_prefill_does_not_recompile_across_batch_shapes(
    device: str, require
) -> None:
    require("attention", "mha_prefill", "fa4", torch.bfloat16, "q")
    torch.manual_seed(0)
    # The engine's startup warmup runs one long prefill; the leaf compiles the
    # short-prefill variant then too.
    first = _batch([300, 77], 8, 2, 128, device)
    with kernel_warmup(enabled=True):
        _prefill(first[0], first[1], first[2], first[3], first[4], -1)
    with assert_no_cute_compile():
        for lengths in ([1], [4096], [511, 512, 513, 1000], [17] * 40):
            batch = _batch(lengths, 8, 2, 128, device)
            _prefill(batch[0], batch[1], batch[2], batch[3], batch[4], -1)


def _int32_at_end_of_mapped_memory(values: list[int]) -> torch.Tensor:
    """An int32 CUDA tensor whose last byte is the last mapped byte: one element more faults.

    Built on the driver's virtual-memory API, so that the page after the tensor is unmapped.
    The caching allocator never places a small tensor there reliably, which hid the bug
    this guards in most runs.
    """
    from cuda.bindings import driver

    def check(result):
        assert result[0] == driver.CUresult.CUDA_SUCCESS, result[0]
        return result[1] if len(result) == 2 else None

    torch.zeros(1, device="cuda")  # makes the primary context current
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    granularity = check(
        driver.cuMemGetAllocationGranularity(
            prop,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    )
    base = int(check(driver.cuMemAddressReserve(2 * granularity, 0, 0, 0)))
    handle = check(driver.cuMemCreate(granularity, prop, 0))
    check(driver.cuMemMap(driver.CUdeviceptr(base), granularity, 0, handle, 0))
    access = driver.CUmemAccessDesc()
    access.location = prop.location
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(driver.cuMemSetAccess(driver.CUdeviceptr(base), granularity, [access], 1))

    class Raw:
        def __init__(self, ptr: int, nbytes: int) -> None:
            self.__cuda_array_interface__ = {
                "shape": (nbytes,),
                "typestr": "|u1",
                "data": (ptr, False),
                "version": 2,
            }

    nbytes = 4 * len(values)
    raw = torch.as_tensor(Raw(base + granularity - nbytes, nbytes), device="cuda")
    tensor = raw.view(torch.int32)
    tensor.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
    return tensor


def _prefill_with_cu_seqlens_at_end_of_mapped_memory() -> None:
    # Lengths that are multiples of the 32 query rows a 128-row tile holds (4 query heads
    # per KV head) leave the scheduler's tile bound above the tiles needed, so surplus tiles run.
    lengths = [320, 96, 64, 128]
    q, k, v, _, cu_cpu = _batch(lengths, 32, 8, 128, "cuda")
    cu_seqlens = _int32_at_end_of_mapped_memory(cu_cpu)
    out = _prefill(q, k, v, cu_seqlens, cu_cpu, -1)
    torch.cuda.synchronize()
    ref = _reference(q, k, v, lengths, -1, 1 / math.sqrt(128))
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_fa4_prefill_does_not_read_past_cu_seqlens(require) -> None:
    pytest.importorskip("cuda.bindings.driver")
    require("attention", "mha_prefill", "fa4", torch.bfloat16, "q")
    # A fault is sticky for the whole process, so the check runs in its own.
    done = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert done.returncode == 0, done.stderr[-2000:]


if __name__ == "__main__":
    _prefill_with_cu_seqlens_at_end_of_mapped_memory()


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [
        (64, "fa4_mha_prefill_sm120"),
        (128, "fa4_mha_prefill_sm120"),
        # 256 does not fit FA4's SM80-class kernel in SM120's 99 KB of shared memory
        (256, "triton_mha_prefill"),
    ],
)
def test_sm120_selects_fa4_prefill_up_to_head_dim_128(
    sm120_platform: PlatformInfo, head_dim: int, expected: str
) -> None:
    import tokenspeed_kernel.ops.attention  # noqa: F401  (registers the leaves)

    signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k=dense_tensor_format(torch.bfloat16),
        v=dense_tensor_format(torch.bfloat16),
    )
    chosen = select_kernel(
        "attention",
        "mha_prefill",
        signature,
        platform=sm120_platform,
        traits={
            "head_dim": head_dim,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "skip_softmax": False,
            "sliding_window": False,
        },
    )
    assert chosen.name == expected
