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
"""The two-rank copy-engine all-reduce equals NCCL's all-reduce bit for bit."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.copy_engine import CopyEngineAllReduceState

# A small chunk so most cases run several chunks and an uneven last one.
CHUNK_BYTES = 64 * 1024


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def worker_fn(rank, port, error_dict):
    try:
        worker_main(rank, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def payload(rank: int, numel: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed * 2 + rank)
    return (torch.randn(numel, generator=gen, device="cuda") * 8).to(dtype)


def worker_main(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        state = CopyEngineAllReduceState(
            dist.group.WORLD, CHUNK_BYTES, torch.device("cuda", rank)
        )
        cases = [
            (torch.bfloat16, 3),  # below one aligned half: rank 1 sums it all
            (torch.bfloat16, 4096),  # one chunk
            (torch.bfloat16, 3 * 32768 + 7),  # several chunks, odd tail
            (torch.float16, 50001),
            (torch.float32, 40000),
            (torch.bfloat16, 0),
        ]
        # Back-to-back calls on new data reuse the buffers across chunks and
        # calls; every one must equal NCCL.
        for seed, (dtype, numel) in enumerate(cases * 2):
            x = payload(rank, numel, dtype, seed)
            expected = x.clone()
            dist.all_reduce(expected)
            assert state.all_reduce_(x) is x
            assert torch.equal(x, expected), f"{dtype} {numel} differs from NCCL"

        # A row view of a larger tensor reduces only its rows.
        base = payload(rank, 8 * 4096, torch.bfloat16, 99).view(8, 4096)
        untouched = base[:2].clone()
        rows = base[2:]
        expected = rows.clone()
        dist.all_reduce(expected)
        state.all_reduce_(rows)
        assert torch.equal(base[2:], expected)
        assert torch.equal(base[:2], untouched)

        # Captured once, replayed on new inputs written into the same tensor.
        x = torch.empty(5 * 4096, dtype=torch.bfloat16, device="cuda")
        x.copy_(payload(rank, x.numel(), torch.bfloat16, 7))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            state.all_reduce_(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.all_reduce_(x)
        for seed in (11, 12, 13):
            fresh = payload(rank, x.numel(), torch.bfloat16, seed)
            expected = fresh.clone()
            dist.all_reduce(expected)
            x.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(x, expected), f"graph replay {seed} differs"
        del graph
        torch.cuda.synchronize()
    finally:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not torch.cuda.can_device_access_peer(0, 1),
    reason="requires two GPUs with peer access",
)
def test_copy_engine_all_reduce_matches_nccl():
    port = get_open_port()
    manager = mp.Manager()
    errors = manager.dict()
    mp.spawn(worker_fn, args=(port, errors), nprocs=2, join=True)
    assert not errors, "\n".join(errors.values())
