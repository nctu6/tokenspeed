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
"""The copy-engine all-reduce tier on a real two-rank PCIe group: armed only
where it applies, and the public all-reduce keeps NCCL's bits through it."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_backend.copy_engine import (
    copy_engine_all_reduce_applies,
)

HIDDEN = 3072


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def worker_fn(rank, port, error_dict):
    try:
        worker_main(rank, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    from tokenspeed.runtime.distributed.comm_backend import get_global_backend
    from tokenspeed.runtime.distributed.comm_ops import all_reduce
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(rank=rank, world_size=2, attn_tp_size=2, dense_tp_size=2)
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["batch_invariant_collectives"] = False
    global_server_args_dict["force_deterministic_rsag"] = False
    world_pg = pg_manager.get_process_group("nccl", group)
    try:
        backend = get_global_backend()
        # Chunks of 1 MiB: the 8 MiB payload below runs in several.
        assert backend.configure_copy_engine_all_reduce(group, 1 << 20)
        gen = torch.Generator(device=device).manual_seed(rank)
        for rows in (1024, 32, 1365):  # 6 MiB, 192 KiB (one-shot window), odd
            x = torch.randn(rows, HIDDEN, generator=gen, device=device)
            x = x.to(torch.bfloat16)
            expected = x.clone()
            dist.all_reduce(expected, group=world_pg)
            assert all_reduce(x, group) is x
            assert torch.equal(x, expected), f"{rows} rows differ from NCCL"
        # A collection past the window: every tensor in place, NCCL's bits.
        pair = tuple(
            torch.randn(rows, HIDDEN, generator=gen, device=device).to(torch.bfloat16)
            for rows in (512, 700)
        )
        expected = tuple(value.clone() for value in pair)
        for value in expected:
            dist.all_reduce(value, group=world_pg)
        reduced = backend.all_reduce(pair, group)
        for got, value, want in zip(reduced, pair, expected):
            assert got is value
            assert torch.equal(got, want)
    finally:
        dist.barrier(group=world_pg)
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not torch.cuda.can_device_access_peer(0, 1)
    or not copy_engine_all_reduce_applies((0, 1)),
    reason="requires two PCIe-connected NVIDIA GPUs with peer access",
)
def test_public_all_reduce_through_copy_engine_matches_nccl():
    port = get_open_port()
    manager = mp.Manager()
    errors = manager.dict()
    mp.spawn(worker_fn, args=(port, errors), nprocs=2, join=True)
    assert not errors, "\n".join(errors.values())


def test_copy_engine_tier_needs_two_ranks():
    assert not copy_engine_all_reduce_applies((0, 1, 2, 3))
    assert not copy_engine_all_reduce_applies((0,))
