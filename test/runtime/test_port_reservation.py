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

"""Servers started at the same moment must not share control-plane addresses."""

import multiprocessing
import os
import socket

import pytest
import zmq

from tokenspeed.runtime.utils.network import is_port_available
from tokenspeed.runtime.utils.port_reservation import (
    claim_port,
    create_ipc_dir,
    release_port,
    reserve_port,
)
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

_PORTS_PER_PROCESS = 40


def _reserve_ports(barrier, results, stop):
    barrier.wait()
    ports = [reserve_port() for _ in range(_PORTS_PER_PROCESS)]
    results.put(ports)
    stop.wait(60)


def _init_port_args(http_port, barrier, results, stop):
    barrier.wait()
    port_args = PortArgs.init_new(ServerArgs(model="x", port=http_port))
    results.put(
        (
            port_args.tokenizer_ipc_name,
            port_args.scheduler_input_ipc_name,
            port_args.metrics_ipc_name,
            port_args.rpc_ipc_name,
            port_args.dist_init_addr,
            port_args.nccl_port,
        )
    )
    stop.wait(60)


def _run_concurrently(target, args_per_process):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(len(args_per_process))
    results = ctx.Queue()
    stop = ctx.Event()
    procs = [
        ctx.Process(target=target, args=(*args, barrier, results, stop))
        for args in args_per_process
    ]
    for proc in procs:
        proc.start()
    try:
        # Claims live as long as the processes, so collect before stopping them.
        return [results.get(timeout=120) for _ in procs], stop, procs
    except BaseException:
        stop.set()
        raise


def _finish(stop, procs):
    stop.set()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0


def test_concurrent_processes_reserve_distinct_ports():
    results, stop, procs = _run_concurrently(_reserve_ports, [()] * 4)
    _finish(stop, procs)
    ports = [port for process_ports in results for port in process_ports]
    assert len(ports) == len(set(ports)) == 4 * _PORTS_PER_PROCESS


def test_concurrent_servers_on_adjacent_http_ports_do_not_collide():
    # Adjacent HTTP ports used to derive overlapping control-plane clusters.
    results, stop, procs = _run_concurrently(_init_port_args, [(8000,), (8001,)])
    try:
        first, second = results
        for name in range(4):
            assert first[name] != second[name]
        assert first[4] != second[4]
        assert first[5] != second[5]
        # Both servers' ipc endpoints bind at once.
        context = zmq.Context()
        sockets = []
        try:
            for endpoints in results:
                for endpoint in endpoints[:3]:
                    sock = context.socket(zmq.PULL)
                    sock.bind(endpoint)
                    sockets.append(sock)
        finally:
            for sock in sockets:
                sock.close(linger=0)
            context.term()
    finally:
        _finish(stop, procs)


def test_claim_is_exclusive_across_processes_and_follows_the_holder():
    port = reserve_port()
    ctx = multiprocessing.get_context("spawn")
    parent_claim_visible = ctx.Queue()
    proc = ctx.Process(target=_try_claim, args=(port, parent_claim_visible))
    proc.start()
    assert parent_claim_visible.get(timeout=120) is False
    proc.join(timeout=60)
    release_port(port)
    proc = ctx.Process(target=_try_claim, args=(port, parent_claim_visible))
    proc.start()
    assert parent_claim_visible.get(timeout=120) is True
    proc.join(timeout=60)


def _try_claim(port, out):
    out.put(claim_port(port))


def test_claim_rejects_a_port_another_program_listens_on():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen(1)
        assert claim_port(listener.getsockname()[1]) is False


def _free_cluster_base():
    while True:
        base = reserve_port()
        release_port(base)
        if all(is_port_available(base + offset) for offset in range(8)):
            return base


def test_explicit_dist_init_addr_is_honored():
    base = _free_cluster_base()
    args = ServerArgs(model="x", dist_init_addr=f"127.0.0.1:{base}")

    port_args = PortArgs.init_new(args)

    assert port_args.dist_init_addr == f"127.0.0.1:{base}"
    assert port_args.tokenizer_ipc_name == f"tcp://127.0.0.1:{base + 1}"


def test_explicit_dist_init_addr_fails_when_a_port_is_busy():
    base = _free_cluster_base()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("", base + 1))
        busy.listen(1)
        args = ServerArgs(model="x", dist_init_addr=f"127.0.0.1:{base}")
        with pytest.raises(ValueError, match=f"\\[{base + 1}\\] are already in use"):
            PortArgs.init_new(args)


def test_ipc_dirs_are_distinct_per_call():
    first = create_ipc_dir()
    second = create_ipc_dir()
    assert first != second
    assert os.path.isdir(first) and os.path.isdir(second)
