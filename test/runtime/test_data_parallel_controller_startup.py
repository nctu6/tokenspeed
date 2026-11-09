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

"""A scheduler that fails to start must surface its own error.

The attention-DP controller starts one thread per DP rank. An exception in a
thread used to vanish, the scheduler-info fields stayed unset and the caller
failed later with ``AttributeError: ... multimodal_encoder_dtype``.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
import zmq

from tokenspeed.runtime.engine.data_parallel_controller import DataParallelController


class SchedulerStartupError(RuntimeError):
    pass


def _free_port_block(count: int) -> int:
    """Return a base port whose next ``count`` ports are free to bind."""
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        socks = []
        try:
            for i in range(1, count + 1):
                t = socket.socket()
                socks.append(t)
                t.bind(("127.0.0.1", base + i))
            return base
        except OSError:
            continue
        finally:
            for t in socks:
                t.close()


def _controller(failing_ranks: set[int], dp_size: int = 2):
    controller = DataParallelController.__new__(DataParallelController)
    controller.context = zmq.Context(1 + dp_size)
    controller.workers = [None] * dp_size
    started: list[int] = []

    def launch_tensor_parallel_group(server_args, port_args, dp_rank):
        if dp_rank in failing_ranks:
            raise SchedulerStartupError(f"rank {dp_rank} ran out of memory")
        started.append(dp_rank)

    controller.launch_tensor_parallel_group = launch_tensor_parallel_group
    server_args = SimpleNamespace(
        node_rank=0,
        mapping=SimpleNamespace(
            attn=SimpleNamespace(dp_size=dp_size, has_dp=True),
            nnodes=1,
        ),
    )
    base = _free_port_block(dp_size)
    port_args = SimpleNamespace(
        tokenizer_ipc_name="ipc://t",
        scheduler_input_ipc_name=f"tcp://127.0.0.1:{base}",
        nccl_port=1,
        dist_init_addr="127.0.0.1:4000",
        rpc_ipc_name="ipc://r",
        metrics_ipc_name="ipc://m",
        metrics_producer_configs=None,
        tokenizer_worker_ipc_name="ipc://w",
    )
    return controller, server_args, port_args, started, base


def test_launch_error_is_reraised():
    controller, server_args, port_args, started, _ = _controller({1})
    try:
        with pytest.raises(SchedulerStartupError, match="rank 1 ran out of memory"):
            controller.launch_dp_schedulers(server_args, port_args)
    finally:
        controller.context.destroy(linger=0)
    assert started == [0]


def test_launch_without_error_returns_port_args():
    controller, server_args, port_args, started, base = _controller(set())
    try:
        dp_port_args = controller.launch_dp_schedulers(server_args, port_args)
    finally:
        controller.context.destroy(linger=0)
    assert sorted(started) == [0, 1]
    assert [p.scheduler_input_ipc_name for p in dp_port_args] == [
        f"tcp://127.0.0.1:{base + 1}",
        f"tcp://127.0.0.1:{base + 2}",
    ]
