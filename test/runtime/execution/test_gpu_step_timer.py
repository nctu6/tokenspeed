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

"""GpuStepTimer: per-window GPU durations of bracketed forwards."""

from __future__ import annotations

import os
import re
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.gpu_step_timer import GpuStepTimer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _forward(timer: GpuStepTimer, kind: str, x: torch.Tensor, steps: int) -> None:
    start = timer.start()
    x.mul_(1.0)  # prologue work
    launch = timer.start()
    for _ in range(steps):
        x.mul_(1.0001)
    timer.finish(kind, start, launch, 0.5)


def test_window_reports_each_kind_and_its_step():
    timer = GpuStepTimer(window_s=0.0)
    x = torch.ones(1 << 22, device="cuda")
    _forward(timer, "decode", x, 50)
    _forward(timer, "extend", x, 200)
    torch.cuda.synchronize()
    # The next forward drains the two completed ones and closes the window.
    _forward(timer, "decode", x, 1)
    report = timer.last_report
    assert report is not None
    decode = re.search(r"\| decode: n=(\d+) sum_ms=([\d.]+)", report)
    step = re.search(r"\| decode\.step: n=(\d+) sum_ms=([\d.]+)", report)
    host = re.search(r"\| decode\.step_host: n=1 sum_ms=(\d+) p50_ms=0\.50", report)
    assert decode and step and host
    # The step bracket lies inside the forward bracket.
    assert int(decode.group(1)) >= 1
    assert float(step.group(2)) <= float(decode.group(2))
    assert re.search(r"\| extend: n=1 ", report)
