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

"""Profiler-free GPU timing of forwards on the execution stream.

TOKENSPEED_GPU_STEP_TIMING=1 brackets each forward with two timing events on
the execution stream and logs, every TOKENSPEED_GPU_STEP_TIMING_WINDOW_S
seconds (default 6), the GPU duration of decode and extend forwards and the
GPU idle between consecutive forwards.

Why not the torch profiler: under profiling ROCm attaches a completion signal
to every AQL packet of a graph launch, which changes how a single-queue graph
is submitted (DEBUG_HIP_FORCE_GRAPH_QUEUES=1 runs ~2x slower traced than
untraced). Two events per forward perturb neither path.
"""

from __future__ import annotations

import collections
import os
import statistics
import time

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

ENABLED = os.environ.get("TOKENSPEED_GPU_STEP_TIMING", "0") == "1"
WINDOW_S = float(os.environ.get("TOKENSPEED_GPU_STEP_TIMING_WINDOW_S", "6"))


class GpuStepTimer:
    """Collects (kind, start, end) event pairs and reports them per window."""

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._pending: collections.deque = collections.deque()
        self._prev_end: torch.cuda.Event | None = None
        # The latest window's report line, also logged.
        self.last_report: str | None = None
        self._t0 = 0.0
        self._dur: collections.defaultdict = collections.defaultdict(list)
        self._gap_ms = 0.0
        self._gpu_first: torch.cuda.Event | None = None
        self._gpu_last: torch.cuda.Event | None = None
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self._t0 = now
        self._dur = collections.defaultdict(list)
        self._gap_ms = 0.0
        self._gpu_first: torch.cuda.Event | None = None
        self._gpu_last: torch.cuda.Event | None = None

    def start(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def finish(
        self,
        kind: str,
        start: torch.cuda.Event,
        launch: torch.cuda.Event,
        launch_host_ms: float,
    ) -> None:
        """Close a forward opened by ``start``.

        ``launch`` was recorded just before the model step (graph replay or
        eager model launches) and ``launch_host_ms`` is the host time that
        step's call took, so a report separates the prologue's GPU time, the
        step's GPU time and how long the host was held by the step's launch.
        """
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((kind, start, launch, end, launch_host_ms))
        self._drain()

    def _drain(self) -> None:
        while self._pending and self._pending[0][3].query():
            kind, start, launch, end, launch_host_ms = self._pending.popleft()
            self._dur[kind].append(start.elapsed_time(end))
            self._dur[f"{kind}.step"].append(launch.elapsed_time(end))
            self._dur[f"{kind}.step_host"].append(launch_host_ms)
            if self._prev_end is not None:
                self._gap_ms += max(0.0, self._prev_end.elapsed_time(start))
            if self._gpu_first is None:
                self._gpu_first = start
            self._gpu_last = end
            self._prev_end = end
        now = time.monotonic()
        if now - self._t0 >= self._window_s and self._gpu_first is not None:
            self._report()
            self._reset(now)

    def _report(self) -> None:
        span = self._gpu_first.elapsed_time(self._gpu_last)
        parts = [f"gpu_step_timing span_ms={span:.0f} gap_ms={self._gap_ms:.1f}"]
        for kind in sorted(self._dur):
            v = self._dur[kind]
            parts.append(
                f"{kind}: n={len(v)} sum_ms={sum(v):.0f} "
                f"p50_ms={statistics.median(v):.2f} mean_ms={sum(v) / len(v):.2f}"
            )
        self.last_report = " | ".join(parts)
        logger.info(self.last_report)
