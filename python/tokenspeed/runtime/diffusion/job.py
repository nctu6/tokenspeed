# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""Video job queue and GPU dispatch batching for the diffusion HTTP surface.

Generation is minutes-long; the OpenAI-shaped video API is async
(``POST /v1/videos`` -> 202). All requests, sync and async, meet in one
FIFO that a single dispatcher drains (``next_dispatch``), so the GPUs only
ever see one dispatch at a time.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum

from tokenspeed.runtime.diffusion.pipeline import H3GenerateRequest, H3GenerateResult
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["VideoJobQueue", "JobStatus", "VideoJob"]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class VideoJob:
    id: str
    request: H3GenerateRequest
    status: JobStatus = JobStatus.queued
    created_at: float = field(default_factory=time.time)
    progress_step: int = 0
    progress_total: int = 0
    error: str | None = None
    result: H3GenerateResult | None = None
    model: str = "test"


class VideoJobQueue:
    """Async ``/v1/videos`` jobs.

    ``num_runners`` jobs may be in flight at once, matching how many requests
    one GPU dispatch can take; ``generate_fn`` is the server's serialized
    submit path, so runners never touch the GPUs concurrently themselves.
    """

    def __init__(
        self,
        generate_fn: Callable[[H3GenerateRequest], H3GenerateResult],
        *,
        num_runners: int,
        max_queue: int = 64,
    ):
        if num_runners < 1:
            raise ValueError(f"num_runners must be >= 1, got {num_runners}")
        self._generate = generate_fn
        self._max_queue = max_queue
        self._jobs: dict[str, VideoJob] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._workers = [
            threading.Thread(target=self._loop, name=f"ts-h3-job-{i}", daemon=True)
            for i in range(num_runners)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, request: H3GenerateRequest, *, model: str = "test") -> VideoJob:
        with self._lock:
            if len(self._pending) >= self._max_queue:
                raise RuntimeError(
                    f"video job queue full (max_queue={self._max_queue})"
                )
            job = VideoJob(id=f"video_{uuid.uuid4().hex}", request=request, model=model)
            job.progress_total = int(request.num_inference_steps or 0)
            self._jobs[job.id] = job
            self._pending.append(job.id)
        self._wake.set()
        return job

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _loop(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._pending or self._stop:
                        job_id = None
                    else:
                        job_id = self._pending.pop(0)
                        job = self._jobs[job_id]
                        job.status = JobStatus.running
                if job_id is None:
                    break
                try:
                    logger.info("h3 job %s start task=%s", job_id, job.request.task)
                    result = self._generate(job.request)
                    with self._lock:
                        job.result = result
                        job.status = JobStatus.completed
                        job.progress_step = job.progress_total
                    logger.info(
                        "h3 job %s done bytes=%s", job_id, len(result.mp4_bytes)
                    )
                except Exception as exc:
                    logger.exception("h3 job %s failed", job_id)
                    with self._lock:
                        job.status = JobStatus.failed
                        job.error = str(exc)
                finally:
                    self._cleanup_staged(job.request)

    @staticmethod
    def _cleanup_staged(request: H3GenerateRequest) -> None:
        extra = request.extra_params or {}
        paths = extra.pop("_ts_staged_refs", None) if isinstance(extra, dict) else None
        if not paths:
            return
        from tokenspeed.runtime.diffusion.references import (
            ReferenceUpload,
            cleanup_staged,
        )

        staged = [ReferenceUpload(path=p, kind="image") for p in paths]
        cleanup_staged(staged)

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()
        for worker in self._workers:
            worker.join(timeout=5.0)


# How long the dispatcher waits for more requests after the first one arrives
# before deciding the dispatch width. Requests of one burst reach the server a
# few ms apart; 50 ms is noise next to a minute-long generation.
DISPATCH_GATHER_WINDOW_S = 0.05


@dataclass
class PendingRequest:
    """One request waiting for the GPU dispatcher, with its caller's future."""

    request: H3GenerateRequest
    future: Future


SHUTDOWN = object()


def next_dispatch(
    pending: queue.Queue, *, width: int, gather_window_s: float
) -> list[PendingRequest] | None:
    """Block for the next dispatch: 1 request, or ``width`` if that many wait.

    A dispatch runs either one request (across every GPU) or exactly ``width``
    requests (one per GPU); in between, extra requests stay queued for the
    next dispatch, in arrival order. Returns ``None`` on ``SHUTDOWN``.
    """
    first = pending.get()
    if first is SHUTDOWN:
        return None
    batch = [first]
    if width <= 1:
        return batch
    deadline = time.monotonic() + gather_window_s
    taken: list[PendingRequest] = []
    while len(batch) + len(taken) < width:
        remaining = deadline - time.monotonic()
        try:
            item = (
                pending.get(timeout=max(0.0, remaining))
                if remaining > 0
                else pending.get_nowait()
            )
        except queue.Empty:
            break
        if item is SHUTDOWN:
            pending.put(SHUTDOWN)
            break
        taken.append(item)
    if len(batch) + len(taken) == width:
        return batch + taken
    # Not enough for a full-width dispatch: run the first alone and put the
    # others back at the head of the queue, in arrival order.
    _requeue_front(pending, taken)
    return batch


def _requeue_front(pending: queue.Queue, items: list) -> None:
    """Put ``items`` back at the front of ``pending``, preserving order."""
    if not items:
        return
    with pending.mutex:
        for item in reversed(items):
            pending.queue.appendleft(item)
        pending.not_empty.notify(len(items))
