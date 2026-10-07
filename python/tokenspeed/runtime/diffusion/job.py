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

"""Single-flight video job queue for the diffusion HTTP surface.

Generation is minutes-long; the OpenAI-shaped video API is async
(``POST /v1/videos`` -> 202). Concurrent DiTs would thrash staged residency,
so depth defaults to 1 running job with an in-process FIFO.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

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
    def __init__(self, generate_fn: Callable[[H3GenerateRequest], H3GenerateResult], *, max_queue: int = 8):
        self._generate = generate_fn
        self._max_queue = max_queue
        self._jobs: dict[str, VideoJob] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._worker = threading.Thread(target=self._loop, name="ts-h3-worker", daemon=True)
        self._worker.start()

    def submit(self, request: H3GenerateRequest, *, model: str = "test") -> VideoJob:
        with self._lock:
            if len(self._pending) >= self._max_queue:
                raise RuntimeError(f"video job queue full (max_queue={self._max_queue})")
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
                    logger.info("h3 job %s done bytes=%s", job_id, len(result.mp4_bytes))
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
        from tokenspeed.runtime.diffusion.references import ReferenceUpload, cleanup_staged

        staged = [ReferenceUpload(path=p, kind="image") for p in paths]
        cleanup_staged(staged)

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()
        self._worker.join(timeout=5.0)
