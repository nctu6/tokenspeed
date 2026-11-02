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

"""MiniMax-Music3 serving engine: batched AR stage + streamed acoustic stage.

Two threads, one per device, both always resident:

* the AR thread admits songs (prefill), then runs batched frame steps
  (``Music3AREngine``) and hands every new frame to the song's acoustic state;
* the acoustic thread batches every window that is ready across songs
  (``Music3Acoustic``) and appends the cropped PCM to the song's job.

So a song's acoustic windows run while its AR stage is still going, and the
acoustic stage of many songs shares DiT forwards.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from tokenspeed.runtime.diffusion.music3_acoustic import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    AcousticSong,
    Music3Acoustic,
)
from tokenspeed.runtime.diffusion.music3_ar import (
    Music3AREngine,
    Music3ARRequest,
    build_prompt_ids,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["Music3Engine", "Music3Job"]

_END = object()

# The overflow acoustic worker (on the AR GPU) only runs while every song in
# its AR stage is at least this many windows ahead of its acoustic stage.
OVERFLOW_LEAD_WINDOWS = 2


@dataclass
class Music3Job:
    """One song's progress, consumed by the HTTP thread."""

    seed: int | None
    # (np [2, S], is_last) windows in order, then _END; or an exception.
    pieces: queue.Queue = field(default_factory=queue.Queue)
    num_frames: int = 0
    # Stage timestamps (time.monotonic) for the per-song timing log line.
    submitted_at: float = field(default_factory=time.monotonic)
    admitted_at: float | None = None
    ar_done_at: float | None = None

    def iter_pieces(self):
        """Yield ``(float32 [2, samples], is_last)`` windows; raise the job's error."""
        while True:
            item = self.pieces.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def waveform(self) -> np.ndarray:
        """Block until the song is done; the stitched ``[2, samples]`` waveform."""
        return np.concatenate([piece for piece, _ in self.iter_pieces()], axis=-1)


@dataclass
class _Song:
    job: Music3Job
    acoustic: AcousticSong
    # Set right after construction: its callbacks close over this song.
    ar: Music3ARRequest | None = None
    error: BaseException | None = None


class Music3Engine:
    def __init__(
        self,
        *,
        tokenizer,
        ar: Music3AREngine,
        acoustic: list[Music3Acoustic],
        max_window_batch: int,
    ):
        """``acoustic[0]`` is the dedicated acoustic device; any further entry
        shares a device with the AR stage and only takes windows while
        ``acoustic[0]`` is busy and another song's window is waiting.
        """
        if max_window_batch < 1:
            raise ValueError(f"max_window_batch must be >= 1, got {max_window_batch}")
        if not acoustic:
            raise ValueError("Music3Engine needs at least one acoustic worker")
        self.tokenizer = tokenizer
        self.ar = ar
        self.acoustic = acoustic
        self.max_window_batch = max_window_batch
        self._submitted: queue.Queue = queue.Queue()
        self._waiting: deque[_Song] = deque()
        self._songs: list[_Song] = []  # songs with acoustic work left
        self._busy = [False] * len(acoustic)
        self._lock = threading.Condition()
        self._stop = False
        self._ar_thread = threading.Thread(
            target=self._ar_loop, name="ts-music3-ar", daemon=True
        )
        self._acoustic_threads = [
            threading.Thread(
                target=self._acoustic_loop,
                args=(index,),
                name=f"ts-music3-acoustic-{index}",
                daemon=True,
            )
            for index in range(len(acoustic))
        ]
        self._ar_thread.start()
        for thread in self._acoustic_threads:
            thread.start()

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        *,
        prompt: str,
        lyrics: str,
        max_frames: int,
        seed: int | None,
        num_inference_steps: int,
    ) -> Music3Job:
        """Queue a song; returns its job. Tokenization runs on the caller thread."""
        ids = build_prompt_ids(self.tokenizer, prompt, lyrics)
        job = Music3Job(seed=seed)
        generator = None
        if seed is not None:
            # CPU generator: the song's windows may run on any acoustic device.
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        acoustic = AcousticSong(
            seed=seed, num_inference_steps=int(num_inference_steps), generator=generator
        )
        song = _Song(job=job, acoustic=acoustic)
        song.ar = Music3ARRequest(
            prompt_ids=ids,
            max_frames=int(max_frames),
            # An unseeded song still needs a sampling key; derive one per job.
            seed=int(seed) if seed is not None else (id(job) & 0x7FFFFFFF),
            on_frames=lambda frame, index, s=song: self._on_frames(s, frame, index),
            on_done=lambda total, error, s=song: self._on_ar_done(s, total, error),
        )
        self._submitted.put(song)
        with self._lock:
            self._lock.notify_all()
        return job

    # --------------------------------------------------------------- AR side
    def _on_frames(self, song: _Song, frame: torch.Tensor, index: int) -> None:
        with self._lock:
            song.acoustic.frames.append(frame)
            song.acoustic.frames_available = index + 1
            self._lock.notify_all()

    def _on_ar_done(self, song: _Song, total: int, error: BaseException | None) -> None:
        with self._lock:
            song.job.num_frames = total
            song.job.ar_done_at = time.monotonic()
            if error is not None:
                song.error = error
                song.acoustic.finished = True
                song.job.pieces.put(error)
                if song in self._songs:
                    self._songs.remove(song)
            else:
                song.acoustic.total_frames = total
            self._lock.notify_all()

    def _ar_loop(self) -> None:
        torch.cuda.set_device(self.ar.device)
        while not self._stop:
            while True:
                try:
                    self._waiting.append(self._submitted.get_nowait())
                except queue.Empty:
                    break
            while self._waiting and self.ar.can_admit(self._waiting[0].ar):
                song = self._waiting.popleft()
                try:
                    self.ar.admit(song.ar)
                except Exception as exc:  # noqa: BLE001 - reported to the caller
                    logger.exception("music3: admit failed")
                    song.job.pieces.put(exc)
                    continue
                song.job.admitted_at = time.monotonic()
                with self._lock:
                    self._songs.append(song)
            if not self.ar.active:
                with self._lock:
                    if self._submitted.empty() and not self._stop:
                        self._lock.wait(timeout=0.5)
                continue
            try:
                self.ar.step()
            except Exception as exc:  # noqa: BLE001 - fails the batch, not the server
                logger.exception("music3: AR step failed")
                self.ar.fail_all(exc)

    # --------------------------------------------------------- acoustic side
    def _ready_windows(self):
        ready = []
        for song in self._songs:
            window = song.acoustic.ready_window()
            if window is not None:
                ready.append((song, *window))
        return ready

    def _may_take(self, index: int) -> bool:
        """Whether acoustic worker ``index`` may start a window now.

        The dedicated worker always may. A worker sharing the AR GPU slows the
        AR stage (its fp32 DiT tiles hold the SMs even at low stream priority),
        so it only helps while the dedicated worker is busy and slowing the AR
        stage cannot starve the acoustic stage: every song still in its AR
        stage already has frames for ``OVERFLOW_LEAD_WINDOWS`` windows past
        its next one. If the AR stage falls back, the lead shrinks and the
        overflow worker stops until the AR stage is ahead again.
        """
        if index == 0:
            return True
        if not self._busy[0]:
            return False
        for song in self._songs:
            acoustic = song.acoustic
            if acoustic.total_frames is not None:
                continue  # AR finished; nothing to slow down for this song
            needed = (
                acoustic.next_window + OVERFLOW_LEAD_WINDOWS
            ) * CHUNK_HOP + CHUNK_FRAMES
            if acoustic.frames_available < needed:
                return False
        return True

    def _acoustic_loop(self, index: int) -> None:
        worker = self.acoustic[index]
        torch.cuda.set_device(worker.device)
        while not self._stop:
            with self._lock:
                ready = self._ready_windows() if self._may_take(index) else []
                while not ready and not self._stop:
                    self._lock.wait(timeout=0.5)
                    ready = self._ready_windows() if self._may_take(index) else []
                if self._stop:
                    return
                # One DiT batch: windows of equal length and step count,
                # oldest songs first.
                key = (
                    ready[0][2] - ready[0][1],
                    ready[0][0].acoustic.num_inference_steps,
                )
                group = [
                    item
                    for item in ready
                    if (item[2] - item[1], item[0].acoustic.num_inference_steps) == key
                ][: self.max_window_batch]
                for song, *_ in group:
                    song.acoustic.in_flight = True
                self._busy[index] = True
                self._lock.notify_all()
            try:
                pieces = worker.run_windows(
                    [
                        (song.acoustic, start, end, last)
                        for song, start, end, last in group
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - fails these songs only
                logger.exception("music3: acoustic window failed")
                with self._lock:
                    self._busy[index] = False
                    for song, *_ in group:
                        song.acoustic.in_flight = False
                        song.acoustic.finished = True
                        if song in self._songs:  # not already failed by the AR stage
                            song.job.pieces.put(exc)
                            self._songs.remove(song)
                    self._lock.notify_all()
                continue
            with self._lock:
                self._busy[index] = False
                self._lock.notify_all()
                for (song, *_), (piece, last) in zip(group, pieces):
                    song.acoustic.in_flight = False
                    if song not in self._songs:  # the AR stage failed it meanwhile
                        continue
                    song.job.pieces.put((piece, last))
                    if last:
                        song.job.pieces.put(_END)
                        self._songs.remove(song)
                        self._log_timing(song.job)

    @staticmethod
    def _log_timing(job: Music3Job) -> None:
        t0 = job.submitted_at
        logger.info(
            f"music3 song done: frames={job.num_frames} "
            f"admit={(job.admitted_at or t0) - t0:.2f}s "
            f"ar_done={(job.ar_done_at or t0) - t0:.2f}s "
            f"total={time.monotonic() - t0:.2f}s"
        )

    def shutdown(self) -> None:
        self._stop = True
        with self._lock:
            self._lock.notify_all()
        self._ar_thread.join(timeout=5.0)
        for thread in self._acoustic_threads:
            thread.join(timeout=5.0)
