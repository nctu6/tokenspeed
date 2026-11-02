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

"""CPU-only: the GPU dispatcher takes 1 request or a full-width batch, in order."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

from tokenspeed.runtime.diffusion.job import SHUTDOWN, PendingRequest, next_dispatch
from tokenspeed.runtime.diffusion.pipeline import H3GenerateRequest


def _item(name: str) -> PendingRequest:
    return PendingRequest(request=H3GenerateRequest(prompt=name), future=Future())


def _names(batch) -> list[str]:
    return [item.request.prompt for item in batch]


def test_full_width_batch_when_enough_requests_wait():
    pending = queue.Queue()
    for name in "abc":
        pending.put(_item(name))
    assert _names(next_dispatch(pending, width=2, gather_window_s=0.01)) == ["a", "b"]
    assert _names(next_dispatch(pending, width=2, gather_window_s=0.01)) == ["c"]


def test_partial_batch_runs_the_first_alone_and_keeps_order():
    pending = queue.Queue()
    for name in "ab":
        pending.put(_item(name))
    # Width 3 cannot be filled: run "a" alone, "b" stays first in line.
    assert _names(next_dispatch(pending, width=3, gather_window_s=0.01)) == ["a"]
    pending.put(_item("c"))
    pending.put(_item("d"))
    assert _names(next_dispatch(pending, width=3, gather_window_s=0.01)) == [
        "b",
        "c",
        "d",
    ]


def test_late_arrival_inside_the_window_joins_the_batch():
    pending = queue.Queue()
    pending.put(_item("a"))
    timer = threading.Timer(0.02, lambda: pending.put(_item("b")))
    timer.start()
    try:
        batch = next_dispatch(pending, width=2, gather_window_s=1.0)
    finally:
        timer.cancel()
    assert _names(batch) == ["a", "b"]


def test_width_one_never_waits_and_shutdown_ends_the_loop():
    pending = queue.Queue()
    pending.put(_item("a"))
    pending.put(SHUTDOWN)
    assert _names(next_dispatch(pending, width=1, gather_window_s=10.0)) == ["a"]
    assert next_dispatch(pending, width=1, gather_window_s=10.0) is None


def test_shutdown_inside_the_window_is_kept_for_the_next_call():
    pending = queue.Queue()
    pending.put(_item("a"))
    pending.put(SHUTDOWN)
    assert _names(next_dispatch(pending, width=2, gather_window_s=1.0)) == ["a"]
    assert next_dispatch(pending, width=2, gather_window_s=1.0) is None
