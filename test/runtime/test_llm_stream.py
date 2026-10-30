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

"""The sync ``LLM`` bridge's streaming iterator, consumed from a worker thread.

The bridge is the real one. Only ``AsyncLLM.generate_request`` (scheduler
processes and IPC) is a stand-in: it yields the cumulative text an engine
streams, one piece per step, when the request asks to stream, and the whole
text once otherwise. The held variant then waits for a step that never
comes, the way a request still decoding does, so a test can see whether an
early-closed iterator cancels it. ``AsyncLLM._wait_one_response`` turns that
cancellation into an ``AbortReq`` (covered by ``test_async_llm_cancel``).
"""

import asyncio
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import GenerateReqInput  # noqa: E402
from tokenspeed.runtime.engine.llm import LLM  # noqa: E402

# Generous: cancellation crosses one thread hop and one loop iteration.
_CANCEL_TIMEOUT_S = 5.0


def _output(text: str) -> dict:
    return {"text": text, "meta_info": {"finish_reason": None}}


class ScriptedAsyncLLM:
    """Stands in for the scheduler-backed ``AsyncLLM.generate_request``."""

    def __init__(self, pieces: list[str], hold_after: int | None) -> None:
        self.pieces = list(pieces)
        # After this many streamed pieces, wait forever (a request still
        # decoding); None streams every piece and finishes.
        self.hold_after = hold_after
        self.requests: list[GenerateReqInput] = []
        self.cancelled = threading.Event()

    async def generate_request(self, obj: GenerateReqInput):
        self.requests.append(obj)
        if not obj.stream:
            yield _output("".join(self.pieces))
            return
        text = ""
        try:
            for index, piece in enumerate(self.pieces):
                if index == self.hold_after:
                    await asyncio.Event().wait()
                text += piece
                yield _output(text)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _Failing:
    """A ``generate_request`` that fails after its first streamed output."""

    async def generate_request(self, obj: GenerateReqInput):
        yield _output("partial")
        raise RuntimeError("scheduler gone")


def _request(stream: bool) -> GenerateReqInput:
    return GenerateReqInput(input_ids=[1, 2, 3], stream=stream)


def _pieces(seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f" w{rng.randrange(1000)}" for _ in range(rng.randint(1, 12))]


@pytest.fixture
def make_llm():
    bridges: list[LLM] = []

    def make(async_llm) -> LLM:
        llm = LLM(async_llm)
        bridges.append(llm)
        return llm

    yield make
    for llm in bridges:
        llm.shutdown()


@pytest.mark.parametrize("seed", range(8))
def test_stream_yields_each_output_in_order_on_a_worker_thread(make_llm, seed):
    """**Validates: Requirements 2.6**"""
    pieces = _pieces(seed)
    scheduler = ScriptedAsyncLLM(pieces, hold_after=None)
    llm = make_llm(scheduler)

    # Starlette iterates a sync StreamingResponse body in a threadpool.
    with ThreadPoolExecutor(max_workers=1) as pool:
        outputs = pool.submit(lambda: list(llm.generate_stream(_request(True))))
        texts = [out["text"] for out in outputs.result()]

    assert texts == ["".join(pieces[: i + 1]) for i in range(len(pieces))]
    assert [req.stream for req in scheduler.requests] == [True]
    assert not scheduler.cancelled.is_set()


@pytest.mark.parametrize("seed", range(8))
def test_early_close_cancels_the_request(make_llm, seed):
    """**Validates: Requirements 2.6**"""
    pieces = _pieces(seed) + [" tail"]
    taken = random.Random(seed).randint(1, len(pieces) - 1)
    scheduler = ScriptedAsyncLLM(pieces, hold_after=taken)
    llm = make_llm(scheduler)

    stream = llm.generate_stream(_request(True))
    got = [next(stream)["text"] for _ in range(taken)]
    assert got[-1] == "".join(pieces[:taken])
    stream.close()  # the client went away

    assert scheduler.cancelled.wait(_CANCEL_TIMEOUT_S)
    # The bridge's loop survives the cancel and serves the next request.
    assert llm.generate(_request(False))["text"] == "".join(pieces)


def test_finished_stream_is_not_cancelled(make_llm):
    scheduler = ScriptedAsyncLLM([" a", " b"], hold_after=None)
    llm = make_llm(scheduler)

    stream = llm.generate_stream(_request(True))
    assert [out["text"] for out in stream] == [" a", " a b"]
    stream.close()

    llm.run(asyncio.sleep(0))
    assert not scheduler.cancelled.is_set()


def test_stream_error_reaches_the_consumer(make_llm):
    llm = make_llm(_Failing())

    stream = llm.generate_stream(_request(True))
    assert next(stream)["text"] == "partial"
    with pytest.raises(RuntimeError, match="scheduler gone"):
        next(stream)


def test_generate_returns_the_single_final_output(make_llm):
    scheduler = ScriptedAsyncLLM([" Hello", " world."], hold_after=None)
    llm = make_llm(scheduler)

    assert llm.generate(_request(False)) == _output(" Hello world.")
    assert [req.stream for req in scheduler.requests] == [False]


# --- generate_stream_async: the same bridge consumed on another event loop
# (an ASGI server's), as the ASR endpoint's StreamingResponse body does.


async def _collect(llm: LLM) -> list[str]:
    return [out["text"] async for out in llm.generate_stream_async(_request(True))]


@pytest.mark.parametrize("seed", range(8))
def test_async_stream_yields_each_output_in_order_on_another_loop(make_llm, seed):
    """**Validates: Requirements 2.6**"""
    pieces = _pieces(seed)
    scheduler = ScriptedAsyncLLM(pieces, hold_after=None)
    llm = make_llm(scheduler)
    texts = asyncio.run(_collect(llm))
    assert texts == ["".join(pieces[: i + 1]) for i in range(len(pieces))]
    assert [req.stream for req in scheduler.requests] == [True]
    assert not scheduler.cancelled.is_set()


@pytest.mark.parametrize("seed", range(4))
def test_async_streams_interleave_on_one_loop(make_llm, seed):
    # Many streams on one consumer loop, as concurrent SSE responses are:
    # each gets exactly its own outputs.
    rng = random.Random(seed)
    piece_sets = [_pieces(rng.randrange(10**6)) for _ in range(rng.randint(2, 6))]

    class _PerRequest:
        def __init__(self) -> None:
            self.scripts = {
                id(req): ScriptedAsyncLLM(p, hold_after=None)
                for req, p in zip(requests, piece_sets)
            }

        async def generate_request(self, obj):
            async for out in self.scripts[id(obj)].generate_request(obj):
                await asyncio.sleep(0)
                yield out

    requests = [_request(True) for _ in piece_sets]
    llm = make_llm(_PerRequest())

    async def one(req) -> list[str]:
        return [out["text"] async for out in llm.generate_stream_async(req)]

    async def main():
        return await asyncio.gather(*(one(req) for req in requests))

    for texts, pieces in zip(asyncio.run(main()), piece_sets):
        assert texts == ["".join(pieces[: i + 1]) for i in range(len(pieces))]


@pytest.mark.parametrize("how", ["aclose", "cancel"])
@pytest.mark.parametrize("seed", range(4))
def test_async_early_exit_cancels_the_request(make_llm, how, seed):
    """**Validates: Requirements 2.6**"""
    pieces = _pieces(seed) + [" tail"]
    taken = random.Random(seed).randint(1, len(pieces) - 1)
    scheduler = ScriptedAsyncLLM(pieces, hold_after=taken)
    llm = make_llm(scheduler)

    async def main() -> list[str]:
        stream = llm.generate_stream_async(_request(True))
        got = [(await stream.__anext__())["text"] for _ in range(taken)]
        if how == "aclose":
            await stream.aclose()  # the client went away
        else:
            # A disconnect cancels the task iterating the body.
            task = asyncio.ensure_future(stream.__anext__())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await stream.aclose()
        return got

    got = asyncio.run(main())
    assert got[-1] == "".join(pieces[:taken])
    assert scheduler.cancelled.wait(_CANCEL_TIMEOUT_S)
    assert llm.generate(_request(False))["text"] == "".join(pieces)


def test_async_finished_stream_is_not_cancelled(make_llm):
    scheduler = ScriptedAsyncLLM([" a", " b"], hold_after=None)
    llm = make_llm(scheduler)
    assert asyncio.run(_collect(llm)) == [" a", " a b"]
    llm.run(asyncio.sleep(0))
    assert not scheduler.cancelled.is_set()


def test_async_stream_error_reaches_the_consumer(make_llm):
    llm = make_llm(_Failing())

    async def main():
        stream = llm.generate_stream_async(_request(True))
        assert (await stream.__anext__())["text"] == "partial"
        with pytest.raises(RuntimeError, match="scheduler gone"):
            await stream.__anext__()

    asyncio.run(main())


def test_async_stream_refuses_the_bridge_loop(make_llm):
    scheduler = ScriptedAsyncLLM([" a"], hold_after=None)
    llm = make_llm(scheduler)

    async def on_bridge_loop():
        return await _collect(llm)

    with pytest.raises(RuntimeError, match="other than"):
        llm.run(on_bridge_loop())
    assert scheduler.requests == []
