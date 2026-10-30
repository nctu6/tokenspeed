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

"""Sync facade over ``AsyncLLM`` for blocking callers.

A dedicated daemon thread owns a private ``asyncio`` event loop that
drives ``AsyncLLM.generate_request`` coroutines. Sync callers submit
work via ``asyncio.run_coroutine_threadsafe``; streaming callers
consume results through a ``queue.Queue`` with three-valued
semantics (normal item / terminal sentinel / exception) so
exceptions propagate to the caller thread instead of getting
swallowed by the driver loop. Streaming callers that run on another
event loop (an ASGI server's) use the same protocol through an
``asyncio.Queue`` owned by their loop, fed with
``call_soon_threadsafe``, so they never await the bridge's loop.

Bridging at this layer — rather than maintaining a separate sync
IPC client — keeps ``AsyncLLM`` single-producer and the scheduler
IPC surface single-caller, while still letting ``Engine`` expose a
blocking API to callers that cannot own an event loop themselves.
"""

import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import GenerateReqInput

# Sentinel marking the end of a streaming bridge. Identity-compared,
# so a dict payload happening to equal this value is impossible.
_STREAM_END = object()


class LLM:
    """Sync adapter around an existing ``AsyncLLM`` instance.

    Owns a daemon thread running its own ``asyncio`` event loop; all
    coroutines are driven on that loop via
    ``asyncio.run_coroutine_threadsafe``. The adapter is cheap to
    construct and safe to share across caller threads — the bg loop
    is single-threaded, so ordering of queued work matches the
    submission order.
    """

    def __init__(self, async_llm: AsyncLLM) -> None:
        self.async_llm = async_llm
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tokenspeed-llm-sync",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro) -> Any:
        """Submit ``coro`` to the bg loop and block until it completes.

        This is the single sync entry point used by ``Engine`` for every
        blocking coroutine call (``flush_cache``, ``start_profile``,
        weight sync, internal-state queries, …). It replaces the
        deprecated ``asyncio.get_event_loop()`` +
        ``run_until_complete(coro)`` idiom that previously lived inline
        at every call site.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def generate(self, obj: GenerateReqInput) -> dict:
        """Run ``AsyncLLM.generate_request`` to completion and return the
        final dict. Used by non-streaming sync paths
        (``Engine.generate`` with ``stream=False``).
        """

        async def _one() -> dict:
            gen = self.async_llm.generate_request(obj)
            return await gen.__anext__()

        return self.run(_one())

    def generate_stream(self, obj: GenerateReqInput) -> Iterator[dict]:
        """Run ``AsyncLLM.generate_request`` and yield each output dict
        as it lands. Implements the three-valued queue bridge:

          * dicts are produced by the async generator → ``q.put(item)``
          * terminal → ``q.put(_STREAM_END)``
          * exception → ``q.put(exc)`` then ``q.put(_STREAM_END)``

        Safe to consume from any thread other than the bg loop's own,
        e.g. a Starlette threadpool worker behind a ``StreamingResponse``.

        If the caller stops before the stream ends (``break``, ``close()``,
        or the generator being garbage-collected after a client
        disconnect), the drain task is cancelled. ``CancelledError`` then
        reaches ``AsyncLLM._wait_one_response``, whose ``finally`` sends
        ``AbortReq`` to the scheduler, so the request does not keep
        decoding for nobody. The cancel is scheduled, not awaited: closing
        may happen on an event-loop thread (GC) that must not block.
        """
        q: queue.Queue[Any] = queue.Queue()
        drain = self._start_drain(obj, q.put)

        ended = False
        try:
            while True:
                item = q.get()
                if item is _STREAM_END:
                    ended = True
                    return
                if isinstance(item, BaseException):
                    ended = True
                    raise item
                yield item
        finally:
            if not ended:
                # Consumer left early: cancel the drain so the request is
                # aborted (see the docstring).
                drain.cancel()

    async def generate_stream_async(self, obj: GenerateReqInput) -> AsyncIterator[dict]:
        """:meth:`generate_stream` for a consumer on another event loop.

        For an ASGI server's loop (e.g. a ``StreamingResponse`` body): each
        output is handed to the consumer's loop with ``call_soon_threadsafe``
        into an ``asyncio.Queue`` that loop owns, so the consumer never
        awaits the bridge's loop and no worker thread is parked per stream or
        woken per output. Iterating :meth:`generate_stream` from a threadpool
        instead costs two thread handoffs per output; with many concurrent
        streams those handoffs contend for the GIL with the rest of the
        process. Measured on the Whisper ASR endpoint at 16 concurrent
        streams: the time from upload to engine entry of each request was
        80 ms with the threadpool relay and 18 ms with this one (14 ms
        non-streamed).

        Same three-valued protocol and early-close semantics as
        :meth:`generate_stream`: leaving early (``aclose()``, the consumer
        task being cancelled, or the generator being finalized) cancels the
        drain, which aborts the request. Must not be consumed on the bridge's
        own loop.
        """
        loop = asyncio.get_running_loop()
        if loop is self._loop:
            raise RuntimeError(
                "generate_stream_async must be consumed on a loop other than "
                "the LLM bridge's own; await AsyncLLM.generate_request there"
            )
        q: asyncio.Queue[Any] = asyncio.Queue()

        def put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                # The consumer's loop closed: nobody is left to read.
                pass

        drain = self._start_drain(obj, put)

        ended = False
        try:
            while True:
                item = await q.get()
                if item is _STREAM_END:
                    ended = True
                    return
                if isinstance(item, BaseException):
                    ended = True
                    raise item
                yield item
        finally:
            if not ended:
                drain.cancel()

    def _start_drain(
        self, obj: GenerateReqInput, put: Callable[[Any], None]
    ) -> concurrent.futures.Future:
        """Drive ``generate_request(obj)`` on the bridge's loop, handing every
        output to ``put``, then any exception, then :data:`_STREAM_END`."""

        async def _drain() -> None:
            pending_exc: BaseException | None = None
            try:
                async for item in self.async_llm.generate_request(obj):
                    put(item)
            except BaseException as exc:  # noqa: BLE001 — propagate anything
                pending_exc = exc
            finally:
                if pending_exc is not None:
                    put(pending_exc)
                put(_STREAM_END)

        return asyncio.run_coroutine_threadsafe(_drain(), self._loop)

    def shutdown(self) -> None:
        """Stop the bg event loop and join the thread.

        Idempotent. Safe to call before ``AsyncLLM`` teardown. The
        daemon flag means an accidentally-skipped shutdown will not
        hang interpreter exit, but leaving the loop running past
        ``AsyncLLM`` teardown will surface as "Event loop is closed"
        errors on later submissions, so call this before tearing
        down the engine.
        """
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
