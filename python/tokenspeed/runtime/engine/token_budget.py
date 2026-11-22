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
"""A FIFO token budget for requests that the scheduler must prefill whole."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator


class TokenBudget:
    """Counting semaphore over tokens: at most ``capacity`` held at any time.

    A pooling request reduces over its whole prompt, so the scheduler must
    prefill it in one chunk. The scheduler fills each round's token budget in
    arrival order and cuts the last request that does not fit into a partial
    chunk, which a pooling forward cannot take. Holding each request's prompt
    tokens here until its vector returns keeps the sum of all waiting prompts
    within one round's budget, so none is ever cut.

    Waiters are served in arrival order: a large request is not starved by a
    stream of small ones.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"token budget capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.held = 0
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    async def acquire(self, tokens: int) -> None:
        """Wait until ``tokens`` fit beside what is held, then hold them.

        Raises:
            ValueError: ``tokens`` is not in ``1..capacity``; it could never be
                granted.
        """
        if not 0 < tokens <= self.capacity:
            raise ValueError(
                f"{tokens} tokens cannot be held by a budget of {self.capacity}"
            )
        if not self._waiters and self.held + tokens <= self.capacity:
            self.held += tokens
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (tokens, waiter)
        self._waiters.append(entry)
        try:
            await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                # Granted in the same step the wait was cancelled: give it back.
                self.release(tokens)
            else:
                self._waiters.remove(entry)
                self._grant()
            raise

    def release(self, tokens: int) -> None:
        """Return ``tokens`` and wake the waiters that now fit, in order."""
        if not 0 < tokens <= self.held:
            raise ValueError(f"releasing {tokens} tokens with {self.held} held")
        self.held -= tokens
        self._grant()

    def _grant(self) -> None:
        while self._waiters:
            tokens, waiter = self._waiters[0]
            if self.held + tokens > self.capacity:
                return
            self._waiters.popleft()
            self.held += tokens
            waiter.set_result(None)

    @contextlib.asynccontextmanager
    async def hold(self, tokens: int) -> AsyncIterator[None]:
        """Hold ``tokens`` for the body of an ``async with``."""
        await self.acquire(tokens)
        try:
            yield
        finally:
            self.release(tokens)
