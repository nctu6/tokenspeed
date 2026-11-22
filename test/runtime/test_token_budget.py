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
"""``TokenBudget``: the sum of held tokens never exceeds the capacity."""

import asyncio
import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.engine.token_budget import TokenBudget

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_concurrent_holders_never_exceed_capacity():
    budget = TokenBudget(100)
    peak = 0

    async def job(tokens: int) -> None:
        nonlocal peak
        async with budget.hold(tokens):
            peak = max(peak, budget.held)
            assert budget.held <= budget.capacity
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    async def main() -> None:
        await asyncio.gather(*(job(t) for t in [40, 30, 50, 20, 60, 10, 100, 25] * 5))

    asyncio.run(main())
    assert budget.held == 0
    assert 90 <= peak <= 100


def test_waiters_are_served_in_arrival_order():
    budget = TokenBudget(10)
    order = []

    async def job(name: str, tokens: int) -> None:
        async with budget.hold(tokens):
            order.append(name)
            await asyncio.sleep(0)

    async def main() -> None:
        await budget.acquire(8)
        # "big" arrives first and needs 8: it must not be passed by "small".
        big = asyncio.create_task(job("big", 8))
        await asyncio.sleep(0)
        small = asyncio.create_task(job("small", 2))
        await asyncio.sleep(0)
        assert order == []  # small would fit beside the 8 held, but big is ahead
        budget.release(8)
        await asyncio.gather(big, small)

    asyncio.run(main())
    assert order == ["big", "small"]


def test_cancelled_waiter_gives_its_place_and_tokens_back():
    budget = TokenBudget(10)

    async def main() -> None:
        await budget.acquire(10)
        waiting = asyncio.create_task(budget.acquire(6))
        behind = asyncio.create_task(budget.acquire(4))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        budget.release(10)
        await behind
        assert budget.held == 4

    asyncio.run(main())


def test_cancel_after_grant_releases_the_grant():
    budget = TokenBudget(10)

    async def main() -> None:
        await budget.acquire(10)
        waiting = asyncio.create_task(budget.acquire(10))
        await asyncio.sleep(0)
        budget.release(10)  # grants ``waiting`` (future resolved, task not run yet)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert budget.held == 0

    asyncio.run(main())


@pytest.mark.parametrize("tokens", [0, -1, 11])
def test_a_request_that_can_never_fit_is_rejected(tokens):
    budget = TokenBudget(10)
    with pytest.raises(ValueError):
        asyncio.run(budget.acquire(tokens))


def test_release_more_than_held_fails():
    budget = TokenBudget(10)
    with pytest.raises(ValueError):
        budget.release(1)
