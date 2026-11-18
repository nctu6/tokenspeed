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

"""The request an EPD encode server's ranks take: one multimodal item to
embed and ship to the prefill ranks that wait for it.

The encode server's frontend sends it to the first encode rank over the
connection that rank dialed; the ranks' host (``ts-host`` in encode mode,
``epd/encode_host.py``) relays, batches and ships it. This module only
defines the wire struct.
"""

from __future__ import annotations

from tokenspeed.runtime.cache.embedding_cache import (
    EmbeddingCache,
    TieredEmbeddingCache,
)
from tokenspeed.runtime.engine.io_struct import BaseReq
from tokenspeed.runtime.epd.encode_scheduler import EncodeScheduler, PendingEncodeItem
from tokenspeed.runtime.multimodal.embedder import _item_token_count
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem
from tokenspeed.runtime.utils import get_colorful_logger


class EncodeRequest(BaseReq, kw_only=True):
    """One encode request: its prefill peers' rendezvous and its item.

    ``bootstrap_host``/``port`` name the encode server's bootstrap server
    and ``bootstrap_room`` the room the prefill ranks receive the item's
    embedding under (one room per item, so one item per request).

    A tagged IPC struct (``BaseReq``): it rides the request channel as
    msgpack, its item's pixels as ext-encoded tensor buffers or SHM handles
    (``ts-wire``'s ``EncodeRequest`` decodes it on the host).
    """

    request_id: str
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
    items: list[MultimodalDataItem]


# --- definitions restored from the pre-merge branch (D3-3) ---

logger = get_colorful_logger(__name__)


def _nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


class EncodeWorker:
    """Orchestrates cache + scheduler + executor for the encode role.

    Injected with the executor (real ``DisaggEncodeExecutor`` or a fake), an
    ``EncodeScheduler`` and an embedding cache (single-tier ``EmbeddingCache`` or
    the two-tier ``TieredEmbeddingCache``; only ``get``/``put`` are used) so the
    control flow is testable without a GPU or transport.
    """

    def __init__(
        self,
        executor,
        scheduler: EncodeScheduler,
        cache: EmbeddingCache | TieredEmbeddingCache,
    ):
        self.executor = executor
        self.scheduler = scheduler
        self.cache = cache
        # (request_id, item_index) -> item awaiting the tower
        self._pending: dict = {}

    def submit(self, request: EncodeRequest) -> None:
        self.executor.register(
            request.request_id,
            request.bootstrap_host,
            request.bootstrap_port,
            request.bootstrap_room,
        )
        for idx, item in enumerate(request.items):
            cached = self.cache.get(item.hash)
            if item.feature_shm is not None:
                # EPD pixel-SHM: the servicer published pixels to POSIX SHM and
                # the ZMQ hop carried only this handle (hash/pad_value were set on
                # the real tensor before publish). consume() unlinks, so segments
                # never outlive the item: materialize on a miss, and on a hit still
                # consume-and-drop to unlink the unused segment.
                handle, item.feature_shm = item.feature_shm, None
                handle.attach()
                if cached is None:
                    item.feature = handle.consume()
                else:
                    handle.consume()
            if cached is not None:
                # Cache hit: tower skipped, but the embedding still must reach
                # the prefill peer, so ship it directly. Entries are
                # (main, deepstack) pairs and BOTH halves must be restored, else
                # the prefill publishes a never-written deepstack buffer. Tolerate
                # a bare tensor for legacy/test-seeded entries.
                if isinstance(cached, tuple):
                    item.encoded, item.encoded_deepstack = cached
                else:
                    item.encoded = cached
                self.executor.send_item(request.request_id, item)
            else:
                self.scheduler.add(
                    PendingEncodeItem(
                        request_id=request.request_id,
                        item_index=idx,
                        cost=_item_token_count(item),
                    )
                )
                self._pending[(request.request_id, idx)] = item

    def step(self) -> int:
        """Run one scheduler batch through the tower + transfer. Returns the
        number of items encoded (0 when nothing is pending)."""
        self.executor.reap_concluded_senders({rid for (rid, _idx) in self._pending})
        # Retry sends that couldn't lease a ring slot last tick (non-blocking).
        self.executor.drain_deferred()
        # Backpressure: if sends are STILL deferred after the drain, the bounce
        # ring is saturated (all slots hold in-flight transfers). Pulling more ViT
        # now would only pile fresh embeddings into _deferred_sends -- each pins a
        # GPU embedding tensor with no slot to ship it, growing an unbounded
        # backlog into an OOM. Skip this tick; the loop yields the GIL (encode_loop
        # sees has_deferred) so the transfer daemons free slots, then we resume.
        if self.executor.has_deferred():
            return 0
        batch = self.scheduler.next_batch()
        if not batch:
            return 0
        request_items = [(p.request_id, self._pending[p.key]) for p in batch]
        try:
            self.executor.execute(request_items)
        except Exception as e:
            # A tower-step contract violation (ViT output not matching the items'
            # post-merge token count, or the forward itself) must fail only the
            # rooms in THIS batch, not propagate out into the engine's SIGUSR1
            # handler, which would kill the worker and drop every other request's
            # in-flight image. These raises fire before any send is issued, so
            # concluding the batch Failed never poisons an already-shipped room.
            # Per-item STAGING errors are handled finer-grained inside
            # _stage_and_send -> _fail_staged_room.
            n_failed = self.executor.fail_rooms((rid for rid, _ in request_items), e)
            for p in batch:
                self._pending.pop(p.key, None)
            logger.error(
                f"encode batch failed ({n_failed:d} rooms concluded Failed): {e!s}",
            )
            return 0
        for p in batch:
            item = self._pending.pop(p.key)
            if item.encoded is not None:
                # Cache the (main, deepstack) PAIR: caching only the main half
                # would make every later hit ship a deepstack-less transfer,
                # publishing uninitialized rows on the prefill.
                deep = item.encoded_deepstack
                nbytes = _nbytes(item.encoded) + (
                    _nbytes(deep) if deep is not None else 0
                )
                self.cache.put(item.hash, (item.encoded, deep), nbytes)
        return len(batch)

    def has_pending(self) -> bool:
        return self.scheduler.pending_size() > 0

    def has_deferred(self) -> bool:
        """True while sends are queued waiting for a free ring slot (executor)."""
        return self.executor.has_deferred()
