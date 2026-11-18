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

"""GPU memory a sleeping engine releases (``release_memory_occupation``)
and restores on wake, one :class:`MemoryRegion` per tag.

The Rust control plane drains the engine and orders the regions (the KV
cache first, NCCL last; resume in reverse); the data plane frees and
restores each one. Each tag is released and restored independently, so the
RL loop can resume weights (and NCCL), push fresh weights, then resume the KV
cache:

- ``weights``: model parameters (CPU backup, or discard until a complete
  replica-committed update; frozen draft allocations retain CPU backup);
- ``kv_cache``: the cache arena (discarded; zeroed on resume);
- ``cuda_graph``: the decode CUDA graph memory pool (discarded; every replay
  rewrites it before reading);
- ``nccl``: NCCL communicator buffers (suspended and resumed in place); only
  on an engine whose NCCL can suspend a communicator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    IsSleepingReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)


@dataclass(frozen=True)
class MemoryRegion:
    """How one releasable tag frees and restores its GPU memory.

    Attributes:
        release: Frees the memory; runs once the scheduler has drained.
        resume: Restores the memory and whatever state must accompany it
            (zeroed KV, transfer-engine registration) before scheduling
            resumes.
    """

    release: Callable[[], None]
    resume: Callable[[], None]


def memory_regions(
    device, adapter, *, nccl_suspendable: bool
) -> dict[str, MemoryRegion]:
    """The releasable GPU-memory regions of an engine, keyed by tag.

    The memory-saver ``adapter`` unmaps and re-maps the allocator regions;
    the device handle does the data-plane work that must surround them.
    Which pools and peers that work touches is the handle's business; this
    side only says when. ``nccl_suspendable`` says whether the process's
    NCCL can suspend communicators; without it there is no ``nccl`` region.
    """

    def release_kv_cache() -> None:
        # Wait for in-flight copies and deregister the arena from the PD
        # transfer engine before its memory goes away.
        device.prepare_kv_release()
        adapter.pause(tag="kv_cache")

    def resume_kv_cache() -> None:
        device.before_kv_resume()
        adapter.resume(tag="kv_cache")
        # Re-mapped memory holds garbage: zero it, then register it with
        # the PD transfer engine again.
        device.finish_kv_resume()

    def allocator_region(tag: str) -> MemoryRegion:
        def release() -> None:
            device.synchronize()
            adapter.pause(tag=tag)

        return MemoryRegion(release=release, resume=lambda: adapter.resume(tag=tag))

    regions = {
        "kv_cache": MemoryRegion(release=release_kv_cache, resume=resume_kv_cache),
        "weights": MemoryRegion(
            release=lambda: device.release_weights(adapter),
            resume=lambda: device.resume_weights(adapter),
        ),
        "cuda_graph": allocator_region("cuda_graph"),
    }
    if nccl_suspendable:
        regions["nccl"] = MemoryRegion(
            release=device.suspend_nccl, resume=device.resume_nccl
        )
    return regions


# --- definitions restored from the pre-merge branch (D3-3) ---

if TYPE_CHECKING:
    # The pause module builds this module's regions (PauseHooks), so a
    # runtime import here would be circular.
    from tokenspeed.runtime.engine.pause import PauseController


# Release order: the KV cache first (a PD engine deregisters it from the
# transfer engine before it is unmapped), NCCL last (nothing may issue a
# collective after it is suspended). Resume runs the reverse order, so NCCL and
# the graphs are back before the weights and the KV cache are used again.
VALID_TAGS = ("kv_cache", "weights", "cuda_graph", "nccl")


RESUME_ORDER = tuple(reversed(VALID_TAGS))


def _normalize_tags(
    tags: list[str] | None, order: tuple[str, ...], available
) -> tuple[list[str] | None, str | None]:
    """Return ``(ordered_tags, error)``. ``None`` ⇒ every ``available`` tag.
    The order is fixed (``order``) for determinism; an unknown tag, or one this
    engine cannot release, yields an error and ``None`` tags."""
    if tags is None:
        return [t for t in order if t in available], None
    bad = [t for t in tags if t not in VALID_TAGS]
    if bad:
        return None, f"invalid tags: {bad!r}; valid: {list(VALID_TAGS)}"
    unavailable = [t for t in tags if t not in available]
    if unavailable:
        return None, (
            f"tags not releasable on this engine: {unavailable!r}; "
            f"available: {[t for t in VALID_TAGS if t in available]!r}"
        )
    return [t for t in order if t in tags], None


class MemoryOccupationController:
    """Owns the data-plane release/resume for one scheduler event loop."""

    def __init__(
        self,
        *,
        send_func,
        pause_controller: PauseController,
        regions: dict[str, MemoryRegion],
        enabled: bool,
        reset_caches_fn: Callable[[], bool],
        transfers_idle_fn: Callable[[], bool],
        release_refusal_fn: Callable[[], str | None],
        weights_busy_fn: Callable[[], bool],
        kv_cache_release_allowed: bool = True,
    ) -> None:
        unknown = [t for t in regions if t not in VALID_TAGS]
        if unknown:
            raise ValueError(f"memory regions for unknown tags {unknown!r}")
        self._send = send_func
        self._pause = pause_controller
        # The tags this engine can release (``nccl`` needs an NCCL that can
        # suspend a communicator); a release without tags frees all of them.
        self._regions = regions
        self._enabled = enabled
        self._reset_caches = reset_caches_fn
        # A disaggregated engine must not free the KV cache while a peer can
        # still read or write it: the release waits until no request is
        # bootstrapping or holds pages pinned for a transfer.
        self._transfers_idle = transfers_idle_fn
        # Refuse releases that cannot drain (buffered PD requests or an active
        # rebalance). Already queued weight ops must finish through the normal
        # same-round gate before any region is released.
        self._release_refusal = release_refusal_fn
        self._weights_busy = weights_busy_fn
        # Whether releasing the ``kv_cache`` region is safe in this engine. False
        # when prefix caching is on but the scheduler exposes no prefix-cache
        # reset: discarded KV would leave stale cache entries that survive wake
        # and produce silently wrong hits, so we reject the release up front
        # rather than free KV we cannot invalidate. A declared capability (set
        # at construction), not a runtime duck-typed probe.
        self._kv_cache_release_allowed = kv_cache_release_allowed
        self.released_tags: set[str] = set()
        # True from an accepted release request until it frees memory or
        # fails, so admission can tell a memory release from a plain pause.
        self._release_pending = False

    @property
    def is_sleeping(self) -> bool:
        """True while any GPU-memory region is released (data-plane sleep)."""
        return bool(self.released_tags)

    @property
    def is_releasing_or_released(self) -> bool:
        """True while a release is draining or any region is released.

        A disaggregated engine refuses new requests in this window: their
        transfer registration would name the KV memory the release is about to
        unmap and re-register.
        """
        return self._release_pending or bool(self.released_tags)

    def refuse_weight_load(self) -> str | None:
        """Why a weight load cannot run now, or None when it can.

        A load writes the weights region and may issue collectives, so both
        the ``weights`` and the ``nccl`` regions must be resident.
        """
        released = [t for t in ("weights", "nccl") if t in self.released_tags]
        if not released:
            return None
        return (
            f"memory regions {released!r} are released; resume them with "
            "resume_memory_occupation before updating weights"
        )

    # -- control-request handlers (driven by the request handler) -------------

    def handle_release(self, req: ReleaseMemoryOccupationReqInput) -> None:
        if not self._enabled:
            self._fail_release("memory saver not enabled (--enable-memory-saver)")
            return
        tags, err = _normalize_tags(req.tags, VALID_TAGS, self._regions)
        if err is not None:
            self._fail_release(err)
            return
        if "kv_cache" in tags and not self._kv_cache_release_allowed:
            self._fail_release(
                "cannot release kv_cache: prefix caching is enabled but the "
                "scheduler has no prefix-cache reset, so discarded KV would "
                "leave stale cache entries after wake"
            )
            return
        already = [t for t in tags if t in self.released_tags]
        if already:
            self._fail_release(f"tags already released: {already!r}")
            return
        if self._pause.is_drain_pending:
            self._fail_release("a pause or release is already in progress")
            return
        refusal = self._release_refusal()
        if refusal is not None:
            self._fail_release(f"cannot release memory: {refusal}")
            return
        # Defer the actual free until the scheduler drains. wait-mode: let
        # in-flight requests finish (the RL rollout is idle between steps).
        # A post-finish L2 transfer can outlive the request; keep the release
        # pending until ClearL1Cache can atomically invalidate its cache refs,
        # until no PD transfer still touches the cache, and until no gated op
        # is still writing the weights.
        reset_caches = self._reset_caches if "kv_cache" in tags else None

        def ready() -> bool:
            if self._weights_busy() or not self._transfers_idle():
                return False
            return reset_caches is None or reset_caches()

        self._release_pending = True
        self._pause.request_drain(
            abort_inflight=False,
            on_drained=lambda: self._finish_release(tags),
            on_cancelled=lambda: self._fail_release("resumed before release drained"),
            ready=ready,
        )

    def _finish_release(self, tags: list[str]) -> None:
        self._release_pending = False
        for tag in tags:
            self._regions[tag].release()
            self.released_tags.add(tag)
        self._pause.set_released(True)
        self._send.send_pyobj(ReleaseMemoryOccupationReqOutput(success=True))

    def _fail_release(self, message: str) -> None:
        self._release_pending = False
        self._send.send_pyobj(
            ReleaseMemoryOccupationReqOutput(success=False, message=message)
        )

    def handle_resume(self, req: ResumeMemoryOccupationReqInput) -> None:
        # ``None`` on resume means "wake exactly what is asleep" — NOT all valid
        # tags. Expanding to all (as on release) would make a bare resume after a
        # partial release (e.g. weights-only) fail on the never-released tag,
        # stranding the released region. Resolve against ``released_tags`` so a
        # partial release round-trips with a default resume_memory_occupation().
        if req.tags is None:
            tags = [t for t in RESUME_ORDER if t in self.released_tags]
        else:
            tags, err = _normalize_tags(req.tags, RESUME_ORDER, self._regions)
            if err is not None:
                self._send.send_pyobj(
                    ResumeMemoryOccupationReqOutput(success=False, message=err)
                )
                return
        not_released = [t for t in tags if t not in self.released_tags]
        if not_released:
            self._send.send_pyobj(
                ResumeMemoryOccupationReqOutput(
                    success=False, message=f"tags not released: {not_released!r}"
                )
            )
            return
        if not tags and self._pause.is_drain_pending:
            # No region needs waking; preserve the current drain owner.
            self._send.send_pyobj(ResumeMemoryOccupationReqOutput(success=True))
            return
        for tag in tags:
            self._regions[tag].resume()
            self.released_tags.discard(tag)
        if not self.released_tags:
            # Fully awake → resume scheduling.
            self._pause.set_released(False)
        self._send.send_pyobj(ResumeMemoryOccupationReqOutput(success=True))

    def handle_is_sleeping(self, req: IsSleepingReqInput) -> None:
        self._send.send_pyobj(IsSleepingReqOutput(is_sleeping=self.is_sleeping))
