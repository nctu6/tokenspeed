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

"""Local-disk L3 store: packed Host CacheBlocks in segment files on NVMe.

One ``batch_put_from`` call is written as one or more *segment* files. A
segment is a header (magic, a CRC32 of the entry table, and per object its
key, payload offset, size and payload CRC32) padded to 4 KiB, followed by
the payloads packed back to back and padded to 4 KiB at the end, so the
whole file is read and written with O_DIRECT in one call. Objects written
together are usually read together (the pages of one prefix), so a read
opens each segment once and copies every requested object out of it; small
KV objects (tens of KiB) are otherwise bound by per-file CPU cost, not by
the drive.

Reads and writes of one ``batch_*`` call run in parallel, one contiguous
slice of segments per thread (``os.preadv`` / ``os.pwritev``, ``zlib.crc32``
and ``ctypes.memmove`` release the GIL), through a page-aligned per-thread
bounce buffer.

Puts are atomic (written to a temporary name, then renamed), so a reader
never sees a partial segment; a torn segment fails the header or payload
CRC and reads as a miss. The key -> segment index lives in memory and is
rebuilt from the headers at start-up, so objects survive a restart.
Capacity is bounded per segment: a put that does not fit evicts
least-recently-used segments (with every object in them) first.

An attention-TP group whose ranks cache identical bytes shares one
directory: only its rank 0 writes and deletes, and every rank derives the
same segment names and offsets from the same puts (``FileKvStore``). The
group reads through the page cache, so the drive serves each file once.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import mmap
import os
import struct
import threading
import uuid
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from tokenspeed.runtime.cache.l3.backend import host_buffer_ptr

logger = logging.getLogger(__name__)

ALIGN = 4096
# Smallest header region; a segment with many or long keys takes more.
HEADER_BYTES = ALIGN
# A put is split so no segment exceeds this many payload bytes (one object
# larger than it gets a segment of its own) ...
MAX_SEGMENT_BYTES = 8 << 20
# ... and split across the I/O threads only into segments of at least this
# size: a few small objects (one page of a prefix) stay one file.
MIN_SPLIT_BYTES = 1 << 20
_MAGIC = b"TSL3SEG1"
# magic, header region bytes, entry count, CRC32 of the entry table
_PREFIX = struct.Struct("<8sIII")
# payload offset in the file, payload bytes, payload CRC32 (0 when not
# verified), key bytes; the key follows each entry
_ENTRY = struct.Struct("<QQIH")
_MAX_KEY_BYTES = 1024
_SUFFIX = ".seg"
_LEGACY_SUFFIX = ".kv"


def _round_up(value: int, multiple: int = ALIGN) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class FileStoreConfig:
    """Settings of the local-disk L3 store; every field is required.

    ``path``: directory for the segments (a local NVMe file system).
    ``capacity_gb``: bytes on disk each rank may use, in GiB, headers and
    padding included; least-recently-used segments are evicted beyond it.
    ``io_threads``: parallel reads/writes per ``batch_*`` call.
    ``o_direct``: bypass the page cache (start-up fails if the file system
    refuses O_DIRECT).
    ``verify_crc``: store a CRC32 per object and reject reads that mismatch.
    ``fsync``: flush each segment to the device before it becomes visible.
    """

    path: str
    capacity_gb: float
    io_threads: int
    o_direct: bool
    verify_crc: bool
    fsync: bool

    @property
    def capacity_bytes(self) -> int:
        return int(self.capacity_gb * (1 << 30))


def parse_file_store_config(extra_config: str | None) -> FileStoreConfig:
    """Parse ``--kvstore-storage-backend-extra-config`` for the file store."""

    if not extra_config:
        raise ValueError(
            "the file L3 store needs --kvstore-storage-backend-extra-config with "
            '"path", "capacity_gb", "io_threads", "o_direct", "verify_crc", "fsync"'
        )
    raw = json.loads(extra_config)
    if not isinstance(raw, dict):
        raise ValueError("file L3 extra config must be a JSON object")
    fields = {
        "path": str,
        "capacity_gb": (int, float),
        "io_threads": int,
        "o_direct": bool,
        "verify_crc": bool,
        "fsync": bool,
    }
    missing = [name for name in fields if name not in raw]
    unknown = sorted(set(raw) - set(fields))
    if missing or unknown:
        raise ValueError(f"file L3 extra config: missing {missing}, unknown {unknown}")
    for name, kind in fields.items():
        value = raw[name]
        # bool is an int subclass: keep the numeric fields numeric.
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            raise ValueError(f"file L3 extra config: {name!r} has the wrong type")
    if raw["capacity_gb"] <= 0 or raw["io_threads"] <= 0:
        raise ValueError("file L3 extra config: capacity_gb and io_threads must be > 0")
    return FileStoreConfig(**raw)


class _AlignedBuffer:
    """A page-aligned anonymous mapping that grows on demand (one per thread)."""

    def __init__(self) -> None:
        self._map: mmap.mmap | None = None
        self.address = 0
        self.size = 0

    def ensure(self, size: int) -> None:
        if size <= self.size:
            return
        self._map = mmap.mmap(-1, _round_up(size))
        self.size = len(self._map)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))

    def view(self, size: int, offset: int = 0) -> memoryview:
        return memoryview(self._map)[offset : offset + size]


def _host_view(address: int, size: int) -> memoryview:
    return memoryview((ctypes.c_char * size).from_address(address)).cast("B")


def _skip(views: list[memoryview], done: int) -> list[memoryview]:
    """The part of ``views`` after the first ``done`` bytes."""

    remaining = []
    for view in views:
        if done >= len(view):
            done -= len(view)
            continue
        remaining.append(view[done:])
        done = 0
    return remaining


def _read_full(fd: int, views: list[memoryview], offset: int) -> int:
    """preadv until every view is full or EOF; return bytes read."""

    total = 0
    want = sum(len(view) for view in views)
    while total < want:
        got = os.preadv(fd, _skip(views, total), offset + total)
        if got <= 0:
            break
        total += got
    return total


def _write_full(fd: int, views: list[memoryview]) -> None:
    total = 0
    want = sum(len(view) for view in views)
    while total < want:
        written = os.pwritev(fd, _skip(views, total), total)
        if written <= 0:
            raise OSError(errno.EIO, "short write to L3 file")
        total += written


def _aligned(address: int, size: int) -> bool:
    return address % ALIGN == 0 and size % ALIGN == 0


@dataclass(frozen=True)
class _Entry:
    """Where one object lives: its segment, payload offset, size and CRC."""

    segment: str
    offset: int
    size: int
    crc: int


@dataclass
class _Segment:
    """One segment file: its on-disk bytes and the keys still served from it."""

    disk: int
    live: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _PutObject:
    index: int
    key: str
    address: int
    size: int


def _header_bytes(keys: Sequence[bytes]) -> int:
    table = _PREFIX.size + sum(_ENTRY.size + len(key) for key in keys)
    return _round_up(table)


def _parse_header(
    view: memoryview,
) -> tuple[int, list[tuple[str, int, int, int]]] | None:
    """Return (header bytes, [(key, offset, size, crc)]) or None if invalid."""

    magic, header, count, table_crc = _PREFIX.unpack_from(view, 0)
    if magic != _MAGIC or header < ALIGN or header % ALIGN or header > len(view):
        return None
    entries = []
    pos = _PREFIX.size
    for _ in range(count):
        if pos + _ENTRY.size > header:
            return None
        offset, size, crc, key_len = _ENTRY.unpack_from(view, pos)
        pos += _ENTRY.size
        if pos + key_len > header:
            return None
        key = bytes(view[pos : pos + key_len]).decode(errors="replace")
        pos += key_len
        entries.append((key, offset, size, crc))
    if zlib.crc32(view[_PREFIX.size : pos]) != table_crc:
        return None
    return header, entries


class FileKvStore:
    """L3 byte store on a local directory (see the module docstring)."""

    def __init__(
        self, config: FileStoreConfig, *, rank: int, shared_owner_rank: int | None
    ) -> None:
        """Open (or create) this rank's store.

        ``rank`` is the process's global rank. ``shared_owner_rank`` is None
        when every rank keeps its own copy (one subdirectory per rank), or
        the global rank of the attention-TP rank 0 when the group caches the
        same bytes on every rank (``CacheRecipe.kv_identical_across_attn_tp``):
        the group then shares the owner's subdirectory, only the owner writes
        and deletes files, and every rank indexes the same segment names and
        offsets for the same puts. A non-owner records the CRC of its own
        Host bytes and reads verify against it, so a rank whose bytes differ
        from the owner's reads a miss, never the owner's data.
        """

        self.config = config
        self._shared = shared_owner_rank is not None
        owner = int(rank) if shared_owner_rank is None else int(shared_owner_rank)
        self._writer = owner == int(rank)
        self._verify = config.verify_crc or self._shared
        # One subdirectory per owner: it indexes, sizes and evicts only its
        # own segments even when ranks share the configured path.
        self.root = os.path.join(config.path, f"rank{owner}")
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self._flags = os.O_DIRECT if config.o_direct else 0
        # The ranks of a shared group read the same files at about the same
        # time: through the page cache the drive serves each file once.
        self._read_flags = 0 if self._shared else self._flags
        if config.o_direct:
            self._check_o_direct(rank=int(rank))
        self._lock = threading.Lock()
        self._keys: dict[str, _Entry] = {}
        # segment name -> segment, least recently used first
        self._segments: OrderedDict[str, _Segment] = OrderedDict()
        self._used = 0
        self._local = threading.local()
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=config.io_threads, thread_name_prefix="l3-file"
        )
        self._rebuild_index()

    # -- layout -------------------------------------------------------------

    def _segment_path(self, name: str) -> str:
        return os.path.join(self.root, name[:2], name + _SUFFIX)

    def _buffer(self) -> _AlignedBuffer:
        buf = getattr(self._local, "buffer", None)
        if buf is None:
            buf = _AlignedBuffer()
            self._local.buffer = buf
        return buf

    def _check_o_direct(self, *, rank: int) -> None:
        probe = os.path.join(self.root, f".o_direct_probe.{rank}")
        try:
            fd = os.open(
                probe, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600
            )
        except OSError as exc:
            raise ValueError(
                f"file L3: {self.config.path!r} does not support O_DIRECT "
                f'({exc}); set "o_direct": false'
            ) from exc
        os.close(fd)
        os.unlink(probe)

    def _rebuild_index(self) -> None:
        """Index the segments already on disk; drop temporaries and torn files.

        Only the writer deletes: in a shared directory the other ranks may
        scan while the owner is still writing or cleaning up.
        """

        found = []
        for root, _dirs, files in os.walk(self.root):
            for name in files:
                path = os.path.join(root, name)
                if not name.endswith(_SUFFIX):
                    # Temporaries of an interrupted put, and one-object files
                    # of the earlier layout (".kv"), which nothing reads.
                    if self._writer and (
                        ".tmp." in name or name.endswith(_LEGACY_SUFFIX)
                    ):
                        os.unlink(path)
                    continue
                parsed = self._read_segment_header(path)
                if parsed is None:
                    if self._writer:
                        os.unlink(path)
                    continue
                disk, entries = parsed
                found.append(
                    (os.stat(path).st_mtime, name[: -len(_SUFFIX)], disk, entries)
                )
        for _mtime, name, disk, entries in sorted(found):
            segment = _Segment(disk=disk)
            self._segments[name] = segment
            self._used += disk
            for key, offset, size, crc in entries:
                # A key written twice (two racing puts) is served by the newer
                # segment; the older copy stays until its segment is evicted.
                if key in self._keys:
                    self._drop_key_locked(key)
                self._keys[key] = _Entry(name, offset, size, crc)
                segment.live.add(key)
        for name in [name for name, seg in self._segments.items() if not seg.live]:
            self._delete_segment_locked(name)
        if found:
            logger.info(
                f"[L3 file] indexed {len(self._keys)} objects in "
                f"{len(self._segments)} segments, "
                f"{self._used / (1 << 30):.2f} GiB under {self.root}"
            )

    def _read_segment_header(
        self, path: str
    ) -> tuple[int, list[tuple[str, int, int, int]]] | None:
        try:
            fd = os.open(path, os.O_RDONLY | self._flags)
        except OSError:
            return None
        try:
            disk = os.fstat(fd).st_size
            buf = self._buffer()
            buf.ensure(ALIGN)
            if _read_full(fd, [buf.view(ALIGN)], 0) != ALIGN:
                return None
            header = _PREFIX.unpack_from(buf.view(ALIGN), 0)[1]
            if header > ALIGN:
                if header % ALIGN or header > disk:
                    return None
                buf.ensure(header)
                if _read_full(fd, [buf.view(header)], 0) != header:
                    return None
            parsed = _parse_header(buf.view(max(header, ALIGN)))
            if parsed is None:
                return None
            header, entries = parsed
            end = max(
                (offset + size for _k, offset, size, _c in entries), default=header
            )
            if disk != _round_up(end) or any(
                off < header for _k, off, _s, _c in entries
            ):
                return None
            return disk, entries
        finally:
            os.close(fd)

    # -- KvStoreStorage ---------------------------------------------------------

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        with self._lock:
            results = []
            for key in keys:
                entry = self._keys.get(key)
                if entry is not None:
                    # A probe precedes a prefetch: keep the segment from being
                    # evicted between the two.
                    self._segments.move_to_end(entry.segment)
                results.append(entry is not None)
            return results

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 get")
        base = host_buffer_ptr(host_buffer)
        results = [False] * len(keys)
        # segment -> [(result index, key, entry, host address)], in request order
        jobs: dict[str, list[tuple[int, str, _Entry, int]]] = {}
        with self._lock:
            for i, (key, off, size) in enumerate(zip(keys, offsets, sizes)):
                entry = self._keys.get(key)
                if entry is None or entry.size != int(size):
                    continue
                self._segments.move_to_end(entry.segment)
                jobs.setdefault(entry.segment, []).append(
                    (i, key, entry, base + int(off))
                )
        outcomes = self._run(self._get_segment, list(jobs.items()))
        for indices in outcomes:
            for i in indices:
                results[i] = True
        return results

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 put")
        base = host_buffer_ptr(host_buffer)
        results = [False] * len(keys)
        pending: list[_PutObject] = []
        seen: set[str] = set()
        with self._lock:
            for i, (key, off, size) in enumerate(zip(keys, offsets, sizes)):
                entry = self._keys.get(key)
                if entry is not None:
                    self._segments.move_to_end(entry.segment)
                    results[i] = True
                    continue
                if key in seen or len(key.encode()) > _MAX_KEY_BYTES:
                    continue
                seen.add(key)
                pending.append(_PutObject(i, key, base + int(off), int(size)))
            groups = self._split(pending)
            keep = {self._keys[k].segment for k in keys if k in self._keys}
            todo: list[tuple[list[_PutObject], int]] = []
            for group in groups:
                disk = self._disk_bytes(group)
                if not self._make_room(disk, keep=keep):
                    continue
                # Reserve the bytes now so concurrent puts cannot overcommit.
                self._used += disk
                todo.append((group, disk))
        names = self._run(lambda group, disk: self._put_segment(group), todo)
        with self._lock:
            for (group, disk), written in zip(todo, names):
                if written is None:
                    self._used -= disk
                    continue
                name, entries = written
                if name in self._segments:
                    # Shared names are derived from the put; a repeated put
                    # of the same keys replaced the file under the same name.
                    old = self._segments.pop(name)
                    self._used -= old.disk
                    for key in old.live:
                        del self._keys[key]
                segment = _Segment(disk=disk)
                self._segments[name] = segment
                for obj, entry in zip(group, entries):
                    results[obj.index] = True
                    if obj.key in self._keys:
                        continue  # a racing put stored it first
                    self._keys[obj.key] = entry
                    segment.live.add(obj.key)
                if not segment.live:
                    self._delete_segment_locked(name)
        return results

    def remove_by_prefix(self, prefix: str) -> bool:
        """Delete every segment holding a key under ``prefix``.

        A segment's header lists all its keys, so a deletion must remove the
        whole file to stay deleted across a restart. Keys of another prefix
        written in the same put go with it; they are cache and read as misses.
        """

        with self._lock:
            names = {
                entry.segment
                for key, entry in self._keys.items()
                if key.startswith(prefix)
            }
            ok = True
            for name in names:
                for key in self._segments[name].live:
                    del self._keys[key]
                ok = self._delete_segment_locked(name) and ok
        return ok

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    # -- internals --------------------------------------------------------------

    def _run(self, fn, items: list[tuple]) -> list:
        if not items:
            return []
        if self._pool is None:
            raise RuntimeError("file L3 store is closed")
        workers = min(self.config.io_threads, len(items))
        if workers == 1:
            return [fn(*args) for args in items]
        # One contiguous slice per thread, not one future per item: per-future
        # dispatch and GIL hand-offs otherwise cost as much as a small read.
        step = -(-len(items) // workers)
        slices = [items[start : start + step] for start in range(0, len(items), step)]
        results = self._pool.map(lambda part: [fn(*args) for args in part], slices)
        return [item for part in results for item in part]

    def _split(self, objects: list[_PutObject]) -> list[list[_PutObject]]:
        """Group a put into segments of at most ``MAX_SEGMENT_BYTES`` payload.

        A large put is also split across the I/O threads so it is written in
        parallel, but never into segments under ``MIN_SPLIT_BYTES``; a put of
        a few small objects stays one segment.
        """

        if not objects:
            return []
        total = sum(obj.size for obj in objects)
        share = -(-total // self.config.io_threads)
        limit = min(MAX_SEGMENT_BYTES, max(share, MIN_SPLIT_BYTES))
        groups: list[list[_PutObject]] = [[]]
        filled = 0
        for obj in objects:
            if groups[-1] and filled + obj.size > limit:
                groups.append([])
                filled = 0
            groups[-1].append(obj)
            filled += obj.size
        return groups

    @staticmethod
    def _disk_bytes(group: Sequence[_PutObject]) -> int:
        header = _header_bytes([obj.key.encode() for obj in group])
        return _round_up(header + sum(obj.size for obj in group))

    def _make_room(self, disk: int, *, keep: set[str]) -> bool:
        """Evict LRU segments (not in ``keep``) until ``disk`` more bytes fit.

        Called with the lock held. Files are unlinked here too: an evicted
        key is gone from the index first, so a concurrent get misses cleanly.
        """

        capacity = self.config.capacity_bytes
        if disk > capacity:
            return False
        for name in list(self._segments):
            if self._used + disk <= capacity:
                break
            if name in keep:
                continue
            for key in list(self._segments[name].live):
                del self._keys[key]
            self._delete_segment_locked(name)
        return self._used + disk <= capacity

    def _drop_key_locked(self, key: str) -> bool:
        """Forget ``key``; return True when its segment has no live key left."""

        entry = self._keys.pop(key)
        segment = self._segments[entry.segment]
        segment.live.discard(key)
        return not segment.live

    def _delete_segment_locked(self, name: str) -> bool:
        segment = self._segments.pop(name)
        self._used -= segment.disk
        if not self._writer:
            return True  # the owner deletes the shared file
        try:
            os.unlink(self._segment_path(name))
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception(f"[L3 file] failed to delete segment {name}")
            return False
        return True

    def _get_segment(
        self, name: str, wanted: list[tuple[int, str, _Entry, int]]
    ) -> list[int]:
        """Read one segment and copy the wanted objects; return their indices."""

        try:
            fd = os.open(self._segment_path(name), os.O_RDONLY | self._read_flags)
        except FileNotFoundError:
            return []
        try:
            disk = os.fstat(fd).st_size
            buf = self._buffer()
            wanted = sorted(wanted, key=lambda item: item[2].offset)
            header = wanted[0][2].offset
            # Every object of the file wanted, back to back, 4 KiB aligned on
            # both sides: scatter the payloads straight into the Host slices.
            direct = all(_aligned(address, e.size) for _i, _k, e, address in wanted)
            end = header
            for _i, _k, entry, _a in wanted:
                direct = direct and entry.offset == end
                end = entry.offset + entry.size
            direct = direct and end == disk and header % ALIGN == 0
            if direct:
                buf.ensure(header)
                views = [buf.view(header)] + [
                    _host_view(address, entry.size) for _i, _k, entry, address in wanted
                ]
            else:
                buf.ensure(disk)
                views = [buf.view(disk)]
            if _read_full(fd, views, 0) != disk:
                return []
            parsed = _parse_header(buf.view(header if direct else disk))
            if parsed is None:
                return []
            stored = {key: (offset, size, crc) for key, offset, size, crc in parsed[1]}
            done = []
            for (i, key, entry, address), view in zip(
                wanted, views[1:] if direct else [None] * len(wanted)
            ):
                # The index is checked against the file itself, so a segment
                # file that does not hold this key never serves it.
                if stored.get(key) != (entry.offset, entry.size, entry.crc):
                    continue
                payload = view if direct else buf.view(entry.size, entry.offset)
                if self._verify and zlib.crc32(payload) != entry.crc:
                    continue
                if not direct:
                    ctypes.memmove(address, buf.address + entry.offset, entry.size)
                done.append(i)
            return done
        except OSError:
            logger.exception(f"[L3 file] read failed for segment {name}")
            return []
        finally:
            os.close(fd)

    def _put_segment(self, group: list[_PutObject]) -> tuple[str, list[_Entry]] | None:
        if self._shared:
            # Every rank of the group must arrive at the owner's file name.
            digest = hashlib.sha256()
            for obj in group:
                digest.update(f"{obj.key}\0{obj.size}\0".encode())
            name = digest.hexdigest()[:32]
        else:
            name = uuid.uuid4().hex
        if not self._writer:
            return name, self._shared_entries(name, group)
        path = self._segment_path(name)
        tmp = f"{path}.tmp.{threading.get_ident()}"
        key_bytes = [obj.key.encode() for obj in group]
        header = _header_bytes(key_bytes)
        disk = self._disk_bytes(group)
        # 4 KiB aligned Host slices are written straight from the Host buffer;
        # anything else is packed into the bounce buffer first.
        direct = all(_aligned(obj.address, obj.size) for obj in group)
        buf = self._buffer()
        buf.ensure(header if direct else disk)
        entries = []
        table = bytearray()
        views = []
        offset = header
        for obj, key in zip(group, key_bytes):
            if direct:
                payload = _host_view(obj.address, obj.size)
                views.append(payload)
            else:
                ctypes.memmove(buf.address + offset, obj.address, obj.size)
                payload = buf.view(obj.size, offset)
            crc = zlib.crc32(payload) if self._verify else 0
            table += _ENTRY.pack(offset, obj.size, crc, len(key)) + key
            entries.append(_Entry(name, offset, obj.size, crc))
            offset += obj.size
        if not direct and disk > offset:
            buf.view(disk - offset, offset)[:] = bytes(disk - offset)
        head = buf.view(header)
        head[:] = bytes(header)
        _PREFIX.pack_into(head, 0, _MAGIC, header, len(group), zlib.crc32(table))
        head[_PREFIX.size : _PREFIX.size + len(table)] = table
        views = [head] + views if direct else [buf.view(disk)]
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        try:
            fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | self._flags, 0o600
            )
            try:
                _write_full(fd, views)
                if self.config.fsync:
                    os.fdatasync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            return name, entries
        except OSError:
            logger.exception(f"[L3 file] write failed for segment {name}")
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            return None

    def _shared_entries(self, name: str, group: list[_PutObject]) -> list[_Entry]:
        """The entries the owner writes for ``group``, CRCs from this rank's bytes."""

        offset = _header_bytes([obj.key.encode() for obj in group])
        entries = []
        for obj in group:
            crc = zlib.crc32(_host_view(obj.address, obj.size))
            entries.append(_Entry(name, offset, obj.size, crc))
            offset += obj.size
        return entries
