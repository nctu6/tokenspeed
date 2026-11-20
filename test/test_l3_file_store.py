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

"""Local-disk L3 store: round trips, restart, integrity, capacity and config."""

from __future__ import annotations

import json
import os

import pytest
import torch

from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.file import (
    ALIGN,
    HEADER_BYTES,
    FileKvStore,
    FileStoreConfig,
    parse_file_store_config,
)


def _config(
    path, *, capacity_gb=1.0, o_direct=True, verify_crc=True
) -> FileStoreConfig:
    return FileStoreConfig(
        path=str(path),
        capacity_gb=capacity_gb,
        io_threads=4,
        o_direct=o_direct,
        verify_crc=verify_crc,
        fsync=False,
    )


def _aligned_host(nbytes: int) -> tuple[torch.Tensor, int]:
    """A uint8 host buffer and the first offset whose address is 4 KiB aligned."""

    buf = torch.empty(nbytes + ALIGN, dtype=torch.uint8)
    return buf, (-buf.data_ptr()) % ALIGN


def _pattern(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=gen)


@pytest.mark.parametrize("o_direct", [True, False])
@pytest.mark.parametrize(
    "skew, size",
    [(0, 3 * ALIGN), (0, 3 * ALIGN + 100), (64, 3 * ALIGN), (7, 1000)],
)
def test_round_trip_into_aligned_and_unaligned_host_slices(
    tmp_path, o_direct, skew, size
):
    store = FileKvStore(
        _config(tmp_path, o_direct=o_direct), rank=0, shared_owner_rank=None
    )
    src, base = _aligned_host(4 * size + 4 * ALIGN)
    keys = [f"k{i}" for i in range(3)]
    offsets = [base + skew + i * _round(size) for i in range(3)]
    for i, off in enumerate(offsets):
        src[off : off + size] = _pattern(size, i)
    assert store.batch_put_from(keys, src, offsets, [size] * 3) == [True] * 3
    assert store.batch_exists(keys + ["missing"]) == [True, True, True, False]

    dst, dbase = _aligned_host(4 * size + 4 * ALIGN)
    dst.zero_()
    doffs = [dbase + skew + i * _round(size) for i in range(3)]
    assert store.batch_get_into(keys, dst, doffs, [size] * 3) == [True] * 3
    for i, off in enumerate(doffs):
        assert torch.equal(dst[off : off + size], _pattern(size, i))
    # A get for the wrong size is a miss and leaves the slice untouched.
    dst.zero_()
    assert store.batch_get_into(["k0"], dst, [doffs[0]], [size + ALIGN]) == [False]
    assert int(dst.sum()) == 0
    store.close()


def test_batch_larger_than_io_threads_keeps_per_key_results_in_order(tmp_path):
    # 11 objects over 4 threads: uneven slices, with a miss in the middle.
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    size, count = 1000, 11
    src = torch.empty(count * size, dtype=torch.uint8)
    for i in range(count):
        src[i * size : (i + 1) * size] = _pattern(size, 100 + i)
    keys = [f"k{i}" for i in range(count)]
    offsets = [i * size for i in range(count)]
    assert store.batch_put_from(keys, src, offsets, [size] * count) == [True] * count

    keys[5] = "missing"
    dst = torch.zeros(count * size, dtype=torch.uint8)
    flags = store.batch_get_into(keys, dst, offsets, [size] * count)
    assert flags == [i != 5 for i in range(count)]
    for i in range(count):
        want = (
            torch.zeros(size, dtype=torch.uint8) if i == 5 else _pattern(size, 100 + i)
        )
        assert torch.equal(dst[i * size : (i + 1) * size], want)
    store.close()


def _round(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def test_objects_survive_a_restart_and_temporaries_are_dropped(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    src, base = _aligned_host(2 * ALIGN)
    src[base : base + 5000] = _pattern(5000, 7)
    assert store.batch_put_from(["persist"], src, [base], [5000]) == [True]
    store.close()
    stray = os.path.join(str(tmp_path), "rank0", "ab", "x.seg.tmp.1")
    legacy = os.path.join(str(tmp_path), "rank0", "cd", "y.kv")
    for path in (stray, legacy):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"partial")

    again = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    assert again.batch_exists(["persist"]) == [True]
    assert not os.path.exists(stray)
    assert not os.path.exists(legacy)
    dst, dbase = _aligned_host(2 * ALIGN)
    assert again.batch_get_into(["persist"], dst, [dbase], [5000]) == [True]
    assert torch.equal(dst[dbase : dbase + 5000], _pattern(5000, 7))
    assert again.used_bytes() == HEADER_BYTES + _round(5000)
    again.close()


def _object_file(store: FileKvStore, key: str) -> str:
    return store._segment_path(store._keys[key].segment)


def _segment_files(root) -> list[str]:
    return sorted(
        os.path.join(d, f)
        for d, _, files in os.walk(root)
        for f in files
        if f.endswith(".seg")
    )


def test_one_put_is_one_segment_and_partial_reads_work(tmp_path):
    size, count = 3000, 6
    src = torch.empty(count * size, dtype=torch.uint8)
    for i in range(count):
        src[i * size : (i + 1) * size] = _pattern(size, 200 + i)
    keys = [f"p{i}" for i in range(count)]
    offsets = [i * size for i in range(count)]
    # A small put stays one segment even with several I/O threads.
    single = FileKvStore(
        FileStoreConfig(str(tmp_path / "one"), 1.0, 16, True, True, False),
        rank=0,
        shared_owner_rank=None,
    )
    assert single.batch_put_from(keys, src, offsets, [size] * count) == [True] * count
    assert len(_segment_files(single.root)) == 1
    dst = torch.zeros(count * size, dtype=torch.uint8)
    # Read two objects of the segment, out of order, into other offsets.
    assert single.batch_get_into(["p4", "p1"], dst, [0, size], [size, size]) == [
        True,
        True,
    ]
    assert torch.equal(dst[:size], _pattern(size, 204))
    assert torch.equal(dst[size : 2 * size], _pattern(size, 201))
    single.close()


def test_remove_by_prefix_deletes_whole_segments_and_stays_deleted(tmp_path):
    cfg = FileStoreConfig(str(tmp_path), 1.0, 1, True, True, False)
    store = FileKvStore(cfg, rank=0, shared_owner_rank=None)
    src = torch.empty(3 * 1000, dtype=torch.uint8)
    keys = ["a_0", "b_1", "a_2"]
    assert store.batch_put_from(keys, src, [0, 1000, 2000], [1000] * 3) == [True] * 3
    assert store.batch_put_from(["b_3"], src, [0], [1000]) == [True]
    assert store.remove_by_prefix("a_") is True
    # "b_1" shared a segment with the "a_" keys and goes with it.
    assert store.batch_exists(keys + ["b_3"]) == [False, False, False, True]
    assert store.used_bytes() == HEADER_BYTES + ALIGN
    store.close()
    again = FileKvStore(cfg, rank=0, shared_owner_rank=None)
    assert again.batch_exists(keys + ["b_3"]) == [False, False, False, True]
    assert len(_segment_files(again.root)) == 1
    again.close()


def test_large_put_is_split_across_io_threads(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)  # 4 threads
    size, count = 1 << 20, 8  # 8 MiB: four 2 MiB shares, above MIN_SPLIT_BYTES
    src = torch.empty(count * size, dtype=torch.uint8)
    for i in range(count):
        src[i * size : (i + 1) * size] = _pattern(size, 400 + i)
    keys = [f"s{i}" for i in range(count)]
    offsets = [i * size for i in range(count)]
    assert store.batch_put_from(keys, src, offsets, [size] * count) == [True] * count
    assert len(_segment_files(store.root)) == 4
    dst = torch.zeros(count * size, dtype=torch.uint8)
    assert store.batch_get_into(keys, dst, offsets, [size] * count) == [True] * count
    assert torch.equal(dst, src)
    store.close()


def test_corrupted_segment_header_is_dropped_at_restart(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    src, base = _aligned_host(2 * ALIGN)
    assert store.batch_put_from(["h"], src, [base], [ALIGN]) == [True]
    path = _object_file(store, "h")
    store.close()
    with open(path, "r+b") as f:
        f.seek(40)  # inside the entry table, covered by its CRC
        byte = f.read(1)
        f.seek(40)
        f.write(bytes([byte[0] ^ 0xFF]))
    again = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    assert again.batch_exists(["h"]) == [False]
    assert not os.path.exists(path)
    assert again.used_bytes() == 0
    again.close()


def test_corrupted_payload_reads_as_a_miss(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    src, base = _aligned_host(2 * ALIGN)
    src[base : base + ALIGN] = _pattern(ALIGN, 1)
    assert store.batch_put_from(["bad"], src, [base], [ALIGN]) == [True]
    path = _object_file(store, "bad")
    with open(path, "r+b") as f:
        f.seek(HEADER_BYTES + 10)
        byte = f.read(1)
        f.seek(HEADER_BYTES + 10)
        f.write(bytes([byte[0] ^ 0xFF]))
    dst, dbase = _aligned_host(2 * ALIGN)
    assert store.batch_get_into(["bad"], dst, [dbase], [ALIGN]) == [False]
    store.close()


def test_truncated_file_is_dropped_at_restart(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    src, base = _aligned_host(2 * ALIGN)
    assert store.batch_put_from(["torn"], src, [base], [ALIGN]) == [True]
    path = _object_file(store, "torn")
    store.close()
    with open(path, "r+b") as f:
        f.truncate(100)
    again = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    assert again.batch_exists(["torn"]) == [False]
    assert not os.path.exists(path)
    again.close()


def test_capacity_evicts_least_recently_used(tmp_path):
    per_object = HEADER_BYTES + ALIGN
    capacity_gb = 3 * per_object / (1 << 30)
    store = FileKvStore(
        _config(tmp_path, capacity_gb=capacity_gb), rank=0, shared_owner_rank=None
    )
    src, base = _aligned_host(2 * ALIGN)
    for key in ("a", "b", "c"):
        assert store.batch_put_from([key], src, [base], [ALIGN]) == [True]
    store.batch_exists(["a"])  # touch: "b" is now the oldest
    b_file = _object_file(store, "b")
    assert store.batch_put_from(["d"], src, [base], [ALIGN]) == [True]
    assert store.batch_exists(["a", "b", "c", "d"]) == [True, False, True, True]
    assert not os.path.exists(b_file)
    assert store.used_bytes() == 3 * per_object
    # An object larger than the whole capacity is refused, nothing evicted.
    big, bbase = _aligned_host(4 * per_object)
    assert store.batch_put_from(["huge"], big, [bbase], [4 * per_object]) == [False]
    assert store.batch_exists(["a", "c", "d"]) == [True, True, True]
    store.close()


def test_remove_by_prefix_and_rank_isolation(tmp_path):
    rank0 = FileKvStore(_config(tmp_path), rank=0, shared_owner_rank=None)
    rank1 = FileKvStore(_config(tmp_path), rank=1, shared_owner_rank=None)
    src, base = _aligned_host(2 * ALIGN)
    # Separate puts: a prefix delete removes whole segments.
    assert rank0.batch_put_from(["ns1_x"], src, [base], [64]) == [True]
    assert rank0.batch_put_from(["ns2_y"], src, [base], [64]) == [True]
    assert rank1.batch_put_from(["ns1_x"], src, [base], [64]) == [True]
    assert rank0.remove_by_prefix("ns1_") is True
    assert rank0.batch_exists(["ns1_x", "ns2_y"]) == [False, True]
    assert rank1.batch_exists(["ns1_x"]) == [True]
    rank0.close()
    rank1.close()
    assert (
        FileKvStore(_config(tmp_path), rank=1, shared_owner_rank=None).used_bytes()
        == HEADER_BYTES + ALIGN
    )


def test_config_requires_every_field(tmp_path):
    full = {
        "path": str(tmp_path),
        "capacity_gb": 2,
        "io_threads": 8,
        "o_direct": True,
        "verify_crc": True,
        "fsync": False,
    }
    assert parse_file_store_config(json.dumps(full)).io_threads == 8
    for name in full:
        partial = {k: v for k, v in full.items() if k != name}
        with pytest.raises(ValueError, match="missing"):
            parse_file_store_config(json.dumps(partial))
    with pytest.raises(ValueError, match="unknown"):
        parse_file_store_config(json.dumps({**full, "threads": 4}))
    with pytest.raises(ValueError, match="wrong type"):
        parse_file_store_config(json.dumps({**full, "o_direct": "yes"}))
    with pytest.raises(ValueError, match="wrong type"):
        parse_file_store_config(json.dumps({**full, "io_threads": True}))
    with pytest.raises(ValueError, match="> 0"):
        parse_file_store_config(json.dumps({**full, "capacity_gb": 0}))
    with pytest.raises(ValueError):
        parse_file_store_config(None)


def test_factory_builds_the_file_store(tmp_path):
    extra = json.dumps(
        {
            "path": str(tmp_path),
            "capacity_gb": 1,
            "io_threads": 2,
            "o_direct": False,
            "verify_crc": True,
            "fsync": True,
        }
    )
    backend = create_kvstore_storage_backend(
        "file",
        extra,
        host_buffer=object(),
        tp_size=1,
        pp_size=1,
        rank=3,
        shared_owner_rank=None,
    )
    assert isinstance(backend, FileKvStore)
    assert backend.root == os.path.join(str(tmp_path), "rank3")
    backend.close()


def _shared_pair(tmp_path, *, owner=4):
    cfg = FileStoreConfig(str(tmp_path), 1.0, 2, True, False, False)
    writer = FileKvStore(cfg, rank=owner, shared_owner_rank=owner)
    reader = FileKvStore(cfg, rank=owner + 1, shared_owner_rank=owner)
    return cfg, writer, reader


def _page(seed: int, size: int = 6000, count: int = 3) -> torch.Tensor:
    return torch.cat([_pattern(size, seed + i) for i in range(count)])


def test_shared_group_keeps_one_copy_that_every_rank_reads(tmp_path):
    _cfg, writer, reader = _shared_pair(tmp_path)
    keys, offsets, sizes = ["k0", "k1", "k2"], [0, 6000, 12000], [6000] * 3
    data = _page(500)
    # Both ranks hold the same bytes and issue the same put.
    assert writer.batch_put_from(keys, data, offsets, sizes) == [True] * 3
    assert reader.batch_put_from(keys, data.clone(), offsets, sizes) == [True] * 3
    assert writer.root == reader.root == os.path.join(str(tmp_path), "rank4")
    assert len(_segment_files(writer.root)) == 1
    assert reader.used_bytes() == writer.used_bytes()
    dst = torch.zeros_like(data)
    assert reader.batch_get_into(keys, dst, offsets, sizes) == [True] * 3
    assert torch.equal(dst, data)
    writer.close()
    reader.close()


def test_shared_rank_with_different_bytes_reads_a_miss(tmp_path):
    _cfg, writer, reader = _shared_pair(tmp_path)
    keys, offsets, sizes = ["k0", "k1", "k2"], [0, 6000, 12000], [6000] * 3
    assert writer.batch_put_from(keys, _page(600), offsets, sizes) == [True] * 3
    # This rank's own bytes differ, so the owner's file must not serve it.
    assert reader.batch_put_from(keys, _page(700), offsets, sizes) == [True] * 3
    dst = torch.zeros(18000, dtype=torch.uint8)
    assert reader.batch_get_into(keys, dst, offsets, sizes) == [False] * 3
    assert int(dst.sum()) == 0
    writer.close()
    reader.close()


def test_only_the_owner_deletes_shared_files(tmp_path):
    cfg, writer, reader = _shared_pair(tmp_path)
    data = _page(800, count=1)
    for store in (writer, reader):
        assert store.batch_put_from(["ns_a"], data, [0], [6000]) == [True]
    path = _segment_files(writer.root)[0]
    stray = os.path.join(writer.root, "ab", "x.seg.tmp.1")
    os.makedirs(os.path.dirname(stray), exist_ok=True)
    open(stray, "wb").write(b"partial")
    assert reader.remove_by_prefix("ns_") is True
    assert reader.batch_exists(["ns_a"]) == [False]
    assert os.path.exists(path)  # the reader only forgot it
    reader.close()
    # A restarting non-owner indexes the owner's files and leaves the rest.
    again = FileKvStore(cfg, rank=5, shared_owner_rank=4)
    assert again.batch_exists(["ns_a"]) == [True]
    assert os.path.exists(stray)
    again.close()
    assert writer.remove_by_prefix("ns_") is True
    assert not os.path.exists(path)
    writer.close()
