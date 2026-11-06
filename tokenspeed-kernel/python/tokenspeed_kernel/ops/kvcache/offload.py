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

"""Public sparse KV residency API; NVIDIA uses the CuTe DSL implementation."""

import functools

import torch
from tokenspeed_kernel.ops.kvcache.cute_dsl import (
    accepted_ids,
    current_slots,
)
from tokenspeed_kernel.ops.kvcache.cute_dsl import (
    cute_dsl_offload_copy_rows as copy_rows,
)
from tokenspeed_kernel.ops.kvcache.cute_dsl import (
    cute_dsl_offload_materialize as materialize,
)
from tokenspeed_kernel.ops.kvcache.cute_dsl import (
    reset_lru,
    seed_locations,
    seed_rows,
)

# Shared hash tables are two int32 arrays of ``slots`` entries next to the
# resolver's 8-int32 scan scratch (16-byte aligned).
_SCAN_SCRATCH_BYTES = 32
# Upper bound kept from the SM90/SM100 tuning: larger tables use global memory.
_MAX_SHARED_TABLE_SLOTS = 16384


@functools.cache
def _max_shared_bytes_per_block(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def hash_geometry(queries: int, topk: int) -> tuple[int, bool]:
    """Return table slots and shared placement for a fixed selection geometry.

    Tables of at most 16384 slots stay in shared memory when they fit the
    current CUDA device's per-block limit (all of SM90/SM100; up to 8192 slots
    on SM120's ~99 KiB). Larger shapes use budgeted global tables.
    """
    entries = queries * topk
    if entries <= 0:
        raise ValueError("offload selection width must be positive")
    slots = 1 << (2 * entries - 1).bit_length()
    shared = slots <= _MAX_SHARED_TABLE_SLOTS
    if shared and torch.cuda.is_available():
        limit = _max_shared_bytes_per_block(torch.cuda.current_device())
        shared = slots * 8 + _SCAN_SCRATCH_BYTES <= limit
    return slots, shared


__all__ = [
    "accepted_ids",
    "copy_rows",
    "current_slots",
    "hash_geometry",
    "materialize",
    "reset_lru",
    "seed_locations",
    "seed_rows",
]
