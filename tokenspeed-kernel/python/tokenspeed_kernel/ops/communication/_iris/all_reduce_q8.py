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

"""Pull two-shot BF16 all-reduce with 8-bit payloads on the peer links.

A local kernel first quantizes the BF16 input into this rank's symmetric heap
(INT8 or FP8 E4M3, one FP32 scale per GROUP consecutive elements). The
communication kernel then runs the usual pull two-shot: the owner of each
1/WORLD_SIZE partition pulls the peers' 8-bit partition, sums in FP32 in rank
order and republishes the sum in 8 bits; every rank then pulls all reduced
partitions and writes BF16. Peer traffic is about half of the BF16 two-shot's
(1 byte per element plus 4 bytes per group), at the cost of two 8-bit
roundings (each input and the reduced sum).

The communication grid is fixed, so every tile always belongs to the same
workgroup; with the per-workgroup rendezvous this makes buffer reuse across
calls safe without an exit barrier: a peer workgroup reaches the next call's
first rendezvous only after it finished reading this rank's buffers, and the
next quantize kernel starts only after this rank's communication kernel
(and with it every peer's reduce phase) has finished.
"""

import os

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_drain_subgroup_vmem,
    _iris_peer_pointers,
    _iris_rendezvous,
)

# Measured on 4x MI350P over PCIe (168 MB BF16 message): group 128, 64-row
# tiles, 32 programs and 8 warps reach ~46 GB/s of payload per rank, the
# BF16 two-shot's link rate. Group 32 lowers the error on heavy-tailed data
# (1.5e-2 vs 2.8e-2) for ~10% more time.
Q8_GROUP = int(os.environ.get("TOKENSPEED_IRIS_Q8_GROUP", "128"))
Q8_ROWS = int(os.environ.get("TOKENSPEED_IRIS_Q8_ROWS", "64"))
Q8_PROGRAMS = int(os.environ.get("TOKENSPEED_IRIS_Q8_PROGRAMS", "32"))
Q8_WARPS = int(os.environ.get("TOKENSPEED_IRIS_Q8_WARPS", "8"))
# Each lane moves 16 bytes of 8-bit payload per load.
Q8_VEC = 16
_QUANTIZE_ROWS = 64
_QUANTIZE_WARPS = 8


@gluon.jit
def _q8_layout(GROUP: gl.constexpr, VEC: gl.constexpr, NUM_WARPS: gl.constexpr):
    LANES_PER_GROUP: gl.constexpr = GROUP // VEC
    return gl.BlockedLayout(
        [1, VEC], [64 // LANES_PER_GROUP, LANES_PER_GROUP], [NUM_WARPS, 1], [1, 0]
    )


@gluon.jit
def _q8_quantize(values, FP8: gl.constexpr):
    """Round [rows, GROUP] FP32 values to 8 bits with one FP32 scale per row."""
    amax = gl.max(gl.abs(values), axis=1)
    if FP8:
        qmax: gl.constexpr = 448.0
    else:
        qmax: gl.constexpr = 127.0
    inv = gl.where(amax > 0, qmax / amax, 0.0)
    scaled = values * gl.expand_dims(inv, 1)
    if FP8:
        q = scaled.to(gl.float8e4nv)
    else:
        q = gl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(gl.int32).to(gl.int8)
    return q, amax * (1.0 / qmax)


@gluon.jit(do_not_specialize=["groups"])
def iris_q8_quantize(
    x_ptr,
    q_ptr,
    s_ptr,
    groups,
    GROUP: gl.constexpr,
    ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    VEC: gl.constexpr,
    FP8: gl.constexpr,
):
    """Quantize ROWS groups of GROUP BF16 elements per program."""
    layout: gl.constexpr = _q8_layout(GROUP, VEC, NUM_WARPS)
    group = gl.program_id(0) * ROWS + gl.arange(
        0, ROWS, layout=gl.SliceLayout(1, layout)
    )
    cols = gl.arange(0, GROUP, layout=gl.SliceLayout(0, layout))
    gmask = group < groups
    emask = gl.expand_dims(gmask, 1)
    offsets = gl.expand_dims(group * GROUP, 1) + gl.expand_dims(cols, 0)
    values = gl.load(x_ptr + offsets, mask=emask, other=0.0).to(gl.float32)
    q, scale = _q8_quantize(values, FP8)
    gl.store(q_ptr + offsets, q, mask=emask)
    gl.store(s_ptr + group, scale, mask=gmask)


@gluon.jit(do_not_specialize=["numel"])
def iris_pull_twoshot_q8(
    q_ptr,
    s_ptr,
    qr_ptr,
    sr_ptr,
    out_ptr,
    flags,
    epochs,
    heaps,
    numel,
    RANK: gl.constexpr,
    GROUP: gl.constexpr,
    ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    VEC: gl.constexpr,
    FP8: gl.constexpr,
):
    """Reduce-scatter and all-gather numel quantized elements into BF16 out_ptr.

    numel must be a multiple of WORLD_SIZE * GROUP. q/s hold this rank's
    quantized input and qr/sr its reduced partition, all in the symmetric
    heap; flags hold [programs, WORLD_SIZE] int32 counters and epochs one
    local int32 per program.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    layout: gl.constexpr = _q8_layout(GROUP, VEC, NUM_WARPS)
    rows = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, GROUP, layout=gl.SliceLayout(0, layout))
    block = gl.program_id(0)
    programs = gl.num_programs(0)
    epoch = gl.load(epochs + block) + 2
    # The hint keeps the partition offsets on 16-byte vectors.
    part = gl.multiple_of(numel // WORLD_SIZE, GROUP)
    part_groups = part // GROUP
    tiles = gl.cdiv(part_groups, ROWS)
    # Inputs were quantized by the previous kernel on this stream.
    _iris_rendezvous(flags, heaps, block, epoch - 1, RANK, NUM_WARPS, True)

    # 1. Reduce this rank's partition from every rank's copy, in rank order.
    peer_q = _iris_peer_pointers(q_ptr, heaps, RANK)
    peer_s = _iris_peer_pointers(s_ptr, heaps, RANK)
    for tile in range(block, tiles, programs):
        group = tile * ROWS + rows
        gmask = group < part_groups
        emask = gl.expand_dims(gmask, 1)
        local = gl.expand_dims(group * GROUP, 1) + gl.expand_dims(cols, 0)
        qs = ()
        ss = ()
        # Issue every peer's loads before consuming any of them.
        for peer in gl.static_range(WORLD_SIZE):
            qs += (
                gl.load(
                    peer_q[peer] + RANK * part + local,
                    mask=emask,
                    cache_modifier=".cg",
                ),
            )
            ss += (
                gl.load(
                    peer_s[peer] + RANK * part_groups + group,
                    mask=gmask,
                    other=0.0,
                    cache_modifier=".cg",
                ),
            )
        acc = qs[0].to(gl.float32) * gl.expand_dims(ss[0], 1)
        for peer in gl.static_range(1, WORLD_SIZE):
            acc += qs[peer].to(gl.float32) * gl.expand_dims(ss[peer], 1)
        q, scale = _q8_quantize(acc, FP8)
        gl.store(qr_ptr + local, q, mask=emask, cache_modifier=".wt")
        gl.store(sr_ptr + group, scale, mask=gmask, cache_modifier=".wt")
    _iris_drain_subgroup_vmem()
    gl.barrier()
    _iris_rendezvous(flags, heaps, block, epoch, RANK, NUM_WARPS, False)

    # 2. Gather every rank's reduced partition as BF16.
    peer_qr = _iris_peer_pointers(qr_ptr, heaps, RANK)
    peer_sr = _iris_peer_pointers(sr_ptr, heaps, RANK)
    for tile in range(block, tiles, programs):
        group = tile * ROWS + rows
        gmask = group < part_groups
        emask = gl.expand_dims(gmask, 1)
        local = gl.expand_dims(group * GROUP, 1) + gl.expand_dims(cols, 0)
        qs = ()
        ss = ()
        for step in gl.static_range(WORLD_SIZE):
            qs += (
                gl.load(
                    peer_qr[(RANK + step) % WORLD_SIZE] + local,
                    mask=emask,
                    cache_modifier=".cg",
                ),
            )
            ss += (
                gl.load(
                    peer_sr[(RANK + step) % WORLD_SIZE] + group,
                    mask=gmask,
                    other=0.0,
                    cache_modifier=".cg",
                ),
            )
        for step in gl.static_range(WORLD_SIZE):
            value = qs[step].to(gl.float32) * gl.expand_dims(ss[step], 1)
            owner = (RANK + step) % WORLD_SIZE
            gl.store(out_ptr + owner * part + local, value.to(gl.bfloat16), mask=emask)
    gl.store(epochs + block, epoch)


class IrisQ8AllReduce:
    """Own the symmetric 8-bit buffers for one group and run the reduction.

    Args:
        peers: the prepared Iris group; the buffers come from its heap.
        max_numel: largest message in elements; a multiple of
            world_size * group.
        fp8: transport FP8 E4M3 (True) or INT8 (False) payloads.
        group: elements per FP32 scale (power of two, >= 16).
        rows: scale groups per tile; at least the 64 * 16 / group rows one
            pass of warps covers, times warps.
        programs: fixed communication grid.
        warps: warps per communication workgroup.
    """

    @staticmethod
    def heap_bytes(
        max_numel: int,
        world_size: int,
        group: int = Q8_GROUP,
        programs: int = Q8_PROGRAMS,
    ) -> int:
        groups = max_numel // group
        return (
            allocation_bytes((max_numel,), torch.int8)
            + allocation_bytes((groups,), torch.float32)
            + allocation_bytes((max_numel // world_size,), torch.int8)
            + allocation_bytes((groups // world_size,), torch.float32)
            + allocation_bytes((programs, world_size), torch.int32)
        )

    def __init__(
        self,
        peers: IrisGroup,
        max_numel: int,
        *,
        fp8: bool,
        group: int = Q8_GROUP,
        rows: int = Q8_ROWS,
        programs: int = Q8_PROGRAMS,
        warps: int = Q8_WARPS,
    ):
        world = peers.world_size
        if max_numel % (world * group):
            raise ValueError("max_numel must be a multiple of world_size * group")
        # Fewer rows than one pass of the layout would replicate the loads.
        if group < Q8_VEC or rows % (64 * Q8_VEC // group * warps):
            raise ValueError(f"rows={rows} does not tile group={group}, warps={warps}")
        self.peers = peers
        self.max_numel = max_numel
        self.fp8 = fp8
        self.group = group
        self.rows = rows
        self.programs = programs
        self.warps = warps
        groups = max_numel // group
        qdtype = torch.float8_e4m3fn if fp8 else torch.int8
        self.q = peers.ctx.zeros((max_numel,), dtype=torch.int8).view(qdtype)
        self.s = peers.ctx.zeros((groups,), dtype=torch.float32)
        self.qr = peers.ctx.zeros((max_numel // world,), dtype=torch.int8).view(qdtype)
        self.sr = peers.ctx.zeros((groups // world,), dtype=torch.float32)
        self.flags = peers.ctx.zeros((programs, world), dtype=torch.int32)
        self.epochs = torch.zeros((programs,), dtype=torch.int32, device=peers.device)

    def can_run(self, tensor: torch.Tensor) -> bool:
        n = tensor.numel()
        return (
            tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and 0 < n <= self.max_numel
            and n % (self.peers.world_size * self.group) == 0
        )

    def run(self, tensor: torch.Tensor, out: torch.Tensor) -> None:
        """Write the sum of every rank's tensor into out (may alias tensor).

        Every rank must call with the same numel; out is a contiguous BF16
        tensor of the same size.
        """
        groups = tensor.numel() // self.group
        iris_q8_quantize[(triton.cdiv(groups, _QUANTIZE_ROWS),)](
            tensor,
            self.q,
            self.s,
            groups,
            GROUP=self.group,
            ROWS=_QUANTIZE_ROWS,
            NUM_WARPS=_QUANTIZE_WARPS,
            VEC=Q8_VEC,
            FP8=self.fp8,
            num_warps=_QUANTIZE_WARPS,
        )
        iris_pull_twoshot_q8[(self.programs,)](
            self.q,
            self.s,
            self.qr,
            self.sr,
            out,
            self.flags,
            self.epochs,
            self.peers.addresses[: self.peers.world_size],
            tensor.numel(),
            RANK=self.peers.rank,
            GROUP=self.group,
            ROWS=self.rows,
            NUM_WARPS=self.warps,
            VEC=Q8_VEC,
            FP8=self.fp8,
            num_warps=self.warps,
        )
