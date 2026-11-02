# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""CPU, 2 gloo ranks: clip-parallel decode equals the single-process decode."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("diffusers")

from diffusers import AutoencoderKLMiniMaxH3  # noqa: E402

from tokenspeed.runtime.diffusion.vae_parallel import (  # noqa: E402
    ClipParallelVideoDecode,
)

LATENT_FRAMES = 31  # 124 pixel frames, the 5 s bench workload: 6 clips


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=[8, 8, 8, 8, 8, 8],
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
    ).eval()


def _latents() -> torch.Tensor:
    return torch.randn(
        1, 4, LATENT_FRAMES, 2, 3, generator=torch.Generator().manual_seed(3)
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        vae = _tiny_vae()
        decode = ClipParallelVideoDecode(vae, group=None)
        decode.enabled = True
        with torch.no_grad():
            out = vae.decode(_latents(), return_dict=False)[0]
        torch.save(out, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@torch.no_grad()
def test_clip_parallel_decode_matches_single_process(tmp_path):
    vae = _tiny_vae()
    z = _latents()
    expected = vae.decode(z, return_dict=False)[0]
    # The workload really is multi-clip, otherwise the test proves nothing.
    assert len(ClipParallelVideoDecode(_tiny_vae(), group=None).clip_inputs(z)) == 6

    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        got = torch.load(tmp_path / f"rank{rank}.pt")
        assert torch.equal(got, expected)


@torch.no_grad()
def test_disabled_decode_is_the_original():
    vae = _tiny_vae()
    z = _latents()
    expected = vae.decode(z, return_dict=False)[0]
    ClipParallelVideoDecode(vae, group=None)  # enabled stays False
    assert torch.equal(vae.decode(z, return_dict=False)[0], expected)
