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
"""MiniMax-H3-Image family detection, request validation and HTTP surface."""

import base64
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tokenspeed.runtime.diffusion.family import detect_diffusion_family
from tokenspeed.runtime.diffusion.image_config import H3ImageConfig, parse_image_size
from tokenspeed.runtime.diffusion.image_pipeline import H3ImageRequest, H3ImageResult
from tokenspeed.runtime.entrypoints.image_http import (
    create_app,
    parse_image_argv,
    request_from_fields,
)
from tokenspeed.runtime_select import resolve_serving_runtime


def _checkpoint(tmp_path: Path, class_name: str) -> Path:
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    return tmp_path


def test_image_checkpoint_is_its_own_family(tmp_path: Path):
    root = _checkpoint(tmp_path, "MiniMaxH3ImagePipeline")
    assert detect_diffusion_family(str(root)) == "h3image"
    assert resolve_serving_runtime(str(root)) == "diffusion"


def test_video_checkpoint_stays_h3(tmp_path: Path):
    root = _checkpoint(tmp_path, "MiniMaxH3ModularPipeline")
    assert detect_diffusion_family(str(root)) == "h3"


def test_parse_image_size():
    assert parse_image_size("4736x3552", 32) == (4736, 3552)
    assert parse_image_size(" 1024 X 2048 ", 32) == (1024, 2048)
    for bad in ("1024", "1000x1024", "0x0", "axb", ""):
        with pytest.raises(ValueError):
            parse_image_size(bad, 32)


def test_residency_resolution():
    def cfg(gpus: int, residency: str) -> H3ImageConfig:
        return H3ImageConfig(
            model_path="m",
            served_model_name="test",
            num_gpus=gpus,
            dtype="bfloat16",
            enable_cpu_offload=False,
            memory_reserve_margin="12GB",
            residency=residency,
        )

    assert cfg(2, "auto").resolve_residency() == "device_split"
    assert cfg(1, "auto").resolve_residency() == "single"
    assert cfg(1, "single").resolve_residency() == "single"
    with pytest.raises(ValueError):
        cfg(1, "device_split").resolve_residency()
    with pytest.raises(ValueError):
        cfg(2, "ulysses").resolve_residency()


def test_serve_argv_rejects_context_parallel():
    with pytest.raises(SystemExit):
        parse_image_argv(["m", "--ulysses-degree", "2"])
    args = parse_image_argv(["m", "--tensor-parallel-size", "2", "--task-type", "x"])
    assert args.num_gpus == 2


def test_request_validation():
    ok = request_from_fields(
        {"prompt": "a fox", "size": "1024x512", "seed": 3, "output_format": "jpg"},
        [],
        32,
    )
    assert (ok.width, ok.height, ok.seed, ok.output_format) == (1024, 512, 3, "jpeg")
    assert ok.reference_images == []
    for bad in (
        {"size": "1024x1024"},
        {"prompt": "x"},
        {"prompt": "x", "size": "1024x1024", "num_inference_steps": 4},
        {"prompt": "x", "size": "1024x1024", "n": 2},
        {"prompt": "x", "size": "1024x1024", "response_format": "url"},
        {"prompt": "x", "size": "1024x1024", "output_format": "gif"},
        {"prompt": "x", "size": "1024x1024", "generator_device": "cuda"},
        {"prompt": "x", "size": "1024x1024", "bogus": 1},
    ):
        with pytest.raises(ValueError):
            request_from_fields(bad, [], 32)
    with pytest.raises(ValueError):
        request_from_fields({"prompt": "x", "size": "1024x1024"}, [b"x"] * 17, 32)


class _EchoPipeline:
    """Stands in for the GPU pipeline: returns a solid image of the requested size."""

    canvas_multiple = 32

    def __init__(self):
        self.requests: list[H3ImageRequest] = []

    def generate(self, req: H3ImageRequest) -> H3ImageResult:
        self.requests.append(req)
        buffer = io.BytesIO()
        Image.new("RGB", (req.width, req.height), (9, 8, 7)).save(buffer, "PNG")
        return H3ImageResult(
            image_bytes=buffer.getvalue(),
            output_format="png",
            width=req.width,
            height=req.height,
            seed=req.seed if req.seed is not None else 0,
        )


def _png(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_generations_and_edits_endpoints():
    pipeline = _EchoPipeline()
    client = TestClient(create_app(pipeline, "test"))

    assert client.get("/health").json()["family"] == "h3image"

    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox", "size": "128x96", "seed": 5},
    )
    assert response.status_code == 200
    body = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.size == (128, 96)
    assert body["seed"] == 5
    assert pipeline.requests[-1].reference_images == []

    response = client.post(
        "/v1/images/generations", json={"prompt": "x", "size": "100x96"}
    )
    assert response.status_code == 400

    refs = [_png((255, 0, 0)), _png((0, 255, 0))]
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "swap <Picture 1> and <Picture 2>", "size": "64x64"},
        files=[("image", ("a.png", refs[0], "image/png")), ("image", ("b.png", refs[1], "image/png"))],
    )
    assert response.status_code == 200
    assert pipeline.requests[-1].reference_images == refs

    response = client.post(
        "/v1/images/edits", data={"prompt": "x", "size": "64x64"}
    )
    assert response.status_code == 400
