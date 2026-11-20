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
"""``/v1/images/generations`` and ``/v1/images/edits`` for MiniMax-H3-Image.

Sibling of :mod:`diffusion_http` (video) and :mod:`music3_http` (audio); the
smg gateway has no image RPC. The wire format follows the OpenAI Images API
and the checkpoint's own ``examples/request.t2i.json``:

* ``POST /v1/images/generations`` (JSON): ``prompt``, ``size`` ("WxH"),
  optional ``seed``, ``output_format`` (png | jpeg | webp),
  ``min_generation_area``. Returns ``{"created", "data": [{"b64_json"}]}``.
* ``POST /v1/images/edits`` (multipart): the same fields plus one or more
  ``image`` files (up to 16, referred to in the prompt as ``<Picture N>``).

The checkpoint is distilled to a fixed 6-step schedule without guidance, so
``n``, ``num_inference_steps``, ``guidance_scale`` and ``negative_prompt`` are
rejected instead of silently ignored. ``response_format`` is ``b64_json`` only.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from tokenspeed.runtime.diffusion.image_config import (
    IMAGE_FORMATS,
    MAX_REFERENCE_IMAGES,
    H3ImageConfig,
    parse_image_size,
)
from tokenspeed.runtime.diffusion.image_pipeline import (
    H3ImagePipeline,
    H3ImageRequest,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve", "parse_image_argv", "request_from_fields"]

_UNSUPPORTED = (
    "n",
    "num_inference_steps",
    "guidance_scale",
    "negative_prompt",
    "quality",
    "style",
)
_KNOWN = frozenset(
    {
        "prompt",
        "model",
        "size",
        "seed",
        "response_format",
        "output_format",
        "generator_device",
        "min_generation_area",
        "image",
        "user",
    }
)


def parse_image_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tokenspeed serve (h3image)", add_help=False)
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--model", dest="model_flag", default=None)
    parser.add_argument("--served-model-name", default="test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-cpu-offload", action="store_true", default=False)
    parser.add_argument("--memory-reserve-margin", default="12GB")
    parser.add_argument("--residency", default="auto")
    # Ignore video/music-only flags so a shared launcher can pass a superset.
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--task-type", "--model-variant", dest="task_type", default=None)
    parser.add_argument("--ulysses-degree", "--usp", type=int, default=None)
    parser.add_argument("--ring", type=int, default=None)
    parser.add_argument("--text-encoder-tp-size", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.info(f"h3image serve: ignoring non-image flags: {' '.join(unknown)}")
    if (args.ulysses_degree or 1) > 1 or (args.ring or 1) > 1:
        raise SystemExit(
            "MiniMax-H3-Image serve does not support --ulysses-degree/--ring yet; "
            "use --residency device_split on 2 GPUs"
        )
    if (args.text_encoder_tp_size or 1) > 1:
        raise SystemExit("MiniMax-H3-Image serve does not support --text-encoder-tp-size")
    model = args.model_flag or args.model
    if not model:
        raise SystemExit("tokenspeed serve <model> is required for h3image")
    args.model = model
    if args.num_gpus is None:
        args.num_gpus = args.tensor_parallel_size or max(
            1,
            len(
                [
                    x
                    for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
                    if x.strip()
                ]
            ),
        )
    return args


def request_from_fields(
    fields: dict[str, Any], references: list[bytes], canvas_multiple: int
) -> H3ImageRequest:
    """Validate request fields (JSON body or multipart form) into a request."""
    for key in _UNSUPPORTED:
        if fields.get(key) is not None:
            if key == "n" and int(fields[key]) == 1:
                continue
            raise ValueError(
                f"MiniMax-H3-Image does not support {key!r} (fixed 6-step distilled "
                "schedule, no guidance, one image per request); omit it"
            )
    unknown = sorted(set(fields) - _KNOWN - set(_UNSUPPORTED))
    if unknown:
        raise ValueError(f"unknown request field(s): {', '.join(unknown)}")
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    size = fields.get("size")
    if not isinstance(size, str):
        raise ValueError("'size' is required, e.g. '4096x4096'")
    width, height = parse_image_size(size, canvas_multiple)
    response_format = fields.get("response_format") or "b64_json"
    if response_format != "b64_json":
        raise ValueError("response_format must be 'b64_json'")
    output_format = str(fields.get("output_format") or "png").lower()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in IMAGE_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(IMAGE_FORMATS)}")
    if (fields.get("generator_device") or "cpu") != "cpu":
        raise ValueError("generator_device must be 'cpu'")
    if len(references) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"at most {MAX_REFERENCE_IMAGES} reference images")
    seed = fields.get("seed")
    area = fields.get("min_generation_area")
    return H3ImageRequest(
        prompt=prompt,
        width=width,
        height=height,
        seed=int(seed) if seed is not None else None,
        output_format=output_format,
        reference_images=references,
        min_generation_area=int(area) if area is not None else None,
    )


def create_app(pipeline: H3ImagePipeline, served_model_name: str) -> FastAPI:
    import base64

    app = FastAPI(title="TokenSpeed MiniMax-H3-Image server")
    app.state.pipeline = pipeline
    app.state.served = served_model_name

    @app.get("/health")
    def health():
        return {"status": "ok", "runtime": "diffusion", "family": "h3image"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tokenspeed",
                }
            ],
        }

    async def _run(fields: dict[str, Any], references: list[bytes]):
        try:
            req = request_from_fields(fields, references, pipeline.canvas_multiple)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            # Blocking GPU call: keep it off the event loop so /health answers.
            result = await run_in_threadpool(pipeline.generate, req)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.exception("h3image generation failed")
            return JSONResponse({"error": str(exc)}, status_code=500)
        return {
            "created": int(time.time()),
            "data": [
                {"b64_json": base64.b64encode(result.image_bytes).decode("ascii")}
            ],
            "size": f"{result.width}x{result.height}",
            "seed": result.seed,
            "output_format": result.output_format,
        }

    @app.post("/v1/images/generations")
    async def generations(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON object required"}, status_code=400)
        return await _run(body, [])

    @app.post("/v1/images/edits")
    async def edits(request: Request):
        form = await request.form()
        fields: dict[str, Any] = {}
        references: list[bytes] = []
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                if key not in {"image", "image[]"}:
                    return JSONResponse(
                        {"error": f"unexpected file field {key!r}; use 'image'"},
                        status_code=400,
                    )
                references.append(await value.read())
            else:
                fields[key] = value
        if not references:
            return JSONResponse(
                {"error": "edits needs at least one 'image' file"}, status_code=400
            )
        return await _run(fields, references)

    return app


def serve(argv: list[str], *, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    args = parse_image_argv(argv)
    host = host or args.host
    port = port or args.port
    config = H3ImageConfig(
        model_path=args.model,
        served_model_name=args.served_model_name,
        num_gpus=int(args.num_gpus),
        dtype=args.dtype,
        enable_cpu_offload=not args.no_cpu_offload,
        memory_reserve_margin=args.memory_reserve_margin,
        residency=args.residency,
    )
    pipeline = H3ImagePipeline(config)
    logger.info(
        f"h3image server: model={config.model_path} served={config.served_model_name} "
        f"gpus={config.num_gpus} residency={pipeline.residency}"
    )
    app = create_app(pipeline, served_model_name=config.served_model_name)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        pipeline.shutdown()
