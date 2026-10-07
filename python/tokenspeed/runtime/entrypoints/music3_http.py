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

"""``/v1/audio/speech`` for MiniMax-Music3 (lyrics + caption → WAV).

Sibling of :mod:`diffusion_http` (video) and :mod:`asr_http` (transcription).
The smg gateway has no music RPC, so this process owns the Diffusers
``MiniMaxMusic3ModularPipeline`` and the OpenAI-shaped speech surface.

Request contract (JSON body, OpenAI speech-shaped):

* ``input`` -- lyrics (structure tags like ``[verse]`` on their own line)
* ``instructions`` -- music description (genre / BPM / vocals / arrangement)
* ``seed`` -- optional int
* ``response_format`` -- ``wav`` only (mp3/opus rejected)
* TokenSpeed extras: ``audio_duration`` (seconds upper bound), 
  ``num_inference_steps``, ``max_new_tokens`` (AR frames at 25 fps)

Unsupported TTS knobs (``voice``, ``speed``, ``temperature``) are rejected
rather than silently ignored.

Streaming (TokenSpeed-native, OpenAI speech-shaped):

* ``stream: true`` (default framing ``stream_format="sse"``) or
  ``stream_format="sse"|"audio"`` streams progressive acoustic windows after
  the AR stage — Diffusers' 200-frame denoise/vocode/crop path, one window at
  a time (real first-audio after the first window; not a late full-song
  buffer labeled as a stream).
* ``stream_format="sse"`` → ``text/event-stream`` with ``speech.audio.delta`` /
  ``speech.audio.done`` / ``speech.audio.error`` (same SSE family as TokenSpeed
  chat).
* ``stream_format="audio"`` → raw ``audio/wav`` or ``audio/pcm`` bytes (WAV
  uses an OpenAI-style placeholder-size header on the first chunk).
* Streaming requires ``response_format`` ``wav`` or ``pcm``.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tokenspeed.runtime.diffusion.music3_config import (
    DEFAULT_AUDIO_DURATION_S,
    DEFAULT_NUM_INFERENCE_STEPS,
    REFERENCE_SAMPLE_RATE,
    Music3Config,
)
from tokenspeed.runtime.diffusion.music3_pipeline import (
    Music3GenerateRequest,
    Music3Pipeline,
)
from tokenspeed.runtime.diffusion.music3_stream import (
    iter_speech_audio_bytes,
    iter_speech_sse,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve", "parse_music3_argv"]

_UNSUPPORTED = (
    "voice",
    "speed",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def parse_music3_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tokenspeed serve (music3)", add_help=False)
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--model", dest="model_flag", default=None)
    parser.add_argument("--served-model-name", default="test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--no-cpu-offload", action="store_true", default=False)
    parser.add_argument("--memory-reserve-margin", default="8GB")
    parser.add_argument("--residency", default="auto")
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=REFERENCE_SAMPLE_RATE,
        help="WAV sample rate for clients (default 32000 to match reference servers; "
        "set 44100 for Diffusers native)",
    )
    # Ignore video/H3-only flags so a shared launcher can pass a superset.
    parser.add_argument("--task-type", "--model-variant", dest="task_type", default=None)
    parser.add_argument("--ulysses-degree", "--usp", type=int, default=None)
    parser.add_argument("--ring", type=int, default=None)
    parser.add_argument("--text-encoder-tp-size", type=int, default=None)
    parser.add_argument("--diffusion-attention-backend", default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.info("music3 serve: ignoring non-music flags: %s", " ".join(unknown))
    model = args.model_flag or args.model
    if not model:
        raise SystemExit("tokenspeed serve <model> is required for music3")
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


def _request_from_body(body: dict[str, Any]) -> Music3GenerateRequest:
    if not isinstance(body, dict):
        raise ValueError("JSON body required")

    for key in _UNSUPPORTED:
        if key in body and body[key] is not None:
            # speed=1.0 is the OpenAI default; allow only the identity value.
            if key == "speed" and float(body[key]) == 1.0:
                continue
            raise ValueError(
                f"MiniMax-Music3 does not support {key!r}; omit it from the request"
            )

    lyrics = body.get("input")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("MiniMax-Music3 requires non-empty 'input' (lyrics)")

    prompt = body.get("instructions")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "MiniMax-Music3 requires non-empty 'instructions' (music description)"
        )

    stream_flag = body.get("stream")
    stream_format = body.get("stream_format")
    if stream_format is not None:
        stream_format = str(stream_format).lower().strip()
        if stream_format not in {"sse", "audio"}:
            raise ValueError("stream_format must be 'sse' or 'audio'")
    want_stream = bool(stream_flag) or stream_format is not None
    if want_stream and stream_format is None:
        # TokenSpeed default matches chat SSE rather than opaque raw bytes.
        stream_format = "sse"
    if stream_flag is False and stream_format is None:
        want_stream = False

    fmt = body.get("response_format") or body.get("format") or "wav"
    fmt_l = str(fmt).lower()
    if fmt_l in {"wave", "audio/wav"}:
        fmt_l = "wav"
    if fmt_l in {"audio/pcm"}:
        fmt_l = "pcm"
    if want_stream:
        if fmt_l not in {"wav", "pcm"}:
            raise ValueError(
                "MiniMax-Music3 streaming requires response_format 'wav' or 'pcm'"
            )
    elif fmt_l not in {"wav"}:
        raise ValueError("MiniMax-Music3 only supports response_format='wav' when not streaming")

    audio_duration = body.get("audio_duration")
    max_new_tokens = body.get("max_new_tokens")
    if audio_duration is None and max_new_tokens is None:
        audio_duration = DEFAULT_AUDIO_DURATION_S
    elif audio_duration is not None:
        audio_duration = float(audio_duration)

    steps = body.get("num_inference_steps")
    if steps is None:
        steps = DEFAULT_NUM_INFERENCE_STEPS
    seed = body.get("seed", 42)
    if seed is not None:
        seed = int(seed)

    return Music3GenerateRequest(
        lyrics=lyrics,
        prompt=prompt,
        audio_duration=float(audio_duration) if audio_duration is not None else DEFAULT_AUDIO_DURATION_S,
        num_inference_steps=int(steps),
        seed=seed,
        max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else None,
        extra_params={k: v for k, v in body.items() if k not in {
            "input", "instructions", "model", "response_format", "format",
            "audio_duration", "num_inference_steps", "seed", "max_new_tokens",
            "voice", "speed", "stream", "stream_format",
        }},
        stream=want_stream,
        stream_format=stream_format,
        response_format=fmt_l,
    )



def create_app(pipeline: Music3Pipeline, served_model_name: str) -> FastAPI:
    app = FastAPI(title="TokenSpeed MiniMax-Music3 server")
    app.state.pipeline = pipeline
    app.state.served = served_model_name

    @app.get("/health")
    def health():
        return {"status": "ok", "runtime": "diffusion", "family": "music3"}

    @app.get("/v1/models")
    def models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "tokenspeed",
                }
            ],
        }

    @app.post("/v1/audio/speech")
    async def create_speech(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON object required"}, status_code=400)

        model = body.get("model") or served_model_name
        if model and model != served_model_name:
            # Soft accept aliases: served name is canonical.
            logger.info("music3: request model=%s served=%s", model, served_model_name)

        try:
            req = _request_from_body(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        want_stream = bool(req.stream)
        stream_format = req.stream_format
        response_format = req.response_format

        if want_stream:
            try:
                chunk_iter = pipeline.generate_stream(req)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:
                logger.exception("music3 speech stream setup failed")
                return JSONResponse({"error": str(exc)}, status_code=500)

            if stream_format == "audio":
                media = "audio/pcm" if response_format == "pcm" else "audio/wav"

                def byte_gen():
                    try:
                        yield from iter_speech_audio_bytes(
                            chunk_iter, response_format=response_format
                        )
                    except Exception:
                        logger.exception("music3 speech audio stream failed")
                        raise

                return StreamingResponse(
                    byte_gen(),
                    media_type=media,
                    headers={
                        "X-TokenSpeed-Stream": "audio",
                        "X-TokenSpeed-Response-Format": response_format,
                    },
                )

            # Default / explicit SSE
            def sse_gen():
                try:
                    yield from iter_speech_sse(
                        chunk_iter, response_format=response_format
                    )
                except Exception:
                    logger.exception("music3 speech SSE stream failed")
                    raise

            return StreamingResponse(
                sse_gen(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-TokenSpeed-Stream": "sse",
                    "X-TokenSpeed-Response-Format": response_format,
                },
            )

        try:
            result = pipeline.generate(req)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.exception("music3 speech failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

        headers = {
            "X-TokenSpeed-Sample-Rate": str(result.sample_rate),
            "X-TokenSpeed-Duration-Seconds": f"{result.duration_s:.3f}",
            "X-TokenSpeed-Channels": str(result.channels),
        }
        return Response(
            content=result.wav_bytes,
            media_type="audio/wav",
            headers=headers,
        )

    return app


def serve(argv: list[str], *, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    args = parse_music3_argv(argv)
    host = host or args.host
    port = port or args.port

    config = Music3Config(
        model_path=args.model,
        served_model_name=args.served_model_name,
        num_gpus=int(args.num_gpus),
        dtype=args.dtype,
        memory_reserve_margin=args.memory_reserve_margin,
        enable_cpu_offload=not args.no_cpu_offload,
        residency=args.residency,
        output_sample_rate=int(args.output_sample_rate),
    )
    # Fail closed early on bad residency knobs.
    residency = config.resolve_residency()
    pipeline = Music3Pipeline(config)
    logger.info(
        "music3 server: model=%s served=%s gpus=%s residency=%s out_sr=%s",
        config.model_path,
        config.served_model_name,
        config.num_gpus,
        residency,
        config.output_sample_rate,
    )
    app = create_app(pipeline, served_model_name=config.served_model_name)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    import sys

    serve(sys.argv[1:])
