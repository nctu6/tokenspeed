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

"""``/v1/videos`` (async job) and ``/v1/videos/sync`` for MiniMax-H3.

Sibling of :mod:`asr_http` and :mod:`music3_http`. The smg gateway speaks a
text-in/text-out proto, so a DiT video model cannot be reached through it.
``serve()`` also dispatches MiniMax-Music3 to :mod:`music3_http`
(``/v1/audio/speech``) when the checkpoint family is ``music3``.

Flags intentionally mirror familiar vLLM / vLLM-Omni names where the meaning
matches (``--tensor-parallel-size``, ``--served-model-name``, ``--host``,
``--port``, ``--dtype``, ``--trust-remote-code``, ``--task-type``), but the
implementation is TokenSpeed-native (no omni tree).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from tokenspeed.runtime.diffusion.config import H3Config
from tokenspeed.runtime.diffusion.job import JobStatus, VideoJobQueue
from tokenspeed.runtime.diffusion.pipeline import H3GenerateRequest, H3Pipeline
from tokenspeed.runtime.diffusion.references import (
    audio_url_from_field,
    cleanup_staged,
    stage_uploads,
    validate_ref2va_counts,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve", "parse_diffusion_argv"]


def parse_diffusion_argv(argv: list[str]) -> argparse.Namespace:
    """Parse ``ts serve <model> [flags...]`` for the diffusion runtime.

    Unknown LLM-only flags are ignored so a shared launch script can pass a
    superset of knobs without failing closed.
    """
    parser = argparse.ArgumentParser(prog="tokenspeed serve (diffusion)", add_help=False)
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--model", dest="model_flag", default=None)
    parser.add_argument("--served-model-name", default="test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--task-type", "--model-variant", dest="task_type", default="fl2va")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--ulysses-degree", "--usp", type=int, default=None)
    parser.add_argument("--ring", type=int, default=None)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=None)
    parser.add_argument("--text-encoder-tp-size", type=int, default=None)
    parser.add_argument("--diffusion-attention-backend", default=None)
    parser.add_argument("--enable-dlo", action="store_true", default=False)
    parser.add_argument("--dlo-resident-layers", type=int, default=None)
    parser.add_argument("--no-cpu-offload", action="store_true", default=False)
    parser.add_argument("--memory-reserve-margin", default="12GB")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.info("diffusion serve: ignoring non-diffusion flags: %s", " ".join(unknown))
    model = args.model_flag or args.model
    if not model:
        raise SystemExit("tokenspeed serve <model> is required for diffusion")
    args.model = model
    # TP size is the vLLM-shaped knob scripts already pass; map to num_gpus.
    if args.num_gpus is None:
        args.num_gpus = args.tensor_parallel_size or max(
            1, len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()])
        )
    return args


def _parse_extra_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_params must be JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("extra_params must be a JSON object")
    return value


def create_app(
    pipeline: H3Pipeline,
    queue: VideoJobQueue,
    served_model_name: str,
    *,
    generate_fn=None,
) -> FastAPI:
    app = FastAPI(title="TokenSpeed diffusion server")
    app.state.pipeline = pipeline
    app.state.queue = queue
    app.state.served = served_model_name
    # Sync path must use the same generate_fn as the job queue so USP collectives
    # stay on the rank-0 main thread (HTTP alone would deadlock workers).
    app.state.generate = generate_fn or pipeline.generate

    @app.get("/health")
    def health():
        return {"status": "ok", "runtime": "diffusion"}

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

    def _collect_upload_bytes(
        files: list[UploadFile] | UploadFile | None,
    ) -> list[tuple[bytes, str | None, str | None]]:
        if files is None:
            return []
        if not isinstance(files, list):
            files = [files]
        out: list[tuple[bytes, str | None, str | None]] = []
        for item in files:
            if item is None:
                continue
            data = item.file.read()
            out.append((data, item.filename, item.content_type))
        return out

    def _request_from_form(
        *,
        prompt: str,
        width: int,
        height: int,
        fps: int,
        num_inference_steps: int,
        seed: int | None,
        extra_params: str | None,
        input_reference: UploadFile | None,
        last_image: UploadFile | None,
        input_references: list[UploadFile] | UploadFile | None,
        audio_reference: str | None,
    ) -> tuple[H3GenerateRequest, list]:
        """Build a generate request; return staged refs that the caller must clean up."""
        extra = _parse_extra_params(extra_params)
        task = str(extra.get("task") or "t2va").lower()
        duration = float(extra.get("duration", 5.0))

        multi = _collect_upload_bytes(input_references)
        single = _collect_upload_bytes(input_reference)
        last = _collect_upload_bytes(last_image)
        audio_url = audio_url_from_field(audio_reference)

        image_bytes = None
        last_bytes = None
        staged: list = []
        reference_uploads: list[tuple[str, str]] | None = None

        if task == "ref2va":
            items = list(multi) if multi else list(single)
            if not items and not audio_url:
                raise ValueError(
                    "ref2va requires multipart input_references (or input_reference) "
                    "and/or audio_reference"
                )
            staged = stage_uploads(items) if items else []
            if audio_url:
                # Stage remote/local audio as a trailing reference via tempfile download.
                import urllib.request

                suffix = ".wav"
                lower = audio_url.lower().split("?")[0]
                for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
                    if lower.endswith(ext):
                        suffix = ext
                        break
                import tempfile as _tempfile

                fd, audio_path = _tempfile.mkstemp(prefix="ts-h3-audio-", suffix=suffix)
                os.close(fd)
                try:
                    if os.path.isfile(audio_url):
                        with open(audio_url, "rb") as src, open(audio_path, "wb") as dst:
                            dst.write(src.read())
                    else:
                        urllib.request.urlretrieve(audio_url, audio_path)
                except Exception:
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass
                    raise
                from tokenspeed.runtime.diffusion.references import ReferenceUpload

                staged.append(
                    ReferenceUpload(
                        path=audio_path, kind="audio", filename=os.path.basename(audio_path)
                    )
                )
            validate_ref2va_counts([s.kind for s in staged])
            reference_uploads = [(s.path, s.kind) for s in staged]
        else:
            # FL2VA / T2VA: prefer repeated input_references as first/last keyframes.
            if multi:
                if len(multi) > 2:
                    raise ValueError("fl2va accepts at most two input_references (first, last)")
                image_bytes = multi[0][0]
                if len(multi) == 2:
                    last_bytes = multi[1][0]
            else:
                image_bytes = single[0][0] if single else None
                last_bytes = last[0][0] if last else None

        req = H3GenerateRequest(
            prompt=prompt,
            task=task,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image_bytes=image_bytes or None,
            last_image_bytes=last_bytes or None,
            reference_uploads=reference_uploads,
            extra_params=extra,
        )
        return req, staged

    @app.post("/v1/videos")
    def create_video(
        prompt: str = Form(...),
        model: str = Form(default=""),
        width: int = Form(default=672),
        height: int = Form(default=384),
        fps: int = Form(default=24),
        num_inference_steps: int = Form(default=50),
        seed: int | None = Form(default=42),
        extra_params: str | None = Form(default=None),
        input_reference: UploadFile | None = File(default=None),
        last_image: UploadFile | None = File(default=None),
        input_references: list[UploadFile] | None = File(default=None),
        audio_reference: str | None = Form(default=None),
    ):
        staged: list = []
        try:
            req, staged = _request_from_form(
                prompt=prompt,
                width=width,
                height=height,
                fps=fps,
                num_inference_steps=num_inference_steps,
                seed=seed,
                extra_params=extra_params,
                input_reference=input_reference,
                last_image=last_image,
                input_references=input_references,
                audio_reference=audio_reference,
            )
            # Keep staged files until the worker finishes this job.
            if staged:
                req.extra_params = dict(req.extra_params or {})
                req.extra_params["_ts_staged_refs"] = [s.path for s in staged]
            job = queue.submit(req, model=model or served_model_name)
        except (ValueError, RuntimeError) as exc:
            cleanup_staged(staged)
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "id": job.id,
                "object": "video",
                "model": job.model,
                "status": job.status.value,
                "created_at": int(job.created_at),
            },
            status_code=202,
        )

    @app.get("/v1/videos/{video_id}")
    def get_video(video_id: str):
        job = queue.get(video_id)
        if job is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        body: dict[str, Any] = {
            "id": job.id,
            "object": "video",
            "model": job.model,
            "status": job.status.value,
            "created_at": int(job.created_at),
        }
        if job.status == JobStatus.running:
            body["progress"] = {"step": job.progress_step, "total": job.progress_total}
        if job.status == JobStatus.failed:
            body["error"] = job.error
        return body

    @app.get("/v1/videos/{video_id}/content")
    def get_video_content(video_id: str):
        job = queue.get(video_id)
        if job is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if job.status != JobStatus.completed or job.result is None:
            return JSONResponse(
                {"error": f"video not ready (status={job.status.value})"},
                status_code=409,
            )
        return Response(content=job.result.mp4_bytes, media_type="video/mp4")

    @app.post("/v1/videos/sync")
    def create_video_sync(
        prompt: str = Form(...),
        model: str = Form(default=""),
        width: int = Form(default=672),
        height: int = Form(default=384),
        fps: int = Form(default=24),
        num_inference_steps: int = Form(default=50),
        seed: int | None = Form(default=42),
        extra_params: str | None = Form(default=None),
        input_reference: UploadFile | None = File(default=None),
        last_image: UploadFile | None = File(default=None),
        input_references: list[UploadFile] | None = File(default=None),
        audio_reference: str | None = Form(default=None),
    ):
        """Blocking smoke path: run one generation and return MP4 bytes."""
        staged: list = []
        try:
            req, staged = _request_from_form(
                prompt=prompt,
                width=width,
                height=height,
                fps=fps,
                num_inference_steps=num_inference_steps,
                seed=seed,
                extra_params=extra_params,
                input_reference=input_reference,
                last_image=last_image,
                input_references=input_references,
                audio_reference=audio_reference,
            )
            result = app.state.generate(req)
        except Exception as exc:
            logger.exception("sync video failed")
            cleanup_staged(staged)
            return JSONResponse({"error": str(exc)}, status_code=500)
        cleanup_staged(staged)
        return Response(content=result.mp4_bytes, media_type="video/mp4")

    return app


def serve(argv: list[str], *, host: str | None = None, port: int | None = None) -> None:
    import queue as queue_mod
    import threading
    from concurrent.futures import Future

    import uvicorn

    from tokenspeed.runtime.diffusion.family import detect_diffusion_family
    from tokenspeed.runtime.diffusion.ulysses import (
        CMD_SHUTDOWN,
        barrier,
        init_ulysses_world,
        leader_signal,
        maybe_reexec_torchrun,
        usp_world_size,
        worker_loop,
    )

    # Peek model path before full H3 parse so Music3 does not inherit video defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("model", nargs="?", default=None)
    pre.add_argument("--model", dest="model_flag", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    model_path = pre_args.model_flag or pre_args.model
    if model_path:
        try:
            family = detect_diffusion_family(model_path)
        except ValueError:
            family = "h3"
        if family == "music3":
            from tokenspeed.runtime.entrypoints.music3_http import serve as music3_serve

            music3_serve(argv, host=host, port=port)
            return

    args = parse_diffusion_argv(argv)
    host = host or args.host
    port = port or args.port
    ulysses = int(args.ulysses_degree or 1)
    ring = int(args.ring or 1)
    world = usp_world_size(ulysses, ring)
    # Diffusers CP needs N processes; re-exec under torchrun when needed.
    maybe_reexec_torchrun(argv, world)

    config = H3Config(
        model_path=args.model,
        workflow=args.task_type,
        served_model_name=args.served_model_name,
        num_gpus=int(args.num_gpus),
        dtype=args.dtype,
        memory_reserve_margin=args.memory_reserve_margin,
        enable_cpu_offload=not args.no_cpu_offload,
        ulysses_degree=ulysses,
        ring_degree=ring,
        text_encoder_tp_size=int(args.text_encoder_tp_size or 1),
        diffusion_attention_backend=args.diffusion_attention_backend,
    )
    # Fail closed before loading weights when weight-TP asks are unsupported.
    residency = config.resolve_residency()
    usp_world = init_ulysses_world(ulysses, ring) if residency == "ulysses" else None
    pipeline = H3Pipeline(config, usp_world=usp_world)

    if usp_world is not None and not usp_world.is_leader:
        # Non-leader ranks only join DiT collectives; no HTTP bind.
        def _worker_generate(payload):
            pipeline.generate_from_payload(payload)

        try:
            worker_loop(_worker_generate)
        finally:
            pipeline.shutdown()
        return

    # USP collectives must run on the main thread (same thread that inited NCCL).
    # HTTP (uvicorn) and VideoJobQueue otherwise call generate from other threads
    # and deadlock against the worker ranks. Bridge via a handoff queue.
    usp_jobs: queue_mod.Queue | None = None
    if usp_world is not None:
        usp_jobs = queue_mod.Queue()

    def _leader_generate(req: H3GenerateRequest):
        if usp_jobs is None:
            return pipeline.generate(req)
        fut: Future = Future()
        usp_jobs.put(("generate", req, fut))
        return fut.result()

    queue = VideoJobQueue(_leader_generate)
    logger.info(
        "diffusion server: model=%s served=%s workflow=%s gpus=%s residency=%s usp=%s ring=%s",
        config.model_path,
        config.served_model_name,
        config.workflow,
        config.num_gpus,
        residency,
        ulysses,
        ring,
    )
    app = create_app(pipeline, queue, served_model_name=config.served_model_name, generate_fn=_leader_generate)

    if usp_jobs is None:
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        finally:
            queue.shutdown()
            pipeline.shutdown()
        return

    http_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port, log_level="info"),
        name="ts-h3-http",
        daemon=True,
    )
    http_thread.start()
    logger.info("h3 usp: HTTP on background thread; main thread owns NCCL generate")
    try:
        while True:
            item = usp_jobs.get()
            kind = item[0]
            if kind == "shutdown":
                try:
                    leader_signal(CMD_SHUTDOWN)
                except Exception:
                    logger.exception("usp shutdown signal failed")
                break
            _, req, fut = item
            try:
                payload = pipeline.request_payload(req)
                leader_signal("generate", payload)
                result = pipeline.generate(req)
                barrier()
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
                try:
                    barrier()
                except Exception:
                    pass
    finally:
        queue.shutdown()
        pipeline.shutdown()


if __name__ == "__main__":
    # torchrun -m tokenspeed.runtime.entrypoints.diffusion_http <serve argv...>
    import sys

    serve(sys.argv[1:])
