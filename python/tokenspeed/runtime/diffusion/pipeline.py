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

"""MiniMax-H3 generation through released HuggingFace Diffusers.

TokenSpeed owns residency, workflow selection, muxing, and the HTTP job
surface. The DiT / VAE / conditioner implementations come from the released
``diffusers`` package (``ModularPipeline`` / ``MiniMaxH3Blocks``) -- the same
pattern Whisper uses for ``transformers``, not a vendored omni tree.

Multi-GPU layouts (vLLM-flag-shaped knobs):

* ``--tensor-parallel-size 2`` / ``--num-gpus 2`` with USP=1 → HF ModularPipeline
  **device_split** (conditioner on ``cuda:1``, DiT+VAEs on ``cuda:0``).
* ``--ulysses-degree/--usp N`` (and/or ``--ring``) → TokenSpeed **ulysses**
  residency: torchrun N ranks, each holds a DiT replica on ``cuda:LOCAL_RANK``,
  Diffusers Context Parallel shards attention sequence (see ``diffusion/ulysses.py``).
  Not a vendored omni USP tree. Weight-TP stays fail-closed (no H3 ``_tp_plan``).
* One GPU / USP=1 without split → full auto CPU offload on ``cuda:0``.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.diffusion.config import (
    AUDIO_SIGMA_SHIFT,
    H3Config,
    SUPPORTED_FPS,
    VIDEO_SIGMA_SHIFT,
    frames_for_duration,
    resolve_diffusion_checkpoint,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["H3Pipeline", "H3GenerateRequest", "H3GenerateResult"]


@dataclass
class H3GenerateRequest:
    prompt: str
    task: str = "t2va"  # t2va | fl2va | ref2va
    width: int = 672
    height: int = 384
    duration: float = 5.0
    fps: int = SUPPORTED_FPS
    num_inference_steps: int = 50
    seed: int | None = 42
    # Optional keyframe bytes (PNG/JPEG) for fl2va.
    image_bytes: bytes | None = None
    last_image_bytes: bytes | None = None
    # Optional reference file paths for ref2va (decoded by diffusers helpers).
    reference_paths: list[str] | None = None
    # Ordered (path, kind) pairs from multipart staging; preferred over paths.
    reference_uploads: list[tuple[str, str]] | None = None
    extra_params: dict[str, Any] | None = None


@dataclass
class H3GenerateResult:
    mp4_bytes: bytes
    width: int
    height: int
    num_frames: int
    sampling_rate: int
    seed: int | None


def _require_diffusers():
    try:
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[ts serve] diffusion runtime needs diffusers/av/imageio-ffmpeg:\n"
            "           re-run ./scripts/install.sh\n"
            "           (or: pip install 'tokenspeed[diffusion]')\n"
            f"           import error: {exc}"
        ) from exc


class H3Pipeline:
    """Load once, generate many. Single-flight: caller serializes jobs."""

    def __init__(self, config: H3Config, *, usp_world=None):
        _require_diffusers()
        self.config = config
        self.model_path = resolve_diffusion_checkpoint(config.model_path)
        self._pipe = None
        self._usp_world = usp_world
        self._load()

    def _dtype(self):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }.get(self.config.dtype.lower(), torch.bfloat16)

    def _load(self) -> None:
        import os

        import torch
        from diffusers import ComponentsManager, ModularPipeline

        workflow = self.config.load_components_workflow()
        dtype = self._dtype()
        n = max(1, int(self.config.num_gpus))
        residency = self.config.resolve_residency()
        # Env override kept for A/B: TOKENSPEED_H3_SPLIT=0 forces single.
        env_split = os.environ.get("TOKENSPEED_H3_SPLIT")
        if env_split == "0" and residency != "ulysses":
            residency = "single"
        elif env_split == "1" and residency != "ulysses":
            residency = "device_split" if n >= 2 else "single"
        want_split = residency == "device_split"
        logger.info(
            "h3 load: path=%s workflow=%s gpus=%s dtype=%s residency=%s usp=%s ring=%s audio_shift=%s video_shift=%s",
            self.model_path,
            workflow or "all",
            n,
            dtype,
            residency,
            self.config.ulysses_degree,
            self.config.ring_degree,
            AUDIO_SIGMA_SHIFT,
            VIDEO_SIGMA_SHIFT,
        )

        if residency == "ulysses":
            self._load_ulysses(workflow=workflow, dtype=dtype)
            return

        if (
            want_split
            and n >= 2
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
        ):
            blocks = ModularPipeline.from_pretrained(self.model_path)
            wf = workflow or "fl2va"
            try:
                flow = blocks.blocks.get_workflow(wf)
            except Exception:
                flow = None
            if (
                flow is not None
                and hasattr(flow, "sub_blocks")
                and "text_encoder" in getattr(flow, "sub_blocks", {})
            ):
                from diffusers.modular_pipelines import SequentialPipelineBlocks

                # Ref2VA/FL2VA need before_encode (normalize refs/keyframes) on the
                # conditioner side; t2va only has text_encoder. VAE encode of
                # references stays with rest@cuda:0 (shares the video/audio VAEs).
                cond_names = [
                    name
                    for name in ("before_encode", "text_encoder")
                    if name in flow.sub_blocks
                ]
                cond_dict = {name: flow.sub_blocks.pop(name) for name in cond_names}
                cond_blocks = (
                    SequentialPipelineBlocks.from_blocks_dict(cond_dict)
                    if len(cond_dict) > 1
                    else next(iter(cond_dict.values()))
                )

                text_mgr = ComponentsManager()
                if self.config.enable_cpu_offload:
                    text_mgr.enable_auto_cpu_offload(
                        device="cuda:1",
                        memory_reserve_margin=self.config.memory_reserve_margin,
                    )
                conditioner = cond_blocks.init_pipeline(
                    self.model_path, components_manager=text_mgr
                )
                conditioner.load_components(dtype=dtype)

                rest_mgr = ComponentsManager()
                if self.config.enable_cpu_offload:
                    rest_mgr.enable_auto_cpu_offload(
                        device="cuda:0",
                        memory_reserve_margin=self.config.memory_reserve_margin,
                    )
                rest = flow.init_pipeline(self.model_path, components_manager=rest_mgr)
                rest.load_components(dtype=dtype)
                self._apply_audio_shift(rest)
                self._pipe = ("split", conditioner, rest)
                logger.info(
                    "h3 load: 2-GPU split (conditioner=%s@cuda:1, rest@cuda:0)",
                    "+".join(cond_names),
                )
                return

        manager = ComponentsManager()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.config.enable_cpu_offload and device.startswith("cuda"):
            manager.enable_auto_cpu_offload(
                device=device,
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
        # Keep all blocks at from_pretrained; select the DiT partition only at
        # load_components (see H3Config.load_components_workflow).
        pipe = ModularPipeline.from_pretrained(
            self.model_path, components_manager=manager
        )
        load_kwargs = {"dtype": dtype}
        if workflow is not None:
            load_kwargs["workflow"] = workflow
        pipe.load_components(**load_kwargs)
        self._apply_audio_shift(pipe)
        self._pipe = ("single", pipe)
        logger.info(
            "h3 load: single-pipeline on %s load_workflow=%s (num_gpus=%s visible)",
            device,
            workflow or "all",
            n,
        )

    def _load_ulysses(self, *, workflow: str | None, dtype) -> None:
        """Leader text-encode + all-rank DiT Ulysses (Diffusers Context Parallel).

        Each rank holds a DiT/VAE **replica** on ``cuda:LOCAL_RANK``. Only rank 0
        also loads the conditioner (``before_encode`` + ``text_encoder``). Generate
        broadcasts encoded conditioning so followers never run the 66GB TE — that
        was the dual-rank encode race that wedged the first Ulysses collective.
        """
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        from tokenspeed.runtime.diffusion.ulysses import enable_dit_context_parallel

        world = self._usp_world
        if world is None:
            raise RuntimeError(
                "residency=ulysses requires init_ulysses_world before H3Pipeline load"
            )
        local = f"cuda:{world.local_rank}" if torch.cuda.is_available() else "cpu"
        blocks = ModularPipeline.from_pretrained(self.model_path)
        wf = workflow or "t2va"
        try:
            flow = blocks.blocks.get_workflow(wf)
        except Exception as exc:
            raise RuntimeError(
                f"ulysses load: workflow={wf!r} missing on ModularPipeline: {exc}"
            ) from exc

        # Conditioner blocks leave the shared flow; rest keeps denoise+decode.
        cond_names = [
            name
            for name in ("before_encode", "text_encoder")
            if name in getattr(flow, "sub_blocks", {})
        ]
        cond_dict = {name: flow.sub_blocks.pop(name) for name in cond_names}

        # One ComponentsManager per rank so auto CPU offload sees both the
        # leader conditioner (text_encoder) and DiT/VAEs — separate managers
        # cannot evict each other's VRAM and OOMs ~96 GiB sm120 cards.
        shared_mgr = ComponentsManager()
        if self.config.enable_cpu_offload and local.startswith("cuda"):
            shared_mgr.enable_auto_cpu_offload(
                device=local,
                memory_reserve_margin=self.config.memory_reserve_margin,
            )

        conditioner = None
        if world.is_leader:
            if not cond_dict:
                raise RuntimeError(
                    "ulysses load: leader needs text_encoder (and optional before_encode) "
                    f"in workflow={wf!r}"
                )
            cond_blocks = (
                SequentialPipelineBlocks.from_blocks_dict(cond_dict)
                if len(cond_dict) > 1
                else next(iter(cond_dict.values()))
            )
            conditioner = cond_blocks.init_pipeline(
                self.model_path, components_manager=shared_mgr
            )
            conditioner.load_components(dtype=dtype)

        rest = flow.init_pipeline(self.model_path, components_manager=shared_mgr)
        rest.load_components(dtype=dtype)
        self._apply_audio_shift(rest)
        enabled = enable_dit_context_parallel(
            rest,
            ulysses_degree=world.ulysses_degree,
            ring_degree=world.ring_degree,
            attention_backend=self.config.diffusion_attention_backend,
        )
        self._pipe = ("ulysses", conditioner, rest)
        logger.info(
            "h3 load: ulysses residency on %s rank=%s/%s transformers_cp=%s conditioner=%s",
            local,
            world.rank,
            world.world_size,
            enabled,
            ("+".join(cond_names) if world.is_leader else "follower"),
        )

    def _apply_audio_shift(self, pipe) -> None:
        """Force audio scheduler shift to 3.0 when the attribute exists."""
        sched = getattr(pipe, "audio_scheduler", None)
        if sched is None:
            return
        if hasattr(sched, "config") and hasattr(sched.config, "shift"):
            try:
                sched.config.shift = AUDIO_SIGMA_SHIFT
            except Exception:
                pass
        if hasattr(sched, "shift"):
            try:
                sched.shift = AUDIO_SIGMA_SHIFT
            except Exception:
                pass

    def _pil_from_bytes(self, data: bytes | None):
        if not data:
            return None
        from PIL import Image

        return Image.open(io.BytesIO(data)).convert("RGB")

    def _build_call_kwargs(self, req: H3GenerateRequest) -> dict[str, Any]:
        import torch

        extra = dict(req.extra_params or {})
        task = (extra.pop("task", None) or req.task or "t2va").lower()
        duration = float(extra.pop("duration", req.duration))
        fps = int(extra.pop("fps", req.fps) or SUPPORTED_FPS)
        num_frames = int(extra.pop("num_frames", 0) or frames_for_duration(duration, fps))
        steps = int(extra.pop("num_inference_steps", req.num_inference_steps))
        seed = req.seed if req.seed is not None else extra.pop("seed", None)

        kwargs: dict[str, Any] = {
            "prompt": req.prompt,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "output": ["videos", "audio", "sampling_rate"],
        }
        if req.width:
            kwargs["width"] = int(req.width)
        if req.height:
            kwargs["height"] = int(req.height)
        if seed is not None:
            kwargs["generator"] = torch.Generator().manual_seed(int(seed))

        image = self._pil_from_bytes(req.image_bytes)
        last_image = self._pil_from_bytes(req.last_image_bytes)
        if task in {"fl2va", "t2va"} and image is not None:
            kwargs["image"] = image
        if task == "fl2va" and last_image is not None:
            kwargs["last_image"] = last_image

        if task == "ref2va":
            refs = self._build_references(req)
            if refs:
                kwargs["references"] = refs

        # Pass through remaining extras that the modular blocks understand.
        for key in ("guidance_scale", "output_type", "attention_kwargs"):
            if key in extra:
                kwargs[key] = extra[key]
        return kwargs, seed, num_frames

    def _build_references(self, req: H3GenerateRequest) -> list[Any]:
        """Decode ordered Ref2VA references from staged uploads or path list."""
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        from tokenspeed.runtime.diffusion.references import (
            classify_media,
            validate_ref2va_counts,
        )

        entries: list[tuple[str, str]] = []
        if req.reference_uploads:
            entries.extend(req.reference_uploads)
        elif req.reference_paths:
            for path in req.reference_paths:
                entries.append((path, classify_media(filename=path)))
        if not entries:
            return []
        validate_ref2va_counts([kind for _, kind in entries])
        refs: list[Any] = []
        for path, kind in entries:
            if kind == "image":
                refs.append(MiniMaxH3ImageReference.from_file(path))
            elif kind == "video":
                refs.append(MiniMaxH3VideoReference.from_file(path))
            elif kind == "audio":
                refs.append(MiniMaxH3AudioReference.from_file(path))
            else:
                raise ValueError(f"unsupported reference kind {kind!r} for {path}")
        return refs

    def generate(self, req: H3GenerateRequest) -> H3GenerateResult:
        return self._generate_local(req)

    def generate_from_payload(self, payload: dict) -> H3GenerateResult | None:
        """Worker entry: rebuild request from a broadcast payload."""
        req = _request_from_payload(payload)
        result = self._generate_local(req)
        # Non-leader ranks participate in CP collectives but discard the MP4.
        if self._usp_world is not None and not self._usp_world.is_leader:
            return None
        return result

    def _generate_local(self, req: H3GenerateRequest) -> H3GenerateResult:
        call_kwargs, seed, num_frames = self._build_call_kwargs(req)
        kind = self._pipe[0]
        if kind == "single":
            pipe = self._pipe[1]
            results = pipe(**call_kwargs)
        elif kind == "ulysses":
            results = self._generate_ulysses(call_kwargs)
        else:
            conditioner, rest = self._pipe[1], self._pipe[2]
            # Conditioner: prompt + refs/keyframes (+ geometry for before_encode).
            # Rest: VAE-encode refs, denoise, decode.
            cond_keys = {
                "prompt",
                "image",
                "last_image",
                "references",
                "height",
                "width",
                "num_frames",
            }
            cond_kwargs = {k: v for k, v in call_kwargs.items() if k in cond_keys}
            rest_kwargs = {k: v for k, v in call_kwargs.items() if k not in cond_keys}
            state = conditioner(**cond_kwargs)
            results = rest(state=state, **rest_kwargs)

        videos = results["videos"]
        audio = results["audio"]
        rate = int(results["sampling_rate"])
        video0 = videos[0] if isinstance(videos, (list, tuple)) else videos
        audio0 = audio[0] if isinstance(audio, (list, tuple)) and hasattr(audio[0], "shape") else audio

        mp4 = self._mux_mp4(video0, audio0, rate, fps=req.fps or SUPPORTED_FPS)
        height = int(getattr(video0, "shape", [0, 0, 0])[-3]) if hasattr(video0, "shape") else req.height
        width = int(getattr(video0, "shape", [0, 0, 0])[-2]) if hasattr(video0, "shape") else req.width
        # PIL list fallback
        if isinstance(video0, list) and video0:
            im = video0[0]
            width, height = im.size
        return H3GenerateResult(
            mp4_bytes=mp4,
            width=width or req.width,
            height=height or req.height,
            num_frames=num_frames,
            sampling_rate=rate,
            seed=seed,
        )

    def _mux_mp4(self, video, audio, sample_rate: int, fps: int) -> bytes:
        """Encode H.264 + AAC into one MP4 byte string."""
        from diffusers.utils.export_utils import encode_video

        with tempfile.TemporaryDirectory(prefix="ts-h3-") as tmp:
            out = os.path.join(tmp, "out.mp4")
            try:
                encode_video(
                    video,
                    fps=fps,
                    output_path=out,
                    audio=audio,
                    audio_sample_rate=sample_rate,
                )
            except TypeError:
                # Older diffusers may not take audio=; write video then try pyav mux.
                encode_video(video, fps=fps, output_path=out)
            with open(out, "rb") as handle:
                return handle.read()

    def shutdown(self) -> None:
        self._pipe = None

    def _generate_ulysses(self, call_kwargs: dict[str, Any]):
        """Leader-only text encode, broadcast conditioning, all-rank denoise/decode."""
        from tokenspeed.runtime.diffusion.ulysses import (
            barrier,
            broadcast_pyobject,
            pipeline_state_from_payload,
            pipeline_state_to_payload,
        )

        world = self._usp_world
        if world is None:
            raise RuntimeError("ulysses generate without usp_world")
        conditioner, rest = self._pipe[1], self._pipe[2]
        cond_keys = {
            "prompt",
            "image",
            "last_image",
            "references",
            "height",
            "width",
            "num_frames",
        }
        rest_kwargs = {k: v for k, v in call_kwargs.items() if k not in cond_keys}

        # Align ranks before encode so followers wait on the broadcast, not on DiT A2A.
        barrier()
        local = f"cuda:{world.local_rank}"
        import torch

        # Phase VRAM: DiT↔CPU before TE encode, TE↔CPU before denoise. Shared
        # ComponentsManager auto-offload alone left DiT resident across requests
        # and OOMd the next Ref2VA encode (wide refs) on ~140 GiB cards.
        try:
            rest.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        barrier()
        if world.is_leader:
            if conditioner is None:
                raise RuntimeError("ulysses leader missing conditioner pipeline")
            cond_kwargs = {k: v for k, v in call_kwargs.items() if k in cond_keys}
            state = conditioner(**cond_kwargs)
            enc_payload = pipeline_state_to_payload(state)
            try:
                conditioner.to("cpu")
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            state = None
            enc_payload = None
        enc_payload = broadcast_pyobject(enc_payload, src=0)
        if enc_payload is None:
            raise RuntimeError("ulysses conditioning broadcast returned None")
        # Leader keeps the live encode state (float embeds on CUDA, int tags on CPU).
        # Followers rebuild: float tensors → local CUDA; integer layout tags stay CPU.
        if not world.is_leader:
            state = pipeline_state_from_payload(enc_payload, device=local)
        # Same seed on every rank → identical prepare_latents before CP shards.
        results = rest(state=state, **rest_kwargs)
        barrier()
        return results

    def request_payload(self, req: H3GenerateRequest) -> dict:

        """Picklable snapshot for USP broadcast (paths, not open handles)."""
        return {
            "prompt": req.prompt,
            "task": req.task,
            "width": req.width,
            "height": req.height,
            "duration": req.duration,
            "fps": req.fps,
            "num_inference_steps": req.num_inference_steps,
            "seed": req.seed,
            "image_bytes": req.image_bytes,
            "last_image_bytes": req.last_image_bytes,
            "reference_paths": list(req.reference_paths or []),
            "reference_uploads": list(req.reference_uploads or []),
            "extra_params": dict(req.extra_params or {}),
        }


def _request_from_payload(payload: dict) -> H3GenerateRequest:
    return H3GenerateRequest(
        prompt=payload.get("prompt") or "",
        task=payload.get("task") or "t2va",
        width=int(payload.get("width") or 672),
        height=int(payload.get("height") or 384),
        duration=float(payload.get("duration") or 5.0),
        fps=int(payload.get("fps") or SUPPORTED_FPS),
        num_inference_steps=int(payload.get("num_inference_steps") or 50),
        seed=payload.get("seed"),
        image_bytes=payload.get("image_bytes"),
        last_image_bytes=payload.get("last_image_bytes"),
        reference_paths=payload.get("reference_paths"),
        reference_uploads=payload.get("reference_uploads"),
        extra_params=payload.get("extra_params"),
    )
