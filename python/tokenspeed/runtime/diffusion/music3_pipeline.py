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

"""MiniMax-Music3 generation through released HuggingFace Diffusers.

TokenSpeed owns residency, WAV encoding, and the HTTP surface. The Global /
Local LM, flow-matching transformer, and Flow-VAE vocoder come from the
released ``diffusers`` ``ModularPipeline`` / ``MiniMaxMusic3Blocks`` -- the
same library pattern as MiniMax-H3 and Whisper (``transformers``), not a
vendored omni tree.

Layouts:

* ``num_gpus=1`` → full pipeline on ``cuda:0`` with optional auto CPU offload
  (~23 GB bf16 peak per Diffusers docs).
* ``num_gpus>=2`` / ``residency=device_split`` → ``semantic_generator``
  (tokenizer + language_model + rvq_depth_decoder) on ``cuda:1``,
  ``denoise``+``decode`` (condition_encoder / transformer / vocoder) on
  ``cuda:0``, chained with ModularPipeline ``state=`` (``frame_hiddens``).
  Falls back to single-device offload if the Diffusers block graph lacks that
  cut.

Streaming (:meth:`generate_stream`) runs AR once, then Diffusers' own
200-frame acoustic windows one at a time (denoise → vocode → crop → yield
PCM). That is real progressive audio, not a full-song buffer labeled as a
stream.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.diffusion.music3_config import (
    DEFAULT_AUDIO_DURATION_S,
    DEFAULT_NUM_INFERENCE_STEPS,
    NATIVE_SAMPLE_RATE,
    Music3Config,
    duration_for_frames,
    frames_for_duration,
    resolve_music3_checkpoint,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.diffusion.music3_stream import (
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    Music3StreamChunk,
    pcm16le_bytes,
    resample_poly_pcm,
)

logger = get_colorful_logger(__name__)

__all__ = ["Music3Pipeline", "Music3GenerateRequest", "Music3GenerateResult", "Music3StreamChunk"]


@dataclass
class Music3GenerateRequest:
    """Lyrics (``input``) + music description (``instructions``) → song."""

    lyrics: str
    prompt: str  # music description / caption
    audio_duration: float = DEFAULT_AUDIO_DURATION_S
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    seed: int | None = 42
    # Optional AR frame budget (25 fps). When set, overrides audio_duration.
    max_new_tokens: int | None = None
    extra_params: dict[str, Any] | None = None
    # HTTP streaming knobs (ignored by non-stream generate()).
    stream: bool = False
    stream_format: str | None = None  # "sse" | "audio"
    response_format: str = "wav"  # wav | pcm (pcm only when streaming)


@dataclass
class Music3GenerateResult:
    wav_bytes: bytes
    sample_rate: int
    num_samples: int
    channels: int
    duration_s: float
    seed: int | None


def _require_diffusers():
    try:
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[ts serve] music3 runtime needs diffusers/soundfile:\n"
            "           re-run ./scripts/install.sh\n"
            "           (or: pip install 'tokenspeed[diffusion]')\n"
            f"           import error: {exc}"
        ) from exc


class Music3Pipeline:
    """Load once, generate many. Caller serializes jobs (single-flight)."""

    def __init__(self, config: Music3Config):
        _require_diffusers()
        self.config = config
        self.model_path = resolve_music3_checkpoint(config.model_path)
        self._pipe = None
        self._sampling_rate = NATIVE_SAMPLE_RATE
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


    def _load_kwargs(self, dtype) -> dict:
        """Kwargs for ``load_components`` (dtype + tokenizer regex fix).

        MiniMax-Music3 ships a tokenizer whose regex trips transformers'
        mistral-pattern warning; ``fix_mistral_regex`` is tokenizer-only
        (must not be broadcast to Qwen3ForCausalLM).
        """
        return {
            "dtype": dtype,
            "fix_mistral_regex": {"tokenizer": True},
        }

    def _load(self) -> None:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        dtype = self._dtype()
        residency = self.config.resolve_residency()
        n = max(1, int(self.config.num_gpus))
        logger.info(
            "music3 load: path=%s gpus=%s dtype=%s residency=%s offload=%s",
            self.model_path,
            n,
            dtype,
            residency,
            self.config.enable_cpu_offload,
        )

        if (
            residency == "device_split"
            and n >= 2
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
        ):
            loaded = self._try_load_split(dtype=dtype)
            if loaded:
                return
            logger.warning(
                "music3 device_split unavailable for this Diffusers build; "
                "falling back to single-GPU auto offload on cuda:0"
            )

        manager = ComponentsManager()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.config.enable_cpu_offload and device.startswith("cuda"):
            manager.enable_auto_cpu_offload(
                device=device,
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
        pipe = ModularPipeline.from_pretrained(
            self.model_path, components_manager=manager
        )
        pipe.load_components(**self._load_kwargs(dtype))
        if not self.config.enable_cpu_offload and device.startswith("cuda"):
            pipe.to(device)
        self._pipe = ("single", pipe)
        self._sampling_rate = int(getattr(pipe, "sampling_rate", NATIVE_SAMPLE_RATE) or NATIVE_SAMPLE_RATE)
        logger.info(
            "music3 load: single-pipeline on %s sampling_rate=%s",
            device,
            self._sampling_rate,
        )

    def _try_load_split(self, *, dtype) -> bool:
        """AR ``semantic_generator``@cuda:1 / ``denoise+decode``@cuda:0.

        MiniMaxMusic3Blocks exposes three top-level sub_blocks with a clean
        cut: semantic_generator outputs ``frame_hiddens``; denoise+decode
        consume them via ModularPipeline ``state=`` handoff (same pattern as
        H3 conditioner→rest). No vendored omni compose tree.
        """
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        root = ModularPipeline.from_pretrained(self.model_path)
        blocks = getattr(root, "blocks", None)
        if blocks is None or not hasattr(blocks, "sub_blocks"):
            return False
        sub = dict(blocks.sub_blocks)
        if "semantic_generator" not in sub:
            return False
        if "denoise" not in sub or "decode" not in sub:
            return False

        ar_block = sub.pop("semantic_generator")
        rest_blocks = SequentialPipelineBlocks.from_blocks_dict(sub)

        ar_mgr = ComponentsManager()
        if self.config.enable_cpu_offload:
            ar_mgr.enable_auto_cpu_offload(
                device="cuda:1",
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
        ar_pipe = ar_block.init_pipeline(self.model_path, components_manager=ar_mgr)
        ar_pipe.load_components(**self._load_kwargs(dtype))

        rest_mgr = ComponentsManager()
        if self.config.enable_cpu_offload:
            rest_mgr.enable_auto_cpu_offload(
                device="cuda:0",
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
        rest_pipe = rest_blocks.init_pipeline(
            self.model_path, components_manager=rest_mgr
        )
        rest_pipe.load_components(**self._load_kwargs(dtype))

        self._pipe = ("split", ar_pipe, rest_pipe)
        self._sampling_rate = int(
            getattr(rest_pipe, "sampling_rate", None)
            or getattr(ar_pipe, "sampling_rate", None)
            or NATIVE_SAMPLE_RATE
        )
        logger.info(
            "music3 load: 2-GPU device_split "
            "(semantic_generator@cuda:1, denoise+decode@cuda:0) sampling_rate=%s",
            self._sampling_rate,
        )
        return True

    def _resolve_duration(self, req: Music3GenerateRequest) -> float:
        if req.max_new_tokens is not None:
            return duration_for_frames(int(req.max_new_tokens))
        return float(req.audio_duration)

    def generate(self, req: Music3GenerateRequest) -> Music3GenerateResult:
        import numpy as np
        import torch

        if not req.lyrics or not str(req.lyrics).strip():
            raise ValueError("MiniMax-Music3 requires non-empty lyrics (input)")
        if not req.prompt or not str(req.prompt).strip():
            raise ValueError(
                "MiniMax-Music3 requires non-empty music description (instructions)"
            )

        duration = self._resolve_duration(req)
        # Clamp via frame helper so we never exceed the checkpoint ceiling.
        duration = duration_for_frames(frames_for_duration(duration))
        steps = max(1, int(req.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS))
        seed = req.seed

        kind = self._pipe[0]

        logger.info(
            "music3 generate: kind=%s duration=%.2fs steps=%s seed=%s lyrics_chars=%s prompt_chars=%s",
            kind,
            duration,
            steps,
            seed,
            len(req.lyrics),
            len(req.prompt),
        )

        if kind == "single":
            pipe = self._pipe[1]
            generator = None
            if seed is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(seed))
            audio = pipe(
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=generator,
                num_inference_steps=steps,
                output="audios",
            )[0]
        elif kind == "split":
            ar_pipe, rest_pipe = self._pipe[1], self._pipe[2]
            # Separate generators: AR samples on cuda:1, flow-matching on cuda:0.
            gen_ar = gen_ac = None
            if seed is not None:
                gen_ar = torch.Generator(device="cuda:1").manual_seed(int(seed))
                gen_ac = torch.Generator(device="cuda:0").manual_seed(int(seed))
            state = ar_pipe(
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=gen_ar,
            )
            audio = rest_pipe(
                state=state,
                generator=gen_ac,
                num_inference_steps=steps,
                output="audios",
            )[0]
            pipe = rest_pipe
        else:
            raise RuntimeError(f"unknown music3 pipe kind {kind!r}")

        # Diffusers returns (batch, channels, samples) or (channels, samples).
        if hasattr(audio, "detach"):
            tensor = audio.detach().float().cpu()
        else:
            tensor = torch.as_tensor(audio).float()
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise RuntimeError(f"unexpected music3 audio shape: {tuple(tensor.shape)}")

        # (channels, samples) → write WAV
        native_sr = int(getattr(pipe, "sampling_rate", self._sampling_rate) or NATIVE_SAMPLE_RATE)
        out_sr = int(self.config.output_sample_rate or native_sr)
        if out_sr <= 0:
            out_sr = native_sr

        wav_bytes, sample_rate, num_samples, channels = self._encode_wav(
            tensor, native_sr=native_sr, out_sr=out_sr
        )
        return Music3GenerateResult(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            num_samples=num_samples,
            channels=channels,
            duration_s=float(num_samples) / float(sample_rate),
            seed=seed,
        )

    @staticmethod
    def _encode_wav(tensor, *, native_sr: int, out_sr: int) -> tuple[bytes, int, int, int]:
        """``tensor`` is (channels, samples) float in roughly [-1, 1]."""
        import numpy as np

        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit(
                "[ts serve] music3 needs soundfile (pip install soundfile); "
                "re-run ./scripts/install.sh"
            ) from exc

        audio = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise RuntimeError(f"wav encode expected (C, T), got {audio.shape}")
        # soundfile wants (samples, channels)
        pcm = audio.T
        pcm = np.clip(pcm, -1.0, 1.0)

        if out_sr != native_sr:
            pcm = _resample_poly(pcm, native_sr, out_sr)
            sample_rate = out_sr
        else:
            sample_rate = native_sr

        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        return data, sample_rate, int(pcm.shape[0]), int(pcm.shape[1])


    def generate_stream(self, req: Music3GenerateRequest):
        """Yield progressive PCM windows after AR; real acoustic chunking.

        Diffusers' stock ModularPipeline call buffers every 200-frame denoise
        window until the song ends. This driver runs the same Diffusers chunk
        denoise / vocoder crop steps one window at a time and yields PCM so
        HTTP can stream OpenAI ``speech.audio.*`` SSE or raw audio bytes with
        first-audio latency after the first window (AR still completes first —
        that stage is inherently sequential).
        """
        import torch

        if not req.lyrics or not str(req.lyrics).strip():
            raise ValueError("MiniMax-Music3 requires non-empty lyrics (input)")
        if not req.prompt or not str(req.prompt).strip():
            raise ValueError(
                "MiniMax-Music3 requires non-empty music description (instructions)"
            )

        duration = self._resolve_duration(req)
        duration = duration_for_frames(frames_for_duration(duration))
        steps = max(1, int(req.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS))
        seed = req.seed
        kind = self._pipe[0]

        logger.info(
            "music3 stream: kind=%s duration=%.2fs steps=%s seed=%s lyrics_chars=%s prompt_chars=%s",
            kind,
            duration,
            steps,
            seed,
            len(req.lyrics),
            len(req.prompt),
        )

        ar_pipe, acoustic_pipe = self._ar_and_acoustic_pipes()
        gen_ar = gen_ac = None
        if seed is not None:
            if kind == "split":
                gen_ar = torch.Generator(device="cuda:1").manual_seed(int(seed))
                gen_ac = torch.Generator(device="cuda:0").manual_seed(int(seed))
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                gen_ar = torch.Generator(device=device).manual_seed(int(seed))
                gen_ac = torch.Generator(device=device).manual_seed(int(seed))

        # AR only → PipelineState with frame_hiddens (no acoustic yet).
        if kind == "split":
            state = ar_pipe(
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=gen_ar,
            )
        else:
            state = self._run_semantic_only(
                ar_pipe,
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=gen_ar,
            )

        yield from self._iter_acoustic_stream_chunks(
            acoustic_pipe, state, steps=steps, generator=gen_ac
        )

    def _ar_and_acoustic_pipes(self):
        kind = self._pipe[0]
        if kind == "single":
            pipe = self._pipe[1]
            return pipe, pipe
        if kind == "split":
            return self._pipe[1], self._pipe[2]
        raise RuntimeError(f"unknown music3 pipe kind {kind!r}")

    def _run_semantic_only(self, pipe, *, prompt, lyrics, audio_duration, generator):
        """Run Tokenizer+AR blocks only on a full ModularPipeline."""
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        blocks = getattr(getattr(pipe, "blocks", None), "sub_blocks", None)
        if not blocks or "semantic_generator" not in blocks:
            raise RuntimeError(
                "music3 stream: single pipeline lacks semantic_generator block"
            )
        sg = blocks["semantic_generator"]
        state = PipelineState()
        inputs = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "generator": generator,
        }
        for expected in sg.inputs:
            name = expected.name
            if name in inputs and inputs[name] is not None:
                state.set(name, inputs[name], expected.kwargs_type)
            elif name not in state.values:
                state.set(name, expected.default, expected.kwargs_type)
        _, state = sg(pipe, state)
        return state

    def _iter_acoustic_stream_chunks(self, components, state, *, steps: int, generator):
        """Drive Diffusers chunk denoise one window at a time; yield PCM."""
        prep, loop = self._resolve_denoise_blocks(components)
        state.set("num_inference_steps", int(steps))
        if generator is not None:
            state.set("generator", generator)

        components, state = prep(components, state)
        block_state = loop.get_block_state(state)
        block_state.latent_chunks = []
        block_state.previous_latent = None
        block_state.previous_condition = None
        block_state.num_inference_steps = int(steps)

        chunk_starts = list(block_state.chunk_starts or [])
        num_chunks = len(chunk_starts)
        if num_chunks == 0:
            raise RuntimeError("music3 stream: no acoustic chunks from frame_hiddens")

        hop = int(getattr(components, "latent_hop_length", 512) or 512)
        native_sr = int(
            getattr(components, "sampling_rate", None) or self._sampling_rate or NATIVE_SAMPLE_RATE
        )
        out_sr = int(self.config.output_sample_rate or native_sr)
        if out_sr <= 0:
            out_sr = native_sr

        logger.info(
            "music3 stream acoustic: chunks=%s steps=%s hop=%s native_sr=%s out_sr=%s",
            num_chunks,
            steps,
            hop,
            native_sr,
            out_sr,
        )

        with loop.progress_bar(total=num_chunks * int(steps)) as progress_bar:
            block_state.progress_bar = progress_bar
            for k in range(num_chunks):
                components, block_state = loop.loop_step(components, block_state, k=k)
                latents = block_state.latent_chunks[-1]
                waveform = components.vocoder(latents.to(components.vocoder.dtype))
                left = 0 if k == 0 else CROP_LEFT_LATENT * hop
                right = 0 if k == num_chunks - 1 else CROP_RIGHT_LATENT * hop
                end = waveform.shape[-1] - right
                if end <= left:
                    raise RuntimeError(
                        f"music3 stream: empty crop window k={k} shape={tuple(waveform.shape)} "
                        f"left={left} right={right}"
                    )
                piece = waveform[..., left:end].float().clamp(-1.0, 1.0)
                if piece.ndim == 3:
                    piece = piece[0]
                pcm, sr, n_samples, channels = pcm16le_bytes(
                    piece, native_sr=native_sr, out_sr=out_sr
                )
                yield Music3StreamChunk(
                    pcm_s16le=pcm,
                    sample_rate=sr,
                    channels=channels,
                    chunk_index=k,
                    num_chunks=num_chunks,
                    num_samples=n_samples,
                )
        block_state.progress_bar = None

    @staticmethod
    def _resolve_denoise_blocks(components):
        """Prefer the loaded pipeline's denoise sub-blocks; else construct Diffusers steps."""
        from diffusers.modular_pipelines.minimax_music3.before_denoise import (
            MiniMaxMusic3PrepareChunksStep,
        )
        from diffusers.modular_pipelines.minimax_music3.denoise import (
            MiniMaxMusic3ChunkDenoiseStep,
        )

        blocks = getattr(getattr(components, "blocks", None), "sub_blocks", None)
        if blocks and "denoise" in blocks:
            denoise_root = blocks["denoise"]
            sub = getattr(denoise_root, "sub_blocks", None)
            if sub and "prepare_chunks" in sub and "denoise" in sub:
                return sub["prepare_chunks"], sub["denoise"]
        # Split rest_pipe is Sequential(denoise=CoreDenoise, decode=Vocoder).
        if blocks and "prepare_chunks" in blocks and "denoise" in blocks:
            return blocks["prepare_chunks"], blocks["denoise"]
        return MiniMaxMusic3PrepareChunksStep(), MiniMaxMusic3ChunkDenoiseStep()

    def shutdown(self) -> None:
        self._pipe = None


def _resample_poly(pcm, src_sr: int, dst_sr: int):
    """Lightweight polyphase resample; (T, C) float32 → (T', C)."""
    return resample_poly_pcm(pcm, src_sr, dst_sr)
