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

"""MiniMax-Music3 generation: TokenSpeed engine over Diffusers modules.

The Global / Local LM, flow-matching transformer, and Flow-VAE vocoder are the
released ``diffusers`` / ``transformers`` modules of the
``MiniMaxMusic3ModularPipeline``; TokenSpeed owns how they run.

Layouts (``Music3Config.resolve_residency``):

* ``device_split`` (``num_gpus>=2``): the TokenSpeed serving engine
  (``music3_engine.py``). The batched AR stage runs on ``cuda:0`` and the
  windowed acoustic stage on ``cuda:1``, both resident; songs are batched in
  both stages and acoustic windows stream while AR is still running.
* ``single`` (``num_gpus=1``): the stock Diffusers ``ModularPipeline`` on
  ``cuda:0``, one song at a time, with optional auto CPU offload for small
  GPUs (~23 GB bf16 peak per Diffusers docs).

``generate_stream`` yields progressive PCM windows on both layouts.
"""

from __future__ import annotations

import copy
import io
import threading
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
from tokenspeed.runtime.diffusion.music3_stream import (
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    Music3StreamChunk,
    pcm16le_bytes,
    resample_poly_pcm,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = [
    "Music3Pipeline",
    "Music3GenerateRequest",
    "Music3GenerateResult",
    "Music3StreamChunk",
]


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


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }[name.lower()]


def _fold_weight_norm(module) -> int:
    """Fold ``weight_norm`` reparametrizations into plain weights; return the count.

    ``weight = g * v / ||v||`` is what the hook recomputes before every
    forward; folding computes it once (same value), skips the per-call
    recompute, and makes the module deep-copyable for a second replica.
    """
    import torch

    folded = 0
    for sub in module.modules():
        if hasattr(sub, "weight_g") and hasattr(sub, "weight_v"):
            torch.nn.utils.remove_weight_norm(sub)
            folded += 1
    return folded


_LM_COMPONENTS = ("language_model", "rvq_depth_decoder")
_ACOUSTIC_COMPONENTS = ("condition_encoder", "transformer", "vocoder")


class Music3Pipeline:
    """Load once, generate many; thread-safe ``generate`` / ``generate_stream``."""

    def __init__(self, config: Music3Config):
        _require_diffusers()
        self.config = config
        self.model_path = resolve_music3_checkpoint(config.model_path)
        self._pipe = None  # single: Diffusers ModularPipeline
        self._engine = None  # device_split: Music3Engine
        # The single layout runs one Diffusers call at a time.
        self._single_lock = threading.Lock()
        self._sampling_rate = NATIVE_SAMPLE_RATE
        self._load()

    def _load_kwargs(self) -> dict:
        """``load_components`` kwargs: per-component dtype + tokenizer regex fix.

        ``fix_mistral_regex`` is tokenizer-only (must not reach Qwen3ForCausalLM).
        """
        lm = _torch_dtype(self.config.dtype)
        acoustic = _torch_dtype(self.config.acoustic_dtype)
        dtype = {name: lm for name in _LM_COMPONENTS}
        dtype.update({name: acoustic for name in _ACOUSTIC_COMPONENTS})
        dtype["default"] = lm
        return {"dtype": dtype, "fix_mistral_regex": {"tokenizer": True}}

    def _load(self) -> None:
        import torch

        residency = self.config.resolve_residency()
        logger.info(
            f"music3 load: path={self.model_path} gpus={self.config.num_gpus} "
            f"lm_dtype={self.config.dtype} acoustic_dtype={self.config.acoustic_dtype} "
            f"residency={residency} offload={self.config.enable_cpu_offload}"
        )
        if self.config.acoustic_dtype == "float32":
            # fp32 acoustic with TF32 matmuls, as the vLLM-Omni reference runs it.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if residency == "device_split":
            self._load_engine()
        else:
            self._load_single()

    def _load_single(self) -> None:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

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
        pipe.load_components(**self._load_kwargs())
        if not self.config.enable_cpu_offload and device.startswith("cuda"):
            pipe.to(device)
        self._pipe = pipe
        self._sampling_rate = int(
            getattr(pipe, "sampling_rate", NATIVE_SAMPLE_RATE) or NATIVE_SAMPLE_RATE
        )
        logger.info(f"music3 load: single Diffusers pipeline on {device}")

    def _load_engine(self) -> None:
        import torch
        from diffusers import ModularPipeline

        from tokenspeed.runtime.diffusion.music3_acoustic import Music3Acoustic
        from tokenspeed.runtime.diffusion.music3_ar import Music3AREngine
        from tokenspeed.runtime.diffusion.music3_engine import Music3Engine

        if self.config.enable_cpu_offload:
            raise ValueError(
                "residency=device_split keeps the Music3 engine resident on its GPUs; "
                "pass --no-cpu-offload (CPU offload would swap weights every frame)"
            )
        if torch.cuda.device_count() < 2:
            raise RuntimeError("residency=device_split needs 2 visible CUDA devices")
        ar_device, acoustic_device = "cuda:0", "cuda:1"
        pipe = ModularPipeline.from_pretrained(self.model_path)
        pipe.load_components(**self._load_kwargs())
        for name in _LM_COMPONENTS:
            getattr(pipe, name).to(ar_device).eval()
        _fold_weight_norm(pipe.vocoder)
        for name in _ACOUSTIC_COMPONENTS:
            getattr(pipe, name).to(acoustic_device).eval()
        torch.cuda.set_device(ar_device)
        ar = Music3AREngine(
            pipe.language_model,
            pipe.rvq_depth_decoder,
            device=ar_device,
            kv_cache_gib=self.config.ar_kv_cache_gib,
            max_pairs=self.config.max_batch_songs,
            use_cuda_graph=not self.config.enforce_eager,
        )
        # Acoustic workers: the dedicated device first; the AR device runs a
        # second replica as overflow capacity when many songs are in flight.
        workers = [
            Music3Acoustic(
                pipe.condition_encoder,
                pipe.transformer,
                pipe.vocoder,
                pipe.scheduler,
                device=acoustic_device,
                latent_hop=pipe.latent_hop_length,
                low_priority=False,
            )
        ]
        if self.config.acoustic_on_ar_device:
            workers.append(
                Music3Acoustic(
                    copy.deepcopy(pipe.condition_encoder).to(ar_device),
                    copy.deepcopy(pipe.transformer).to(ar_device),
                    copy.deepcopy(pipe.vocoder).to(ar_device),
                    pipe.scheduler,
                    device=ar_device,
                    latent_hop=pipe.latent_hop_length,
                    # Shares the GPU with the AR stage, which must go first.
                    low_priority=True,
                )
            )
        self._sampling_rate = int(pipe.sampling_rate)
        logger.info(
            f"music3 load: acoustic workers on {[str(w.device) for w in workers]}"
        )
        self._engine = Music3Engine(
            tokenizer=pipe.tokenizer,
            ar=ar,
            acoustic=workers,
            max_window_batch=self.config.max_window_batch,
        )
        logger.info(
            f"music3 load: engine AR@{ar_device} acoustic@{acoustic_device} "
            f"mem={torch.cuda.memory_allocated(ar_device) / 2**30:.1f}/"
            f"{torch.cuda.memory_allocated(acoustic_device) / 2**30:.1f} GiB"
        )

    # ------------------------------------------------------------- requests
    def _validated(self, req: Music3GenerateRequest) -> tuple[float, int]:
        if not req.lyrics or not str(req.lyrics).strip():
            raise ValueError("MiniMax-Music3 requires non-empty lyrics (input)")
        if not req.prompt or not str(req.prompt).strip():
            raise ValueError(
                "MiniMax-Music3 requires non-empty music description (instructions)"
            )
        if req.max_new_tokens is not None:
            duration = duration_for_frames(int(req.max_new_tokens))
        else:
            duration = float(req.audio_duration)
        # Clamp via the frame helper so we never exceed the checkpoint ceiling.
        duration = duration_for_frames(frames_for_duration(duration))
        steps = max(1, int(req.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS))
        return duration, steps

    def _out_rate(self) -> int:
        out_sr = int(self.config.output_sample_rate or self._sampling_rate)
        return out_sr if out_sr > 0 else self._sampling_rate

    def _submit(self, req: Music3GenerateRequest):
        duration, steps = self._validated(req)
        logger.info(
            f"music3 submit: duration={duration:.2f}s steps={steps} seed={req.seed} "
            f"lyrics_chars={len(req.lyrics)} prompt_chars={len(req.prompt)}"
        )
        return self._engine.submit(
            prompt=req.prompt,
            lyrics=req.lyrics,
            max_frames=frames_for_duration(duration),
            seed=req.seed,
            num_inference_steps=steps,
        )

    def generate(self, req: Music3GenerateRequest) -> Music3GenerateResult:
        if self._engine is not None:
            waveform = self._submit(req).waveform()
        else:
            waveform = self._generate_single(req)
        wav_bytes, sample_rate, num_samples, channels = self._encode_wav(
            waveform, native_sr=self._sampling_rate, out_sr=self._out_rate()
        )
        return Music3GenerateResult(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            num_samples=num_samples,
            channels=channels,
            duration_s=float(num_samples) / float(sample_rate),
            seed=req.seed,
        )

    def _generate_single(self, req: Music3GenerateRequest):
        import torch

        duration, steps = self._validated(req)
        with self._single_lock:
            generator = None
            if req.seed is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(req.seed))
            audio = self._pipe(
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=generator,
                num_inference_steps=steps,
                output="audios",
            )[0]
        tensor = (
            audio.detach().float().cpu()
            if hasattr(audio, "detach")
            else torch.as_tensor(audio).float()
        )
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise RuntimeError(f"unexpected music3 audio shape: {tuple(tensor.shape)}")
        return tensor.numpy()

    @staticmethod
    def _encode_wav(
        audio, *, native_sr: int, out_sr: int
    ) -> tuple[bytes, int, int, int]:
        """``audio`` is (channels, samples) float in roughly [-1, 1]."""
        import numpy as np

        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit(
                "[ts serve] music3 needs soundfile (pip install soundfile); "
                "re-run ./scripts/install.sh"
            ) from exc

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise RuntimeError(f"wav encode expected (C, T), got {audio.shape}")
        pcm = np.clip(audio.T, -1.0, 1.0)  # soundfile wants (samples, channels)
        sample_rate = native_sr
        if out_sr != native_sr:
            pcm = resample_poly_pcm(pcm, native_sr, out_sr)
            sample_rate = out_sr
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), sample_rate, int(pcm.shape[0]), int(pcm.shape[1])

    # ------------------------------------------------------------ streaming
    def generate_stream(self, req: Music3GenerateRequest):
        """Yield progressive PCM windows (``Music3StreamChunk``)."""
        if self._engine is not None:
            job = self._submit(req)
            for index, (piece, is_last) in enumerate(job.iter_pieces()):
                yield self._stream_chunk(piece, index, is_last)
            return
        yield from self._generate_stream_single(req)

    def _stream_chunk(self, piece, index: int, is_last: bool) -> Music3StreamChunk:
        pcm, sr, n_samples, channels = pcm16le_bytes(
            piece, native_sr=self._sampling_rate, out_sr=self._out_rate()
        )
        return Music3StreamChunk(
            pcm_s16le=pcm,
            sample_rate=sr,
            channels=channels,
            chunk_index=index,
            # The window count is only known at the end; until then it is a
            # lower bound (one more window follows).
            num_chunks=index + 1 if is_last else index + 2,
            num_samples=n_samples,
        )

    def _generate_stream_single(self, req: Music3GenerateRequest):
        """Single layout: AR once, then Diffusers' acoustic windows one at a time."""
        import torch

        duration, steps = self._validated(req)
        with self._single_lock:
            pipe = self._pipe
            generator = None
            if req.seed is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(req.seed))
            state = self._run_semantic_only(
                pipe,
                prompt=req.prompt,
                lyrics=req.lyrics,
                audio_duration=duration,
                generator=generator,
            )
            yield from self._iter_acoustic_stream_chunks(
                pipe, state, steps=steps, generator=generator
            )

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
        from diffusers.modular_pipelines.minimax_music3.before_denoise import (
            MiniMaxMusic3PrepareChunksStep,
        )
        from diffusers.modular_pipelines.minimax_music3.denoise import (
            MiniMaxMusic3ChunkDenoiseStep,
        )

        blocks = getattr(getattr(components, "blocks", None), "sub_blocks", None)
        prep, loop = MiniMaxMusic3PrepareChunksStep(), MiniMaxMusic3ChunkDenoiseStep()
        if blocks and "denoise" in blocks:
            sub = getattr(blocks["denoise"], "sub_blocks", None)
            if sub and "prepare_chunks" in sub and "denoise" in sub:
                prep, loop = sub["prepare_chunks"], sub["denoise"]
        state.set("num_inference_steps", int(steps))
        if generator is not None:
            state.set("generator", generator)

        components, state = prep(components, state)
        block_state = loop.get_block_state(state)
        block_state.latent_chunks = []
        block_state.previous_latent = None
        block_state.previous_condition = None
        block_state.num_inference_steps = int(steps)
        num_chunks = len(block_state.chunk_starts or [])
        if num_chunks == 0:
            raise RuntimeError("music3 stream: no acoustic chunks from frame_hiddens")
        hop = int(getattr(components, "latent_hop_length", 512) or 512)

        with loop.progress_bar(total=num_chunks * int(steps)) as progress_bar:
            block_state.progress_bar = progress_bar
            for k in range(num_chunks):
                components, block_state = loop.loop_step(components, block_state, k=k)
                latents = block_state.latent_chunks[-1]
                waveform = components.vocoder(latents.to(components.vocoder.dtype))
                left = 0 if k == 0 else CROP_LEFT_LATENT * hop
                right = 0 if k == num_chunks - 1 else CROP_RIGHT_LATENT * hop
                piece = (
                    waveform[..., left : waveform.shape[-1] - right]
                    .float()
                    .clamp(-1.0, 1.0)
                )
                if piece.ndim == 3:
                    piece = piece[0]
                yield self._stream_chunk(piece, k, k == num_chunks - 1)
        block_state.progress_bar = None

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
        self._engine = None
        self._pipe = None
