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
"""MiniMax-H3-Image generation through released HuggingFace Diffusers.

``MiniMaxH3ImageModularPipeline`` (diffusers >= 0.43) supplies the blocks:
``before_encode``, ``text_encoder``, ``vae_encoder``, ``denoise``, ``decode``.
TokenSpeed owns residency and the HTTP surface. One ``image`` input switches
the same blocks between text-to-image and reference-image editing, so there is
one load and one call path for both.

Residency (see :class:`H3ImageConfig`): ``single`` runs everything on
``cuda:0``; ``device_split`` runs the conditioner blocks (``before_encode`` +
``text_encoder``) on ``cuda:1`` and the rest on ``cuda:0``, handing the
conditioning over as pipeline state.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

from tokenspeed.runtime.diffusion.config import resolve_diffusion_checkpoint
from tokenspeed.runtime.diffusion.image_config import IMAGE_FORMATS, H3ImageConfig
from tokenspeed.runtime.diffusion.ulysses import (
    pipeline_state_from_payload,
    pipeline_state_to_payload,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["H3ImagePipeline", "H3ImageRequest", "H3ImageResult"]

# Blocks that run on the conditioner GPU in the device_split layout.
_CONDITIONER_BLOCKS = ("before_encode", "text_encoder")
# Inputs only the conditioner blocks consume; the rest pipeline gets the state.
_CONDITIONER_ONLY_KEYS = frozenset({"prompt", "image"})


@dataclass
class H3ImageRequest:
    prompt: str
    width: int
    height: int
    # None draws a fresh seed per request.
    seed: int | None
    output_format: str
    # Encoded reference images (PNG/JPEG/...), in `<Picture N>` order. Empty
    # means text-to-image.
    reference_images: list[bytes]
    # Smallest generation area in pixels; None keeps the checkpoint default.
    min_generation_area: int | None


@dataclass
class H3ImageResult:
    image_bytes: bytes
    output_format: str
    width: int
    height: int
    seed: int


class H3ImagePipeline:
    """Load once, generate many. ``generate`` is single-flight (internal lock)."""

    def __init__(self, config: H3ImageConfig):
        try:
            import diffusers  # noqa: F401
            from diffusers import MiniMaxH3ImageModularPipeline  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "[ts serve] MiniMax-H3-Image needs diffusers>=0.43.0 "
                "(MiniMaxH3ImageModularPipeline): re-run ./scripts/install.sh\n"
                f"           import error: {exc}"
            ) from exc
        self.config = config
        self.model_path = resolve_diffusion_checkpoint(config.model_path)
        self.residency = config.resolve_residency()
        self._lock = threading.Lock()
        # Set by _load: one pipeline (single) or conditioner + rest (device_split).
        self._conditioner = None
        self._rest = None
        self._load()

    @property
    def canvas_multiple(self) -> int:
        return self._rest.canvas_multiple

    def _dtype(self):
        import torch

        table = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        name = {"bf16": "bfloat16", "fp16": "float16"}.get(
            self.config.dtype.lower(), self.config.dtype.lower()
        )
        if name not in table:
            raise ValueError(f"dtype={self.config.dtype!r} must be one of {sorted(table)}")
        return table[name]

    def _load(self) -> None:
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax-H3-Image serve needs CUDA GPUs")
        if torch.cuda.device_count() < self.config.num_gpus:
            raise RuntimeError(
                f"num_gpus={self.config.num_gpus} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        dtype = self._dtype()
        logger.info(
            f"h3image load: path={self.model_path} gpus={self.config.num_gpus} "
            f"dtype={dtype} residency={self.residency} "
            f"cpu_offload={self.config.enable_cpu_offload}"
        )

        def manager_for(device: str):
            if not self.config.enable_cpu_offload:
                return None
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(
                device=device,
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
            return manager

        if self.residency == "single":
            pipe = ModularPipeline.from_pretrained(
                self.model_path, components_manager=manager_for("cuda:0")
            )
            pipe.load_components(dtype=dtype)
            if not self.config.enable_cpu_offload:
                pipe.to("cuda:0")
            self._rest = pipe
            return

        blocks = ModularPipeline.from_pretrained(self.model_path).blocks
        sub_blocks = blocks.sub_blocks
        cond_dict = {name: sub_blocks.pop(name) for name in _CONDITIONER_BLOCKS}
        conditioner = SequentialPipelineBlocks.from_blocks_dict(
            cond_dict
        ).init_pipeline(self.model_path, components_manager=manager_for("cuda:1"))
        conditioner.load_components(dtype=dtype)
        rest = blocks.init_pipeline(
            self.model_path, components_manager=manager_for("cuda:0")
        )
        rest.load_components(dtype=dtype)
        if not self.config.enable_cpu_offload:
            conditioner.to("cuda:1")
            rest.to("cuda:0")
        self._conditioner = conditioner
        self._rest = rest
        logger.info("h3image load: device_split (conditioner@cuda:1, rest@cuda:0)")

    def generate(self, req: H3ImageRequest) -> H3ImageResult:
        import secrets

        import torch
        from PIL import Image

        if req.output_format not in IMAGE_FORMATS:
            raise ValueError(
                f"output_format={req.output_format!r} must be one of {sorted(IMAGE_FORMATS)}"
            )
        seed = req.seed if req.seed is not None else secrets.randbits(31)
        kwargs: dict = {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "generator": torch.Generator().manual_seed(seed),
            "output": "images",
        }
        if req.min_generation_area is not None:
            kwargs["min_generation_area"] = req.min_generation_area
        if req.reference_images:
            kwargs["image"] = [
                Image.open(io.BytesIO(data)).convert("RGB")
                for data in req.reference_images
            ]
        with self._lock, torch.no_grad():
            if self._conditioner is None:
                images = self._rest(**kwargs)
            else:
                cond_kwargs = {
                    k: v for k, v in kwargs.items() if k in _CONDITIONER_ONLY_KEYS
                }
                rest_kwargs = {
                    k: v for k, v in kwargs.items() if k not in _CONDITIONER_ONLY_KEYS
                }
                # Conditioning is produced on cuda:1; denoise reads it on cuda:0.
                # Floats (prompt_embeds) move, integer layout tags stay on CPU.
                state = pipeline_state_from_payload(
                    pipeline_state_to_payload(self._conditioner(**cond_kwargs)),
                    device="cuda:0",
                )
                images = self._rest(state=state, **rest_kwargs)
        image = images[0]
        buffer = io.BytesIO()
        image.save(buffer, format=IMAGE_FORMATS[req.output_format])
        return H3ImageResult(
            image_bytes=buffer.getvalue(),
            output_format=req.output_format,
            width=image.width,
            height=image.height,
            seed=seed,
        )

    def shutdown(self) -> None:
        self._conditioner = None
        self._rest = None
