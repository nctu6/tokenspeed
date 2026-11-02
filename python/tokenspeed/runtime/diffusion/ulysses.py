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

"""TokenSpeed DiT Ulysses / ring orchestration for MiniMax-H3.

Design (TokenSpeed-native, not a vLLM-Omni dump):

* **CLI / residency / HTTP job sync** live in TokenSpeed — same pattern as
  Whisper (TokenSpeed owns the serve surface; the library owns the math).
* **Sequence-parallel attention** uses released Diffusers Context Parallel
  (``ContextParallelConfig`` + ``MiniMaxH3Transformer3DModel._cp_plan`` +
  ``enable_parallelism``). MiniMax-H3 already declares a CP plan (56 heads,
  divisible by 2) and the AttnProcessor forwards ``parallel_config`` into
  ``dispatch_attention_fn``.
* **Collectives** go through ``torch.distributed`` DeviceMesh (ring × ulysses)
  that Diffusers builds; TokenSpeed already has matching A2A helpers in
  ``runtime/distributed/comm_ops.py`` (``all_to_all_transpose`` /
  ``all_to_all_head_scatter``) for future TokenSpeed-owned processors.
* **Dispatch degree** is a per-dispatch parameter of the same path: one waiting
  request runs across all ranks with CP on; ``world_size`` waiting requests
  run one per rank with CP off (``DitContextParallel``). Commands and CPU
  payloads travel on a gloo control group, so idle workers never sit inside an
  NCCL collective.

Weight-TP is separate: H3's Diffusers ``_tp_plan`` is empty today, so
``--text-encoder-tp-size`` / DiT weight-TP stay fail-closed. Multi-GPU
*component placement* remains ``residency=device_split``. USP **replicates**
DiT weights across ranks (classic Ulysses), it does not shard Linear layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = [
    "UlyssesWorld",
    "init_ulysses_world",
    "DitContextParallel",
    "enable_dit_context_parallel",
    "broadcast_pyobject",
    "gather_pyobject",
    "leader_signal",
    "worker_loop",
    "is_usp_worker",
    "usp_world_size",
    "maybe_reexec_torchrun",
    "pipeline_state_to_payload",
    "pipeline_state_from_payload",
    "barrier",
]


# MiniMax-H3 FL2VA/Ref2VA DiT attention head count (transformer/config.json).
H3_NUM_ATTENTION_HEADS = 56


@dataclass(frozen=True)
class UlyssesWorld:
    rank: int
    world_size: int
    local_rank: int
    ulysses_degree: int
    ring_degree: int
    # Gloo group for leader -> worker commands and CPU payloads. Workers block on
    # it between requests; NCCL would hit its watchdog timeout while idle.
    control_group: Any

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def usp_world_size(ulysses_degree: int, ring_degree: int = 1) -> int:
    return max(1, int(ulysses_degree)) * max(1, int(ring_degree))


def is_usp_worker() -> bool:
    """True when this process is a non-leader torchrun worker."""
    if (
        usp_world_size(
            int(os.environ.get("TOKENSPEED_H3_ULYSSES", "1")),
            int(os.environ.get("TOKENSPEED_H3_RING", "1")),
        )
        <= 1
    ):
        return False
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() != 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) != 0


def maybe_reexec_torchrun(argv: list[str], world: int) -> None:
    """Re-exec under ``torchrun`` when USP/ring needs N processes and we are not one yet.

    ``tokenspeed serve`` is single-process; Diffusers CP requires an initialized
    process group with ``world_size == ring * ulysses``. Re-exec keeps the same
    module entry so scripts can keep calling ``tokenspeed serve``.
    """
    if world <= 1:
        return
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        return

    torchrun = _find_torchrun()
    cmd = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={world}",
        "-m",
        "tokenspeed.runtime.entrypoints.diffusion_http",
        *argv,
    ]
    logger.info("h3 usp: re-exec under torchrun world=%s: %s", world, " ".join(cmd))
    os.execvp(cmd[0], cmd)


def _find_torchrun() -> str:
    import shutil

    path = shutil.which("torchrun")
    if path:
        return path
    # venv sibling of python
    bindir = os.path.dirname(os.path.realpath(os.sys.executable))
    candidate = os.path.join(bindir, "torchrun")
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(
        "ulysses/ring > 1 needs torchrun on PATH (install PyTorch in the TokenSpeed venv)"
    )


def init_ulysses_world(
    ulysses_degree: int, ring_degree: int = 1
) -> UlyssesWorld | None:
    """Initialize NCCL + return world metadata. ``None`` when degree product is 1."""
    world = usp_world_size(ulysses_degree, ring_degree)
    if world <= 1:
        return None

    import torch
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for DiT Ulysses/ring")

    if not dist.is_initialized():
        # torchrun sets env; default to env://
        # H200/PRO hosts in this fleet often break NVLS multicast bind (CUDA 401),
        # same class of failure TokenSpeed already auto-disables for LLM TP.
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = None
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device_id = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend=backend, device_id=device_id)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != world:
        raise RuntimeError(
            f"USP/ring world mismatch: torch.distributed world_size={world_size} but "
            f"ulysses_degree={ulysses_degree} * ring_degree={ring_degree} = {world}. "
            f"Launch with torchrun --nproc_per_node={world}."
        )
    local_rank = int(
        os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count()))
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Stash for worker detection without importing config.
    os.environ["TOKENSPEED_H3_ULYSSES"] = str(ulysses_degree)
    os.environ["TOKENSPEED_H3_RING"] = str(ring_degree)

    from datetime import timedelta

    # Workers wait on this group for the next request for as long as the
    # server is idle, so its timeout is effectively unbounded.
    control_group = dist.new_group(backend="gloo", timeout=timedelta(days=365))

    logger.info(
        "h3 usp: rank=%s/%s local_rank=%s ulysses=%s ring=%s",
        rank,
        world_size,
        local_rank,
        ulysses_degree,
        ring_degree,
    )
    return UlyssesWorld(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        ulysses_degree=int(ulysses_degree),
        ring_degree=int(ring_degree),
        control_group=control_group,
    )


class DitContextParallel:
    """Switch Diffusers Context Parallel on/off on the DiTs between dispatches.

    The DiT replicas stay where they are; only the sequence-sharding hooks and
    the attention processors' parallel config change. ``enabled=True`` runs one
    request across every rank (Ulysses, lowest latency); ``enabled=False`` lets
    each rank run its own request (data parallel, highest throughput).
    """

    def __init__(self, transformers: list[Any]):
        # enable_parallelism resolved each DiT's mesh into dit._parallel_config.
        self.transformers = transformers
        self.parallel_configs = [dit._parallel_config for dit in transformers]
        self.plans = [dit._cp_plan for dit in transformers]
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        from diffusers.hooks.context_parallel import (
            apply_context_parallel,
            remove_context_parallel,
        )
        from diffusers.models.attention import AttentionModuleMixin

        for dit, config, plan in zip(
            self.transformers, self.parallel_configs, self.plans
        ):
            for sub in dit.modules():
                if isinstance(sub, AttentionModuleMixin):
                    sub.processor._parallel_config = config if enabled else None
            if enabled:
                apply_context_parallel(dit, config.context_parallel_config, plan)
            else:
                remove_context_parallel(dit, plan)
        self.enabled = enabled


def enable_dit_context_parallel(
    module: Any,
    *,
    ulysses_degree: int,
    ring_degree: int,
    attention_backend: str | None,
    ulysses_anything: bool = True,
) -> DitContextParallel:
    """Enable Diffusers Context Parallel on every MiniMax-H3 DiT under ``module``.

    Returns the controller that toggles it per dispatch. Requires an
    initialized process group and a CP degree product > 1.
    """
    world = usp_world_size(ulysses_degree, ring_degree)
    if world <= 1 or module is None:
        raise ValueError(
            "enable_dit_context_parallel needs ulysses*ring > 1 and a pipeline"
        )

    from diffusers.models._modeling_parallel import (
        ContextParallelConfig,
        ParallelConfig,
    )

    transformers = list(_iter_h3_transformers(module))
    if not transformers:
        raise RuntimeError(
            "ulysses/ring > 1 but no MiniMaxH3Transformer3DModel found on the pipeline; "
            "load the DiT partition before enable_dit_context_parallel"
        )

    # Head divisibility: strict Ulysses wants heads % degree == 0. H3 has 56 heads.
    if (
        not ulysses_anything
        and ulysses_degree > 1
        and H3_NUM_ATTENTION_HEADS % int(ulysses_degree) != 0
    ):
        raise ValueError(
            f"MiniMax-H3 has {H3_NUM_ATTENTION_HEADS} heads; ulysses_degree="
            f"{ulysses_degree} does not divide evenly (enable ulysses_anything or pick a divisor)"
        )

    cp = ContextParallelConfig(
        ulysses_degree=int(ulysses_degree),
        ring_degree=int(ring_degree),
        ulysses_anything=bool(
            ulysses_anything and ulysses_degree > 1 and ring_degree == 1
        ),
        ring_anything=bool(ring_degree > 1 and ulysses_degree == 1),
    )
    parallel = ParallelConfig(context_parallel_config=cp)

    # No per-forward barrier: the conditioning broadcast already aligns the
    # ranks before denoise, and in data-parallel dispatch the ranks run
    # independent requests (a barrier there could pair unrelated steps).
    for dit in transformers:
        if attention_backend:
            _set_attention_backend(dit, attention_backend)
        else:
            _ensure_cp_attention_backend(dit)
        dit.enable_parallelism(config=parallel)
        logger.info(
            f"h3 usp: enable_parallelism on {dit.__class__.__name__} "
            f"(ulysses={ulysses_degree} ring={ring_degree} "
            f"anything={cp.ulysses_anything or cp.ring_anything})"
        )
    return DitContextParallel(transformers)


def _iter_h3_transformers(root: Any):
    """Yield DiT modules from a ModularPipeline / ComponentsManager tree."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        cls = obj.__class__.__name__
        if cls == "MiniMaxH3Transformer3DModel":
            yield obj
            continue
        # ModularPipeline exposes loaded components as attributes.
        for name in ("transformer", "transformer_ref"):
            child = getattr(obj, name, None)
            if child is not None:
                stack.append(child)
        # ComponentsManager / dict-like
        components = getattr(obj, "components", None)
        if isinstance(components, dict):
            stack.extend(components.values())
        elif components is not None:
            for name in ("transformer", "transformer_ref"):
                try:
                    stack.append(components.get(name))
                except Exception:
                    pass


def _ensure_cp_attention_backend(dit: Any) -> None:
    """Pick a Diffusers attention backend that supports context parallel."""
    try:
        from diffusers.models.attention_dispatch import (
            _AttentionBackendRegistry,
        )
    except Exception:
        return
    supported = set(_AttentionBackendRegistry._supports_context_parallel)
    # cuDNN SDPA first: on sm120 at H3 shapes ([1, 8k-31k, 28-56, 128] bf16,
    # unmasked) it is 7-9% faster than the SDPA flash kernel and 2x FA4-CuTe
    # (tmp microbench, minimax-h3-analysis.md). Then flash variants, then math.
    preferred = [
        "_native_cudnn",
        "flash",
        "_native_flash",
        "native",
    ]
    for name in preferred:
        if name in supported:
            try:
                dit.set_attention_backend(name)
            except Exception as exc:  # noqa: BLE001 - try the next backend
                logger.info(f"h3 usp: attention backend {name} unavailable: {exc}")
                continue
            logger.info(f"h3 usp: attention backend {name}")
            return
    # Last resort: leave default; enable_parallelism will raise with the list.


def _set_attention_backend(dit: Any, name: str) -> None:
    try:
        dit.set_attention_backend(name)
    except Exception as exc:
        raise RuntimeError(
            f"failed to set diffusion attention backend {name!r}: {exc}"
        ) from exc


def broadcast_pyobject(obj: Any, *, src: int, group: Any) -> Any:
    """Broadcast a picklable Python object from ``src`` to every rank of ``group``."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj
    payload = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src, group=group)
    return payload[0]


def gather_pyobject(obj: Any, *, dst: int, group: Any) -> list[Any] | None:
    """Gather one picklable object per rank on ``dst`` (others get ``None``)."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [obj]
    out = [None] * dist.get_world_size() if dist.get_rank() == dst else None
    dist.gather_object(obj, out, dst=dst, group=group)
    return out


def barrier(group: Any) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier(group=group)


def pipeline_state_to_payload(state: Any) -> dict[str, Any]:
    """Pickle-friendly snapshot of a Diffusers ``PipelineState`` (tensors on CPU)."""
    import torch

    values: dict[str, Any] = {}
    raw = getattr(state, "values", None) or {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            values[key] = value.detach().to("cpu").contiguous()
        else:
            values[key] = value
    return {
        "values": values,
        "kwargs_mapping": dict(getattr(state, "kwargs_mapping", {}) or {}),
    }


def _should_move_tensor_to_device(value: Any) -> bool:
    """Float/complex activations go to CUDA; integer layout tags stay on CPU.

    MiniMax-H3 ``PrepareLayout`` builds ``token_tags = torch.empty(..., dtype=long)``
    on CPU then assigns ``text_token_tags`` into it — tags must remain CPU.
    ``prompt_embeds`` (bf16/fp) must be on the local CUDA device for denoise.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return False
    return value.is_floating_point() or value.is_complex()


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        if _should_move_tensor_to_device(value):
            return value.to(device, non_blocking=False).contiguous()
        return value
    if isinstance(value, (list, tuple)):
        moved = [_move_value_to_device(v, device) for v in value]
        return type(value)(moved)
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    return value


def pipeline_state_from_payload(payload: dict[str, Any], *, device: str) -> Any:
    """Rebuild a ``PipelineState`` on ``device`` from a CPU payload.

    Floating-point tensors (e.g. ``prompt_embeds``) move onto ``device``.
    Integer layout tags (e.g. ``text_token_tags``) stay on CPU so Diffusers
    ``PrepareLayout`` can index them into CPU ``token_tags`` before ``.to(device)``.
    """
    import torch
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    torch_device = torch.device(device)
    values: dict[str, Any] = {}
    for key, value in (payload.get("values") or {}).items():
        values[key] = _move_value_to_device(value, torch_device)
    return PipelineState(
        values=values,
        kwargs_mapping=dict(payload.get("kwargs_mapping") or {}),
    )


CMD_GENERATE = "generate"
CMD_SHUTDOWN = "shutdown"


def leader_signal(cmd: str, payload: Any | None, *, group: Any) -> None:
    """Rank-0 helper: publish a worker command and its payload."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    broadcast_pyobject((cmd, payload), src=0, group=group)


def worker_loop(dispatch_fn, *, group: Any) -> None:
    """Non-leader ranks: run each dispatch rank 0 publishes until shutdown."""
    import torch.distributed as dist

    assert dist.is_initialized()
    while True:
        cmd, payload = broadcast_pyobject(None, src=0, group=group)
        if cmd == CMD_SHUTDOWN:
            logger.info("h3 usp worker: shutdown")
            break
        if cmd == CMD_GENERATE:
            dispatch_fn(payload)
            continue
        logger.warning(f"h3 usp worker: unknown cmd {cmd!r}")
