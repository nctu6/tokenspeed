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

"""Detect which TokenSpeed diffusion family a ModularPipeline checkpoint is.

MiniMax-H3 ships ``model_index.json`` (and often ``modular_model_index.json``).
MiniMax-Music3 ships only ``modular_model_index.json`` plus a root
``config.json`` with ``model_type=minimax_music3``. Diffusers loads
``modular_model_index.json`` first, then falls back to ``model_index.json``.
"""

from __future__ import annotations

import json
import os

__all__ = [
    "DIFFUSION_FAMILIES",
    "detect_diffusion_family",
    "is_modular_checkpoint",
    "resolve_modular_root",
]

DIFFUSION_FAMILIES = ("h3", "music3")

_INDEX_NAMES = ("modular_model_index.json", "model_index.json")


def _read_json(path: str) -> dict:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _class_blob(meta: dict) -> str:
    return " ".join(
        str(meta.get(key) or "")
        for key in ("_class_name", "_blocks_class_name", "model_type")
    )


def is_modular_checkpoint(model_path: str) -> bool:
    """True when the directory looks like a Diffusers ModularPipeline root."""
    if not model_path or not os.path.isdir(model_path):
        return False
    for name in _INDEX_NAMES:
        meta = _read_json(os.path.join(model_path, name))
        if not meta:
            continue
        blob = _class_blob(meta)
        if "Modular" in blob or meta.get("_blocks_class_name"):
            return True
    # Music3 root config alone (no index yet) -- rare, but cheap.
    cfg = _read_json(os.path.join(model_path, "config.json"))
    if str(cfg.get("model_type") or "").lower() == "minimax_music3":
        return True
    arch = cfg.get("architectures") or []
    if any("Music3" in str(a) for a in arch):
        return True
    return False


def detect_diffusion_family(model_path: str) -> str:
    """Return ``h3`` or ``music3`` for a modular checkpoint.

    Fail closed on unknown ModularPipeline families so we do not mis-route a
    future video/audio model onto the H3 ``/v1/videos`` surface.
    """
    path = os.path.abspath(os.path.expanduser(model_path))
    for name in _INDEX_NAMES:
        meta = _read_json(os.path.join(path, name))
        if not meta:
            continue
        blob = _class_blob(meta)
        lower = blob.lower()
        if "music3" in lower or "minimaxmusic3" in lower.replace(" ", ""):
            return "music3"
        if "h3" in lower or "minimaxh3" in lower.replace(" ", ""):
            return "h3"

    cfg = _read_json(os.path.join(path, "config.json"))
    model_type = str(cfg.get("model_type") or "").lower()
    if model_type == "minimax_music3":
        return "music3"
    if model_type in {"minimax_h3", "minimax-h3"}:
        return "h3"
    arch = " ".join(str(a) for a in (cfg.get("architectures") or []))
    if "Music3" in arch:
        return "music3"
    if "H3" in arch and "MiniMax" in arch:
        return "h3"

    raise ValueError(
        f"unsupported diffusion checkpoint at {path}: expected MiniMax-H3 or "
        "MiniMax-Music3 (modular_model_index.json / model_index.json)"
    )


def resolve_modular_root(model_path: str) -> str:
    """Return the directory ``ModularPipeline.from_pretrained`` should open.

    Accepts the HF repo root. For H3, also remaps a ``FL2VA`` / ``Ref2VA``
    partition path to the parent modular root when present.
    """
    path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"diffusion checkpoint not found: {path}")

    if is_modular_checkpoint(path):
        return path

    base = os.path.basename(path.rstrip(os.sep))
    parent = os.path.dirname(path)
    if base in {"FL2VA", "Ref2VA"} and is_modular_checkpoint(parent):
        return parent

    for name in _INDEX_NAMES:
        if os.path.isfile(os.path.join(path, name)):
            return path

    raise FileNotFoundError(
        f"{path} has no modular_model_index.json / model_index.json "
        "(need MiniMax-H3 or MiniMax-Music3 repo root)"
    )
