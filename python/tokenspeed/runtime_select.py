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

"""Which serving runtime a checkpoint needs, decided from the checkpoint.

``ts serve <model>`` should be the whole command for every model this engine
can run -- a generative LLM, an embedding model, a reranker, Whisper, and in
time a diffusion pipeline. The operator knows the path; the checkpoint knows
what it is. Making them also pass ``--is-embedding`` or pick a different
subcommand is asking them to restate something already written on disk, and
getting it wrong is not loud: an embedding checkpoint served generatively
builds a vocab-sized lm_head that no weight fills.

A LEAF, for the same reason :mod:`tokenspeed.flat_kvcache_select` is one: the
answer is a couple of fields of a couple of JSON files, and the decision is
wanted at the CLI boundary, before the first ``import tokenspeed.runtime.*``
drags in transformers, tokenspeed_kernel and flashinfer. stdlib only, no
package siblings.

One caveat, measured rather than assumed: ``import tokenspeed`` *already*
costs torch (2.2s), because ``tokenspeed/_logging.py`` imports the noisy
third-party loggers by name in order to filter them. So "before torch" is
true of this module's own imports and not of the import statement that
reaches it -- the same is true of ``flat_kvcache_select``, whose docstring
makes the stronger claim. The test that guards this measures the *delta*
against ``import tokenspeed``, which is the part a sibling import would
regress.

Unparseable or absent metadata resolves to ``generate`` rather than raising:
this runs before the loader, which reports a bad checkpoint far better than
an argument parser can.
"""

from __future__ import annotations

import json
import os

__all__ = ["RUNTIMES", "resolve_serving_runtime", "describe_runtime"]

RUNTIMES = ("generate", "embed", "rerank", "asr", "diffusion")

# Encoder-only embedding architectures. Kept as a literal list rather than
# imported from the model registry, which is exactly the heavy import this
# module exists to avoid. It only has to stay in step with
# ``runtime/models/bert.py``'s EntryClass; a name missing here costs an
# ``--is-embedding`` flag, not a wrong answer, because the engine's own
# ``resolve_pooling_config`` reads 1_Pooling independently.
_ENCODER_ONLY_EMBEDDING = frozenset(
    {
        "BertModel",
        "RobertaModel",
        "XLMRobertaModel",
        "NewModel",
        "GteModel",
        "NomicBertModel",
        "ModernBertModel",
    }
)

_ASR = frozenset({"WhisperForConditionalGeneration"})

_CROSS_ENCODER_SUFFIX = "ForSequenceClassification"


def _architectures(model_path: str) -> list[str]:
    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            declared = json.load(handle).get("architectures") or []
    except (OSError, ValueError):
        return []
    return [str(a) for a in declared]


def resolve_serving_runtime(model_path: str) -> str:
    """One of :data:`RUNTIMES`, from the checkpoint's own metadata.

    Ordered most specific first, because the tests overlap: a reranker is
    also a pooling model, and a diffusers pipeline directory also contains
    a ``config.json`` under each component.
    """
    if not model_path or not os.path.isdir(model_path):
        return "generate"

    # A diffusers ModularPipeline is a *directory of components* described by
    # modular_model_index.json (MiniMax-Music3) and/or model_index.json
    # (MiniMax-H3). Nothing else in this tree ships those files at the root.
    if os.path.isfile(os.path.join(model_path, "modular_model_index.json")) or os.path.isfile(
        os.path.join(model_path, "model_index.json")
    ):
        return "diffusion"
    # Music3 also declares model_type on a root config.json even when indexes
    # are present; keep this as a belt-and-suspenders signal for incomplete trees.
    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            cfg = json.load(handle)
        if str(cfg.get("model_type") or "").lower() == "minimax_music3":
            return "diffusion"
        arch = cfg.get("architectures") or []
        if any("Music3" in str(a) for a in arch):
            return "diffusion"
    except (OSError, ValueError):
        pass

    architectures = _architectures(model_path)

    # Before the pooling test: a cross-encoder pools too, but it scores a
    # pair rather than embedding a text, and the two answer different routes.
    if any(a.endswith(_CROSS_ENCODER_SUFFIX) for a in architectures):
        return "rerank"

    if any(a in _ASR for a in architectures):
        return "asr"

    # sentence-transformers says so directly. This is the only signal that
    # distinguishes a *decoder* trained as an embedder (Qwen3-Embedding) from
    # the same architecture trained to generate.
    if os.path.isfile(os.path.join(model_path, "1_Pooling", "config.json")):
        return "embed"

    if any(a in _ENCODER_ONLY_EMBEDDING for a in architectures):
        return "embed"

    return "generate"


def describe_runtime(runtime: str) -> str:
    """One line naming what this runtime serves, for the startup log."""
    return {
        "generate": "generative LLM (tokens in, tokens out)",
        "embed": "embedding (one pooled vector per request, no decode)",
        "rerank": "cross-encoder reranker (one score per query/document pair)",
        "asr": "speech recognition (encoder-decoder)",
        "diffusion": "image/video/audio diffusion pipeline (MiniMax-H3 / Music3)",
    }.get(runtime, runtime)
