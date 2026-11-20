# Copyright (c) 2026 LightSeek Foundation
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Resolve pooling settings declared by embedding checkpoints."""

import json
import os
from dataclasses import dataclass

from tokenspeed.runtime.layers.pooler import PoolingType


@dataclass(frozen=True)
class PoolingConfig:
    """How an embedding checkpoint reduces hidden states to one vector."""

    pooling_type: PoolingType
    normalize: bool


# sentence-transformers spells its pooling choice as mutually exclusive
# booleans. Only the modes implemented here are listed; a checkpoint selecting
# any other is rejected, because a wrong reduction still returns a well-formed
# vector that nothing downstream can catch.
_ST_POOLING_MODES = {
    "pooling_mode_lasttoken": PoolingType.LAST,
    "pooling_mode_cls_token": PoolingType.CLS,
    "pooling_mode_mean_tokens": PoolingType.MEAN,
}


def read_sentence_transformers_pooling(model_path: str) -> PoolingConfig | None:
    """Read ``1_Pooling/config.json`` and ``modules.json`` from a checkpoint.

    Returns None when the checkpoint ships no sentence-transformers pooling
    config, which is how a generative checkpoint looks.
    """
    pooling_file = os.path.join(model_path, "1_Pooling", "config.json")
    if not os.path.isfile(pooling_file):
        return None
    with open(pooling_file) as handle:
        raw = json.load(handle)

    selected = [
        pooling_type for key, pooling_type in _ST_POOLING_MODES.items() if raw.get(key)
    ]
    enabled_unsupported = [
        key
        for key, value in raw.items()
        if key.startswith("pooling_mode_") and value and key not in _ST_POOLING_MODES
    ]
    if enabled_unsupported:
        raise ValueError(
            f"{pooling_file} selects unsupported pooling mode(s) "
            f"{sorted(enabled_unsupported)}; supported: "
            f"{sorted(_ST_POOLING_MODES)}. Pass --pooling-type to override."
        )
    if len(selected) != 1:
        raise ValueError(
            f"{pooling_file} selects {len(selected)} pooling modes; expected "
            "exactly one. Pass --pooling-type to override."
        )

    # Normalization is its own sentence-transformers module, not a field of
    # the pooling config.
    normalize = False
    modules_file = os.path.join(model_path, "modules.json")
    if os.path.isfile(modules_file):
        with open(modules_file) as handle:
            modules = json.load(handle)
        normalize = any(
            str(module.get("type", "")).endswith("Normalize") for module in modules
        )

    return PoolingConfig(pooling_type=selected[0], normalize=normalize)


def resolve_pooling_config(
    model_path: str,
    *,
    is_embedding: bool | None = None,
    pooling_type: str | None = None,
    pooling_normalize: bool | None = None,
) -> PoolingConfig | None:
    """Decide whether this server pools, and how.

    Returns None for a generative server. Raises when embedding mode is
    requested on a checkpoint that declares no pooling and no override is
    given: a server that guessed the reduction would answer every request
    with a plausible wrong vector.
    """
    declared = read_sentence_transformers_pooling(model_path)
    if is_embedding is False:
        return None
    if is_embedding is None and declared is None:
        return None

    if declared is None and pooling_type is None:
        raise ValueError(
            f"--is-embedding was requested but {model_path} ships no "
            "1_Pooling/config.json to say how to pool. Pass --pooling-type "
            "(last|cls|mean), and --pooling-normalize if the checkpoint "
            "expects L2-normalized output."
        )

    resolved_type = (
        PoolingType(pooling_type) if pooling_type is not None else declared.pooling_type
    )
    if pooling_normalize is not None:
        normalize = pooling_normalize
    else:
        normalize = declared.normalize if declared is not None else False
    return PoolingConfig(pooling_type=resolved_type, normalize=normalize)
