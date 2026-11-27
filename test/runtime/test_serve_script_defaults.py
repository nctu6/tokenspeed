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
"""``scripts/qwen3.5-122b-a10b.sh`` and ``scripts/qwen3.6-35b-a3b.sh``: which
serve flags the arch defaults and the MTP default select.

CPU-only. The script runs for real; only the ``tokenspeed`` executable in a
throwaway venv is replaced by one that prints its arguments.
"""

import json
import os
import pathlib
import stat
import struct
import subprocess

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
SCRIPT = SCRIPTS / "qwen3.5-122b-a10b.sh"
SCRIPT_QWEN36 = SCRIPTS / "qwen3.6-35b-a3b.sh"


def _serve_args(
    tmp_path: pathlib.Path, script: pathlib.Path = SCRIPT, **env: str
) -> list[str]:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
    exe = venv / "bin" / "tokenspeed"
    exe.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    run_env = {
        "PATH": os.environ["PATH"],
        "VENV_DIR": str(venv),
        "MODEL_PATH": str(tmp_path / "no-such-checkpoint"),
        "HOST_PORT": "1",
        "CUDA_DEVICES": "0,1",
        **env,
    }
    out = subprocess.run(
        ["bash", str(script)],
        env=run_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"TOKENSPEED_CUDA_ARCH": "sm120"}, True),
        ({"TOKENSPEED_CUDA_ARCH": "sm120", "ENABLE_FP8_ATTN_PROJ": "0"}, False),
        ({"TOKENSPEED_CUDA_ARCH": "sm120", "QUANTIZATION": "nvfp4"}, False),
        ({"TOKENSPEED_CUDA_ARCH": "sm90"}, False),
        ({"TOKENSPEED_CUDA_ARCH": "sm90", "ENABLE_FP8_ATTN_PROJ": "1"}, True),
    ],
)
def test_fp8_attn_proj_follows_arch_and_quantization(tmp_path, env, expected):
    args = _serve_args(tmp_path, **env)
    assert ("--enable-fp8-attn-proj" in args) is expected


def _checkpoint(
    root: pathlib.Path,
    *,
    config_layers: int | None,
    weights: str | None,
    nested_config: bool = False,
) -> pathlib.Path:
    """A checkpoint directory holding only what the MTP check reads.

    ``weights``: ``"index"`` writes ``model.safetensors.index.json`` naming an
    ``mtp.`` tensor, ``"header"`` a lone safetensors file whose header names
    one, ``"none"`` an index without any, ``None`` no weight files at all.
    """
    root.mkdir()
    config: dict = {}
    if config_layers is not None:
        config = {"mtp_num_hidden_layers": config_layers}
        if nested_config:
            config = {"text_config": config}
    (root / "config.json").write_text(json.dumps(config))
    names = ["model.layers.0.mlp.weight"]
    if weights in ("index", "header"):
        names.append("mtp.fc.weight")
    if weights in ("index", "none"):
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: "model-00001.safetensors" for n in names}})
        )
    elif weights == "header":
        header = json.dumps(
            {n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for n in names}
        ).encode()
        (root / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header
        )
    return root


MTP_FLAGS = ["--speculative-algorithm", "MTP"]
SERVE_SCRIPTS = [
    pytest.param(SCRIPT, {"CUDA_DEVICES": "6,7"}, id="qwen3.5-122b"),
    pytest.param(SCRIPT_QWEN36, {"CUDA_DEVICES": "6"}, id="qwen3.6-35b"),
]


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
@pytest.mark.parametrize(
    "layout",
    [
        {"config_layers": 1, "weights": "index"},
        {"config_layers": 1, "weights": "header"},
        {"config_layers": 1, "weights": "index", "nested_config": True},
    ],
)
def test_mtp_default_on_when_checkpoint_ships_mtp_layer(
    tmp_path, script, devices, layout
):
    model = _checkpoint(tmp_path / "model", **layout)
    args = _serve_args(
        tmp_path, script, MODEL_PATH=str(model), MAX_NUM_SEQS="64", **devices
    )
    at = args.index("--speculative-algorithm")
    assert args[at : at + 2] == MTP_FLAGS
    assert args[args.index("--speculative-num-steps") + 1] == "2"
    assert args[args.index("--speculative-draft-model-path") + 1] == str(model)


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
@pytest.mark.parametrize(
    "layout",
    [
        {"config_layers": None, "weights": "index"},
        {"config_layers": 0, "weights": "index"},
        {"config_layers": 1, "weights": "none"},
        {"config_layers": 1, "weights": None},
    ],
)
def test_mtp_default_off_without_mtp_layer_or_weights(
    tmp_path, script, devices, layout
):
    model = _checkpoint(tmp_path / "model", **layout)
    args = _serve_args(tmp_path, script, MODEL_PATH=str(model), **devices)
    assert not any(a.startswith("--speculative") for a in args)


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
def test_mtp_env_controls(tmp_path, script, devices):
    model = _checkpoint(tmp_path / "model", config_layers=1, weights="index")
    run = lambda **env: _serve_args(  # noqa: E731
        tmp_path, script, MODEL_PATH=str(model), **devices, **env
    )
    assert "--speculative-algorithm" not in run(ENABLE_MTP="0")
    steps = run(MTP_NUM_STEPS="3")
    assert steps[steps.index("--speculative-num-steps") + 1] == "3"
    # An explicit --speculative-* in EXTRA_SERVE_ARGS replaces the default.
    explicit = run(EXTRA_SERVE_ARGS="--speculative-algorithm EAGLE3")
    assert explicit.count("--speculative-algorithm") == 1
    assert explicit[explicit.index("--speculative-algorithm") + 1] == "EAGLE3"
    assert "--speculative-num-steps" not in explicit
    # ENABLE_MTP=0 does not remove what the user passed.
    kept = run(ENABLE_MTP="0", EXTRA_SERVE_ARGS="--speculative-algorithm EAGLE3")
    assert kept.count("--speculative-algorithm") == 1


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
def test_mtp_forced_on_requires_mtp_weights(tmp_path, script, devices):
    bare = _checkpoint(tmp_path / "bare", config_layers=1, weights="none")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _serve_args(tmp_path, script, MODEL_PATH=str(bare), ENABLE_MTP="1", **devices)
    assert "has no MTP layer" in exc.value.stderr
    full = _checkpoint(tmp_path / "full", config_layers=1, weights="index")
    args = _serve_args(
        tmp_path, script, MODEL_PATH=str(full), ENABLE_MTP="1", **devices
    )
    assert "--speculative-algorithm" in args
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _serve_args(
            tmp_path, script, MODEL_PATH=str(full), ENABLE_MTP="maybe", **devices
        )
    assert "ENABLE_MTP must be" in exc.value.stderr


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
@pytest.mark.parametrize(
    "max_num_seqs, steps", [("1", "3"), ("32", "3"), ("33", "2"), ("512", "2")]
)
def test_mtp_draft_length_follows_decode_batch_bound(
    tmp_path, script, devices, max_num_seqs, steps
):
    model = _checkpoint(tmp_path / "model", config_layers=1, weights="index")
    args = _serve_args(
        tmp_path,
        script,
        MODEL_PATH=str(model),
        MAX_NUM_SEQS=max_num_seqs,
        **devices,
    )
    assert args[args.index("--speculative-num-steps") + 1] == steps
    explicit = _serve_args(
        tmp_path,
        script,
        MODEL_PATH=str(model),
        MAX_NUM_SEQS=max_num_seqs,
        MTP_NUM_STEPS="4",
        **devices,
    )
    assert explicit[explicit.index("--speculative-num-steps") + 1] == "4"


@pytest.mark.parametrize(
    "script, devices",
    [pytest.param(SCRIPT_QWEN36, {"CUDA_DEVICES": "6"}, id="qwen3.6-35b")],
)
@pytest.mark.parametrize(
    "env, expected",
    [
        ({"MAX_NUM_SEQS": "32"}, None),
        ({"MAX_NUM_SEQS": "128"}, None),
        ({"MAX_NUM_SEQS": "129"}, "128"),
        ({"MAX_NUM_SEQS": "512"}, "128"),
        ({"MAX_NUM_SEQS": "512", "MAX_CUDAGRAPH_CAPTURE_SIZE": "192"}, "192"),
        ({"MAX_NUM_SEQS": "512", "MAX_CUDAGRAPH_CAPTURE_SIZE": ""}, None),
        ({"MAX_NUM_SEQS": "32", "MAX_CUDAGRAPH_CAPTURE_SIZE": "16"}, "16"),
    ],
)
def test_decode_graph_cap_follows_decode_batch_bound(
    tmp_path, script, devices, env, expected
):
    args = _serve_args(tmp_path, script, **devices, **env)
    if expected is None:
        assert "--max-cudagraph-capture-size" not in args
    else:
        assert args[args.index("--max-cudagraph-capture-size") + 1] == expected


def test_decode_graph_ladder_is_the_engine_default_for_the_two_gpu_model(tmp_path):
    """The cap gave qwen3.5-122b-a10b.sh nothing at C=256: no default cap."""
    args = _serve_args(tmp_path, SCRIPT, CUDA_DEVICES="6,7", MAX_NUM_SEQS="512")
    assert "--max-cudagraph-capture-size" not in args


def _mixed_budget(args: list[str]) -> str | None:
    if "--mixed-prefill-token-budget" not in args:
        assert "--enable-mixed-batch" not in args
        return None
    assert "--enable-mixed-batch" in args
    return args[args.index("--mixed-prefill-token-budget") + 1]


@pytest.mark.parametrize("script, devices", SERVE_SCRIPTS)
def test_mixed_rounds_default_on_with_mtp(tmp_path, script, devices):
    model = _checkpoint(tmp_path / "model", config_layers=1, weights="index")
    run = lambda **env: _serve_args(  # noqa: E731
        tmp_path, script, **{"MODEL_PATH": str(model), **devices, **env}
    )
    assert _mixed_budget(run()) == "4096"
    assert _mixed_budget(run(MIXED_PREFILL_TOKEN_BUDGET="2048")) == "2048"
    assert _mixed_budget(run(MIXED_PREFILL_TOKEN_BUDGET="0")) is None
    # Mixed rounds are tied to the MTP default: off without MTP.
    assert _mixed_budget(run(ENABLE_MTP="0")) is None
    bare = _checkpoint(tmp_path / "bare", config_layers=None, weights="index")
    assert _mixed_budget(run(MODEL_PATH=str(bare))) is None
    # An explicit mixed-batch flag replaces the default and is kept once.
    for extra in ("--disable-mixed-batch", "--mixed-prefill-token-budget 1024"):
        args = run(EXTRA_SERVE_ARGS=extra)
        assert args.count("--mixed-prefill-token-budget") == extra.count("budget")
        assert "--enable-mixed-batch" not in args
