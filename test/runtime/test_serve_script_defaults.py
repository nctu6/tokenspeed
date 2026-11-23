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
"""``scripts/qwen3.5-122b-a10b.sh``: which serve flags the arch defaults select.

CPU-only. The script runs for real; only the ``tokenspeed`` executable in a
throwaway venv is replaced by one that prints its arguments.
"""

import os
import pathlib
import stat
import subprocess

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "qwen3.5-122b-a10b.sh"
)


def _serve_args(tmp_path: pathlib.Path, **env: str) -> list[str]:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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
        ["bash", str(SCRIPT)],
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
