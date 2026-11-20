#!/usr/bin/env bash
#
# minimax-h3-image.sh -- serve MiniMax-H3-Image (text-to-image + editing)
# =============================================================================
#
# MiniMax-H3-Image is a 33B DiT image model (up to ~16 MP, 6 distilled steps,
# no guidance), not a chat LLM. `tokenspeed serve` detects model_index.json
# (MiniMaxH3ImagePipeline), selects the diffusion runtime, and dispatches to
# entrypoints/image_http.py, which wraps the released HuggingFace Diffusers
# MiniMaxH3ImageModularPipeline (diffusers>=0.43) and exposes:
#   POST /v1/images/generations   (JSON: prompt, size "WxH", seed, ...)
#   POST /v1/images/edits         (multipart: prompt, size, image=@ref ...)
#   GET  /health, /v1/models
#
# Deps: ./scripts/install.sh (diffusion deps, diffusers>=0.43.0)
#
# Knobs (override via env):
#   MODEL_PATH           default: /models/MiniMaxAI/MiniMax-H3-Image
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8782
#   CUDA_DEVICES         default: 6,7
#   NUM_GPUS             default: 2 (must match CUDA_DEVICES count)
#   RESIDENCY            default: auto (device_split when NUM_GPUS>=2: Qwen3-VL
#                        text encoder on the 2nd GPU, DiT + VAE on the 1st)
#   DTYPE                default: bfloat16
#   ENABLE_CPU_OFFLOAD   default: 0 (both ~62 GB weight sets stay resident
#                        with device_split); 1 enables auto CPU offload, which
#                        is what a single 80 GB GPU needs (NUM_GPUS=1)
#   VENV_DIR             default: ./.venv
#
# The text encoder is loaded from the `text_encoder/` folder of the checkpoint
# (model_index.json points at MiniMaxAI/MiniMax-H3; keep a local copy or the
# HF cache populated).
#
# Smoke probe (after the server is up):
#   curl -sS -X POST "http://127.0.0.1:${HOST_PORT}/v1/images/generations" \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"A red fox in fresh snow, photorealistic","size":"1024x1024","seed":42}' \
#     | python -c 'import sys,json,base64;open("h3-image.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
#
# Edit probe (prompt refers to references as <Picture 1>, <Picture 2>, ...):
#   curl -sS -X POST "http://127.0.0.1:${HOST_PORT}/v1/images/edits" \
#     -F 'prompt=Keep <Picture 1> as it is, but make it night.' \
#     -F size=1024x1024 -F seed=1 -F image=@ref.png
#
# Usage:
#   ./scripts/minimax-h3-image.sh
#   CUDA_DEVICES=6,7 HOST_PORT=8782 ./scripts/minimax-h3-image.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/models/MiniMaxAI/MiniMax-H3-Image}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8782}"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
RESIDENCY="${RESIDENCY:-auto}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-0}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

_NDEV="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
if [[ "${_NDEV}" -ne "${NUM_GPUS}" ]]; then
    echo "ERROR: CUDA_DEVICES='$CUDA_DEVICES' has ${_NDEV} device(s) but NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/model_index.json" ]]; then
    echo "ERROR: $MODEL_PATH/model_index.json not found (set MODEL_PATH)." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./scripts/install.sh first (or set VENV_DIR)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if ! python -c "from diffusers import MiniMaxH3ImageModularPipeline" >/dev/null 2>&1; then
    echo "ERROR: diffusers>=0.43.0 not installed. Run: ./scripts/install.sh" >&2
    exit 2
fi

echo "[minimax-h3-image] CUDA_DEVICES=${CUDA_DEVICES} NUM_GPUS=${NUM_GPUS} residency=${RESIDENCY} offload=${ENABLE_CPU_OFFLOAD} dtype=${DTYPE} model=${MODEL_PATH}" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --num-gpus "$NUM_GPUS"
    --dtype "$DTYPE"
    --residency "$RESIDENCY"
)
if [[ "$ENABLE_CPU_OFFLOAD" == "0" ]]; then
    cmd+=(--no-cpu-offload)
fi
exec "${cmd[@]}"
