#!/usr/bin/env bash
#
# gemma3.sh -- serve gemma-3-27b-it (bf16) from a local venv (see install.sh)
# =====================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/google/gemma-3-27b-it
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8977
#   CUDA_DEVICES         default: 1
#   MAX_NUM_SEQS         default: 452
#   MAX_MODEL_LEN        default: 131072
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: auto (unset/empty = omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: legacy (F2; not auto)
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: off (F3; not 512/all)
#
# Usage:
#   ./scripts/gemma3.sh
#   CUDA_DEVICES=0 HOST_PORT=8977 ./scripts/gemma3.sh
#   ATTENTION_BACKEND=triton ./scripts/gemma3.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/models/google/gemma-3-27b-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
CUDA_DEVICES="${CUDA_DEVICES:-6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-452}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# F1–F3 attention: conservative defaults (match kernel). Override for
# DECODE_KV_SPLITS=auto / FA2_EXTEND=512 after A/B. (The sliding-window
# KV-range trim is always on in the kernel, upstream #1559.)
export TOKENSPEED_TRITON_PREFILL_SKIP_OOR="${TOKENSPEED_TRITON_PREFILL_SKIP_OOR:-1}"
export TOKENSPEED_TRITON_DECODE_KV_SPLITS="${TOKENSPEED_TRITON_DECODE_KV_SPLITS:-legacy}"
export TOKENSPEED_FLASHINFER_FA2_EXTEND="${TOKENSPEED_FLASHINFER_FA2_EXTEND:-off}"

if [[ ! -x "$VENV_DIR/bin/tokenspeed" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./scripts/install.sh first (or set VENV_DIR)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --dtype "$DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-prefix-caching
    --max-model-len "$MAX_MODEL_LEN"
)
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
    cmd+=(--attention-backend "$ATTENTION_BACKEND")
fi

exec "${cmd[@]}"
