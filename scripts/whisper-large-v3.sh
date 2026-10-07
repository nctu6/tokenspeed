#!/usr/bin/env bash
#
# whisper-large-v3.sh -- serve openai/whisper-large-v3 from a local venv
# =============================================================================
#
# Speech recognition (encoder-decoder). Probe OpenAI-compatible
#   POST /v1/audio/transcriptions
# (multipart form: file=...&model=...&language=en), not chat/completions.
#
# Serving path (TokenSpeed-native ASR HTTP + engine):
#   `tokenspeed serve` detects WhisperForConditionalGeneration and launches
#   `runtime/entrypoints/asr_http.py` (feature extract + decoder prompt +
#   resample in-process). SMG's gRPC transcription adapter remains Qwen3-ASR
#   only; see whisper-large-v3.md.
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/openai/whisper-large-v3
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778
#   CUDA_DEVICES         default: 6  (single GPU; large-v3 bf16 ~3 GiB weights
#                        + cross-KV buffer sized by max_num_seqs)
#   TENSOR_PARALLEL_SIZE default: 1
#   MAX_NUM_SEQS         default: 4  (cross-KV is decoder_layers×1500×d_model×2
#                        per slot; keep modest for smoke)
#   MAX_MODEL_LEN        default: 448 (Whisper max_target_positions)
#   GPU_MEMORY_UTILIZATION  default: empty (engine default)
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: bfloat16 (match dtype; avoid fp16/bf16 mix)
#   ATTENTION_BACKEND    default: empty (engine auto)
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#
# Arch → defaults (CUDA compute capability of first selected GPU):
#   sm90  (H200 ~141 GiB, aws-mewtwo): CUDA_DEVICES=6 TP=1
#   sm120 (PRO 6000 ~96 GiB, aws-raichu / Blackwell): CUDA_DEVICES=6 TP=1
#     (only 6,7 available on raichu; prefer GPU 6)
#
# Recommended invoke:
#   CUDA_DEVICES=6 ./scripts/whisper-large-v3.sh
#
# Smoke probe (after server is up):
#   curl -sS http://127.0.0.1:${HOST_PORT}/v1/audio/transcriptions \
#     -F model=test -F language=en -F file=@/tmp/smoke.wav
#
# Usage:
#   ./scripts/whisper-large-v3.sh
#   CUDA_DEVICES=6 HOST_PORT=8778 ./scripts/whisper-large-v3.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/whisper-large-v3.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# --- CUDA arch detection (prefer compute_cap; allow env override) ------------
_normalize_cuda_arch() {
    local raw="${1:-}"
    raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    raw="${raw#sm_}"
    raw="${raw#sm}"
    if [[ "$raw" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        printf 'sm%s%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        printf 'sm%s' "$raw"
        return 0
    fi
    if [[ "$raw" =~ ^sm[0-9]+$ ]]; then
        printf '%s' "$raw"
        return 0
    fi
    echo ""
}

_detect_cuda_arch() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return 0
    fi
    local idx=0
    if [[ -n "${CUDA_DEVICES:-}" ]]; then
        idx="${CUDA_DEVICES%%,*}"
        idx="$(echo "$idx" | tr -d '[:space:]')"
    fi
    local cap
    cap="$(nvidia-smi --id="$idx" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]')"
    if [[ -z "$cap" ]]; then
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d '[:space:]')"
    fi
    _normalize_cuda_arch "$cap"
}

_detect_gpu_mem_mib() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return 0
    fi
    local idx=0
    if [[ -n "${CUDA_DEVICES:-}" ]]; then
        idx="${CUDA_DEVICES%%,*}"
        idx="$(echo "$idx" | tr -d '[:space:]')"
    fi
    nvidia-smi --id="$idx" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]'
}

_CUDA_ARCH_RAW="${TOKENSPEED_CUDA_ARCH:-$(_detect_cuda_arch)}"
_CUDA_ARCH="$(_normalize_cuda_arch "${_CUDA_ARCH_RAW}")"
_GPU_MEM_MIB="$(_detect_gpu_mem_mib)"

_IS_SM120=0
if [[ "${_CUDA_ARCH}" == "sm120" ]]; then
    _IS_SM120=1
fi

MODEL_PATH="${MODEL_PATH:-/models/openai/whisper-large-v3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
CUDA_DEVICES="${CUDA_DEVICES:-6}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-448}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

_NDEV="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
if [[ "${_NDEV}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
    echo "ERROR: CUDA_DEVICES='$CUDA_DEVICES' has ${_NDEV} device(s) but TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}." >&2
    echo "  Prefer single-GPU TP=1, e.g. CUDA_DEVICES=6 TENSOR_PARALLEL_SIZE=1 $0" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [[ ! -x "$VENV_DIR/bin/tokenspeed" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./scripts/install.sh first (or set VENV_DIR)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[whisper-large-v3] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} dtype=${DTYPE} KV=${KV_CACHE_DTYPE:-auto} task=transcriptions" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
)
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
    cmd+=(--attention-backend "$ATTENTION_BACKEND")
fi

exec "${cmd[@]}"
