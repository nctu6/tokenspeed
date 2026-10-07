#!/usr/bin/env bash
#
# minimax-music3.sh -- serve MiniMax-Music3 (lyrics+caption → WAV) via TokenSpeed
# =============================================================================
#
# MiniMax-Music3 is a text-to-music ModularPipeline (Global 8B LM + Local depth
# decoder + Flow-Matching / Flow-VAE), not a chat LLM. `tokenspeed serve`
# detects modular_model_index.json, selects the diffusion runtime, and
# dispatches to entrypoints/music3_http.py which wraps released HuggingFace
# Diffusers ModularPipeline and exposes:
#   POST /v1/audio/speech   (JSON: input=lyrics, instructions=caption → WAV)
#   GET  /health
#   GET  /v1/models
#
# Internals are TokenSpeed-owned (runtime/diffusion/music3_*). Diffusers
# supplies the LM/DiT/vocoder implementations the same way transformers
# supplies Whisper -- not a vendored vLLM-Omni / SGLang tree.
# Deps: ./scripts/install.sh (diffusion + soundfile by default)
#
# Matched baseline knobs (override via env; flag names follow vLLM where useful):
#   MODEL_PATH           default: /models/MiniMaxAI/MiniMax-Music3
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8781
#   CUDA_DEVICES         default: 6,7  (device_split: AR@cuda:1 / acoustic@cuda:0)
#   NUM_GPUS             default: 2 (must match CUDA_DEVICES count)
#   TENSOR_PARALLEL_SIZE default: equal to NUM_GPUS (vLLM-shaped; maps to num_gpus)
#   DTYPE                default: bfloat16
#   ENABLE_CPU_OFFLOAD   default: 1 (ComponentsManager auto offload)
#   OUTPUT_SAMPLE_RATE   default: 32000 (reference servers; use 44100 for native)
#   RESIDENCY            default: auto (device_split when NUM_GPUS>=2)
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#
# Arch notes:
#   sm90  (H200 ~141 GiB): CUDA_DEVICES=6,7 NUM_GPUS=2 device_split
#   sm120 (PRO 6000 ~96 GiB): same; keep smoke audio_duration short
#   Single-GPU still works (~23 GB): CUDA_DEVICES=6 NUM_GPUS=1
#
# Recommended invoke:
#   CUDA_DEVICES=6,7 ./scripts/minimax-music3.sh
#
# Streaming (real acoustic windows after AR; not a late full-song buffer):
#   stream=true → OpenAI speech.audio.* SSE (default stream_format=sse)
#   stream_format=audio → raw audio/wav|pcm bytes (WAV placeholder header)
#
# Smoke probe (after server is up) — NOT chat/completions:
#   curl -sS -X POST "http://127.0.0.1:${HOST_PORT}/v1/audio/speech" \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model":"test",
#       "input":"[verse]\nMorning light through the pine\n[chorus]\nSoftly the world begins to breathe",
#       "instructions":"Genre: acoustic pop. BPM: 96. Soft female vocal, fingerpicked guitar.",
#       "seed":7,
#       "audio_duration":8,
#       "num_inference_steps":4,
#       "response_format":"wav"
#     }' \
#     -o /tmp/music3-smoke.wav
#
# Streaming SSE smoke:
#   curl -sS -N -X POST "http://127.0.0.1:${HOST_PORT}/v1/audio/speech" \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model":"test",
#       "input":"[verse]\nMorning light through the pine\n[chorus]\nSoftly the world begins to breathe",
#       "instructions":"Genre: acoustic pop. BPM: 96. Soft female vocal, fingerpicked guitar.",
#       "seed":7,
#       "audio_duration":8,
#       "num_inference_steps":4,
#       "response_format":"pcm",
#       "stream":true
#     }'
#
# Usage:
#   ./scripts/minimax-music3.sh
#   CUDA_DEVICES=6 HOST_PORT=8781 ./scripts/minimax-music3.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/minimax-music3.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

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

MODEL_PATH="${MODEL_PATH:-/models/MiniMaxAI/MiniMax-Music3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8781}"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-1}"
OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-32000}"
RESIDENCY="${RESIDENCY:-auto}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

_NDEV="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
if [[ "${_NDEV}" -ne "${NUM_GPUS}" ]]; then
    echo "ERROR: CUDA_DEVICES='$CUDA_DEVICES' has ${_NDEV} device(s) but NUM_GPUS=${NUM_GPUS}." >&2
    echo "  MiniMax-Music3 default: CUDA_DEVICES=6,7 NUM_GPUS=2 $0" >&2
    exit 1
fi
if [[ "${TENSOR_PARALLEL_SIZE}" -ne "${NUM_GPUS}" ]]; then
    echo "WARN: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} != NUM_GPUS=${NUM_GPUS}; using NUM_GPUS for residency." >&2
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

if ! python -c "import diffusers" >/dev/null 2>&1; then
    echo "ERROR: diffusers not installed. Run: ./scripts/install.sh (diffusion deps are default)" >&2
    exit 2
fi
if ! python -c "import soundfile" >/dev/null 2>&1; then
    echo "ERROR: soundfile not installed. Run: ./scripts/install.sh (Music3 deps are default)" >&2
    exit 2
fi

echo "[minimax-music3] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} NUM_GPUS=${NUM_GPUS} TP=${TENSOR_PARALLEL_SIZE} residency=${RESIDENCY} dtype=${DTYPE} out_sr=${OUTPUT_SAMPLE_RATE} model=${MODEL_PATH}" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$NUM_GPUS"
    --num-gpus "$NUM_GPUS"
    --dtype "$DTYPE"
    --residency "$RESIDENCY"
    --output-sample-rate "$OUTPUT_SAMPLE_RATE"
)
if [[ "$ENABLE_CPU_OFFLOAD" == "0" ]]; then
    cmd+=(--no-cpu-offload)
fi

exec "${cmd[@]}"
