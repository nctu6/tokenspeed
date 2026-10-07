#!/usr/bin/env bash
#
# minimax-h3.sh -- serve MiniMax-H3 (video+audio) via TokenSpeed-native diffusion
# =============================================================================
#
# MiniMax-H3 is an omni-modal video+audio generative system (H3-Base FL2VA /
# Ref2VA), not a chat LLM. `tokenspeed serve` detects model_index.json, selects
# the diffusion runtime, and launches entrypoints/diffusion_http.py which wraps
# released HuggingFace Diffusers ModularPipeline and exposes:
#   POST /v1/videos          (async job, 202)
#   GET  /v1/videos/{id}
#   GET  /v1/videos/{id}/content
#   POST /v1/videos/sync     (blocking MP4; smoke-friendly)
#
# Internals are TokenSpeed-owned (runtime/diffusion/*). Diffusers supplies the
# DiT/VAE/conditioner implementations the same way transformers supplies Whisper
# -- not a vendored vLLM-Omni tree. Deps: ./scripts/install.sh (diffusion by default)
#
# Matched baseline knobs (override via env; flag names follow vLLM where useful):
#   MODEL_PATH           default: /models/MiniMaxAI/MiniMax-H3  (repo ROOT)
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8780
#   CUDA_DEVICES         default: 6,7  (2 GPUs on both sm90 and sm120)
#   NUM_GPUS             default: 2 (must match CUDA_DEVICES count)
#   TENSOR_PARALLEL_SIZE default: equal to NUM_GPUS (vLLM-shaped; maps to num_gpus)
#   MODEL_VARIANT / TASK_TYPE  default: fl2va  (fl2va | ref2va | t2va | all)
#   DTYPE                default: bfloat16
#   ENABLE_CPU_OFFLOAD   default: 1 (ComponentsManager auto offload)
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#
# Arch notes (2-GPU, devices 6,7):
#   sm90  (H200 ~141 GiB): default device_split (text_encoder@cuda:1 / rest@cuda:0)
#   sm120 (PRO 6000 ~96 GiB): same; keep short-edge smoke canvases small
#   ULYSSES_DEGREE/USP>1 enables TokenSpeed ulysses residency (Diffusers CP); text-encoder-TP still rejected
#
# Recommended invoke:
#   CUDA_DEVICES=6,7 ./scripts/minimax-h3.sh
#   MODEL_VARIANT=ref2va CUDA_DEVICES=6,7 ./scripts/minimax-h3.sh
#
# Smoke probe (after server is up) — NOT chat/completions:
#   curl -sS -X POST "http://127.0.0.1:${HOST_PORT}/v1/videos/sync" \
#     -F model=test \
#     -F 'prompt=A quiet cinematic night scene with matching ambient sound.' \
#     -F width=672 -F height=384 -F fps=24 \
#     -F num_inference_steps=2 -F seed=42 \
#     -F 'extra_params={"task":"t2va","duration":5.0}' \
#     -o /tmp/h3-smoke.mp4
#
# Ref2VA multipart (server started with TASK_TYPE=ref2va or all):
#   curl -sS -X POST "http://127.0.0.1:${HOST_PORT}/v1/videos/sync" \
#     -F model=test \
#     -F 'prompt=Image 1 subject matches Video 1 motion.' \
#     -F width=672 -F height=384 -F num_inference_steps=2 -F seed=42 \
#     -F 'extra_params={"task":"ref2va","duration":5.0}' \
#     -F "input_references=@/path/a.png;type=image/png" \
#     -F "input_references=@/path/b.mp4;type=video/mp4" \
#     -o /tmp/h3-ref2va.mp4
#
# Usage:
#   ./scripts/minimax-h3.sh
#   CUDA_DEVICES=6,7 HOST_PORT=8780 ./scripts/minimax-h3.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/minimax-h3.sh

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

MODEL_PATH="${MODEL_PATH:-/models/MiniMaxAI/MiniMax-H3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8780}"
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_VARIANT="${MODEL_VARIANT:-${TASK_TYPE:-fl2va}}"
TASK_TYPE="${TASK_TYPE:-$MODEL_VARIANT}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-1}"
# DiT Context Parallel (TokenSpeed ulysses residency). Default 1 = device_split path.
ULYSSES_DEGREE="${ULYSSES_DEGREE:-${USP:-1}}"
RING_DEGREE="${RING_DEGREE:-1}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

case "${TASK_TYPE}" in
    fl2va|ref2va|t2va|all) ;;
    *)
        echo "ERROR: TASK_TYPE/MODEL_VARIANT='${TASK_TYPE}' must be fl2va, ref2va, t2va, or all." >&2
        exit 1
        ;;
esac

_NDEV="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
if [[ "${_NDEV}" -ne "${NUM_GPUS}" ]]; then
    echo "ERROR: CUDA_DEVICES='$CUDA_DEVICES' has ${_NDEV} device(s) but NUM_GPUS=${NUM_GPUS}." >&2
    echo "  MiniMax-H3 2-GPU layout: CUDA_DEVICES=6,7 NUM_GPUS=2 $0" >&2
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

# Prefer modular repo root; remap .../FL2VA to parent when present.
if [[ "$(basename "$MODEL_PATH")" == "FL2VA" || "$(basename "$MODEL_PATH")" == "Ref2VA" ]]; then
    _parent="$(dirname "$MODEL_PATH")"
    if [[ -f "$_parent/model_index.json" ]]; then
        echo "[minimax-h3] remapping partition path to modular root: $_parent" >&2
        MODEL_PATH="$_parent"
    fi
fi

echo "[minimax-h3] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} NUM_GPUS=${NUM_GPUS} TP=${TENSOR_PARALLEL_SIZE} usp=${ULYSSES_DEGREE} ring=${RING_DEGREE} task=${TASK_TYPE} dtype=${DTYPE} model=${MODEL_PATH}" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$NUM_GPUS"
    --num-gpus "$NUM_GPUS"
    --dtype "$DTYPE"
    --task-type "$TASK_TYPE"
    --ulysses-degree "$ULYSSES_DEGREE"
    --ring "$RING_DEGREE"
)
if [[ "$ENABLE_CPU_OFFLOAD" == "0" ]]; then
    cmd+=(--no-cpu-offload)
fi

# USP/ring>1: diffusion_http re-execs under torchrun; ensure torchrun is on PATH.
if [[ "${ULYSSES_DEGREE}" -gt 1 || "${RING_DEGREE}" -gt 1 ]]; then
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "ERROR: ULYSSES_DEGREE/RING>1 needs torchrun (activate venv / install torch)." >&2
        exit 2
    fi
    # Avoid NVLS multicast bind failures on broken fabric hosts (CUDA 401).
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
fi

exec "${cmd[@]}"
