#!/usr/bin/env bash
#
# gemma3.sh -- serve gemma-3-27b-it (bf16) from a local venv (see install.sh)
# =====================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/google/gemma-3-27b-it
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778 (the script exits if it is in use)
#   CUDA_DEVICES         default: 6
#   MAX_NUM_SEQS         default: 452
#   MAX_MODEL_LEN        default: 131072
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: empty (omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   VENV_DIR             default: ./.venv
#   DISABLE_KVSTORE      default: 0. Set 1 to pass --disable-kvstore.
#   EXTRA_ARGS           default: empty. Extra `tokenspeed serve` flags.
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: unset (kernel default auto)
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: unset (kernel default 512)
#   VLLM_BATCH_INVARIANT default: empty. Truthy = --force-deterministic-rsag +
#                        --disable-nccl-nvls (for TP>1 on hosts with broken
#                        NVLS multicast, e.g. sm90). Single-GPU default is fine
#                        without it.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#
# SM120 defaults (RTX PRO 6000, compute capability 12.x of the first
# CUDA_DEVICES GPU; explicit env always wins, other archs are unchanged):
#   No attention/GEMM backend is forced: the engine picks the SM120 attention
#   leaves and the DeepGEMM SM120 GEMM leaves, and the CUDA graph ladder runs up
#   to MAX_NUM_SEQS (no capture cap is passed). Prefill graphs and KV events keep
#   the engine defaults.
#   /v1/completions: the gateway adds special tokens by the tokenizer's own
#   rules, like vLLM (task 87), so gemma-3 raw prompts get <bos> automatically;
#   do not prefix <bos> yourself.
#
# Multi-model (SMG IGW): override SERVED_MODEL_NAME to a unique id when
# registering beside other models; pair with gemma4.sh on another port/GPUs
# and enable IGW on the gateway. See multi-model-gemma.md.
#
# Usage:
#   ./scripts/gemma3.sh
#   CUDA_DEVICES=0 HOST_PORT=8778 ./scripts/gemma3.sh
#   ATTENTION_BACKEND=triton ./scripts/gemma3.sh
#   # multi-model: override served name when needed
#   SERVED_MODEL_NAME=gemma-3-27b-it ./scripts/gemma3.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/models/google/gemma-3-27b-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
# gemma templates are not a Symphony "hf" table (the gateway would refuse
# them); passthrough returns text as generated, nctu6's proven config.
REASONING_PARSER="${REASONING_PARSER:-passthrough}"
HOST_PORT="${HOST_PORT:-8778}"
# Fail fast when HOST_PORT is taken (another server, possibly another user's).
if (exec 3<>"/dev/tcp/127.0.0.1/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: port ${HOST_PORT} is already in use. Set HOST_PORT to a free port, e.g. HOST_PORT=18778 $0" >&2
    exit 1
fi
CUDA_DEVICES="${CUDA_DEVICES:-6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-452}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
# SM120 (compute capability 12.x) of the first selected GPU. TOKENSPEED_CUDA_ARCH
# (e.g. sm120 / 12.0) overrides the nvidia-smi query. Same variables as the
# other model scripts: _CUDA_ARCH_RAW, _IS_SM120.
_CUDA_ARCH_RAW="${TOKENSPEED_CUDA_ARCH:-$(nvidia-smi --id="${CUDA_DEVICES%%,*}" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 || true)}"
_IS_SM120=0
case "$(tr '[:upper:]' '[:lower:]' <<<"${_CUDA_ARCH_RAW}" | tr -d '[:space:]_')" in
    12.*|sm12*) _IS_SM120=1 ;;
esac
DISABLE_KVSTORE="${DISABLE_KVSTORE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE_DETERMINISTIC_RSAG="${FORCE_DETERMINISTIC_RSAG:-}"
DISABLE_NCCL_NVLS="${DISABLE_NCCL_NVLS:-}"
case "${VLLM_BATCH_INVARIANT:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        FORCE_DETERMINISTIC_RSAG="${FORCE_DETERMINISTIC_RSAG:-1}"
        DISABLE_NCCL_NVLS="${DISABLE_NCCL_NVLS:-1}"
        ;;
esac
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# F1–F3 attention: the kernel defaults apply unless set here (split-KV
# decode auto, FA2 extend for head_dim 512). (The sliding-window KV-range
# trim is always on in the kernel, upstream #1559.)
export TOKENSPEED_TRITON_PREFILL_SKIP_OOR="${TOKENSPEED_TRITON_PREFILL_SKIP_OOR:-1}"

if [[ ! -x "$VENV_DIR/bin/tokenspeed" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./scripts/install.sh first (or set VENV_DIR)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[gemma3] sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto}" >&2
cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --reasoning-parser "$REASONING_PARSER"
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
if [[ -n "$FORCE_DETERMINISTIC_RSAG" ]]; then
    cmd+=(--force-deterministic-rsag)
fi
if [[ -n "$DISABLE_NCCL_NVLS" ]]; then
    cmd+=(--disable-nccl-nvls)
    export NCCL_NVLS_ENABLE=0
fi

if [[ "$DISABLE_KVSTORE" == "1" ]]; then
    cmd+=(--disable-kvstore)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    cmd+=($EXTRA_ARGS)
fi

if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi

exec "${cmd[@]}"
