#!/usr/bin/env bash
#
# qwen3-embedding-8b.sh -- serve Qwen3-Embedding-8B from a local venv
# =============================================================================
#
# Embedding model (last-token pooling). Probe OpenAI-compatible
#   POST /v1/embeddings
# not chat/completions. Chat-only knobs (reasoning / tool parsers) are omitted.
#
# Embedding serving (TokenSpeed-native + SMG):
#   `tokenspeed serve` fronts SMG + TokenSpeed gRPC. With Embed RPC + pooler
#   support, POST /v1/embeddings returns vectors for this checkpoint.
#   Pass `--is-embedding` (or rely on auto-detect from 1_Pooling/config.json).
#   Prefix caching / overlap schedule are forced off for pooling models.
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/Qwen/Qwen3-Embedding-8B
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778
#   CUDA_DEVICES         default: 6  (single GPU; 8B bf16 ~15 GiB weights)
#   TENSOR_PARALLEL_SIZE default: 1  (must match device count; minimal TP)
#   QUANTIZATION         default: empty (bf16 fits one PRO 6000 ~96 GiB or H200)
#   MAX_NUM_SEQS         default: 128 on both arches
#   MAX_MODEL_LEN        default: 32768 (model cord supports up to 40960;
#                        leave headroom for KV on a single ~96 GiB card)
#   GPU_MEMORY_UTILIZATION  default: empty (engine default)
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: fp8 (unset/empty = omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: legacy (F2; not auto)
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: off (F3; not 512/all)
#   VLLM_BATCH_INVARIANT default: empty. Truthy (1/true/yes/on) = deterministic
#                        comm preset: --force-deterministic-rsag +
#                        --disable-nccl-nvls (NCCL_NVLS_ENABLE=0). Needed on
#                        hosts with broken NVLS multicast (e.g. mewtwo) when
#                        TP>1. Harmless to leave unset for TP=1.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#
# Arch → defaults (CUDA compute capability of first selected GPU):
#   sm90  (H200 ~141 GiB, aws-mewtwo):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1
#     bf16 weights ~15 GiB — fits one H200. No --quantization.
#   sm120 (PRO 6000 ~96 GiB, aws-raichu / Blackwell):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1  (only 6,7 available on raichu;
#     single GPU 6 preferred for 8B embeds)
#     bf16 weights ~15 GiB — fits ~96 GiB without online FP8.
#   Override CUDA_DEVICES / TENSOR_PARALLEL_SIZE / QUANTIZATION /
#   TOKENSPEED_CUDA_ARCH to pin. For TP=2 on free 6,7:
#     CUDA_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 ./scripts/qwen3-embedding-8b.sh
#
# Recommended invoke:
#   # mewtwo (H200 / sm90) — single GPU 6
#   CUDA_DEVICES=6 ./scripts/qwen3-embedding-8b.sh
#   # raichu (PRO 6000 / sm120) — single GPU 6
#   CUDA_DEVICES=6 ./scripts/qwen3-embedding-8b.sh
#   # optional TP=2 on 6,7 (mewtwo: also set VLLM_BATCH_INVARIANT=1)
#   CUDA_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 VLLM_BATCH_INVARIANT=1 \
#     ./scripts/qwen3-embedding-8b.sh
#
# Smoke probe (after server is up):
#   curl -sS http://127.0.0.1:${HOST_PORT}/v1/embeddings \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"test","input":"hello embedding"}'
#
# Usage:
#   ./scripts/qwen3-embedding-8b.sh
#   CUDA_DEVICES=6 HOST_PORT=8778 ./scripts/qwen3-embedding-8b.sh
#   ATTENTION_BACKEND=triton ./scripts/qwen3-embedding-8b.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/qwen3-embedding-8b.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# --- CUDA arch detection (prefer compute_cap; allow env override) ------------
# Normalize "12.0" / "sm_120" / "SM120" → "sm120".
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

# Probe compute_cap of the first GPU in CUDA_DEVICES (if already set), else GPU 0.
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

# sm120 (Blackwell / PRO 6000) and sm90 (H200): bf16 TP=1 on GPU 6.
_IS_SM120=0
if [[ "${_CUDA_ARCH}" == "sm120" ]]; then
    _IS_SM120=1
fi

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
if [[ "${_IS_SM120}" -eq 1 ]]; then
    # 8B bf16 ~15 GiB fits one PRO 6000 ~96 GiB; prefer minimal TP on free GPU 6.
    CUDA_DEVICES="${CUDA_DEVICES:-6}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    QUANTIZATION="${QUANTIZATION:-}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
else
    # sm90 H200 (and unknown arch): same minimal-TP defaults.
    CUDA_DEVICES="${CUDA_DEVICES:-6}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    QUANTIZATION="${QUANTIZATION:-}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
# Hosts with broken NVLS multicast need these when TP>1; default off.
FORCE_DETERMINISTIC_RSAG="${FORCE_DETERMINISTIC_RSAG:-}"
DISABLE_NCCL_NVLS="${DISABLE_NCCL_NVLS:-}"
case "${VLLM_BATCH_INVARIANT:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        FORCE_DETERMINISTIC_RSAG="${FORCE_DETERMINISTIC_RSAG:-1}"
        DISABLE_NCCL_NVLS="${DISABLE_NCCL_NVLS:-1}"
        ;;
esac
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

# Device count must match TP (comma-separated CUDA_DEVICES).
_NDEV="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
if [[ "${_NDEV}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
    echo "ERROR: CUDA_DEVICES='$CUDA_DEVICES' has ${_NDEV} device(s) but TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}." >&2
    echo "  Qwen3-Embedding-8B bf16 is ~15 GiB — prefer single-GPU TP=1, e.g.:" >&2
    echo "    CUDA_DEVICES=6 TENSOR_PARALLEL_SIZE=1 $0" >&2
    echo "  Or TP=2 on free 6,7:" >&2
    echo "    CUDA_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 $0" >&2
    exit 1
fi

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

echo "[qwen3-embedding-8b] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} quant=${QUANTIZATION:-none} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default} task=embeddings" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --is-embedding
    --max-model-len "$MAX_MODEL_LEN"
)
if [[ -n "$QUANTIZATION" ]]; then
    cmd+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
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

exec "${cmd[@]}"
