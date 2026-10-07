#!/usr/bin/env bash
#
# qwen3.6-35b-a3b.sh -- serve Qwen3.6-35B-A3B from a local venv
# =============================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/Qwen/Qwen3.6-35B-A3B
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778
#   CUDA_DEVICES         arch-aware default (see below); override to pin GPUs
#   TENSOR_PARALLEL_SIZE arch-aware default (see below); must match device count
#   QUANTIZATION         default: empty (bf16 fits TP=1 on both sm90 and sm120;
#                        set e.g. fp8 only if needed)
#   MOE_BACKEND          default: empty (engine auto). Set e.g. triton if online
#                        FP8 MoE is forced on sm120.
#   MAX_NUM_SEQS         default: 32 on both (TP=1 cudagraph + KV headroom)
#   MAX_MODEL_LEN        default: 8192 on both (bf16 weights ~67 GiB on one GPU
#                        at TP=1; longer ctx OOMs during KV/cudagraph reserve
#                        even on H200 ~141 GiB. Override upward if you have
#                        headroom. Use fp8 KV.)
#   GPU_MEMORY_UTILIZATION  default: empty (engine default; VLM may clamp to 0.9).
#                           Set e.g. 0.85 on tight cards after weights fit.
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: fp8 (unset/empty = omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   ENABLE_EXPERT_PARALLEL  default: empty. Non-empty = --enable-expert-parallel
#                           (ep_size = stage world). Same weight bytes as TP-only
#                           at the same world size; optional MoE routing layout.
#   REASONING_PARSER     default: qwen3_thinking (chat template opens <think>
#                           on the assistant turn when thinking is on; this
#                           parser assumes reasoning from the first generated
#                           token. Override to qwen3 if you always pass
#                           enable_thinking=false / empty think block.)
#   ENABLE_AUTO_TOOL_CHOICE default: empty. Non-empty = --enable-auto-tool-choice
#                           (current SMG builds do not accept this flag; leave
#                           off — tool calling is enabled via TOOL_CALL_PARSER)
#   TOOL_CALL_PARSER     default: qwen_coder (alias of qwen_xml; Qwen3.6 XML
#                           <tool_call>/<function=...> format. Omit by setting
#                           empty)
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: legacy (F2; not auto)
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: off (F3; not 512/all)
#   VLLM_BATCH_INVARIANT default: empty. Truthy (1/true/yes/on) = deterministic
#                        comm preset: --force-deterministic-rsag +
#                        --disable-nccl-nvls (NCCL_NVLS_ENABLE=0). Needed on
#                        multi-GPU hosts with broken NVLS multicast (e.g.
#                        mewtwo). Harmless no-op for single-GPU TP=1. The
#                        engine also honors it directly from the environment.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#
# Arch → defaults (CUDA compute capability of first selected GPU):
#   sm90  (H200 ~141 GiB, aws-mewtwo):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1
#     MAX_MODEL_LEN=8192  MAX_NUM_SEQS=32
#     bf16 weights ~67 GiB on one GPU at TP=1 — fits H200. No --quantization.
#     MoE backend = engine auto (no forced triton).
#   sm120 (PRO 6000 ~96 GiB, aws-raichu / Blackwell):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1
#     MAX_MODEL_LEN=8192  MAX_NUM_SEQS=32
#     bf16 weights ~67 GiB on one GPU at TP=1 — fits ~96 GiB without online FP8
#     if context/seqs stay modest (cudagraph + KV need the leftover ~25 GiB).
#     MoE backend = engine auto (no forced triton / FP8).
#   Override CUDA_DEVICES / TENSOR_PARALLEL_SIZE / QUANTIZATION / MOE_BACKEND /
#   TOKENSPEED_CUDA_ARCH to pin.
#
# Recommended invoke:
#   # mewtwo (H200 / sm90) — single-GPU TP=1 on GPU 6
#   CUDA_DEVICES=6 ./scripts/qwen3.6-35b-a3b.sh
#   # raichu (PRO 6000 / sm120) — bf16 TP=1 on free GPU 6
#   CUDA_DEVICES=6 ./scripts/qwen3.6-35b-a3b.sh
#
# Usage:
#   ./scripts/qwen3.6-35b-a3b.sh
#   CUDA_DEVICES=6 HOST_PORT=8778 ./scripts/qwen3.6-35b-a3b.sh
#   ATTENTION_BACKEND=triton ./scripts/qwen3.6-35b-a3b.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/qwen3.6-35b-a3b.sh
#   REASONING_PARSER=qwen3 TOOL_CALL_PARSER=qwen_xml ./scripts/qwen3.6-35b-a3b.sh

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

# sm120 (Blackwell / PRO 6000) and sm90 (H200): bf16 TP=1 on GPU 6 — no forced FP8.
_IS_SM120=0
if [[ "${_CUDA_ARCH}" == "sm120" ]]; then
    _IS_SM120=1
fi

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
if [[ "${_IS_SM120}" -eq 1 ]]; then
    # bf16 MoE ~67 GiB on one GPU at TP=1 fits PRO 6000 ~96 GiB only with
    # shorter ctx / lower max-num-seqs (cudagraph capture + KV need leftover).
    # No online FP8 / triton MoE required unlike 122B-A10B.
    CUDA_DEVICES="${CUDA_DEVICES:-6}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    QUANTIZATION="${QUANTIZATION:-}"
    MOE_BACKEND="${MOE_BACKEND:-}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
else
    # sm90 H200 (and unknown arch): bf16 TP=1 on GPU 6, no forced FP8.
    # 32768×128 / 16384×64 can still SIGKILL during KV reserve; 8192×32 is
    # the reliable single-GPU smoke baseline (override upward if you have room).
    CUDA_DEVICES="${CUDA_DEVICES:-6}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    QUANTIZATION="${QUANTIZATION:-}"
    MOE_BACKEND="${MOE_BACKEND:-}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
fi
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}"
REASONING_PARSER="${REASONING_PARSER:-qwen3_thinking}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen_coder}"
# Hosts with broken NVLS multicast need these; default off for other hosts.
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
    echo "  bf16 Qwen3.6-35B-A3B needs ~67 GiB on one GPU at TP=1 (OK on H200 ~141 GiB" >&2
    echo "  and PRO 6000 ~96 GiB / sm120 with shorter MAX_MODEL_LEN). Use e.g.:" >&2
    echo "    CUDA_DEVICES=6 TENSOR_PARALLEL_SIZE=1 $0" >&2
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

echo "[qwen3.6-35b] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} quant=${QUANTIZATION:-none} moe_backend=${MOE_BACKEND:-auto} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default} reasoning=${REASONING_PARSER:-none} tool_parser=${TOOL_CALL_PARSER:-none}" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-prefix-caching
    --max-model-len "$MAX_MODEL_LEN"
)
if [[ -n "$REASONING_PARSER" ]]; then
    cmd+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$QUANTIZATION" ]]; then
    cmd+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$MOE_BACKEND" ]]; then
    cmd+=(--moe-backend "$MOE_BACKEND")
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
if [[ -n "$ENABLE_EXPERT_PARALLEL" ]]; then
    cmd+=(--enable-expert-parallel)
fi
if [[ -n "$ENABLE_AUTO_TOOL_CHOICE" ]]; then
    cmd+=(--enable-auto-tool-choice)
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    cmd+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$FORCE_DETERMINISTIC_RSAG" ]]; then
    cmd+=(--force-deterministic-rsag)
fi
if [[ -n "$DISABLE_NCCL_NVLS" ]]; then
    cmd+=(--disable-nccl-nvls)
    export NCCL_NVLS_ENABLE=0
fi

exec "${cmd[@]}"
