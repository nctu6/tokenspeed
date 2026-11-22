#!/usr/bin/env bash
#
# qwen3.5-122b-a10b.sh -- serve Qwen3.5-122B-A10B from a local venv
# =============================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/Qwen/Qwen3.5-122B-A10B
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778 (the script exits if it is in use)
#   CUDA_DEVICES         arch-aware default (see below); override to pin GPUs
#   TENSOR_PARALLEL_SIZE arch-aware default (see below); must match device count
#   QUANTIZATION         arch-aware (fp8 on sm120 / PRO 6000; empty on sm90 H200);
#                        nvfp4 for an NVFP4 checkpoint (detected from the
#                        checkpoint's quantization config, not its path)
#   MOE_BACKEND          arch-aware (triton required for online FP8 MoE on sm120;
#                        empty = engine auto on bf16 H200 / sm90 path;
#                        auto for an NVFP4 checkpoint)
#   MAX_NUM_SEQS         default: 128 on sm90; 64 on sm120 (~96 GiB)
#   MAX_MODEL_LEN        default: 32768 on sm90; 8192 on sm120 FP8 TP=2
#                        (KV ~23 GiB at 8k with ~9 GiB free on 96 GiB; 32k KV OOMs)
#   GPU_MEMORY_UTILIZATION  default: empty (engine default; VLM may clamp to 0.9).
#                           Set e.g. 0.85 on tight cards after weights fit.
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: fp8 (unset/empty = omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   ENABLE_EXPERT_PARALLEL  default: empty. Non-empty = --enable-expert-parallel
#                           (ep_size = stage world). Same weight bytes as TP-only
#                           at the same world size; optional MoE routing layout.
#   ENABLE_AUTO_TOOL_CHOICE default: empty. Non-empty = --enable-auto-tool-choice
#                           (some SMG builds reject this flag; leave off on the sm120 host)
#   TOOL_CALL_PARSER     default: qwen_coder (omit by setting empty)
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm120 / sm90 / 12.0 / 9.0)
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: legacy (F2; not auto); unset
#                        (kernel default) on SM120
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: off (F3; not 512/all); unset
#                        (kernel default) on SM120
#   VLLM_BATCH_INVARIANT default: empty. Truthy (1/true/yes/on) = deterministic
#                        comm preset: --force-deterministic-rsag +
#                        --disable-nccl-nvls (NCCL_NVLS_ENABLE=0). Needed on
#                        hosts with broken NVLS multicast (e.g. some sm90 hosts). The
#                        engine also honors it directly from the environment.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#
# Arch → defaults (CUDA compute capability of first selected GPU):
#   sm90  (H200 ~141 GiB):
#     CUDA_DEVICES=6,7  TENSOR_PARALLEL_SIZE=2
#     bf16 weights ~114 GiB/GPU at TP=2 — fits H200. No --quantization.
#     MoE backend = engine auto (no forced triton).
#   sm120 (PRO 6000 ~96 GiB, Blackwell):
#     CUDA_DEVICES=6,7  TENSOR_PARALLEL_SIZE=2  (only 6,7 available on the sm120 host)
#     Online --quantization fp8 + --moe-backend triton:
#       MoE weights ~59.6 GB/GPU; total ~86 GiB used at max_model_len=8192.
#     Default flashinfer_trtllm FP8 MoE crashes on sm120:
#       unexpected hidden_states_scale shape (196608,); expected (24, 8192).
#       MoE backend=triton fixes it. BF16 at TP=2 OOMs (~114 GiB weights/GPU).
#     NVFP4 checkpoint (e.g. /models/nvidia/Qwen3.5-122B-A10B-NVFP4, ~39GiB/GPU
#     at TP=2): the script detects it from the checkpoint config and defaults to
#     --quantization nvfp4 --moe-backend auto (the triton MoE has no NVFP4
#     kernel). Explicit QUANTIZATION / MOE_BACKEND still win.
#   Override CUDA_DEVICES / TENSOR_PARALLEL_SIZE / QUANTIZATION / MOE_BACKEND /
#   TOKENSPEED_CUDA_ARCH to pin.
#
# SM120 defaults (compute capability 12.x of the first CUDA_DEVICES GPU; explicit
# env always wins, other archs are unchanged):
#   - ATTENTION_BACKEND is not forced (engine picks the SM120 attention leaves),
#     the TOKENSPEED_TRITON_DECODE_KV_SPLITS / FA2_EXTEND pins are not exported,
#     and no CUDA graph capture cap is passed: the ladder runs up to MAX_NUM_SEQS.
#   - NVFP4 checkpoint: --quantization nvfp4 --moe-backend auto (DeepGEMM / CUTLASS
#     FP4 leaves), detected from the checkpoint config.
#   - BF16 checkpoint: --quantization fp8 as before (BF16 OOMs at ~114 GiB/GPU on
#     2 x 96 GB). --moe-backend triton stays for this online-FP8 case because the
#     default FlashInfer TRT-LLM FP8 MoE crashes on SM120 (bring-up on the SM120
#     host: hidden_states_scale shape error); `MOE_BACKEND=auto` re-tests it.
#   - MAX_MODEL_LEN 8192 / MAX_NUM_SEQS 64 / KV fp8 are memory limits (~86 GiB
#     used at 8192 context), not sm90 leftovers (32k KV OOMs, see above).
#
# Recommended invoke:
#   # sm90 (H200) — NVLS multicast broken → batch-invariant preset
#   CUDA_DEVICES=6,7 VLLM_BATCH_INVARIANT=1 ./scripts/qwen3.5-122b-a10b.sh
#   # sm120 (PRO 6000) — FP8 TP=2 + triton MoE on free GPUs 6,7
#   CUDA_DEVICES=6,7 ./scripts/qwen3.5-122b-a10b.sh
#   # sm120 NVFP4 checkpoint (no quantization / MoE env needed)
#   MODEL_PATH=/models/nvidia/Qwen3.5-122B-A10B-NVFP4 ./scripts/qwen3.5-122b-a10b.sh
#
# Usage:
#   ./scripts/qwen3.5-122b-a10b.sh
#   CUDA_DEVICES=6,7 HOST_PORT=8778 ./scripts/qwen3.5-122b-a10b.sh
#   ATTENTION_BACKEND=triton ./scripts/qwen3.5-122b-a10b.sh
#   TOKENSPEED_CUDA_ARCH=sm120 ./scripts/qwen3.5-122b-a10b.sh

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

# Default GPUs first: the arch below is read from the first selected device.
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
_CUDA_ARCH_RAW="${TOKENSPEED_CUDA_ARCH:-$(_detect_cuda_arch)}"
_CUDA_ARCH="$(_normalize_cuda_arch "${_CUDA_ARCH_RAW}")"
_GPU_MEM_MIB="$(_detect_gpu_mem_mib)"

# NVFP4 checkpoint (modelopt quant_algo NVFP4 in config.json or
# hf_quant_config.json; the path name is not used) -> QUANTIZATION=nvfp4,
# MOE_BACKEND=auto.
_checkpoint_is_nvfp4() {
    python3 - "$1" <<'PY'
import json
import os
import sys

root = sys.argv[1]
algos = []
for name, keys in (("config.json", ("quantization_config",)), ("hf_quant_config.json", ("quantization",))):
    path = os.path.join(root, name)
    if os.path.isfile(path):
        with open(path) as f:
            section = json.load(f).get(keys[0]) or {}
        algos.append(str(section.get("quant_algo", "")).upper())
sys.exit(0 if "NVFP4" in algos else 1)
PY
}
# sm120 (Blackwell / PRO 6000) → FP8 + triton MoE + TP=2 on 6,7.
# sm90 / other high-mem (H200) → bf16 TP=2 on 6,7; no forced FP8/triton.
_IS_SM120=0
if [[ "${_CUDA_ARCH}" == sm12* ]]; then
    _IS_SM120=1
fi

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
# Fail fast when HOST_PORT is taken (another server, possibly another user's).
if (exec 3<>"/dev/tcp/127.0.0.1/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: port ${HOST_PORT} is already in use. Set HOST_PORT to a free port, e.g. HOST_PORT=18778 $0" >&2
    exit 1
fi
_NVFP4=0
if [[ -d "$MODEL_PATH" ]] && _checkpoint_is_nvfp4 "$MODEL_PATH"; then
    _NVFP4=1
fi
if [[ "${_IS_SM120}" -eq 1 ]]; then
    # Online FP8 shrinks weights enough for TP=2 on ~96 GiB (verified on the sm120 host
    # GPUs 6,7). Triton MoE required; default flashinfer_trtllm FP8 MoE crashes
    # on sm120 (hidden_states_scale shape mismatch).
    CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    if [[ "${_NVFP4}" -eq 1 ]]; then
        # Pre-quantized NVFP4: the triton MoE has no NVFP4 kernel.
        QUANTIZATION="${QUANTIZATION:-nvfp4}"
        MOE_BACKEND="${MOE_BACKEND:-auto}"
    else
        QUANTIZATION="${QUANTIZATION:-fp8}"
        MOE_BACKEND="${MOE_BACKEND:-triton}"
    fi
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
else
    # sm90 H200 (and unknown arch): prior working defaults — TP=2, no forced FP8.
    CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    if [[ "${_NVFP4}" -eq 1 ]]; then
        QUANTIZATION="${QUANTIZATION:-nvfp4}"
        MOE_BACKEND="${MOE_BACKEND:-auto}"
    else
        QUANTIZATION="${QUANTIZATION:-}"
        MOE_BACKEND="${MOE_BACKEND:-}"
    fi
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}"
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
    echo "  bf16 Qwen3.5-122B-A10B needs ~114 GiB/GPU at TP=2 (OK on H200 ~141 GiB;" >&2
    echo "  OOM on PRO 6000 ~96 GiB / sm120). On sm120 use online FP8 + TP=2, e.g.:" >&2
    echo "    CUDA_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 QUANTIZATION=fp8 MOE_BACKEND=triton $0" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# F1–F3 attention: conservative defaults (match kernel). Override for
# DECODE_KV_SPLITS=auto / FA2_EXTEND=512 after A/B. (The sliding-window
# KV-range trim is always on in the kernel, upstream #1559.)
export TOKENSPEED_TRITON_PREFILL_SKIP_OOR="${TOKENSPEED_TRITON_PREFILL_SKIP_OOR:-1}"
if [[ "${_IS_SM120}" -eq 1 ]]; then
    # SM120: leave both unset (kernel defaults) so the SM120 attention leaves are
    # not pinned to the older split-KV sizing / extend path. Exporting them
    # yourself still wins.
    :
else
    export TOKENSPEED_TRITON_DECODE_KV_SPLITS="${TOKENSPEED_TRITON_DECODE_KV_SPLITS:-legacy}"
    export TOKENSPEED_FLASHINFER_FA2_EXTEND="${TOKENSPEED_FLASHINFER_FA2_EXTEND:-off}"
fi

if [[ ! -x "$VENV_DIR/bin/tokenspeed" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run ./scripts/install.sh first (or set VENV_DIR)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[qwen3.5-122b] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} quant=${QUANTIZATION:-none} moe_backend=${MOE_BACKEND:-auto} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default}" >&2

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
    --reasoning-parser qwen3
)
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

# Extra serve args (space-separated), appended verbatim, e.g. EXTRA_SERVE_ARGS="--chunked-prefill-size 4096".
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi
exec "${cmd[@]}"
