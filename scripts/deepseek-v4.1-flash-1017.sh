#!/usr/bin/env bash
#
# deepseek-v4.1-flash-1017.sh -- serve DeepSeek-V4.1-Flash-1017 (TP=2 or TP=4)
# =============================================================================
#
# Single-node, aggregated serve of deepseek-ai/DeepSeek-V4.1-Flash-1017 from a
# local venv (see ./scripts/install.sh). For the two-node 1P1D disaggregated
# topology use test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh instead.
#
# DeepSeek V4.1 (deepseek_v41) runs on its FlatKV attention backend with a
# four-group KV cache; the serve knobs below follow the recipe in
# docs/recipes/models.md ("DeepSeek V4.1-Flash"). Same-checkpoint DSpark
# speculative decoding is on by default (the CI/eval default for this
# checkpoint); it requires --disable-kvstore and prefix caching on.
#
# MoE backend is arch-aware:
#   Datacenter Blackwell (sm100/sm103, B200/GB200): mega_moe + expert parallel
#     (the FlashInfer TRT-LLM vendor MoE, validated only on this window).
#   Workstation Blackwell (sm120/sm121, RTX PRO 6000): auto -> CUTLASS MXFP4 /
#     Marlin W4A16, plain TP (mega_moe is NOT admitted on sm120).
#   Hopper (sm90): marlin, plain TP.
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: deepseek-ai/DeepSeek-V4.1-Flash-1017
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8977
#   TENSOR_PARALLEL_SIZE default: 4  (set 2 for TP=2; must match CUDA_DEVICES count)
#   CUDA_DEVICES         default: arch/TP-aware (0..TP-1); override to pin GPUs
#   MOE_BACKEND          default: arch-aware (mega_moe on sm100/sm103; auto on
#                        sm120/sm121; marlin on sm90). 'auto' omits the flag so
#                        the engine resolves it. Overriding to mega_moe forces
#                        expert parallel.
#   ENABLE_EXPERT_PARALLEL  default: 1 when MOE_BACKEND=mega_moe, else 0.
#                        Non-empty/1 = --enable-expert-parallel (ep_size=world).
#   ENABLE_DSPARK        default: 1 (--speculative-algorithm DSPARK). 0 to disable.
#   DTYPE                default: bfloat16
#   MAX_MODEL_LEN        default: 32768
#   MAX_TOTAL_TOKENS     default: 262144  (KV budget; caps admission)
#   MAX_NUM_SEQS         default: 32
#   CHUNKED_PREFILL_SIZE default: 8192
#   MAX_CUDAGRAPH_CAPTURE_SIZE  default: 32
#   GPU_MEMORY_UTILIZATION  default: empty (engine default). Set e.g. 0.9 if tight.
#   REASONING_PARSER     default: deepseek_v41. Fanned out to engine + gateway
#                        (the serve orchestrator splits it). Set empty to omit.
#   TOOL_CALL_PARSER     default: deepseek_v41 (gateway-only). Set empty to omit.
#   ENABLE_AUTO_TOOL_CHOICE default: empty (off). Non-empty = --enable-auto-tool-choice.
#                        This is a vLLM-shaped frontend flag; some SMG/tokenspeed
#                        serve builds reject it (tool calling already works via
#                        TOOL_CALL_PARSER). Leave off unless your build accepts it.
#   ENGRAM_HOST_TABLE    default: 1 on sm120/sm121 (RTX PRO 6000), else empty.
#                        1 = --engram-host-table: the two ~203 GB FP8 Engram tables
#                        go to host RAM (UVA), freeing HBM. Required to fit TP=4 on
#                        96 GiB cards. Set empty/0 to keep them GPU-sharded.
#   ENGRAM_HOST_TABLE_LAYOUT default: sharded (auto|shared|sharded; used when ENGRAM_HOST_TABLE on).
#   LANGUAGE_MODEL_ONLY  default: 1 on sm120/sm121, else empty. 1 = --language-model-only
#                        (drop the vision tower; text-only serving).
#   ALLOW_TP2_OOM        default: empty. 1 = bypass the TP=2 weight-fit guard and
#                        attempt TP=2 anyway (expected CUDA OOM on <128 GiB cards).
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_CUDA_ARCH override detected arch (e.g. sm100 / sm90 / 10.0 / 9.0)
#   VLLM_BATCH_INVARIANT default: empty. Truthy (1/true/yes/on) = deterministic
#                        comm preset: --force-deterministic-rsag +
#                        --disable-nccl-nvls (NCCL_NVLS_ENABLE=0). Needed on
#                        hosts with broken NVLS multicast.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#   EXTRA_SERVE_ARGS     extra serve args (space-separated), appended verbatim.
#
# Recommended invoke:
#   # TP=4 on GPUs 0-3
#   TENSOR_PARALLEL_SIZE=4 CUDA_DEVICES=0,1,2,3 ./scripts/deepseek-v4.1-flash-1017.sh
#   # TP=2 on GPUs 0-1
#   TENSOR_PARALLEL_SIZE=2 CUDA_DEVICES=0,1 ./scripts/deepseek-v4.1-flash-1017.sh
#
# Usage:
#   ./scripts/deepseek-v4.1-flash-1017.sh
#   TENSOR_PARALLEL_SIZE=2 ./scripts/deepseek-v4.1-flash-1017.sh
#   ENABLE_DSPARK=0 ./scripts/deepseek-v4.1-flash-1017.sh
#   MOE_BACKEND=marlin ./scripts/deepseek-v4.1-flash-1017.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# --- CUDA arch detection (prefer compute_cap; allow env override) ------------
# Normalize "10.0" / "sm_100" / "SM100" -> "sm100".
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

# MoE policy is arch-aware. The FlashInfer TRT-LLM vendor MoE (mega_moe +
# expert parallel) is validated only on datacenter Blackwell SM100-SM103
# (B200/GB200). Workstation Blackwell SM120 (RTX PRO 6000, compute_cap 12.0)
# is NOT in that window: deepseek_v41 leaves moe_backend 'auto' there, which
# resolves to the CUTLASS MXFP4 / Marlin W4A16 path. Hopper SM90 uses marlin.
#   _MOE_POLICY: datacenter_blackwell | auto | marlin
_MOE_POLICY=marlin
case "${_CUDA_ARCH}" in
    sm100|sm103) _MOE_POLICY=datacenter_blackwell ;;
    sm120|sm121) _MOE_POLICY=auto ;;
    sm90)        _MOE_POLICY=marlin ;;
    "")          _MOE_POLICY=auto ;;  # arch unknown: let the engine decide
esac

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8977}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
if [[ "${TENSOR_PARALLEL_SIZE}" != "2" && "${TENSOR_PARALLEL_SIZE}" != "4" ]]; then
    echo "ERROR: TENSOR_PARALLEL_SIZE='${TENSOR_PARALLEL_SIZE}' must be 2 or 4." >&2
    exit 1
fi

# Default device list is 0..TP-1 unless pinned via CUDA_DEVICES.
_default_devices() {
    local n="$1" out="" i
    for ((i = 0; i < n; i++)); do
        out+="${out:+,}$i"
    done
    printf '%s' "$out"
}
CUDA_DEVICES="${CUDA_DEVICES:-$(_default_devices "$TENSOR_PARALLEL_SIZE")}"

case "${_MOE_POLICY}" in
    datacenter_blackwell) MOE_BACKEND="${MOE_BACKEND:-mega_moe}" ;;
    auto)                 MOE_BACKEND="${MOE_BACKEND:-auto}" ;;
    *)                    MOE_BACKEND="${MOE_BACKEND:-marlin}" ;;
esac
# mega_moe needs expert parallelism; other backends run plain TP by default.
if [[ "${MOE_BACKEND}" == "mega_moe" ]]; then
    ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
else
    ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
fi

ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
REASONING_PARSER="${REASONING_PARSER-deepseek_v41}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-deepseek_v41}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-}"

# --- Weight/memory fit for the ~510 GB checkpoint (MEASURED footprint) --------
# The two FP8 Engram n-gram tables are ~203 GB; host-offloading them (UVA) frees
# that HBM. --language-model-only drops the vision tower (text-only serving).
# On sm120 (RTX PRO 6000, ~96 GiB) these are ON by default: they are what makes
# TP=4 fit (~77 GB/GPU on-GPU weight). Set empty to disable.
#   On-GPU weight that CANNOT be offloaded (no weight/expert CPU-offload exists):
#   ~307 GB total, dominated by ~296 GB MXFP4 experts -> ~77 GB/GPU at TP=4 (fits),
#   ~154 GB/GPU at TP=2 (does NOT fit 96 GiB; see deepseek-v4.1-flash-1017-research.md).
if [[ "${_CUDA_ARCH}" == "sm120" || "${_CUDA_ARCH}" == "sm121" ]]; then
    ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE-1}"
    LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY-1}"
else
    ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE-}"
    LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY-}"
fi
ENGRAM_HOST_TABLE_LAYOUT="${ENGRAM_HOST_TABLE_LAYOUT:-sharded}"

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
    echo "  Use e.g.: TENSOR_PARALLEL_SIZE=4 CUDA_DEVICES=0,1,2,3 $0" >&2
    echo "       or:  TENSOR_PARALLEL_SIZE=2 CUDA_DEVICES=0,1 $0" >&2
    exit 1
fi

# TP=2 weight-fit guard. The non-offloadable on-GPU weight (~307 GB, mostly MXFP4
# experts) needs ~154 GB/GPU at TP=2. Cards below ~128 GiB cannot hold it, and no
# weight/expert CPU-offload or sub-4-bit path exists to close the gap (quantization
# does NOT help: experts are already MXFP4). Refuse early with a clear message.
# Override with ALLOW_TP2_OOM=1 to attempt anyway (expected CUDA OOM at weight load).
if [[ "${TENSOR_PARALLEL_SIZE}" == "2" && "${ALLOW_TP2_OOM:-}" != "1" ]]; then
    _mem="${_GPU_MEM_MIB:-0}"
    if [[ "${_mem}" =~ ^[0-9]+$ && "${_mem}" -lt 131072 ]]; then
        echo "ERROR: TP=2 needs ~154 GB/GPU of non-offloadable weight (MXFP4 experts);" >&2
        echo "  this GPU has ~$(( _mem / 1024 )) GiB. It will OOM at weight load." >&2
        echo "  Engram host-table + language-model-only + quantization CANNOT close the gap" >&2
        echo "  (experts are already ~4-bit MXFP4). Use TP>=4, more GPUs, or a smaller" >&2
        echo "  checkpoint. See deepseek-v4.1-flash-1017-research.md. Set ALLOW_TP2_OOM=1" >&2
        echo "  to attempt anyway." >&2
        exit 1
    fi
fi

# DSpark replay requires KVStore off and prefix caching on (scheduler_utils).
if [[ "${ENABLE_DSPARK}" != "0" && "${ENABLE_DSPARK}" != "1" ]]; then
    echo "ERROR: ENABLE_DSPARK must be 0 or 1." >&2
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

echo "[deepseek-v4.1-flash-1017] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB moe_policy=${_MOE_POLICY} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} moe_backend=${MOE_BACKEND} expert_parallel=${ENABLE_EXPERT_PARALLEL} dspark=${ENABLE_DSPARK} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_NUM_SEQS=${MAX_NUM_SEQS} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default} reasoning=${REASONING_PARSER:-none} tool_parser=${TOOL_CALL_PARSER:-none} auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE:-off} engram_host_table=${ENGRAM_HOST_TABLE:-off} lm_only=${LANGUAGE_MODEL_ONLY:-off} model=${MODEL_PATH}" >&2

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --enable-prefix-caching
    --disable-kvstore
)
if [[ -n "$ENGRAM_HOST_TABLE" && "$ENGRAM_HOST_TABLE" != "0" ]]; then
    cmd+=(--engram-host-table --engram-host-table-layout "$ENGRAM_HOST_TABLE_LAYOUT")
fi
if [[ -n "$LANGUAGE_MODEL_ONLY" && "$LANGUAGE_MODEL_ONLY" != "0" ]]; then
    cmd+=(--language-model-only)
fi
if [[ -n "$MOE_BACKEND" && "$MOE_BACKEND" != "auto" ]]; then
    cmd+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$ENABLE_EXPERT_PARALLEL" && "$ENABLE_EXPERT_PARALLEL" != "0" ]]; then
    cmd+=(--enable-expert-parallel)
fi
if [[ "$ENABLE_DSPARK" == "1" ]]; then
    # Same-checkpoint draft; the block width comes from the checkpoint.
    cmd+=(--speculative-algorithm DSPARK)
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "$REASONING_PARSER" ]]; then
    cmd+=(--reasoning-parser "$REASONING_PARSER")
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

# Extra serve args (space-separated), e.g. EXTRA_SERVE_ARGS="--disable-autotune".
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi

exec "${cmd[@]}"
