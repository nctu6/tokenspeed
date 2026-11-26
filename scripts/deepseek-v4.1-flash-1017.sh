#!/usr/bin/env bash
#
# deepseek-v4.1-flash-1017.sh -- serve DeepSeek-V4.1-Flash-1017 (TP=2 on SM12x, else TP=4)
# =============================================================================
#
# Single-node, aggregated serve of deepseek-ai/DeepSeek-V4.1-Flash-1017 from a
# local venv (see ./scripts/install.sh). For the two-node 1P1D disaggregated
# topology use test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh instead.
#
# DeepSeek V4.1 (deepseek_v41) runs on its FlatKV attention backend with a
# four-group KV cache. With no env set, an SM12x host (RTX PRO 6000, 96 GiB)
# serves on two GPUs (TP=2, CUDA_DEVICES=0,1) with the first 128 of the 384 routed
# experts, DSPARK on and `--moe-backend auto`; other archs default to TP=4 with
# all experts. The serve knobs below follow the recipe in
# docs/recipes/models.md ("DeepSeek V4.1-Flash"). Same-checkpoint DSpark
# speculative decoding is on by default (the CI/eval default for this
# checkpoint). It needs prefix caching on; V4.1 keeps the draft windows in
# its SWA cache group, so the Host KVStore (and L3 under it) works with
# DSpark too. The KVStore stays off unless ENABLE_KVSTORE=1.
#
# MoE backend is arch-aware:
#   Datacenter Blackwell (sm100/sm103, B200/GB200): mega_moe + expert parallel
#     (the FlashInfer TRT-LLM vendor MoE, validated only on this window).
#   Workstation Blackwell (sm120/sm121, RTX PRO 6000): auto, plain TP. The
#     engine resolves `auto` to the SM120 MegaMoE leaf (DeepGEMM MXFP4) when the
#     launch passes its gates (task 60: attention TP = MoE TP, MXFP4 experts, no
#     speculative decoding / PD / all-to-all / EPLB) and to CUTLASS MXFP4 /
#     Marlin W4A16 otherwise. DSPARK (ENABLE_DSPARK=1, the default here) is
#     speculative decoding, so with the defaults `auto` stays Marlin; use
#     ENABLE_DSPARK=0 to get MegaMoE (the engine logs `DeepSeek V4.1 default
#     moe_backend=...` with the reason). MegaMoE gain over Marlin, measured
#     without DSPARK on the 128-expert TP2 subset: TPOT -1% to -13%, input
#     tok/s +15% at C=128. DSPARK is still the faster default on SM120: RAG
#     12000-char prompts, 256 output tokens, TP2, MAX_NUM_SEQS=256, out tok/s at
#     C=1/16/64/256: DSPARK (Marlin) 276.9/1046.1/2197.7/3310.3 vs
#     ENABLE_DSPARK=0 (MegaMoE) 110.7/668.6/1567.7/3271.2 (+150%/+56%/+40%/+1%).
#   Hopper (sm90): marlin, plain TP.
#
# CUDA-graph memory with DSPARK: the decode graph scratch is about 11-13 MB per
# verified row per rank at MAX_MODEL_LEN=32768 (measured on SM120 TP2: 22 GB
# decode reserve at MAX_NUM_SEQS=512 with 4 rows per request, 6.6 GB for 512 rows
# without DSPARK). On a 96 GB GPU, MAX_NUM_SEQS=512 + DSPARK +
# GPU_MEMORY_UTILIZATION=0.9 leaves no KV budget and the engine stops at startup
# with "cache budget leaves -1.52 GiB". The reserve grows with MAX_NUM_SEQS, so
# lower it (the default 32 is far below the limit) or set
# MAX_CUDAGRAPH_CAPTURE_SIZE to cap the graph ladder.
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 when
#                        that directory exists, else deepseek-ai/DeepSeek-V4.1-Flash-1017
#   SERVED_MODEL_NAME    default: test. SPACE-separated for multiple names
#                        (e.g. "DeepSeek-V4.1-Flash-1017 test"); NOT comma-separated.
#   HOST_PORT            default: 8977 (the script exits if it is in use)
#   TENSOR_PARALLEL_SIZE default: the number of CUDA_DEVICES when that is set,
#                        else 2 on sm120/sm121 and 4 on other archs. Must be 2 or 4
#                        and match the CUDA_DEVICES count.
#   CUDA_DEVICES         default: 0..TP-1 (0,1 on sm12x); override to pin GPUs
#   MOE_BACKEND          default: arch-aware (mega_moe on sm100/sm103; auto on
#                        sm120/sm121; marlin on sm90). 'auto' omits the flag so
#                        the engine resolves it. Overriding to mega_moe forces
#                        expert parallel.
#   ENABLE_EXPERT_PARALLEL  default: 1 when MOE_BACKEND=mega_moe, else 0.
#                        Non-empty/1 = --enable-expert-parallel (ep_size=world).
#   ENABLE_DSPARK        default: 1 (--speculative-algorithm DSPARK). 0 to disable.
#   ENABLE_KVSTORE       default: 0 (--disable-kvstore). 1 keeps the Host KVStore
#                        (L2) on; add --kvstore-* / L3 flags via EXTRA_SERVE_ARGS.
#   DTYPE                default: bfloat16
#   MAX_MODEL_LEN        default: 32768
#   MAX_TOTAL_TOKENS     default: 262144  (KV budget; caps admission)
#   MAX_NUM_SEQS         default: 32
#   CHUNKED_PREFILL_SIZE default: 8192
#   MAX_CUDAGRAPH_CAPTURE_SIZE  default: 32 on sm90/sm100; empty on sm120/sm121
#                        (flag omitted: the default ladder runs up to MAX_NUM_SEQS)
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
#   LANGUAGE_MODEL_ONLY  default: empty = MULTIMODAL (text+image), the default.
#                        Set 1 for text-only (adds --language-model-only, drops
#                        the vision tower; slightly less memory). Multimodal TP=4
#                        is verified on sm120 (~85 GB/GPU; vision uses triton_attn).
#   ROUTED_EXPERTS       default: 128 at TP=2 on <128 GiB cards, else empty (all
#                        checkpoint experts). N = load only the first N routed
#                        experts per layer via --hf-overrides
#                        '{"n_routed_experts": N, "num_checkpoint_experts": <ckpt>}'
#                        (docs/recipes/models.md). The router keeps those N rows,
#                        so this is a different model than the checkpoint: use it
#                        for bring-up/numerics on 2 GPUs, not for quality.
#                        Set empty to load all experts.
#   ALLOW_TP2_OOM        default: empty. 1 = bypass the TP=2 weight-fit guard and
#                        attempt TP=2 with all experts anyway (expected CUDA OOM
#                        on <128 GiB cards).
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
#   # TP=2 on GPUs 0-1 is the SM12x default (96 GiB cards: partial MoE, first
#   # 128 of 384 experts), so a bare ./scripts/deepseek-v4.1-flash-1017.sh does it
#   # TP=4 on GPUs 0-3 (all 384 experts)
#   TENSOR_PARALLEL_SIZE=4 CUDA_DEVICES=0,1,2,3 ./scripts/deepseek-v4.1-flash-1017.sh
#   # TP=2 on other GPUs, e.g. 6 and 7 (TP follows the device count)
#   CUDA_DEVICES=6,7 ./scripts/deepseek-v4.1-flash-1017.sh
#
# Usage:
#   ./scripts/deepseek-v4.1-flash-1017.sh
#   TENSOR_PARALLEL_SIZE=4 ./scripts/deepseek-v4.1-flash-1017.sh
#   ENABLE_DSPARK=0 ./scripts/deepseek-v4.1-flash-1017.sh
#   MOE_BACKEND=marlin ./scripts/deepseek-v4.1-flash-1017.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# SM120 defaults (compute capability 12.x of the first CUDA_DEVICES GPU, here
# sm120 / sm121; explicit env always wins, other archs are unchanged): MOE_BACKEND
# auto (MegaMoE when admitted, see above), the 128 of 384 routed experts subset at
# TP=2 on <128 GiB cards, MODEL_PATH from /models, no CUDA graph capture cap
# (ladder up to MAX_NUM_SEQS), engine-default attention (V4.1 native indexer and
# SM120 attention leaves) and KV events at the engine default. MAX_NUM_SEQS 32 and
# MAX_TOTAL_TOKENS 262144 are the DSPARK / KV-budget recipe values, not sm90 tuning.

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
    sm12[0-9])   _MOE_POLICY=auto ;;
    sm90)        _MOE_POLICY=marlin ;;
    "")          _MOE_POLICY=auto ;;  # arch unknown: let the engine decide
esac

# Prefer the local checkpoint when it exists; otherwise use the hub id.
_LOCAL_MODEL=/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
if [[ -d "$_LOCAL_MODEL" ]]; then
    MODEL_PATH="${MODEL_PATH:-$_LOCAL_MODEL}"
else
    MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8977}"
# Fail fast when HOST_PORT is taken (another server, possibly another user's).
if (exec 3<>"/dev/tcp/127.0.0.1/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: port ${HOST_PORT} is already in use. Set HOST_PORT to a free port, e.g. HOST_PORT=18778 $0" >&2
    exit 1
fi

# TP follows an explicit CUDA_DEVICES; without one it is 2 on SM12x (96 GiB cards
# cannot hold the all-experts TP=2 weight, so the 128-expert subset below applies)
# and 4 elsewhere.
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
    if [[ -n "${CUDA_DEVICES:-}" ]]; then
        TENSOR_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<<"$CUDA_DEVICES")"
    elif [[ "${_CUDA_ARCH}" == sm12* ]]; then
        TENSOR_PARALLEL_SIZE=2
    else
        TENSOR_PARALLEL_SIZE=4
    fi
fi
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
ENABLE_KVSTORE="${ENABLE_KVSTORE:-0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
# SM120: no cap, so the engine's default ladder runs up to MAX_NUM_SEQS.
if [[ "${_CUDA_ARCH}" == sm12* ]]; then
    MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
else
    MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}"
fi
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
# Multimodal (text+image) is the DEFAULT. Verified on sm120 TP=4: the vision
# weights are tiny (~0.8 GB), the encoder uses the portable triton_attn backend,
# and it fits alongside the host-offloaded Engram tables (~85 GB/GPU).
# To serve TEXT-ONLY (slightly less memory), set LANGUAGE_MODEL_ONLY=1, which
# adds --language-model-only and drops the vision tower.
if [[ "${_CUDA_ARCH}" == sm12* ]]; then
    ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE-1}"
else
    ENGRAM_HOST_TABLE="${ENGRAM_HOST_TABLE-}"
fi
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY-}"   # empty = multimodal (default)
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
# experts) needs ~154 GB/GPU at TP=2 with every expert loaded. Cards below
# ~128 GiB cannot hold it, and quantization does not help (experts are already
# MXFP4). There the default is a partial MoE: only the first ROUTED_EXPERTS
# (128 of 384) routed experts per layer, ~1/3 of the expert bytes (~50 GB/GPU).
_tp2_small_gpu=0
if [[ "${TENSOR_PARALLEL_SIZE}" == "2" ]]; then
    _mem="${_GPU_MEM_MIB:-0}"
    if [[ "${_mem}" =~ ^[0-9]+$ && "${_mem}" -lt 131072 ]]; then
        _tp2_small_gpu=1
    fi
fi
if [[ "${_tp2_small_gpu}" == "1" && "${ALLOW_TP2_OOM:-}" != "1" ]]; then
    ROUTED_EXPERTS="${ROUTED_EXPERTS-128}"
else
    ROUTED_EXPERTS="${ROUTED_EXPERTS-}"
fi
# Routed experts the checkpoint holds (text_config.n_routed_experts); 384 for
# V4.1-Flash when the config cannot be read (e.g. MODEL_PATH is a hub id).
CHECKPOINT_EXPERTS="${CHECKPOINT_EXPERTS:-}"
if [[ -n "$ROUTED_EXPERTS" && -z "$CHECKPOINT_EXPERTS" ]]; then
    CHECKPOINT_EXPERTS="$(python3 - "$MODEL_PATH/config.json" <<'PY' 2>/dev/null || true
import json, sys
config = json.load(open(sys.argv[1]))
print(config.get("text_config", config)["n_routed_experts"])
PY
)"
    CHECKPOINT_EXPERTS="${CHECKPOINT_EXPERTS:-384}"
fi
if [[ -n "$ROUTED_EXPERTS" ]]; then
    if ! [[ "$ROUTED_EXPERTS" =~ ^[0-9]+$ && "$ROUTED_EXPERTS" -ge 1 \
            && "$ROUTED_EXPERTS" -le "$CHECKPOINT_EXPERTS" ]]; then
        echo "ERROR: ROUTED_EXPERTS='$ROUTED_EXPERTS' must be 1..${CHECKPOINT_EXPERTS}." >&2
        exit 1
    fi
elif [[ "${_tp2_small_gpu}" == "1" && "${ALLOW_TP2_OOM:-}" != "1" ]]; then
    echo "ERROR: TP=2 with all experts needs ~154 GB/GPU of non-offloadable weight;" >&2
    echo "  this GPU has ~$(( _mem / 1024 )) GiB. Keep the default ROUTED_EXPERTS=128" >&2
    echo "  (partial MoE), use TP=4, or set ALLOW_TP2_OOM=1 to attempt anyway." >&2
    exit 1
fi
if [[ "${ENABLE_DSPARK}" != "0" && "${ENABLE_DSPARK}" != "1" ]]; then
    echo "ERROR: ENABLE_DSPARK must be 0 or 1." >&2
    exit 1
fi
if [[ "${ENABLE_KVSTORE}" != "0" && "${ENABLE_KVSTORE}" != "1" ]]; then
    echo "ERROR: ENABLE_KVSTORE must be 0 or 1." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# Activate a venv if one is present; otherwise fall back to a tokenspeed that
# is already on PATH (the container / system-python case, e.g. the Docker
# image installs the stack system-wide with no .venv). Only error when neither
# a venv nor an on-PATH tokenspeed exists.
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif [[ -x "$VENV_DIR/bin/tokenspeed" ]]; then
    export PATH="$VENV_DIR/bin:$PATH"
elif command -v tokenspeed >/dev/null 2>&1; then
    echo "[deepseek-v4.1-flash-1017] no venv at $VENV_DIR; using tokenspeed on PATH: $(command -v tokenspeed)" >&2
else
    echo "ERROR: no venv at $VENV_DIR and no \"tokenspeed\" on PATH. Run ./scripts/install.sh, set VENV_DIR, or use an image with tokenspeed installed." >&2
    exit 1
fi

echo "[deepseek-v4.1-flash-1017] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB moe_policy=${_MOE_POLICY} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} moe_backend=${MOE_BACKEND} expert_parallel=${ENABLE_EXPERT_PARALLEL} dspark=${ENABLE_DSPARK} kvstore=${ENABLE_KVSTORE} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_NUM_SEQS=${MAX_NUM_SEQS} cudagraph_cap=${MAX_CUDAGRAPH_CAPTURE_SIZE:-max_num_seqs} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default} reasoning=${REASONING_PARSER:-none} tool_parser=${TOOL_CALL_PARSER:-none} auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE:-off} routed_experts=${ROUTED_EXPERTS:-all}${ROUTED_EXPERTS:+/${CHECKPOINT_EXPERTS}} engram_host_table=${ENGRAM_HOST_TABLE:-off} lm_only=${LANGUAGE_MODEL_ONLY:-off} model=${MODEL_PATH}" >&2

# --served-model-name is nargs="+": SPACE-separated names, each its own argv
# item (NOT comma-separated -- a comma becomes part of a single literal name).
# e.g. SERVED_MODEL_NAME="DeepSeek-V4.1-Flash-1017 test" advertises both.
# shellcheck disable=SC2206
_served_names=($SERVED_MODEL_NAME)

cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "${_served_names[@]}"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --enable-prefix-caching
)
if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
    cmd+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi
if [[ "$ENABLE_KVSTORE" == "0" ]]; then
    cmd+=(--disable-kvstore)
fi
if [[ -n "$ROUTED_EXPERTS" ]]; then
    cmd+=(--hf-overrides "{\"n_routed_experts\": ${ROUTED_EXPERTS}, \"num_checkpoint_experts\": ${CHECKPOINT_EXPERTS}}")
fi
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
