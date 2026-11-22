#!/usr/bin/env bash
#
# gemma4.sh -- serve gemma-4-31B-it (bf16, TP=2) from a local venv (see install.sh)
# =====================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/google/gemma-4-31B-it
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8779 (the script exits if it is in use)
#   CUDA_DEVICES         default: 6,7
#   TENSOR_PARALLEL_SIZE default: 2 (must match CUDA_DEVICES count)
#   MAX_NUM_SEQS         default: 452
#   MAX_MODEL_LEN        default: 131072
#   CHUNKED_PREFILL_SIZE default: empty (engine default 8192). On SM120 with
#                        MAX_MODEL_LEN above 199000 it defaults to 1024, see
#                        "SM120 defaults" below.
#   PREFILL_GRAPH_MAX_TOKENS  default: 1024 on SM120, else empty (engine default).
#                        Largest prefill CUDA-graph bucket; see "SM120 defaults".
#   MAX_CUDAGRAPH_CAPTURE_SIZE  default: empty (ladder up to MAX_NUM_SEQS).
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: empty (omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   VENV_DIR             default: ./.venv
#   TOKENSPEED_TRITON_PREFILL_SKIP_OOR   default: 1 (F1; kernel default on)
#   TOKENSPEED_TRITON_DECODE_KV_SPLITS   default: unset (kernel default auto)
#   TOKENSPEED_FLASHINFER_FA2_EXTEND    default: unset (kernel default 512)
#   VLLM_BATCH_INVARIANT default: empty. Truthy (1/true/yes/on) = deterministic
#                        comm preset: --force-deterministic-rsag +
#                        --disable-nccl-nvls (NCCL_NVLS_ENABLE=0). Needed on
#                        hosts with broken NVLS multicast (e.g. sm90). The
#                        engine also honors it directly from the environment.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#   GPU_MEMORY_UTILIZATION default: empty (engine default). TP1 on one 141GB sm90
#                        card needs ~0.85 to fit the 31B weights + KV.
#   MM_ENCODER_TP_MODE   default: empty (engine default: weights). "data" runs
#                        the vision tower data-parallel across the TP ranks.
#
# SM120 defaults (RTX PRO 6000, compute capability 12.x of the first
# CUDA_DEVICES GPU; explicit env always wins, other archs are unchanged):
#   No attention/GEMM backend is forced: the engine picks the SM120 attention
#   leaves and the DeepGEMM SM120 GEMM leaves, and the CUDA graph ladder runs up
#   to MAX_NUM_SEQS (no capture cap is passed).
#   Prefill graphs stop at 1024 tokens (--prefill-graph-max-tokens 1024): the
#   8192-token bucket ladder reserves 2.3 GB of the 96 GB card (CUDA-graph
#   memory reserve, 56 buckets), which the KV pool gets back (19.9 -> 21.7 GB at
#   TP1; more concurrent 12000-char RAG requests fit). Longer chunks still
#   run, without a graph. RAG, 256 output tokens, MAX_NUM_SEQS=512, out tok/s at
#   C=1/16/64/256 (vLLM 0.32.0 in brackets):
#     TP1 23.1/264.7/364.8/360.6 (22.3/250.6/379.5/378.1); 339.7/344.7 at C=64/256
#         before this default;
#     TP2 42.9/479.8/931.4/1090.4 (41.2/444.1/869.1/975.2).
#   MAX_CUDAGRAPH_CAPTURE_SIZE=128 adds 4% at TP1 C=64/256 (379.9/377.5) by
#   shrinking the decode graph reserve, but decode batches above 128 then run
#   without a graph, so it is opt-in.
#   Long context: one 8192-token prefill chunk is reserved per sliding-window
#   group, so a 262144-token request does not fit the KV pool of one 96 GB GPU
#   (single-request limit 199614 at TP1). With MAX_MODEL_LEN above 199000 the
#   script passes --chunked-prefill-size 1024 (BF16 KV, no --kv-cache-dtype):
#   single-request limit 261886-262143 at TP1 (measured, tasks 52 and 84).
#   An explicit CHUNKED_PREFILL_SIZE or KV_CACHE_DTYPE (fp8 also reaches 262143)
#   disables this default.
#   /v1/completions: the gateway adds special tokens by the tokenizer's own
#   rules, like vLLM (task 87); no <bos> prefix is needed for gemma-3, and
#   gemma-4's tokenizer adds none for raw prompts.
#
# Recommended sm90 (H200) invoke:
#   CUDA_DEVICES=6,7 VLLM_BATCH_INVARIANT=1 ./scripts/gemma4.sh
#
# Multi-model (SMG IGW): override SERVED_MODEL_NAME to a unique id when
# registering beside other models; HOST_PORT already differs from gemma3.sh.
# See multi-model-gemma.md.
#
# Usage:
#   ./scripts/gemma4.sh
#   # SM120, TP1, 256K context (chunk 1024 is added automatically)
#   CUDA_DEVICES=6 TENSOR_PARALLEL_SIZE=1 MAX_MODEL_LEN=262144 GPU_MEMORY_UTILIZATION=0.9 ./scripts/gemma4.sh
#   CUDA_DEVICES=6,7 HOST_PORT=8779 ./scripts/gemma4.sh
#   ATTENTION_BACKEND=triton ./scripts/gemma4.sh
#   # multi-model: override served name when needed
#   SERVED_MODEL_NAME=gemma-4-31B-it ./scripts/gemma4.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/models/google/gemma-4-31B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
# gemma templates are not a Symphony "hf" table (the gateway would refuse
# them); passthrough returns text as generated, nctu6's proven config.
REASONING_PARSER="${REASONING_PARSER:-passthrough}"
HOST_PORT="${HOST_PORT:-8779}"
# Fail fast when HOST_PORT is taken (another server, possibly another user's).
if (exec 3<>"/dev/tcp/127.0.0.1/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: port ${HOST_PORT} is already in use. Set HOST_PORT to a free port, e.g. HOST_PORT=18778 $0" >&2
    exit 1
fi
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
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
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
PREFILL_GRAPH_MAX_TOKENS="${PREFILL_GRAPH_MAX_TOKENS:-}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
if [[ -z "$PREFILL_GRAPH_MAX_TOKENS" && "$_IS_SM120" == "1" ]]; then
    PREFILL_GRAPH_MAX_TOKENS=1024
fi
if [[ -z "$CHUNKED_PREFILL_SIZE" && -z "$KV_CACHE_DTYPE" && "$_IS_SM120" == "1" && "$MAX_MODEL_LEN" -gt 199000 ]]; then
    CHUNKED_PREFILL_SIZE=1024
fi
# Hosts with broken NVLS multicast need these; default off for other hosts.
FORCE_DETERMINISTIC_RSAG="${FORCE_DETERMINISTIC_RSAG:-}"
DISABLE_NCCL_NVLS="${DISABLE_NCCL_NVLS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-}"
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

echo "[gemma4] sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto} chunked_prefill=${CHUNKED_PREFILL_SIZE:-engine_default} prefill_graph_max_tokens=${PREFILL_GRAPH_MAX_TOKENS:-engine_default} cudagraph_cap=${MAX_CUDAGRAPH_CAPTURE_SIZE:-max_num_seqs}" >&2
cmd=(
    tokenspeed serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --reasoning-parser "$REASONING_PARSER"
    --host 0.0.0.0
    --port "$HOST_PORT"
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
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
if [[ -n "$CHUNKED_PREFILL_SIZE" ]]; then
    cmd+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
fi
if [[ -n "$PREFILL_GRAPH_MAX_TOKENS" ]]; then
    cmd+=(--prefill-graph-max-tokens "$PREFILL_GRAPH_MAX_TOKENS")
fi
if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
    cmd+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "$MM_ENCODER_TP_MODE" ]]; then
    cmd+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
fi
if [[ -n "$FORCE_DETERMINISTIC_RSAG" ]]; then
    cmd+=(--force-deterministic-rsag)
fi
if [[ -n "$DISABLE_NCCL_NVLS" ]]; then
    cmd+=(--disable-nccl-nvls)
    export NCCL_NVLS_ENABLE=0
fi

# Extra serve/gateway args (space-separated), e.g. EXTRA_SERVE_ARGS="--reasoning-parser passthrough".
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi

exec "${cmd[@]}"
