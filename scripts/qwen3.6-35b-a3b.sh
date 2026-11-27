#!/usr/bin/env bash
#
# qwen3.6-35b-a3b.sh -- serve Qwen3.6-35B-A3B from a local venv
# =============================================================================
#
# Matched baseline knobs (override via env):
#   MODEL_PATH           default: /models/Qwen/Qwen3.6-35B-A3B
#   SERVED_MODEL_NAME    default: test
#   HOST_PORT            default: 8778 (the script exits if it is in use)
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
#   PREFILL_GRAPH_MAX_TOKENS  default: empty (engine default; the engine caps it
#                        at 1024 on SM120, see "SM120 defaults"). Set to override.
#   DTYPE                default: bfloat16
#   KV_CACHE_DTYPE       default: fp8 (unset/empty = omit flag)
#   ATTENTION_BACKEND    default: empty (engine auto). Set e.g. triton to force.
#   ENABLE_EXPERT_PARALLEL  default: empty. Non-empty = --enable-expert-parallel
#                           (ep_size = stage world). Same weight bytes as TP-only
#                           at the same world size; optional MoE routing layout.
#   REASONING_PARSER     default: qwen3. The chat template has an
#                           enable_thinking toggle (default on), so the gateway
#                           arms the parser only when thinking is effectively
#                           on; enable_thinking=false then returns the answer in
#                           content. qwen3_thinking is always-in-reasoning and
#                           would put a thinking-off answer in reasoning_content.
#   ENABLE_AUTO_TOOL_CHOICE default: empty. Non-empty = --enable-auto-tool-choice
#                           (current SMG builds do not accept this flag; leave
#                           off — tool calling is enabled via TOOL_CALL_PARSER)
#   TOOL_CALL_PARSER     default: qwen_coder (alias of qwen_xml; Qwen3.6 XML
#                           <tool_call>/<function=...> format. Omit by setting
#                           empty)
#   MAX_CUDAGRAPH_CAPTURE_SIZE  default: 128 when MAX_NUM_SEQS > 128, else empty (engine
#                        default: the ladder reaches MAX_NUM_SEQS). Set but empty
#                        = engine default. The decode graphs are priced out of
#                        the KV pool; MAX_NUM_SEQS=512 with MTP: 533 pages and
#                        RAG 2169/2147 out tok/s at C=128/256, with cap 128 792
#                        pages and 2767/2726 (cap 192: C=256 2578); MTP off
#                        C=256 2287 -> 2324; 256 concurrent short requests (128
#                        in, 256 out) MTP 2184 -> 2602, off 3424 -> 3549. No
#                        change at C<=64 (the same graphs). The two-GPU model
#                        of qwen3.5-122b-a10b.sh gains nothing from it and
#                        keeps the engine ladder.
#   ENABLE_MTP           default: auto = MTP speculative decoding when the checkpoint
#                        ships an MTP layer (config mtp_num_hidden_layers >= 1
#                        and mtp.* tensors in the weight index or safetensors
#                        headers; the model name is not used). 1 = force on
#                        (error if there is no MTP layer), 0 = off. A
#                        --speculative-* flag in EXTRA_SERVE_ARGS replaces it.
#   MTP_NUM_STEPS        default: 3 when MAX_NUM_SEQS <= 32, else 2 (draft tokens
#                        per step; the draft model is MODEL_PATH itself).
#                        RAG out tok/s on one SM120 GPU, greedy. This script's
#                        own defaults (MAX_NUM_SEQS=32), C=1/8/16/32: MTP off
#                        170/546/731/977, steps=2 256/687/949/1359, steps=3
#                        265/718/1009/1445. MAX_NUM_SEQS=512 (32768 context,
#                        GPU_MEMORY_UTILIZATION=0.9), C=1/16/32/48/64/128/256:
#                        off 169/717/-/-/1388/1999/2285; steps=2
#                        256/946/1341/1704/1999/2169/2151; steps=3
#                        264/1008/1450/1777/1695/-/-. Steps=3 beats steps=2 up
#                        to C=48 and loses 15 % at C=64 (queued prompts wait
#                        seconds: TTFT p50 2.7 s vs 0.2 s), so the default
#                        follows the decode batch bound. With the graph cap
#                        above MTP wins at every concurrency measured (C=256
#                        2726 vs off 2324, TTFT p50 12 s vs 2.9 s). Without the
#                        cap it wins up to about C=128 and loses 6-7 % at
#                        C=256 (cache pool 533 pages instead of 751, about 84
#                        requests in flight instead of 210; vLLM MTP loses 10
#                        % there). The accuracy probe (400 items x 2) is
#                        equal within its +-0.01 noise. Measured on SM120 only.
#   MIXED_PREFILL_TOKEN_BUDGET  default: 4096 when MTP is on (else unused).
#                        > 0 = --enable-mixed-batch --mixed-prefill-token-budget
#                        N: prompt chunks of up to N tokens run in the same
#                        round as the MTP verify windows instead of separate
#                        prefill rounds; 0 = off. An explicit mixed-batch flag
#                        in EXTRA_SERVE_ARGS replaces it. RAG out tok/s on one
#                        SM120 GPU, MAX_NUM_SEQS=512 (32768 context,
#                        GPU_MEMORY_UTILIZATION=0.9), steps=2, C=1/16/64/256,
#                        2 runs: off 256/938/1987/2714, 4096 257/1011/2281/3187;
#                        2048 257/1010/2248/3099, 8192 256/1014/2278/3225 (1
#                        run). vLLM 0.34 MTP 233/998/2020/2023. Measured on
#                        SM120 only.
#   EXTRA_SERVE_ARGS     default: empty; appended verbatim to `tokenspeed serve`.
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
#                        multi-GPU hosts with broken NVLS multicast (e.g.
#                        some sm90 hosts). Harmless no-op for single-GPU TP=1. The
#                        engine also honors it directly from the environment.
#   FORCE_DETERMINISTIC_RSAG / DISABLE_NCCL_NVLS
#                        legacy per-knob switches (non-empty = on); still work.
#
# Arch → defaults (CUDA compute capability of first selected GPU):
#   sm90  (H200 ~141 GiB):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1
#     MAX_MODEL_LEN=8192  MAX_NUM_SEQS=32
#     bf16 weights ~67 GiB on one GPU at TP=1 — fits H200. No --quantization.
#     MoE backend = engine auto (no forced triton).
#   sm120 (PRO 6000 ~96 GiB, Blackwell):
#     CUDA_DEVICES=6  TENSOR_PARALLEL_SIZE=1
#     MAX_MODEL_LEN=8192  MAX_NUM_SEQS=32
#     bf16 weights ~67 GiB on one GPU at TP=1 — fits ~96 GiB without online FP8
#     if context/seqs stay modest (cudagraph + KV need the leftover ~25 GiB).
#     MoE backend = engine auto (no forced triton / FP8).
#   Override CUDA_DEVICES / TENSOR_PARALLEL_SIZE / QUANTIZATION / MOE_BACKEND /
#   TOKENSPEED_CUDA_ARCH to pin.
#
# SM120 defaults (compute capability 12.x of the first CUDA_DEVICES GPU; explicit
# env always wins, other archs are unchanged): MOE_BACKEND, QUANTIZATION and
# ATTENTION_BACKEND stay on the engine's auto choice (SM120 leaves), the
# TOKENSPEED_TRITON_DECODE_KV_SPLITS / FA2_EXTEND pins are not exported, and no
# CUDA graph capture cap is passed (the ladder runs up to MAX_NUM_SEQS).
# MAX_MODEL_LEN 8192, MAX_NUM_SEQS 32, KV fp8 and GPU_MEMORY_UTILIZATION 0.85 are
# memory limits of the ~67 GiB bf16 weights on one 96 GB GPU, not sm90 leftovers.
# The engine caps the prefill graphs at 1024 tokens on SM120
# (--prefill-graph-max-tokens, logged as "auto: prefill_graph_max_tokens=1024",
# docs/configuration/server.md); the KV pool gets the memory of the larger
# buckets back and longer chunks run without a graph. RAG, 256 output tokens, MAX_NUM_SEQS=512, MAX_MODEL_LEN=32768,
# GPU_MEMORY_UTILIZATION=0.9, out tok/s at C=16/64/256: 726.5/1369.5/2236.2 and
# 725.5/1371.5/2233.1 (repeat) vs 726.1-726.5/1368.1-1370.0/2124.6-2149.6
# without it (vLLM 0.32.0 699.2/1385.8/2205.3).
#
# Recommended invoke:
#   # sm90 (H200) — single-GPU TP=1 on GPU 6
#   CUDA_DEVICES=6 ./scripts/qwen3.6-35b-a3b.sh
#   # sm120 (PRO 6000) — bf16 TP=1 on free GPU 6
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

# MTP speculative decoding: on when the checkpoint ships an MTP layer. The
# check reads the files, not the model name: config.json declares
# mtp_num_hidden_layers >= 1 (top level or text_config) and the weight index (or
# the safetensors headers when there is no index) holds mtp.* tensors.
_checkpoint_has_mtp() {
    python3 - "$1" <<'PY'
import glob
import json
import os
import struct
import sys
root = sys.argv[1]
config_path = os.path.join(root, "config.json")
if not os.path.isfile(config_path):
    sys.exit(1)
with open(config_path) as f:
    config = json.load(f)
layers = max(int(c.get("mtp_num_hidden_layers") or 0) for c in (config, config.get("text_config") or {}))
if layers < 1:
    sys.exit(1)
index_path = os.path.join(root, "model.safetensors.index.json")
if os.path.isfile(index_path):
    with open(index_path) as f:
        names = list(json.load(f)["weight_map"])
else:
    names = []
    for path in glob.glob(os.path.join(root, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            names += json.loads(f.read(header_len))
sys.exit(0 if any(n.startswith("mtp.") for n in names) else 1)
PY
}
# Default GPUs first: the arch below is read from the first selected device.
CUDA_DEVICES="${CUDA_DEVICES:-6}"
_CUDA_ARCH_RAW="${TOKENSPEED_CUDA_ARCH:-$(_detect_cuda_arch)}"
_CUDA_ARCH="$(_normalize_cuda_arch "${_CUDA_ARCH_RAW}")"
_GPU_MEM_MIB="$(_detect_gpu_mem_mib)"

# sm120 (Blackwell / PRO 6000) and sm90 (H200): bf16 TP=1 on GPU 6 — no forced FP8.
_IS_SM120=0
if [[ "${_CUDA_ARCH}" == sm12* ]]; then
    _IS_SM120=1
fi

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"
HOST_PORT="${HOST_PORT:-8778}"
# Fail fast when HOST_PORT is taken (another server, possibly another user's).
if (exec 3<>"/dev/tcp/127.0.0.1/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: port ${HOST_PORT} is already in use. Set HOST_PORT to a free port, e.g. HOST_PORT=18778 $0" >&2
    exit 1
fi
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
PREFILL_GRAPH_MAX_TOKENS="${PREFILL_GRAPH_MAX_TOKENS:-}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
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

echo "[qwen3.6-35b] arch=${_CUDA_ARCH:-unknown} gpu_mem=${_GPU_MEM_MIB:-unknown}MiB sm120=${_IS_SM120} CUDA_DEVICES=${CUDA_DEVICES} TP=${TENSOR_PARALLEL_SIZE} quant=${QUANTIZATION:-none} moe_backend=${MOE_BACKEND:-auto} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV=${KV_CACHE_DTYPE:-auto} gpu_mem_util=${GPU_MEMORY_UTILIZATION:-engine_default} prefill_graph_max_tokens=${PREFILL_GRAPH_MAX_TOKENS:-engine_default} reasoning=${REASONING_PARSER:-none} tool_parser=${TOOL_CALL_PARSER:-none}" >&2

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
if [[ -n "$PREFILL_GRAPH_MAX_TOKENS" ]]; then
    cmd+=(--prefill-graph-max-tokens "$PREFILL_GRAPH_MAX_TOKENS")
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

# ENABLE_MTP: auto (default) = on when the checkpoint has MTP weights, 1 = on
# (error when it has none), 0 = off. MTP_NUM_STEPS is the draft length; its
# default follows the decode batch bound: 3 when MAX_NUM_SEQS <= 32 (the wider
# verify step wins at that concurrency), else 2. Any --speculative-* flag in
# EXTRA_SERVE_ARGS replaces this default (explicit wins).
ENABLE_MTP="${ENABLE_MTP:-auto}"
if [[ "$MAX_NUM_SEQS" -le 32 ]]; then
    MTP_NUM_STEPS="${MTP_NUM_STEPS:-3}"
else
    MTP_NUM_STEPS="${MTP_NUM_STEPS:-2}"
fi
_mtp=0
case "$ENABLE_MTP" in
    auto)
        if [[ "${EXTRA_SERVE_ARGS:-}" != *--speculative* ]] && _checkpoint_has_mtp "$MODEL_PATH"; then
            _mtp=1
        fi
        ;;
    1)
        if ! _checkpoint_has_mtp "$MODEL_PATH"; then
            echo "ERROR: ENABLE_MTP=1 but $MODEL_PATH has no MTP layer (mtp_num_hidden_layers / mtp.* weights)." >&2
            exit 1
        fi
        if [[ "${EXTRA_SERVE_ARGS:-}" == *--speculative* ]]; then
            echo "ERROR: ENABLE_MTP=1 conflicts with --speculative-* in EXTRA_SERVE_ARGS; set only one." >&2
            exit 1
        fi
        _mtp=1
        ;;
    0) ;;
    *)
        echo "ERROR: ENABLE_MTP must be auto, 1 or 0 (got '$ENABLE_MTP')." >&2
        exit 1
        ;;
esac
if [[ "$_mtp" -eq 1 ]]; then
    cmd+=(--speculative-algorithm MTP --speculative-num-steps "$MTP_NUM_STEPS" --speculative-draft-model-path "$MODEL_PATH")
    echo "[qwen3.6-35b] MTP speculative decoding on (steps=${MTP_NUM_STEPS}; ENABLE_MTP=0 turns it off)" >&2
    # Mixed prefill+decode rounds: prompt chunks of up to
    # MIXED_PREFILL_TOKEN_BUDGET tokens ride beside the MTP verify windows
    # instead of taking rounds of their own (the engine keeps mixed rounds
    # opt-in under speculative decoding). 0 = off; an explicit mixed-batch
    # flag in EXTRA_SERVE_ARGS replaces this default.
    MIXED_PREFILL_TOKEN_BUDGET="${MIXED_PREFILL_TOKEN_BUDGET:-4096}"
    if [[ "$MIXED_PREFILL_TOKEN_BUDGET" -gt 0 && "${EXTRA_SERVE_ARGS:-}" != *-mixed-* ]]; then
        cmd+=(--enable-mixed-batch --mixed-prefill-token-budget "$MIXED_PREFILL_TOKEN_BUDGET")
        echo "[qwen3.6-35b] mixed prefill+decode rounds on (prompt budget ${MIXED_PREFILL_TOKEN_BUDGET} tokens; MIXED_PREFILL_TOKEN_BUDGET=0 turns it off)" >&2
    fi
fi
# Decode CUDA-graph ladder cap. The engine's default ladder reaches MAX_NUM_SEQS
# and its graphs are priced out of the KV pool (with MTP the verify and draft
# graphs cost about 3 GB at 512). A decode step above 128 requests runs 50 ms or
# more here, which hides the eager launch cost, so graphs above 128 only take
# cache pages: the default cap is 128 when MAX_NUM_SEQS > 128. An explicit
# MAX_CUDAGRAPH_CAPTURE_SIZE wins; set but empty = the engine default.
if [[ -z "${MAX_CUDAGRAPH_CAPTURE_SIZE+x}" ]]; then
    if [[ "$MAX_NUM_SEQS" -gt 128 ]]; then
        MAX_CUDAGRAPH_CAPTURE_SIZE=128
    else
        MAX_CUDAGRAPH_CAPTURE_SIZE=""
    fi
fi
if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
    cmd+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi
# Extra serve args (space-separated), appended verbatim, e.g. EXTRA_SERVE_ARGS="--chunked-prefill-size 4096".
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi
exec "${cmd[@]}"
