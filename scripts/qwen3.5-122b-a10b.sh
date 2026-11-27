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
#   MOE_BACKEND          arch-aware (auto on sm120: the engine selects the
#                        Triton block-FP8 MoE there for online FP8 and the FP4
#                        leaves for NVFP4; empty = engine auto on the bf16 H200
#                        / sm90 path; auto for an NVFP4 checkpoint)
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
#   ENABLE_FP8_ATTN_PROJ default: 1 on sm120 with online FP8 (QUANTIZATION=fp8),
#                        else empty. Non-empty and not 0 = --enable-fp8-attn-proj
#                        (block-FP8 attention and GDN projections); 0 = off.
#   ENABLE_MTP           default: auto = MTP speculative decoding when the checkpoint
#                        ships an MTP layer (config mtp_num_hidden_layers >= 1
#                        and mtp.* tensors in the weight index or safetensors
#                        headers; the model name is not used), for the BF16,
#                        FP8 and NVFP4 checkpoints alike. 1 = force on (error if
#                        there is no MTP layer), 0 = off. A --speculative-*
#                        flag in EXTRA_SERVE_ARGS replaces it.
#   MTP_NUM_STEPS        default: 3 when MAX_NUM_SEQS <= 32, else 2 (draft tokens
#                        per step; the draft model is MODEL_PATH itself).
#                        RAG out tok/s on two SM120 GPUs, greedy, NVFP4 with
#                        this script's own defaults (MAX_NUM_SEQS=64), C=1/16/32:
#                        steps=2 235/1031/1392, steps=3 247/1083/1448. With
#                        MAX_NUM_SEQS=512 (32768 context, GPU_MEMORY_UTILIZATION
#                        =0.9), C=1/16/64, FP8 off 167/691/1227, steps=2
#                        254/838/1591; NVFP4 off 134/847/1569, steps=2
#                        231/1039/1871. At C=256 MTP gives FP8 2114 (off 2122)
#                        and NVFP4 2176 (off 2551; vLLM MTP loses 3 % there): for
#                        an NVFP4 server held at C>~160 set ENABLE_MTP=0. The
#                        accuracy probe (400 items x 2) is equal within its
#                        +-0.01 noise. The decode CUDA-graph ladder keeps the
#                        engine default (it reaches MAX_NUM_SEQS): capping it at
#                        128 as qwen3.6-35b-a3b.sh does gives this model on two
#                        GPUs no throughput (NVFP4 MTP C=256 2167/2192 capped,
#                        2181/2171 not), its cache pool is not the limit.
#                        Measured on SM120 only.
#   MIXED_PREFILL_TOKEN_BUDGET  default: 4096 when MTP is on (else unused).
#                        > 0 = --enable-mixed-batch --mixed-prefill-token-budget
#                        N: prompt chunks of up to N tokens run in the same
#                        round as the MTP verify windows instead of separate
#                        prefill rounds; 0 = off. An explicit mixed-batch flag
#                        in EXTRA_SERVE_ARGS replaces it. RAG out tok/s on two
#                        SM120 GPUs, MAX_NUM_SEQS=512 (32768 context,
#                        GPU_MEMORY_UTILIZATION=0.9), steps=2, C=1/16/64/256,
#                        2 runs: FP8 off 254/840/1570/2110, 4096
#                        254/892/1768/2332; NVFP4 off 236/1032/1874/2164, 4096
#                        235/1089/2066/2272. Measured on SM120 only.
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
#     Online --quantization fp8 (MoE backend auto = Triton block-FP8 on sm120):
#       MoE weights ~59.6 GB/GPU; total ~86 GiB used at max_model_len=8192.
#     BF16 at TP=2 OOMs (~114 GiB weights/GPU). The FlashInfer TRT-LLM FP8 MoE has
#     no sm12x kernel (it crashed with a hidden_states_scale shape error); the
#     kernel registry no longer offers it there, so auto skips it.
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
#   - BF16 checkpoint: --quantization fp8 (BF16 OOMs at ~114 GiB/GPU on 2 x 96 GB).
#     Online FP8 changes the numerics, so it stays an explicit choice of this
#     script; the MoE backend is the engine auto choice (Triton block-FP8).
#   - Online FP8: --enable-fp8-attn-proj (block-FP8 QKV/O and GDN projections) is
#     on. 2 x 96 GB, out tok/s C=1/16/64/256: 168.1/665.9/1178.8/1971.6 vs
#     146.3/634.2/1141.1/1902.1 without (+14.9/+5.0/+3.3/+3.7 %). 400 multi-step
#     word problems, two greedy runs each: accuracy 0.9375/0.9375 without vs
#     0.935/0.9375 with (two runs of one config differ in 1 of 400 answers).
#     `ENABLE_FP8_ATTN_PROJ=0` turns it off.
#   - MAX_MODEL_LEN 8192 / MAX_NUM_SEQS 64 / KV fp8 are memory limits (~86 GiB
#     used at 8192 context), not sm90 leftovers (32k KV OOMs, see above).
#
# Recommended invoke:
#   # sm90 (H200) — NVLS multicast broken → batch-invariant preset
#   CUDA_DEVICES=6,7 VLLM_BATCH_INVARIANT=1 ./scripts/qwen3.5-122b-a10b.sh
#   # sm120 (PRO 6000) — FP8 TP=2 on free GPUs 6,7
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
# sm120 (Blackwell / PRO 6000) → online FP8 + TP=2 on 6,7 (MoE backend auto).
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
    # GPUs 6,7). The engine picks the Triton block-FP8 MoE with auto.
    CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    if [[ "${_NVFP4}" -eq 1 ]]; then
        # Pre-quantized NVFP4: the triton MoE has no NVFP4 kernel.
        QUANTIZATION="${QUANTIZATION:-nvfp4}"
        MOE_BACKEND="${MOE_BACKEND:-auto}"
    else
        QUANTIZATION="${QUANTIZATION:-fp8}"
        MOE_BACKEND="${MOE_BACKEND:-auto}"
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
if [[ "${_IS_SM120}" -eq 1 && "${QUANTIZATION}" == "fp8" ]]; then
    ENABLE_FP8_ATTN_PROJ="${ENABLE_FP8_ATTN_PROJ:-1}"
else
    ENABLE_FP8_ATTN_PROJ="${ENABLE_FP8_ATTN_PROJ:-}"
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
    echo "    CUDA_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 QUANTIZATION=fp8 $0" >&2
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
if [[ -n "$ENABLE_FP8_ATTN_PROJ" && "$ENABLE_FP8_ATTN_PROJ" != "0" ]]; then
    cmd+=(--enable-fp8-attn-proj)
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
    echo "[qwen3.5-122b] MTP speculative decoding on (steps=${MTP_NUM_STEPS}; ENABLE_MTP=0 turns it off)" >&2
    # Mixed prefill+decode rounds: prompt chunks of up to
    # MIXED_PREFILL_TOKEN_BUDGET tokens ride beside the MTP verify windows
    # instead of taking rounds of their own (the engine keeps mixed rounds
    # opt-in under speculative decoding). 0 = off; an explicit mixed-batch
    # flag in EXTRA_SERVE_ARGS replaces this default.
    MIXED_PREFILL_TOKEN_BUDGET="${MIXED_PREFILL_TOKEN_BUDGET:-4096}"
    if [[ "$MIXED_PREFILL_TOKEN_BUDGET" -gt 0 && "${EXTRA_SERVE_ARGS:-}" != *-mixed-* ]]; then
        cmd+=(--enable-mixed-batch --mixed-prefill-token-budget "$MIXED_PREFILL_TOKEN_BUDGET")
        echo "[qwen3.5-122b] mixed prefill+decode rounds on (prompt budget ${MIXED_PREFILL_TOKEN_BUDGET} tokens; MIXED_PREFILL_TOKEN_BUDGET=0 turns it off)" >&2
    fi
fi
# Extra serve args (space-separated), appended verbatim, e.g. EXTRA_SERVE_ARGS="--chunked-prefill-size 4096".
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    _extra=($EXTRA_SERVE_ARGS)
    cmd+=("${_extra[@]}")
fi
exec "${cmd[@]}"
