#!/bin/bash
# vLLM 基準 server。規則：只改「啟動參數」，絕不改 vLLM 程式碼；每個偏離 recipe 的地方寫註解說明原因。
# 用法：serve_vllm.sh <CONC>
# 以下 per-concurrency 參數是【MI350P / InferenceX dsv41flash mi355x recipe TP4】的實例：換硬體時從該硬體官方 recipe 重抄。
set -euo pipefail
source "$(dirname "$0")/env.sh"
CONC=${1:?usage: serve_vllm.sh <CONC>}
IMAGE=${VLLM_IMAGE:?set VLLM_IMAGE to the vendor image pinned by digest}
NAME=bench-server
case "$CONC" in
  1|2)  PER_CONC=(--max-cudagraph-capture-size 8 --max-num-batched-tokens 32768 --max-num-seqs 1) ;;
  # 【MI350P】recipe 在 c4/c8 用預設 max-num-seqs，144 GB 上 KV 只剩 2.27 GiB（< 一個 1M 請求）→ 2×CONC
  4|8)  PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 16384 --max-num-seqs $(( 2 * CONC ))) ;;
  16)   PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs 4) ;;
  32)   PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs 8) ;;
  # recipe 只到 c32（CONC/4）；c64/c128 沿用同比例
  64|128|256) PER_CONC=(--max-cudagraph-capture-size 1024 --max-num-batched-tokens 8192 --max-num-seqs $(( CONC / 4 ))) ;;
  *) echo "no recipe knobs for CONC=$CONC" >&2; exit 1 ;;
esac
# 第二個基準系列：VLLM_MAX_NUM_SEQS=$((2*CONC))；每格取兩系列中 RPM 較高者為「vLLM 最佳」
[[ -n "${VLLM_MAX_NUM_SEQS:-}" ]] && PER_CONC+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
SPEC_CFG=${VLLM_SPEC_CFG:-'{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'}
"$(dirname "$0")/stop_servers.sh" || true
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --network host --ipc host --shm-size 64g --ulimit core=0 \
  -v "$MODEL_DIR_HOST":/models:ro -v "$ROOT/logs":/logs \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  ${VLLM_EXTRA_ENV:-} \
  --entrypoint vllm "$IMAGE" serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$TP" --speculative-config "$SPEC_CFG" \
  "${PER_CONC[@]}" ${VLLM_EXTRA_ARGS:-}
# VLLM_EXTRA_ENV 例【MI350P】："-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE -e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
# VLLM_EXTRA_ARGS 例【MI350P】：recipe 其餘旗標；移除 --monokernel（需要 256 CU，MI350P 只有 128）
echo "started $NAME (vllm, CONC=$CONC) on port $PORT"
