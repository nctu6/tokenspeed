#!/bin/bash
# 在長駐容器內啟動 UnieInfra（包 repo 內建的模型啟動腳本 $LAUNCHER）。
# 用法：serve_unieinfra.sh <CONC>
#   實驗開關用環境變數給：TOKENSPEED_* / TS_EXP_* / HIP_* / HSA_* / GPU_* / DEBUG_* / AMD_* / NCCL_* / RCCL_*
#   開頭的變數會「自動」傳進容器（避免「加了 env 卻忘了加 -e」的陷阱）。
#   額外 serve 參數：EXTRA_SERVE_ARGS="--sampling-backend triton_full ..."
#   實驗程式碼：ENGINE_PYTHONPATH=/workspace/.wt-<name>/python:...（每個腳本都要明確設定！）
set -euo pipefail
source "$(dirname "$0")/env.sh"
CONC=${1:?usage: serve_unieinfra.sh <CONC>}
LOG_NAME=unieinfra-server-c$CONC-$(date -u +%Y%m%dT%H%M%SZ).log   # 每次啟動一份 log，不覆蓋崩潰的 log

# 1) 停掉舊的 server，等記憶體釋放；釋放不了就拒絕啟動（否則 RCCL init OOM）
"$(dirname "$0")/stop_servers.sh" || { echo "serve_unieinfra: GPUs still hold $(vram_used) MB, refusing to start" >&2; exit 1; }

# 2) 依 concurrency 的預設
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$(( 2 * CONC ))}                    # AgentX lane 會分出 subagent
SPEC_ROWS=${SPEC_ROWS:-6}                                         # 每請求每步 verify 列數（draft+1）
CAPTURE=$(( MAX_NUM_SEQS * SPEC_ROWS )); (( CAPTURE > 512 )) && CAPTURE=512   # graph 不要吃掉 KV
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-1073741824}                  # 2^30 = 由記憶體預算決定容量
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.86}            # 【MI350P】0.9 會 OUT_OF_RESOURCES
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
ENGINE_PYTHONPATH=${ENGINE_PYTHONPATH:?set ENGINE_PYTHONPATH explicitly (e.g. /workspace/python)}

# 3) 安全的預設：有損通訊關閉；scratch 不回收；kernarg 放 device
export TOKENSPEED_IRIS_Q8_ALLREDUCE=${TOKENSPEED_IRIS_Q8_ALLREDUCE:-0}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}

# 4) 自動傳遞實驗 env（只傳有值的；用 VAR=value 形式，避免 docker 刪掉映像內建值）
ENV_ARGS=()
while IFS='=' read -r k v; do
  [[ $k =~ ^(TOKENSPEED_|TS_EXP_|HIP_|HSA_|GPU_MAX|DEBUG_|AMD_LOG|NCCL_|RCCL_|MOE_BACKEND|ENABLE_EXPERT_PARALLEL|ATTN_DP_SIZE) ]] && ENV_ARGS+=(-e "$k=$v")
done < <(env)

docker exec -d -u root \
  -e PYTHONPATH="$ENGINE_PYTHONPATH" \
  -e MODEL_PATH="$MODEL_PATH" -e SERVED_MODEL_NAME="$MODEL_NAME" \
  -e HOST_PORT="$PORT" -e TENSOR_PARALLEL_SIZE="$TP" -e CUDA_DEVICES="$GPU_DEVICES" \
  -e MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" -e MAX_CUDAGRAPH_CAPTURE_SIZE="$CAPTURE" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" \
  -e HF_HUB_OFFLINE=1 -e PYTHONFAULTHANDLER=1 \
  ${MODEL_EXTRA_ENV:-} \
  "${ENV_ARGS[@]}" \
  "$CONTAINER" bash -c "ulimit -c 0; cd /workspace && exec $LAUNCHER > $CONTAINER_LOG_DIR/$LOG_NAME 2>&1"
# MODEL_EXTRA_ENV：模型/硬體特有的固定 env，例如【MI350P】"-e ENGRAM_HOST_TABLE=1 -e LANGUAGE_MODEL_ONLY=1 -e MAX_MODEL_LEN=1048576"
ln -sfn "$LOG_NAME" "$ROOT/logs/unieinfra-server.log"
echo "started unieinfra (CONC=$CONC) log: $ROOT/logs/$LOG_NAME"
echo "verify env inside: docker exec $CONTAINER bash -c 'tr \"\\0\" \"\\n\" < /proc/\$(pgrep -f \"tokenspeed[:][:]scheduler\" | head -1)/environ | grep TOKENSPEED_'"
