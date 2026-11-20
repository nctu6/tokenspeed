#!/bin/bash
# 所有腳本共用的設定。換機器/換模型時只改這個檔。
# 用法：在其他腳本開頭 `source "$(dirname "$0")/env.sh"`。
export ROOT=${ROOT:-/path/to/project}                 # 專案根目錄（含 logs/ results/ experiments.json）
export UNIEINFRA=${UNIEINFRA:-$ROOT/unieinfra}        # UnieInfra repo（host 路徑）
export CONTAINER=${CONTAINER:-ts-dev}                 # 跑 UnieInfra 的長駐容器；repo 掛在 /workspace
export CONTAINER_LOG_DIR=${CONTAINER_LOG_DIR:-/logs}  # 容器內看到的 $ROOT/logs
export MODEL_DIR_HOST=${MODEL_DIR_HOST:-/data/models} # host 上的模型目錄（放 NVMe）
export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4.1-Flash-1017}         # 容器內路徑
export MODEL_NAME=${MODEL_NAME:-deepseek-ai/DeepSeek-V4.1-Flash-1017}     # served model name
export TOKENIZER=${TOKENIZER:-$MODEL_DIR_HOST/DeepSeek-V4.1-Flash-1017}
export LAUNCHER=${LAUNCHER:-./scripts/deepseek-v4.1-flash-1017.sh}         # repo 內建模型啟動腳本（相對 /workspace）
export TP=${TP:-4}
export GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
export PORT=${PORT:-8000}
export SERVER_URL=${SERVER_URL:-http://127.0.0.1:$PORT}
export AGENTX_BIN=${AGENTX_BIN:-$ROOT/.venv-agentx/bin/agentx}
export AGENTX_DATASET=${AGENTX_DATASET:-semianalysis_cc_traces_weka_062126}
export EVALSCOPE_BIN=${EVALSCOPE_BIN:-$ROOT/.venv-eval/bin/evalscope}
export LOCK=${LOCK:-$ROOT/logs/GPU_LOCK}
export VENDOR=${VENDOR:-amd}                          # amd | nvidia
mkdir -p "$ROOT/logs" "$ROOT/results"

# 所有 GPU 已用記憶體總和（MB）
vram_used() {
  if [[ $VENDOR == amd ]]; then
    amd-smi metric --mem-usage 2>/dev/null | awk '/USED_VRAM/ {s+=$2} END {print s+0}'
  else
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'
  fi
}
