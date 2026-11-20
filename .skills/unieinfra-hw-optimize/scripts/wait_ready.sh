#!/bin/bash
# 等 server 真的能生成（1 token chat completion 成功）。/v1/models 有回應不代表 engine 活著。
# 用法：wait_ready.sh [timeout_s] [這次啟動的 server log]
# 給 log 時：log 出現 READY_RE 後才開始探測；出現 FATAL 立刻失敗（不要誤把下一格的 server 當 ready）。
source "$(dirname "$0")/env.sh"
T=${1:-1800}; LOG=${2:-}
FATAL=${FATAL_RE:-'Segmentation fault|HIP failure|CUDA error|startup failed|engine never reached SERVING|OUT_OF_RESOURCES'}
READY_RE=${READY_RE:-'engine ready: max_total_num_tokens'}
BODY="{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"
log_ready=0
for ((s=0; s<T; s+=10)); do
  if [[ -n $LOG ]]; then
    if grep -qE "$FATAL" "$LOG" 2>/dev/null; then
      echo "SERVER DIED after ${s}s: $(grep -E "$FATAL" "$LOG" | tail -1 | cut -c1-200)"; exit 1
    fi
    (( log_ready )) || { grep -qE "$READY_RE" "$LOG" 2>/dev/null && log_ready=1; }
    (( log_ready )) || { sleep 10; continue; }
  fi
  curl -sf -m 60 "$SERVER_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | grep -q '"choices"' && { echo "READY after ${s}s"; exit 0; }
  sleep 10
done
echo "NOT READY after ${T}s"; exit 1
