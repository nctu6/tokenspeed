#!/bin/bash
# 對已啟動的 server 跑一格 AgentX。
# 用法：run_agentx.sh <engine-label> <CONC> [DURATION_s|official]
#   STRESS=1：--ignore-trace-delays（壓力模式；本專案的比較基準）
#   TAG=<tag>：寫進結果目錄名，A/B 解析靠它
set -eo pipefail
source "$(dirname "$0")/env.sh"
ENGINE=${1:?engine label}; CONC=${2:?concurrency}; DURATION=${3:-}
DURLABEL=${DURATION:+-d$DURATION}
[[ "$DURATION" == official ]] && { DURATION=""; DURLABEL=-dofficial; }
OUT=$ROOT/results/agentx/${ENGINE}${TAG:+-$TAG}-c${CONC}${DURLABEL}-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
EXTRA=()
if [[ -n "$DURATION" ]]; then
  EXTRA+=(--benchmark-duration "$DURATION" --warmup-requests-per-lane 2)
  (( DURATION < 900 )) && EXTRA+=(--unsafe-override)     # < 900 s 為非正式
fi
if [[ "${STRESS:-0}" == 1 ]]; then
  EXTRA+=(--ignore-trace-delays)
  [[ " ${EXTRA[*]} " == *" --unsafe-override "* ]] || EXTRA+=(--unsafe-override)
fi
echo "$OUT"
"$AGENTX_BIN" profile --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$MODEL_NAME" --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
  --concurrency "$CONC" --public-dataset "$AGENTX_DATASET" --random-seed 42 --ui-type simple \
  --output-artifact-dir "$OUT/aiperf_artifacts" "${EXTRA[@]}" 2>&1 | tee "$OUT/aiperf.log"
