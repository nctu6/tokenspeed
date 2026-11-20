#!/bin/bash
# GSM8K 正確性關卡（與 repo AMD CI 相同設定：greedy、max_tokens 512；DSv4.1 門檻 0.90）。
# 用法：run_gsm8k.sh <label>
set -eo pipefail
source "$(dirname "$0")/env.sh"
OUT=$ROOT/results/gsm8k/${1:?label}-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; cd "$OUT"
"$EVALSCOPE_BIN" eval --model "$MODEL_NAME" --api-url "$SERVER_URL/v1" --api-key EMPTY_TOKEN \
  --datasets gsm8k --dataset-hub huggingface --dataset-args '{"gsm8k":{"dataset_id":"openai/gsm8k"}}' \
  --eval-batch-size 32 --generation-config '{"do_sample":false,"temperature":0.0,"max_tokens":512}' \
  --work-dir "$OUT" 2>&1 | tee "$OUT/eval.log"
