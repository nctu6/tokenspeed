#!/bin/bash
# 範例：用目前最佳設定啟動一格 UnieInfra（【MI350P】的最佳 env；換硬體時逐項重新驗證）。
# 不要直接拿去跑 benchmark —— benchmark 一律走 run_round.sh / ab_abba.sh（會取 GPU 鎖、重啟、檢查）。
source scripts/gpu_lock.sh
gpu_lock_acquire "$(whoami)(manual c32)"
CONC=32
env STRESS=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-base/python \
  TOKENSPEED_PREFILL_TBO=1 \
  TOKENSPEED_DSPARK_BLOCK_VERIFY=1 \
  TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  TOKENSPEED_CACHE_AWARE_ADMISSION=1 \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*CONC)) \
  MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  MODEL_EXTRA_ENV="-e ENGRAM_HOST_TABLE=1 -e LANGUAGE_MODEL_ONLY=1 -e MAX_MODEL_LEN=1048576" \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/serve_unieinfra.sh $CONC
scripts/wait_ready.sh 3000 "$ROOT/logs/unieinfra-server.log"
python3 scripts/needle_test.py --cases 6 --tokens 30000
scripts/stop_servers.sh
