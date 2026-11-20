#!/bin/bash
# A B B A 交錯 A/B 範本。複製成 ab_<name>.sh 後只改「設定區」。
# 每格：AgentX 壓力 600 s。兩臂除了 ARM_*_ENV 以外完全相同。
# 結果：results/agentx/unieinfra-<TAG><A|B><conc><rep>-c<conc>-d600-*；解析：summarize_cells.py <TAG>A <TAG>B
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/gpu_lock.sh"
# ======== 設定區 ========
TAG=${TAG:-AB}                                  # 結果標籤前綴（短、唯一）
CONCS=${CONCS:-"32 64 16"}                      # 先跑最可能有效的 concurrency
DUR=${DUR:-600}
COMMON_ENV="STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 \
TOKENSPEED_DSPARK_COUPLED_DRAFT=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
CHUNKED_PREFILL_SIZE=8192 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86"
COMMON_ARGS="--sampling-backend triton_full --enable-log-request-stats"
ARM_A_ENV="ENGINE_PYTHONPATH=/workspace/.wt-base/python"                    # A = 目前最佳（基準 commit）
ARM_B_ENV="ENGINE_PYTHONPATH=/workspace/.wt-new/python TOKENSPEED_MY_FEATURE=1"  # B = 只多一個變因
# ========================
M=$ROOT/logs/ab_$TAG.log
gpu_lock_acquire "$(whoami)(ab_$TAG)"
run() {  # arm conc rep
  local arm=$1 c=$2 r=$3 extra
  [[ $arm == A ]] && extra=$ARM_A_ENV || extra=$ARM_B_ENV
  echo "$(date -u +%T) start $arm c$c r$r" >> "$M"
  env $COMMON_ENV $extra MAX_NUM_SEQS=$((2*c)) EXTRA_SERVE_ARGS="$COMMON_ARGS" \
    "$ROOT/scripts/run_round.sh" "$TAG$arm$c$r" "$DUR" unieinfra "$c"
  # 每格檢查崩潰：OUT_OF_RESOURCES 的格不能用來判定
  local slog; slog=$(readlink -f "$ROOT/logs/unieinfra-server.log")
  grep -q OUT_OF_RESOURCES "$slog" && echo "$(date -u +%T) WARNING $arm c$c r$r OUT_OF_RESOURCES in $slog" >> "$M"
  echo "$(date -u +%T) done $arm c$c r$r" >> "$M"
}
for c in $CONCS; do run A "$c" 1; run B "$c" 1; run B "$c" 2; run A "$c" 2; done
echo "AB DONE $(date -u +%FT%TZ)" >> "$M"
python3 "$ROOT/scripts/summarize_cells.py" "${TAG}A" "${TAG}B" >> "$M"
