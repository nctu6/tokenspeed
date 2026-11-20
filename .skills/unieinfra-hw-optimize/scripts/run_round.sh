#!/bin/bash
# 一輪 benchmark：每個 engine × concurrency 重啟 server → wait_ready → AgentX → 停止。
# 用法：run_round.sh <tag> <duration_s|official> "<engines>" "<concs>"
#   例：STRESS=1 run_round.sh base 600 "unieinfra vllm" "1 4 16 32 64 128"
# 注意：本腳本不取 GPU 鎖；由呼叫者（ab_abba.sh 等）取。
set -uo pipefail
source "$(dirname "$0")/env.sh"
TAG=${1:?tag}; DURATION=${2:?duration}; ENGINES=${3:?engines}; CONCS=${4:?concs}
LOG=$ROOT/logs/round-$TAG.log
[[ -f $ROOT/logs/skip-$TAG ]] && { echo "SKIP $TAG" | tee -a "$LOG"; exit 1; }   # 用旗標檔跳過，不要改正在執行的 queue
for engine in $ENGINES; do
  for conc in $CONCS; do
    echo "=== $(date -u +%FT%TZ) $engine c$conc: starting server" | tee -a "$LOG"
    "$ROOT/scripts/stop_servers.sh"
    if ! "$ROOT/scripts/serve_$engine.sh" "$conc" >> "$LOG" 2>&1; then
      echo "=== $engine c$conc: SERVER FAILED TO START" | tee -a "$LOG"; continue
    fi
    ready_log=""; [[ $engine == unieinfra ]] && ready_log=$ROOT/logs/unieinfra-server.log
    if ! "$ROOT/scripts/wait_ready.sh" 3000 "$ready_log" >> "$LOG" 2>&1; then
      echo "=== $engine c$conc: SERVER FAILED TO START" | tee -a "$LOG"
      cp "$ROOT/logs/unieinfra-server.log" "$ROOT/logs/fail-$TAG-$engine-c$conc.log" 2>/dev/null; continue
    fi
    budget=14400; [[ "$DURATION" != official ]] && budget=$(( DURATION + 5400 ))   # watchdog：server 卡死時不會無限等
    run_log=$ROOT/logs/agentx-$TAG-$engine-c$conc.log
    echo "=== $(date -u +%FT%TZ) $engine c$conc: running agentx $DURATION" | tee -a "$LOG"
    timeout -k 60 "$budget" env TAG=$TAG "$ROOT/scripts/run_agentx.sh" "$engine" "$conc" "$DURATION" > "$run_log" 2>&1
    rc=$?; out_dir=$(head -1 "$run_log")
    [[ -f "$out_dir/aiperf_artifacts/profile_export_aiperf.json" ]] || echo "=== $engine c$conc: NO EXPORT (rc=$rc) -- cell INVALID" | tee -a "$LOG"
    echo "=== $(date -u +%FT%TZ) $engine c$conc: done (rc=$rc) -> $out_dir" | tee -a "$LOG"
  done
done
"$ROOT/scripts/stop_servers.sh"
echo "=== $(date -u +%FT%TZ) ROUND $TAG COMPLETE" | tee -a "$LOG"
