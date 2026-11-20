#!/bin/bash
# 在 AgentX 壓力負載下錄 GPU trace（預設 GPU-only；CPU tracer 會放大 host 空隙）。
# 用法：[ACTIVITIES='["GPU"]'] [WARM=240] [WIN=6] prof_under_load.sh <CONC>
#   server 設定沿用呼叫者的環境（與 A 臂相同的 env + ENGINE_PYTHONPATH）
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/gpu_lock.sh"
C=${1:-32}; WARM=${WARM:-240}; WIN=${WIN:-6}; ACT=${ACTIVITIES:-'["GPU"]'}
TAG=$(date -u +%m%d%H%M); OUTD=prof-c$C-$TAG; L=$ROOT/logs/prof_under_load.log
gpu_lock_acquire "$(whoami)(prof c$C)"
STRESS=1 MAX_NUM_SEQS=${MAX_NUM_SEQS:-$((2*C))} "$ROOT/scripts/run_round.sh" PRF$C 300 unieinfra "$C" > "$ROOT/logs/prof.round.out" 2>&1 &
RR=$!
until grep -q "running agentx" "$ROOT/logs/prof.round.out" 2>/dev/null; do
  grep -q "FAILED TO START" "$ROOT/logs/prof.round.out" && { echo "server failed"; exit 1; }; sleep 5; done
sleep "$WARM"
curl -s -X POST "$SERVER_URL/start_profile" -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"$CONTAINER_LOG_DIR/$OUTD\", \"activities\": $ACT}" >> "$L"; echo >> "$L"
sleep "$WIN"; curl -s -X POST "$SERVER_URL/stop_profile" >> "$L"; echo >> "$L"
wait $RR
echo "PROF DONE c$C dir=$ROOT/logs/$OUTD" | tee -a "$L"
echo "next: python3 $ROOT/scripts/trace_exposed.py $ROOT/logs/$OUTD"
