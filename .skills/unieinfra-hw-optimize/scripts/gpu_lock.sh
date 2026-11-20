#!/bin/bash
# GPU 鎖（noclobber 原子建立）。用法：
#   source scripts/gpu_lock.sh
#   gpu_lock_acquire "me(ab_foo c32)"   # 會一直等到拿到
#   ...GPU 工作...
#   （trap 會在結束時自動釋放；也可手動 gpu_lock_release）
# 規則：不要刪別人的鎖；等太久就回報使用者鎖的內容。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gpu_lock_acquire() {
  local owner=${1:?owner(purpose)} waited=0
  until ( set -o noclobber; echo "owner=$owner since=$(date -u +%FT%TZ) pid=$$" > "$LOCK" ) 2>/dev/null; do
    (( waited % 600 == 0 )) && echo "gpu_lock: waiting, held by: $(cat "$LOCK" 2>/dev/null)" >&2
    sleep 15; waited=$((waited + 15))
  done
  trap 'gpu_lock_release' EXIT
}
gpu_lock_release() {
  # 只刪自己的鎖
  grep -q "pid=$$" "$LOCK" 2>/dev/null && rm -f "$LOCK"
}
