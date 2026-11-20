#!/bin/bash
# 停止所有 benchmark server（vLLM 容器 + 容器內 UnieInfra），等 GPU 記憶體釋放。
# pattern 一律用方括號，讓它匹配不到執行它的 bash -c 自己（否則 shell 會自殺、pkill -9 不會執行）。
source "$(dirname "$0")/env.sh"
docker rm -f bench-server >/dev/null 2>&1
docker exec -u root "$CONTAINER" bash -c '
  P="smg[_]grpc_servicer|smg[:][:]|tokenspeed[:][:]|tokenspeed[ ]serve|multiprocessing[.]spawn"
  pkill -f "$P"; sleep 6; pkill -9 -f "$P"; sleep 2
  stray=$(ps -eo pid,args --no-headers | grep -E "$P" | grep -v "grep -E" | awk "{print \$1}" | tr "\n" " ")
  [ -n "$stray" ] && { echo "stop_servers: leftover pid(s) $stray"; kill -9 $stray 2>/dev/null; sleep 3; }
  exit 0' >/dev/null 2>&1
for _ in $(seq 1 60); do
  (( $(vram_used) < 8000 )) && exit 0
  sleep 2
done
echo "WARNING: GPU memory still in use: $(vram_used) MB" >&2
exit 1
