# 03 基準與固定 benchmark 協定

## 1. 基準引擎（vLLM / SGLang）規則

- **只改啟動參數**，絕不改程式碼。每個改動在 `serve_vllm.sh` 中寫一行註解說明為什麼
  （例：「MI350P 144 GB，recipe 的 max-num-seqs 預設會讓 KV 只剩 2.27 GiB，改 2×CONC」）。
- 從廠商官方 recipe 出發（例：InferenceX `dsv41flash/vllm/mi355x-*.yaml`），copy 每個 concurrency
  的 override 區塊原文。
- 移除硬體不支援的旗標並註明（【MI350P】`--monokernel` 需要 256 CU）。
- 跑兩個 vLLM 系列：recipe 的 running-request cap 與 2×CONC cap。每格取 RPM 較高者為「vLLM 最佳」。
- 如果 vLLM 某格異常（例如 c1 ITL 36 ms 而健康時 6 ms），重跑；重現了就標記為「設定錯誤格」
  並排除，**不要**拿來當分母。
- SGLang：若明顯落後 vLLM，可以經使用者同意後只比 vLLM。

## 2. 公平性清單（每次報「vs vLLM」之前逐項確認）

- [ ] 同一個 checkpoint、同 dtype、同 TP、同機器。
- [ ] **取樣模式相同**：AgentX 請求帶 temperature 1.0。AMD 上 UnieInfra 預設 `--sampling-backend greedy`
      會忽略 temperature → 必須 `--sampling-backend triton_full`。【MI350P】greedy 時看起來 1.0–1.2×，
      改成真的取樣後是 0.82–0.91×。
- [ ] **雙方通訊都不量化**（`TOKENSPEED_IRIS_Q8_ALLREDUCE=0`，並在 `/proc/<pid>/environ` 驗證）。
- [ ] 同 speculative 設定（draft 數、adaptive verification）。從 server log 讀實際生效值，不要假設。
- [ ] 我方所有格用同一個 build、同一個 `ENGINE_PYTHONPATH`、同 `CHUNKED_PREFILL_SIZE`。
- [ ] cache 命中率只算 profiling 階段（warmup 是冷的）。
- [ ] 同 benchmark 模式（壓力 vs 官方）、同時長。

## 3. Benchmark 協定【通用；工作負載為 MI350P 專案所用】

### 3.1 兩種模式

| | 官方模式 | 壓力模式（本專案的比較基準） |
|---|---|---|
| think time | 保留 trace 中的間隔 | `--ignore-trace-delays`（`STRESS=1`） |
| 時長 | preset（3600 s） | 600 s |
| 用途 | 正式提交 | 量引擎本身的容量 |
| 問題 | client 58–63% 時間在等間隔，在途請求只有 4–6，壓不到引擎 | 非正式提交 |

使用者指定以**壓力模式**作比較。報告數字時寫清楚是哪一種。

### 3.2 單格協定

1. `stop_servers.sh`（每格重啟 server）。
2. `serve_<engine>.sh <CONC>`。
3. `wait_ready.sh`（真的生成 1 token；偵測啟動錯誤）。
4. `run_agentx.sh <engine> <CONC> 600`，`STRESS=1`，seed 42，固定 dataset。
5. 檢查 `profile_export_aiperf.json` 存在，否則格無效。
6. 停 server。

`run_round.sh <tag> 600 "<engines>" "<concs>"` 包好了以上全部，加 watchdog（DURATION + 5400 s）。

### 3.3 矩陣

concurrency：**1, 4, 16, 32, 64, 128**，每個引擎都要。
`MAX_NUM_SEQS = 2 × CONC`（AgentX lane 會分出 subagent）。

### 3.4 主要指標

- 標題指標：**RPM**（requests/min）相對「vLLM 最佳」的倍數。
- 同時記錄：output tok/s、ITL avg/p90、TTFT avg/p90、prefix 命中率、error rate、server KV token capacity。
- 解析：`python3 scripts/summarize_cells.py <tag-prefix>`。

### 3.5 無效格判定

以下任何一項成立就是無效格，要重跑：
- 沒有 export 檔（server 掛了、warmup 失敗）。
- error rate > 1% 且原因是引擎（不是 trace 本身的特殊請求）。
- 跑的時候有其他工作佔 GPU（查 GPU_LOCK 歷史與 `amd-smi`/`nvidia-smi` process）。
- 開了 `TOKENSPEED_KERNEL_VERBOSE` 或 profiler。
- server log 中有 `OUT_OF_RESOURCES` / OOM 警告數異常。

## 4. 便宜的篩選 benchmark（在上 AgentX 之前用）

| 名稱 | 量什麼 | 雜訊 |
|---|---|---|
| 固定 batch 長上下文 decode（例：16/32/64 users 共用 130K 前綴） | ms/token、tok/s/user | 低（~2%） |
| 單一 100K–155K prompt prefill ×3 | prefill 秒數 | 低 |
| suffix prefill（105K 快取前綴 + 1K/4K/8K 新 token） | TTFT | 低 |
| 兩引擎同 shape decode profile | 每步 ms 與逐 kernel 差 | 低 |

用它們淘汰沒希望的想法；只有通過的才上 600 s AgentX ABBA。
但要記得：篩選 bench 上的改善**經常**不會轉成 AgentX 吞吐（【MI350P】hold-results 在 trace 上 −8%，
AgentX 中性）。
