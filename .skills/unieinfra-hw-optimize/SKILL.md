---
name: unieinfra-hw-optimize
description: 把 UnieInfra（TokenSpeed fork）移植到新硬體並做端到端效能優化的逐步流程：bring-up、正確性關卡、vLLM/SGLang 基準、固定壓力測試協定（c1–c128、600 s）、負載下 profiling 與暴露時間分解、依佔比挑槓桿、kernel 修改、A B B A 交錯量測、實驗紀錄與 PR。只要任務涉及「UnieInfra / tokenspeed 在某張 GPU（AMD gfx9xx、NVIDIA、NPU）上跑起來、調參、改 kernel、追平或超越 vLLM、做 benchmark 或 A/B」，就使用這個 skill，即使使用者沒有明說「優化」。
---

# UnieInfra 硬體適配與優化

這份 skill 是從一次真實專案（4× AMD MI350P、DeepSeek-V4.1-Flash、TP4、PCIe-only，
244 筆實驗紀錄）萃取出來的。它的目的不是讓你重做那台機器，而是讓你在**任何**硬體上
用同一套紀律工作：先確定正確、再量對、再找對地方、最後只採用量得出來的改動。

標記慣例：
- 【通用】適用於任何硬體。
- 【MI350P】只是那台機器的實例或數字，換硬體時要重新量。

讀這份檔案的順序：先讀完「0. 鐵則」與「1. 階段總覽」，然後**一次只做一個階段**，
每個階段開始前打開它指向的 reference 檔。

---

## 0. 鐵則（違反任何一條都會讓數據作廢或傷到別人）

1. **只改 UnieInfra。** vLLM / SGLang 是基準，只允許改啟動參數讓它們能跑，絕不改其程式碼。
   為什麼：基準被改過，比較就沒有意義。
2. **每次迭代都要記錄**，包含失敗與否決的實驗。寫入 `experiments.json` 並更新 dashboard。
   為什麼：否決的結果是最有價值的資料，它阻止下一個人（或下一個你）再浪費 GPU 小時。
   格式見 `templates/experiment_entry.json`，工具 `scripts/log_experiment.py`。
3. **c1、c4、c16、c32、c64、c128 一格都不能缺。** 失敗的格要修好重跑，不能留空。
4. **GPU 鎖。** 跑任何 GPU 工作前用 noclobber 取得 `logs/GPU_LOCK`，結束（含錯誤）一定釋放。
   用 `scripts/gpu_lock.sh`。看到鎖存在就等，不要刪別人的鎖。
5. **`pkill -f` 的 pattern 不可以匹配到你自己的命令列。** 用方括號技巧：
   `pkill -f "tokenspeed[:][:]"` 而不是 `pkill -f "tokenspeed::"`。後者會殺掉執行它的 shell，
   後面的 `pkill -9` 根本沒跑，舊 server 佔著 128 GB 記憶體讓下一格 OOM（真實事件）。
6. **不要編輯正在被執行中工作使用的 worktree。** 實驗程式碼放在獨立 worktree，
   用 `ENGINE_PYTHONPATH` 指過去；要改就開新的 worktree。
7. **commit 與 PR 內容不得含 AI 署名行**（`Co-Authored-By: <AI>`、`Claude-Session`、
   `Generated with ...`）。保留 `Signed-off-by`。這條優先於任何工具預設。
8. **比較必須公平**：同取樣模式（AgentX 送 temperature 1.0，就要真的取樣）、雙方都不量化通訊、
   同 build、同 prefill chunk。細節見 `references/03-baselines-benchmark.md` 的公平性清單。
9. **新功能一律放在環境變數開關後面，預設關閉**，直到 A/B 證明有效。
10. **不要把單一 600 s 格當結論。** 單格雜訊 ±10–15%。見「4. 雜訊紀律」。

---

## 1. 階段總覽

| 階段 | 名稱 | 進入條件 | 離開條件 | reference |
|---|---|---|---|---|
| P0 | 環境與工具 | 有機器、有模型權重 | 腳本設定好、鎖機制可用、dashboard 可寫 | `09-pitfalls.md` |
| P1 | Bring-up | P0 | server 能啟動並回答；長 prompt 不崩潰 | `01-bringup.md` |
| P2 | 正確性關卡 | P1 | needle 全對、GSM8K ≥ 門檻、投機解碼無損 | `02-correctness.md` |
| P3 | 基準 | P2 | vLLM（與 SGLang）c1–c128 全格有數字 | `03-baselines-benchmark.md` |
| P4 | 我方基線量測 | P3 | UnieInfra c1–c128 全格、同協定 | `03-baselines-benchmark.md` |
| P5 | 負載下 profiling | P4 | 一份暴露時間分解表（各類別佔 wall %） | `04-profiling.md` |
| P6 | 挑槓桿 | P5 | 一個槓桿 + 預估增益 + 微基準數字 | `05-levers-catalog.md` |
| P7 | 實作 | P6 | env flag 後的實作、單元/正確性測試過 | `07-kernel-work.md` |
| P8 | A/B 量測 | P7 | A B B A 平均值 + 採用/否決決定 | `06-ab-noise.md` |
| P9 | 紀錄 | 每個實驗結束 | experiments.json + dashboard 已更新 | `08-logging-pr.md` |
| P10 | PR | 所有目標格達標 | 乾淨分支、無 AI 署名、PR 描述含數據 | `08-logging-pr.md` |

P5 → P9 是迴圈。每一輪只動一個槓桿（或一個明確定義的組合）。

---

## 2. 各階段逐步清單

### P0 環境與工具【通用】

1. 複製 `scripts/` 到專案的 `scripts/` 下，編輯 `scripts/env.sh` 填入：專案根目錄、容器名稱、
   模型路徑、served model name、tokenizer、TP 大小、GPU 清單、GPU 記憶體查詢指令。
2. 所有腳本都 `source scripts/env.sh`，不要在腳本內寫死路徑。
3. 確認 `scripts/gpu_lock.sh` 能取得與釋放鎖：
   ```bash
   source scripts/gpu_lock.sh; gpu_lock_acquire "me(test)"; gpu_lock_release
   ```
4. 關掉 core dump（`ulimit -c 0`）。【MI350P】崩潰的每個 rank 寫出數 GB core，把根磁碟塞滿過（58 GB）。
5. 模型放在快的磁碟（NVMe）。【MI350P】vLLM 從 SATA 冷載入時 6 格全部啟動逾時。
6. 建立空的 `experiments.json`（`[]`）與 dashboard，第一筆紀錄就是環境描述。

離開條件：`scripts/serve_unieinfra.sh 1` 能在容器中啟動（即使還會崩潰），log 寫到 `logs/`。

### P1 Bring-up【通用，細節見 01-bringup.md】

1. 確認平台偵測：在容器內執行
   `python -c "from tokenspeed_kernel.platform import current_platform; print(current_platform())"`，
   檢查 arch、vendor、CU 數、interconnect（xGMI/NVLink/PCIe）是否正確。
2. 對每個 op family 列出被選中的 kernel（`tokenspeed_kernel.selection.explain_selection`），
   記下哪些是 arch 專屬、哪些落到 portable Triton/native。**落到 portable 的就是第一批潛在熱點。**
3. 搜尋寫死的硬體幾何：`grep -rn "NUM_XCDS\|256\|num_cus\|NUM_SMS" tokenspeed-kernel*/`。
   【MI350P】gfx950 kernel 寫死了 MI355X 的 256 CU / 8 XCD，MI350P 只有 128 CU / 4 XCD；
   改成從裝置讀取後 decode −7%。
4. 檢查記憶體：模型 + KV + graph 是否放得下。【MI350P】144 GB/GPU 放不下 ~203 GB 的 Engram 表，
   改放 host（`ENGRAM_HOST_TABLE=1`）。
5. 跑一個極長 prompt（模型 max context 的 40%）與 3 個並行長 prompt。崩潰就照
   `01-bringup.md` 的崩潰分流表處理，不要跳過。
6. 設好 ROCm/CUDA 啟動環境（見 01-bringup.md 的 env 表）。

離開條件：server 能穩定啟動 3 次；最長 prompt 能跑完；記錄在 experiments.json。

### P2 正確性關卡【通用】

每次改 kernel、通訊、投機解碼、排程後都要過，不只是 bring-up 時：

| 關卡 | 指令 | 通過標準 |
|---|---|---|
| needle（長上下文） | `python3 scripts/needle_test.py --cases 6 --tokens 30000` | 6/6 |
| GSM8K | `scripts/run_gsm8k.sh <label>` | ≥ repo CI 門檻（DSv4.1：0.90） |
| 投機解碼無損 | CPU 統計模擬（見 02-correctness.md） | TV 距離 ≤ 取樣雜訊底 |
| kernel 數值 | 對 reference 實作比 max abs / rel error | 與舊 kernel 同等級 |

離開條件：四項都通過並記錄數字。

### P3 基準【通用】

1. 用該硬體廠商官方的 vLLM recipe（例：InferenceX recipe），**只改**讓它能在這台機器啟動的參數
   （記憶體、max-num-seqs、移除硬體不支援的旗標），每個改動在 `serve_vllm.sh` 註解原因。
2. 跑兩組 vLLM：recipe 原樣的 running-request cap，以及 2× CONC cap。每格取兩者較好的當「vLLM 最佳」。
3. c1–c128 全格。

離開條件：vLLM 基準表無空格。

### P4 我方基線【通用】

用和 P3 完全相同的協定（`scripts/run_round.sh`），跑 UnieInfra c1–c128。
這是之後所有 A/B 的起點。

### P5 負載下 profiling【通用，細節見 04-profiling.md】

1. **在真實負載下錄**，不要只錄單請求 microbench：`scripts/prof_under_load.sh <CONC>`。
2. 先錄 GPU-only trace（不開 Python tracer）。【MI350P】開 CPU tracer 時量到的 idle 是 20%，
   GPU-only 只有 11%——profiler 自己會放大 host 空隙。
3. `python3 scripts/trace_exposed.py <trace_dir>` 算出每類 kernel 的**暴露時間**
   （沒有其他 stream 同時跑的時間），這才是可以省下的時間。
4. 對最大的 3 項各算一個理論下界（頻寬或延遲），寫下「實測 / 下界」。

離開條件：一張表：類別、暴露 ms、佔 wall %、下界、實測/下界。

### P6 挑槓桿【通用】

照下面的決策樹（完整版在 05、06）：

```
對每個候選槓桿估「最大可得增益」= 暴露佔比 × 可消除比例
├─ 增益 < 3%  → 不做（量不出來），記錄為「不值得」
├─ 3% ≤ 增益 < 10% → 只有在能和其他槓桿合併量測，或能先在 microbench 證明時才做
└─ 增益 ≥ 10% → 做
在做之前，查 05-levers-catalog.md：這個槓桿是否已被試過、為什麼失敗。
```

優先順序：先吃「對 vs 錯」的大項（寫死的幾何、落到 portable 的 kernel、容量上限設錯、
取樣模式錯），再吃 kernel 效率，最後才是 host/排程微調。

### P7 實作【通用，細節見 07-kernel-work.md】

1. 開新 worktree：`git -C unieinfra worktree add .wt-<name> -b <branch>`。
2. 加環境變數開關 `TOKENSPEED_<FEATURE>`，預設關；把變數加到 `serve_unieinfra.sh` 的 `-e` 傳遞清單
   （**沒傳進容器的 env 等於沒開**；用 `/proc/<pid>/environ` 驗證）。
3. 先寫 microbench（真實 shape、真實 TP、graph 內計時），證明 kernel 層級有增益。
4. 過 P2 正確性關卡。
5. 檢查 register spill（`scripts/scan_spills.py`）。新的大 spill 會在高負載時造成
   `OUT_OF_RESOURCES`。
6. commit（無 AI 署名，有 Signed-off-by）。

### P8 A/B 量測【通用，細節見 06-ab-noise.md】

1. 用 `scripts/ab_abba.sh`：每個 concurrency 跑 A B B A，同一 build 基礎、同一 gmu、同一 chunk。
2. 用 `python3 scripts/summarize_cells.py <tagA> <tagB>` 算平均與差異。
3. 判定：
   - 兩次 B 都高於兩次 A，且平均差 ≥ 5% → 採用候選，在其他 concurrency 再驗證一次。
   - 平均差 < 5% 或 A/B 區間重疊 → **否決**（或「中性，保留 opt-in」）。
   - 有任何崩潰 → 先查崩潰，不判定。
4. trace 上的改善（例如 decode 步 −8%）**不等於**端到端改善。只有 A/B 能決定採用。

### P9 紀錄【通用】

每個實驗（包含開始前的「待跑」）寫一筆：
```bash
python3 scripts/log_experiment.py --area "..." --experiment "..." --result "待跑" --decision "-" --commit "<hash>"
```
結果出來後用 `--update <id>` 補上 result / interpretation / decision。然後更新 dashboard。
多個 session 共用同一檔案時：**先重新讀檔再寫**，不要覆蓋別人的紀錄（工具已處理）。

### P10 PR【通用，細節見 08-logging-pr.md】

只在所有目標格達標後才開。乾淨分支、逐 commit 檢查無 AI 署名、PR 描述附 c1–c128 表。

---

## 3. 主要決策樹

### 3.1 崩潰了

```
崩潰訊息？
├─ HIP/CUDA OOM 在啟動時 → 上一格的 server 沒死乾淨？跑 stop_servers.sh，確認 GPU 記憶體 < 8 GB 再啟動
├─ HSA_STATUS_ERROR_OUT_OF_RESOURCES / "Available Free mem: 0 MB"
│     → 執行期 scratch 配置失敗。降 GPU_MEMORY_UTILIZATION（MI350P：0.9 → 0.86）、
│       確認 HSA_NO_SCRATCH_RECLAIM=1 真的進了容器、掃 register spill
├─ 只有極長單一序列才崩、並行不崩 → 某 kernel 的 workspace/預算寫死（MI350P：indexer 32 MiB → 512 MiB）
├─ HIP_LAUNCH_BLOCKING=1 或關 overlap 就不崩 → host 端 race 或 workspace 不夠，不是數值 bug
└─ 其他 → 縮小 repro（單請求、固定長度），二分 env/flag，不要直接跑 sweep
```

### 3.2 量測結果看起來很好

```
增益 > 10%？
├─ 檢查：取樣模式相同？雙方都沒量化？同 chunk？同 ENGINE_PYTHONPATH（有沒有混進別的 worktree）？
│     【MI350P】「BV +5–15%」其實混了另一個 worktree 的 MoE kernel；「1.0–1.2× vLLM」其實是 greedy 對 sampling。
├─ 檢查：是單格還是 ABBA 平均？單格 → 重跑
└─ 全部通過 → 換一個 concurrency 再驗證，然後記錄
```

### 3.3 增益小於雜訊

```
預期增益 < 單格雜訊（~10%）
├─ 能在 microbench / 固定 batch decode bench 量到嗎？能 → 先用它篩選
├─ 和其他小槓桿合併成一個 B arm，用 ABBA 量合併效果
│     【MI350P】hold + E2 合併：c16 +1.2%、c32 −0.8%、c64 −3.7% → 整組否決
└─ 合併仍量不出來 → 否決，記錄「量不出來」，轉向更大的槓桿（通常是 kernel 本身）
```

---

## 4. 雜訊紀律【通用；數字為 MI350P】

- AgentX 壓力模式 600 s 單格的 run-to-run 雜訊約 ±10–15%（同設定 c64 量過 158.2 與 141.0）。
- 所以：
  - 每個 A/B 至少 **A B B A**（每臂 2 次、交錯），比平均值。
  - 每個 concurrency 分開判定；c16 和 c64 常常方向相反。
  - 小於 5% 的差異視為中性。
  - 不跨日比較單格絕對值；只比同一時段內的配對。
  - 不開 per-call kernel verbose log（`TOKENSPEED_KERNEL_VERBOSE`），它會污染量測。
- 便宜的篩選工具（固定 batch decode bench、單一長 prompt prefill 時間）雜訊小很多，
  先用它們淘汰，再上 AgentX。

---

## 5. 槓桿目錄（摘要；完整數字與原因見 references/05-levers-catalog.md）

| 槓桿 | 結果 | 一句話原因 |
|---|---|---|
| 從裝置讀 CU/XCD 幾何 | ✅ 採用 | 寫死別張卡的幾何是純粹的錯 |
| indexer 預算 / 向量化載入 / tile 調整 | ✅ 採用 | 3.8× kernel、修崩潰 |
| KV 容量依記憶體預算（不被 max-total-tokens 卡住） | ✅ 採用 | c128 3×；先查「容量是被什麼限制的」 |
| 正確的取樣後端 + coupled drafting + block verify | ✅ 採用 | 公平比較的前提；接受長度回升 |
| prefill 兩 micro-batch 通訊重疊（TBO） | ✅ 採用 | c16 +27%、c32 +11%；side stream 必須高 priority |
| Iris 小訊息 32 programs（≤2 MB） | ✅ 採用 | decode −10% |
| HIP_FORCE_DEV_KERNARG=1 | ✅ 採用 | 與 vLLM 映像相同的啟動環境 |
| gmu 0.86（留 scratch 餘裕） | ✅ 採用 | 消除 OUT_OF_RESOURCES |
| 全部集合通訊改 RCCL | ❌ | 低 batch 時 Iris 較快 |
| INT8 all-reduce | ❌（公平比較中不允許） | 有損；decode 尺寸也無效 |
| Lamport/push all-reduce | ❌ | PCIe 上 6 rows 120 µs vs Iris 19 µs |
| Expert parallel | ❌ | TP4 上 −27~33% RPM |
| DP attention | ❌ | 複製權重的 GEMM 變慢、rank 不同步 |
| mixed batch | ❌ | c32/c64 −2~4% |
| prefill graph 擴到 8192 | ❌ | 端到端中性 |
| hold results / GIL interval / host prep | ❌ | trace 改善沒變成吞吐 |
| aiter MoE（vLLM 同款） | ❌ | ABBA 平手，多一份相依 |
| chunk 32K | ❌ | 比 16K 慢 |
| TBO 門檻 1024 + Iris cap 8 MB（E1+E2） | ❌ | −6% |
| shard-local block verify | ⚪ 中性，opt-in | 省 RCCL all-gather，但端到端在雜訊內 |

---

## 6. 腳本一覽（scripts/）

| 腳本 | 用途 |
|---|---|
| `env.sh` | 所有路徑與硬體設定（先編輯它） |
| `gpu_lock.sh` | `gpu_lock_acquire` / `gpu_lock_release`（noclobber） |
| `stop_servers.sh` | 安全停止所有 server（方括號 pattern），等記憶體釋放 |
| `serve_unieinfra.sh <CONC>` | 啟動 UnieInfra；所有實驗 env 在這裡傳進容器 |
| `serve_vllm.sh <CONC>` | 啟動 vLLM 基準（只有啟動參數） |
| `wait_ready.sh` | 真的生成 1 token 才算 ready；偵測啟動失敗 |
| `run_agentx.sh` / `run_round.sh` | 單格 benchmark；`STRESS=1` 為壓力模式 |
| `ab_abba.sh` | A B B A 交錯 A/B 模板 |
| `summarize_cells.py` | 讀結果目錄，印 RPM/ITL/TTFT 與每臂平均 |
| `prof_under_load.sh` | 負載中錄 GPU-only trace |
| `trace_exposed.py` | 暴露時間分解 |
| `bench_allreduce.py` | decode 尺寸 all-reduce 微基準（graph 內計時） |
| `scan_spills.py` | 掃 kernel 快取的 register spill |
| `needle_test.py` / `run_gsm8k.sh` | 正確性關卡 |
| `log_experiment.py` | 安全地新增/更新 experiments.json |
| `rocm_overlay.sh` | 同一 Python 環境換另一版 ROCm user-space 做 A/B（見 references/10-rocm-versions.md） |
| `rocm_micro.py` / `rocm_allreduce.py` | 換 ROCm 版本時的單元件微基準：launch、graph、GEMM、頻寬、RCCL |

範本在 `templates/`：實驗紀錄、server 啟動、A/B 腳本、PR 描述。

---

## 7. 每天開始工作時的檢查（照做）

1. `cat logs/GPU_LOCK` —— 有人在用嗎？
2. `tail -3 experiments.json` 的最後幾筆 —— 有沒有「待跑」但已經跑完的？先補紀錄。
3. 結果矩陣有沒有缺格？有就先補。
4. 目前的「基準設定」是哪一組 env + commit？寫在最近一筆 decision 裡。所有新 A/B 的 A 臂就是它。
5. 再決定今天的槓桿。
