# 09 陷阱清單（都是真實發生過的事）

每一條：發生了什麼 → 後果 → 規則。

## 共用資源

1. **GPU 鎖**
   兩個 session 共用同一台機器。沒取鎖就跑 → 兩個 server 搶記憶體、數據互相污染。
   規則：
   ```bash
   until ( set -o noclobber; echo "owner=<you>(<purpose>) since=$(date -u +%FT%TZ)" > logs/GPU_LOCK ) 2>/dev/null; do sleep 15; done
   trap 'rm -f logs/GPU_LOCK' EXIT
   ```
   `noclobber` 讓「檢查 + 建立」成為原子操作。`trap EXIT` 保證錯誤時也釋放。
   別人的鎖不要刪；等不及就問使用者。鎖內容寫清楚是誰、做什麼、預計何時釋放。
   長時間等鎖時（【MI350P】一次等了 ~5 小時）查佔用者是誰，回報給使用者。

2. **pkill 自殺**
   `docker exec ... bash -c 'pkill -f "tokenspeed::"; sleep 6; pkill -9 -f "tokenspeed::"'`
   第一個 pkill 匹配到 `bash -c` 自己的命令列，shell 被殺，`pkill -9` 沒執行，
   一個 OOM 後的 scheduler 佔著 127.7 GB 進入下一格。
   規則：pattern 用方括號：`"tokenspeed[:][:]"`、`"smg[_]grpc_servicer"`。
   停完之後用記憶體查詢確認 < 8 GB，不夠就依 pid kill。

3. **改到正在使用中的 worktree**
   benchmark 用 `.wt-X`，同時在 `.wt-X` 改程式碼 → Python 在 server 重啟時載入半改的程式碼，
   該格作廢。規則：要改就開新 worktree；用中的 worktree 唯讀。

4. **別的 worktree 混進來**
   `ENGINE_PYTHONPATH` 預設值裡留著 `.wt-ait`（未 commit 的 aiter MoE），整個 BV 系列實際上跑的是
   不同的 MoE，KV 少 14 GB。規則：每個腳本**明確**設定 `ENGINE_PYTHONPATH`；從 server log 核對
   載入的 kernel 名稱與 MoE 權重大小。

## 量測

5. **取樣模式不同**：greedy 對 sampling，「贏 1.2×」其實輸 0.85×。見 03 的公平性清單。
6. **量化偷渡**：啟動腳本預設開了 INT8 all-reduce，整個 sweep 帶著有損通訊。
   規則：有損功能預設關，比較前在 `/proc/<pid>/environ` 驗證。
7. **單格當結論**：見 06。
8. **VERBOSE log 污染**：`TOKENSPEED_KERNEL_VERBOSE=1` 每格多 15K 行。量測時一律關閉。
9. **跨日比較單格絕對值**：同配方不同日 ms/step 47→72，原因是工作負載狀態（cache 壓力、會話樹）。
   只比同時段的配對。
10. **profiler 放大 host 空隙**：CPU tracer 開著時 idle 20%，GPU-only 11%。
11. **log 取樣造成歸因錯誤**：decode 每 40 輪印一行 → 把 decode 時間算到 prefill。
12. **dashboard 分類 bug**：把壓力模式的 vLLM 系列分到官方，倍數被灌水（c16 2.99× 實為 1.03×）。
    規則：倍數出現大跳動時，回頭對原始資料。
13. **vLLM 異常格**：c1 ITL 36 ms（健康 6 ms）。重跑重現 → 標記設定錯誤並排除，不當分母。

## 啟動與環境

14. **docker `-e VAR` 不帶值**：host 未設時 docker 把映像內建的 `HSA_NO_SCRATCH_RECLAIM=1` 刪掉。
    規則：`-e VAR=${VAR:-<default>}`；驗證 `/proc/<pid>/environ`。
15. **新 env 沒加到 `-e` 傳遞清單**：開關在 host 設了，容器內看不到，「A/B 無差異」。
16. **core dump 塞滿磁碟**：`ulimit -c 0`。
17. **模型在慢磁碟**：冷載入逾時。放 NVMe。
18. **`/v1/models` 回應 ≠ ready**：gateway 活著、engine 死了。用 `wait_ready.sh` 生成 1 token，
    並從**這次**啟動的 log 偵測致命錯誤（否則會誤把下一格的 server 當成 ready）。
19. **上一格記憶體沒釋放就啟動**：RCCL init OOM，倖存的 scheduler 卡住整輪半小時。
    serve 腳本要等記憶體 < 8 GB，否則拒絕啟動。

## 記憶體 / 穩定性

20. **KV 被 token 上限卡住而不是記憶體**：調 gmu 完全沒效果才發現。先查容量是被什麼限制。
21. **KV 吃滿 → runtime scratch 失敗**：`OUT_OF_RESOURCES`。gmu 留餘裕（MI350P 0.86）。
22. **權重前處理碎片**：比較 `reserved − allocated`。

## 腳本

23. **在腳本頂層用 `local`**：`set -u` 下算術失敗，整輪中止。
24. **改寫正在執行的 bash 腳本**：bash 邊讀邊執行，改檔會讓正在跑的 queue 執行到錯位的行。
    規則：複製成新檔再改；或用 `logs/skip-<tag>` 這種旗標檔控制。
25. **結果迴圈遇到缺檔就 exit 1**：解析腳本要容忍無效格（印 INVALID 繼續）。
26. **中斷 queue 後殘留 worker**：下一格啟動 OOM。中斷後跑 `stop_servers.sh`。

## 工作方式

27. **trace 改善當成吞吐改善**：hold-results（trace −8%）、HOST_PREP、GIL interval、只驗證前 k 個 draft
    （microbench −12%）全部端到端中性。只信 ABBA。
28. **microbench 只量一部分成本**：DP attention 的通訊 microbench 預估 +8–17%，實際 2.7× 慢——
    沒量到複製權重後 GEMM 形狀變差。microbench 要涵蓋改動影響到的所有 kernel。
29. **不先算上限就投入**：prefix cache 已在 96.5%（理論 96.6%）；decode AR 實測已是延遲主導，
    壓縮無效。先算，再做。
30. **有損技術沒問使用者**：INT8 all-reduce 有效但有損，要使用者明確同意才能列入結果。

## 換 ROCm 版本（細節見 10-rocm-versions.md）

31. **只換一部分 ROCm 函式庫**：兩套 HIP runtime 同時載入，直接 segfault。一次換齊。
32. **少了 `.kpack`**：RCCL `invalid device function`，沒有其他提示。
33. **單元件全部變快就以為端到端會變快**：7.14／10.0 的 GEMM、RCCL、launch 都更快，端到端卻慢 6–29%。
    一定要跑同日交錯的端到端，再用 GPU 計時找停頓在哪。
