# 05 槓桿目錄：試過什麼、結果、為什麼

每一項格式：**槓桿** — 結果 — 原因 / 教訓（#實驗編號，指 MI350P 專案 experiments.json）。
數字都是【MI350P】的；「教訓」欄是【通用】的。

做任何新槓桿前先在這裡搜尋。被否決的不是不能再試，而是要先說明這次哪裡不同。

## 目錄
1. Kernel（arch 專屬）
2. 通訊
3. KV cache / 容量 / 排程
4. 投機解碼
5. Prefill
6. Host / runtime
7. 平行策略
8. 穩定性
9. 合併與 env 篩選

---

## 1. Kernel

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| 從裝置讀 CU/XCD（取代寫死的 256 CU/8 XCD） | ✅ decode 28.8→26.8 ms/token（#59） | 先修「錯」再調「快」 |
| MXFP8 decode GEMM `NUM_XCDS` 8→4 單獨改 | ❌ +0.4%（#11） | 幾何影響依 kernel 而定，要量 |
| indexer logits 預算 32 MiB→512 MiB | ✅ 修長 prompt 崩潰、overlap 回來（#7） | 寫死的 workspace 是崩潰源 |
| indexer key 載入允許 specialize（16 B 向量），waves_per_eu 2→8 | ✅ 2332→617 µs，3.8×（#8） | `do_not_specialize` 讓 stride 退化成 byte load |
| indexer 6 個 spec row 共用 key 載入（第一版） | ❌ 617→3536 µs（#12） | 天真的共用破壞了平行度 |
| indexer tile sweep（16/32/64/128 rows × waves） | ✅ 16 rows 1 wave + chunk 256/512/1024 bucket（#13） | 做完整 sweep，不要猜 |
| row-block indexer logits（aiter 風格，6 rows/CTA） | ✅ 155K prefill 5.6→5.0 s（#60） | 參考 vendor 庫的資料流 |
| candidate-mode indexer 用 32×32×64 MFMA | ✅ 64 users 27.3→25.4 ms/token（#61） | 用對矩陣指令 |
| 兩段式 LDS 壓縮 top-k select | ✅ 3.6→1.5 ms/step（#62） | |
| selected attention rows（dim-split waves、2 WG/CU） | ✅（#66） | |
| decode query 只補到 16 heads（原補到 64） | ✅ 24.3→22.8 ms/token（#67） | 不必要的 padding = 白算 |
| selected attention 6 rows/WG + LDS dedup | ❌ 正確但不更快（#79） | |
| attention rows tile split | ❌ 66→69–79 µs（#198） | |
| 移植 vLLM/aiter block32 MXFP8 GEMM、依 128 CU 重調 | ✅ M=96 2–3× 快，每步省 ~2.8 ms（#75） | 小 M decode GEMM 是常見熱點 |
| 拿掉 FP32 -inf logits pre-fill | ✅ FillFunctor 從 profile 消失（#77） | 看 profile 裡的 Fill/memset |
| draft attention 寬度 134 落到 portable Triton → 補到 192 走 gfx950 kernel | ✅ 每步省 2.3 ms（#56） | **落到 portable 的 kernel 是隱形熱點** |
| MoE：aiter FlyDSL a8w4（vLLM 同款）+ 128 CU tuning | ❌ ABBA 平手（#222） | 單次的 +15% 是雜訊 |
| MoE decode e4m3 | ❌ 速度（#198） | |
| MoE decode 頻寬：實測 72% 峰值（#153） | 開放 | 剩餘 ~28% 要改 kernel 內部 |

## 2. 通訊

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| PCIe-only 主機所有集合通訊改 RCCL | ❌ n=1 155 vs Iris 202 tok/s（#14） | 低 batch 時 Iris（GPU 直接 pull）延遲低 |
| prefill 尺寸 Iris cap 提到 256 MB | ✅ 104K prefill −10%（#25） | 大訊息 RCCL 在 PCIe 上慢 |
| Iris 小訊息（≤2 MB）用 32 programs | ✅ 16 users −10%（#73） | 預設 84 programs 對 1–4 MB 太多 |
| Iris small cap 提到 8 MB（E2） | ❌/不確定（#242, #244） | c64 158.2/141.0 散佈太大 |
| Iris small program 數 sweep 16–84 | ⚪ 不是槓桿（#127） | |
| INT8 all-reduce（prefill） | ⚠ 有效（c32 +12%、c128 +28%）但有損，公平比較中禁止（#93, #110） | 有損技術要使用者明確同意 |
| INT8 all-reduce（decode 尺寸） | ❌ 23.3 vs 21.6 ms（#99, #108） | decode AR 是延遲不是頻寬 |
| Lamport / push all-reduce | ❌ 6 rows 120 µs vs Iris 19 µs（#234） | PCIe 上 push/flag 輪詢貴 |
| 換 producer-direct 幾何 | ❌ 無變化（#109） | 停止調 transport |
| side-stream all-reduce 與計算重疊：RCCL vs Iris | RCCL 藏 64%、Iris 84 programs 只藏 30%、Iris 32 programs 藏 64%（#31, #45） | Iris 的 spinning WG 佔 CU；重疊時要降 programs |
| decode AR 與 MoE 重疊 | ❌ 無設計達 3 ms/步門檻（#199） | 先算可藏的上限再實作 |

## 3. KV cache / 容量 / 排程

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| KV 依記憶體預算（`MAX_TOTAL_TOKENS=2^30`）而非 16.7M 上限 | ✅ c128 41.5→123.5 RPM，3×（#81） | **先查容量被什麼限制** |
| `MAX_TOTAL_TOKENS` 67M / 47M | ❌ OOM（#83, #85） | 留餘裕 |
| compressor_tail 每頁 2→32 rows | ✅ c64 +8%（#69, #72） | block table 打包成本 |
| cache-aware admission（依已快取前綴排序） | ✅ c128 19.6→30.2 RPM、命中 0→65%（#47） | |
| decode-every-N-prefill-rounds=1 | ✅ c128 0→21.4 RPM（#39） | prefill-first 排程讓 decode 餓死 |
| mixed batch（小 chunk） | 低並行有幫助、高並行不行（#43, #44） | |
| mixed batch（8K chunk，TBO 後） | ❌ c32/c64 −2~4%（#240） | |
| prefix cache 進一步改善 | ❌ 已近上限（96.5% vs 理論 96.6%）（#174, #175） | 先算上限再投入 |
| GIL：ForwardBatch 複製 MB 級 block table | ✅ 25.4→24.5 ms/token（#64） | Python 熱路徑上的大物件複製 |
| 排程器 post_process 快路徑 | ✅ 每步省 ~50 µs（#225） | |

## 4. 投機解碼

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| `--sampling-backend triton_full`（真的取樣） | 必要；吞吐 −5~25%（#96） | 公平比較的前提 |
| coupled drafting（draft 與 target 共用隨機數） | ✅ c16 +11%（#100, #102） | |
| draft 溫度縮放 | ❌（#101） | |
| 只驗證前 k 個 draft | ❌ AgentX 不一致（#80, #84） | microbench −12% 沒轉成吞吐 |
| shadow n-gram drafter | ❌（#89） | |
| block verification（整塊驗證） | ✅ c1 +19%、c4–c16 +3~8%（#172–#214） | 統計無損測試先行 |
| BV 在 bs>128 時記憶體爆 → chunk 16 + max_bs 128 | ✅ c128 修好（#209–#211） | |
| shard-local BV（TP 下只交換 scalar） | ⚪ 中性，opt-in（#236, #238） | 省 RCCL 3%，端到端雜訊內 |

## 5. Prefill

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| chunk 32K vs 16K | ❌ 較慢（#27） | |
| chunk 4K/8K/16K（AgentX） | 8K 為最終（#98, #226） | 依負載選 |
| row-chunked FFN 通訊重疊（同 stream / Iris） | ❌ 變慢（#32, #49） | ROCm 同 priority stream 被序列化 |
| **prefill TBO：兩 micro-batch，attn(A) attn(B) MoE(A) MoE(B)，AR 在高 priority side stream** | ✅ 長 prefill −24%，c16 +27%、c32 +11%（#228–#231） | side stream 要 `priority=-1`，否則 overlap = 0 |
| TBO 門檻 4096→1024 rows（E1） | ❌（#237） | 小 chunk 拆兩半開銷 > 收益 |
| prefill graph 擴到 8192 | ❌ 中性（#226） | |

## 6. Host / runtime

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG=1` | ✅ c1 139→191 tok/s（#16, #182） | 跟 vLLM 映像一致 |
| `GPU_MAX_HW_QUEUES=2`、`DEBUG_CLR_GRAPH_PACKET_CAPTURE` | ❌ 不穩定（#16） | |
| zero-on-exec / upload stream 變體 | ❌（#63） | |
| HOLD_RESULTS_UNTIL_LAUNCH | ❌ trace −8%、AgentX 中性（#217, #223, #244） | trace 改善 ≠ 吞吐 |
| GIL switch interval 500 µs | ❌（#206） | |
| DECODE_HOST_PREP（背景執行緒準備） | ❌（#197） | |
| graph segment scheduling / HC inline | ❌（#197, #201） | |

## 7. 平行策略

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| Expert parallel（TP4 上） | ❌ −27~33% RPM（#117–#120） | dispatch/all-to-all 成本回不來 |
| DP attention（TP1×DP4） | ❌ 64 users 2.7× 慢（#104, #105） | 複製權重 GEMM 形狀變差、每 rank 獨立 event loop 不同步；microbench 只看通訊漏掉 GEMM |
| 移植 vLLM mono decode | ❌ 不做（#176） | 要 256 CU；只支援 M≤48；估 6–10 週 |

## 8. 穩定性

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM=1`（正確傳入容器） | 必要但不足（#232, #233） | docker `-e VAR` 不帶值會刪掉映像值 |
| gmu 0.9 → 0.86 | ✅ OUT_OF_RESOURCES 消失（#233, #243, #244） | 留 ~6 GB/GPU 給 runtime scratch |
| 掃 register spill | 開放：208 個 kernel spill，最高 2024 B/lane（#243） | 根治：KV 預算計入 scratch，或在配 KV 前預熱 |
| aiter 權重前處理碎片 | ⚠ 吃掉 ~12 GB；預配 padded buffer 原地 shuffle（#212–#220） | 比較 reserved vs allocated |

## 9. 合併與 env 篩選

| 槓桿 | 結果 | 原因 / 教訓 |
|---|---|---|
| E1+E2（TBO 門檻 1024 + Iris small cap 8 MB） | ❌ −6%（#237） | |
| 合併 B = hold + E2，ABBA c16/c32/c64 | ❌ +1.2% / −0.8% / −3.7%（#244） | 單項 3–5% 合起來仍量不出來 → 轉向 kernel 本身 |

## 目前最佳設定（【MI350P】；作為新 A/B 的 A 臂範例）

```
STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1
HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1
TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*CONC))
MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 HSA_NO_SCRATCH_RECLAIM=1
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats"
```
結果（壓力 600 s RPM，vs vLLM 最佳）：c1 17.8（1.62×）、c4 48.1（1.29×）、c16 83.2（1.30×）、
c32 122.2（1.09×）、c64 151.2（1.12×）、c128 129.1（5.56×）。


## MoE kernel roofline（#245–246）
| 槓桿 | 結果 | 決定 |
|---|---|---|
| decode MoE kernel 優化 | 已達讀取峰值的 85–97%（2.6–3.05 / 3.13 TB/s） | 不做 |
| prefill MoE 調參（block_m、warps、reduce） | M=8192 −18%；但 TBO 拆半後的 4096 是 0% | 不合入 |
| stage1 XCD swizzle + GROUP_M | ±1% | 還原 |
| prefill MoE GEMM 重寫 | FP4 算力只用 6–13%；估 c32 GPU 時間 −3.7% | 待決定 |
