# 04 Profiling：先知道時間花在哪

## 1. 原則【通用】

1. **在真實負載下錄。** 單請求 microbench 的分佈與 AgentX 下完全不同（prefill/decode 混合、
   batch 大小變動、prefix cache 命中）。
2. **先錄 GPU-only。** CPU/Python tracer 會放大 host 空隙。【MI350P】c32：開 CPU tracer 量到
   GPU busy 80%，GPU-only 為 89%。
3. **看暴露時間，不看 kernel 總時間。** 多 stream 時，被其他 stream 蓋住的 kernel 時間省下來也沒用。
4. **對最大項算下界。** 知道離理論極限多遠，才知道還值不值得做。
5. **不要用每 N 輪才印一行的 log 做歸因。**【MI350P】decode log 每 40 輪一行，曾誤判「短 extend 483 ms」。

## 2. 錄 trace

```bash
scripts/prof_under_load.sh 32        # 在 c32 壓力負載進行約 4 分鐘時錄 6 s GPU trace
```
它做的事：取得 GPU 鎖 → 用 `run_round.sh` 起 server 跑 300 s 壓力 → 等 240 s →
`POST /start_profile {"output_dir": ..., "activities": ["GPU"]}` → 6 s → `POST /stop_profile`。
需要 CPU 視角時把 activities 改成 `["CPU","GPU"]`，但數字要打折看。

## 3. 暴露時間分解

```bash
python3 scripts/trace_exposed.py logs/prof-<...>
```
輸出：
- `union busy`：任一 stream 有 kernel 的時間比例；`idle` = GPU 完全閒置。
- 每個（類別, stream）：kernel 數、總時間、**暴露時間**、佔 wall %。

類別依 kernel 名稱正則分（`AR-iris`、`RCCL`、`MoE`、`attn`、`indexer`、`mHC`、`GEMM`、`copy/misc`）。
**換模型或換硬體時要更新 `cat()` 裡的正則**——第一次跑先把 `other` 的前 30 名 kernel 名稱印出來，
把它們歸類。

【MI350P c32 TBO 後】busy 86%：MoE 17.4%、decode AR 12.0%、GEMM 7.8%、prefill AR（未重疊）5.8%、
attn 5.0%、indexer 3.5%、RCCL（BV all-gather）3.0%；TBO side-stream AR 1289 ms 只暴露 97 ms。

## 4. 閒置時間分類

GPU idle 再分：
- **eager launch 小空檔**（< 100 µs，同 stream 連續 kernel 之間）：kernel 太多太小。解法是進 graph
  或融合。【MI350P】stream 1 有 56,422 個、平均 6.8 µs、共 385 ms（6.4% wall）；graph 內只有 0.9 µs。
- **步間 host 空檔**（> 100 µs）：排程、配頁、H2D 拷貝、GIL。看前後是哪兩個 kernel/拷貝。
  【MI350P】最大項 `zero_page_fields → HtoD`。
- 嘗試過的 host 端修法（hold results、GIL switch interval、host prep thread、graph segment
  scheduling）在 trace 上有改善，但 AgentX 都中性——host 空檔被 batch 動態吸收了。

## 5. 下界計算範本

### MoE decode（頻寬下界）
```
每步讀的權重 = 啟用 expert 數 × 每 expert 權重大小 × 層數 / TP
下界 = 每步讀取量 / HBM 峰值頻寬
```
【MI350P】192 rows 時啟用 ~367/384 experts、每步 ~75 GB/GPU MXFP4、8 TB/s → 9.4 ms，
實測 13.1 ms = 72% 峰值。剩餘空間 ~28%，要改 kernel 本身（load 排程、expert 排序、split）。

### decode all-reduce（延遲下界）
```
two-shot 每卡收發 ≈ 2 × (TP-1)/TP × 訊息大小
頻寬下界 = 收發量 / 單卡互連頻寬
```
【MI350P】1.5 MB、PCIe ~55 GB/s → 41 µs，實測 110–130 µs → 主要是同步/往返延遲而非頻寬。
所以壓縮（INT8 減半傳輸）、one-shot↔two-shot、與 norm 融合都無效。

### prefill all-reduce
【MI350P】168 MB：RCCL 6.19 ms、RS+AG 7.42、Iris two-shot 2.6 ms/84 MB。PCIe 主機上 prefill
尺寸走 Iris；要隱藏它只能和計算重疊（TBO）。

## 6. 逐 kernel 對照 vLLM

兩引擎跑同 shape（同前綴長度、同 users、同 temperature），各錄一份 trace，取 rank 0 穩態步，
逐類別比 µs/步。這會直接告訴你「哪個 kernel 比 vLLM 慢多少」。
【MI350P】正常步中位數 45.1 vs 40.0 ms；差距分散在 indexer、top-k、attention，最後逐一移植/重寫。

## 7. 產出（P5 離開條件）

一張表，寫進 experiments.json 的 result：
| 類別 | 暴露 ms | 佔 wall % | 下界 | 實測/下界 | 已試過的方法（查 05） |
