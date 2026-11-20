# 02 正確性關卡

正確性是每個改動的入場券，不是 bring-up 時做一次就好。效能數字只有在正確性通過後才有意義。

## 1. 什麼時候要跑哪些關卡

| 改動類型 | needle | GSM8K | 投機解碼 TV | kernel 數值 |
|---|---|---|---|---|
| attention / indexer / KV cache kernel | ✅ | ✅ | | ✅ |
| GEMM / MoE kernel | | ✅ | | ✅ |
| 通訊（all-reduce 實作、量化） | ✅ | ✅ | | ✅（max err vs RCCL） |
| 投機解碼（drafter、verify、取樣） | | ✅（T=1.0） | ✅ | |
| 排程 / batch 組成 / prefill 切分 | ✅ | | | |
| 只改啟動參數 | 視情況 | | | |

## 2. Needle（長上下文檢索）

```bash
python3 scripts/needle_test.py --cases 6 --tokens 30000
```
在 ~30K token 的填充文字中不同深度藏一組代碼，要求模型回答。通過：`NEEDLE 6/6`。
長上下文相關改動也要用更長的 `--tokens`（例如 120000）。
5/6 不算過：重跑一次；若仍 5/6，比較改動前是否也是 5/6（【MI350P】TBO off 曾 5/6、on 6/6）。

## 3. GSM8K

```bash
scripts/run_gsm8k.sh <label>
```
與 repo CI 相同設定（evalscope、greedy、max_tokens 512）。通過：≥ CI 門檻（DSv4.1 為 0.90）。
改投機解碼的取樣時，額外用 T=1.0 sampled 跑 500 題，與改動前比較；差距應在 500 題的取樣雜訊內
（±~1.5 pp）。【MI350P】block verify F=1 95.8% vs F=0 96.2%。

## 4. 投機解碼無損（統計測試）

投機解碼的驗證規則（rejection sampling、block verify）必須讓輸出分佈等於目標模型分佈。
這不能靠 GSM8K 證明，要做統計測試：

1. 用 PyTorch 寫 reference 實作（照論文或照 vLLM 的 rejection sampler）。
2. 用隨機 logits 產生 target/draft 分佈，跑 N 次（≥ 6000；向量化可到 20 萬列）。
3. 量測輸出 token 邊際分佈對 target 的 TV 距離（總變異距離）。
4. 同時量**雜訊底**：直接從 target 取樣同樣次數的 TV。
5. 通過：實作的 TV ≤ 雜訊底（【MI350P】shard-local BV：TV 0.028–0.043，雜訊底 0.025–0.049）。

TP 分片版（每個 rank 只有部分 vocab）：在 CPU 上用多 thread 模擬 rank，比較分片版與單卡版的
接受長度均值與 TV。範例在專案 `unieinfra/.bench/test_bv_tp.py`（4-thread TP 模擬）。

## 5. Kernel 數值

- 對 reference（PyTorch 或舊 kernel）比 max abs error 與 relative error。
- 量化 kernel 與 BF16 reference 比；新 kernel 的誤差應與它取代的 kernel **同等級**。
  【MI350P】INT8 all-reduce rel err 7.7e-3 vs BF16 1.7e-3 → 有損，公平比較中不允許。
- 在 graph capture 下也要測一次（有些 bug 只在 graph 中出現）。

## 6. 端到端錯誤率

每個 benchmark 格都要看 error rate。非 0 就要查是哪一個請求、為什麼。
【MI350P】唯一一筆錯誤是 `max_new_tokens=2` 的 trace 請求遇到投機解碼一次接受多 token 的邊界情況；
另一次 10% 錯誤率是 admission（請求被拒）而非模型錯誤。分清楚再決定是否作廢。
