# 06 A/B 量測與雜訊紀律

## 1. 為什麼要這麼麻煩

閉環 benchmark（agent 跑完一輪才送下一輪）讓每格走到 trace 的不同位置；prefix cache 狀態、
batch 組成、prefill 長度分佈每次不同。結果：**同一設定的 600 s 格可以差 ±10–15%**。

【MI350P】真實例子：
- aiter MoE 單次 c16 80.6、c64 89.8 看起來大勝 → ABBA 後 70.5 vs 70.05，平手。
- hold-results c16 兩次 on：78.3、65.8（同設定差 19%）。
- E2 c64：158.2 與 141.0。

## 2. 標準流程：A B B A

```bash
cp scripts/ab_abba.sh scripts/ab_<name>.sh
# 編輯 ARM_A_ENV / ARM_B_ENV / CONCS / TAG
scripts/ab_<name>.sh            # 會自己等 GPU 鎖
python3 scripts/summarize_cells.py <TAG>A <TAG>B
```

規則：
- A 臂 = 目前的最佳設定（**同 commit 基礎**，最好就是 B 的 parent）。
- 兩臂只差一個變因（或一個明確定義的組合）。gmu、chunk、cap、取樣、`ENGINE_PYTHONPATH` 以外的
  路徑全部相同。
- 順序 A B B A，抵銷時間漂移（cache 狀態、溫度、別的工作）。
- 每個 concurrency 各自一組 ABBA。先跑最可能有效的 concurrency。
- 一格約 17 分鐘（啟動 + warmup + 600 s + drain）；3 個 concurrency × 4 格 ≈ 3.5 小時。先估時間再排。

## 3. 判定表

令 Ā、B̄ 為兩臂平均，Δ = (B̄ − Ā)/Ā。

| 條件 | 判定 |
|---|---|
| 任一格崩潰或無效 | 不判定；先查原因、補跑 |
| Δ ≥ +5% 且 min(B) > max(A) | 採用候選 → 在另一個 concurrency 再一組 ABBA；兩組都成立才採用 |
| Δ ≥ +5% 但區間重疊 | 再加一組 ABBA |
| −5% < Δ < +5% | 中性：否決，或保留為預設關的 opt-in（若有其他理由，例如省記憶體） |
| Δ ≤ −5% | 否決 |
| 不同 concurrency 方向相反 | 可以考慮依 concurrency 自動切換，但要每段都有 ABBA 支持 |

報告時一律給：每臂每次的數字、平均、Δ、ITL、TTFT、錯誤率。

## 4. 合併策略

當幾個槓桿各自只有 3–5%（在雜訊內），把它們合併成一個 B 臂一次量：
- 合併有效 → 再拆開做消去實驗（ablation）找出誰在貢獻。
- 合併無效 → 全部否決，**不要**再個別重跑；轉向更大的槓桿。

## 5. 便宜的前置篩選

先用低雜訊 bench 淘汰：
- 固定 batch decode（共享長前綴，16/32/64 users）：雜訊 ~2%。
- 單一長 prompt prefill ×3。
- kernel microbench（graph 內計時，真實 shape）。

通過前置篩選 ≠ 會贏 AgentX。但沒通過的幾乎一定不會贏。

## 6. 常見錯誤

- 只跑 A B（兩格）→ 時間漂移直接變成「差異」。
- A 臂用昨天的數字 → 跨時段不可比。
- B 臂偷偷多了別的改動（另一個 worktree 的 kernel、不同 gmu）。
  【MI350P】BV 系列混進了 aiter MoE 的 worktree，KV 少了 14 GB，結論作廢。
- 用 trace 上的 ms/step 代替 A/B 下結論。
- 單一 concurrency 的結果推廣到全部。
