## Summary

<一段話：這個 PR 讓 UnieInfra 在 <硬體> 上 <模型> 的 <場景> 達到什麼程度，主要做了哪三件事。>

## Changes

- **<元件>**：<做了什麼>（env `TOKENSPEED_<X>`，預設 <on/off>）。<一句話效果>。
- ...

## Results

AgentX 壓力模式（`--ignore-trace-delays`），600 s／格，TP<N>，<GPU 型號 ×N>。
vLLM：<image digest>，官方 recipe 只改啟動參數（見 scripts/serve_vllm.sh 註解）；每格取 recipe cap 與 2×CONC cap 中較佳者。
重複量測者取平均（A B B A）。

| CONC | vLLM best RPM | UnieInfra base | + <改動 1> | + <改動 2> | ratio |
|---|---|---|---|---|---|
| 1 | | | | | |
| 4 | | | | | |
| 16 | | | | | |
| 32 | | | | | |
| 64 | | | | | |
| 128 | | | | | |

## Correctness

- Needle 30K：6/6
- GSM8K（CI 設定）：xx.x%（門檻 0.90）
- 投機解碼統計測試：TV x.xxx（雜訊底 x.xxx）

## Rejected directions

<一行一個：方向 — 結果 — 原因>（完整紀錄見 experiments.json）

## Test plan

- [ ] unit tests
- [ ] needle / GSM8K
- [ ] c1–c128 stress sweep
