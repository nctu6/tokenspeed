# 08 紀錄與 PR

## 1. experiments.json

一個 JSON array，每筆一個實驗。**每個**實驗都要有（包含否決、無效、待跑）。

欄位（見 `templates/experiment_entry.json`）：
| 欄位 | 內容 |
|---|---|
| `id` | 遞增整數 |
| `date` | YYYY-MM-DD（UTC） |
| `area` | 類別：Kernels / Comm / Scheduler / Speculative / Prefill / Benchmark / Profile / Methodology / Stability … |
| `experiment` | 做了什麼：腳本名、branch/commit、env、shape、concurrency、重複次數 |
| `result` | 數字：每臂每次、平均、Δ、ITL/TTFT、錯誤率；或「待跑」 |
| `interpretation` | 為什麼是這個結果 |
| `decision` | 採用 / 否決 / opt-in / 下一步 |
| `commit` | hash（branch） |

工具：
```bash
python3 scripts/log_experiment.py --area Kernels --experiment "..." --result "待跑" --decision "-" --commit "abc123 (branch)"
python3 scripts/log_experiment.py --update 245 --result "..." --interpretation "..." --decision "否決"
```
工具會：用檔案鎖、重新讀檔、只附加或只改指定 id、保持 UTF-8 縮排。多個 session 同時寫也安全。

紀錄要寫到別人不看程式碼也能理解：寫清楚「跟什麼比」（A 臂是哪個 commit + env）。

修正錯誤：不要刪舊紀錄。新增一筆「撤回 / 更正 #N」，並在舊紀錄的 decision 加註。

## 2. Dashboard

- 每次實驗結束後更新（使用者要求「dashboard 跑完記得更新」）。
- 實驗紀錄 dataset 從 experiments.json 來；benchmark 結果 dataset 從結果目錄匯出
  （專案有 `export_dashboard.py`）。
- 更新方法依你的 dashboard 系統而定；本專案是把 json 上傳成 artifact asset，再更新 `datasets/log`
  文件的 `source.url`（帶 `if_version`）。

## 3. Commit 規則

- 一個 commit 一個邏輯改動；訊息第一行 `<元件>: <做了什麼>`，body 寫為什麼與數字。
- 結尾 `Signed-off-by: <名字> <email>`（repo AGENTS.md 要求）。用 `git commit -s`。
- **不得**有 AI 署名行：`Co-Authored-By: Claude...`、`Claude-Session: ...`、
  `Generated with Claude Code`、或任何其他 AI 工具的同類行。這條優先於工具預設。
- 檢查：
  ```bash
  git log <base>..HEAD --format=%B | grep -inE "co-authored-by|claude|generated with" && echo "FIX ME"
  ```
- 已經有署名的分支：建乾淨分支，`git cherry-pick` 每個 commit 後 `git commit --amend` 改訊息
  （或 `git rebase` + `exec`），**不要**在共用分支上強制推送，除非使用者要求。

## 4. PR

時機：只有在目標全部達成（或使用者要求）時才開 PR 到 main。

內容（見 `templates/pr_body.md`）：
- 摘要：改了什麼、為什麼。
- c1–c128 結果表（vs vLLM 最佳），寫清楚 benchmark 模式與時長。
- 正確性：needle、GSM8K 數字。
- 新的 env 開關與預設值。
- 否決的方向（簡短，指向 experiments.json）。
- **沒有** AI 署名 footer。

推送與建立 PR 是對外動作：先問使用者，或由使用者自己執行。
