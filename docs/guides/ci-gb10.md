# 目前 CI 設定

## CI 範圍

目前啟用的 GPU CI 目標是 self-hosted runner label：`gb10-altos`。
既有 worker/task YAML 宣告保留原本的 runner label；GB10 流程則透過新增的 GB10 專用 task 宣告與 workflow filtering 提供。

model-test 與 unit-test task 都會在 `gb10-altos` runner 上使用既有 image：

```text
unieinfra:gb10-main
```

不在 workflow 內重新 build image。

Docker image 的 `CUDA_ARCH_LIST` 固定為 `sm121`。`tokenspeed-kernel/python/setup.py` 會將 `sm121` 正規化為 `121`，再產生 `compute_121/sm_121` gencode，避免錯誤附加 `a` suffix。

```text
docker run --gpus all --network host --ipc host
```

在 container 內執行；`/home/altos/models` 以唯讀方式掛載。`gb10-altos` runner 必須安裝 Docker 與 NVIDIA Container Toolkit，並允許 Docker 使用 GPU。

## CI 運作流程

### 1. CI 觸發

CI 主要有兩種入口：

- **自動 CI**：Pull Request 或 push 觸發 GitHub Actions。
- **手動 CI**：透過 K8s dispatch workflow 選擇 task、trigger、runner 與 task type。

目前 active 的 GPU runner 是 `gb10-altos`。

### 2. 變更路徑過濾

自動 CI 先取得這次 Pull Request 或 push 的變更檔案，交由：

```text
test/ci_system/ci_path_filter.py
```

判斷是否需要啟動 GPU CI。文件、與目前 runner 無關的 task，或不屬於 GB10 範圍的變更，不會產生 GB10 task matrix。

### 3. 掃描 CI task 並建立 matrix

workflow 呼叫：

```text
test/ci_system/pipeline.py scan
```

pipeline 會掃描：

```text
test/ci/**/*.yaml
```

並依照以下條件篩選 task：

- `triggers` 是否符合目前事件，例如 `per-commit` 或 `manual`；
- `workflow_stage`，例如 `unit-test` 或 `model-test`；
- runner group 是否為 `gb10`；
- task 的 runner label 是否為 `gb10-altos`；
- multi-node 與排除 runner 設定。

目前產生的 matrix 為：

- `unit-test`：`test/ci/ut/gb10-unit-test.yaml`；
- `model-test`：兩個 Qwen model YAML。

### 4. 派送到 reusable workflow

建立 matrix 後，workflow 將它交給：

```text
.github/workflows/run-ci-task-matrix.yml
```

這個 reusable workflow 會針對 matrix 中的每個 task 建立一個 GitHub Actions job，並將 job 派送到：

```text
runs-on: gb10-altos
```

### 5. 執行單一 CI task

每個 task job 會依序執行：

1. checkout source code；
2. 建立 job 專用工作目錄；
3. 安裝 pipeline dependency；
4. 讀取 task YAML；
5. 執行 task 的 `install` 階段；
6. 執行 task 的主要階段：
   - unit task 執行 `ut.commands`；
   - model task 啟動 `server.command`；
   - 等待 `ready.url` 通過健康檢查；
   - 執行 `eval.command`；
   - 依 `score_threshold` 驗證評測分數；
7. 上傳結果與失敗 log；
8. 清理 job 工作目錄。

### 6. Unit 與 model 的執行關係

自動 workflow 會先建立並執行 unit-test matrix。model-test 依賴 unit-test job；只有 unit-test 成功，或沒有符合條件的 unit task 時，才會繼續執行 model-test。

目前 benchmark workflow 已停用，因此不會插入額外的 kernel benchmark stage。

### 7. 結果回報

每個 task 會產生 CI result artifact。GitHub Actions 會依照：

- task exit code；
- model evaluation score；
- task 是否標記為 optional；
- matrix 中所有 job 的結果；

決定整體 CI 是否成功。失敗時可從 workflow artifact 取得 `result.json` 與 task log。

## 目前啟用的 CI

### Unit test

`test/ci/ut/gb10-unit-test.yaml` 是 GB10 的 unit-test 入口，目前執行：

- runtime 1-GPU CUDA 測試；
- kernel numerics 測試；
- CUDA integration/function 測試。

由於此架構不支援 TRT-LLM，TRT-LLM communication 測試已排除；CLI 測試也不在目前範圍內。目前 active scope 僅限於 Triton/CUDA unit coverage。

這是目前唯一預期在 GB10 上執行的 non-model CI。既有 unit-test YAML 沒有重新標記為 `gb10-altos`。

### Model test

目前 model-test matrix 只有以下兩個 task YAML：

- `test/ci/eval/qwen3.8-27b-gsm8k.yaml`
- `test/ci/eval/qwen3.6-35b-a3b-gsm8k.yaml`

兩個 task 都具備以下設定：

- 在 `gb10-altos` 上執行；
- 從 `/home/altos/models/Qwen/` 載入本機 checkpoint；
- 使用設定好的 Triton attention/MoE backend；
- 使用 `chunked-prefill-size=8192`、`max-model-len=8192`、`max-num-seqs=64` 與 `kv-cache-dtype=fp8_e4m3`；
- 目前使用 GSM8K 作為 evaluation smoke test；
- score 至少需要達到 `0.90`。

## 如何新增 model test

可以。每個 model test 都是一個 task declaration，因此通常每個需要獨立 serving 與 evaluation 設定的 model/configuration，都要在 `test/ci/eval/` 下新增一個 YAML 檔案。

獨立 YAML 的好處是可以在同一個可重現的 task 中記錄：

- model checkpoint；
- serving arguments；
- runner label；
- evaluation dataset；
- trigger；
- score threshold。

如果需要手動選擇執行，新增 YAML 後也要將路徑加入 K8s dispatch 的選項。只要新 task 的 `runner.labels` 包含 `gb10-altos`，且 `workflow_stage` 為 `model-test`，它就會被自動 GB10 model matrix 掃描到。

如果多個 model 完全共用相同設定，未來可以重構成統一的 matrix 以減少重複；但目前 CI pipeline 以 task YAML 作為 source of truth，尚未使用獨立的 model list/matrix 檔案。

## Maintainer 手動觸發與 merge gate

目前 merge gate 分成兩部分：

### Lint：PR/push 自動執行

`.github/workflows/lint.yml` 會在 Pull Request 與 push 到 `main` 時自動執行，使用 GitHub 提供的：

```text
ubuntu-latest
```

這個 workflow 的 job/check 名稱是：

```text
Lint / lint
```

它是 required check，必須成功才能 merge。lint 不需要 `/ci-run`。

### GB10 CI：maintainer 執行 `/ci-run`

GB10 unit/model CI 不會因為 PR push 或 merge 到 `main` 自動執行。maintainer 需要在 PR comment 輸入完全符合的指令：

```text
/ci-run
```

`.github/workflows/ci-run.yml` 會：

1. 確認 comment 位於 Pull Request；
2. 透過 GitHub API 檢查留言者 repository permission，只有 `admin`、`maintain` 或 `push` 可以執行；
3. 讀取 PR 的最新 head SHA；
4. 以該 SHA dispatch `nvidia-gb10-tests.yml`；
5. 等待 GB10 unit-test 與 model-test 完成；
6. 將 `/ci-run` 的 `ci-run / GB10` job 設為成功或失敗。

`ci-run / GB10` 的 orchestration job 使用 `ubuntu-latest`，實際 GPU unit/model task 使用 `gb10-altos`。GB10 workflow 會 checkout PR head SHA，而不是 `main`，因此測試的是 maintainer 執行 `/ci-run` 當下的 PR 內容。

要讓 CI 阻擋 merge，repository maintainer 需要到 GitHub：

```text
Settings → Branches → Branch protection rules
```

在 `main` branch rule 的 required status checks 中加入：

```text
Lint / lint
ci-run / GB10
```

也就是必須同時滿足：

- `Lint / lint` 通過；
- maintainer 執行 `/ci-run`；
- `ci-run / GB10` 通過，代表 GB10 unit/model CI 通過。

建議同時啟用：

- Require status checks to pass before merging；
- Require branches to be up to date before merging（依團隊需求）；
- Restrict who can push to matching branches。

注意：GitHub branch protection 的 required checks 是 repository 設定，無法只透過 workflow YAML 自動建立；必須由 repository admin 設定。

## 已暫停的 CI

Kernel benchmark CI 目前已暫停：

- `.github/workflows/amd-kernel-benchmark.yml` 的 jobs 已停用；
- `test/ci/perf/kernel-benchmark-amd-gfx950.yaml` 已從 K8s dispatch 選項中註解；
- benchmark task 不會出現在 active GB10 unit/model matrix 中。

其他 model 原本的 model-test task 仍維持停用狀態，並未納入目前 active 的 GB10 model matrix。

## 驗證方式

目前設定已完成以下檢查：

- 解析所有 GitHub Actions 與 `test/ci` YAML 檔案；
- 編譯變更過的 CI Python modules；
- 使用 `runner_group=gb10` 掃描 pipeline。

最後產生的 matrix 包含一個 GB10 unit task，以及兩個 GB10 model task。
