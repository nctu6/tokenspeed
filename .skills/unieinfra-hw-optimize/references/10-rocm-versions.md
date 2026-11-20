# 10. 換 ROCm 版本的 A/B【通用；數字為 MI350P】

目的：判斷換 ROCm user-space（HIP runtime、RCCL、hipBLASLt、comgr）會不會讓引擎變快，
而且只改這一個變數——同一份程式碼、同一個 Python 環境、同一台機器。

## 做法：torch overlay

新版映像的 torch 版本通常和引擎需要的不同，換映像會同時改兩個變數。改用 overlay：
複製一份 torch 套件，把它自帶的 ROCm 函式庫換成另一版，靠 `PYTHONPATH` / `LD_LIBRARY_PATH`
讓它優先載入（`scripts/rocm_overlay.sh`）。

必做的事（每一項都踩過）：

1. **一次換齊** torch/lib 裡所有 runtime 函式庫（HIP、HSA、RCCL、hipBLASLt、rocBLAS、comgr、
   hiprtc、rocprofiler、roctracer、roctx、aqlprofile、rocm_smi、rocm-core）。只換一部分，
   程序裡會同時有兩套 HIP runtime，直接 segfault。
2. **帶上 `.kpack`**：TheRock 建置把 GPU code object 另外放在 `lib/` 旁邊的 `.kpack/`。
   少了它只會看到 RCCL `invalid device function`；`AMD_LOG_LEVEL=2` 才會顯示
   `kpack_load_code_object failed with error: 13`。
3. **上游 triton 要換成同一套 LLVM 的建置**：TheRock 的 comgr 以全域方式載入 shared libLLVM，
   靜態連結 LLVM 的 libtriton.so（torch 或取樣 kernel 會 import）在靜態初始化時 crash。
   引擎自己的 kernel 若用自帶 LLVM 的 triton fork，不受影響、兩邊相同。
4. **確認真的換到了**：`hipRuntimeGetVersion` 要從**已載入**的 libamdhip64 取（讀
   `/proc/self/maps`）；用名稱另外 dlopen 會再載入一份 runtime。`torch.version.hip` 是編譯時版本，不準。
5. **限制**：torch 本身的 kernel 仍是舊編譯器產物（載入時約 100 筆 `register fat binary failed`，
   常用 op、CUDA graph、torch.distributed 都正常）。這不是官方支援組合；要給 vendor 看的結論，
   要寫明這一點。

## 量測順序

1. **單元件 microbenchmark**（`scripts/rocm_micro.py`、`scripts/rocm_allreduce.py`）：兩個 runtime 各跑兩次。
2. **端到端**：同一天交錯（新 → 舊 → 新 → 舊），每次重啟 server，至少 c4／c16／c64。
3. **若端到端與 microbenchmark 方向相反**：
   - 從 server log 比投機解碼接受長度、每請求 decode 速度（gen tok/s ÷ running-req）、執行期 JIT 次數。
   - 冷長 prompt prefill 每輪延遲，TBO 開／關各一次（看重疊是否還有效、每輪是否穩定）。
   - 開 `TOKENSPEED_GPU_STEP_TIMING=1` 看停頓在 GPU 端（`extend` 變長、`step_host` 不變）還是主機端。

## MI350P 的結果（7.2.4 → 7.14.1／10.0）

- 端到端變慢：7.14 c4 −22%、c16 −11%、c64 −6%；10.0 c4 −8%、c16 −29%、c64 +2%。
- 單元件幾乎全部持平或更快：hipBLASLt GEMM 快 5–30%、RCCL 小封包快約 20%、eager launch 7.14 快 12%
  （10.0 的 eager 提交慢 6%）；graph replay、HBM 頻寬持平。
- 接受長度、每請求 decode 速度相同；prefill TBO 仍有效（開比關快 13–20%）。
- 差別在 prefill 的穩定性：7.2.4 每輪差異 ≤ 0.1 s；新版偶發 +30–50%，GPU 計時顯示同一批 kernel
  在同一個 server 裡逐漸變慢 10–30%，偶爾整步多 2 s（4 個 rank 同時），主機提交時間不變。
- 結論：維持 7.2.4，把上述數據交給 AMD；下一步是在執行中記錄時脈／功耗／溫度。

## 順帶發現

- ROCm 7.2.4 的 HIP runtime 在同一個 50 節點 graph 連續 replay 約 2,000 次（中間不同步）後 segfault；
  HIP 7.15 正常。microbenchmark 要每次 replay 後同步。
