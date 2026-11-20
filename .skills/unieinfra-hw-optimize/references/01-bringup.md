# 01 Bring-up：讓 UnieInfra 在新硬體上跑起來

## 目錄
1. UnieInfra 的硬體抽象在哪裡
2. Kernel 選擇機制與如何檢查
3. 新硬體 bring-up 逐步流程
4. 寫死的硬體幾何
5. 記憶體規劃
6. 啟動環境變數
7. 崩潰分流表
8. 【MI350P】實例

---

## 1. 硬體抽象在哪裡【通用】

| 路徑 | 內容 |
|---|---|
| `tokenspeed-kernel/python/tokenspeed_kernel/platform.py` | `current_platform()`：vendor、arch 版本、CU/SM 數、interconnect（`_detect_rocm_interconnect` 讀 `/sys/class/kfd` 的 xGMI link；CUDA 讀 NVLink） |
| `tokenspeed-kernel/python/tokenspeed_kernel/registry.py` | `register_kernel(family, mode, name=, solution=, capability=CapabilityRequirement(...), signatures=, traits=, priority=)` |
| `tokenspeed-kernel/python/tokenspeed_kernel/selection.py` | 依 `(oracle, priority)` 排序選 kernel；`explain_selection()`；`TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>` 強制指定 |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/<family>/` | 各 op family 的 portable 實作（triton / gluon / native / flashinfer …） |
| `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/` | AMD gfx950 專屬 kernel（attention、gemm、moe、layernorm、sampling、mhc …） |
| `tokenspeed-kernel-npu/` | NPU 專屬 kernel |
| `python/tokenspeed/...` | 模型、排程前端、attention backend |
| `tokenspeed-scheduler/csrc/` | C++ 排程器（混合 batch、admission 等 env 在這裡讀） |
| `scripts/<model>.sh` | repo 內建的模型啟動腳本；外層 `serve_unieinfra.sh` 包它 |

新硬體 = 新的 arch 專屬目錄（例如 `ops/gfx942/`、`ops/sm_120/`）+ 在 `CapabilityRequirement`
裡宣告適用範圍 + （必要時）在 `platform.py` 補偵測。

## 2. Kernel 選擇機制與如何檢查【通用】

每個 op 在執行時由 selection 從所有已註冊、capability 符合的 kernel 中挑分數最高者。
一個 kernel 不符合時會**靜默**落到下一個，通常是 portable Triton——慢但能跑。
這就是新硬體上最常見的隱形熱點。

檢查方式（容器內）：
```python
from tokenspeed_kernel.selection import explain_selection
# family/mode/format_signature 依各 op；在 ops/<family>/__init__.py 找呼叫 select_kernel 的地方抄參數
print(explain_selection("attention", "dsa_decode", <format_signature>))
```
輸出會列出所有候選、分數、哪個被選中、是否有 override。

快速盤點：啟動一次 server，加 `TOKENSPEED_KERNEL_VERBOSE=1`，跑幾個請求，grep log 中
每個 op 選中的 kernel 名稱，列成表。**盤點完立刻關掉 VERBOSE**，它會污染之後的量測。

強制指定做 A/B：`TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY=<kernel_name>`。
注意：override 到一個不支援當前格式的 kernel 會在第一個 batch 直接 ValueError
（【MI350P】Triton MoE 不支援 MXFP4 → 不能當對照組）。

## 3. 逐步流程【通用】

1. **確認平台偵測**
   ```bash
   docker exec $CONTAINER python -c "from tokenspeed_kernel.platform import current_platform as p; print(p())"
   ```
   檢查：vendor、arch（例 gfx950 / sm_100）、CU/SM 數、interconnect topology。
   interconnect 錯會讓通訊路徑選錯（PCIe-only 主機和 xGMI/NVLink 主機最佳 all-reduce 不同）。
2. **列出被選中的 kernel**（上一節），標記：arch 專屬 / portable / 第三方（aiter、flashinfer）。
3. **找寫死的幾何**（下一節）。
4. **算記憶體**（第 5 節），決定哪些東西放 host、KV 預算多少。
5. **最小啟動**：c1、短 prompt、`--disable-overlap-schedule` 先排除 race；能回答再打開 overlap。
6. **長序列壓力**：單一序列 40% max context；3 條並行 ~250K。這兩個 repro 在【MI350P】分出了
   「依單序列深度崩潰、與並行無關」的 bug（indexer 預算寫死）。
7. **啟動 3 次**確認穩定，每次記錄 KV token capacity（log 中 `engine ready: max_total_num_tokens`）。
8. 記錄到 experiments.json：硬體描述、kernel 盤點表、記憶體配置、已知問題。

## 4. 寫死的硬體幾何【通用；MI350P 為實例】

搜尋：
```bash
grep -rnE "NUM_XCDS|NUM_SMS|num_cus|= 256\b|= 304\b|= 132\b|multi_processor_count" \
  tokenspeed-kernel*/python | grep -v test
```
常見的寫死點：
- grid 大小 / persistent kernel 的常駐 CTA 數（= CU 數）
- XCD swizzle（AMD 多 die；PID 依 XCD round-robin 分配）
- split-K 份數（依 CU 數計算）
- LDS / shared memory 大小、wave/warp 大小（AMD 64、NVIDIA 32）
- 每種格式的 MFMA/MMA 指令形狀

修正原則：**從裝置讀**（`torch.cuda.get_device_properties().multi_processor_count`、
KFD `num_xcc`），不要換成另一個寫死值。

【MI350P】gfx950 kernel 依 MI355X（256 CU、8 XCD）寫死；MI350P 為 128 CU、4 XCD。
MXFP8 GEMM split-K、MXFP4 MoE/GEMM swizzle、DSv4 prefill、selected attention 改為讀裝置後：
64 人 decode 28.8 → 26.8 ms/token。另一方面，單純把 MXFP8 decode GEMM 的 `NUM_XCDS` 8→4 只 +0.4%
——幾何修正的收益依 kernel 而定，要量。

【MI350P】vLLM 的 mono decode kernel 要求 256 CU，在 128 CU 上拒絕啟動，所以 vLLM 基準不開 `--monokernel`。

## 5. 記憶體規劃【通用】

每張 GPU：`權重/TP + KV 預算 + CUDA/HIP graph + activation workspace + runtime scratch + 碎片`。

- 先查 **KV 容量是被什麼限制的**：記憶體預算（`GPU_MEMORY_UTILIZATION`）還是 token 上限
  （`MAX_TOTAL_TOKENS`）。【MI350P】KV 被 16.7M token 上限卡住而非記憶體，gmu 0.9→0.97 完全不變；
  改成依預算後 c128 RPM 3×。
- 留 runtime scratch 餘裕。AMD 上有 register spill 的 kernel 每個硬體佇列需要 scratch
  （每 lane 私有大小 × 常駐 wave 數，2 KB/lane 時約 0.5 GB/佇列）。KV 吃滿後首次在新佇列上跑
  大 spill kernel 會失敗 → `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。【MI350P】gmu 0.9 崩潰率 5.3%，
  0.86 為 0。
- 權重前處理（重排、shuffle）可能造成 allocator 碎片。【MI350P】aiter MoE 前處理碎片 ~12 GB，
  KV 少了 14 GB/GPU。比較 `reserved − allocated`。
- 放不下的大表放 host。【MI350P】Engram 表 ~203 GB → `ENGRAM_HOST_TABLE=1`。
- decode graph 上限：`MAX_CUDAGRAPH_CAPTURE_SIZE = min(MAX_NUM_SEQS × (spec_tokens+1), 512)`，
  否則 graph 吃掉 KV。

## 6. 啟動環境變數【AMD 為主；NVIDIA 對應項自行查】

| 變數 | 值 | 原因 |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG` | 1 | kernel 參數放 device memory；c1 decode 139→191 tok/s；vLLM 映像預設也是 1 |
| `HSA_NO_SCRATCH_RECLAIM` | 1 | 不回收 scratch。**用 `-e VAR=${VAR:-1}` 傳**：`-e VAR` 不帶值且 host 未設時，docker 會把映像內建的值刪掉 |
| `GPU_MAX_HW_QUEUES` | 預設 | 改 2 無穩定增益 |
| `ulimit -c 0` | — | 避免 core dump 塞滿磁碟 |
| `TOKENSPEED_KERNEL_VERBOSE` | 不設 | 只在盤點時開 |
| `PYTHONFAULTHANDLER` | 1 | 崩潰時有 Python stack |

驗證 env 真的進了 server：
```bash
docker exec $CONTAINER bash -c 'for p in $(pgrep -f "tokenspeed[:][:]scheduler"); do tr "\0" "\n" < /proc/$p/environ | grep TOKENSPEED_; done'
```

## 7. 崩潰分流表【通用】

| 症狀 | 先做的實驗 | 可能原因 |
|---|---|---|
| 啟動時 OOM（RCCL/NCCL init） | `stop_servers.sh`，確認記憶體 < 8 GB 再啟動 | 上一格的 rank 沒死 |
| 單一極長序列崩潰、並行多條較短不崩 | 固定長度二分（330K、381K、425K…） | kernel workspace/預算寫死 |
| `HIP_LAUNCH_BLOCKING=1` 或 `--disable-overlap-schedule` 就不崩 | 兩者分開試 | host 端 race / overlap 下 workspace 衝突 |
| `--max-total-tokens` 加大、關 prefill graph、降 gmu 都還崩 | 排除記憶體壓力 | 不是 KV 預算問題 |
| `OUT_OF_RESOURCES`、`Available Free mem: 0 MB`、高負載長 prefill 中 | 降 gmu 0.04；查 scratch reclaim；`scan_spills.py` | runtime scratch 配置失敗 |
| NCCL/RCCL watchdog timeout | 看哪個 rank 先死 | 某 rank 已崩潰，其他 rank 在等 |
| 第一個 batch ValueError | 看 kernel 名稱 | override 到不支援的格式 |
| server 回 `/v1/models` 但不生成 | `wait_ready.sh`（真的要 1 token） | gateway 活著、engine 已死 |

原則：**縮小 repro 再修**。不要在 sweep 中 debug。

## 8. 【MI350P】實例摘要

- 4× MI350P gfx950、每卡 128 CU / 4 XCD、144 GB HBM、**PCIe-only（無 xGMI）**、TP4。
- DSv4.1-Flash：hidden 5120、moe_intermediate 2304、384 experts top-6 MXFP4、40 層、vocab 129280、
  DSpark 投機解碼 block 5。
- bring-up 中遇到並修掉的：indexer logits 預算寫死 32 MiB（長 prompt 崩潰）→ 512 MiB；
  indexer key 以 byte 載入（`do_not_specialize`）→ 允許 specialize 用 16 B 向量載入，3.8×；
  寫死 MI355X 幾何；core dump 塞滿磁碟；docker env 傳遞吃掉 `HSA_NO_SCRATCH_RECLAIM`。
