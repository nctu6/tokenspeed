# 07 直接改 kernel 的流程

host/排程/env 層級的小槓桿用完之後（【MI350P】c32/c64 剩下的都 ≤5%），剩下的收益在 kernel 本身。
這份檔案是改 kernel 的固定流程。照順序做，不要跳步。

## 目錄
1. 選 kernel
2. 開 worktree 與 env 開關
3. 讀懂現有 kernel
4. Microbench（真實 shape）
5. 離線編譯與資源檢查（不需要 GPU）
6. 調參
7. 數值正確性
8. 整合與端到端
9. 常見 kernel 改法清單
10. AMD 特有注意事項

---

## 1. 選 kernel【通用】

從 P5 的暴露時間表選：
- 暴露佔比最大、且實測/下界 ≥ 1.3 的類別。
- 在該類別內，用 trace 找總時間最大的 3 個 kernel 名稱與它們的 shape（grid、M/N/K、rows）。
- 確認該 kernel 是不是 arch 專屬的（`explain_selection`）。若落在 portable 實作，
  第一步就是寫/移植 arch 專屬版。

## 2. worktree 與 env 開關【通用】

```bash
cd $UNIEINFRA
git worktree add .wt-<name> -b <branch> <base-commit>
```
- 實驗用 `ENGINE_PYTHONPATH=/workspace/.wt-<name>/python:/workspace/.wt-<name>/tokenspeed-kernel/python:...`
  指過去（依需要包含 kernel 套件路徑）。主 checkout 保持在基準 commit。
- **正在被 benchmark 使用的 worktree 不能改。** 要改就再開一個。
- 新路徑放在 env 開關後：
  ```python
  _ENABLE = os.environ.get("TOKENSPEED_<FEATURE>", "0") == "1"
  ```
  並把 `-e TOKENSPEED_<FEATURE>` 加到 `serve_unieinfra.sh` 的傳遞清單。
- 新的 kernel 也可以註冊成另一個候選（`register_kernel(..., priority=低)`），用
  `TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>=<name>` 切換做 A/B。

## 3. 讀懂現有 kernel【通用】

寫下（記在 experiments.json 的 experiment 欄）：
- grid 怎麼分（每個 program/CTA 負責什麼）、persistent 與否。
- 每個 program 的 load：從 HBM 讀幾 bytes、是否向量化（16 B）、是否重複讀。
- 用了哪個矩陣指令（MFMA/MMA 形狀）。
- LDS/shared 用量、register 數、spill。
- 它的理論下界（bytes / 頻寬，或 FLOPs / 峰值）。

對照：vLLM / aiter / FlashInfer / repo 內 CUDA 版本怎麼做同一件事。
「port from CUDA」= 讀 repo 內 CUDA kernel 的資料流設計，移植到目標硬體；不是去調 vLLM。

## 4. Microbench【通用】

規則：
- **真實 shape**：從 trace 取 M（rows）、N、K、expert 數、context 長度。decode 要涵蓋
  每個 concurrency 會出現的 rows（例：6、48、96、192、384）。
- **真實 TP**：通訊 kernel 要多 GPU 跑；計算 kernel 用 TP 後的 shape。
- **graph 內計時**：把 N 次呼叫 capture 進 CUDA/HIP graph，replay 多次取平均。eager 計時會把 launch
  開銷算進去。範例：`scripts/bench_allreduce.py` 的 `timed()`。
- 同一支程式內同時跑舊版與新版，輸出表格：shape、舊 µs、新 µs、max err。
- 把 microbench 放在 worktree 的 `.bench/` 下（不進 PR），或專案 `scripts/`。

範本：
```python
def timed(fn, inner=43, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (iters * inner) * 1e6   # µs per call
```

## 5. 離線編譯與資源檢查（GPU 被佔用時也能做）【AMD 實例；NVIDIA 用 ptxas -v 對應】

Triton/Gluon kernel 可以在沒有 GPU 的情況下指定 target 編譯，檢查 LDS、register、spill：
```python
import tokenspeed_triton as triton
from tokenspeed_triton.backends.compiler import GPUTarget
from tokenspeed_triton.experimental.gluon._runtime import GluonASTSource as ASTSource
target = GPUTarget("hip", "gfx950", 64)          # arch、warp size
src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs)
k = triton.compile(src, target=target, options={"num_warps": 4, "warp_size": 64})
print(k.metadata.shared)                          # LDS bytes
open("out.amdgcn","w").write(k.asm["amdgcn"])     # 搜尋 .amdhsa_private_segment_fixed_size（spill）
```
執行期快取也可以直接掃：`python3 scripts/scan_spills.py <triton cache dir>`。

為什麼重要：
- LDS 超過硬體上限 → 執行期失敗。
- spill（`private_segment_fixed_size > 0`）→ 慢，且在 AMD 上每個硬體佇列需要 scratch，
  KV 吃滿時會 `OUT_OF_RESOURCES`。【MI350P】`_mxfp8_block32_tiled` 2024 B/lane、
  `selected_attention_rows` 932 B/lane。新 kernel 不應引入新的大 spill。

## 6. 調參【通用】

- 小 M decode GEMM 這類對 shape 敏感的 kernel：每個 (N, K) 一張表、依 M 分段選 config。
  【MI350P】`tokenspeed-kernel-amd/.../gfx950/gemm/mxfp8/block32.py` 的 `BLOCK32_CONFIGS`
  就是 `{(N,K): [(max_M, Block32Config(...)), ...]}`。
- sweep 的維度：tile M/N/K、num_warps、waves_per_eu、split-K、stages、XCD remap。
- 換硬體時 config 一定要重調；CU 數不同時 split-K 與 grid 最先受影響。
- 用 vendor 的 tuning 工具時（例 aiter `gemm_moe_tune`）設定正確的 `cu_num`。
- 把 sweep 結果存檔（csv），寫進 experiments.json。

## 7. 數值正確性

見 02-correctness.md 第 5 節。新 kernel 在所有 microbench shape 上與 reference 比，
並在 graph capture 中比一次。然後 needle + GSM8K。

## 8. 整合與端到端

1. 在 worktree commit（無 AI 署名、有 Signed-off-by）。
2. 低雜訊 bench：固定 batch decode 或單 prompt prefill，對照 env 開/關。
3. 通過 → AgentX ABBA（06-ab-noise.md）。
4. 記錄 kernel 層級數字與端到端數字**兩者**。kernel 快 2× 但端到端中性也要記錄——
   代表瓶頸不在這裡，下一個人不必再做。

## 9. 常見 kernel 改法清單（依【MI350P】經驗的命中率排序）

1. 從裝置讀幾何，不寫死（CU、XCD、SM）。
2. 去掉不必要的 padding（heads、寬度）與 pre-fill（-inf 填充、zero）。
3. 讓寬度落進 arch 專屬 kernel 的支援範圍（補齊到支援的寬度，而不是落到 portable）。
4. 向量化載入（16 B）；檢查 `do_not_specialize` 是否讓 stride 失去對齊資訊。
5. 用正確的矩陣指令形狀（例：AMD 32×32×64 MFMA）。
6. 共用重複讀取（多個 spec row 共用 page load）——但要保持平行度，第一版常常更慢。
7. 兩段式演算法（histogram + compaction）取代多次全掃描（top-k）。
8. 移植 vendor 庫的小 M kernel（tiled + packed、fused split-K），依 CU 數重調。
9. 融合相鄰的小 kernel（尤其在 eager 路徑上：每個 launch 有 ~7 µs 空隙）。
10. 降低 register 壓力以消除 spill（拆 kernel、減 tile、改 accumulate 位置）。

## 10. AMD 特有注意事項【AMD】

- wave64；`waves_per_eu` 影響 occupancy 與 register 上限。
- 多 XCD：PID 依 XCD round-robin 分配，L2 每 XCD 獨立 → 需要 XCD remap 讓相鄰 tile 在同一 XCD。
- 同 priority 的多個 stream 會 round-robin 到 4 個 HW queue，可能和 compute 被序列化；
  要讓通訊與計算重疊，side stream 用 `torch.cuda.Stream(priority=-1)`。
  【MI350P】TBO 第一版 overlap = 0 ms，改 priority 後 2574 ms 的 AR 藏掉 2236 ms。
- Iris（GPU 直接 pull 的 all-reduce）的 programs 會佔 CU；和計算重疊時把 programs 降到 32。
- `HIP_FORCE_DEV_KERNARG=1`。
- ROCm graph：segment 數多且平均節點少時 clr 會判成 complex graph 走較慢的路徑（ROCm 7.2.4 clr
  `ScheduleNodesIntoBatches`）。

## 改 kernel 前必做：先量 roofline（#245–246 的教訓）

【通用】不要用規格表上的頻寬或算力估上限，要自己量：
1. `scripts/read_bw.py`：純讀 Triton kernel，掃 0.5–4 GB 與幾組 BLOCK/warps，取最大值當「實測讀取峰值」。`torch` 的 copy/sum 會低估。【MI350P】實測 3.13 TB/s（不是 MI355X 的 8 TB/s）。
2. 對目標 kernel 做逐 stage 微基準（範例：`scripts/moe_bw.py`），在實際會出現的 M（decode 批次、prefill chunk，以及 TBO 拆半後的大小）下算出有效 TB/s 與 TFLOPS。
3. 判斷：
   - 有效頻寬 ≥ 峰值的 85%：已經在頻寬上限，**不要改**。只有減少讀取的 bytes（例如更低精度、讓每次讀取的權重服務更多 token）才有用。
   - 頻寬與算力都低於 30%：才值得改。先試調參（tile、warps、atomic 還是 reduce），再試 grid 順序（XCD swizzle），最後才重寫 kernel。
4. 調參前，先確認生產環境實際會遇到的 M。【MI350P】調參在 M=8192 快 18%，但 TBO 把 8K chunk 拆成兩個 4096，生產上效果 ≈0。
5. 調參表常綁定特定模型的 shape 或啟用條件。例如 Kimi 的表要求 `swiglu_limit == 0`，DSv4.1 的 clamp 是 10，所以根本用不到那張表。要檢查你的模型是不是在走通用 heuristic。
6. XCD swizzle 或 GROUP_M 只有在權重重讀真的打到 HBM 時才有用。【MI350P】這兩個改動都在 ±1% 內，因為 MALL 已經接住重讀。
