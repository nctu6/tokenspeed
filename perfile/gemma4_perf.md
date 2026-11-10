# gemma-4-31B-it perf on sm90 (GPU5, bf16)

Workload: `vllm bench serve` random, out 256, ignore_eos.
- 1K: in 1024, C=1/8/32, N=max(16,8*C), max-model-len 131072
- 128K: in 126976, C=1 (N=3) / C=4 (N=8)
Scripts: remote `g4perf/` on sm90 build tree (`/workspace/nctu6/build/merge1012/g4perf/`), tmux `nctu6_test` window `g4chain*`.
Model: `/models/google/gemma-4-31B-it`. SERVED_MODEL_NAME=test.

Kernel / tree: build merge1012 tokenspeed @ 19d4dc1d (detached HEAD; plane-packing on 4e34b409) + local uncommitted gemma3-era wins (mixed-batch / prefill-graph / kvstore DMA etc.).

## Chain 1 (12:38–12:57 Taipei) — vLLM baseline + TS base start fail

TokenSpeed `base` FAILED START (timeout before `/v1/completions` ready). vLLM `vllm/vllm-openai:latest` on :8800 GPU5 finished full suite. CHAIN1DONE 04:57:37 UTC.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| vllm 1K | 1 | 45.02 | 588.52 | 6611.71 | 19.99 | 20.29 | 20.77 |
| vllm 1K | 8 | 338.81 | 713.61 | 808.74 | 20.90 | 23.22 | 21.48 |
| vllm 1K | 32 | 809.75 | 1787.23 | 3309.84 | 32.64 | 38.89 | 94.98 |
| vllm 128K | 1 | 4.85 | 30491.00 | 30537.07 | 87.21 | 87.28 | 88.78 |
| vllm 128K | 4 | 7.25 | 58099.90 | 118536.25 | 322.05 | 424.63 | 2825.38 |

Notes vs gemma3-27b (same host): gemma4 C1 TTFT ~589 ms vs gemma3 ~106 ms; C32 out ~810 vs ~1068. Expect larger model + different attn (Gemma4) to dominate; still the comparison target for TokenSpeed.

## Chain 2 (12:57–13:18 Taipei) — TS base + mix DONE (CHAIN2DONE 05:18 UTC)

tmux `g4chain2` on GPU5. TokenSpeed `base` (no mixed-batch, chunked_prefill_size=8192, token_capacity=153984, attn_backend=None, prefill graph buckets up to 2048) finished 1K suite; 128K all Bad Request. Then re-served `mix` (`--enable-mixed-batch --chunked-prefill-size 4096`, token_capacity=152704, buckets to 4096); mix 1K finished, 128K still fail. CHAIN2DONE 05:18:02 UTC.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS base 1K | 1 | 49.26 | 133.35 | 147.66 | 19.85 | 19.91 | 21.17 |
| TS base 1K | 8 | 312.97 | 819.82 | 3000.76 | 22.44 | 25.18 | 25.01 |
| TS base 1K | 32 | 403.45 | 12704.37 | 15843.42 | 25.00 | 25.84 | 27.35 |
| TS base 128K | 1 | 0 (fail) | — | — | — | — | — |
| TS base 128K | 4 | 0 (fail) | — | — | — | — | — |

vs vLLM 1K: C1 TS wins (49.3 vs 45.0 tok/s; TTFT 133 vs 589); C8 slightly behind (313 vs 339); C32 far behind (403 vs 810) with TTFT blow-up (12.7s vs 1.8s) — concurrency/scheduling bottleneck, not decode TPOT (TPOT still ~25ms).

128K root cause (server log): `The input (126976 tokens) exceeds the engine's maximum input length (61182 tokens)` despite `--max-model-len 131072`. Effective max input ~61K from KV/page budget on 31B (token_capacity≈154K but usable max-input capped). gemma4-specific long-context capacity issue vs gemma3; need raise usable context (mem/page/parent sizing) before 128K benches can run on TokenSpeed.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS mix 1K | 1 | 48.73 | 450.23 | 4433.97 | 18.83 | 22.67 | 22.09 |
| TS mix 1K | 8 | 311.15 | 778.14 | 3074.14 | 22.75 | 24.32 | 25.43 |
| TS mix 1K | 32 | 401.42 | 12444.74 | 16114.25 | 26.52 | 27.96 | 29.90 |
| TS mix 128K | 1 | 0 (fail) | — | — | — | — | — |
| TS mix 128K | 4 | 0 (fail) | — | — | — | — | — |

mix ≈ base on 1K (no C32 win); C1 TTFT worse than base (450 vs 133). 128K still capped at max_req_input_len=61182.

## Chain 3 (13:56–14:52 Taipei) — plane packing; CHAIN3DONE 06:52 UTC

Root cause of tiny KV: `_mha_plane_prefix` had scoped planes per cache group for layer_kv_geometry models, so each parent carried all 6 groups' planes while each group filled only its own (~5/6 idle). Uncommitted dirty fix in `ordinary.py` returns group-agnostic `unit` for every model (pack places two full CacheBlocks per parent on shared planes). Effect vs chain2 base: parent_bytes 62.9MB→10.5MB, parents 1203→7228, token_capacity 153984→925184, max_req_input_len 61182→131071. Quality OK (fox/lazy dog).

tmux `g4chain3` on GPU5: `base2` + `mix2` full suite DONE. CHAIN3DONE 06:52:10 UTC.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS base2 1K | 1 | 49.25 | 136.49 | 148.45 | 19.84 | 19.98 | 23.02 |
| TS base2 1K | 8 | 312.69 | 913.98 | 941.95 | 22.09 | 22.20 | 26.96 |
| TS base2 1K | 32 | 753.27 | 2307.31 | 3657.14 | 33.58 | 41.97 | 41.50 |
| TS base2 128K | 1 | 2.24 | 83755.61 | 83962.00 | 120.07 | 120.15 | 126.59 |
| TS base2 128K | 4 | 2.79 | 209394.99 | 331346.18 | 616.28 | 1096.91 | 142.72 |

base2 128K C4: 4/8 successful. vs vLLM 1K: C1 still wins; C8 slightly behind; C32 recovered 403→753 (vLLM 810). vs vLLM 128K: TS slower C1 2.24 vs 4.85, C4 2.79 vs 7.25.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS mix2 1K | 1 | 49.24 | 132.87 | 136.48 | 19.87 | 20.04 | 20.45 |
| TS mix2 1K | 8 | 310.87 | 676.49 | 970.37 | 23.17 | 24.46 | 28.05 |
| TS mix2 1K | 32 | 752.17 | 1428.17 | 6887.36 | 37.05 | 41.12 | 457.39 |
| TS mix2 128K | 1 | 2.25 | 83013.03 | 83031.76 | 120.41 | 120.49 | 122.76 |
| TS mix2 128K | 4 | 2.77 | 158694.76 | 340345.40 | 805.59 | 1068.89 | 2440.85 |

mix2 ≈ base2 (1K C32 752≈753; 128K C1 2.25≈2.24; 128K C4 2.77≈2.79). Still ~2× slower than vLLM 128K. Cache: token_capacity≈917K–925K, max_req_input_len=131071. Plane packing later committed as `19d4dc1d` (after chain4).

## Prof1 (14:52–14:59 Taipei) — PROF1DONE; long-prefill hotspots

Warm ~30K prompt + profile 64-out on :8801/:8802. Trace `g4perf/trace/20261014-065844-TP0.trace.json.gz`. Top kernels (~11.5s total):

| ms | % | count | kernel |
|---|---|---|---|
| 5762 | 50.0% | 40 | `_fwd_kernel` (Triton prefill attn) |
| 1663 | 14.4% | 640 | `_fwd_grouped_kernel_stage1` |
| ~3500 | ~30% | many | nvjet GEMM (sm90 tst_*) |
| 149 | 1.3% | 3400 | FlashAttnFwdSm90 (FA3) |

Bottleneck for ~30K prefill is Triton `_fwd_kernel` (50%), not FA3. GEMM second. Matches gemma4-specific long-context gap vs vLLM.

## Chain 4 (15:00–15:26 Taipei) — FA2 extend + KVS auto; CHAIN4DONE 07:26 UTC

tmux `g4chain4` on GPU5. `FA2X=512` serve `fa2x` (FlashInfer FA2 extend JIT head_dim 512), then `KVS=auto` serve `fa2kvs`. Quality OK. CHAIN4DONE 07:26:24 UTC. Plane packing committed locally as `19d4dc1d` after this chain.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS fa2x 1K | 1 | 49.29 | 128.98 | 137.34 | 19.86 | 19.93 | 23.41 |
| TS fa2x 1K | 32 | 766.59 | 2126.14 | 3459.14 | 33.54 | 41.25 | 36.72 |
| TS fa2x 128K | 1 | 3.42 | 44117.18 | 44134.28 | 120.10 | 120.16 | 125.65 |
| TS fa2kvs 1K | 1 | 51.09 | 130.64 | 149.03 | 19.14 | 19.22 | 24.14 |
| TS fa2kvs 1K | 32 | 775.48 | 2193.40 | 3445.27 | 32.80 | 40.79 | 35.94 |
| TS fa2kvs 128K | 1 | 5.03 | 44249.08 | 44317.88 | 25.86 | 25.93 | 33.56 |

vs base2: FA2 alone cuts 128K C1 TTFT 84s→44s and lifts out 2.24→3.42; KVS=auto then slashes decode TPOT 120→25.9 ms and lifts out to **5.03** (beats vLLM 4.85). 1K C1 fa2kvs 51.1 vs vLLM 45.0; 1K C32 775 vs vLLM 810 (close). Prefill still ~2× slower TTFT than vLLM on 128K (44s vs 30s) — Triton `_fwd_kernel` from PROF1 remains the long-prefill hotspot.

## Chain 5 (15:28–15:40 Taipei) — fa2kvs 1K C8 DONE; interrupted mid 128K C4

tmux `g4chain5` on GPU5. `FA2X=512 KVS=auto` serve `fa2kvs2`. 1K C8 finished; 128K C4 reached ~12% (1/8) then the chain was interrupted when a corrupted `chain5.sh` (mistakenly rewritten to gemma3 `def3`/`g3def`) spawned a second competing serve on GPU5. Both stopped ~07:40 UTC. `serve.sh` now maps `FA2X`→`TOKENSPEED_FLASHINFER_FA2_EXTEND` and `KVS`→`TOKENSPEED_TRITON_DECODE_KV_SPLITS`. Local commit `19d4dc1d` still in place.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL |
|---|---|---|---|---|---|---|---|
| TS fa2kvs 1K | 8 | 323.00 | 714.09 | 885.20 | 22.06 | 24.37 | 25.53 |

vs vLLM 1K C8 338.8/713.6/20.9 — slightly behind on out tok/s, TTFT similar. Combined with chain4: fa2kvs 1K C1 51.1 / C8 323 / C32 775 vs vLLM 45.0 / 338.8 / 809.8.

## Chain 6 (15:44–16:04 Taipei) — fa2kvs vs fa2x 128K C4; CHAIN6DONE 08:04 UTC

tmux `g4chain6` on GPU5. Redo `fa2kvs` 128K C4 (`FA2X=512 KVS=auto`), then `fa2x2` 128K C4 (`FA2X=512 KVS=legacy`). CHAIN6DONE 08:04:27 UTC.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | notes |
|---|---|---|---|---|---|---|---|---|
| TS fa2kvs 128K | 4 | 5.57 | 110118.17 | 173960.51 | 286.11 | 538.18 | 36.45 | 4/8 ok |
| TS fa2x 128K | 4 | 4.91 | 110072.55 | 173746.05 | 382.92 | 634.40 | 141.35 | 4/8 ok |

vs vLLM 128K C4 7.25 / 58099 / 322 — fa2kvs closer on out (5.57 vs 7.25) with better TPOT (286 vs 322) but ~2× worse TTFT. KVS=auto wins decode vs fa2x legacy (TPOT 286 vs 383; out 5.57 vs 4.91). Prefill TTFT still the gap.

## Prof2 / Chain 7–8 (16:04–16:12 Taipei) — PROF2DONE + CHAIN7/8DONE

tmux `g4chain7` after CHAIN6DONE: FA2X=512 KVS=auto ~30K profile (trace `g4perf/trace2/20261014-081106-TP0.trace.json.gz`). Total kernel ~6.6s (vs prof1 ~11.5s). CHAIN7DONE 08:12:08; CHAIN8DONE 08:12:10 (summary only).

| ms | % | count | kernel |
|---|---|---|---|
| 2293 | 34.7% | 40 | FlashInfer `BatchPrefillWithPagedKVCacheKernel` (FA2 extend) |
| ~3130 | ~47% | many | nvjet GEMM (sm90 tst_*) |
| 155 | 2.3% | 3400 | FlashAttnFwdSm90 (FA3) |
| 111 | 1.7% | 640 | `_fwd_grouped_kernel_stage1` |

vs PROF1: Triton `_fwd_kernel` (50%) gone — FA2 extend takes prefill attn. Remaining long-prefill cost is GEMM-heavy; FA3 still tiny. Local commit still `19d4dc1d`.

## xe_chain (16:12–17:16 Taipei) — xdef/xmix DONE; xmix 128K hung

tmux `g4xe` / `xe_chain.sh` on GPU5. Defaults suite (`xdef`, no FA2X/KVS env) then mixed (`xmix`). Planned `xg3def` not reached — hung kill at 09:16:51 UTC. Plane packing commit `19d4dc1d` still local-only.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | notes |
|---|---|---|---|---|---|---|---|---|
| TS xdef 1K | 1 | 51.03 | 127.38 | 137.90 | 19.17 | 19.28 | 19.64 | |
| TS xdef 1K | 8 | 323.24 | 686.62 | 876.01 | 22.14 | 24.37 | 23.02 | |
| TS xdef 1K | 32 | 773.97 | 2165.22 | 3494.58 | 32.99 | 40.82 | 39.42 | |
| TS xdef 128K | 1 | 5.04 | 44208.41 | 44254.42 | 25.81 | 25.94 | 30.23 | |
| TS xdef 128K | 4 | 5.58 | 109917.80 | 173575.28 | 285.70 | 537.19 | 72.76 | 4/8 ok |
| TS xmix 1K | 1 | 50.16 | 129.89 | 137.41 | 19.50 | 23.59 | 23.82 | |
| TS xmix 1K | 8 | 319.90 | 649.82 | 984.77 | 22.54 | 23.73 | 27.66 | |
| TS xmix 1K | 32 | 770.33 | 1333.88 | 3600.86 | 36.42 | 40.18 | 437.04 | TTFT better than xdef |
| TS xmix 128K | 1 | 0 (hang) | — | — | — | — | — | hung ~32m at first Mix batch |

xdef ≈ fa2kvs → defaults already pick FA2 extend + KVS=auto. vs vLLM: 1K C1 wins; C8/C32 close; 128K C1 5.04 beats 4.85; 128K C4 5.58 vs 7.25 (TTFT still ~2×).

xmix 128K C1 hang: py-spy MainThread in `Event.synchronize` → `_commit_forward_results`; forward thread in `decode_num_kv_splits` / triton decode during `prefill_graph.replay` (2 sampling params / Mix batch). Manual kill 09:16:51 UTC.

## xe_diag / xe_diagchain (17:18–17:38 Taipei) — root cause; XDIAGDONE; diag stopped early

tmux `g4xd` then manual stop. Blocking `CUDA_LAUNCH_BLOCKING=1` diag: XDIAGDONE 09:25:43 UTC; short curls stayed 200; Mean TTFT 47609.82 (1/1) — hang not reproduced under blocking.

`xe_diag2` async (default mix): hang suspected 09:35:03 (same stack as xmix: MainThread `Event.synchronize` → `_commit_forward_results`; forward in decode/prefill_graph.replay with 2 sampling params). Legacy leg started 09:35:21 then **diag chain stopped manually 09:38:29 (root cause found)** — no nopg leg, no XDCHAINDONE.

**Root cause / fix (uncommitted on `19d4dc1d` tree):** `event_loop._should_commit_head` — do not commit an extend dispatched in the *same* round (that blocks the scheduler thread on the whole prefill and races planning). Env `TOKENSPEED_COMMIT_DISPATCHED_EXTEND=1` restores old behavior. Also uncommitted: length-aware short-prompt-ahead admission + `mixed_long_chunk` cap when decodes share a round; prefill_graph replay/eager stats logging. Large dirty tree remains (FA2 extend, KVS, plane packing already committed as `19d4dc1d`).

## xe_chain2 (17:41–… Taipei) — xmixf DONE; xg3f mid-run on tmux `g4xe2`

Script: mixed 128K after FA2 plan-race fix (`xmixf`), then gemma3 `xg3f` regression. Started 09:41:49 UTC. `xmixf` serve: post-capture ~3.5m stall then :8801 up 09:47:23; quality OK. xmixf 128K suite DONE ~09:57 UTC (no hang). Then `down` + `g3perf/serve_def.sh xg3f` loading on GPU5 (~09:58); XE2DONE not yet. Plane packing still `19d4dc1d` local-only; commit-head / mixed_long_chunk / short-ahead still uncommitted.

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | notes |
|---|---|---|---|---|---|---|---|---|
| TS xmixf 128K | 1 | 5.00 | 44682.78 | 44781.87 | 25.70 | 25.77 | 28.57 | 3/3 ok |
| TS xmixf 128K | 4 | 5.48 | 86856.23 | 178130.72 | 387.88 | 545.41 | 1204.79 | **8/8 ok** |

vs hung xmix 128K C1 and vs fa2kvs 128K C4 5.57/110118/286 (4/8): plan-race fix unblocks mix+128K; C4 success 4→8 and TTFT 110s→87s (still worse than vLLM 7.25/58099/322). C1 ≈ xdef/fa2kvs (~5.0 vs vLLM 4.85). health fails: 2 in xmixf serve log. Next: gemma3 xg3f 1K C1/C8/C32 regression then XE2DONE.

## xe_chain2 DONE (XE2DONE 10:08:35 UTC) + commits e21b18f2, d1a04425

**Hang root cause (verified):** FlashInfer FA2 paged-extend `plan(non_blocking=True)` stages the work partition in the wrapper's pinned host buffer + async H2D. One wrapper per geometry was re-planned by the next forward (host runs ahead: overlap + launch queue) before the previous copy executed → forward ran on another batch's partition → GPU spin (serving) / `cudaErrorInvalidAddressSpace` (isolated test). Not KV-split related: the decode/KV-split frames were only where the host blocked on a full launch queue (native py-spy: `cuLaunchKernel` → `sched_yield`); 2nd repro hung in a pure prefill replay. `CUDA_LAUNCH_BLOCKING=1` hides it.
Fix `e21b18f2`: ring of 2 wrappers per geometry + event after each plan; slot re-planned only after its event. Test `test_fa2_extend_replan_while_gpu_lags_keeps_each_plan` (fails pre-fix with illegal address, passes 3/3 post-fix) + ring unit test. Note: remote tree also carries uncommitted event_loop commit-head edits from another session; xmixf ran with both.

Mixed default for Gemma4 (`d1a04425`, `MIXED_BATCH_DEFAULT_ARCHITECTURES` + chunk 4096; `--disable-mixed-batch` overrides; test in test_cli_config_compat.py, 74 passed).

### Before (defaults without mixed = xdef)
| C | 1K tok/s / TTFT / P99 ITL | 128K tok/s / TTFT / P99 ITL | 128K ok |
|---|---|---|---|
| 1 | 51.0 / 131 / – | 5.04 / 44208 / 30.2 | 3/3 |
| 32 / 4 | 774 / 2165 / 39.4 | 5.58 / 109918 / 72.8 | **4/8** |

### After (mixed default = xmix 1K, xmixf 128K)
| C | 1K tok/s / TTFT / P99 ITL | 128K tok/s / TTFT / P99 ITL | 128K ok |
|---|---|---|---|
| 1 | 50.2 / 130 / 23.8 | 5.00 / 44683 / 28.6 | 3/3 |
| 8 | 319.9 / 650 / 27.7 | – | – |
| 32 / 4 | 770 / 1334 / 437 | 5.48 / 86856 / 1205 | **8/8** |

P99 ITL cost accepted (same trade as gemma3); success + TTFT win.

gemma3 regression xg3f (vs la43 60.2/105/16.25, 397/497/18.27, 987/955/28.77): C1 60.61/105.2/16.15, C8 394.5/504.1/18.37, C32 978.6/983.7/28.92 → within noise.

## xe_chain3 (18:10–18:30 Taipei) — chunk sweep DONE; XE3DONE 10:30:50 UTC

Script: 128K C1 TTFT vs `--chunked-prefill-size` / `--max-prefill-tokens` 8192 then 16384 (mixed default).

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | notes |
|---|---|---|---|---|---|---|---|---|
| TS xck8192 128K | 1 | 5.05 | 44157.82 | 44176.59 | 25.69 | 25.74 | 28.72 | 2/2 ok |
| TS xck16384 128K | 1 | 5.10 | 43555.64 | 43612.94 | 25.85 | 25.96 | 27.16 | 2/2 ok |

vs prior defaults (chunk 4096): xdef 5.04/44208/25.81, xmixf 5.00/44683/25.70 — **chunk 4K/8K/16K ≈ same 128K C1 TTFT** (~43.6–44.7 s; not the lever). Still ~1.45× slower TTFT than vLLM 4.85/30491/87. Decode still much better (TPOT ~26 vs 87). Local commits through `d1a04425` (see gap analysis + xe_chain4 for FA4).

## 128K prefill gap analysis (Oct 14 evening)

- Chunk size is not the lever (mixed default, 128K C1 TTFT): 4096 44.68 s, 8192 44.16 s, 16384 43.56 s.
- GEMMs are already efficient: prof2 (32K) nvjet ≈ 3.4 s for 2.0 PFLOP ≈ 590 TFLOP/s → ~13.6 s at 128K.
- Head-512 attention is the gap. FlashInfer FA2 microbench (q chunk 4096, 32/4 heads, d512, page 64): 171–174 TFLOP/s at prefix 8K/64K → per 64K-prefix chunk ≈ 104 ms × 10 layers × 31 chunks ≈ 32 s of the 44 s TTFT. vLLM (30.5 s) runs its own FA4-cute fork with an SM90 hdim≤512 path; upstream Dao-AILab flash_attn/cute (main, 2026-10-13) and our venv still cap SM90 at 256 — not copyable/usable.
- TokenSpeed-native Triton prototype (GQA-packed rows: BT tokens × 8 q-heads per kv head, paged K/V, online softmax), `g4perf/kb/bench512.py`: best BT8/BN32/w8/s2 = 161/149 TFLOP/s (≈ FA2, correct to 5e-4); larger tiles exceed 227 KB smem. Not a win → not committed.
- Next step needed for ~2×: a warp-specialized wgmma/TMA hdim-512 forward (CuTe DSL or CUTLASS): page=64 matches a 64-row KV tile (2 × 256-col TMA boxes), split the 512-wide O accumulator across two consumer warpgroups (each 64×256 fp32), producer warp for paged TMA loads. Estimated target ≥300 TFLOP/s → 128K C1 TTFT ≈ 30 s.

## xe_chain4 (18:48–… Taipei) — FA4 CuTe SM90 hd512 extend; mid-run on tmux `g4xe4`

Local commit `a828ca84` (Mac `nctu6` ahead 8, unpushed): SM90 head_dim 512 paged extend on vendored CuTe-DSL FlashAttention (`tokenspeed_kernel.thirdparty.fa4_sm90`). Serve `xfa4` defaults (mixed + FA4 path; no FA2X/KVS env). Quality OK. Engine: token_capacity=910976, max_req_input_len=131071. JIT warmup emits cutlass AuxData UserWarning (non-fatal).

| setup | C | out tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | notes |
|---|---|---|---|---|---|---|---|---|
| TS xfa4 1K | 1 | 51.02 | 131.62 | 140.03 | 19.16 | 19.44 | 22.52 | ≈ xdef/fa2kvs |
| TS xfa4 1K | 8 | (mid-run) | — | — | — | — | — | started ~10:56 UTC |
| TS xfa4 1K | 32 | — | — | — | — | — | — | queued |
| TS xfa4 128K | 1 | — | — | — | — | — | — | queued (key TTFT vs FA2 ~44s / vLLM 30.5s) |
| TS xfa4 128K | 4 | — | — | — | — | — | — | queued |

1K C1 ≈ prior FA2 defaults (51.0/131 vs xdef 51.0/127) — short-context parity expected; FA4 win target is 128K prefill TTFT. XE4DONE not yet. sm90 build tree still shows HEAD `19d4dc1d` + dirty; FA4 kernel is live via installed `tokenspeed_kernel` / Mac commit `a828ca84`.

## a828ca84 — SM90 hd512 CuTe extend (vendored FA4 fork, user-authorized) — xe_chain4 (XE4DONE 11:06:47 UTC)

Kernel microbench (page 64, chunk 4096, 32/4 heads): FA4 345–360 TFLOP/s vs FA2 155–174. Kernel tests: new test_fa4_sm90_hd512 + fa2 file 16 passed/1 skip; whole tokenspeed-kernel/test/ops/attention 1377 passed (test_triton_dsv4_attention 2 fails pre-existing, same with FA4 off).

### Before (mixed default, FA2 hd512: xmix 1K / xmixf 128K)
| run | tok/s | mean TTFT ms | mean TPOT | P99 ITL |
|---|---|---|---|---|
| 1K C1 | 50.16 | 130 | 19.50 | 23.8 |
| 1K C8 | 319.9 | 650 | 22.54 | 27.7 |
| 1K C32 | 770.3 | 1334 | 36.42 | 437 |
| 128K C1 | 5.00 | 44683 | 25.70 | 28.6 |
| 128K C4 | 5.48 | 86856 | 387.9 | 1205 (8/8) |

### After (xfa4, defaults, no env)
| run | tok/s | mean TTFT ms | mean TPOT | P99 ITL | vLLM tok/s / TTFT |
|---|---|---|---|---|---|
| 1K C1 | 51.02 | 132 | 19.16 | 22.5 | 45.0 / 589 |
| 1K C8 | 321.2 | 639 | 22.48 | 28.4 | 338.8 / 714 |
| 1K C32 | 773.8 | 1247 | 36.57 | 432 | 809.8 / 1787 |
| 128K C1 | **7.11** | **29432** | 25.76 | 30.3 | 4.85 / 30491 |
| 128K C4 | **8.09** | **56761** | 268.8 | 731 (8/8) | 7.25 / 58100 |

128K now beats vLLM on both throughput and TTFT. Remaining gaps: 1K C8/C32 throughput (−5% / −4%), driven by mixed-round TPOT (36.6 vs vLLM 32.6) and P99 ITL.

## Follow-up (Oct 14 night): FA4 hd512 tile search + 1K C8/C32 gap triage

Tile search (prefix 64K, chunk 4096; `g4perf/kb/tune_fa4.py`): (64,64,RS=F,overlap=T) default 347 TF; RS=T 345; overlap=F 353 (+1.7%); n=32 210; n=48 illegal access; n=80/96 and m=128 fail to compile. Config space is exhausted at ~350 TF; more needs kernel work (e.g. split d_v=512 over two consumer WGs). Not committed.

1K gap triage: mixed is not the cause — xdef (non-mixed) and xfa4 (mixed) have equal throughput (774) and duration (84.7 s vs vLLM 80.9 s). Median ITL 27.95 vs vLLM 27.11 (C32), 21.40 vs 20.65 (C8); bs1 decode TPOT is faster than vLLM (19.16 vs 20.0). So the loss is in rounds that interleave 1K prefills with decodes at concurrency, not the steady decode kernels. xe_prof3 (3 s GPU trace at C32) landed in the ramp-up window (prefill GEMMs 79%), and decode kernels inside CUDA graphs are not in the trace — needs a later window + graph-node tracing (or nsys) to attribute.

## Follow-up (Oct 14 late): decode split, kernel breakdown, qkv-norm copy fix

xe_split (C32, defaults before fix; cols duration inTok outTok/s meanTTFT medTPOT medITL):

```
cols: duration inTok outTok/s meanTTFT medTPOT medITL
TS dec 50.11 2048 1307.96 188.63 24.31 24.23
TS pre 55.01 524288 9.31 3344.05 0.00 1.71
vLLM dec 52.28 2048 1253.64 2086.27 23.51 23.47
vLLM pre 56.01 524288 9.14 3422.18 0.00 0.00
XSPLITDONE
```

xe_dprof (eager bs32 decode, per-kernel): GEMMs identical to vLLM (same nvjet tiles, same avg us). TokenSpeed extra: 3 direct_copy kernels per layer (q/k/v reshape of split slices before the norms), ~0.45 ms/step. Fixed by commit gemma4: run q/k/v norms on strided head views of the fused projection.

After fix (xe_chain5, cols tok/s meanTTFT P99TTFT meanTPOT P99TPOT P99ITL):

```
dec Benchmark duration (s): 49.02  Output token throughput (tok/s): 1336.84  Median TPOT (ms): 23.80  Median ITL (ms): 23.70
xqkn 1K c1 51.12 124.61 138.71 19.15 19.26 22.30
xqkn 1K c8 328.60 813.23 6223.00 21.24 23.03 27.05
xqkn 1K c32 788.15 1297.06 3494.24 35.62 39.37 423.16
xqkn 128K c1 7.17 29137.93 29204.81 25.75 25.77 30.10
xqkn 128K c4 8.16 56342.17 117604.54 266.39 365.32 730.10
  graph:: 512 replayed (362 mixed), 0 eager, padding 5.3% of replayed tokens (prefill_graph.py:1565)[0m
XC5DONE
```

Mixed-round chunk sweep (xe_chain6) and master:

```
xch2048 1K c8 330.26 539.73 2819.33 22.16 23.72 118.65
xch2048 1K c32 794.60 920.88 3537.67 36.72 39.51 223.11
xch8192 1K c8 328.59 667.20 900.83 21.81 22.79 30.96
xch8192 1K c32 786.33 1887.12 3401.38 33.42 37.92 55.49
XC6DONE
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
58 passed, 5 skipped, 52 warnings in 9.63s
pick: base c32=788.15 c8=328.60 best=none (788.15)
```

Conclusion: no chunk beat 4096 by >1% at 1K C32 (2048: +0.8% within noise, halves P99 ITL; 8192: -0.2%). Default kept at 4096. Remaining 1K C32 gap vs vLLM ~2.7% (788 vs 810), decode TPOT 23.80 vs 23.51.

## xfuse quality-verified (Oct 14 evening)

False alarm: raw `/v1/completions` without chat template on gemma-4-it produced
repetitive garbage ("France is France…"). Chat `/v1/chat/completions` via
`g3perf/quality.py` on fused HEAD **matches** `xfa4_quality` baseline exactly
(XQ3DONE). Bisect `xe_san2` (a828 gemma4 + FA4 on/off/eager) also looked
"broken" under the same raw-completions probe — FA4/fused were not the cause.

### Before (xqkn = strided q/k/v norms, no fused kernels)
| run | tok/s | mean TTFT | mean TPOT | P99 ITL |
|---|---|---|---|---|
| 1K C1 | 51.12 | 125 ms | 19.15 | 22.3 |
| 1K C8 | 328.60 | 813 ms | 21.24 | 27.1 |
| 1K C32 | 788.15 | 1297 ms | 35.62 | 423 |
| 128K C1 | 7.17 | 29.1 s | 25.75 | 30.1 |
| 128K C4 | 8.16 | 56.3 s | 266.4 | 730 |

### After (xfuse = fused block-boundary + q/k norm+RoPE, chunk 4096)
| run | tok/s | mean TTFT | mean TPOT / med ITL | vLLM tok/s / TTFT | result |
|---|---|---|---|---|---|
| 1K C1 | **53.13** | **120 ms** | 18.42 / — | 45.0 / 589 | win |
| 1K C8 | **340.75** | **618 ms** | 21.13 / 20.11 | 338.8 / 714 | win |
| 1K C32 | 803.86 | **1235 ms** | 35.07 / **26.89** | 809.8 / 1787 | −0.73% (med ITL already better) |
| 128K C1 | **7.26** | **28.9 s** | 25.02 | 4.85 / 30.5 s | win |
| 128K C4 | **8.19** | 58.1 s | 257.7 | 7.25 / 58.1 s | win |

Decode-only (in32/out1024, C32): xfuse 1375.62 tok/s med TPOT 23.10 vs xqkn 1336.84 / 23.80.
Quality: MATCH xfa4 baseline. Commits: `481859a8` kernels, `2c9e66d1` gemma4 wiring.
Remaining: close 1K C32 ~0.7% (duration 81.53 s vs vLLM 80.93 s).

## xvn — v_norm folded into q/k norm+RoPE launch (`95adc60b`), quality MATCH

| run | tok/s | mean TTFT | mean TPOT | P99 ITL | vLLM |
|---|---|---|---|---|---|
| 1K C1 | 53.32 | 120 ms | 18.36 | 18.8 | 45.0 |
| 1K C8 | 342.41 | 611 ms | 21.04 | 23.6 | 338.8 |
| 1K C32 | 802.53 | 1282 ms | 34.95 | 419 | 809.8 |
| 128K C1 | 7.21 | 29.1 s | 25.02 | 27.1 | 4.85 |
| 128K C4 | 8.17 | 57.2 s | 262.3 | 727 | 7.25 / 58.1 s |

Decode-only C32: 1380.93 tok/s (xfuse 1375.62). 128K no regression. C32 gap is the mixed-round P99 ITL (419 vs vLLM 95).

## xdyn16 — decode-count-aware mixed prefill budget (`14408c73`, TOKENSPEED_MIXED_DECODE_REF=16 default), quality MATCH

Before = xvn (above). After:

| run | tok/s | mean TTFT | mean TPOT | P99 ITL | vLLM |
|---|---|---|---|---|---|
| 1K C1 | 53.26 | 123 ms | 18.36 | 21.3 | 45.0 |
| 1K C8 | 340.92 | 786 ms | 20.46 | 25.5 | 338.8 |
| 1K C32 | 801.64 | **1108 ms** | 35.64 | **272** (was 419) | 809.8 |
| 128K C1 | 7.22 | 29.1 s | 25.04 | 28.2 | 4.85 |
| 128K C4 | 8.18 | **56.9 s** | 263.0 | 728 | 7.25 / 58.1 s |

P99 ITL −35% and C32 TTFT −14%, 128K unchanged, but 1K C32 throughput unchanged (−1.0% vs vLLM): the C32 throughput gap is not the prefill bursts. Unit tests: scheduler C++ 624 passed, pytest 66 passed / 5 skipped. ref=8 / ref=24 sweep queued in xe_dyn.sh.

## MIXED_DECODE_REF sweep + C32 steady-state profile (Oct 15 00:00–01:00)

| ref | 1K C32 tok/s | 1K C8 TTFT | 1K C8 tok/s | C32 P99 ITL | 128K C4 TTFT |
|---|---|---|---|---|---|
| 8 | 800.64 | **623 ms** | 341.51 | **155 ms** | 56.9 s |
| 16 | 801.64 | 786 ms | 340.92 | 272 ms | 56.9 s |
| 24 | 801.61 | 790 ms | 341.60 | 350 ms | 56.9 s |

Default set to ref=8 (amended `1ce04649`). Throughput is flat across ref, so the C32 gap is not prefill bursting.

C32 steady 5 s torch-profiler window, graphs on (ref=8): TokenSpeed GPU idle 2.1%, GEMM 72%, attention 21%, norm 4%; mixed-output `torch.cat` 0.4% (not the cause). vLLM same window: GEMM 68%, attention 24%, but the vLLM profiler run slowed to 218 tok/s (TS 726), so per-step comparison is not apples to apples. vLLM runs prefill as rare 8K steps (5.5 ms GEMMs), TS as many smaller mixed rounds (~2K-token GEMMs, 112x128 / 256x128 tiles). Next: per-shape GEMM microbench of the mixed-round M sizes vs one 8K step.

## xref8 — defaults (no env), `1ce04649` (MIXED_DECODE_REF=8), quality MATCH

| run | tok/s | mean / median TTFT | mean TPOT | P99 ITL | vLLM |
|---|---|---|---|---|---|
| 1K C1 | 53.31 | 412 / 121 ms (one client-side cold outlier, P99 4.1 s) | 17.21 | 19.8 | 45.0 |
| 1K C8 | 341.60 | 623 ms | 21.06 | 24.7 | 338.8 |
| 1K C32 | 800.80 | 787 ms | 36.86 | 154 | 809.75 |
| 128K C1 | 7.22 | 29.1 s | 24.96 | 27.5 | 4.85 |
| 128K C4 | 8.17 | 59.1 s | 255.5 | 730 | 7.25 / 58.1 s |

C1 outlier: server prefill→first decode for the first request is 0.69 s (later 0.38 s); the rest of the 4 s is the bench client's first request (cold docker client / connection). Not a server bug.

## GEMM microbench vs M (gemma-4-31B weight shapes, 60 layers/step, `g4perf/gbench.py`)

| M | step GEMM ms | us/token |
|---|---|---|
| 32 (decode) | 15.75 | — |
| 1024 / 1056 | 78.1 / 90.9 | 76 / 86 |
| 2048 / 2056 / 2080 / 2176 | 178.6 / 173.7 / 181.7 / 189.0 | 87 / 84 / 87 / 87 |
| 2560 | 249.8 | 98 (only bad point) |
| 4096 / 4128 | 358.5 / 362.1 | 88 / 88 |
| 8192 | 708.2 | 86 |

Per-token GEMM cost is flat (85–89 us) from 1K to 8K; M alignment and mixed-round splitting are not the C32 gap. Mixing 32 decodes into a 1024 prefill (90.9 ms) beats running them apart (93.9 ms).

## Noise check (Oct 15 01:20–01:45): same seeds both sides, 1K C32

| seed | TokenSpeed | vLLM | diff |
|---|---|---|---|
| 11 | 802.46 | 732.23 (first bench after vLLM start) | +9.6% |
| 22 | 800.47 | 801.63 | −0.14% |

TokenSpeed seeded runs: 800.5–802.5 (TTFT 869–878 ms vs vLLM 1764–2829 ms). vLLM's earlier 809.75 is a favourable single sample; on a matched seed C32 is at parity within noise. 128K C4 single rerun on defaults: 8.17 tok/s, TTFT 58.18 s (previous 59.08 s; range across runs 56.3–59.1 s) — run-to-run variance, not a regression (MIXED_DECODE_REF cannot bind with ≤4 decodes).

## Attention microbench at C32 mixed-round shapes (`g4perf/abench.py`)

| shape | auto | fa3 | triton |
|---|---|---|---|
| decode hd256 sliding bs32 kv1152 | 159.0 us | 152.0 | 182.7 |
| decode hd256 sliding bs32 kv1280 | 156.9 | 154.3 | 178.3 |
| extend hd256 sliding 1/2/4×1024 | 269 / 301 / 369 | 264 / 296 / 368 | 389 / 511 / 763 |
| decode hd512 global bs32 kv1152 | 227.8 | n/a | 217.5 |
| extend hd512 global 1/2/4×1024 (auto = FA4 sm90) | 342 / 458 / 687 | n/a | 986 / 1754 / 3776 |

Auto selection is the fastest or within 1–4% on every shape; vLLM uses TRITON_FLASH_ATTN. No attention root cause; no code change this round.

## FA3 hd256 sliding decode: num_splits / pack_gqa sweep (`g4perf/fa3bench.py`, Oct 15 ~01:50)

Serving already runs hd256 decode on FA3 (profile shows FlashAttnFwdSm90 + FlashAttnFwdCombine, i.e. split-KV + combine); the abench "auto" row used no backend solution and is not the serving path.

| shape | heuristic (splits=0) | best forced | worst forced |
|---|---|---|---|
| bs32 kv1100 | 151.2 us | 151.2 (0/None) | 168.5 (splits≥2) |
| bs32 kv1280 | 158.7 | 155.3 (splits=1) | 161.3 |
| bs24 kv1200 | 129.4 | 129.4 (0/None) | 180.4 (splits=2) |

All outputs bit-identical. Heuristic is best or within 2%. bs32×1.1K×16 kv heads×256×bf16 K+V ≈ 577 MB per layer step → 151 us ≈ 3.8 TB/s, i.e. at HBM roofline. No FA3 change made.

## Follow-up (Oct 14 late): decode split, kernel breakdown, qkv-norm copy fix

xe_split (C32, defaults before fix; cols duration inTok outTok/s meanTTFT medTPOT medITL):

```
cols: duration inTok outTok/s meanTTFT medTPOT medITL
TS dec 50.11 2048 1307.96 188.63 24.31 24.23
TS pre 55.01 524288 9.31 3344.05 0.00 1.71
vLLM dec 52.28 2048 1253.64 2086.27 23.51 23.47
vLLM pre 56.01 524288 9.14 3422.18 0.00 0.00
XSPLITDONE
```

xe_dprof (eager bs32 decode, per-kernel): GEMMs identical to vLLM (same nvjet tiles, same avg us). TokenSpeed extra: 3 direct_copy kernels per layer (q/k/v reshape of split slices before the norms), ~0.45 ms/step. Fixed by commit gemma4: run q/k/v norms on strided head views of the fused projection.

After fix (xe_chain5, cols tok/s meanTTFT P99TTFT meanTPOT P99TPOT P99ITL):

```
dec Benchmark duration (s): 49.02  Output token throughput (tok/s): 1336.84  Median TPOT (ms): 23.80  Median ITL (ms): 23.70
xqkn 1K c1 51.12 124.61 138.71 19.15 19.26 22.30
xqkn 1K c8 328.60 813.23 6223.00 21.24 23.03 27.05
xqkn 1K c32 788.15 1297.06 3494.24 35.62 39.37 423.16
xqkn 128K c1 7.17 29137.93 29204.81 25.75 25.77 30.10
xqkn 128K c4 8.16 56342.17 117604.54 266.39 365.32 730.10
  graph:: 512 replayed (362 mixed), 0 eager, padding 5.3% of replayed tokens (prefill_graph.py:1565)[0m
XC5DONE
```

Mixed-round chunk sweep (xe_chain6) and master:

```
xch2048 1K c8 330.26 539.73 2819.33 22.16 23.72 118.65
xch2048 1K c32 794.60 920.88 3537.67 36.72 39.51 223.11
xch8192 1K c8 328.59 667.20 900.83 21.81 22.79 30.96
xch8192 1K c32 786.33 1887.12 3401.38 33.42 37.92 55.49
XC6DONE
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
58 passed, 5 skipped, 52 warnings in 9.63s
pick: base c32=788.15 c8=328.60 best=none (788.15)
```

## 2026-10-15 hd512 decode num_stages=3（commit fb37dbc5）
- bytes 更正：global 層 K==V（attention_k_eq_v），bs4×128K 723 µs = 2.97 TB/s ≈ 89% HBM；C32 shape (bs32×~1.1K) 僅 ~0.68 TB/s，延遲受限。
- 單元測試：kernel 200 passed、runtime 66 passed（exit code 0）。品質逐字一致。

| case | before | after | vLLM |
|---|---|---|---|
| 1K C32 s22 tok/s | ~801 | 801.27 | 801.63 |
| 1K C32 s33 tok/s | – | 801.15 | 809.53 |
| C32 TPOT med ms | 37.6 | 36.5 | 32.3/33.0 |
| 128K C4 TTFT med s | 57.2 (xvn) / 58.2,59.1 (xref8) | 58.15 | – |
| ref=4 C32 s22/s33 | – | 800.5/799.6, TPOT 37.15 (更差) | – |

## 2026-10-15 hd512 decode for the C32 shape (bs32 × ~1.1K): split-KV / K==V tile reuse — not adopted
Microbench (µs, triton grouped; 128K C4 shape for reference):

| config | bs32 kv1152 | bs32 kv1280 | bs24 kv1200 | bs4 kv131072 | bit-exact |
|---|---|---|---|---|---|
| default (2 splits, 3 stages) | 220.3 | 218.6 | 216.8 | 720.3 | – |
| K==V tile reuse (tl.trans) | 225–277 | 222–289 | 225–238 | 1107 | yes |
| TARGET_WAVES=4, MIN=256 | 238.7 | 236.3 | 219.8 | 721.0 | no (0.0156) |
| TARGET_WAVES=4, MIN=128 | 216.5 | 217.2 | 221.7 | 731.5 | no |
| TARGET_WAVES=8, MIN=256 | 217.2 | 211.5 | 216.2 | 720.1 | no |
| TARGET_WAVES=8, MIN=128 | 210.0 | 215.8 | 215.5 | 722.8 | no |

Best gain ~10 µs/global layer (~0.1 ms of a 36.5 ms step) and it breaks verbatim output; the transpose costs more than the saved load. Both reverted.

## 2026-10-15 1K C32 decode-only profile (32 × 1K in / 1536 out, 3 s window, graphs on)

| | TokenSpeed | vLLM |
|---|---|---|
| kernel time / window | 2976 / 3034 ms | 3007 / 3057 ms |
| GPU idle | 2.2% | 1.6% |
| gaps > 100 µs (excl. profiler start) | 3 (0.9 ms) | 8 (3.3 ms) |
| GEMM / attn / norm / other | 1809 / 1052 / 70 / 45 ms | 1800 / 1024 / 90 / 92 ms |
| median ITL (decode only) | **27.13 ms** | 27.71 ms |

Decode steps are already graph-captured end to end with <2.5% host gaps; TokenSpeed's pure decode step is faster
than vLLM's. The C32 suite TPOT gap (36.5 vs 33.0 ms) therefore comes from mixed prefill+decode rounds, not from
inter-step host gaps or sampling. FA3 prepare_varlen + combine is 19 ms/3 s (0.17 ms/step) — upper bound of any fusion.
Next: chunked_prefill_size sweep (xe_chk.sh).

## 2026-10-15 chunked_prefill_size sweep (xe_chk.sh), seeds 22/33

| chunk | 1K C32 s22 | 1K C32 s33 | C32 TPOT med | C32 P99 ITL | 1K C8 s22 (TTFT med) | 128K C1 TTFT | 128K C4 TTFT med |
|---|---|---|---|---|---|---|---|
| 4096 (default, xst3) | 801.27 | 801.15 | 36.5 | 155 | 341.4 (710) | 28.96 s | 58.15 s |
| **2048** | **823.24** | **820.21** | 35.7 | 89 | 348.7 (441) | 29.09 s | **68.19 s (+17%)** |
| 1024 | 687.26 | 651.39 | 42.3 | 91 | 342.2 (326) | – | – |
| vLLM | 801.63 | 809.53 | 33.0 | ~100 | – | – | – |

2048 beats vLLM on C32 (quality verbatim) but regresses 128K C4 TTFT, so it is not the default. Next (xe_knob.sh):
keep chunk 4096 and apply the mixed-round knobs that 2048 implies one at a time
(MIXED_LONG_CHUNK=1024, LONG_PREFILL_FLOOR=512, SHORT_PROMPT_TOKENS=2048, all three).

## 2026-10-15 mixed prefill cap gets its own floor, ref default 4 (commit c632438d)
Root cause: capPrefillBesideDecodes floored the shrunk prefill share at the admission floor (chunk/4 = 1024),
so beside 32 decodes every mixed round still carried 1024 prefill tokens and the ref knob did nothing
(ref=4 alone: 800 tok/s). chunk 2048 only helped because it halved that floor, but it also halved
MIXED_LONG_CHUNK and regressed 128K C4 TTFT to 68 s (knob sweep: MIXED_LONG_CHUNK=1024 alone -> 67.9 s;
LONG_PREFILL_FLOOR=512 or SHORT_PROMPT_TOKENS=2048 alone -> C32 ~805-810, 128K unchanged).
Fix: TOKENSPEED_MIXED_PREFILL_FLOOR (default chunk/8) for the cap, admission floor unchanged, MIXED_DECODE_REF default 4.
Tests: gtest 625 passed (rc 0, + MixedPrefillFloorSuite), pytest 66 passed (rc 0). Quality: verbatim match.
Numbers below are the default path (no env), fixed seeds; columns tok/s | mean TTFT ms | P99 TTFT ms | mean TPOT | P99 TPOT | P99 ITL.

| case | before (xst3 / xb22) | after (xmfl, default) | vLLM |
|---|---|---|---|
| 1K C1 | 52.65 / 124 | **53.59** / 103 / P99 ITL 21.2 | 45.02 / 589 |
| 1K C8 | 341.36 / 710 | **341.84** / 623 / P99 ITL 39.9 | 338.81 / 714 |
| 1K C32 s22 | 801.27 / 873 / P99 ITL 154.6 | **813.43** / 865 / P99 ITL 94.3 | 801.63 |
| 1K C32 s33 | 801.15 / 878 / P99 ITL 155.1 | **815.57** / 837 / P99 ITL 91.4 | 809.53 (TPOT 33.0) |
| 128K C1 | 7.29 / 28962 | **7.29** / 28930 | 4.85 / 30491 |
| 128K C4 s22 | 8.23 / 56849 (xb22) | **8.24** / 56443 | 7.25 / 58100 |
| 128K C4 s33 | 8.23 / 56108 (xb22) | **8.22** / 56381 | – |

Remaining: C32 TPOT 35.8 vs vLLM 33.0 ms (mixed rounds); C8 margin is thin (+0.9%).

## 2026-10-15 decode-skip + 8192 chunk with a 4096 mixed reference + first-chunk stream fix

Changes measured here (all default path, no env knobs): (1) Gemma 4 takes an 8192-token chunk budget while the mixed-round
sizing (`capPrefillBesideDecodes`, floors, long chunk) stays anchored to a 4096 reference chunk (`TOKENSPEED_MIXED_REF_CHUNK`);
Gemma 3 stays at 4096. (2) The overlap schedule no longer plans a decode for a request whose declared budget the forwards in
flight already cover (that step ran after the request was done and the next prompt queued behind it, ~7 ms at C1).
(3) smg always sends a stream's first chunk, even when the first token decodes to empty text (chat and completions), so such a
request no longer reports its whole generation time as TTFT.

Protocol: cold server per case group, `vllm bench serve` random, out 256, ignore_eos, same seeds on every side, mean of the seeds
(C1 s41/s42, C8 s52/s63, C32 s22/s33, 128K C1 s71, 128K C4 s72/s73), N=max(16,8*C). before = the pushed default (b18e37bf),
after = this change, vLLM = 0.31.0 (`vllm/vllm-openai:latest`). Quality: verbatim identical to the reference output for the
after build.

| Case | System | TPS (tok/s) | TTFT mean (ms) | TPOT median (ms) |
|---|---|---|---|---|
| 1K C1 | before | 53.38 | 123.2 | 18.32 |
| 1K C1 | after | **55.39** | **112.6** | **17.67** |
| 1K C1 | vLLM 0.31.0 | 48.53 | 113.4 | 20.23 |
| 1K C8 | before | 341.98 | 744.6 | 21.07 |
| 1K C8 | after | **348.22** | **632.8** | **20.25** |
| 1K C8 | vLLM 0.31.0 | 334.30 | 700.9 | 20.87 |
| 1K C32 | before | 813.61 | 807.5 | 37.19 |
| 1K C32 | after | **834.02** | **750.0** | 36.22 |
| 1K C32 | vLLM 0.31.0 | 804.73 | 1712.7 | **32.80** |
| 128K C1 | before | 7.28 | 28991.7 | 24.21 |
| 128K C1 | after | **7.44** | **28387.8** | **23.53** |
| 128K C1 | vLLM 0.31.0 | 4.83 | 30670.6 | 87.44 |
| 128K C4 | before | 8.22 | 58228.0 | 301.82 |
| 128K C4 | after | **8.28** | 58432.8 | **293.68** |
| 128K C4 | vLLM 0.31.0 | 7.23 | 60219.5 | 355.11 |

Tail latency (prose only, same seeds, mean of seeds): 1K C8 P99 ITL is 24.1 ms vs vLLM 22.3 ms and P99 TTFT 884 ms vs 811 ms
(before: 4367 ms, the empty-first-chunk artifact fixed by the smg change); 1K C32 P99 TTFT is 3.47 s vs 3.30 s and P99 ITL
97 ms vs 96 ms; C1 P99 ITL 22.4 ms vs 21.0 ms. Correction: the "1K C8 P99 ITL 39.9 ms regression" recorded in the tables
above was single-run noise, not a regression; same-seed cold runs give 24.1 ms (before 25.7 ms).

Remaining loss: **1K C32 TPOT, 36.22 ms vs vLLM 32.80 ms.** It is mostly a metric effect, not a slower decode step:
TPOT = (latency - TTFT) / 255, so the 960 ms lower TTFT (750 vs 1713 ms) by itself adds ~3.8 ms to TPOT at the same end-to-end
time. End to end the build is faster: mean request latency 9742 ms vs 10173 ms (-4.2%) and throughput +3.6%. Pure decode is
already ahead (decode-only profile at C32: 27.1 vs 27.7 ms median ITL, same GEMM kernels, faster attention). Trading TTFT
back toward vLLM's (DECODE_REF 16-24 variants) moves TPOT to ~31-33 ms but costs 1-3% throughput without the decode-skip, so
the default keeps the lower TTFT and the higher throughput. 128K C4 TTFT is flat within run-to-run spread (58.2 vs 58.4 s;
range across runs 56.3-59.1 s) while TPS and TPOT improve; 128K C1/C4 do not regress.
