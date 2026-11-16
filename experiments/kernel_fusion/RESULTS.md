# Where could custom or fused kernels pay off? Measurements on the small driving model, and what they imply for the big one

Host: Linux x86_64, 4 cores, Intel Xeon @ 2.1 GHz (AVX-512 capable), clang 18, Python 3.12, tinygrad 0.14.0, onnxruntime 1.32.0.
Nothing here ran on an iPhone or a Mac. **No phone speed is claimed anywhere.** Absolute times are for one x86 core and do not transfer;
structure (which layers dominate, how many launches, what fusion changes) is what to take from this.

**Noise warning.** The machine was shared: load average was 15-25 on 4 cores for almost the whole session. Every timing is pinned to one core
(`taskset -c 3`) and is the minimum (or median, as labelled) of repeated runs, and step times are reported as process CPU time as well as wall time
because wall time was inflated 1.2-2x by other jobs. Differences under ~10% between configurations are not resolved.
Stated precisely, the numbers are "best case seen under load"; treat them as +-10-15%.

## Answer first

**Small model, CPU.** Launch overhead is not the problem: a jitted launch costs about 15 us here, 161 launches is 2.4 ms, 1.4-1.7% of the step; 36 launches over 1 ms make
81% of the kernel time, and the 63 launches under 50 us make 0.6%. What the data does show is **generated-code quality**: tinygrad's kernels
run at 12-20 GOPS (matvec-style layers at 1.8) where a hand-written AVX2 kernel for the same arithmetic reaches 60-70 GFLOP/s on the same core,
and onnxruntime reaches 70-120. For one encoder block's FFN (two 1x1 convs + GELU + layer scale + residual, about 50 M MACs) tinygrad needs 8.5 / 5.6 ms
(stage 0 / stage 2); my C kernel needs 1.6 / 1.4 ms, a factor 5.3 / 4.0. Of that, **fusion itself** (5 passes -> 1 pass, hidden tensor kept in L1) is worth 1.9x at stage 0 (1.1x at stage 2) and
**kernel quality** (tinygrad's structure with a good GEMM micro-kernel) is worth about 3x. So the pay-off on a CPU is in arithmetic kernels (1x1 convs = 75% of the ops, 54-58% of the time, the stem conv 11%, and the
M=1 linear layers and temporal matmuls which tinygrad's CPU backend generates badly: 27% of time for 6% of ops). Hand fusion matters mostly where the intermediate is large (stage 0: 1.6 MB hidden tensor).
tinygrad's own scheduler already fuses elementwise work into the reducing kernel (377 ONNX nodes -> 161 launches, only 9 pure elementwise launches, 0.5% of time), and none of its fusion-related
switches that exist in 0.14.0 reduces the count by more than 2 without breaking things.
**Caveat that matters:** I cannot tell from here whether the same holds on Metal. The matvec problem is CPU-specific (tinygrad only applies its matvec optimisation when the backend has local memory, `codegen/opt/heuristic.py:62`), and a GPU's launch cost and
memory behaviour are different. The scripts are built to be re-run on a Mac (commands at the end).

**Big model.** From the kernel list alone (not run): 474 launches, 96.2 Gop, and the cost is concentrated in *large* kernels: 73 launches of >= 1 Gop carry 83% of the ops, 217 launches below 1 Mop carry 0.1%.
So launch overhead cannot matter, and fusing small kernels cannot matter. Custom kernels would matter only if (a) the phone GPU path (tinygrad/Metal or Core ML) delivers well below what a hand kernel could on the 73
big GEMM/conv launches (on this CPU the gap was 3-5x; unknown on Metal), and (b) the model is compute-bound rather than weight-bandwidth-bound: its 766 MB of fp16 weights read once per step is 15 GB/s at 20 Hz,
about a quarter of a third-party-quoted ~60 GB/s for an A18 Pro class memory system (see section 5); if the weights stream from DRAM every step, kernel work cannot beat that floor and only weight compression (int8/int4) or keeping fewer weights moves it.

## Measured / read / estimated

- **Measured here**: sections 1-4, the launch-cost microbenchmark, all tables with a time or a kernel count.
- **Read from the installed source**: the option list and its meaning (file:line below).
- **Read from the pickle analysis (static, not run)**: big-model launch/ops distribution.
- **Estimates**: anything labelled "estimate"; the phone-chip comparison is claims from third-party pages and vendor-quoted figures, not measurements.

---

## 1. Per-layer cost (ONNX nodes, `01_node_cost.py`)

Model `models/driving_supercombo_fp32.onnx`, 377 nodes. MACs are analytic from onnx shape inference (conv: out elements x Cin/group x kh x kw; Gemm: M*N*K; MatMul: out x K).
Total 847.0 M MACs = 1.694 G ops at 2 ops/MAC (tinygrad counts 1.836 G for the same graph, 1.720 G in the QCOM pickle; they count GELU's polynomial and other elementwise work).
Time: onnxruntime profiler, `SessionOptions.enable_profiling = True`, `profile_file_prefix`, `InferenceSession.end_profiling()` (listed in the installed
`onnxruntime/capi/onnxruntime_inference_collection.py:494-510` and the `SessionOptions` docstrings: "Enable profiling for this session", "The prefix of the profile file");
1 intra-op thread, sequential execution, `ORT_DISABLE_ALL` so each profile event is one original node (I assert 377 events, one per node; I did not check what the events look like with optimisation on, I only know that optimisation changes the wall time: below).
33 runs with state carried, first 3 discarded, **median per node over 30 runs**. The profile timer is 1 us; 80 nodes are below 5 us.
Whole run, no profiler: 46.6 ms median / 41.5 ms min with optimisations off; **31.4 / 28.3 ms with `ORT_ENABLE_ALL`** (ORT's own fusion and layout work is worth 1.5x). The per-node sum is 46.5 ms.
Single-thread onnxruntime uses AVX-512 kernels on this CPU; tinygrad below is limited to x86-64-v3 (AVX2), so ORT-vs-tinygrad compares different instruction sets as well as different code.

Stages (found by reading the graph; nodes 0-23 input/state prep, 24-25 normalisation, 26-134 encoder by `stem`/`stages.N`/`final_conv` weight names, 135-283 per-frame heads and three summarizers, 284-289 glue, 290-341 temporal transformer, 342-372 temporal heads and output concat, 373-376 state carry):

| stage | nodes | MMACs | MAC share | time ms | time share |
|---|---|---|---|---|---|
| encoder stage 2 (6 blocks, 128 px, 256 ch) | 45 | 323.4 | 38.2% | 11.94 | 25.7% |
| encoder stage 0 (2 blocks, 2048 px, 64 ch) | 14 | 115.9 | 13.7% | 6.97 | 15.0% |
| per-frame heads + summarizers | 149 | 16.5 | 2.0% | 6.09 | 13.1% |
| encoder stem | 6 | 122.8 | 14.5% | 5.36 | 11.5% |
| encoder stage 1 | 17 | 119.9 | 14.2% | 5.27 | 11.3% |
| encoder stage 3 | 17 | 111.8 | 13.2% | 4.27 | 9.2% |
| temporal transformer | 52 | 33.7 | 4.0% | 2.36 | 5.1% |
| state / input prep (original model: slices, gathers, casts of the image queue) | 24 | 0 | 0% | 2.03 | 4.4% |
| temporal heads + output concat | 31 | 2.6 | 0.3% | 0.96 | 2.1% |
| input normalisation (Sub, Div) | 2 | 0 | 0% | 0.74 | 1.6% |
| final conv + SE + pool, state carry, glue | 20 | 0.4 | 0.2% | 0.53 | 1.1% |

By op type (full table in `data/node_cost_report.md`): Conv 60 nodes, 93.8% of MACs, 67% of time; of the conv time **1x1 convs are 75% of MACs but 27% of time, depthwise 7x7 convs are 3.6% of MACs but 22.6% of time (12 nodes, 10.5 ms)**,
depthwise 3x3 6%, the one dense 3x3 stem conv 7.9%. Gemm (82 M=1 linear layers) 2.3% of MACs, 14.5% of time. Gelu 4.9%, Concat 2.0%, Cast 1.2%, all other elementwise ~4%.
So time is not proportional to arithmetic: low-arithmetic-intensity depthwise convs and M=1 Gemms are about 3-10x more expensive per MAC.

Top 15 nodes by time (of 46.5 ms): stem conv `node_conv2d` 3.67 ms (7.9%, 113 MMACs); ten depthwise 7x7 convs (`node_Conv_173/175/177/179/181/183/185/187/189/191`) 0.73-1.59 ms each;
`node_conv2d_1` (stem depthwise 3x3) 0.85; `node_linear` (the 2048->1024 Gemm) 0.72; fc2 of stage-0 block 0 0.72; one downsample conv 0.66. Full list `data/node_cost_report.md`.

Concentration: the **first 3 convolutions (full resolution) are 14.5% of MACs and 10.1% of time**; first 6: 18.4% / 15.2%; first 10: 25.2% / 21.7%. Top 3 nodes by time 14.5%, top 10 26.6%, top 20 40.8%;
97 of 377 nodes are needed for 90% of the time. The whole encoder is 93.8% of MACs and 73.4% of time. 227 nodes with zero MACs (movement, elementwise, norm, softmax) take 14.3% of time.
Cost is spread: no single layer exceeds 8%, the encoder's 12 identical blocks repeat the same three or four shapes.

## 2. Per-kernel cost in tinygrad on CPU (`tg_harness.py --per-kernel --map-weights`, `02_kernel_analysis.py`, `02b_launch_overhead.py`)

`DEV="CPU:CLANG:x86_64,x86-64-v3"` (`-march=native` crashes clang 18 here, as `experiments/tinygrad_metal/RESULTS.md` records), 1 pinned core, so the generator emits single-thread kernels
(`NUM_CPU_THREADS` comes from `sched_getaffinity`: `tinygrad/helpers.py:255-263`, threads in `renderer/cstyle.py:261`, `codegen/opt/heuristic.py:183`).
161 launches, 78 distinct kernels, 1.836 G ops (tinygrad `Estimates.ops`), 510 KiB generated C. Outputs match the reference: worst max abs diff 5.3e-4, worst mean 8.9e-6 over 24 steps.
Per-launch time is taken **in step order** (the whole step's launches run in sequence, repeated 15 times, minimum per launch, then minimum over 3 separate process runs): sum 139 ms. A launch run alone repeatedly
(hot cache) sums to 141-148 ms, so cache state is a minor effect here. Whole jitted step: 155 ms CPU-time minimum, 171 median (wall 189 / 251). The step also needs a state clone (about 18 ms for the original model's 7.8 MB image queue; reported separately in the JSON).
Each launch is mapped to a layer by matching its input buffers' contents to the ONNX initializers (sorted-value hash), so the layer names below are exact, not guessed.

By kernel class (`data/tinygrad_kernel_report.md`):

| class | launches | ops share | time ms | time share | GOPS |
|---|---|---|---|---|---|
| 1x1 convs (fc1, fc2 with fused scale+residual, downsample, stem.2) | 31 | 74.9% | 75.5 | 54.3% | 18.2 |
| linear layers, M=1 (matvec) | 72 | 2.1% | 20.6 | 14.8% | 1.8 |
| temporal transformer matmuls | 6 | 3.7% | 17.0 | 12.2% | 3.9 |
| stem dense 3x3 conv (normalisation fused in) | 1 | 13.6% | 16.0 | 11.5% | 15.6 |
| depthwise 7x7, with epilogue | 15 | 4.5% | 7.1 | 5.1% | 11.6 |
| depthwise 3x3, with epilogue | 14 | 1.1% | 1.8 | 1.3% | 11.3 |
| elementwise `E_*` | 9 | 0.1% | 0.65 | 0.5% | 2.5 |
| other reductions (norm, softmax, pooling) | 13 | 0.1% | 0.35 | 0.2% | 5.2 |

By stage: encoder stage 2 24.4% of time, stage 0 14.4%, stem 13.6%, temporal transformer 12.9%, per-frame heads 12.1%, stage 1 10.4%, stage 3 8.7%, temporal heads 2.1%.
Top launches: #5 stem conv (`r_16_64_32_4_4_24_3_3`) 16.0 ms (11.5%); the two temporal MLP matmuls (`r_3_128_4_3_512_4`, `r_3_512_4_3_128_4`) 7.2 and 6.4 ms; stage-0 fc1 (`r_64_512_4_3_16_4`) 5.4 ms x2; stage-1 fc1 3.8 x2; the stage-2/3 fc1s ~3.2 each.
Each of the 12 blocks contributes fc1(+GELU) and fc2(+scale+residual) as two launches.

**Fixed per-launch cost vs arithmetic.**
- Microbenchmark (`02b_launch_overhead.py`): K = 10..400 independent 16-element kernels in one TinyJit replay, time vs K: **14.8-15.1 us per launch** (fits on median and on min, same with 2 pinned cores). The intercepts are noise (-80..+140 us).
  This is the cost of one launch through tinygrad's CPU command ring when there is no arithmetic.
- Direct calls of tiny kernels (< 3000 ops) take 2.4-3.5 us; that is the harness cost (ctypes + timer), not the launch cost.
- 161 launches x 15 us = 2.4 ms = **1.4-1.7%** of kernel time + launch time. Launch time by size: 0-10 us launches (39) are 0.1% of kernel time, 10-50 us (24) 0.5%, 50-200 us (10) 0.7%, 200 us-1 ms (52) 17.5%, > 1 ms (36) 81%.
  Fixed cost would be 75% of the cost of the sub-10-us launches and 3% of the 0.2-1 ms ones, 0 of the big ones.
- Step CPU time (155 ms) minus summed kernel time (139 ms) is 16 ms (10%); about 2.4 ms of that is the launch estimate, the rest I cannot attribute (different timing paths, worker thread).
- A least-squares fit of time on ops over all launches (R^2 0.81) gives a 190 us intercept and 17 GOPS slope, but the classes differ in efficiency by 10x, so the intercept is not a launch cost and I do not use it.
- **On Metal this will be different** and cannot be separated here. The earlier GitHub-Mac run (51 ms for 161 launches, time spread evenly) is consistent with per-launch costs mattering more, but that is a virtual GPU and no decomposition exists. Run `02b_launch_overhead.py` with `DEV=METAL`.

Where the CPU time goes relative to a good implementation: encoder 1x1 convs 18 GOPS (tinygrad) vs 62-71 GFLOP/s for my hand kernel (AVX2) vs 70-125 for onnxruntime (AVX-512); M=1 linear layers 1.8 GOPS.
Reading the generated C for `r_512_256_4`: the reduction is one scalar accumulator updated with 4 dependent adds per iteration (`buf0 = buf0 + a0*b0 + a1*b1 + ...`), a latency-bound serial chain with no vectorised accumulators, which explains 1.8 GOPS. ORT does all 82 Gemms in 6.7 ms against 20.6 ms for tinygrad's 72 matvec launches (different ISA, so only indicative).

## 3. Fusion and related options (`03_sweep.sh`, `03b_variants.sh`, `04_graph_rewrites.py`)

Every run replays 24 steps of the reference with state carried and checks every step against `reference/driving_reference_fp32.npz`. Step time is CPU time of the jitted step, min over runs / median over runs (2 passes, interleaved; `data/sweep_table.md`).

### Options that exist in 0.14.0 (read from source) and what they did

| option | where (installed `tinygrad/`) | what it does | result here |
|---|---|---|---|
| `PCONTIG` (0 default) | `helpers.py:274`; `schedule/rangeify.py:256,267`; `schedule/indexing.py:257,274` | "partial contiguous in rangeify": lets a buffer be removed and its producer fused into the consumer more often (levels 1,2,3) | **1: 156 launches but wrong output (max diff 7972)**; **2 and 3: crash** (`AssertionError: CPU programs support at most 63 arguments, got 250`, a mega-kernel with 250 buffers; also 172 with `MAX_KERNEL_BUFFERS=8`, 196 with 16, 320 with 1000) |
| `SPLIT_REDUCEOP` (1) / `REDUCEOP_SPLIT_THRESHOLD` | `helpers.py:240`; `schedule/rangeify.py:73,86` | splits big reductions into two kernels | `=0`: no change (161 launches); nothing in this model was being split |
| `NOOPT` | `helpers.py:232`; `codegen/opt/postrange.py:346` | disables the hand-coded kernel optimisations (unroll, upcast, threads) | same 161/78; **step 1132 ms CPU, 7x slower**, ops count 2.448 G (the unoptimised kernels recompute more) |
| `MAX_KERNEL_BUFFERS` (device default CPU 31) | `helpers.py:248`; `schedule/rangeify.py:354,363` | caps buffers per kernel; fusion that would exceed it is cut | 4: 231 launches; 8: 169; default 31: 161; **1000: 159**. Time within noise except 4 (slower wall) |
| `WINO` | `helpers.py:236`; `mixin/op.py:1594` | Winograd for 3x3 stride-1 convs | no change at all (161, same ops): this model has no dense 3x3 stride-1 conv |
| `TRANSCENDENTAL=2` | `helpers.py:239`; `codegen/__init__.py:372` | forces tinygrad's own polynomial exp2/log2/sin even where the backend has them | 161 launches, same ops, time within noise (170 vs 171 ms median) |
| `BEAM=n` | `helpers.py:232`; `engine/realize.py:307`, `codegen/opt/search.py` | per-kernel search for opts by timing candidates | 1 run only (lean model, BEAM=1): 159 launches, ops 1.984 G (up), outputs ok, CPU min 140 / median 219 ms vs 164 / 191 for the same model without it. **Not resolved by this noise, one run**, and the search itself costs minutes |
| `JIT=2` | `helpers.py:234`; `engine/jit.py:73` | skips `graph_split_rewrite` (batching launches into one graph call), used on Metal | no effect on CPU (no graph runtime); the prior report says the same. `JIT_BATCH_SIZE` (`helpers.py:234`) only matters for devices that have a graph runtime, i.e. Metal; not measurable here |
| `NUM_CPU_THREADS`, `THREADS` | `helpers.py:263`; `renderer/cstyle.py:261` | CPU kernel thread count | not swept: the machine was oversubscribed (load 15-25), a multi-core run would be meaningless |
| `FLOAT16`, `IMAGE` | `helpers.py:233`; `mixin/op.py:1485,1593` | QCOM image path | not used (no effect on CPU, previous report) |

Flags that I expected and that **do not exist in 0.14.0**: `FUSE_ARANGE`, `FUSE_CONV_BW`, `NOFUSE`, and any per-op "do not fuse" switch; grep of `getenv(` and `ContextVar(` over `schedule/ engine/ codegen/ uop/ tensor.py mixin/` finds none. The only fusion controls are the ones above.
Also `PCONTIG=2/3` can only be tried on backends without the 63-argument limit (the CPU runtime asserts it in `runtime/ops_cpu.py` `__call__`); Metal's limit is 31 buffers (`DEVICE_MAX_BUFS`, `rangeify.py:354`), so it may fail there in the same way.

Table (`data/sweep_table.md`, `data/variants_table.md`):

| config | launches | distinct | ops G | worst max diff | step CPU ms min / median |
|---|---|---|---|---|---|
| base (original model) | 161 | 78 | 1.836 | 5.3e-4 | 155 / 171 |
| PCONTIG=1 | 156 | 76 | 1.886 | **7972, wrong** | 280 / 301 |
| PCONTIG=2, 3 | crash | | | | |
| SPLIT_REDUCEOP=0 | 161 | 78 | 1.836 | 5.3e-4 | 157 / 176 |
| NOOPT=1 | 161 | 78 | 2.448 | 6.8e-4 | 1132 / 1358 |
| MAX_KERNEL_BUFFERS=4 / 8 / 1000 | 231 / 169 / 159 | 109 / 86 / 76 | 1.836 | 5.3e-4 | 153 / 156 / 154 min |
| WINO=1 | 161 | 78 | 1.836 | 5.3e-4 | 157 / 165 |
| TRANSCENDENTAL=2 | 161 | 78 | 1.836 | 4.7e-4 | 159 / 170 |
| lean model | 159 | 76 | 1.886 | 5.3e-4 | 164 / 191 (this run) |

### Graph-level changes before tinygrad sees the model (`04_graph_rewrites.py`, lean fp32 model as base; ORT-verified against the reference, then run through tinygrad)

| variant | launches | distinct | ops G | worst max diff | step CPU ms min / median |
|---|---|---|---|---|---|
| original model (image queue inside) | 161 | 78 | 1.836 | 5.3e-4 | 155 / 171 |
| lean (`models/driving_supercombo_lean_fp32.onnx`) | 159 | 76 | **1.886** | 5.3e-4 | 164 / 191 |
| + normalisation folded into stem conv | 159 | 76 | 1.843 | 7.9e-4 | 161 / 172 |
| + layer_scale folded into fc2 (12 Mul nodes) | 159 | 76 | 1.843 | 9.5e-4 | 160 / 175 |
| + shared-input Gemms concatenated (3+5+5+2 Gemms -> 4 wide Gemms + slices) | **148** | 92 | 1.843 | 9.5e-4 | 156 / 173 |

- Lean vs original: 2 fewer launches (the image-queue slice/concat kernels) but 0.05 G more ops. Hypothesis (not verified by reading the lean kernels): with the image queue gone, tinygrad fuses cast+normalise into the stem conv's loads and recomputes them; folding the normalisation removes 0.043 G, which is consistent with that.
- Folding normalisation is exact only with an explicit `Pad` of the raw image with 127.5 (the original conv zero-pads the *normalised* image); a plain weight/bias fold would change the 1-pixel border. With the Pad, ORT max diff to reference is 4.6e-4, same as ORT's own fp32 noise level here. Launch count does not change: tinygrad already fused the normalise into the conv.
- Folding layer_scale also does not change the launch count (tinygrad already fused the Mul into fc2's kernel, as the buffer-content mapping shows: one launch reads fc2.weight, fc2.bias and layer_scale.gamma).
- Concatenating the first layers of parallel heads (the 3 summarizers, the 5+5 hydra heads' input layers, the 2 temporal heads) is the one rewrite that removes launches: **-11 (7%)**, at the cost of +16 distinct kernels. CPU time unchanged within noise. It is the kind of horizontal fusion a scheduler does not find; it would matter in proportion to launch cost, which is 15 us here (11 x 15 us = 0.17 ms), i.e. nothing on this CPU. The accuracy deltas (9.5e-4 vs 5.3e-4) are float reassociation, at the model's fp32 noise level.
- I did not run ORT's offline-optimised graph through tinygrad (it can emit non-standard ops).

## 4. Hand-written fused kernel (`fused_block.c`, `05_fused_block.py`)

Hot spot chosen from section 2: the 1x1-conv FFN of an encoder block (fc1 + GELU(tanh, as the model's Gelu nodes specify) + fc2 + layer-scale + residual add), 12 blocks, 54% of tinygrad time.
Two shapes, both 50.3 M MACs: stage 0 (C=64, hidden 192, 2048 px, hidden tensor 1.57 MB) and stage 2 (C=256, hidden 768, 128 px, hidden tensor 0.39 MB).
Inputs are the real activations of one reference step captured with onnxruntime; the check is against the real model's block output. All implementations (C, tinygrad, ORT) agree to <= 3e-5 absolute (output magnitudes 65 and 3.9).
C is AVX2+FMA (the ISA tinygrad generated for here), 4 output channels x 16 pixels register block, one thread, `clang -O3 -mavx2 -mfma`; GELU via a Cephes-style vector exp (`gelu(x) = x / (1 + exp(-2u))`).
Min of 60-100 calls, two separate runs (`data/fused_block_run1.txt`, `run2.txt`); run 2 shown (run 1 within 5%):

| implementation | stage 0 ms | GFLOP/s | stage 2 ms | GFLOP/s |
|---|---|---|---|---|
| tinygrad generated (2 launches) | 8.50 | 11.8 | 5.65 | 17.8 |
| C, unfused, 5 passes (gemm, bias+gelu, gemm, bias+scale+res) | 3.04 | 33.1 | 1.55 | 64.8 |
| C, 2 passes with fused epilogues (tinygrad's structure) | 2.59 | 38.8 | 1.57 | 64.1 |
| **C, 1 pass, hidden tile (16 px) kept in L1** | **1.61** | **62.4** | **1.42** | **71.0** |
| onnxruntime, optimisation off (AVX-512, 1 thread) | 1.48 | 68.0 | 0.85 | 119 |
| onnxruntime, optimisation on | 1.18 | 85.1 | 0.81 | 125 |

(GFLOP/s counts the two GEMMs at 2 flops per MAC and ignores the GELU's own work.)
Reading: (i) tinygrad's kernels are 4.0-5.3x slower than the hand kernel on this core; most of that gap (about 3x) is micro-kernel quality, not fusion: the 2-pass C version with the same structure as tinygrad is 3.3x / 3.6x faster. (ii) Going from 2 passes to a single fused pass adds 1.6x at stage 0, where the hidden tensor (1.57 MB, power-of-two row stride) spills out of L1, and only 1.1x at stage 2 where it is small. (iii) The unfused 5-pass version is 1.1-1.9x slower than the fused one, so fusion by itself is a real but second-order effect on a CPU with big caches. (iv) The mature library with AVX-512 is still 1.1-1.8x faster than my AVX2 fused kernel, so the "headroom" over generated code is bounded by what a good GEMM does, about 4-7x over tinygrad on this machine. (v) The GELU epilogue (an exp and a division per element) is a visible cost: stage 0's unfused version spends a full pass on it.
Limits: single kernel, hand-tuned for these shapes, AVX2 only, not applicable to Metal (different programming model), correctness checked on one frame. It says nothing about the depthwise 7x7 convs (22.6% of ORT time) or the M=1 layers, which I did not write kernels for.

## 5. Sizing the big model (`06_big_distribution.py`; static, from the pickle's per-launch estimates, not run)

Source: `data/big_small_launches.txt` (same allowlisted loader as `experiments/big_model`). These are tinygrad `Estimates` for each recorded launch, not measurements.

| | big | small (QCOM pickle) |
|---|---|---|
| launches / distinct names | 474 / 56 | 166 / 66 |
| ops per step | 96.23 G | 1.720 G |
| launches for 50% / 90% / 99% of ops | 44 / 88 / 132 | 14 / 30 / 80 |
| top 1 / 10 / 50 launches' share | 1.2% / 11.6% / 57.4% | 13.7% / 41.0% / 96.6% |
| median / mean / max Mop per launch | 0.6 / 203 / 1130 | 1.1 / 10.4 / 235 |
| launches by size: < 1 Mop | 246 launches, 0.1% of ops | 80 launches, 0.6% |
| 1 Mop - 100 Mop | 113 launches, 2.4% | 85 launches, 85.7% |
| 100 Mop - 1 Gop | 42 launches, 14.1% | 1 launch, 13.7% |
| >= 1 Gop | **73 launches, 83.3%** | 0 |
| ops / load-store bytes (`lds`) | 13.6 | 0.9 |

The big model is **many medium kernels, not a few huge ones**: 88 launches carry 90% of the ops, and a few kernel shapes repeat many times (the two heaviest names are 27 and 26 repeats of identical shapes, about 29 Gop each, i.e. 60% of all ops).
The big model's median launch is similar to the small model's (0.6 vs 1.1 Mop) but its heavy launches are 5-100x larger (up to 1.13 Gop vs 235 Mop), so per-launch overhead is negligible in comparison and the work is arithmetic (and weights).
The small model, by contrast, has 80% of its ops in launches of 10-100 Mop; it is the small-kernel regime where launch overhead is largest (still 1.7% on this CPU).

Sustained throughput needed at 20 Hz (arithmetic from the table): **1.92 Tops/s** (96.23 G x 20) for the big model versus 34 Gops/s for the small one (56x). Weights: 382.9 M fp16 values = 766 MB; reading them once per step is **15.3 GB/s** at 20 Hz if they are not cache resident (they cannot be); the pickle's `mem` estimate (unique bytes per launch, summed) is 1.216 GB/step = **24.3 GB/s**; the `lds` estimate (loads+stores, counts reuse) is 7.06 GB/step = 141 GB/s, which is an upper bound on traffic that good tiling reduces.
Scaling what I measured, as an *estimate*: tinygrad's 12-18 GOPS on this core would need 5-8 s per big step; a hand-kernel-grade 60-70 GFLOP/s about 1.5 s. Neither is a phone number; they only show the ratio to the small model (56x the work) and that kernel quality is a 3-5x lever, on this CPU.

Published chip figures (vendor or third-party claims; I did not verify any, I found no cited A18/A19 GPU TFLOPS figure and give none):
- Neural Engine: 16 cores, **35 TOPS for A18 / A18 Pro** per Apple's announcement as reported by Tom's Hardware / PhoneArena (https://www.tomshardware.com/phones/iphone/apples-a18-processor-powers-the-iphone-16-and-apple-intelligence, https://www.phonearena.com/news/apple-a18-pro-processor-specs-features_id161928); cpu-monkey and eatyourbytes list 38 TOPS (third-party databases, conflicting). The precision basis of the figure (int8 vs fp16) is not stated in what I saw. For A19 Pro I found 16 cores but **no reliable TOPS figure**; one listing repeats 35, others say Apple gave none (MacRumors https://www.macrumors.com/2025/09/09/iphone-17-a19-chip/ covers the A19/A19 Pro differences; core counts only).
- Memory: A18 Pro 8 GB LPDDR5X-7500 on a 64-bit bus, ~60 GB/s theoretical (third-party: https://lowendmac.com/2025/apple-silicon-a18-pro-chip-specs/); A19 Pro 12 GB, 76.8 GB/s per a third-party comparison table (https://apple-silicon.eustasy.workers.dev/current/iphones/). Notebookcheck quotes 75.8 GB/s in a different context. Theoretical peak, shared by CPU, GPU and ANE, never reached in practice.
- GPU: A18 Pro 6 cores, A19 Pro (iPhone 17 Pro) 6 cores (MacRumors / Apple); no cited FLOP/s figure found.

Comparison and its uncertainty: 1.92 Tops/s is 5.5% of 35 TOPS and the weight stream is 25% (A18 Pro, 15.3/60) or 20% (A19 Pro, 15.3/76.8) of the theoretical memory bandwidth, and the 1.2 GB/step `mem` figure is 40% of 60 GB/s. These ratios say the big model is not obviously impossible on paper and that **memory traffic, not arithmetic, is the tighter budget**; they say nothing about whether it runs, because (a) "TOPS" is a marketing peak for a precision this model may not use, (b) the ANE cannot take custom kernels and has per-layer shape/size limits I did not look up, (c) the GPU path's real efficiency is unmeasured, (d) 766 MB of weights plus activations against an app memory limit I did not look up, (e) sustained thermal behaviour at 20 Hz. The honest range is "unknown within one to two orders of magnitude".

What would have to be true for custom/fused kernels to matter for the big model: the measured evidence says launches are large enough that fusion of small kernels cannot move the total; custom kernels pay off only if the generated large GEMM/conv kernels on the phone GPU are far from the hardware's efficient frontier (on this CPU they were 3-5x), **and** if the weight stream (15 GB/s) leaves room, i.e. weights are fp16/int8 and not re-read more than once per step. If the large kernels are already memory-bound, kernels cannot help and weight compression/quantisation is the lever (`experiments/weight_compression`).

## 6. What failed or was skipped

- `PCONTIG=2` and `3` crash on the CPU backend (63-argument limit; 250 buffers needed, and 172, 196, 320 with `MAX_KERNEL_BUFFERS` 8, 16, 1000). `PCONTIG=1` produces wrong output (max diff 7972); the sweep's correctness check caught it.
- `WINO`, `SPLIT_REDUCEOP=0`, `TRANSCENDENTAL=2` changed nothing measurable. `BEAM=1`: a single run, noisy, not conclusive; BEAM=2+ not tried.
- No multi-thread/multi-core sweep (machine oversubscribed). No Metal, ANE, or any phone runs; `JIT_BATCH_SIZE`/`GRAPH_ONE_KERNEL` effects therefore unmeasured. Per-launch cost on a GPU is unmeasured.
- tinygrad per-launch timings use direct runtime calls (one launch, `wait=True`), min of repeats; launch cost in a jitted replay from a separate microbenchmark; no attempt to get hardware counters.
- Layer mapping covers launches that read an ONNX weight; the 19 weight-less launches (state, glue, softmax) are grouped as one bucket.
- ORT profile events with optimisation on were not inspected; ORT timings above are node-level with optimisation off plus whole-run with it on.
- The hand kernel is one block type, AVX2 only, no depthwise conv or matvec kernel, validated on one captured frame.
- Big-model numbers are static estimates from the pickle; the `lds`/`mem` meaning is tinygrad's (`Estimates`), not verified against a counter.
- Wall-clock times in all JSON are inflated by other jobs (load 15-25); use the CPU-time columns.
- Git status shows other experiments' files (`experiments/ane_variants/*`, `aneforge-cache/`) and my files `02_kernel_analysis.py`, `tg_harness.py` as modified (tracked): another process added this directory to git while I worked; I committed nothing.

## Commands

```
PY=/path/to/cvenv/bin/python           # tinygrad 0.14.0, onnx, onnxruntime
cd experiments/kernel_fusion
taskset -c 3 $PY -I 01_node_cost.py                                   # per-node MACs + ORT profile -> data/node_cost.csv
DEV="CPU:CLANG:x86_64,x86-64-v3" taskset -c 3 $PY -I tg_harness.py --per-kernel k.csv --map-weights --reps 15   # per launch
$PY -I 02_kernel_analysis.py k1.csv k2.csv k3.csv                    # min over files
DEV="CPU:CLANG:x86_64,x86-64-v3" taskset -c 3 $PY -I 02b_launch_overhead.py
PY=$PY OUT=sweep.jsonl ./03_sweep.sh ; python 03c_summarize.py sweep.jsonl
$PY -I 04_graph_rewrites.py --outdir DIR ; PY=$PY VARDIR=DIR OUT=variants.jsonl ./03b_variants.sh
DEV="CPU:CLANG:x86_64,x86-64-v3" taskset -c 3 $PY -I 05_fused_block.py
<tgv-python> -I 06_big_distribution.py big.pkl driving_tinygrad_small.pkl   # pinned-commit venv of experiments/big_model
# On a Mac (Metal): same scripts, no taskset
DEV=METAL JIT=2 python -I tg_harness.py --per-kernel k.csv --map-weights --reps 15
DEV=METAL python -I 02b_launch_overhead.py
DEVBASE=METAL CORE= PY=python OUT=sweep_metal.jsonl ./03_sweep.sh
```
Data: `data/` (node costs, per-launch labelled CSV, sweep JSONL and tables, fused-block runs, big/small launch distributions), all small.
