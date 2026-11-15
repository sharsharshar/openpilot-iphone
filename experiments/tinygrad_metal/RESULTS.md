# tinygrad on the openpilot driving model: measurements

Host: Linux x86_64, 4 cores, Intel Xeon (Sapphire Rapids class) @ 2.1 GHz, no GPU, clang 18.1.3, Python 3.12.
tinygrad **0.14.0** (pip). Model `models/driving_supercombo.onnx` (fp16 weights) unless stated. Reference: `reference/driving_reference_fp32.npz`, 40 steps.
Pinned in CI as `tinygrad==0.14.0`.

## What the installed version supports (read from site-packages)
- ONNX frontend: `tinygrad.nn.onnx.OnnxRunner(path)(dict_of_tensors)`. No onnx package is needed to load.
- Device selection is `DEV=<DEVICE>[:<RENDERER>[:<ARCH>]]` (`CPU:CLANG`, `CPU:LLVM`, `METAL`). `CLANG=1` / `METAL=1` style variables are rejected with an assertion.
- `JIT=2` is supported (`engine/jit.py`: graph batching is skipped when `JIT >= 2`). `HCQ_RUNTIME_DEV` exists only in the experimental `HCQ2` path (AMD/CPU), not used here.
- `FLOAT16=1` only has an effect together with `IMAGE=1` (openpilot's QCOM path); on CPU it changed nothing (identical output). fp16 weights run as native `__fp16` in the generated C.
- `DEV=CPU:CLANG` (default `-march=native`) crashes clang 18 in the backend on this machine (compile of `r_4_512_4_4_4_16_4`). Workaround used: `DEV="CPU:CLANG:x86_64,x86-64-v3"`.

## Commands
```
source <venv>/bin/activate     # tinygrad 0.14.0, numpy
python -I experiments/tinygrad_metal/run_tinygrad.py --dump-kernels DIR            # with DEV set as below
DEV="CPU:CLANG:x86_64,x86-64-v3" ...                      # run A (fp16 model)
DEV="CPU:CLANG:x86_64,x86-64-v3" JIT=2 ...                # run B
DEV="CPU:CLANG:x86_64,x86-64-v3" FLOAT16=1 ...            # run C
DEV=CPU:LLVM ...                                          # run D
DEV="CPU:CLANG:x86_64,x86-64-v3" ... --model models/driving_supercombo_fp32.onnx      # run E
DEV="CPU:CLANG:x86_64,x86-64-v3" ... --ref reference/driving_reference.npz            # run F
```
State is kept on the device between steps (the jitted function returns `next_state_*`; they are `.clone().realize()`d and fed back). Only frames/desire/etc. are uploaded and `outputs` is read back.

## Numerics versus reference (40 steps, abs difference over the 2576 outputs)
| run | model | reference | worst max | worst mean | median ms/step |
|---|---|---|---|---|---|
| A clang | fp16 | fp32 ref | 1.3877 | 0.02204 | 357.1 |
| B clang, JIT=2 | fp16 | fp32 ref | 1.3877 | 0.02204 | 348.9 |
| C clang, FLOAT16=1 | fp16 | fp32 ref | 1.3877 | 0.02204 | 354.3 |
| D LLVM | fp16 | fp32 ref | 1.3121 | 0.02527 | 437.6 |
| E clang | fp32 | fp32 ref | 0.0005 | 0.00001 | 180.7 |
| F clang | fp16 | fp16 ref | 1.7521 | 0.03218 | 350.6 |

Overall mean abs diff for run A: 0.00866. The known fp16-vs-fp32 noise floor is max 1.71 / mean 0.010 (mean over steps); runs A-D are inside it on max. Run E (fp32 weights) matches onnxruntime to 5e-4, which confirms the step loop, uint8 inputs, state carry on device and the JIT buffer handling are correct; no JIT-reuse problem was observed on CPU (also not with `JIT=2`, which on CPU is identical to `JIT=1` because the CPU backend has no graph batching). Run F: tinygrad fp16 is not closer to onnxruntime fp16 than to the fp32 reference.

median ms is the median of steps 3..39 including the three small on-device state clones and device synchronisation.

## Kernels
- Kernels per jitted step: **161** (161 program calls, 0 copies). 152 are reduce kernels (`r_*`: conv, gemm, matmul, layernorm, softmax), 9 elementwise (`E_*`).
- Distinct kernels (by program key, run A): **78**, total generated C source **646,338 bytes (631.2 KiB)**.
- Some kernels take an integer variable argument (e.g. `E_6_2048_4_2_2_4` has `const int data5_`); it is a bound variable of the JIT.
- Generated C uses `__fp16` for the fp16 model.

### Top 15 kernels, run A (best of 5 runs of each kernel alone; sum 328.2 ms, whole step median 357.1 ms)
| # | kernel | ms | share |
|---|---|---|---|
| 1 | r_4_64_32_4_4_4_24_3_3 | 50.388 | 15.4% |
| 2 | r_16_512_4_4_3_16_4 | 14.244 | 4.3% |
| 3 | r_16_512_4_4_3_16_4 | 13.991 | 4.3% |
| 4 | r_32_128_4_4_3_32_4 | 10.531 | 3.2% |
| 5 | r_32_128_4_4_3_32_4 | 10.356 | 3.2% |
| 6 | r_128_8_4_4_3_128_4 | 8.864 | 2.7% |
| 7-12 | r_64_32_4_4_3_64_4 (x6) | 8.25-8.36 each | 2.5% each |
| 13 | r_128_8_4_4_3_128_4 | 8.187 | 2.5% |
| 14 | r_32_8_4_4_4_384_4 | 7.618 | 2.3% |
| 15 | r_32_8_4_4_4_384_4 | 7.355 | 2.2% |

Time by category (run A): reduce `r_*` 152 kernels 327.75 ms (99.9%); elementwise `E_*` 9 kernels 0.40 ms (0.1%). Kernel names do not distinguish conv from gemm, so no finer split is given. Run E (fp32): 144.7 ms summed, 99.7% reduce. Run D (LLVM): 413.0 ms summed, 99.5% reduce.
Per-kernel times come from running each captured call alone with `wait=True`; they sum to less than the jitted step, which also includes input upload handling and state clones.

## Not yet measured
Everything on Metal: numerics versus the reference, kernels per step with the Metal renderer, per-kernel time and share, distinct kernel count and MSL source size, compile time, and whether `JIT=2` matters there. The workflow `.github/workflows/tinygrad-metal.yml` is written but has not been run. Timings from GitHub's macOS runners (virtual machine, paravirtualised GPU) are not representative of real hardware. Also not measured: `IMAGE=1`/`NOLOCALS=1` (openpilot's QCOM flags), BEAM search, and the Core ML comparison on identical inputs.
