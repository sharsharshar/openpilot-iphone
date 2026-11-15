# ANEForge on the openpilot driving model: results so far

Facts only. "Observed" means I ran it. Nothing here was run on a Mac or on ANE hardware.

## What was used

- Repository: https://github.com/sbryngelson/ANEForge (author Spencer H. Bryngelson; matches the docs: `_targets.preflight`, `cross_compile_check`, `onnx_to_tensor`, `e5rt`, `_lib/build.sh`).
- Commit analysed: `34c18a7eee1b6be129cbaadb088f96a89f395420` (2026-11-13, "ci: replace the retired macos-14 runner ..."). Package version reads `0+unknown` from a raw checkout (no git tag in the clone).
- Licence: MIT (`LICENSE`, Copyright 2026 Spencer H. Bryngelson; `NOTICE` also shipped). The README notes the Apple symbols it uses are private and undocumented.
- Model: `models/driving_supercombo.onnx` (opset 20, fp16 weights, 377 nodes).

## What each ANEForge entry point needs (read from source, then observed on Linux)

| Entry point | Linux | macOS | native dylib | ANE hardware |
|---|---|---|---|---|
| `af.onnx_to_tensor`, `coverage_report` | works (observed) | works | no | no |
| `_targets.preflight(out, Family)` | works (observed) | works | no | no |
| MIL + weights lowering (`_compile._lower_fused_to_dir`, private) | works (observed; 176 KB MIL, 60 MB weights) | works | no | no |
| `_compile.cross_compile_check(out, arch)` | fails (observed): `RuntimeError: libane_e5rt_dispatch.dylib can only be built on Apple Silicon macOS (got linux)` | needs dylib; calls `preflight` first, then `_runtime.compile_check(... TargetArchitecture=<arch>)` | yes (private Espresso/e5rt via dlopen) | not by design ("without creating the device op"); not tested |
| `af.compile(out, target=...)` | same RuntimeError (observed) | `target` only selects the static gate (`_retarget_for`: preflight, raises on reject/oversize). It then compiles for the host and loads a device program, so it is not a cross-compile | yes | yes, for the load step |

`preflight` is a lookup, not a compiler run. Per aneforge graph op it checks (a) a small floor table (`sin`/`cos`/`dropout`/`random`/`global_argmax` need A15+; `crop_resize`/`resample`/`affine`/`gather_hw`/`topk`/`sort`/`dynamic_slice` need A14+; everything else is "native" from A13), and (b) dimension caps (A16: spatial/contraction 65536, channels 65536, transpose 16777215, conv kernel width 15). "Native" in the tables below means only "not in that table and within those caps". It does not mean a compiler accepted it.

The importer (`aneforge/onnx.py`) takes only `graph.output[0]`, builds every graph input as an fp16 tensor of its declared static shape (dtype is ignored), and treats float-to-float `Cast` as identity.

## Result on Linux

Reproduce: `python3 -I run_preflight.py <ANEForge clone> models/driving_supercombo.onnx <out dir> [--ref models/driving_supercombo_fp32.onnx reference/driving_reference_fp32.npz]`.

1. Whole model import, exact error:
   `NotImplementedError: ONNX Slice: step=4 on axis 0 needs the sliced extent (5) to divide by the step; this model needs 1 op the importer does not have: GatherND`
2. Importing each node alone (fresh inputs; 377 nodes): 372 import, 5 fail:
   - nodes 7 and 10, `Slice` with step 4 on axis 0 of a 5-long axis (the image-queue bookkeeping): the same `Slice step=4` error.
   - node 291, `GatherND`: `NotImplementedError: ONNX op 'GatherND' not supported`.
   - node 327, `Where` (mask fill, constant mask and constant -inf): `TypeError: select expects three graph Tensors (cond, a, b)`.
   - node 370, `Concat` with a constant (all-zero [1,2]) operand: `AttributeError: 'numpy.ndarray' object has no attribute 'op'`.
   (The standalone runs of nodes 327 and 370 hit the same handler paths as in the full graph, where their inputs are the same constants. Those two are therefore importer limits on constant operands, not artefacts of the per-node method. I did not separately confirm them inside an unedited full-graph import, because the import stops at the first failing node.)
3. Rank-5 tensors, uint8 inputs, multiple inputs and opset 20: no importer error was attributable to any of them. The importer ignores input dtype; the uint8 bookkeeping is only blocked by the Slice step.

### Per-op verdict (A16 family = M4/M5 and every A17/A18 iPhone chip; contrast A13 = M1, A15 = M3)

Preflight verdict per node, each node preflighted alone. A13, A15 and A16 gave identical verdicts for every op, so the contrast family adds no information here.

| ONNX op (nodes) | Importer | Preflight A16 / A15 / A13 | Becomes (aneforge op) |
|---|---|---|---|
| Gemm 82 | ok | native | matmul |
| Relu 64 | ok | native | relu |
| Conv 60 (29 depthwise/grouped) | ok | native (kernel width and extents within caps) | conv |
| Add 47, Mul 19, Sub 1, Div 3, Sigmoid 4 | ok | native | add, mul, sub, real_div, sigmoid |
| Gelu 20 (`approximate=tanh`) | ok | native | decomposed by the importer into pow(3), tanh and scalar/tensor mul and add (about 8 ops each) |
| MatMul 8 | ok | native | matmul (6), bmm (2) |
| Reshape 9, Transpose 7, Unsqueeze 4, Squeeze 3, Identity 3, Concat 7 of 8 | ok | native | reshape, transpose, expand_dims, squeeze, concat |
| Slice 9 of 11 | ok | native | slice_by_size |
| Slice 2 of 11 (nodes 7, 10) | FAILS (step 4 on a 5-long axis) | not reached | - |
| Gather 3 | ok | native | squeeze + slice_by_size + concat |
| Cast 7 | ok | native | identity (float to float) |
| ReduceMean 2, ReduceMax 1, ReduceL2 2 | ok | native | reduce_mean, reduce_max, sqrt + reduce_sum_square |
| Clip 2 | ok | native | relu (the model's min is 0.0, so this matches) |
| Expand 2 | ok | native | tile |
| LayerNormalization 2 | ok | native | layer_norm |
| Softmax 1 | ok | native | softmax |
| GatherND 1 (node 291) | FAILS: op not supported | not reached | - |
| Where 1 (node 327) | FAILS: constant operands | not reached | - |
| Concat 1 (node 370) | FAILS: constant operand | not reached | - |

No op was reported rejected, decomposed by preflight, or oversize for A13, A15 or A16. Oversize was checked on every tensor, including [1,24,128,256] and the [1,64,64,128] stem output.

## Adaptation and result of the edited graph

Cut: everything up to and including `model/div` (node 25, float16 [1,24,128,256]) is dropped, which removes the uint8 queue bookkeeping (and its two failing Slices). The cut model keeps 347 nodes. Inputs: `model/div` [1,24,128,256], `_to_copy_1` [1,24,512], `_to_copy` [1,25,8], `_to_copy_2` [1,2], all fp16. Only the `outputs` [1,2576] output is kept (the importer reads one output). The next-state outputs are therefore not covered; `next_state_feat_q` needs `state_feat_q`, which only feeds the queue update.

Three edits so the importer accepts it (all in `run_preflight.py`, function `adapt`):
1. `GatherND` with constant indices -9..-1 on axis 0 replaced by `Slice [16:25]`. Exact.
2. `Where(mask, -inf, x)` with a constant mask replaced by `Add(x, bias)`, bias = -60000 where masked, else 0. Same result unless a whole softmax row is masked; the mask here is [1,1,9,9] and a numeric check passed (below).
3. `Concat` with the constant all-zero [1,2] pad: the zeros are made in-graph as `x - x` of a slice of a sibling tensor. Exact unless that tensor holds inf or nan.

Numeric check with onnxruntime (3 steps of the reference loop, state carried, the cut model fed with the full model's own intermediate tensors): max abs difference of the edited sub-graph vs the committed fp16 full model's `outputs`: 0.031, 0.031, 0.016 (reference outputs reach 51.7 in magnitude). Full fp16 model vs the fp32 reference `outputs`: 0.150, 0.154, 0.169; the edited sub-graph vs the fp32 reference: the same values to the digits printed.
This checks the ONNX edits only. It does not check what ANEForge would compute.

Edited graph through ANEForge on Linux (observed):
- Import OK: 540 aneforge graph nodes (60 conv, 88 matmul, 66 relu, 68 add, 39 mul, 20 each of tanh/pow/adds from Gelu, 2 layer_norm, 2 bmm, 1 softmax, others).
- `preflight` for A13, A15, A16: ok, 540 native, 0 decompose, 0 reject, 0 oversize.
- MIL lowering OK: `func main<ios18>`, 60 conv, 90 matmul, 66 relu, 105 mul, 234 add, 20 tanh, 20 pow, 2 rsqrt, 1 softmax; weights.bin 60,036,324 bytes.
- `cross_compile_check` and `af.compile(target="h16s")`: the `RuntimeError` in the table above (no dylib on Linux).
- `cross_compile_check` also emits a warning (on a Linux host assumed family A13) that 5 ops may diverge in fp16 between families: layer_norm, reduce_mean, reduce_sum_square, softmax (<=1 ULP) and slice_by_size (value saturation above 4094 on A13/A14 only).

## macOS workflow

`.github/workflows/aneforge.yml` (macos-15, pinned to the commit above) runs `run_preflight.py`, builds the dylib with `sh aneforge/_lib/build.sh`, then `try_cross_compile.py` for candidates `stem` (first 2 convs and Gelu) and `full` (the 347-node sub-graph). Each stage runs in a child process with its own timeout; the job always succeeds. Not yet run.

## Not yet known

- Whether `_ANEInMemoryModel`-free e5rt `compile_check` works on a hosted macOS VM with `hasANE = NO`. The earlier probe through other classes failed with "Host too old, ANE hardware not available".
- Whether the dylib builds on the runner (needs `xcrun clang++` and its private-framework header shim).
- Whether the compiler accepts any of the 60 conv (29 grouped) at these shapes, regardless of what preflight says. The docs state the cross-compiler does not enforce another family's limits, so a pass would not prove an iPhone chip accepts it.
- Whether results from `h16s` apply to every A17/A18 chip (the source folds `h17s`, `h17d`, `h18` into the same family).
- Numerical agreement of ANEForge's fp16 output with the reference: needs hardware.
- Next-state outputs and the uint8 queue bookkeeping: no ANEForge path found (Slice step 4 is rejected; only one output per import).
- Behaviour of `ANEFORGE_TARGET` auto-detection on the runner (`detect_family` reads the CPU brand string).

## macOS runner result (GitHub `macos-15`, Apple Silicon VM, no Neural Engine)

Run on commit c4865b0, ANEForge 34c18a7.

- ANEForge's native library built without changes.
- Import and static preflight reproduced the Linux results exactly (540 native, 0 rejected, 0 oversize for A13, A15, A16).
- `cross_compile_check(..., "h16s")` returned False in 0.3 s for the full trimmed model AND for a 31-op
  convolution-only stem. A failure on a plain conv stem that fast points at the host, not at the model.
- `af.compile(..., target="h16s")` failed for both with `ane_e5rt_program_compile failed (... mask=0x4)`.
  The underlying Espresso error text was not captured.

So on a GitHub VM neither private route compiles for the Neural Engine: the `_ANEInMemoryModel` route says
"Host too old, ANE hardware not available", and ANEForge's e5rt route fails at program compile. The claim in
ANEForge's docs that cross-compilation needs no device was not reproduced here; it may hold on a physical Mac.
