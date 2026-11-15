# openpilot big driving model ("chestnut", USB+AMD) vs small model: measured facts

Machine: Linux, 4 CPU cores, no GPU, ~15 GB RAM. No model file is stored in the repo.

## Sources and versions
- openpilot master `edde587a77847261fada475a15831b42732b1459` (blobless shallow clone). `tinygrad_repo` submodule pin: **tinygrad `ec91b74b5b6b0fc924b72d3f27689679ca2438b9`** (2026-11-02, "fix mark_gated for gate past symbolic end (#19154)"), checked out and installed editable (reports version 0.14.0) in a separate venv.
- Download note: the `media.githubusercontent.com` URL in the task returns 404 for every LFS file of this repo and the GitHub LFS endpoint is refused by the git proxy. `.lfsconfig` points LFS to `https://huggingface.co/commaai/openpilot-lfs.git/info/lfs`. The LFS batch API there returned a public CDN link and the object downloaded. Big pickle sha256 verified = `b4dead3e...e960`, 767,228,907 bytes. `big_driving_warp_1344x760_tinygrad.pkl` (462,613 B, sha256 `1c80960c...3e97`) also fetched.
- Small pickle: `driving_tinygrad.pkl` from the `nightly` branch via raw.githubusercontent.com (not LFS), 67,855,219 B, sha256 `bb13a3e4829774c0c7cd48d7ab79a75784fc34f9912b4e34d915eddd5c627ed4`.
- Small ONNX: /home/user/openpilot-iphone/models/driving_supercombo.onnx (61 MB).

## Container format (what "out of band" means)
`dump_pickle` (tinygrad `examples/openpilot/helpers.py`): `<int64 length N><N bytes of pickle opcodes><raw buffer bytes>`. Each buffer is written at a 256-byte-aligned offset; the opcode stream refers to it by persistent id `(nbytes, offset)`. So the weights are raw bytes after the opcode stream: big = 747,231 B opcodes + 766,481,668 B raw; small = 765,419 B opcodes + 67,089,792 B raw. `load_pickle(out_of_band=True)` copies the whole raw region to the default device in one transfer and hands out views.

## Safe inspection (never executed pickle code)
`inspect_pkl.py` (pickletools.genops only). Globals referenced, identical in big and small: only 16 tinygrad names (BufferSpec, AddrSpace, DType, CapturedJit, _TinyJit, Target, Estimates, runtime.support.hcq2.HCQInfo, uop.Ops, AxisType, CallInfo, CustomFunction, KernelInfo, ParamArg, ProgramInfo, UOp). No numpy, no builtins, nothing outside tinygrad. These 16 are the allowlist in `load_safe.py`; `analyze_pkl.py` and `run_big.py` load through a `pickle.Unpickler` subclass that rejects anything else. Weight bytes were replaced by (nbytes, offset) placeholders in `analyze_pkl.py`. The warp pickle is a plain pickle that additionally references builtins.bytearray, tinygrad.device.Buffer and tinygrad.tensor.Tensor; it was inspected statically only, not loaded.

Device strings baked in: big = `AMD` (buffers, input/output specs) plus `CPU`/`LLVM`/`x86_64,native` (LLVM host-side helper programs named `usb_poke`, `usb_stream`, `usb_patch`, `amd_push`, `fence_amd`, `do_get_time_ms`, `hcq_submit`, with buffers `amd_usb_host`, `libusb_control_transfer`, `libusb_bulk_transfer`). Small = `QCOM` plus `CPU`/`LLVM` (a `struct_kgsl_gpu_command` buffer). The warp pickle: AMD + CPU/LLVM.

## Comparison (big vs small)
| | big (chestnut) | small (comma device) |
|---|---|---|
| pickle file bytes | 767,228,907 | 67,855,219 |
| compiled for | AMD via USB (HCQ command stream) | QCOM (HCQ) |
| f16 weight elements in pickle buffers | 382,904,772 | 33,538,670 |
| f16 weight bytes | 765,809,544 | 67,077,340 |
| other buffers | 8 bool (663,552 B), 1 i64 (9 el.) | 1 bool (81), 1 i64 (9) |
| buffers (out-of-band) | 593 | 314 |
| parameter count, ONNX initializers | no ONNX exists, not measured | 30,003,201 f16 (+81 bool, 58 int64) |
| image input | `new_img` (2,6,128,256) u8 | same (2,6,128,256) u8 |
| `state_img_q` | (2,5,6,128,256) u8 | same |
| `state_desire_q` | (132,1,8) f32 | (100,1,8) f32 |
| `state_feat_q` | (128,1,16384) f32 = 8.39 MB | (96,1,512) f32 = 0.20 MB |
| other inputs | desire (8), traffic_convention (1,2), action_t (1,2): same | same |
| `outputs` | (1,18452) f32 | (1,2576) f32 |
| `hidden_state` slice of outputs | 16384 values | 512 |
| kernel launches per step (pickle) | 474 (56 distinct names) | 166 (66 distinct) |
| ops per step, tinygrad `Estimates.ops` summed over launches | 96,233,854,572 (96.2 G) | 1,719,778,910 (1.72 G) |
| ops per step, ONNX route, `GlobalCounters.global_ops` | not available (no ONNX) | 1,835,779,500 (1.84 G) |
| `Estimates.lds` (bytes loaded+stored) per step | 7.06 GB | 1.83 GB |
| `Estimates.mem` per step | 1.216 GB | 95.7 MB (pickle); `GlobalCounters.global_mem` 89.3 MB (ONNX route) |
| ops in matmul/conv-type (`r_*`) launches | 99.94 % (382 launches) | 99.89 % (151 launches) |
| peak RSS | not run. Only copying the weights to a CPU buffer: 1,514 MB | ONNX route on CPU: 254 MB |
| CPU ms per step | not measured (cannot run) | ONNX route, tinygrad CPU:CLANG x86-64-v3: median 366.0 ms over 10 warm steps (2 warm-ups discarded; 12 steps total); output finite |

Ratios (big/small): file 11.3x, f16 weight elements 11.4x, state_feat_q 42.7x, outputs 7.2x, kernels 2.9x, ops 56x (Estimates, same counter both sides).

Cross-check of the two measurement paths for the small model: pickle `Estimates.ops` 1.720 G vs ONNX-route `GlobalCounters.global_ops` 1.836 G (pickle is 6.3 % lower); kernels 166 vs 159 per step. The two are built by different tinygrad optimisation paths (the pickle is the QCOM-optimised build and may be from a different tinygrad commit than the pinned one), so they agree to ~6 %, not exactly. The pickle holds 33.54 M f16 elements against 30.00 M ONNX initializer elements (+3.54 M, +11.8 %); the cause (e.g. weight re-layout/padding in the compiled graph) was not investigated. Hence the big pickle's element count is an upper bound on its true parameter count, by an unknown margin.

## Load / retarget / run: how far it got
1. Safe structural load: works with the pinned tinygrad (tinygrad commit above). The 0.14.0 pip release was not tried; the pinned commit was used throughout.
2. Real load onto a CPU backend (`DEV=CPU:CLANG:x86_64,x86-64-v3`, `run_big.py`): the 766 MB raw region copied to a CPU buffer (peak RSS 1,514 MB) and the allowlisted unpickle succeeded. `lower_and_compile` returned without doing anything (tinygrad's `hcq2` returns the linear unchanged when it already contains an `HCQInfo` call, "compiled already"). `input_specs` and `output_specs` still say device `AMD`.
3. Calling it fails: `JitError: args mismatch in JIT: expected_input_info != ...` (tinygrad/engine/jit.py:172), because the captured JIT expects inputs on device `AMD`, ours are on `CPU` (full traceback in `run_big.log`). Creating a Tensor on `AMD` here fails with "No interface for AMD:0 is available".
4. Why the device cannot simply be renamed: the captured `_linear` contains no kernel source or ASTs, no `PROGRAM`/`SINK` of model kernels, and none of `make_retargetable`'s CPU targets. It is one `CALL` with `HCQInfo` aux, ~6,100 uops (3,214 CONST, 1,176 GETADDR, 593 BUFFER, 87 BINARY blobs totalling 500,380 B, STOREs into a pre-recorded AMD command buffer `cmdbuf_compute_0` of 5,530 B kernargs / 5,531 slot), plus LLVM helper programs that emulate the USB host and submit the commands. The 474 model kernels exist only as names and `Estimates` in `HCQInfo.kernels`, and as AMD machine code/command packets in the binary blobs. This is the form that `load_oob` loads for `USB+AMD:LLVM`; `lower_and_compile` in modeld.py is a no-op for it. Running it needs the AMD GPU on USB. The `--retargetable` option of `compile_onnx.py` is how a CPU-retargetable pickle is made; this pickle was not built with it. No ONNX of the big model is in the repo. Conclusion: it cannot run on this CPU from this artifact; not run, so no timing, no peak memory, no finite-output check.

Because of that, the big model's arithmetic is taken from tinygrad's own per-launch `Estimates` stored in the pickle (the same counter the compiler uses; summed over the 474 launches, equal to the stored total). It is not a runtime measurement. Per-launch detail is in `analysis_big.txt`: the largest launches are ~1.1 Gop each; top 10 launches = 11.6 % of ops, top 50 = 57.4 %. Small: top 10 = 41 %, top 50 = 96.6 %.

Output slice layout (offsets into `outputs`): big: lane_lines 0:528, lane_lines_prob 528:536, road_edges 536:800, meta 800:855, desire_pred 855:887, pose 887:899, wide_from_device_euler 899:905, road_transform 905:917, plan 917:1907 (990), lead 1907:2051, lead_prob 2051:2054, desire_state 2054:2062, action 2062:2066, hidden_state 2066:18450, pad 18450:18452. Small (from `analysis_small.txt`): meta 0:55, desire_pred 55:87, pose, wide_from_device_euler, road_transform, lane_lines 117:645, lane_lines_prob, road_edges, lead, lead_prob, hidden_state 1064:1576 (512), plan 1576:2566, desire_state 2566:2574, pad = last 2. The small model has no `action` field in its outputs; the big model has `action` (4 values) at 2062:2066.

## Commands
```
# env: uv venv tgv; uv pip install numpy -e ./tg   (tg = tinygrad at ec91b74b5b6b0fc924b72d3f27689679ca2438b9)
curl -sS -X POST -H "Accept: application/vnd.git-lfs+json" -H "Content-Type: application/vnd.git-lfs+json" \
  -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"b4dead3e...e960","size":767228907}]}' \
  https://huggingface.co/commaai/openpilot-lfs.git/info/lfs/objects/batch      # then GET the returned href
python -I inspect_pkl.py big.pkl            # also driving_tinygrad_small.pkl, big_warp_1344.pkl
python -I analyze_pkl.py big.pkl            # allowlisted structural load; same for the small pickle
DEV="CPU:CLANG:x86_64,x86-64-v3" python -I run_big.py big.pkl
DEV="CPU:CLANG:x86_64,x86-64-v3" python -I onnx_small_counters.py driving_supercombo.onnx 12
```
Outputs: `inspect_big.txt`, `inspect_small.txt`, `analysis_big.txt`, `analysis_small.txt`, `run_big.log`.

## Not measured
- Any CPU run of the big model: wall time, peak resident memory of a full run, output finiteness.
- Big model true parameter count (no ONNX; pickle element count is 11.8 % above ONNX count for the small model for an unknown reason).
- Big model ops from runtime counters (`GlobalCounters`); only the compiler's static estimates were read.
- The warp pickles (`big_driving_warp_*`) were not loaded; their ops are not included in either model's figures. The 1928x1208 warp was not fetched.
- AMD-side timing, and any other measurement involving the chestnut hardware.
- Cause of the 6 % ops gap and the +3.54 M element gap between the small pickle and the small ONNX route.
- Small model with fp32 weights, other backends, other tinygrad versions (0.14.0 pip release not tried on the big pickle).
- Precision of `Estimates.ops` (how tinygrad counts a multiply-add) was not independently verified against an analytic MAC count.
