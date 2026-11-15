"""Run the openpilot driving model (ONNX) with tinygrad's ONNX frontend + TinyJit and compare to the fp32 reference.

Device selection is tinygrad's normal environment (written for tinygrad 0.14.0):
  DEV=CPU:CLANG | DEV=CPU:LLVM | DEV=METAL        (JIT=2 disables graph batching, as openpilot does on a Mac)

Usage:  python run_tinygrad.py [--steps 40] [--dump-kernels DIR] [--model PATH] [--ref PATH]

State (state_img_q, state_desire_q, state_feat_q) is kept on the device between steps: the jitted function's
next_state_* output Tensors are cloned on the device and passed back as the next step's state_* inputs.
Only the per-step inputs (frame, desire, ...) are uploaded from numpy and `outputs` is read back.
"""
import argparse, os, re, sys, time, statistics, collections, pathlib
import numpy as np
from tinygrad import Tensor, Device, TinyJit, GlobalCounters, dtypes
from tinygrad.helpers import Context, DEBUG
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.uop.ops import Ops, UOp
from tinygrad.engine.realize import ExecContext, pm_exec, get_call_name

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STATE = ["state_img_q", "state_desire_q", "state_feat_q"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/driving_supercombo.onnx"))
ap.add_argument("--ref", default=str(ROOT / "reference/driving_reference_fp32.npz"))
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--dump-kernels", metavar="DIR", default=None)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()

ref = np.load(args.ref)
N = min(args.steps, len(ref["outputs"]))
dev = Device.DEFAULT
print(f"device={dev} DEV={os.environ.get('DEV','')} JIT={os.environ.get('JIT','1')} steps={N}", flush=True)

t0 = time.perf_counter()
runner = OnnxRunner(args.model)
print(f"loaded onnx in {time.perf_counter()-t0:.1f}s; inputs: " + ", ".join(f"{k}{tuple(v.shape)}:{v.dtype}" for k, v in runner.graph_inputs.items()))

@TinyJit
def step(**inputs):
  out = runner(inputs)
  return tuple(out[k].contiguous() for k in ["outputs", *[f"next_{s}" for s in STATE]])

state = {n: Tensor.zeros(*runner.graph_inputs[n].shape, dtype=runner.graph_inputs[n].dtype).contiguous().realize() for n in STATE}
diffs_max, diffs_mean, times, ok = [], [], [], True
last_inputs = None
for t in range(N):
  feed = {
    "new_img": Tensor(ref["frames"][t]).realize(),
    "desire": Tensor(ref["desire"][t]).realize(),
    "traffic_convention": Tensor(ref["traffic_convention"][t]).realize(),
    "action_t": Tensor(ref["action_t"][t]).realize(),
    **state,
  }
  Device[dev].synchronize()
  st = time.perf_counter()
  res = step(**feed)
  # carry state on the device: clone so the next call's inputs are distinct buffers from the JIT's output buffers
  state = {n: res[1 + i].clone().realize() for i, n in enumerate(STATE)}
  Device[dev].synchronize()
  times.append(time.perf_counter() - st)
  out = res[0].numpy().reshape(-1).astype(np.float32)   # numpy() blocks until the device has finished
  d = np.abs(out - ref["outputs"][t])
  if not np.isfinite(out).all(): ok = False
  diffs_max.append(float(d.max())); diffs_mean.append(float(d.mean()))
  print(f"step {t:2d}  max_abs={diffs_max[-1]:.4f} mean_abs={diffs_mean[-1]:.5f}  {times[-1]*1e3:9.1f} ms{'  (compile/capture)' if t < 2 else ''}", flush=True)
  last_inputs = feed

worst_max, worst_mean = max(diffs_max), max(diffs_mean)
warm = times[3:] if len(times) > 3 else times
median_ms = statistics.median(warm) * 1e3
print(f"worst max abs diff {worst_max:.4f}, worst mean abs diff {worst_mean:.5f}, overall mean {np.mean(diffs_mean):.5f}")
if not (np.isfinite(worst_max) and np.isfinite(worst_mean)): ok = False

# ---------------- kernel inventory and per-kernel timing ----------------
calls = list(step.captured.linear.src)
def kind(c):
  a = c.src[0]
  if a.op is Ops.PROGRAM: return "program"
  if a.op is Ops.COPY: return "copy"
  if a.op is Ops.CUSTOM_FUNCTION: return str(a.arg)
  return str(a.op)
progs = [c for c in calls if kind(c) == "program"]
print(f"jit step has {len(calls)} calls: " + ", ".join(f"{k}={v}" for k, v in collections.Counter(kind(c) for c in calls).items()))
kernels_per_step = len(calls)
# a "graph" call batches several kernels (Metal graph, JIT<2); count the kernels inside it
n_inside = sum(len(c.src[0].src[0].src) for c in calls if kind(c) == "graph")
if n_inside: kernels_per_step = len(calls) - sum(kind(c) == "graph" for c in calls) + n_inside

# per-kernel timing: run each call of the captured linear once with wait=True (device-side time where the runtime reports it)
from tinygrad.engine.jit import _prepare_jit_inputs
input_uops, var_vals, _, _ = _prepare_jit_inputs((), last_inputs)
concrete = tuple(input_uops)
prof = []  # (name, seconds, call)
reps = 5
tot = collections.defaultdict(float)
for c in calls:
  ctx = ExecContext(var_vals, concrete, update_stats=False, jit=True, wait=True)
  tm = []
  for _ in range(reps):
    ets = pm_exec.rewrite(c, ctx)
    tm.append(sum(e for e in (ets or []) if e is not None))
  name = get_call_name(c, [], var_vals) if kind(c) != "program" else c.src[0].arg.name
  name = re.sub(r"\x1b\[[0-9;]*m", "", name)
  prof.append((name, min(tm), c))
total = sum(p[1] for p in prof)
print(f"\nsum of per-kernel times (best of {reps}, kernels run alone): {total*1e3:.1f} ms   (whole jitted step median {median_ms:.1f} ms)")
print(f"\nTop {args.top} kernels by time")
print(f"{'#':>3} {'kernel':<64} {'ms':>9} {'share':>7}")
for i, (n, s, _) in enumerate(sorted(prof, key=lambda p: -p[1])[:args.top]):
  print(f"{i+1:>3} {n[:64]:<64} {s*1e3:9.3f} {100*s/total:6.1f}%")
def cat(n, c):
  if kind(c) != "program": return kind(c)
  if n.startswith("r_"): return "reduce (conv/matmul/norm/softmax) r_*"
  if n.startswith("E_"): return "elementwise E_*"
  return "other"
cats = collections.defaultdict(lambda: [0, 0.0])
for n, s, c in prof: k = cat(n, c); cats[k][0] += 1; cats[k][1] += s
print("\nTime by kernel category")
for k, (n, s) in sorted(cats.items(), key=lambda kv: -kv[1][1]): print(f"  {k:<42} kernels={n:4d}  {s*1e3:9.2f} ms  {100*s/total:5.1f}%")

# ---------------- dump generated sources ----------------
if args.dump_kernels:
  d = pathlib.Path(args.dump_kernels); d.mkdir(parents=True, exist_ok=True)
  seen = {}
  for c in progs:
    p = c.src[0]
    if p.key in seen: continue
    srcs = [s for s in p.src if s.op is Ops.SOURCE]
    if not srcs: continue
    seen[p.key] = (re.sub(r"\x1b\[[0-9;]*m", "", p.arg.name), srcs[0].arg, p.arg.global_size, p.arg.local_size)
  ext = {"METAL": "metal", "CPU": "c"}.get(dev.split(":")[0], "txt")
  size = 0
  for i, (name, src, gs, ls) in enumerate(seen.values()):
    body = f"// global_size={gs} local_size={ls}\n{src}"
    (d / f"{i:04d}_{name[:100]}.{ext}").write_text(body); size += len(src.encode())
  print(f"\nwrote {len(seen)} distinct kernels ({len(progs)} program calls) to {d}; total source size {size} bytes ({size/1024:.1f} KiB)")

print(f"\nTINYGRAD RESULT device={dev} worst_max={worst_max:.4f} worst_mean={worst_mean:.5f} median_ms={median_ms:.1f} kernels_per_step={kernels_per_step}")
sys.exit(0 if ok else 1)
