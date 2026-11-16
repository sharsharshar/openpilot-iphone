"""Shared tinygrad harness for the kernel-fusion experiments (written for tinygrad 0.14.0).

Runs a driving-model ONNX (original or lean) through OnnxRunner + TinyJit for N steps with the state carried on the device,
checks every step against the reference npz, and reports: kernels per step, distinct kernels, estimated ops (tinygrad's own
Estimates, summed over launches), whole-step time (median / min of the warm steps), and, with --per-kernel, per-launch timings.

Device and all options come from tinygrad's environment variables (DEV=..., PCONTIG=..., SPLIT_REDUCEOP=..., NOOPT=..., BEAM=...).
Pin to one core from outside (taskset -c 3 ...). tinygrad reads the CPU count from sched_getaffinity, so a pinned process
has NUM_CPU_THREADS=1 and generates single-thread kernels (tinygrad/helpers.py:263, renderer/cstyle.py:261, codegen/opt/heuristic.py:183).

Usage (stdout ends with one line starting 'RESULT ' holding a JSON dict):
  DEV="CPU:CLANG:x86_64,x86-64-v3" taskset -c 3 python -I tg_harness.py --model models/driving_supercombo_fp32.onnx --label base
Mac / Metal later:  DEV=METAL JIT=2 python tg_harness.py --model ... --label metal   (no taskset; --per-kernel works the same way)
"""
import argparse, collections, csv, json, os, pathlib, re, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import ExecContext, pm_exec, estimate_uop
from tinygrad.helpers import NUM_CPU_THREADS, getenv
from tinygrad.uop.ops import sym_infer

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/driving_supercombo_fp32.onnx"))
ap.add_argument("--ref", default=str(ROOT / "reference/driving_reference_fp32.npz"))
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--warm", type=int, default=3, help="steps excluded from the timing statistics (JIT capture + first replay)")
ap.add_argument("--label", default="run")
ap.add_argument("--per-kernel", metavar="CSV", default=None, help="time every launch alone and write one CSV row per launch")
ap.add_argument("--map-weights", action="store_true", help="with --per-kernel: name each launch's weight inputs by matching buffer contents to ONNX initializers")
ap.add_argument("--reps", type=int, default=9, help="repetitions per launch for --per-kernel")
ap.add_argument("--dump-kernels", metavar="DIR", default=None, help="write the generated source of each distinct kernel")
ap.add_argument("--json", metavar="FILE", default=None, help="append the RESULT dict as a JSON line to FILE")
args = ap.parse_args()

STATE_ORIG = ["state_img_q", "state_desire_q", "state_feat_q"]
ref = np.load(args.ref)
N = min(args.steps, len(ref["outputs"]))
dev = Device.DEFAULT
t0 = time.perf_counter()
runner = OnnxRunner(args.model)
t_load = time.perf_counter() - t0
gi = runner.graph_inputs
lean = "img_stack" in gi
STATE = [n for n in ["state_img_q", "state_desire_q", "state_feat_q"] if n in gi]
out_names = ["outputs", *[f"next_{s}" for s in STATE]]
print(f"[{args.label}] device={dev} DEV={os.environ.get('DEV','')} threads={NUM_CPU_THREADS.value} model={os.path.basename(args.model)} lean={lean} steps={N}", flush=True)

@TinyJit
def step(**inputs):
  out = runner(inputs)
  return tuple(out[k].contiguous() for k in out_names)

class NumpyHistory:  # same as scripts/verify_lean.py: last 5 model images per camera
  def __init__(self): self.q = np.zeros((5, 2, 6, 128, 256), np.uint8)
  def push(self, f): self.q = np.concatenate([self.q[1:], np.asarray(f, np.uint8)[None]], axis=0)
  def stack(self): return np.concatenate([self.q[k, c] for c in (0, 1) for k in (0, 4)], axis=0)[None].astype(np.float32)

hist = NumpyHistory()
state = {n: Tensor.zeros(*gi[n].shape, dtype=gi[n].dtype).contiguous().realize() for n in STATE}
dmax, dmean, times, clone_times, cpu_times = [], [], [], [], []
load0 = os.getloadavg()[0]
last = None
for t in range(N):
  hist.push(ref["frames"][t])
  img = {"img_stack": Tensor(hist.stack()).realize()} if lean else {"new_img": Tensor(ref["frames"][t]).realize()}
  feed = {**img, "desire": Tensor(ref["desire"][t]).realize(), "traffic_convention": Tensor(ref["traffic_convention"][t]).realize(),
          "action_t": Tensor(ref["action_t"][t]).realize(), **state}
  Device[dev].synchronize()
  st = time.perf_counter(); cp = time.process_time()
  res = step(**feed)
  Device[dev].synchronize()
  times.append(time.perf_counter() - st); cpu_times.append(time.process_time() - cp)       # the jitted step alone (all kernel launches) + device sync
  st = time.perf_counter()
  state = {n: res[1 + i].clone().realize() for i, n in enumerate(STATE)}
  Device[dev].synchronize()
  clone_times.append(time.perf_counter() - st)  # state carry (clone of next_state_* into fresh buffers), reported separately
  out = res[0].numpy().reshape(-1).astype(np.float32)
  d = np.abs(out - ref["outputs"][t]); dmax.append(float(d.max())); dmean.append(float(d.mean()))
  if not np.isfinite(out).all(): dmax[-1] = float("inf")
  last = feed
  if t < 2: print(f"  step {t}: {times[-1]:.2f} s (capture/compile)", flush=True)
warm = times[args.warm:]; cwarm = clone_times[args.warm:]

calls = list(step.captured.linear.src)
def kind(c):
  a = c.src[0]
  if a.op is Ops.PROGRAM: return "program"
  if a.op is Ops.COPY: return "copy"
  if a.op is Ops.CUSTOM_FUNCTION: return str(a.arg)
  return str(a.op)
progs = [c for c in calls if kind(c) == "program"]
ests = [estimate_uop(c) for c in progs]
names = [ANSI.sub("", c.src[0].arg.name) for c in progs]
ops = [int(sym_infer(e.ops, {})) for e in ests]
lds = [int(sym_infer(e.lds, {})) for e in ests]
mem = [int(sym_infer(e.mem, {})) for e in ests]
distinct = {c.src[0].key for c in progs}
src_bytes = 0; seen = set()
for c in progs:
  p = c.src[0]
  if p.key in seen: continue
  seen.add(p.key)
  s = [x for x in p.src if x.op is Ops.SOURCE]
  if s: src_bytes += len(s[0].arg.encode())
res_d = dict(label=args.label, model=os.path.basename(args.model), dev=os.environ.get("DEV", ""), threads=NUM_CPU_THREADS.value,
             env={k: os.environ[k] for k in os.environ if k in ("PCONTIG", "SPLIT_REDUCEOP", "NOOPT", "BEAM", "JITBEAM", "WINO", "MAX_KERNEL_BUFFERS", "NOLOCALS", "THREADS", "REDUCEOP_SPLIT_THRESHOLD", "JIT", "TRANSCENDENTAL", "DEBUG")},
             calls=len(calls), kernels=len(progs), distinct=len(distinct), ops_G=sum(ops) / 1e9, lds_MB=sum(lds) / 1e6, mem_MB=sum(mem) / 1e6,
             src_KiB=src_bytes / 1024, worst_max=max(dmax), worst_mean=max(dmean), mean_mean=float(np.mean(dmean)),
             step_ms_median=statistics.median(warm) * 1e3, step_ms_min=min(warm) * 1e3, n_timed=len(warm), step_cpu_ms_median=statistics.median(cpu_times[args.warm:]) * 1e3, step_cpu_ms_min=min(cpu_times[args.warm:]) * 1e3, load_start=load0, load_end=os.getloadavg()[0], clone_ms_median=statistics.median(cwarm) * 1e3, load_s=t_load)

if args.per_kernel:
  from tinygrad.engine.jit import _prepare_jit_inputs
  input_uops, var_vals, _, _ = _prepare_jit_inputs((), last)
  concrete = tuple(input_uops)
  wmap = {}
  if args.map_weights:
    import hashlib, onnx
    from onnx import numpy_helper
    from tinygrad.engine.realize import resolve_params
    def h(a): return hashlib.sha1(np.sort(np.ascontiguousarray(a, np.float32).reshape(-1)).tobytes()).hexdigest()
    inits = {}
    for t in onnx.load(args.model).graph.initializer:
      a = numpy_helper.to_array(t)
      if a.dtype.kind == "f" and a.size > 1: inits.setdefault((a.size, h(a)), []).append(t.name)
    sizes = {k[0] for k in inits}
    for i, c in enumerate(progs):
      found = []
      for b in resolve_params(c, concrete):
        try: buf = b.buffer
        except Exception: continue
        if buf.size in sizes and buf.dtype.name == "float":
          key = (buf.size, h(buf.numpy()))
          if key in inits: found += inits[key]
      wmap[i] = ";".join(dict.fromkeys(found))
  rows = []
  ctx = ExecContext(var_vals, concrete, update_stats=False, jit=True, wait=True)
  # (a) each launch alone, repeated back to back (data stays in cache)
  alone = []
  for i, c in enumerate(progs):
    tm = []
    for _ in range(args.reps):
      ets = pm_exec.rewrite(c, ctx); tm.append(sum(e for e in (ets or []) if e is not None))
    alone.append(tm)
  # (b) in sequence: the whole step's launches in order, repeated; per launch the min/median over repetitions (realistic cache state)
  seq = [[] for _ in progs]
  for _ in range(args.reps):
    for i, c in enumerate(progs):
      ets = pm_exec.rewrite(c, ctx); seq[i].append(sum(e for e in (ets or []) if e is not None))
  for i, c in enumerate(progs):
    p = c.src[0]
    rows.append(dict(i=i, name=names[i], weights=wmap.get(i, ""), ops=ops[i], lds_bytes=lds[i], mem_bytes=mem[i], global_size=str(p.arg.global_size),
                     t_min_us=min(alone[i]) * 1e6, t_median_us=statistics.median(alone[i]) * 1e6,
                     seq_min_us=min(seq[i]) * 1e6, seq_median_us=statistics.median(seq[i]) * 1e6))
  with open(args.per_kernel, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  res_d["sum_kernel_min_ms"] = sum(r["t_min_us"] for r in rows) / 1e3
  res_d["sum_kernel_median_ms"] = sum(r["t_median_us"] for r in rows) / 1e3
  res_d["sum_seq_min_ms"] = sum(r["seq_min_us"] for r in rows) / 1e3
  res_d["sum_seq_median_ms"] = sum(r["seq_median_us"] for r in rows) / 1e3

if args.dump_kernels:
  d = pathlib.Path(args.dump_kernels); d.mkdir(parents=True, exist_ok=True); seen = {}
  for c in progs:
    p = c.src[0]
    if p.key in seen: continue
    s = [x for x in p.src if x.op is Ops.SOURCE]
    if s: seen[p.key] = (ANSI.sub("", p.arg.name), s[0].arg, p.arg.global_size)
  ext = {"METAL": "metal", "CPU": "c"}.get(dev.split(":")[0], "txt")
  for i, (nm, src, gs) in enumerate(seen.values()): (d / f"{i:04d}_{nm[:100]}.{ext}").write_text(f"// global_size={gs}\n{src}")

print(f"kernels={res_d['kernels']} distinct={res_d['distinct']} ops={res_d['ops_G']:.3f}G worst_max={res_d['worst_max']:.4g} worst_mean={res_d['worst_mean']:.4g} "
      f"step median {res_d['step_ms_median']:.1f} ms min {res_d['step_ms_min']:.1f} ms")
print("RESULT " + json.dumps(res_d))
if args.json:
  with open(args.json, "a") as f: f.write(json.dumps(res_d) + "\n")
