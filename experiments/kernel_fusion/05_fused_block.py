"""Step 4: hand-written fused kernel for the ConvFFN part of an encoder block, against what tinygrad and onnxruntime do.

For the chosen block(s) of the lean model it
  1. runs the real model with onnxruntime to capture the block's inputs (dw7x7 output u, residual t) and its output,
  2. extracts the sub-graph  u -> fc1 -> gelu(tanh) -> fc2 -> *gamma -> + t  as its own ONNX,
  3. times that sub-graph in tinygrad (TinyJit, current DEV, one pinned core) and onnxruntime (1 thread, graph optimisation ON and OFF),
  4. times three C versions (fused_block.c, AVX2+FMA, clang -O3): unfused (5 passes), epilogue-fused (2 passes), fully fused (1 pass),
     and checks each against the captured real output.
Usage: DEV="CPU:CLANG:x86_64,x86-64-v3" taskset -c 3 python -I 05_fused_block.py [--blocks 0.0 2.0] [--reps 60]
"""
import argparse, ctypes, os, statistics, subprocess, sys, tempfile, time
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models/driving_supercombo_lean_fp32.onnx"))
ap.add_argument("--ref", default=os.path.join(ROOT, "reference/driving_reference_fp32.npz"))
ap.add_argument("--blocks", nargs="+", default=["0.0", "2.0"], help="stage.block")
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--tmp", default=os.environ.get("TMPDIR", tempfile.gettempdir()))
ap.add_argument("--cc", default="clang")
ap.add_argument("--march", default="-mavx2 -mfma", help="C ISA flags; the tinygrad run here is x86-64-v3 (AVX2+FMA), so the default matches it")
args = ap.parse_args()

so_path = os.path.join(args.tmp, "fused_block.so")
subprocess.check_call([args.cc, "-O3", *args.march.split(), "-ffp-contract=fast", "-shared", "-fPIC", os.path.join(HERE, "fused_block.c"), "-o", so_path])
lib = ctypes.CDLL(so_path)
fp = ctypes.POINTER(ctypes.c_float)
def P(a): return a.ctypes.data_as(fp)

m = onnx.load(args.model); g = m.graph
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
  for i in n.input: cons.setdefault(i, []).append(n)

specs = []
for b in args.blocks:
  st, bl = b.split(".")
  fc1 = next(n for n in g.node if n.op_type == "Conv" and f"stages.{st}.blocks.{bl}.mlp.fc1.weight" in n.input[1])
  fc2 = next(n for n in g.node if n.op_type == "Conv" and f"stages.{st}.blocks.{bl}.mlp.fc2.weight" in n.input[1])
  mul = next(n for n in cons[fc2.output[0]] if n.op_type == "Mul")
  add = next(n for n in cons[mul.output[0]] if n.op_type == "Add")
  t_name = next(i for i in add.input if i != mul.output[0])
  specs.append(dict(tag=f"stage{st}.block{bl}", u=fc1.input[0], t=t_name, out=add.output[0], fc1=fc1, fc2=fc2, mul=mul))

# ---- 1. capture real activations with onnxruntime ----
ref = np.load(args.ref)
class H:
  def __init__(self): self.q = np.zeros((5, 2, 6, 128, 256), np.uint8)
  def push(self, f): self.q = np.concatenate([self.q[1:], np.asarray(f, np.uint8)[None]], axis=0)
  def stack(self): return np.concatenate([self.q[k, c] for c in (0, 1) for k in (0, 4)], axis=0)[None].astype(np.float32)
cap = onnx.load(args.model)
extra = sorted({s[k] for s in specs for k in ("u", "t", "out")})
for e in extra: cap.graph.output.append(helper.make_tensor_value_info(e, TensorProto_FLOAT := 1, None))
cap_path = os.path.join(args.tmp, "capture.onnx"); onnx.save(cap, cap_path)
sess = ort.InferenceSession(cap_path, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
hist = H(); st = {"state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
for t in range(6):
  hist.push(ref["frames"][t])
  res = dict(zip(names, sess.run(names, {"img_stack": hist.stack(), "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **st})))
  st = {k: res["next_" + k] for k in st}

def tmin(fn, reps):
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
  return min(ts) * 1e3, statistics.median(ts) * 1e3

results = []
for s in specs:
  u, t, y = (res[s[k]] for k in ("u", "t", "out"))
  _, C, Hh, Ww = u.shape; Pn = Hh * Ww
  W1 = init[s["fc1"].input[1]].reshape(-1, C); b1 = init[s["fc1"].input[2]]; Hd = W1.shape[0]
  W2 = init[s["fc2"].input[1]].reshape(C, Hd); b2 = init[s["fc2"].input[2]]; gam = init[s["mul"].input[1]].reshape(-1)
  print(f"\n== {s['tag']}: C={C} hidden={Hd} pixels={Pn}  MACs={2*Pn*C*Hd/1e6:.1f}M  hidden tensor {Hd*Pn*4/1e6:.2f} MB (fp32)  tile scratch {Hd*16*4/1024:.0f} KiB")
  # ---- 2. sub-graph ----
  sub_path = os.path.join(args.tmp, f"sub_{s['tag']}.onnx")
  onnx.utils.extract_model(args.model, sub_path, [s["u"], s["t"]], [s["out"]])
  # ---- 3a. onnxruntime on the sub-graph ----
  feed = {s["u"]: u, s["t"]: t}
  row = dict(tag=s["tag"], MMAC=Pn * C * Hd * 2 / 1e6)
  for lab, lvl in (("ort_opt_off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("ort_opt_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.graph_optimization_level = lvl
    ss = ort.InferenceSession(sub_path, so, providers=["CPUExecutionProvider"]); on = ss.get_outputs()[0].name
    out = ss.run([on], feed)[0]; row[lab + "_diff"] = float(np.abs(out - y).max())
    row[lab] = tmin(lambda: ss.run([on], feed), args.reps)[0]
  # ---- 3b. tinygrad on the sub-graph ----
  from tinygrad import Tensor, TinyJit, Device
  from tinygrad.nn.onnx import OnnxRunner
  from tinygrad.uop.ops import Ops
  runner = OnnxRunner(sub_path)
  @TinyJit
  def f(**kw): return runner(kw)[s["out"]].contiguous()
  tu, tt = Tensor(u).realize(), Tensor(t).realize()
  for _ in range(3): o = f(**{s["u"]: tu, s["t"]: tt}).realize()
  row["tg_diff"] = float(np.abs(o.numpy() - y).max())
  row["tg_kernels"] = sum(1 for c in f.captured.linear.src if c.src[0].op is Ops.PROGRAM)
  def run_tg():
    f(**{s["u"]: tu, s["t"]: tt}); Device[Device.DEFAULT].synchronize()
  row["tg"] = tmin(run_tg, args.reps)[0]
  # ---- 4. C kernels ----
  X = np.ascontiguousarray(u.reshape(C, Pn)); T = np.ascontiguousarray(t.reshape(C, Pn)); Yc = np.zeros((C, Pn), np.float32)
  Hs = np.zeros((Hd, Pn), np.float32); Zs = np.zeros((C, Pn), np.float32)
  W1c, W2c = np.ascontiguousarray(W1, np.float32), np.ascontiguousarray(W2, np.float32)
  b1c, b2c, gc = (np.ascontiguousarray(a, np.float32) for a in (b1, b2, gam))
  lib.ffn_unfused.argtypes = [ctypes.c_int] * 3 + [fp] * 10; lib.ffn_epilogue.argtypes = [ctypes.c_int] * 3 + [fp] * 9; lib.ffn_fused.argtypes = [ctypes.c_int] * 3 + [fp] * 8
  calls = {"c_unfused_5pass": lambda: lib.ffn_unfused(Pn, C, Hd, P(X), P(T), P(W1c), P(b1c), P(W2c), P(b2c), P(gc), P(Yc), P(Hs), P(Zs)),
           "c_epilogue_2pass": lambda: lib.ffn_epilogue(Pn, C, Hd, P(X), P(T), P(W1c), P(b1c), P(W2c), P(b2c), P(gc), P(Yc), P(Hs)),
           "c_fused_1pass": lambda: lib.ffn_fused(Pn, C, Hd, P(X), P(T), P(W1c), P(b1c), P(W2c), P(b2c), P(gc), P(Yc))}
  for k, fn in calls.items():
    Yc[:] = 0; fn(); row[k + "_diff"] = float(np.abs(Yc.reshape(1, C, Hh, Ww) - y).max())
    row[k] = tmin(fn, args.reps)[0]
  results.append(row)
  print(f"  correctness, max abs diff vs the real model's block output (output magnitude {np.abs(y).max():.2f}):")
  for k in ("ort_opt_off", "ort_opt_all", "tg", "c_unfused_5pass", "c_epilogue_2pass", "c_fused_1pass"):
    print(f"    {k:18s} {row[k + '_diff']:.2e}")
  print(f"  time, min of {args.reps} (ms); GFLOP/s counts 2 flops per MAC, gelu not counted:")
  for k in ("tg", "ort_opt_off", "ort_opt_all", "c_unfused_5pass", "c_epilogue_2pass", "c_fused_1pass"):
    extra = f"  ({row['tg_kernels']} kernels)" if k == "tg" else ""
    print(f"    {k:18s} {row[k]:8.3f} ms  {2*row["MMAC"]/row[k]:6.1f} GFLOP/s{extra}")
