"""Step 3b: graph-level rewrites on the lean fp32 ONNX model, done before tinygrad sees it.

Variants written to --outdir (cumulative):
  lean            models/driving_supercombo_lean_fp32.onnx unchanged (reference point)
  fold_norm       Cast -> Sub(mean) -> Div(std) -> Conv(stem.0, pads=1)   becomes   Cast -> Pad(raw 127.5) -> Conv(w/std, b - sum(w/std*mean), pads=0)
                  (the zero padding of the original conv acts on the normalised image, i.e. raw value = mean, so the padding is made explicit)
  fold_scale      Conv(fc2) -> Mul(layer_scale gamma)  becomes  Conv(fc2 with gamma folded into weight and bias)   (20 blocks... see log)
  concat_heads    Gemms that read the same tensor (the first layer of the parallel output heads) become one wide Gemm + Slices
Each variant is checked with onnxruntime against reference/driving_reference_fp32.npz (history kept in numpy as scripts/verify_lean.py does).
Usage: python -I 04_graph_rewrites.py --outdir DIR
"""
import argparse, copy, os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(ROOT, "models/driving_supercombo_lean_fp32.onnx"))
ap.add_argument("--ref", default=os.path.join(ROOT, "reference/driving_reference_fp32.npz"))
ap.add_argument("--outdir", required=True)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def arr(m, name):
  for t in m.graph.initializer:
    if t.name == name: return numpy_helper.to_array(t)
  raise KeyError(name)
def set_init(m, name, a):
  for t in m.graph.initializer:
    if t.name == name:
      t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a, np.float32), name)); return
  raise KeyError(name)
def attr(n, k, d=None):
  for a in n.attribute:
    if a.name == k: return helper.get_attribute_value(a)
  return d

def fold_norm(m):
  g = m.graph
  nodes = list(g.node)
  sub = next(n for n in nodes if n.op_type == "Sub" and "_mean" in n.input[1])
  div = next(n for n in nodes if n.op_type == "Div" and n.input[0] == sub.output[0])
  conv = next(n for n in nodes if n.input[0] == div.output[0])
  assert conv.op_type == "Conv" and attr(conv, "pads") == [1, 1, 1, 1] and attr(conv, "group") == 1
  mean = arr(m, sub.input[1]).reshape(-1); std = arr(m, div.input[1]).reshape(-1)
  assert np.all(mean == mean[0]), "per-channel means differ: explicit Pad with one constant is not exact"
  W = arr(m, conv.input[1]); B = arr(m, conv.input[2])
  W2 = W / std[None, :, None, None]
  B2 = B - (W2 * mean[None, :, None, None]).sum(axis=(1, 2, 3))
  set_init(m, conv.input[1], W2); set_init(m, conv.input[2], B2)
  pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "fold_norm_pads")
  cval = numpy_helper.from_array(np.array(float(mean[0]), np.float32), "fold_norm_cval")
  g.initializer.extend([pads, cval])
  pad = helper.make_node("Pad", [sub.input[0], "fold_norm_pads", "fold_norm_cval"], ["fold_norm_padded"], name="fold_norm_pad", mode="constant")
  for a in list(conv.attribute):
    if a.name == "pads": conv.attribute.remove(a)
  conv.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))
  conv.input[0] = "fold_norm_padded"
  idx = nodes.index(sub)
  new = nodes[:idx] + [pad] + [n for n in nodes[idx:] if n is not sub and n is not div]
  del g.node[:]; g.node.extend(new)
  return m

def fold_scale(m):
  g = m.graph; nodes = list(g.node); drop = []; count = 0
  prod = {o: n for n in nodes for o in n.output}
  for n in nodes:
    if n.op_type != "Mul": continue
    a, b = n.input
    cv = prod.get(a)
    if cv is None or cv.op_type != "Conv" or "mlp.fc2" not in cv.input[1]: continue
    gamma = arr(m, b).reshape(-1); W = arr(m, cv.input[1]); B = arr(m, cv.input[2])
    set_init(m, cv.input[1], W * gamma[:, None, None, None]); set_init(m, cv.input[2], B * gamma)
    cv.output[0] = n.output[0]; drop.append(n); count += 1
  keep = [n for n in nodes if n not in drop]
  del g.node[:]; g.node.extend(keep)
  print(f"  fold_scale: folded {count} layer_scale Mul nodes")
  return m

def concat_heads(m):
  g = m.graph; nodes = list(g.node)
  by_in = {}
  for n in nodes:
    if n.op_type == "Gemm" and attr(n, "transB", 0) == 1 and attr(n, "transA", 0) == 0 and len(n.input) == 3:
      by_in.setdefault(n.input[0], []).append(n)
  groups = [v for v in by_in.values() if len(v) >= 2]
  drop = set(); insert_at = {}
  for gi, grp in enumerate(groups):
    Ws = [arr(m, n.input[1]) for n in grp]; Bs = [arr(m, n.input[2]) for n in grp]
    assert len({w.shape[1] for w in Ws}) == 1
    wn, bn, on = f"cat{gi}_W", f"cat{gi}_B", f"cat{gi}_out"
    g.initializer.extend([numpy_helper.from_array(np.concatenate(Ws, 0), wn), numpy_helper.from_array(np.concatenate(Bs, 0), bn)])
    new = [helper.make_node("Gemm", [grp[0].input[0], wn, bn], [on], name=f"cat{gi}_gemm", alpha=1.0, beta=1.0, transB=1)]
    off = 0
    for k, n in enumerate(grp):
      w = Ws[k].shape[0]
      for nm, v in ((f"cat{gi}_{k}_st", [off]), (f"cat{gi}_{k}_en", [off + w]), (f"cat{gi}_{k}_ax", [1])):
        g.initializer.append(numpy_helper.from_array(np.array(v, np.int64), nm))
      new.append(helper.make_node("Slice", [on, f"cat{gi}_{k}_st", f"cat{gi}_{k}_en", f"cat{gi}_{k}_ax"], [n.output[0]], name=f"cat{gi}_{k}_slice"))
      off += w; drop.add(id(n))
    insert_at[id(grp[0])] = new
  out = []
  for n in nodes:
    if id(n) in insert_at: out += insert_at[id(n)]
    elif id(n) in drop: continue
    else: out.append(n)
  del g.node[:]; g.node.extend(out)
  print(f"  concat_heads: {len(groups)} groups of Gemms with a shared input, sizes {[len(v) for v in groups]}")
  return m

# ---- verification with onnxruntime ----
class NumpyHistory:
  def __init__(self): self.q = np.zeros((5, 2, 6, 128, 256), np.uint8)
  def push(self, f): self.q = np.concatenate([self.q[1:], np.asarray(f, np.uint8)[None]], axis=0)
  def stack(self): return np.concatenate([self.q[k, c] for c in (0, 1) for k in (0, 4)], axis=0)[None].astype(np.float32)
def check(path):
  import onnxruntime as ort
  ref = np.load(args.ref); so = ort.SessionOptions(); so.intra_op_num_threads = 1
  s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); outs = [o.name for o in s.get_outputs()]
  st = {"state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}; h = NumpyHistory(); d = []
  for t in range(len(ref["frames"])):
    h.push(ref["frames"][t])
    r = dict(zip(outs, s.run(outs, {"img_stack": h.stack(), "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **st})))
    st = {k: r["next_" + k] for k in st}; d.append(np.abs(r["outputs"][0] - ref["outputs"][t]))
  d = np.array(d); return float(d.max()), float(d.mean())

variants = [("lean", []), ("lean_foldnorm", [fold_norm]), ("lean_foldnorm_foldscale", [fold_norm, fold_scale]),
            ("lean_foldnorm_foldscale_concat", [fold_norm, fold_scale, concat_heads])]
for name, fns in variants:
  m = onnx.load(args.src)
  for f in fns: m = f(m)
  onnx.checker.check_model(m)
  p = os.path.join(args.outdir, name + ".onnx"); onnx.save(m, p)
  mx, mn = check(p)
  print(f"{name}: nodes={len(m.graph.node)} ORT worst max abs diff vs reference {mx:.3e}, mean {mn:.3e}  -> {p}", flush=True)
