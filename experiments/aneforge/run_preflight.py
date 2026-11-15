"""ANEForge static analysis of the openpilot driving model, with no ANE hardware.

Usage: python3 -I run_preflight.py <aneforge_clone> <model.onnx> <out_dir> [--ref <fp32.onnx> <ref.npz>]

Runs on Linux or macOS. Needs numpy and onnx (onnxruntime only for the optional numeric check).
It never compiles and never touches the ANE. Stages:
  1 whole-model import with aneforge.onnx_to_tensor (records the exact exception)
  2 per-node import: each ONNX node alone, fed fresh inputs, then preflight per family
  3 cut the graph after the uint8 queue bookkeeping, import the rest, preflight it
Writes <out_dir>/preflight.json and <out_dir>/preflight_table.md. Prints ANEFORGE-prefixed lines.
"""
import json, os, sys, traceback, warnings
from collections import Counter, defaultdict
import numpy as np

AF_DIR, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REF = sys.argv[sys.argv.index("--ref") + 1:sys.argv.index("--ref") + 3] if "--ref" in sys.argv else None
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("ANEFORGE_CACHE_DIR", os.path.join(os.path.abspath(OUT), "aneforge-cache"))  # keep MIL output out of ~/Models
sys.path.insert(0, os.path.abspath(AF_DIR))
import onnx
from onnx import shape_inference
import aneforge as af
from aneforge import _targets as tg
from aneforge import onnx as afonnx
from aneforge.graph import input as af_input, Tensor

def P(*a): print("ANEFORGE", *a, flush=True)
FAMS = {"A13": tg.Family.A13, "A15": tg.Family.A15, "A16": tg.Family.A16}
CUT_AFTER = "model/div"   # float16 [1,24,128,256], first conv input

P("version", af.__version__, "platform", sys.platform)
m = onnx.load(MODEL)
P("model opset", [o.version for o in m.opset_import], "nodes", len(m.graph.node))
res = {"aneforge_version": af.__version__, "platform": sys.platform}

def verdict(rep):
  """Collapse a Preflight to one verdict string plus detail."""
  if rep.reject: return "rejected", sorted({r.op for r in rep.reject})
  if rep.oversize: return "oversize", sorted({f"{r.op}{tuple(r.shape)}" for r in rep.oversize})
  if rep.decompose: return "decomposed", sorted({r.op for r in rep.decompose})
  return "native", []

# ---- stage 1: whole model
try:
  ins, out = af.onnx_to_tensor(MODEL)
  P("stage1 whole-model import OK", [t.shape for t in ins])
  res["stage1"] = "ok"
except Exception as e:
  res["stage1"] = f"{type(e).__name__}: {e}"
  P("stage1 whole-model import FAILED", res["stage1"])

# ---- stage 2: per node, standalone
mi = shape_inference.infer_shapes(m); g = mi.graph
shp = {}
for v in list(g.value_info) + list(g.input) + list(g.output):
  shp[v.name] = tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
inits = afonnx._inits(g)
rows = []
for idx, n in enumerate(g.node):
  vals = {k: v for k, v in inits.items()}
  ins_t = []
  for nm in n.input:
    if nm and nm not in inits:
      t = af_input(shp[nm]); vals[nm] = t; ins_t.append(t)
  row = {"idx": idx, "op": n.op_type, "name": n.name, "out_shape": list(shp.get(n.output[0], ()))}
  try:
    if n.op_type not in afonnx._ONNX: raise NotImplementedError(f"ONNX op '{n.op_type}' not supported")
    ins = [vals.get(x) if x else None for x in n.input]
    o = afonnx._ONNX[n.op_type](n, ins, afonnx._attrs(n), inits)
    o = o[0] if isinstance(o, (list, tuple)) else o
    if not isinstance(o, Tensor): row["import"] = "constant-folded"; row["fam"] = {f: ("native", []) for f in FAMS}
    else:
      row["import"] = "ok"
      # aneforge ops this node lowered to
      seen, st, lowered = set(), [o], Counter()
      stop = {id(t) for t in ins_t}
      while st:
        t = st.pop()
        if id(t) in seen or id(t) in stop: continue
        seen.add(id(t)); lowered[t.op] += 1; st.extend(t.srcs)
      row["af_ops"] = dict(lowered)
      row["fam"] = {}
      for fn, f in FAMS.items():
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          vd, det = verdict(tg.preflight(o, f))
        # preflight also walks the fresh inputs; only they may be oversize because of the shape, which is a real limit
        row["fam"][fn] = (vd, det)
  except Exception as e:
    row["import"] = f"{type(e).__name__}: {e}"
  rows.append(row)
res["nodes"] = rows
bad = [r for r in rows if r["import"] not in ("ok", "constant-folded")]
P("stage2 per-node import: ok", len(rows) - len(bad), "failed", len(bad))
for r in bad: P("stage2 node-import-fail", r["idx"], r["op"], r["name"], r["import"])

# ---- stage 3: cut after the bookkeeping, import the rest
cut_idx = next(i for i, n in enumerate(g.node) if CUT_AFTER in n.output) + 1
produced = {o for n in g.node[:cut_idx] for o in n.output}
need = []
for n in g.node[cut_idx:]:
  for x in n.input:
    if x and x not in inits and x in produced and x not in need: need.append(x)
    if x and x in {i.name for i in g.input} and x not in need: need.append(x)
# later nodes' internal tensors are not boundary tensors
need = [x for x in need if x in produced or x in {i.name for i in g.input}]
P("stage3 cut after node", cut_idx - 1, "boundary tensors", [(x, shp[x]) for x in need])
import onnx.utils
CUT = os.path.join(OUT, "driving_cut.onnx")
ext = onnx.utils.Extractor(mi)
cm = ext.extract_model(need, ["outputs"])
onnx.save(cm, CUT)
P("stage3 cut model nodes", len(cm.graph.node), "inputs", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in cm.graph.input])
res["cut"] = {"after": CUT_AFTER, "nodes": len(cm.graph.node), "inputs": {x: list(shp[x]) for x in need}}
res["cut_ops"] = dict(Counter(n.op_type for n in cm.graph.node))

def try_import(model, label):
  try:
    af.onnx_to_tensor(model); P(label, "import OK"); return None
  except Exception as e:
    P(label, "import FAILED", f"{type(e).__name__}: {e}"); return f"{type(e).__name__}: {e}"
res["stage3_import_unadapted"] = try_import(cm, "stage3 (unadapted cut)")

# ---- adaptations (importer limits), each exact or near-exact, listed in RESULTS.md
from onnx import numpy_helper, helper, TensorProto
def adapt(cm):
  """Edit the cut graph so the ANEForge importer can take it. Returns (model, list of change descriptions)."""
  g2 = cm.graph; info = shape_inference.infer_shapes(cm).graph
  S = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in list(info.value_info) + list(info.input) + list(info.output)}
  I = {t.name: numpy_helper.to_array(t) for t in g2.initializer}
  log, new_nodes = [], []
  def add_init(name, arr):
    g2.initializer.append(numpy_helper.from_array(np.asarray(arr), name)); I[name] = np.asarray(arr)
  for n in list(g2.node):
    if n.op_type == "GatherND" and n.input[1] in I and I[n.input[1]].ndim == 2 and I[n.input[1]].shape[1] == 1:
      dim = S[n.input[0]][0]; ix = [int(v) % dim for v in I[n.input[1]][:, 0]]
      assert ix == list(range(ix[0], ix[0] + len(ix))), "GatherND indices not contiguous"
      for nm, v in ((n.name + "_st", [ix[0]]), (n.name + "_en", [ix[-1] + 1]), (n.name + "_ax", [0])): add_init(nm, np.array(v, np.int64))
      new_nodes.append(helper.make_node("Slice", [n.input[0], n.name + "_st", n.name + "_en", n.name + "_ax"], list(n.output), name=n.name + "_as_slice"))
      log.append(f"GatherND {n.name}: constant contiguous indices -> Slice [{ix[0]}:{ix[-1]+1}] on axis 0 (exact)")
    elif n.op_type == "Where" and n.input[0] in I and n.input[1] in I and n.input[2] not in I:
      cond, val = I[n.input[0]].astype(bool), I[n.input[1]]
      bias = np.where(cond, np.float16(-60000.0) if np.isinf(val).all() else val, np.float16(0.0)).astype(np.float16)
      add_init(n.name + "_bias", bias)
      new_nodes.append(helper.make_node("Add", [n.input[2], n.name + "_bias"], list(n.output), name=n.name + "_as_add"))
      log.append(f"Where {n.name}: constant mask {cond.shape} and constant fill {float(val.ravel()[0])} -> Add of constant bias {{0, -60000}} (matches only if no row is fully masked)")
    elif n.op_type == "Concat" and any(x in I for x in n.input) and not all(x in I for x in n.input):
      ins2 = list(n.input)
      for k, x in enumerate(ins2):
        if x in I:
          assert not I[x].any(), "constant Concat operand is not all zeros"
          src = next(y for y in ins2 if y not in I)
          shp_c = I[x].shape; ax = int(next((a.i for a in n.attribute if a.name == "axis"), 0)) % len(shp_c)
          assert shp_c[ax] > 0
          for nm, v in ((x + "_st", [0]), (x + "_en", [shp_c[ax]]), (x + "_ax", [ax])): add_init(nm, np.array(v, np.int64))
          # zeros built from the graph itself: slice of a sibling tensor minus itself
          new_nodes.append(helper.make_node("Slice", [src, x + "_st", x + "_en", x + "_ax"], [x + "_s"], name=x + "_zs"))
          new_nodes.append(helper.make_node("Sub", [x + "_s", x + "_s"], [x + "_z"], name=x + "_zsub"))
          ins2[k] = x + "_z"
      new_nodes.append(helper.make_node("Concat", ins2, list(n.output), name=n.name, axis=ax))
      log.append(f"Concat {n.name}: all-zero constant operand {shp_c} -> zeros made in-graph as x-x of a slice of a sibling tensor (exact unless that tensor holds inf/nan)")
    else:
      new_nodes.append(n)
  del g2.node[:]; g2.node.extend(new_nodes)
  used = {x for n in g2.node for x in n.input}
  for t in [t for t in g2.initializer if t.name not in used]: g2.initializer.remove(t)
  for vi in [v for v in g2.input if v.name not in used]: g2.input.remove(vi); log.append(f"dropped unused graph input {vi.name}")
  return cm, log

import copy
cm2, changes = adapt(copy.deepcopy(cm))
for c in changes: P("adaptation", c)
res["adaptations"] = changes
onnx.checker.check_model(cm2)
ADAPTED = os.path.join(OUT, "driving_cut_adapted.onnx"); onnx.save(cm2, ADAPTED)
res["stage3_import_adapted"] = try_import(cm2, "stage3 (adapted cut)")
cm = cm2; CUT = ADAPTED

try:
  ins3, out3 = af.onnx_to_tensor(cm)
  res["stage3_import"] = "ok"; P("stage3 import OK")
  for fn, f in FAMS.items():
    rep = tg.preflight(out3, f)
    vd, det = verdict(rep)
    cnt = {k: len(getattr(rep, k)) for k in ("native", "decompose", "reject", "oversize")}
    res[f"stage3_preflight_{fn}"] = {"verdict": vd, "detail": det, "counts": cnt, "ok": rep.ok,
      "reject_nodes": [[r.op, list(r.shape)] for r in rep.reject], "oversize_nodes": [[r.op, list(r.shape)] for r in rep.oversize]}
    P("stage3 preflight", fn, vd, cnt, det[:12])
    res.setdefault("stage3_ops", {})[fn] = dict(Counter(r.op for r in rep.native + rep.decompose + rep.reject))
except Exception as e:
  res["stage3_import"] = f"{type(e).__name__}: {e}"
  P("stage3 import FAILED", res["stage3_import"])

# ---- stage 4: lower the imported graph to MIL text (no compile, no device) and count MIL ops
if res.get("stage3_import") == "ok":
  try:
    import re
    from aneforge import _compile as C
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      d = C._lower_fused_to_dir(out3, None, int8=False)
    mil = open(os.path.join(str(d), "model.mil")).read()
    hist = Counter(x for x in re.findall(r"=\s+([a-z_0-9]+)\(", mil) if x not in ("string", "uint64", "bool", "fp16", "int32"))
    res["stage4_mil"] = {"bytes": len(mil), "weights_bytes": os.path.getsize(os.path.join(str(d), "weights.bin")), "op_histogram": dict(hist)}
    open(os.path.join(OUT, "driving_cut.mil"), "w").write(mil)
    P("stage4 MIL lowering OK, mil bytes", len(mil), "weights.bin bytes", res["stage4_mil"]["weights_bytes"], dict(hist.most_common()))
  except Exception as e:
    res["stage4_mil"] = f"{type(e).__name__}: {e}"; P("stage4 MIL lowering FAILED", res["stage4_mil"])

# ---- optional numeric check of the cut graph
if REF and os.path.exists(REF[0]) and os.path.exists(REF[1]):
  try:
    import onnxruntime as ort
    r = np.load(REF[1]); steps = 3
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    full = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    mm = onnx.load(MODEL)
    for x in need:
      if x not in [o.name for o in mm.graph.output]: mm.graph.output.append(onnx.helper.make_empty_tensor_value_info(x))
    full_x = ort.InferenceSession(mm.SerializeToString(), so, providers=["CPUExecutionProvider"])
    cutsess = ort.InferenceSession(CUT, so, providers=["CPUExecutionProvider"])
    outn = [o.name for o in full_x.get_outputs()]
    state = {"state_img_q": np.zeros((2,5,6,128,256), np.uint8), "state_desire_q": np.zeros((100,1,8), np.float32), "state_feat_q": np.zeros((96,1,512), np.float32)}
    d_full_cut, d_cut_ref, d_full_ref = [], [], []
    for t in range(steps):
      feed = {"new_img": r["frames"][t], "desire": r["desire"][t], "traffic_convention": r["traffic_convention"][t], "action_t": r["action_t"][t], **state}
      o = dict(zip(outn, full_x.run(outn, feed)))
      cin = {i.name: o[i.name] for i in cutsess.get_inputs()}
      for i in cutsess.get_inputs():
        if i.type == "tensor(float16)": cin[i.name] = cin[i.name].astype(np.float16)
      co = cutsess.run(None, cin)[0]
      d_full_cut.append(float(np.abs(co.astype(np.float32) - o["outputs"].astype(np.float32)).max()))
      d_full_ref.append(float(np.abs(o["outputs"].astype(np.float32)[0] - r["outputs"][t]).max()))
      d_cut_ref.append(float(np.abs(co.astype(np.float32)[0] - r["outputs"][t]).max()))
      state = {k: o[f"next_{k}"] for k in state}
    res["numeric"] = {"steps": steps, "max_abs_cut_vs_full_fp16": d_full_cut, "max_abs_cut_vs_fp32_reference": d_cut_ref, "max_abs_full_fp16_vs_fp32_reference": d_full_ref, "ref_output_max_abs": float(np.abs(r["outputs"][:steps]).max())}
    P("numeric cut-vs-full(fp16) max abs per step", d_full_cut, "cut-vs-fp32-reference", d_cut_ref, "full-fp16-vs-fp32-reference", d_full_ref)
  except Exception as e:
    res["numeric"] = f"{type(e).__name__}: {e}"; P("numeric check FAILED", res["numeric"])

# ---- table
agg = defaultdict(lambda: {"n": 0, "imp_fail": 0, "fam": {f: Counter() for f in FAMS}, "af": Counter(), "detail": defaultdict(set), "err": set()})
for r in rows:
  a = agg[r["op"]]; a["n"] += 1
  if r["import"] not in ("ok", "constant-folded"):
    a["imp_fail"] += 1; a["err"].add(r["import"][:160]); continue
  for f, (vd, det) in r["fam"].items():
    a["fam"][f][vd] += 1; a["detail"][f].update(det)
  for k, v in r.get("af_ops", {}).items(): a["af"][k] += v
lines = ["| ONNX op | nodes | import | A16 | A15 | A13 | lowers to (aneforge ops, all nodes) |", "|---|---|---|---|---|---|---|"]
for op, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
  def cell(f):
    c = a["fam"][f]
    if not c: return "-"
    s = ", ".join(f"{v} {k}" for k, v in c.items())
    d = sorted(a["detail"][f])
    return s + (f" ({'; '.join(d)[:90]})" if d and set(c) != {"native"} else "")
  imp = "ok" if not a["imp_fail"] else f"{a['imp_fail']}/{a['n']} FAILED: {'; '.join(sorted(a['err']))[:140]}"
  lines.append(f"| {op} | {a['n']} | {imp} | {cell('A16')} | {cell('A15')} | {cell('A13')} | {', '.join(f'{k}x{v}' for k, v in a['af'].most_common())} |")
open(os.path.join(OUT, "preflight_table.md"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "preflight.json"), "w"), indent=1, default=str)
P("wrote", OUT)
