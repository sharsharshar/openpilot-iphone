"""Step 1: per-ONNX-node cost of the small driving model.

Analytic: multiply-accumulates (MACs), output elements and bytes per node (from onnx shape inference).
Measured: per-node CPU time from onnxruntime's built-in profiler (SessionOptions.enable_profiling), one thread,
graph optimisations OFF (ORT_DISABLE_ALL) so every profile event is one original ONNX node (with optimisations on,
ORT fuses nodes and the events are named after the fused nodes).

Usage:  python -I 01_node_cost.py [--model PATH] [--runs 30] [--warm 3] [--out DIR]
Pin to one core from outside:  taskset -c 3 python -I 01_node_cost.py
State (desire queue, feature queue, image queue) is carried between steps exactly like reference_run.py.
Writes <out>/node_cost.csv (one row per node) and prints markdown tables.
"""
import argparse, collections, csv, json, os, re, statistics, sys, tempfile, time
import numpy as np
import onnx
from onnx import shape_inference, numpy_helper
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models/driving_supercombo_fp32.onnx"))
ap.add_argument("--ref", default=os.path.join(ROOT, "reference/driving_reference_fp32.npz"))
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--out", default=os.path.join(HERE, "data"))
ap.add_argument("--tmp", default=os.environ.get("TMPDIR", tempfile.gettempdir()))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

m = onnx.load(args.model)
g = m.graph
# unique, ordered names so each profile event maps to exactly one node (some nodes in the file have empty names)
orig_names = [n.name for n in g.node]
for i, n in enumerate(g.node):
    n.name = f"{i:03d}"
tmp_model = os.path.join(args.tmp, "named_model.onnx")
onnx.save(m, tmp_model)

mi = shape_inference.infer_shapes(m, data_prop=True)
shape = {}
for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
    shape[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
init = {t.name: t for t in g.initializer}
for t in g.initializer: shape.setdefault(t.name, list(t.dims))
const_out = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value": shape[n.output[0]] = list(a.t.dims)

def prod(x):
    r = 1
    for v in x: r *= v
    return r

def attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default

def macs(n):
    out = shape[n.output[0]]
    if n.op_type == "Conv":
        w = shape[n.input[1]]; group = attr(n, "group", 1)
        return prod(out) * w[1] * prod(w[2:])  # w = [Cout, Cin/group, kh, kw]
    if n.op_type == "Gemm":
        a = shape[n.input[0]]; ta = attr(n, "transA", 0)
        k = a[0] if ta else a[1]
        return out[0] * out[1] * k
    if n.op_type == "MatMul":
        a = shape[n.input[0]]
        return prod(out) * a[-1]
    return 0

# stage assignment: by position / by weight names
def kind_of(n):
    if n.op_type != "Conv": return n.op_type
    w = shape[n.input[1]]; grp = attr(n, "group", 1); k = "x".join(map(str, w[2:]))
    if grp == w[0] and grp > 1: return f"Conv depthwise {k}"
    if grp > 1: return f"Conv grouped {k}"
    return "Conv 1x1" if k == "1x1" else f"Conv dense {k}"

def stage_of(i, n, cur):
    names = " ".join(n.input)
    mm = re.search(r"_en\.stages\.(\d)", names)
    if mm: cur["enc"] = f"encoder stage {mm.group(1)}"
    if "_en.stem" in names: cur["enc"] = "encoder stem"
    if "_en.final_conv" in names: cur["enc"] = "encoder final conv+SE+pool"
    if i <= 23: return "state/input prep (slice, gather, cast, concat)"
    if i <= 25: return "input normalisation (Sub, Div)"
    if i <= 134: return cur["enc"]
    if i <= 283: return "per-frame heads (vision policy + no_bottleneck + summarizer)"
    if i <= 289: return "glue (slice/reshape/concat)"
    if i <= 341: return "temporal transformer"
    if i <= 372: return "temporal heads + output concat"
    return "state carry (slice/concat)"

cur = {"enc": "encoder stem"}
rows = []
for i, n in enumerate(g.node):
    out = shape[n.output[0]]
    nel = prod(out)
    wbytes = sum(init[x].raw_data and len(init[x].raw_data) or numpy_helper.to_array(init[x]).nbytes for x in n.input if x in init)
    rows.append(dict(idx=i, name=orig_names[i] or f"(unnamed {n.op_type})", op=n.op_type, kind=kind_of(n), stage=stage_of(i, n, cur),
                     out_shape="x".join(map(str, out)), out_elems=nel, out_bytes_f32=nel * 4, weight_bytes=wbytes, macs=macs(n)))

# ---------------- onnxruntime profiling ----------------
ref = np.load(args.ref)
def make_session(level, profile):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = level
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(args.tmp, "ortprof")
    return ort.InferenceSession(tmp_model, so, providers=["CPUExecutionProvider"])

def run_steps(sess, nsteps):
    outs = [o.name for o in sess.get_outputs()]
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32),
          "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    wall = []
    for t in range(nsteps):
        k = t % len(ref["frames"])
        feed = {"new_img": ref["frames"][k], "desire": ref["desire"][k], "traffic_convention": ref["traffic_convention"][k],
                "action_t": ref["action_t"][k], **st}
        t0 = time.perf_counter(); res = sess.run(outs, feed); wall.append(time.perf_counter() - t0)
        d = dict(zip(outs, res)); st = {s: d["next_" + s] for s in st}
    return wall

sess = make_session(ort.GraphOptimizationLevel.ORT_DISABLE_ALL, True)
wall_prof = run_steps(sess, args.runs + args.warm)
pf = sess.end_profiling()
ev = json.load(open(pf))
per = collections.defaultdict(list); opn = {}
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        nm = e["name"][: -len("_kernel_time")]
        per[nm].append(e["dur"]); opn[nm] = e["args"].get("op_name")
n_per = {k: len(v) for k, v in per.items()}
assert len(per) == len(rows), (len(per), len(rows))
assert set(n_per.values()) == {args.runs + args.warm}, collections.Counter(n_per.values())
for r in rows:
    d = per[f"{r['idx']:03d}"][args.warm:]
    r["us_median"] = statistics.median(d); r["us_min"] = min(d)
    assert opn[f"{r['idx']:03d}"] == r["op"]
os.remove(pf)

# unprofiled wall time, optimisations off and on, for reference
walls = {}
for label, lvl in (("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("ORT_ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
    s = make_session(lvl, False); w = run_steps(s, args.runs + args.warm)[args.warm:]
    walls[label] = (statistics.median(w) * 1e3, min(w) * 1e3)

with open(os.path.join(args.out, "node_cost.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

# ---------------- tables ----------------
tot_us = sum(r["us_median"] for r in rows); tot_mac = sum(r["macs"] for r in rows)
print(f"model {os.path.relpath(args.model, ROOT)}: {len(rows)} nodes, {tot_mac/1e6:.1f} M MACs (= {2*tot_mac/1e9:.3f} G ops at 2 ops/MAC)")
print(f"ORT 1 thread: sum of per-node median kernel time (opt off) {tot_us/1e3:.1f} ms; profiled whole-run median {statistics.median(wall_prof[args.warm:])*1e3:.1f} ms")
for k, (med, mn) in walls.items(): print(f"ORT unprofiled whole run {k}: median {med:.1f} ms, min {mn:.1f} ms (n={args.runs})")
print(f"nodes with median kernel time < 5 us (timer resolution is 1 us): {sum(r['us_median']<5 for r in rows)}")

def table(key, title):
    agg = collections.OrderedDict()
    for r in rows:
        a = agg.setdefault(r[key], [0, 0, 0.0, 0]); a[0] += 1; a[1] += r["macs"]; a[2] += r["us_median"]; a[3] += r["out_bytes_f32"]
    print(f"\n### {title}\n| {key} | nodes | MMACs | MAC share | time ms | time share | MB out |\n|---|---|---|---|---|---|---|")
    for k, (c, mc, us, ob) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print(f"| {k} | {c} | {mc/1e6:.1f} | {100*mc/tot_mac:.1f}% | {us/1e3:.2f} | {100*us/tot_us:.1f}% | {ob/1e6:.2f} |")
table("stage", "Share per stage")
table("op", "Share per op type")
table("kind", "Share per op type, convolutions split by kind")
print("\n### Top 15 nodes by measured time\n| # | idx | node | op | out shape | MMACs | ms | time share | cum |\n|---|---|---|---|---|---|---|---|---|")
cum = 0
for i, r in enumerate(sorted(rows, key=lambda r: -r["us_median"])[:15]):
    cum += r["us_median"]
    print(f"| {i+1} | {r['idx']} | {r['name'].replace('model/','')} | {r['op']} | {r['out_shape']} | {r['macs']/1e6:.1f} | {r['us_median']/1e3:.2f} | {100*r['us_median']/tot_us:.1f}% | {100*cum/tot_us:.1f}% |")
print("\n### Top 15 nodes by MACs\n| # | idx | node | MMACs | MAC share | ms | time share |\n|---|---|---|---|---|---|---|")
for i, r in enumerate(sorted(rows, key=lambda r: -r["macs"])[:15]):
    print(f"| {i+1} | {r['idx']} | {r['name'].replace('model/','')} | {r['macs']/1e6:.1f} | {100*r['macs']/tot_mac:.1f}% | {r['us_median']/1e3:.2f} | {100*r['us_median']/tot_us:.1f}% |")
convs = [r for r in rows if r["op"] == "Conv"]
print("\n### Concentration")
for k in (3, 5, 10, 20):
    t = sorted(rows, key=lambda r: -r["us_median"])[:k]
    mc = sorted(rows, key=lambda r: -r["macs"])[:k]
    print(f"top {k} nodes: {100*sum(r['us_median'] for r in t)/tot_us:.1f}% of time; top {k} by MACs: {100*sum(r['macs'] for r in mc)/tot_mac:.1f}% of MACs")
for lim in (3, 6, 10):
    c = convs[:lim]
    print(f"first {lim} convolutions (graph order): {100*sum(r['macs'] for r in c)/tot_mac:.1f}% of MACs, {100*sum(r['us_median'] for r in c)/tot_us:.1f}% of time")
enc = [r for r in rows if r["stage"].startswith("encoder")]
print(f"all encoder: {100*sum(r['macs'] for r in enc)/tot_mac:.1f}% MACs, {100*sum(r['us_median'] for r in enc)/tot_us:.1f}% time")
nm = [r for r in rows if r["macs"] == 0]
print(f"nodes with zero MACs (data movement / elementwise / norm / softmax): {len(nm)} nodes, {100*sum(r['us_median'] for r in nm)/tot_us:.1f}% of time")
print(f"nodes needed for 90% of time: {next(i+1 for i in range(len(rows)) if sum(r['us_median'] for r in sorted(rows, key=lambda r: -r['us_median'])[:i+1]) >= 0.9*tot_us)}")
# resolution of first convs
print("\nFirst convolutions in graph order:")
for r in convs[:6]: print(f"  idx {r['idx']} {r['name']} out {r['out_shape']} {r['macs']/1e6:.1f} MMACs {r['us_median']/1e3:.2f} ms")
