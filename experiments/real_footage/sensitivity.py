"""Sensitivity experiments (float16, GELU variants, weight compression) on the real-footage reference AND the synthetic
reference, with raw output differences (same three numbers as the existing scripts) plus driving-relevant differences.
The variant builders are taken verbatim from experiments/gelu_sensitivity/gelu_variants.py and
experiments/weight_compression/quantize_study.py (their source text is exec'd up to the point where they start running).

Usage: python -I sensitivity.py {f16|gelu|quant} [threads]     -> writes $RF_SCRATCH/sens_<group>.json
       python -I sensitivity.py report                          -> prints markdown tables from the json files
"""
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
from rf_common import *      # noqa

FP32 = f"{ROOT}/models/driving_supercombo_fp32.onnx"; FP16 = f"{ROOT}/models/driving_supercombo.onnx"
REFS = {"real": f"{SCRATCH}/real_reference_fp32.npz", "synthetic": f"{ROOT}/reference/driving_reference_fp32.npz"}
WARM = 100

def run_bytes(model, ref, threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]); outs = [o.name for o in sess.get_outputs()]
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    res = []
    for t in range(len(ref["frames"])):
        o = dict(zip(outs, sess.run(outs, {"new_img": ref["frames"][t], "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **st})))
        res.append(o["outputs"][0].astype(np.float32)); st = {k: o["next_" + k] for k in st}
    return np.stack(res)

def load_refs():
    R = {}
    for k, p in REFS.items():
        r = load_ref(p); n = len(r["frames"])
        v = np.load(f"{SCRATCH}/real_meta.npz")["v_ego"] if k == "real" else np.full(n, 15.0)   # synthetic has no speed: 15 m/s
        R[k] = dict(ref=r, v=v, q=driving_quantities(r["outputs"], v))
    return R

def evaluate(model, R, threads, label):
    out = {}
    for k, d in R.items():
        t0 = time.time(); y = run_bytes(model, d["ref"], threads); q = driving_quantities(y, d["v"])
        n = len(y); sls = {"all": slice(None)} if k == "synthetic" else {"all": slice(None), "warm": slice(WARM, None)}
        out[k] = {s: dict(raw=raw_diff(y, d["ref"]["outputs"], sl), drv=driving_diff(q, d["q"], sl)) for s, sl in sls.items()}
        print(f"{label:44} {k:9} raw(all) worst {out[k]['all']['raw'][0]:.3f} mean {out[k]['all']['raw'][2]:.5f}  [{time.time() - t0:.0f}s]", flush=True)
    return out

def gelu_cases():
    src = open(f"{ROOT}/experiments/gelu_sensitivity/gelu_variants.py").read().split("xs = np.linspace")[0]
    sys.argv = [sys.argv[0], FP32, REFS["synthetic"]]; g = {}; exec(compile(src, "gelu_variants.py", "exec"), g)
    cases = [("tanh form (as shipped)", "tanh", {}), ("exact erf form", "exact", {}), ("x * sigmoid(1.702 x)", "sigmoid", {})]
    cases += [(f"piecewise linear, {K} segments", "pwl", {"segments": K}) for K in (8, 16, 32, 64)]
    return [(l, (lambda kind=kind, kw=kw: g["variant_model"](kind, **kw))) for l, kind, kw in cases]

def quant_cases():
    src = open(f"{ROOT}/experiments/weight_compression/quantize_study.py").read().split("def run(fn):")[0]
    sys.argv = [sys.argv[0], FP32, REFS["synthetic"]]; g = {}; exec(compile(src, "quantize_study.py", "exec"), g)
    base, used = g["base"], g["used_as_weight"]
    def build(fn):
        m = onnx.ModelProto(); m.CopyFrom(base)
        for i, init in enumerate(m.graph.initializer):
            if init.name in used and init.data_type == onnx.TensorProto.FLOAT:
                w = numpy_helper.to_array(init)
                if w.size < 1024: continue
                m.graph.initializer[i].CopyFrom(numpy_helper.from_array(fn(w).astype(np.float32), init.name))
        return m.SerializeToString()
    cs = [("float16 weights (what ships today)", g["fp16"]), ("8-bit linear, per output channel", g["int8_per_channel"]), ("8-bit linear, per tensor", g["int8_per_tensor"]),
          ("8-bit lookup table (256 values per tensor)", g["palettize"](8)), ("6-bit lookup table (64 values)", g["palettize"](6)), ("4-bit lookup table (16 values)", g["palettize"](4))]
    return [(l, (lambda fn=fn: build(fn))) for l, fn in cs]

def report():
    C = {}
    for grp in ("f16", "gelu", "quant"):
        p = f"{SCRATCH}/sens_{grp}.json"
        if os.path.exists(p): C[grp] = json.load(open(p))
    titles = {"f16": "Float16 model vs float32 reference", "gelu": "GELU variants (float32 model)", "quant": "Weight compression (float32 model, weights replaced)"}
    for grp, res in C.items():
        print(f"\n### {titles[grp]}: raw output differences (all 2576 outputs)\n")
        print("| variant | real warm: worst / worst-step-mean / mean | real all steps | synthetic (40 steps) |\n|---|---|---|---|")
        f = lambda r: "%.3f / %.4f / %.5f" % tuple(r)
        for lab, d in res.items(): print(f"| {lab} | {f(d['real']['warm']['raw'])} | {f(d['real']['all']['raw'])} | {f(d['synthetic']['all']['raw'])} |")
        print(f"\n### {titles[grp]}: driving-relevant differences, mean / worst over steps (real = warm steps 100-599; synthetic = all 40, speed assumed 15 m/s)\n")
        hdr = "| variant | data | " + " | ".join(n for n, _ in METRICS) + " |"; print(hdr); print("|---|---|" + "---|" * len(METRICS))
        for lab, d in res.items():
            for k, s in (("real", "warm"), ("synthetic", "all")):
                print(f"| {lab} | {k} | " + " | ".join("%.2g / %.2g" % tuple(d[k][s]["drv"][m]) for _, m in METRICS) + " |")

if __name__ == "__main__":
    grp = sys.argv[1]
    if grp == "report": report(); sys.exit()
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cases = {"f16": lambda: [("float16 model (original ONNX)", lambda: FP16)], "gelu": gelu_cases, "quant": quant_cases}[grp]()
    R = load_refs(); res = {}
    for label, mk in cases:
        res[label] = evaluate(mk(), R, threads, label)
        json.dump(res, open(f"{SCRATCH}/sens_{grp}.json", "w"), indent=1)
