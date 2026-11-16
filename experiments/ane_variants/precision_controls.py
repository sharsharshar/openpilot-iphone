"""Controls that say how to read the differences in verify_variants.py.

Usage: python -I precision_controls.py <variants_dir> <real_reference.npz> <out.json>

1. ulp_stem / ulp_all: the lean model with the stem conv's (or every) weight and bias multiplied by (1 + u * 2^-24), u uniform in
   [-1, 1]: a different but equally valid float32 rounding of the same network. Its drift from the reference over 600 recurrent
   steps is the yardstick for "float32 rounding level" (variant B's fold, E5 and C's re-ordered sums are of this kind).
2. fp16w: base / B / ALL_DROPIN with every float weight rounded to the nearest float16 (activations stay float32). This says
   whether folding the normalisation into the stem conv (a bigger bias, w/63.75) makes the model more sensitive to float16
   WEIGHTS than the original; it does not model float16 activations or the Neural Engine.
"""
import json, os, sys, time
import numpy as np, onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verify_variants as V

vdir, realp, jout = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = os.path.dirname(os.path.dirname(HERE))
syn = np.load(os.path.join(ROOT, "reference", "driving_reference_fp32.npz")); syn = {k: syn[k] for k in syn.files}

class D(dict):
    files = property(lambda s: list(s.keys()))
syn = D(syn)
rr = np.load(realp); rr = D({k: rr[k] for k in ("frames", "desire", "traffic_convention", "action_t", "outputs")})
data = [("synthetic", syn, V.precompute_stacks(syn["frames"])), ("real", rr, V.precompute_stacks(rr["frames"]))]
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))      # SEED=1 JOBS=ulp_stem,ulp_all repeats the ulp controls with other noise

def transform(src, fn):
    m = onnx.load(src)
    for i, t in enumerate(m.graph.initializer):
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32:
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(fn(t.name, a), t.name))
    return m

def ulp(sel):
    def f(name, a):
        if not sel(name) or a.size < 2:
            return a
        return (a.astype(np.float64) * (1 + rng.uniform(-1, 1, a.shape) * 2.0 ** -24)).astype(np.float32)
    return f

def fp16(name, a):
    return a.astype(np.float16).astype(np.float32) if a.size > 1 else a

jobs = [("ulp_stem", "base", ulp(lambda n: "stem.0.reparam_conv" in n)), ("ulp_all", "base", ulp(lambda n: True)),
        ("fp16w_base", "base", fp16), ("fp16w_B", "B", fp16), ("fp16w_ALL_DROPIN", "ALL_DROPIN", fp16)]
res = {}
if os.environ.get("JOBS"):
    jobs = [j for j in jobs if j[0] in os.environ["JOBS"].split(",")]
for label, src, fn in jobs:
    p = os.path.join(vdir, f"_ctl_{label}.onnx")
    onnx.save(transform(os.path.join(vdir, src + ".onnx"), fn), p)
    print(label, flush=True)
    res[label] = {k: V.run_variant(p, ref, st, len(ref["outputs"]), k) for k, ref, st in data}
    os.remove(p)
    json.dump(res, open(jout, "w"), indent=1)
