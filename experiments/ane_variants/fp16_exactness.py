"""Static check: how many weight elements of each variant are not exactly representable as float16.
The shipped model's weights are float16 numbers stored as float32 (scripts/make_fp32.py), and the Core ML package stores float16 weights,
so a variant whose weights are all float16 numbers loses nothing when converted. A rewrite that folds a constant into weights
(variant B's normalisation, E_fold's layer scales) makes new float32 numbers that float16 storage rounds.
Usage: python -I fp16_exactness.py <variants_dir>  -> prints a table, writes fp16_exactness.json"""
import json, os, sys
import numpy as np, onnx
from onnx import numpy_helper
vd = sys.argv[1]
base = {i.name: numpy_helper.to_array(i) for i in onnx.load(os.path.join(vd, "base.onnx")).graph.initializer}
res = {}
print(f"{'variant':16s} {'float32 elements':>17s} {'not fp16-exact':>15s} {'fraction':>9s}  worst relative rounding error")
for f in sorted(os.listdir(vd)):
    if not f.endswith(".onnx") or f.startswith("_"):
        continue
    n = bad = 0; worst = 0.0
    for i in onnx.load(os.path.join(vd, f)).graph.initializer:
        a = numpy_helper.to_array(i)
        if a.dtype != np.float32 or a.size <= 1:
            continue
        r = a.astype(np.float16).astype(np.float32)
        m = r != a
        n += a.size; bad += int(m.sum())
        if m.any():
            worst = max(worst, float((np.abs(r - a)[m] / np.abs(a)[m]).max()))
    res[f[:-5]] = {"elements": n, "not_fp16_exact": bad, "worst_rel_err": worst}
    print(f"{f[:-5]:16s} {n:17d} {bad:15d} {100 * bad / n:8.4f}%  {worst:.2e}")
json.dump(res, open(os.path.join(vd, "fp16_exactness.json"), "w"), indent=1)
