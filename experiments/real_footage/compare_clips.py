"""Cross-clip comparison of the sensitivity results (markdown tables). Reads sens_{f16,gelu,quant}.json from each clip directory.
Usage: python -I compare_clips.py label=dir [label=dir ...]      (the first clip's dir is the original $RF_SCRATCH)
Numbers are the mean over warm steps (steps 100..599) of |variant - float32 reference|, per metric; "worst" tables give the maximum over steps.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
clips = [a.split("=", 1) for a in sys.argv[1:]]
data = {}
for lab, d in clips:
    for grp in ("f16", "gelu", "quant"):
        p = f"{d}/sens_{grp}.json"
        if os.path.exists(p):
            for v, r in json.load(open(p)).items(): data.setdefault(v, {})[lab] = r["real"]["warm"]
VARS = ["float16 model (original ONNX)", "exact erf form", "x * sigmoid(1.702 x)", "piecewise linear, 64 segments", "piecewise linear, 32 segments",
        "8-bit linear, per output channel", "8-bit linear, per tensor", "8-bit lookup table (256 values per tensor)", "6-bit lookup table (64 values)", "4-bit lookup table (16 values)"]
COLS = [("raw mean", None), ("curvature (1/m)", "curv"), ("accel (m/s^2)", "acc"), ("plan x @10s (m)", "x10"), ("plan y @10s (m)", "y10"), ("lane-line prob", "lane"), ("lead prob", "lead")]
def cell(r, key, which):
    if key is None: return "%.2g" % r["raw"][2] if which == 0 else "%.3g" % r["raw"][0]
    return "%.2g" % r["drv"][key][which]
for which, title in ((0, "mean over warm steps"), (1, "worst step (raw column: worst single output)")):
    print(f"\n#### {title}\n")
    for name, key in COLS:
        print(f"\n{name}\n\n| variant | " + " | ".join(l for l, _ in clips) + " |\n|---|" + "---|" * len(clips))
        for v in VARS:
            if v in data: print(f"| {v} | " + " | ".join(cell(data[v][l], key, which) if l in data[v] else "-" for l, _ in clips) + " |")
