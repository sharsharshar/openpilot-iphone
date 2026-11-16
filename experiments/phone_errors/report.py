"""Print markdown tables from $RF_SCRATCH/phone_errors/results/*.json. Usage: python -I report.py > tables.md"""
import os, sys, json, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pe_common import RES, OUT, DEV, quantities, vs_device, V_EGO
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(RES + "/*.json")}
g = lambda x: "%.2g" % x
def mw(v): return "%s / %s" % (g(v[0]), g(v[1]))
HEAD = "| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |\n|---|---|---|---|---|---|---|---|---|---|"
def row(n, label=None):
    if n not in R: return "| %s | (not run) |||||||||" % (label or n)
    v = R[n]["vs_base"]
    return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (label or n, mw(v["curv"]), mw(v["latacc"]), mw(v["acc"]), mw(v["y2"]), mw(v["y10"]), mw(v["x10"]), mw(v["lane"]), mw(v["lead"]), g(v["bias_curv"]), g(v["bias_y10"]))
DH = "| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |\n|---|---|---|---|---|---|---|"
ob = np.load(f"{OUT}/baseline_outputs.npy"); dbase = vs_device(quantities(ob))
def drow(n, label=None):
    if n == "baseline": d = dbase
    elif n in R: d = R[n]["vs_device"]
    else: return "| %s | (not run) ||||||" % (label or n)
    return "| %s | %s | %s | %s | %s | %s | %s |" % (label or n, g(d["curv"]), g(d["acc"]), g(d["y10"]), g(d["x10"]), g(d["lane"]), g(d["lead"]))
def table(title, names, labels=None, dev=True):
    print("\n### %s (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)\n" % title); print(HEAD)
    for n in names: print(row(n, (labels or {}).get(n)))
    if dev:
        print("\nSame cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)\n"); print(DH)
        print(drow("baseline", "unperturbed (yardstick)"))
        for n in names: print(drow(n, (labels or {}).get(n)))

def sgn(s): return "+" if s > 0 else "-"
cal = ["cal_zero"] + ["cal_%s_%s%g" % (ax, s, m) for ax in ("pitch", "yaw", "roll") for m in (0.5, 1, 2, 5) for s in "+-"]
table("1. Calibration error (offsets added to the logged rpy, both cameras; rpy = [roll, pitch, yaw])", cal)
foc = ["foc_%s_%s%d" % (w, s, p) for w in ("both", "narrow", "wide") for p in (2, 5, 10) for s in "+-"]
table("2. Focal length error (assumed = true x (1 +/- pct))", foc)
table("3. One physical camera feeding both inputs", ["one_wide", "one_narrow", "ios_narrowsrc", "ios_widesrc", "ios_narrowsrc_truef"])
table("4. Timing", ["drop5_repeat", "drop2_repeat", "drop5_skip", "drop2_skip", "lag_wide1", "lag_wide2"])
table("5. Image differences", ["img_down2", "img_down4", "img_gain0.7", "img_gain1.3", "img_blur3"])
print("\n### Experiment descriptions\n")
for n in sorted(R): print("- `%s`: %s" % (n, R[n]["desc"]))
if os.path.exists(f"{OUT}/oob.json"):
    print("\n### Fraction of luma samples whose source coordinate is outside the source frame (clamped to the edge pixel by the C code)\n\n| case | outside | by x | by y |\n|---|---|---|---|")
    for k, v in json.load(open(f"{OUT}/oob.json")).items(): print("| %s | %.3f | %.3f | %.3f |" % (k, v["outside_frac"], v["outside_x"], v["outside_y"]))
print("\nspeed used for lateral acceleration: v_ego warm mean %.1f m/s, range %.1f..%.1f" % (V_EGO[100:].mean(), V_EGO[100:].min(), V_EGO[100:].max()))
