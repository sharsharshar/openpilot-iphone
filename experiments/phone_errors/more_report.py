"""Markdown tables for the "More clips" study, from $PE/more/*.npz written by `run_all.py more`.
Usage: python -I more_report.py NAME=DIR [NAME=DIR ...]    (DIR = the phone_errors output directory of each clip, i.e. $RF_SCRATCH/phone_errors or $PE_OUT)
No imports from pe_common (that module reads one clip at import time). Warm steps = 100..599. 'turn' steps = warm steps where the
UNPERTURBED run's |desired curvature| > TURN_CURV (0.01 1/m); 'rest' = the other warm steps. Change = |perturbed - unperturbed| per step, then mean / worst."""
import os, sys, glob, json
import numpy as np
TURN_CURV = 0.01
KEYS = [("curv", "curvature 1/m"), ("acc", "desired accel m/s^2"), ("x10", "plan x @10s m"), ("y10", "plan y @10s m"), ("lane", "lane prob"), ("lead", "lead prob")]
g = lambda x: "%.2g" % x
def load(d):
    b = np.load(f"{d}/more/_base.npz"); base = {k[2:]: b[k] for k in b.files if k.startswith("q_")}
    dev = {k[4:]: b[k] for k in b.files if k.startswith("dev_")}
    cases = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in sorted(glob.glob(f"{d}/more/*.npz")) if not os.path.basename(p).startswith("_")}
    return dict(base=base, dev=dev, dev_ok=b["dev_ok"], has_action=bool(b["has_action"]), v=b["v_ego"], cases=cases)
def masks(c):
    n = len(c["v"]); warm = np.arange(n) >= 100; turn = warm & (np.abs(c["base"]["curv"]) > TURN_CURV)
    return {"all": warm, "turn": turn, "rest": warm & ~turn}
def change(c, name, mask):
    q, b = c["cases"][name], c["base"]; r = {}
    for k, _ in KEYS:
        d = np.abs(q[k][mask] - b[k][mask]); r[k] = (float(d.mean()), float(d.max())) if d.size else (np.nan, np.nan)
    return r
def devdist(c, q):
    ok = c["dev_ok"]; w = ok >= 100; o = ok[w]; r = {}
    for k in ("curv", "acc", "y10", "x10"):
        r[k] = float(np.abs(c["dev"][k][w] - q[k][o]).mean()) if (c["has_action"] or k in ("x10", "y10")) else None
    r["lane"] = float(np.abs(c["dev"]["lane"][w] - q["lane"][o][:, 1:3]).mean()); r["lead"] = float(np.abs(c["dev"]["lead"][w] - q["lead"][o]).mean()); return r
FLOOR = ["ctl_noise1", "ctl_pitch_+0.1", "ctl_yaw_+0.1"]
ORDER = FLOOR + ["cal_zero", "cal_yaw_+1", "cal_yaw_-1", "cal_yaw_+2", "cal_yaw_-2", "cal_pitch_+1", "cal_pitch_-1", "cal_pitch_+2", "cal_pitch_-2", "cal_roll_+2",
         "foc_both_+5", "foc_both_-5", "foc_narrow_+5", "foc_narrow_-5", "foc_wide_+5", "foc_wide_-5",
         "wide_lag2", "wide_frozen", "wide_grey", "one_narrow", "narrow_lag2", "narrow_frozen", "narrow_grey", "one_wide",
         "ios_narrowsrc", "ios_widesrc", "ios_narrowsrc_truef", "drop5_repeat", "drop2_skip", "img_down2"]
LABEL = {"one_narrow": "one_narrow (wide input from narrow-camera pixels; both inputs narrow cam)", "one_wide": "one_wide (narrow input from wide-camera pixels; both inputs wide cam)"}
def table(c, sub, title):
    m = masks(c)[sub]; print("\n#### %s (%d steps)\n" % (title, m.sum()))
    if m.sum() == 0: print("(no such steps)"); return
    print("| case | " + " | ".join(l for _, l in KEYS) + " |\n|---|" + "---|" * len(KEYS))
    for n in ORDER:
        if n not in c["cases"]: continue
        r = change(c, n, m); print("| %s | %s |" % (LABEL.get(n, n), " | ".join("%s / %s" % (g(r[k][0]), g(r[k][1])) for k, _ in KEYS)))
SHORT_DEV = {"unperturbed", "ctl_noise1", "cal_zero", "cal_yaw_+2", "cal_yaw_-2", "cal_pitch_+2", "cal_pitch_-2", "foc_both_+5", "wide_lag2", "wide_frozen", "wide_grey", "one_narrow",
             "narrow_grey", "one_wide", "ios_narrowsrc", "drop5_repeat", "drop2_skip", "img_down2"}
def devtable(c, short=False):
    print("\n#### Distance to the device's own modelV2 (mean abs difference, warm steps%s)\n" % ("" if c["has_action"] else "; this log has no modelV2.action, so curvature and acceleration are not comparable"))
    print("| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |\n|---|---|---|---|---|---|---|")
    rows = [("unperturbed", c["base"])] + [(n, c["cases"][n]) for n in ORDER if n in c["cases"]]
    for n, q in rows:
        if short and n not in SHORT_DEV: continue
        d = devdist(c, q); f = lambda x: "n/a" if x is None else g(x)
        print("| %s | %s | %s | %s | %s | %s | %s |" % (n, f(d["curv"]), f(d["acc"]), f(d["y10"]), f(d["x10"]), f(d["lane"]), f(d["lead"])))
def ratio_tables(C, sub, names, title):
    """mean change divided by the clip's own noise floor (ctl_noise1) on the same steps"""
    print("\n### %s\n\nCell = mean |change| of the case divided by mean |change| of `ctl_noise1` (+-1 grey level of random noise) in the same clip and on the same steps. Above 1 means larger than the clip's own noise floor.\n" % title)
    for k, label in KEYS:
        print("\n**%s** (floor, absolute: %s)\n\n| case | %s |\n|---|%s" % (label, ", ".join("%s %s" % (n, g(change(C[n], "ctl_noise1", masks(C[n])[sub])[k][0])) for n in names), " | ".join(names), "---|" * len(names)))
        for n in ORDER[1:]:
            print("| %s | %s |" % (n, " | ".join(("%.3g" % (change(C[t], n, masks(C[t])[sub])[k][0] / change(C[t], "ctl_noise1", masks(C[t])[sub])[k][0])) if n in C[t]["cases"] else "-" for t in names)))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; short = "--short-dev" in sys.argv
    C = {}
    for a in args:
        n, d = a.split("=", 1); C[n] = load(d)
    for n, c in C.items():
        w = masks(c); v = c["v"]
        print("\n### Clip %s\n\nunperturbed |curvature| warm: mean %.4f max %.4f 1/m; steps with |curvature| > %g: %d of 500 (v_ego there %s m/s); v_ego warm %.1f..%.1f mean %.1f" % (
            n, np.abs(c["base"]["curv"][w["all"]]).mean(), np.abs(c["base"]["curv"][w["all"]]).max(), TURN_CURV, w["turn"].sum(),
            "%.1f..%.1f" % (v[w["turn"]].min(), v[w["turn"]].max()) if w["turn"].any() else "-", v[w["all"]].min(), v[w["all"]].max(), v[w["all"]].mean()))
        table(c, "all", "%s: change vs unperturbed run, all warm steps (mean / worst)" % n)
        if w["turn"].sum() >= 20:
            table(c, "turn", "%s: steps with |reference curvature| > %g" % (n, TURN_CURV)); table(c, "rest", "%s: all other warm steps" % n)
        devtable(c, short)
    print("\n### Cross-clip summary: mean change over warm steps (worst in the per-clip tables)\n")
    names = list(C)
    ratio_tables(C, "all", names, "Cross-clip summary as multiples of each clip's noise floor, all warm steps")
    tn0 = [n for n, c in C.items() if masks(c)["turn"].sum() >= 20]
    if tn0: ratio_tables(C, "turn", tn0, "Cross-clip summary as multiples of the noise floor, turn steps only (|reference curvature| > %g)" % TURN_CURV)
    for k, label in KEYS:
        print("\n**%s**\n\n| case | %s |\n|---|%s" % (label, " | ".join(names), "---|" * len(names)))
        for n in ORDER:
            print("| %s | %s |" % (n, " | ".join(g(change(c, n, masks(c)["all"])[k][0]) if n in c["cases"] else "-" for c in C.values())))
    tn = [n for n, c in C.items() if masks(c)["turn"].sum() >= 20]
    if tn:
        print("\n### Cross-clip summary, turn steps only (|reference curvature| > %g), mean change\n" % TURN_CURV)
        for k, label in KEYS[:4]:
            print("\n**%s**\n\n| case | %s |\n|---|%s" % (label, " | ".join(tn), "---|" * len(tn)))
            for n in ORDER:
                print("| %s | %s |" % (n, " | ".join(g(change(C[t], n, masks(C[t])["turn"])[k][0]) if n in C[t]["cases"] else "-" for t in tn)))
