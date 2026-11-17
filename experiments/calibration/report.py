"""Print the Markdown tables of RESULTS.md from the per-clip summaries.
Usage: python -I report.py name=<clip dir> name=<clip dir> ...      (reads <clip>/calibration/results/summary.json and obs_check.json; no cal_common import)
Optional: CAL_PLOT=<png path> writes one small PNG (angle in use minus logged, versus time, zero start, unmodified-with-loops and short-block variants).
"""
import sys, os, json
import numpy as np
clips = [a.split("=", 1) for a in sys.argv[1:]]
SUM = {n: json.load(open(f"{d}/calibration/results/summary.json")) for n, d in clips}
OBS = {n: json.load(open(f"{d}/calibration/results/obs_check.json")) for n, d in clips}
names = [n for n, _ in clips]
def f(x, p=2): return "-" if x is None else ("%." + str(p) + "f") % x
def g(x): return "-" if x is None else ("%.2g" % x)
def row(*c): print("| " + " | ".join(str(x) for x in c) + " |")
def hdr(*c): row(*c); row(*["---"] * len(c))
STATUS = {0: "uncalibrated", 1: "calibrated", 2: "invalid", 3: "recalibrating"}

def tw(m, thr):
    v = m["stays_within"][str(thr)]
    return "never" if v is None else "%s s (%s s valid)" % (f(v["t_s"], 1), f(v["gated_s"], 1))
def errs(m): return "%s / %s" % tuple(f(x, 2) for x in m["final_used_err_deg_roll_pitch_yaw"][1:])

print("### Summary across clips (start from zero; ratios are mean abs difference to the logged-calibration run divided by the clip's own +-1 grey level noise floor)")
hdr("clip", "unmodified, 30 s pass: error at 30 s (pitch / yaw deg), valid blocks, status", "loop variant: first 'calibrated' (valid driving)", "loop variant: settled error, mean over last pass (pitch / yaw deg)", "worst final error over the 8 wrong starts, loop variant (pitch / yaw deg)",
    "driving diff / noise floor after settling: curvature / accel / plan x 10 s / plan y 10 s / lane prob", "same for fixed zero calibration")
def ratios(D, a, b):
    if a not in D or b not in D: return "-"
    return " / ".join("%.1f" % (D[a][k][0] / max(D[b][k][0], 1e-12)) for k in ("curv", "acc", "x10", "y10", "lane"))
for n in names:
    S_ = SUM[n]; r1 = S_["runs"].get("cl1_zero"); rl = S_["runs"].get("clL_zero"); D = S_["driving"]
    worst = np.array([[abs(S_["runs"][f"clL_{s}"]["final_used_err_deg_roll_pitch_yaw"][1]), abs(S_["runs"][f"clL_{s}"]["final_used_err_deg_roll_pitch_yaw"][2])] for s in ["yaw+2", "yaw-2", "yaw+5", "yaw-5", "pitch+2", "pitch-2", "pitch+5", "pitch-5"] if f"clL_{s}" in S_["runs"]]).max(0)
    fc = rl["first_calibrated"]
    row(n, "%s, %d blocks, %s" % (errs(r1), r1["final_valid_blocks"], STATUS[r1["final_status"]]), "never" if fc is None else "%s s (%s s)" % (f(fc["t_s"], 0), f(fc["gated_s"], 0)),
        "%s / %s" % tuple(f(x, 2) for x in rl["last_pass_100_599_used_err_abs_mean_pitch_yaw"]), "%s / %s" % (f(worst[0], 2), f(worst[1], 2)),
        ratios(D, "clL_zero last pass 100-599 vs reference", "noiseL last pass 100-599 vs openL_logged last pass"), ratios(D, "open_zero vs reference, steps 100-599", "noise1 vs reference, steps 100-599"))
print()
print("### Clips and gating (logged-calibration reference run, one 600-step pass)")
hdr("clip", "device / sensor", "logged calibration at frame 0: roll / pitch / yaw (deg)", "frames with speed + yaw-rate gate met (of 600)", "of those in steps 100-599", "also trans-angle std < 0.25 deg", "median trans-angle std (deg)", "loops used in the loop variant")
for n in names:
    s, o = SUM[n], OBS[n]; ob = o["ours"]
    row(n, f"{s['device']}/{o['init_device']} / {s['sensor']}", " / ".join(f(v, 3) for v in s["logged_rpy_deg"]), ob["gated_speed_and_yawrate"], ob["gated_warm"], ob["also_std_certain"], f(ob["trans_angle_std_deg_median"], 3), s["loops"])

print("\n### Experiment 1: one observation versus the logged calibration (reference run, steps 100-599, gated frames; degrees; observation minus logged)")
hdr("clip", "n gated", "pitch bias", "pitch std", "yaw bias", "yaw std", "block-of-100 means (pitch, yaw)", "device odometry: n / pitch bias / std / yaw bias / std", "corr ours vs device (pitch / yaw)")
for n in names:
    o = OBS[n]; a = o["ours"]["gated"]; d = o.get("device_odometry", {}).get("gated"); c = o.get("ours_vs_device_obs", {})
    bm = "; ".join("(%s, %s)" % (f(x[0], 2), f(x[1], 2)) for x in o["ours"]["block100_means_pitch_yaw_deg"]) or "none (fewer than 100 gated samples)"
    row(n, a["n"], f(a["pitch_bias_deg"], 3), f(a["pitch_std_deg"], 3), f(a["yaw_bias_deg"], 3), f(a["yaw_std_deg"], 3), bm,
        "-" if d is None else "%d / %s / %s / %s / %s" % (d["n"], f(d["pitch_bias_deg"], 3), f(d["pitch_std_deg"], 3), f(d["yaw_bias_deg"], 3), f(d["yaw_std_deg"], 3)),
        "-" if not c else "%s / %s" % (f(c["pitch_corr"], 2), f(c["yaw_corr"], 2)))
print("\nAll frames (no gating) steps 100-599, yaw std: " + "; ".join("%s %s deg" % (n, f(OBS[n]["ours"]["all_warm"]["yaw_std_deg"], 2)) for n in names))
print("\nModel speed (pose trans x) minus carState.vEgo, steps 100-599 mean (m/s): " + "; ".join("%s %s" % (n, f(OBS[n]["model_speed_minus_vego_warm_mean"], 2)) for n in names))
print("\nHeight and wide-camera outputs (steps 100-599 means): ")
hdr("clip", "model road_transform z (m)", "logged height (m)", "model wide_from_device_euler pitch / yaw (deg)", "logged wideFromDeviceEuler pitch / yaw (deg)", "logged rpyCalib pitch / yaw (deg)")
for n in names:
    h = OBS[n]["height"]; w = OBS[n]["wide_from_device_euler_deg"]
    row(n, f(h["model_road_transform_z_warm_mean"], 3), f(h["logged_height_mean"], 3), "%s / %s" % (f(w["model_warm_mean"][1], 2), f(w["model_warm_mean"][2], 2)),
        "-" if w["logged_mean"] is None else "%s / %s" % (f(w["logged_mean"][1], 2), f(w["logged_mean"][2], 2)), "%s / %s" % (f(w["logged_rpy_calib_mean"][1], 2), f(w["logged_rpy_calib_mean"][2], 2)))

print("\n### Frame / sign check: fixed warp angles = logged + offset, no feedback (steps 100-599, gated frames, medians; degrees)")
hdr("clip", "offset in warp", "observed pitch / yaw of travel", "observation corrected angle minus logged (pitch / yaw)", "fraction of the offset removed by one observation")
for n in names:
    for k, v in SUM[n]["sign"].items():
        if "offset_deg" not in v: continue
        row(n, f"{k} ({f(v['offset_deg'], 2)})", "%s / %s" % tuple(f(x, 2) for x in v["obs_pitch_yaw_deg_median"]), "%s / %s" % tuple(f(x, 2) for x in v["new_minus_logged_pitch_yaw_deg_median"]), f(v["fraction_of_offset_removed"], 3))
print("\nFixed zero warp (the iOS app today), same quantities:")
hdr("clip", "start error pitch / yaw (zero minus logged)", "observed pitch / yaw", "corrected minus logged")
for n in names:
    v = SUM[n]["sign"].get("zero")
    if v: row(n, "%s / %s" % tuple(f(x, 2) for x in v["start_err_deg_pitch_yaw"]), "%s / %s" % tuple(f(x, 2) for x in v["obs_pitch_yaw_deg_median"]), "%s / %s" % tuple(f(x, 2) for x in v["new_minus_logged_pitch_yaw_deg_median"]))

print("\n### Experiment 2a: calibrationd's unmodified rules, one 30 s pass, start from zero (angle in use minus logged, pitch / yaw deg)")
hdr("clip", "start error", "at 5 s", "10 s", "15 s", "20 s", "30 s", "gated samples in the pass", "valid blocks at end", "status at end (calPerc)", "seeded at step")
for n in names:
    m = SUM[n]["runs"].get("cl1_zero")
    if not m: continue
    ck = m["used_err_at_t_s_pitch_yaw"]
    row(n, "%s / %s" % tuple(f(x, 2) for x in m["start_err_deg_pitch_yaw"]), *["%s / %s" % tuple(f(x, 2) for x in ck[t]) for t in ("5", "10", "15", "20", "30")], m["gated_total"], m["final_valid_blocks"], "%s (%d %%)" % (STATUS[m["final_status"]], m["final_cal_perc"]), m["seeded_at_step"])

print("\n### Experiment 2b: VARIANT block size 20 (instead of 100), one 30 s pass, start from zero")
hdr("clip", "gated samples", "valid blocks at end", "status at end", "stays within 1.0 deg", "stays within 0.5 deg", "stays within 0.25 deg", "error at end, pitch / yaw (deg)", "first 'calibrated'")
for n in names:
    m = SUM[n]["runs"].get("cls_zero")
    if not m: continue
    fc = m["first_calibrated"]
    row(n, m["gated_total"], m["final_valid_blocks"], STATUS[m["final_status"]], tw(m, 1.0), tw(m, 0.5), tw(m, 0.25), errs(m), "never" if fc is None else "%s s (%s s valid)" % (f(fc["t_s"], 1), f(fc["gated_s"], 1)))

print("\n### Experiment 2c: VARIANT clip replayed in a loop (unmodified constants), start from zero")
hdr("clip", "passes", "gated samples per pass", "first 'calibrated' (replayed time, valid driving)", "stays within 1.0 deg", "stays within 0.5 deg", "stays within 0.25 deg", "error at end, pitch / yaw (deg)", "last pass: mean |error| pitch / yaw (deg)", "last pass: max |error| pitch / yaw")
for n in names:
    m = SUM[n]["runs"].get("clL_zero")
    if not m: continue
    fc = m["first_calibrated"]
    row(n, m["loops"], "/".join(str(x) for x in m["gated_per_pass"][:2]) + "..", "never" if fc is None else "%s s (%s s)" % (f(fc["t_s"], 0), f(fc["gated_s"], 0)), tw(m, 1.0), tw(m, 0.5), tw(m, 0.25), errs(m),
        "%s / %s" % tuple(f(x, 2) for x in m["last_pass_100_599_used_err_abs_mean_pitch_yaw"]), "%s / %s" % tuple(f(x, 2) for x in m["last_pass_100_599_used_err_max_abs_pitch_yaw"]))
print("\nAngle in use minus logged (pitch / yaw, deg) at checkpoints, loop variant from zero:")
hdr("clip", "30 s", "60 s", "90 s", "120 s", "150 s", "180 s", "240 s")
for n in names:
    m = SUM[n]["runs"].get("clL_zero")
    if not m: continue
    ck = m["used_err_at_t_s_pitch_yaw"]; row(n, *[("%s / %s" % tuple(f(x, 2) for x in ck[t])) if t in ck else "-" for t in ("30", "60", "90", "120", "150", "180", "240")])

starts = ["zero", "yaw+2", "yaw-2", "yaw+5", "yaw-5", "pitch+2", "pitch-2", "pitch+5", "pitch-5"]
for tagp, title in (("cls", "VARIANT block size 20, one 30 s pass"), ("clL", "VARIANT loop, unmodified constants"), ("cl1", "unmodified, one 30 s pass")):
    print(f"\n### Experiment 3: starts off by 2 and 5 degrees ({title}): start error -> error at end of run (pitch / yaw, deg); time the error stays below 0.5 deg")
    hdr("clip", *starts)
    for n in names:
        cells = []
        for s in starts:
            m = SUM[n]["runs"].get(f"{tagp}_{s}")
            if not m: cells.append("-"); continue
            v = m["stays_within"]["0.5"]
            cells.append("%s/%s -> %s ; %s" % (f(m["start_err_deg_pitch_yaw"][0], 1), f(m["start_err_deg_pitch_yaw"][1], 1), errs(m), "never" if v is None else f(v["t_s"], 0) + " s"))
        row(n, *cells)

print("\n### Experiment 4: driving outputs versus the reference run with the logged calibration (mean abs difference over steps 100-599 of the pass; in brackets the worst step)")
cols = [("curv", "curvature (1/m)"), ("acc", "accel (m/s^2)"), ("x10", "plan x 10 s (m)"), ("y10", "plan y 10 s (m)"), ("lane", "lane-line prob")]
def drow(name, d): return [name] + ["%s (%s)" % (g(d[k][0]), g(d[k][1])) for k, _ in cols] + [g(d["latacc"][0]), g(d["curv_signed_mean"])]
for n in names:
    print(f"\n**{n}**\n")
    hdr("case", *[c for _, c in cols], "lateral accel change (m/s^2, mean)", "curvature signed mean (1/m)")
    D = SUM[n]["driving"]
    for key, label in (("noise1 vs reference, steps 100-599", "noise floor: +-1 grey level random input noise, 1 pass"),
                       ("noiseL last pass 100-599 vs openL_logged last pass", "noise floor in the loop setting (vs loop run with logged calibration)"),
                       ("openL_logged last pass 100-599 vs reference", "check: logged calibration, loop last pass vs reference"),
                       ("open_zero vs reference, steps 100-599", "zero calibration, fixed (iOS app today)"),
                       ("cl1_zero vs reference, steps 100-599", "closed loop from zero, unmodified, 1 pass"),
                       ("cls_zero vs reference, steps 100-599", "closed loop from zero, block size 20, 1 pass"),
                       ("clL_zero last pass 100-599 vs reference", "closed loop from zero, loop variant, last pass"),
                       ):
        if key in D: row(*drow(label, D[key]))
    for s in starts[1:]:
        key = f"clL_{s} last pass 100-599 vs reference"
        if key in D: row(*drow(f"loop variant, start {s}, last pass", D[key]))
    for s in starts[1:]:
        key = f"open_{s} vs reference, steps 100-599"
        if key in D: row(*drow(f"fixed offset {s}, no recovery", D[key]))

if os.environ.get("CAL_PLOT"):
    from PIL import Image, ImageDraw
    Wd, Hd = 1100, 150 * len(names) + 50; im = Image.new("RGB", (Wd, Hd), "white"); dr = ImageDraw.Draw(im)
    dr.text((10, 5), "angle in use minus logged calibration (deg), start from zero; blue = pitch, red = yaw; thick = loop variant (unmodified rules), thin = block size 20, 1 pass", fill="black")
    dr.text((10, 20), "grey band = +-0.5 deg; clipped at +-3.2 deg; x axis: replayed time, scale differs per clip (total in the right corner)", fill="black")
    for r, n in enumerate(names):
        y0 = 50 + 150 * r; x0 = 70; x1 = Wd - 20; ymid = y0 + 65; sc = 18.0
        dr.rectangle([x0, ymid - 0.5 * sc, x1, ymid + 0.5 * sc], fill=(232, 232, 232)); dr.line([x0, ymid, x1, ymid], fill=(140, 140, 140)); dr.text((5, y0 + 55), n, fill="black")
        runs = SUM[n]["runs"]; tl = runs["clL_zero"]["steps"] / 20 if "clL_zero" in runs else 30.0
        dr.text((x1 - 60, y0 + 2), "%d s" % tl, fill="black")
        for tag, w in (("cls_zero", 1), ("clL_zero", 2)):
            if tag not in runs: continue
            ser = np.array(runs[tag]["_series"])
            for c, col in ((0, (30, 90, 200)), (1, (200, 40, 40))):
                pts = [(x0 + (x1 - x0) * (i * 5 / 20) / tl, ymid - float(np.clip(ser[i, c], -3.2, 3.2)) * sc) for i in range(len(ser))]
                dr.line(pts, fill=col, width=w)
    im.save(os.environ["CAL_PLOT"], optimize=True)
    print("\nplot:", os.environ["CAL_PLOT"], os.path.getsize(os.environ["CAL_PLOT"]), "bytes")
