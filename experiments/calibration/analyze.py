"""Reduce the replays of one clip (<clip>/calibration/results/*.npz) to <clip>/calibration/results/summary.json.
Usage: CAL_CLIP_DIR=<clip dir> python -I analyze.py
Conventions: errors are (estimate - logged calibration of the device) in degrees; 'used' = the angles in the warp (what modeld would have used),
time in seconds of replayed clip time (20 steps per second); 'gated seconds' = gated samples / 20 = seconds of valid driving as calibrationd counts them
(speed and yaw-rate conditions met; before 5 valid blocks nothing else is required).
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cal_common as cc
from cal_common import DEG, NF

R = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{cc.RES}/*.npz") if not p.endswith(".tmp.npz")}
def load(tag): return dict(np.load(R[tag])) if tag in R else None
LOG_RPY = cc.RPY_LOG
Q = ("curv", "acc", "x2", "y2", "x10", "y10", "lane", "lead")
def qdiff(a, b, sa, sb):
    """mean / worst |a-b| of the driving quantities over steps sa (in a) and sb (in b); lane = mean over the 4 lines per step; signed means for curv, y10."""
    d = {}
    for k in Q:
        x = a["q_" + k][sa] - b["q_" + k][sb]
        ax = np.abs(x) if x.ndim == 1 else np.abs(x).mean(1)
        d[k] = [float(ax.mean()), float(ax.max())]
    d["curv_signed_mean"] = float((a["q_curv"][sa] - b["q_curv"][sb]).mean()); d["y10_signed_mean"] = float((a["q_y10"][sa] - b["q_y10"][sb]).mean())
    v = cc.V_EGO[np.arange(NF)[slice(100, None)]]
    d["latacc"] = [float((v ** 2 * np.abs(a["q_curv"][sa] - b["q_curv"][sb])).mean()), float((v ** 2 * np.abs(a["q_curv"][sa] - b["q_curv"][sb])).max())]
    return d

def trace_metrics(r):
    n = len(r["rpy_used"]); loops = n // NF
    used_err = (r["rpy_used"] - np.tile(LOG_RPY, (loops, 1))) / DEG
    cal_err = (r["cal_rpy"] - np.tile(LOG_RPY, (loops, 1))) / DEG
    e = np.maximum(np.abs(used_err[:, 1]), np.abs(used_err[:, 2]))       # worst of pitch and yaw error
    gated_cum = np.cumsum(r["gated"].astype(int))
    m = {"steps": int(n), "loops": int(loops), "final_used_err_deg_roll_pitch_yaw": used_err[-1].tolist(), "final_cal_err_deg_roll_pitch_yaw": cal_err[-1].tolist(),
         "final_status": int(r["status"][-1]), "final_valid_blocks": int(r["valid_blocks"][-1]), "final_cal_perc": int(r["cal_perc"][-1]), "gated_total": int(gated_cum[-1]),
         "gated_per_pass": [int(r["gated"][p * NF:(p + 1) * NF].sum()) for p in range(loops)], "seeded_at_step": (int(np.argmax(r["seeded"])) if r["seeded"].any() else None),
         "start_err_deg_pitch_yaw": used_err[0, 1:].tolist()}
    tw = {}
    for thr in (0.25, 0.5, 1.0):
        ok = e < thr
        if not ok[-1]: tw[str(thr)] = None; continue
        bad = np.where(~ok)[0]; k = 0 if len(bad) == 0 else int(bad[-1]) + 1      # first step after which the error stays below thr until the end
        tw[str(thr)] = {"t_s": k / 20, "gated_s": float(gated_cum[k - 1] / 20) if k > 0 else 0.0}
    m["stays_within"] = tw
    cal_steps = np.where(r["status"] == cc.CAL)[0]
    m["first_calibrated"] = None if len(cal_steps) == 0 else {"t_s": float(cal_steps[0] / 20), "gated_s": float(gated_cum[cal_steps[0]] / 20)}
    m["first_valid_block_step"] = int(np.argmax(r["valid_blocks"] > 0)) if (r["valid_blocks"] > 0).any() else None
    ck = {}
    for t in (5, 10, 15, 20, 25, 30, 40, 60, 90, 120, 150, 180, 240):
        k = int(t * 20) - 1
        if k < n: ck[str(t)] = used_err[k, 1:].tolist()
    m["used_err_at_t_s_pitch_yaw"] = ck
    # error of the angles in use, averaged over the last pass (200 steps -> 10 s windows)
    last = slice(n - NF + 100, n)
    m["last_pass_100_599_used_err_abs_mean_pitch_yaw"] = np.abs(used_err[last, 1:]).mean(0).tolist()
    m["last_pass_100_599_used_err_max_abs_pitch_yaw"] = np.abs(used_err[last, 1:]).max(0).tolist()
    m["last_pass_end_gated_blocks"] = int(r["valid_blocks"][-1])
    # fraction of steps with status calibrated
    m["frac_steps_calibrated"] = float((r["status"] == cc.CAL).mean())
    # series for plots: used error every 5 steps
    m["_series_used_err_pitch_yaw"] = used_err[::5, 1:].round(4).tolist()
    return m

ref = load("open_logged")
S = {"clip": os.path.basename(cc.CLIP.rstrip("/")), "loops": int(os.environ.get("CAL_LOOPS", 4)), "device": cc.LOG["dev"], "init_device": cc.LOG.get("init"), "sensor": cc.LOG["sensor"],
     "logged_rpy_deg": (LOG_RPY[0] / DEG).tolist(), "runs": {}, "driving": {}, "sign": {}}
if ref is not None:
    stored = np.load(f"{cc.CLIP}/real_reference_fp32.npz")["outputs"]
    S["reference_check_max_abs_diff_all_2576_outputs_600_steps"] = float(np.abs(ref["outputs"] - stored).max())
warm = slice(100, NF)
# --- sign / gain from the fixed-angle runs
for tag in sorted(R):
    if not tag.startswith("open_") or tag in ("open_logged", "open_zero"): continue
    r = load(tag); off = tag[5:]; ax = 1 if off.startswith("pitch") else 2
    g = r["gated"][warm]; tr = r["pose30"][warm, :3].astype(np.float64)
    obs = np.stack([-np.arctan2(tr[:, 2], tr[:, 0]), np.arctan2(tr[:, 1], tr[:, 0])], 1)[g] / DEG          # observed pitch, yaw (direction of travel in the warped frame)
    newerr = (r["new_rpy"][warm][g][:, 1:] - LOG_RPY[warm][g][:, 1:]) / DEG
    delta = (r["rpy_used"][0, ax] - LOG_RPY[0, ax]) / DEG
    S["sign"][off] = {"offset_deg": float(delta), "n": int(g.sum()), "obs_pitch_yaw_deg_median": np.median(obs, 0).tolist(), "obs_pitch_yaw_deg_p10_p90": [np.percentile(obs, 10, 0).tolist(), np.percentile(obs, 90, 0).tolist()],
                      "new_minus_logged_pitch_yaw_deg_median": np.median(newerr, 0).tolist(), "fraction_of_offset_removed": float(1 - np.median(newerr[:, ax - 1]) / delta)}
if ref is not None:
    # same for the reference (offset 0) and zero: what the observation says when the warp is the logged / zero angles
    for tag in ("open_logged", "open_zero"):
        r = load(tag) if tag != "open_logged" else ref
        if r is None: continue
        g = r["gated"][warm]; tr = r["pose30"][warm, :3].astype(np.float64)
        obs = np.stack([-np.arctan2(tr[:, 2], tr[:, 0]), np.arctan2(tr[:, 1], tr[:, 0])], 1)[g] / DEG
        newerr = (r["new_rpy"][warm][g][:, 1:] - LOG_RPY[warm][g][:, 1:]) / DEG
        S["sign"][tag[5:]] = {"n": int(g.sum()), "obs_pitch_yaw_deg_median": np.median(obs, 0).tolist(), "new_minus_logged_pitch_yaw_deg_median": np.median(newerr, 0).tolist(),
                              "start_err_deg_pitch_yaw": ((r["rpy_used"][0, 1:] - LOG_RPY[0, 1:]) / DEG).tolist()}
# --- closed-loop traces
for tag in sorted(R):
    if tag.split("_")[0] in ("cl1", "cls", "clL"): S["runs"][tag] = trace_metrics(load(tag))
# --- driving outputs
def add(name, a, b, sa, sb): S["driving"][name] = qdiff(a, b, sa, sb)
if ref is not None:
    for tag in ("open_zero", "noise1", "cl1_zero", "cls_zero") + tuple(t for t in R if t.startswith("open_") and "+" in t or "-" in t and t.startswith("open_")):
        if tag in R: add(tag + " vs reference, steps 100-599", load(tag), ref, warm, warm)
    for tag in sorted(R):
        if tag.startswith("cl1_") or tag.startswith("cls_"): add(tag + " vs reference, steps 100-599", load(tag), ref, warm, warm)
    for tag in sorted(R):
        if tag.startswith("clL_") or tag in ("openL_logged", "openL_zero", "noiseL"):
            r = load(tag); n = len(r["q_curv"]); last = slice(n - NF + 100, n)
            add(tag + " last pass 100-599 vs reference", r, ref, last, warm)
            if tag != "openL_logged" and "openL_logged" in R:
                b = load("openL_logged"); add(tag + " last pass 100-599 vs openL_logged last pass", r, b, last, last)
# --- estimate quality in the same windows (single pass, 100-599)
S["gated_steps_logged_run_warm"] = int(ref["gated"][warm].sum()) if ref is not None else None
for k in list(S["runs"]):
    S["runs"][k]["_series"] = S["runs"][k].pop("_series_used_err_pitch_yaw")
json.dump(S, open(f"{cc.RES}/summary.json", "w"))
print("wrote", f"{cc.RES}/summary.json", "runs", len(S["runs"]), "driving", len(S["driving"]))
