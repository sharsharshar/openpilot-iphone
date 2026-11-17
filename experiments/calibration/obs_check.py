"""Experiment 1: what calibrationd's observation looks like on the stored float32 reference run (warp = the device's logged calibration).
Needs only log data and real_reference_fp32.npz (no video). Usage: CAL_CLIP_DIR=<clip dir> python -I obs_check.py   -> <clip>/calibration/results/obs_check.json
For every frame: the model's pose output -> observed direction of travel -> new_rpy as in Calibrator.handle_cam_odom (composition with the angles used in the warp,
which here are the logged ones), compared with the logged calibration; the same from the device's own cameraOdometry (device's model version).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cal_common as cc
from cal_common import DEG

out = np.load(f"{cc.CLIP}/real_reference_fp32.npz")["outputs"]
cal = cc.Calibrator(mici=cc.MICI)
n = cc.NF; warm = np.arange(n) >= 100
res = {"clip": os.path.basename(cc.CLIP.rstrip("/")), "device": cc.LOG["dev"], "init_device": cc.LOG.get("init"), "sensor": cc.LOG["sensor"], "camera_cfg": str(cc.CFG_KEY)}

def run(source):
    """source: function frame -> dict trans, rot, trans_std, rt_std or None"""
    err = np.full((n, 3), np.nan); gate = np.zeros(n, bool); certain = np.zeros(n, bool); sp = np.zeros(n, bool); yw = np.zeros(n, bool); ang_std = np.full(n, np.nan)
    for i in range(n):
        p = source(i)
        if p is None: continue
        cal.v_ego = cc.V_EGO[i]
        s, y, rc_, hc = cal.gates(p["trans"], p["rot"], p["trans_std"], p["rt_std"])
        sp[i], yw[i] = s, y; gate[i] = s and y; certain[i] = s and y and rc_ and hc
        obs, new = cc.observe(cc.RPY_LOG[i], p["trans"])
        err[i] = (cal.sanity_clip(new) - cc.RPY_LOG[i]) / DEG
        ang_std[i] = np.arctan2(p["trans_std"][1], p["trans"][0]) / DEG
    return err, gate, certain, sp, yw, ang_std

def summarize(err, gate, certain, sp, yw, ang_std, label):
    d = {"frames": int(np.isfinite(err[:, 0]).sum()), "speed_ok": int(sp.sum()), "yawrate_ok": int(yw.sum()), "gated_speed_and_yawrate": int(gate.sum()), "gated_warm": int((gate & warm).sum()),
         "also_std_certain": int(certain.sum()), "trans_angle_std_deg_median": float(np.nanmedian(ang_std)), "trans_angle_std_limit_deg": float(np.degrees(cal.MAX_VEL_ANGLE_STD))}
    for name, m in (("gated", gate & warm), ("gated_certain", certain & warm), ("all_warm", warm & np.isfinite(err[:, 0]))):
        if m.sum() < 2: d[name] = None; continue
        e = err[m]
        d[name] = {"n": int(m.sum()), "pitch_bias_deg": float(e[:, 1].mean()), "yaw_bias_deg": float(e[:, 2].mean()), "pitch_std_deg": float(e[:, 1].std()), "yaw_std_deg": float(e[:, 2].std()),
                   "pitch_rms_deg": float(np.sqrt((e[:, 1] ** 2).mean())), "yaw_rms_deg": float(np.sqrt((e[:, 2] ** 2).mean())), "pitch_p5_p95_deg": [float(np.percentile(e[:, 1], 5)), float(np.percentile(e[:, 1], 95))],
                   "yaw_p5_p95_deg": [float(np.percentile(e[:, 2], 5)), float(np.percentile(e[:, 2], 95))]}
    # block means exactly as calibrationd builds them: consecutive gated samples in blocks of 100 (here: warm gated samples)
    idx = np.where(gate & warm)[0]; blocks = []
    for b in range(len(idx) // 100):
        e = err[idx[b * 100:(b + 1) * 100]]; blocks.append([float(e[:, 1].mean()), float(e[:, 2].mean())])
    d["block100_means_pitch_yaw_deg"] = blocks
    return d

ours = lambda i: cc.split_pose(out[i])
res["ours"] = summarize(*run(ours), "ours")
if cc.ODO_IDX:
    tr, ro, ts, rts = cc.odo_array("trans"), cc.odo_array("rot"), cc.odo_array("trans_std"), cc.odo_array("rt_std")
    def dev(i):
        if not np.isfinite(tr[i, 0]): return None
        return dict(trans=tr[i], rot=ro[i], trans_std=ts[i], rt_std=rts[i] if np.isfinite(rts[i]).all() and len(rts[i]) == 3 else [])
    res["device_odometry"] = summarize(*run(dev), "device")
    # frame-by-frame agreement of the two observations (warm, both gated)
    e1 = run(ours)[0]; e2 = run(dev)[0]; m = warm & np.isfinite(e1[:, 0]) & np.isfinite(e2[:, 0])
    res["ours_vs_device_obs"] = {"n": int(m.sum()), "pitch_corr": float(np.corrcoef(e1[m, 1], e2[m, 1])[0, 1]), "yaw_corr": float(np.corrcoef(e1[m, 2], e2[m, 2])[0, 1]),
                                 "pitch_mean_diff_deg": float((e1[m, 1] - e2[m, 1]).mean()), "yaw_mean_diff_deg": float((e1[m, 2] - e2[m, 2]).mean())}
    res["device_odo_frames_matched"] = len(cc.ODO_IDX)
    res["device_speed_minus_vego_mean"] = float(np.nanmean(tr[warm, 0] - cc.V_EGO[warm]))
pose = np.array([cc.split_pose(r) ["trans"] for r in out])
res["model_speed_minus_vego_warm_mean"] = float((pose[warm, 0] - cc.V_EGO[warm]).mean())
# logged calibration itself (degrees), status and how it moves in the clip
L = cc.RPY_LOG / DEG
res["logged_rpy_deg_first"] = L[0].tolist(); res["logged_rpy_deg_range_in_clip"] = (L.max(0) - L.min(0)).tolist()
ext = cc.CALLOG["ext"] or cc.CALLOG["live"]
res["logged_status_first"] = ext[0]["status"]; res["logged_valid_blocks_first"] = ext[0]["valid_blocks"]; res["logged_calperc_first"] = ext[0]["perc"]; res["logged_spread_deg_first"] = (np.array(ext[0]["spread"]) / DEG).tolist()
# height and wide-from-device: model output vs the device's logged averages
rt = np.array([cc.split_pose(r)["rt"] for r in out]); wd = np.array([cc.split_pose(r)["wide"] for r in out])
res["height"] = {"model_road_transform_z_warm_mean": float(rt[warm, 2].mean()), "model_road_transform_z_warm_std": float(rt[warm, 2].std()), "logged_height_mean": None if np.isnan(cc.HEIGHT_LOG).all() else float(np.nanmean(cc.HEIGHT_LOG))}
res["wide_from_device_euler_deg"] = {"model_warm_mean": (wd[warm].mean(0) / DEG).tolist(), "logged_mean": None if np.isnan(cc.WIDE_LOG).all() else (np.nanmean(cc.WIDE_LOG, 0) / DEG).tolist(), "logged_rpy_calib_mean": (cc.RPY_LOG.mean(0) / DEG).tolist()}
json.dump(res, open(f"{cc.RES}/obs_check.json", "w"), indent=1)
def f(d, k): return d[k] if d and k in d else None
o = res["ours"]
print(res["clip"], res["device"], res["sensor"], "| frames", o["frames"], "gated(speed+yawrate)", o["gated_speed_and_yawrate"], "warm gated", o["gated_warm"], "also std-certain", o["also_std_certain"], "median trans std angle %.3f deg (limit %.2f)" % (o["trans_angle_std_deg_median"], o["trans_angle_std_limit_deg"]))
print(" logged rpy deg", np.round(res["logged_rpy_deg_first"], 3), "status", res["logged_status_first"], "validBlocks", res["logged_valid_blocks_first"])
for name in ("gated", "gated_certain", "all_warm"):
    d = o[name]; print("  ours %-13s" % name, None if d is None else "n %d pitch bias %+.3f std %.3f | yaw bias %+.3f std %.3f" % (d["n"], d["pitch_bias_deg"], d["pitch_std_deg"], d["yaw_bias_deg"], d["yaw_std_deg"]))
print("  ours block100 means (pitch,yaw deg):", np.round(o["block100_means_pitch_yaw_deg"], 3).tolist())
if "device_odometry" in res:
    o = res["device_odometry"]; d = o["gated"]
    print("  device gated warm", o["gated_warm"], None if d is None else "pitch bias %+.3f std %.3f | yaw bias %+.3f std %.3f" % (d["pitch_bias_deg"], d["pitch_std_deg"], d["yaw_bias_deg"], d["yaw_std_deg"]), "block means", np.round(o["block100_means_pitch_yaw_deg"], 3).tolist())
    print("  ours-vs-device", res["ours_vs_device_obs"])
print("  height", res["height"], " wide", res["wide_from_device_euler_deg"])
