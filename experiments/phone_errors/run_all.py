"""Run the perturbation experiments. Usage: python -I run_all.py GROUP [GROUP...]  (groups: calib focal onecam timing image oob)
Each experiment writes $RF_SCRATCH/phone_errors/results/<name>.json (skipped if it already exists; delete to redo)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, numpy as np
from pe_common import *

BASE_OUT = np.load(f"{OUT}/baseline_outputs.npy"); QB = quantities(BASE_OUT)
DEG = np.pi / 180

def finish(name, desc, frames, run_mask=None, extra=None):
    if have(name): return
    t0 = time.time(); out = run_model(frames, run_mask); q = quantities(out)
    d = {"name": name, "desc": desc, "vs_base": vs_baseline(q, QB), "vs_device": vs_device(q), "finite": bool(np.isfinite(out).all())}
    if extra: d.update(extra)
    save_result(name, d); print("%-28s done in %.0fs  curv %.2e  acc %.3f  y10 %.2f" % (name, time.time() - t0, d["vs_base"]["curv"][0], d["vs_base"]["acc"][0], d["vs_base"]["y10"][0]), flush=True)

def calib():
    finish("cal_zero", "rpy = [0,0,0] (no calibration, as the iOS app does today)", baseline_inputs(rpy=np.zeros(3, np.float32)))
    for ax, i in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        for mag in (0.5, 1, 2, 5):
            for sg in (1, -1):
                r = RPY0.copy().astype(np.float64); r[i] += sg * mag * DEG
                finish("cal_%s_%s%g" % (ax, "+" if sg > 0 else "-", mag), "%s offset %+g deg added to the logged calibration" % (ax, sg * mag), baseline_inputs(rpy=r.astype(np.float32)))

def focal():
    for which in ("both", "narrow", "wide"):
        for pct in (2, 5, 10):
            for sg in (1, -1):
                s = 1 + sg * pct / 100
                fn = F_NARROW * (s if which in ("both", "narrow") else 1); fw = F_WIDE * (s if which in ("both", "wide") else 1)
                finish("foc_%s_%s%d" % (which, "+" if sg > 0 else "-", pct), "assumed focal length %s %+d%% (%s)" % (which, sg * pct, "true value x %.2f" % s), baseline_inputs(fn=fn, fw=fw))

def onecam():
    n0 = lambda cam, view, f, rpy: warp_view(cam, view, f, rpy)
    finish("one_wide", "both inputs from the WIDE camera frames, true wide focal and logged rpy", stack_inputs(warp_view(1, 0, F_WIDE, RPY0), warp_view(1, 1, F_WIDE, RPY0)))
    finish("one_narrow", "both inputs from the NARROW camera frames, true narrow focal and logged rpy", stack_inputs(warp_view(0, 0, F_NARROW, RPY0), warp_view(0, 1, F_NARROW, RPY0)))
    # iOS app today: one frame, K from horizontal FOV (fallback 63 deg), rpy = 0, matrices narrow (big=0) and wide (big=1) from the same K
    f63 = (W / 2) / np.tan(63.0 * DEG / 2); z = np.zeros(3, np.float32)
    finish("ios_narrowsrc", "iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the NARROW camera frames", stack_inputs(warp_view(0, 0, f63, z), warp_view(0, 1, f63, z)), extra={"f": float(f63)})
    finish("ios_widesrc", "iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the WIDE camera frames (true wide hfov is 116 deg, so K is wrong)", stack_inputs(warp_view(1, 0, f63, z), warp_view(1, 1, f63, z)), extra={"f": float(f63)})
    finish("ios_narrowsrc_truef", "iOS-app recipe but with the true narrow focal length (rpy 0, same narrow frame in both inputs)", stack_inputs(warp_view(0, 0, F_NARROW, z), warp_view(0, 1, F_NARROW, z)))

def timing():
    base = baseline_inputs(); idx = np.arange(NF)
    for k in (5, 2):
        drop = (idx % k == k - 1)
        fr = base.copy()
        for t in idx[drop]: fr[t] = fr[t - 1]
        finish("drop%d_repeat" % k, "every %dth frame dropped, previous model input repeated (model runs every step)" % k, fr, extra={"dropped": int(drop.sum())})
        finish("drop%d_skip" % k, "every %dth frame dropped, model not run at that step, previous outputs held" % k, base, run_mask=~drop, extra={"dropped": int(drop.sum())})
    for lag in (1, 2):
        fr = base.copy(); fr[:, 1] = base[np.maximum(idx - lag, 0), 1]
        finish("lag_wide%d" % lag, "wide camera input lags the narrow by %d frame(s)" % lag, fr)

def image():
    finish("img_down2", "source frames box-downscaled to 1/2 (672x380) before the warp (focal scaled)", stack_inputs(warp_view(0, 0, F_NARROW, RPY0, lambda f: downscale_nv12(f, 2), "d2"), warp_view(1, 1, F_WIDE, RPY0, lambda f: downscale_nv12(f, 2), "d2")))
    finish("img_down4", "source frames box-downscaled to 1/4 (336x190) before the warp (focal scaled)", stack_inputs(warp_view(0, 0, F_NARROW, RPY0, lambda f: downscale_nv12(f, 4), "d4"), warp_view(1, 1, F_WIDE, RPY0, lambda f: downscale_nv12(f, 4), "d4")))
    for g in (0.7, 1.3):
        xf = lambda f, g=g: (gain_y(f, g), W, H)
        finish("img_gain%g" % g, "luma multiplied by %g (clipped to 0..255), chroma unchanged" % g, stack_inputs(warp_view(0, 0, F_NARROW, RPY0, xf, "g%g" % g), warp_view(1, 1, F_WIDE, RPY0, xf, "g%g" % g)))
    xf = lambda f: (box_blur_y(f, 1), W, H)
    finish("img_blur3", "3x3 box blur of the source luma, chroma unchanged", stack_inputs(warp_view(0, 0, F_NARROW, RPY0, xf, "b3"), warp_view(1, 1, F_WIDE, RPY0, xf, "b3")))

def oob():
    """Fraction of luma samples whose source coordinate falls outside the frame (the C code then clamps to the edge pixel)."""
    res = {}
    f63 = (W / 2) / np.tan(63.0 * DEG / 2)
    cases = {"baseline_narrow_view": (F_NARROW, RPY0, 0), "baseline_wide_view": (F_WIDE, RPY0, 1)}
    for nm, (cam_f, view) in {"narrowcam_in_wide_view": (F_NARROW, 1), "widecam_in_narrow_view": (F_WIDE, 0)}.items(): cases[nm] = (cam_f, RPY0, view)
    cases["ios_f63_rpy0_narrow_view"] = (f63, np.zeros(3, np.float32), 0); cases["ios_f63_rpy0_wide_view"] = (f63, np.zeros(3, np.float32), 1)
    for hf in (40, 50, 63, 78, 90):     # what a camera of other horizontal FOV (1344x760 pixels, rpy 0) would need in the wide model view
        cases["hfov%d_rpy0_wide_view" % hf] = ((W / 2) / np.tan(np.radians(hf) / 2), np.zeros(3, np.float32), 1)
    for nm, (f, rpy, view) in cases.items():
        m = c_matrix(K_of(f), rpy, view).astype(np.float32)
        c, r = np.meshgrid(np.arange(512, dtype=np.float32), np.arange(256, dtype=np.float32))
        sw = m[6] * c + m[7] * r + m[8]; sx = np.rint((m[0] * c + m[1] * r + m[2]) / sw); sy = np.rint((m[3] * c + m[4] * r + m[5]) / sw)
        out = (sx < 0) | (sx > W - 1) | (sy < 0) | (sy > H - 1)
        res[nm] = {"outside_frac": float(out.mean()), "outside_x": float(((sx < 0) | (sx > W - 1)).mean()), "outside_y": float(((sy < 0) | (sy > H - 1)).mean())}
        print("%-28s fraction of luma samples outside the source frame: %.3f (x %.3f, y %.3f)" % (nm, *res[nm].values()))
    json.dump(res, open(f"{OUT}/oob.json", "w"), indent=1)

if __name__ == "__main__":
    for g in sys.argv[1:]: globals()[g]()
