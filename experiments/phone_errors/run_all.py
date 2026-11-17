"""Run the perturbation experiments. Usage: python -I run_all.py GROUP [GROUP...]
Groups of the first study (original clip, 62 cases): calib focal onecam timing image control oob; each experiment writes
$RF_SCRATCH/phone_errors/results/<name>.json (skipped if it already exists; delete to redo).
Group `more` (any clip; the reduced set of the "More clips" study, see RESULTS.md): writes $RF_SCRATCH/phone_errors/more/<name>.json
and <name>.npz (the per-step driving quantities of the perturbed run, from which more_report.py computes everything, including the
turn / non-turn split). Frame size and intrinsics come from the clip's log (pe_common). `more` accepts case-name prefixes as further arguments,
e.g. `run_all.py more cal_ ios_` runs only the cases whose names start with those."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, numpy as np
from pe_common import *

BASE_OUT = np.load(f"{OUT}/baseline_outputs.npy"); QB = quantities(BASE_OUT)
DEG = np.pi / 180

RESDIR = None      # None: results/ (first study); the `more` group sets it to more/
def finish(name, desc, frames, run_mask=None, extra=None):
    if have(name, RESDIR): return
    t0 = time.time(); out = run_model(frames, run_mask); q = quantities(out)
    d = {"name": name, "desc": desc, "vs_base": vs_baseline(q, QB), "vs_device": vs_device(q), "finite": bool(np.isfinite(out).all())}
    if extra: d.update(extra)
    if RESDIR: np.savez(f"{RESDIR}/{name}.npz", **q)
    save_result(name, d, RESDIR); print("%-28s done in %.0fs  curv %.2e  acc %.3f  y10 %.2f" % (name, time.time() - t0, d["vs_base"]["curv"][0], d["vs_base"]["acc"][0], d["vs_base"]["y10"][0]), flush=True)

def calib():
    finish("cal_zero", "rpy = [0,0,0] (no calibration, as the iOS app does today)", baseline_inputs(rpy=np.zeros(3, np.float32)))
    for ax, i in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        for mag in (0.5, 1, 2, 5):
            for sg in (1, -1):
                r = RPY0.copy().astype(np.float64); r[i] += sg * mag * DEG
                finish("cal_%s_%s%g" % (ax, "+" if sg > 0 else "-", mag), "%s offset %+g deg added to the logged calibration" % (ax, sg * mag), baseline_inputs(rpy=r.astype(np.float32)))

def control():
    """tiny perturbations, to see how much the outputs move for almost no input change (a noise floor for the tables)"""
    for ax, i in (("pitch", 1), ("yaw", 2)):
        r = RPY0.copy().astype(np.float64); r[i] += 0.1 * DEG
        finish("ctl_%s_+0.1" % ax, "CONTROL: %s offset +0.1 deg" % ax, baseline_inputs(rpy=r.astype(np.float32)))
    finish("ctl_focal_+0.5", "CONTROL: both focal lengths +0.5%", baseline_inputs(fn=F_NARROW * 1.005, fw=F_WIDE * 1.005))

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

def more(prefixes=()):
    """Reduced set for the other clips. Cases are built lazily so that `prefixes` can select a subset."""
    global RESDIR
    RESDIR = os.path.join(OUT, "more"); os.makedirs(RESDIR, exist_ok=True)
    np.savez(f"{RESDIR}/_base.npz", v_ego=V_EGO, dev_ok=DEV_OK, has_action=HAS_ACTION, **{"q_" + k: v for k, v in QB.items()},
             **{"dev_" + k: v for k, v in DEV.items()})      # unperturbed quantities and the device's modelV2, for more_report.py
    idx = np.arange(NF); z = np.zeros(3, np.float32); f63 = (W / 2) / np.tan(63.0 * DEG / 2)
    cases = []          # (name, desc, thunk -> frames or (frames, run_mask))
    def add(name, desc, thunk, **kw): cases.append((name, desc, thunk, kw))
    base = lambda: baseline_inputs()
    def rpy_off(i, deg):
        r = RPY0.copy().astype(np.float64); r[i] += deg * DEG; return r.astype(np.float32)
    # noise floor of the clip
    def noise():
        fr = base().copy(); rng = np.random.default_rng(0)       # same generator and draw as noise_floor_check.py
        return np.clip(fr.astype(np.int16) + rng.integers(-1, 2, fr.shape, dtype=np.int8), 0, 255).astype(np.uint8)
    add("ctl_noise1", "CONTROL: +-1 grey level of random noise on every model-input sample (seed 0)", noise)
    for ax, i in (("pitch", 1), ("yaw", 2)):
        add("ctl_%s_+0.1" % ax, "CONTROL: %s offset +0.1 deg" % ax, lambda i=i: baseline_inputs(rpy=rpy_off(i, 0.1)))
    # calibration
    add("cal_zero", "rpy = [0,0,0] (no calibration, as the iOS app does today)", lambda: baseline_inputs(rpy=z))
    for ax, i, mags in (("yaw", 2, (1, 2)), ("pitch", 1, (1, 2))):
        for mag in mags:
            for sg in (1, -1):
                add("cal_%s_%s%g" % (ax, "+" if sg > 0 else "-", mag), "%s offset %+g deg added to the logged calibration" % (ax, sg * mag), lambda i=i, d=sg * mag: baseline_inputs(rpy=rpy_off(i, d)))
    add("cal_roll_+2", "roll offset +2 deg added to the logged calibration", lambda: baseline_inputs(rpy=rpy_off(0, 2)))
    # focal 5 percent
    for which in ("both", "narrow", "wide"):
        for sg in (1, -1):
            s_ = 1 + sg * 0.05
            add("foc_%s_%s5" % (which, "+" if sg > 0 else "-"), "assumed focal length %s %+d%% (true value x %.2f)" % (which, sg * 5, s_),
                lambda which=which, s_=s_: baseline_inputs(fn=F_NARROW * (s_ if which != "wide" else 1), fw=F_WIDE * (s_ if which != "narrow" else 1)))
    # how much does the model use each input
    def with_view(view, fn):
        def t():
            b = base().copy(); b[:, view] = fn(b[:, view]); return b
        return t
    for view, nm in ((1, "wide"), (0, "narrow")):
        add("%s_lag2" % nm, "%s input lags the other by 2 frames (frame 0 repeated at the start)" % nm, with_view(view, lambda x: x[np.maximum(idx - 2, 0)]))
        add("%s_frozen" % nm, "%s input frozen: its first frame repeated for the whole clip" % nm, with_view(view, lambda x: np.repeat(x[:1], NF, 0)))
        add("%s_grey" % nm, "%s input replaced by a constant mid-grey image (all 6 channels 128)" % nm, with_view(view, lambda x: np.full_like(x, 128)))
    # one camera feeding both inputs. one_narrow = wide input built from narrow-camera pixels; one_wide = narrow input built from wide-camera pixels
    add("one_narrow", "both inputs from the NARROW camera frames (= wide input built from narrow-camera pixels), true narrow focal and logged rpy",
        lambda: stack_inputs(warp_view(0, 0, F_NARROW, RPY0), warp_view(0, 1, F_NARROW, RPY0)))
    add("one_wide", "both inputs from the WIDE camera frames (= narrow input built from wide-camera pixels), true wide focal and logged rpy",
        lambda: stack_inputs(warp_view(1, 0, F_WIDE, RPY0), warp_view(1, 1, F_WIDE, RPY0)))
    add("ios_narrowsrc", "iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the NARROW camera frames", lambda: stack_inputs(warp_view(0, 0, f63, z), warp_view(0, 1, f63, z)), extra={"f": float(f63)})
    add("ios_widesrc", "iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the WIDE camera frames", lambda: stack_inputs(warp_view(1, 0, f63, z), warp_view(1, 1, f63, z)), extra={"f": float(f63)})
    add("ios_narrowsrc_truef", "iOS-app recipe but with the true narrow focal length (rpy 0, same narrow frame in both inputs)", lambda: stack_inputs(warp_view(0, 0, F_NARROW, z), warp_view(0, 1, F_NARROW, z)))
    # timing
    def drop5():
        fr = base().copy()
        for t in idx[idx % 5 == 4]: fr[t] = fr[t - 1]
        return fr
    add("drop5_repeat", "every 5th frame dropped, previous model input repeated (model runs every step)", drop5)
    add("drop2_skip", "model run only every other frame (every 2nd frame dropped, model not run, previous outputs held)", base, run_mask=(idx % 2 != 1))
    # image
    add("img_down2", "source frames box-downscaled to 1/2 before the warp (focal scaled)", lambda: stack_inputs(warp_view(0, 0, F_NARROW, RPY0, lambda f: downscale_nv12(f, 2), "d2"), warp_view(1, 1, F_WIDE, RPY0, lambda f: downscale_nv12(f, 2), "d2")))
    sel = [c for c in cases if not prefixes or any(c[0].startswith(p) for p in prefixes)]
    for name, desc, thunk, kw in sel:
        if have(name, RESDIR): continue
        fr = thunk(); finish(name, desc, fr, run_mask=kw.get("run_mask"), extra=kw.get("extra"))

if __name__ == "__main__":
    if sys.argv[1] == "more": more(tuple(sys.argv[2:]))       # run_all.py more [case-name-prefix ...]
    else:
        for g in sys.argv[1:]: globals()[g]()
