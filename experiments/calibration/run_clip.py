"""Run the calibration replays for one clip (clip directory in CAL_CLIP_DIR; decoded frames must exist, see decode_clip.sh).
Usage: CAL_CLIP_DIR=<clip dir> CAL_LOOPS=<n> python -I run_clip.py [--part i/n] [--only substring]
Results: <clip>/calibration/results/<tag>.npz (existing files are skipped).

Tags (starts: zero = [0,0,0]; yaw+2 ... pitch-5 = the logged calibration plus that offset in degrees, the same offset triple is used for both cameras):
  open_<start>   fixed angles for all 600 frames (no feedback): how the outputs and the observation depend on the angles (open_logged = the stored reference)
  cl1_<start>    closed loop with calibrationd's unmodified constants, ONE pass (30 s)
  cls_<start>    closed loop, VARIANT: BLOCK_SIZE 20 instead of 100 (everything else unmodified), one pass
  clL_<start>    closed loop, VARIANT: the clip replayed CAL_LOOPS times in a row, unmodified constants
  openL_logged / openL_zero   fixed angles, CAL_LOOPS passes (baselines for clL_*)
  noise1 / noiseL   reference frames + random +-1 grey level, one pass / CAL_LOOPS passes (noise floor)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cal_common as cc
from cal_common import DEG

ap = argparse.ArgumentParser(); ap.add_argument("--part", default="0/1"); ap.add_argument("--only", default=""); ap.add_argument("--list", action="store_true")
a = ap.parse_args()
L = int(os.environ.get("CAL_LOOPS", 4))
OFFS = {"yaw+2": (2, 2), "yaw-2": (2, -2), "yaw+5": (2, 5), "yaw-5": (2, -5), "pitch+2": (1, 2), "pitch-2": (1, -2), "pitch+5": (1, 5), "pitch-5": (1, -5)}
def start_rpy(name):
    if name == "zero": return np.zeros(3)
    if name == "logged": return cc.RPY0.astype(np.float64)
    ax, d = OFFS[name]; r = cc.RPY0.astype(np.float64).copy(); r[ax] += d * DEG; return r
starts = ["zero"] + list(OFFS)
tasks = [("open_logged", dict(init="logged", loops=1, closed=False)), ("open_zero", dict(init="zero", loops=1, closed=False))]
tasks += [(f"open_{s}", dict(init=s, loops=1, closed=False)) for s in OFFS]
tasks += [(f"cl1_{s}", dict(init=s, loops=1, closed=True)) for s in starts]
tasks += [(f"cls_{s}", dict(init=s, loops=1, closed=True, consts=dict(BLOCK_SIZE=20))) for s in starts]
tasks += [("openL_logged", dict(init="logged", loops=L, closed=False)), ("openL_zero", dict(init="zero", loops=L, closed=False))]
tasks += [(f"clL_{s}", dict(init=s, loops=L, closed=True)) for s in starts]
tasks += [("noise1", dict(init="logged", loops=1, closed=False, noise=True)), ("noiseL", dict(init="logged", loops=L, closed=False, noise=True))]
tasks = [t for t in tasks if a.only in t[0]]
tasks.sort(key=lambda t: -t[1]["loops"])
pi, pn = map(int, a.part.split("/")); tasks = tasks[pi::pn]
if a.list:
    print([t[0] for t in tasks]); sys.exit()

ref_frames = None
def noise_frames_fn():
    global ref_frames
    if ref_frames is None: ref_frames = np.load(f"{cc.CLIP}/real_reference_fp32.npz")["frames"]
    def fn(k, i, rpy):
        rng = np.random.default_rng(1000 + k)
        return np.clip(ref_frames[i].astype(np.int16) + rng.integers(-1, 2, ref_frames[i].shape, dtype=np.int8), 0, 255).astype(np.uint8)
    return fn

for tag, kw in tasks:
    path = f"{cc.RES}/{tag}.npz"
    if os.path.exists(path): print("skip", tag); continue
    t0 = time.time(); init = start_rpy(kw["init"])
    out = cc.replay(init, n_loops=kw["loops"], closed=kw["closed"], consts=kw.get("consts"), frames_fn=noise_frames_fn() if kw.get("noise") else None, keep_outputs=(tag == "open_logged"))
    out["init_rpy"] = init; out["loops"] = np.array(kw["loops"])
    np.savez(path + ".tmp.npz", **out); os.replace(path + ".tmp.npz", path)
    print("%-16s %5d steps %6.1fs  final used rpy deg %s" % (tag, kw["loops"] * cc.NF, time.time() - t0, np.round(out["rpy_used"][-1] / DEG, 3)), flush=True)
