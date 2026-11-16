"""Shared harness for the phone-error experiments: warp with the C core, run the fp32 model with state carried, metrics.

Everything lives in $RF_SCRATCH (read-only inputs from experiments/real_footage) and $RF_SCRATCH/phone_errors (our outputs).
"""
import os, sys, ctypes, subprocess, pickle, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "real_footage"))
import rf_common as rc                               # reads RF_SCRATCH
from rf_common import driving_quantities, METRICS
SCRATCH = rc.SCRATCH
OUT = os.path.join(SCRATCH, "phone_errors"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)

W, H, NF = 1344, 760, 600
FSIZE = W * H * 3 // 2
F_NARROW, F_WIDE = 1522.0 * 3 / 4, 567.0 / 4 * 3
WARM = slice(100, None)
MODEL = os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx")

def K_of(f, w=W, h=H):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

# ---- C core
_build = os.path.join(OUT, "build"); os.makedirs(_build, exist_ok=True)
subprocess.run(["cc", "-std=c99", "-O2", "-shared", "-fPIC", "-o", f"{_build}/libopcore.so", f"{ROOT}/core/opcore.c", "-lm"], check=True)
lib = ctypes.CDLL(f"{_build}/libopcore.so")
f32p = ctypes.POINTER(ctypes.c_float)
lib.op_warp_matrix.argtypes = [f32p, f32p, ctypes.c_int, f32p]
lib.op_frame_to_model_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, f32p]
IMG_LEN = 6 * 128 * 256

def c_matrix(K, rpy, big):
    k = np.ascontiguousarray(K, np.float32); r = np.ascontiguousarray(rpy, np.float32); m = np.zeros(9, np.float32)
    lib.op_warp_matrix(k.ctypes.data_as(f32p), r.ctypes.data_as(f32p), int(big), m.ctypes.data_as(f32p)); return m

def c_warp(nv12, m, w=W, h=H):
    """nv12: flat uint8 [w*h*3/2] -> float32 [6,128,256]"""
    out = np.zeros(IMG_LEN, np.float32); fr = np.ascontiguousarray(nv12)
    lib.op_frame_to_model_input(fr.ctypes.data, w, fr.ctypes.data + w * h, w, w, h, m.ctypes.data_as(f32p), out.ctypes.data_as(f32p))
    return out.reshape(6, 128, 256)

# ---- data
META = np.load(f"{SCRATCH}/real_meta.npz"); V_EGO = META["v_ego"].astype(np.float64); RPY0 = META["rpy"].astype(np.float32)
LOG = pickle.load(open(f"{SCRATCH}/log.pkl", "rb"))
_raw = {}
def raw(cam):          # 0 narrow, 1 wide
    if cam not in _raw: _raw[cam] = np.memmap(f"{SCRATCH}/{'fe'[cam]}_nv12.bin", np.uint8, "r")
    return _raw[cam]
def src_frame(cam, i):
    return np.asarray(raw(cam)[i * FSIZE:(i + 1) * FSIZE])

def downscale_nv12(fr, k):
    """box-average the Y plane and each of the U, V planes by k (k=2,4); returns (flat nv12, w, h)"""
    y = fr[:W * H].reshape(H, W).astype(np.float32); uv = fr[W * H:].reshape(H // 2, W // 2, 2).astype(np.float32)
    bx = lambda a: a.reshape(a.shape[0] // k, k, a.shape[1] // k, k, *a.shape[2:]).mean((1, 3))
    y2 = np.round(bx(y)).astype(np.uint8); uv2 = np.round(bx(uv)).astype(np.uint8)
    return np.concatenate([y2.ravel(), uv2.ravel()]), W // k, H // k

def box_blur_y(fr, r=1):
    y = fr[:W * H].reshape(H, W).astype(np.float32); p = np.pad(y, r, mode="edge"); n = 2 * r + 1
    acc = sum(p[dy:dy + H, dx:dx + W] for dy in range(n) for dx in range(n)) / (n * n)
    out = fr.copy(); out[:W * H] = np.round(acc).astype(np.uint8).ravel(); return out

def gain_y(fr, g):
    out = fr.copy(); out[:W * H] = np.clip(np.round(fr[:W * H].astype(np.float32) * g), 0, 255).astype(np.uint8); return out

_cache = {}

_cache = {}
def warp_view(src_cam, view, f, rpy, xform=None, tag=""):
    """Frames [NF,6,128,256] uint8: source camera `src_cam` (0 narrow, 1 wide) warped into model view `view` (0 narrow, 1 wide)
    with assumed focal length f (px, for the full 1344-wide frame) and rpy. xform(frame)->(nv12,w,h) is an optional change of the
    source image (downscale, blur, gain); for a downscaled source f is scaled by w/W. `tag` names the xform for the cache."""
    key = (src_cam, view, round(float(f), 4), tuple(np.round(np.asarray(rpy, np.float64), 9)), tag)
    if key in _cache: return _cache[key]
    out = np.zeros((NF, 6, 128, 256), np.uint8)
    m_full = c_matrix(K_of(f), rpy, view)
    for i in range(NF):
        fr = src_frame(src_cam, i)
        if xform is None: o = c_warp(fr, m_full)
        else:
            fr2, w, h = xform(fr); o = c_warp(fr2, c_matrix(K_of(f * w / W, w, h), rpy, view), w, h)
        out[i] = o.astype(np.uint8)
    if len(_cache) > 12: _cache.pop(next(iter(_cache)))
    _cache[key] = out
    return out

def stack_inputs(a, b):
    return np.stack([a, b], 1)        # [NF,2,6,128,256]

def baseline_inputs(rpy=RPY0, fn=F_NARROW, fw=F_WIDE):
    return stack_inputs(warp_view(0, 0, fn, rpy), warp_view(1, 1, fw, rpy))

# ---- model
import onnxruntime as ort
_sess = None
def run_model(frames, run_mask=None):
    """float32 model, state from zeros and carried, desire zeros, traffic_convention [1,0], action_t [0.2,0.5], as in build_reference.py.
    run_mask: bool [N]; steps with False are not run (model skipped) and the previous outputs are held."""
    global _sess
    if _sess is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    outs = [o.name for o in _sess.get_outputs()]
    n = len(frames)
    desire = np.zeros(8, np.float32)
    tc = np.array([[1, 0]], np.float32); at = np.array([[0.2, 0.5]], np.float32)
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    res = []; last = None
    for t in range(n):
        if run_mask is not None and not run_mask[t] and last is not None:
            res.append(last); continue
        o = dict(zip(outs, _sess.run(outs, {"new_img": frames[t], "desire": desire, "traffic_convention": tc, "action_t": at, **st})))
        last = o["outputs"][0].astype(np.float32); res.append(last); st = {k: o["next_" + k] for k in st}
    return np.stack(res)

# ---- device modelV2 aligned by frame id
_nar = np.array([x[1] for x in LOG["nar"]])[:NF]
_mm = {m["fid"]: m for m in LOG["model"]}
DEV_OK = np.array([i for i in range(NF) if int(_nar[i]) in _mm])
def _dev(f): return np.array([f(_mm[int(_nar[i])]) for i in DEV_OK])
DEV = {"curv": _dev(lambda m: m["curv"]), "acc": _dev(lambda m: m["acc"]), "y10": _dev(lambda m: m["py"][32]), "x10": _dev(lambda m: m["px"][32]),
       "lane": _dev(lambda m: np.array(m["lp"])[1:3]), "lead": _dev(lambda m: m["lead"][0])}
DEV_WARM = DEV_OK >= 100

# ---- metrics
def quantities(outputs):
    return driving_quantities(outputs, V_EGO)

def vs_baseline(qp, qb):
    r = {}
    for name, k in METRICS:
        d = np.abs(qp[k][WARM] - qb[k][WARM]); r[k] = [float(d.mean()), float(d.max())]
    dc = qp["curv"][WARM] - qb["curv"][WARM]; la = V_EGO[WARM] ** 2 * dc
    r["latacc"] = [float(np.abs(la).mean()), float(np.abs(la).max())]
    r["bias_curv"] = float(dc.mean()); r["bias_y10"] = float((qp["y10"][WARM] - qb["y10"][WARM]).mean())
    r["bias_y2"] = float((qp["y2"][WARM] - qb["y2"][WARM]).mean())
    return r

def vs_device(q):
    ok = DEV_OK[DEV_WARM]; r = {}
    for k in ("curv", "acc", "y10", "x10"):
        r[k] = float(np.abs(DEV[k][DEV_WARM] - q[k][ok]).mean())
    r["lane"] = float(np.abs(DEV["lane"][DEV_WARM] - q["lane"][ok][:, 1:3]).mean())
    r["lead"] = float(np.abs(DEV["lead"][DEV_WARM] - q["lead"][ok]).mean())
    return r

def save_result(name, d):
    json.dump(d, open(f"{RES}/{name}.json", "w"))
def have(name): return os.path.exists(f"{RES}/{name}.json")
