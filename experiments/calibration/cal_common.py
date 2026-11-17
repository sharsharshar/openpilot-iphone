"""Shared code for the calibration experiments: a line-by-line port of openpilot's Calibrator, the clip loader, the warp + model runner
and the closed-loop replay.

The clip directory comes from the environment variable CAL_CLIP_DIR (a directory like the ones produced by experiments/real_footage:
log.pkl, real_meta.npz, real_reference_fp32.npz, rlog.*, decoded NV12 frames f_nv12.bin / e_nv12.bin either in the clip directory or in
<clip>/calibration/). Everything this experiment writes goes to <clip>/calibration/ (nothing is written into the repo).
Run with the project python and -I.
"""
import os, sys, ctypes, subprocess, pickle, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLIP = os.environ["CAL_CLIP_DIR"]
os.environ["RF_SCRATCH"] = CLIP                      # rf_common reads this at import time
sys.path.insert(0, os.path.join(ROOT, "experiments", "real_footage"))
import rf_common as rc                                # camera_config, intrinsics, driving_quantities   (not edited, only imported)
CDIR = os.path.join(CLIP, "calibration"); os.makedirs(CDIR, exist_ok=True)
RES = os.path.join(CDIR, "results"); os.makedirs(RES, exist_ok=True)
MODEL = os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx")
NF = 600
DEG = np.pi / 180

# ------------------------------------------------------------------------------------------------------------------------------------
# orientation helpers: copies of openpilot/common/transformations/transformations.py (euler2rot_single, rot2quat_single, quat2euler_single,
# rot2euler_single); orientation.py rot_from_euler = euler2rot, euler_from_rot = rot2euler.
def rot_from_euler(euler):
    phi, theta, psi = euler
    cx, sx, cy, sy, cz, sz = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _rot2quat(rot):
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0); w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s; y = (rot[0, 2] - rot[2, 0]) * s; z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s; x = 0.25 * s; y = (rot[0, 1] + rot[1, 0]) / s; z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s; x = (rot[0, 1] + rot[1, 0]) / s; y = 0.25 * s; z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s; x = (rot[0, 2] + rot[2, 0]) / s; y = (rot[1, 2] + rot[2, 1]) / s; z = 0.25 * s
    return np.array([-w, -x, -y, -z]) if w < 0 else np.array([w, x, y, z])

def euler_from_rot(rot):
    w, x, y, z = _rot2quat(rot)
    gamma = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    theta = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([gamma, theta, psi])

# ------------------------------------------------------------------------------------------------------------------------------------
# Port of openpilot/selfdrive/locationd/calibrationd.py (master, read 2026). Names and line order follow the original; the Params / messaging /
# CarParams parts are dropped and the module constants became instance fields so that variants can change them.
MPH_TO_MS = 0.44704                                   # openpilot/common/constants.py CV.MPH_TO_MS
DEFAULTS = dict(
    MIN_SPEED_FILTER=15 * MPH_TO_MS, MAX_VEL_ANGLE_STD=np.radians(0.25), MAX_YAW_RATE_FILTER=np.radians(2), MAX_HEIGHT_STD=np.exp(-3.5),
    SMOOTH_CYCLES=10, BLOCK_SIZE=100, INPUTS_NEEDED=5, INPUTS_WANTED=50,
    MAX_ALLOWED_YAW_SPREAD=np.radians(2), MAX_ALLOWED_PITCH_SPREAD=np.radians(4))
RPY_INIT = np.array([0.0, 0.0, 0.0]); WIDE_FROM_DEVICE_EULER_INIT = np.array([0.0, 0.0, 0.0]); HEIGHT_INIT = np.array([1.22])
YAW_LIMITS = np.array([-0.06912048084718224, 0.06912048084718235])
def pitch_limits(mici): return np.array([-0.143101, 0.22235988]) if mici else np.array([-0.09074112085129739, 0.17])
UNCAL, CAL, INVALID, RECAL = 0, 1, 2, 3               # log.ExtrinsicsCalibration.Status

def moving_avg_with_linear_decay(prev_mean, new_val, idx, block_size):
    return (idx * prev_mean + (block_size - idx) * new_val) / block_size

class Calibrator:
    def __init__(self, rpy_init=RPY_INIT, valid_blocks=0, mici=False, **consts):
        c = dict(DEFAULTS); c.update(consts)
        for k, v in c.items(): setattr(self, k, v)
        self.PITCH_LIMITS = pitch_limits(mici)
        self.not_car = False
        self.cal_status = UNCAL
        self.reset(np.asarray(rpy_init, float), valid_blocks, WIDE_FROM_DEVICE_EULER_INIT, HEIGHT_INIT)
        self.seeded = self.valid_blocks > 0
        self.update_status()

    def is_calibration_valid(self, rpy):
        return (self.PITCH_LIMITS[0] < rpy[1] < self.PITCH_LIMITS[1]) and (YAW_LIMITS[0] < rpy[2] < YAW_LIMITS[1])

    def sanity_clip(self, rpy):
        if np.isnan(rpy).any(): rpy = RPY_INIT
        return np.array([rpy[0], np.clip(rpy[1], self.PITCH_LIMITS[0] - .005, self.PITCH_LIMITS[1] + .005), np.clip(rpy[2], YAW_LIMITS[0] - .005, YAW_LIMITS[1] + .005)])

    def reset(self, rpy_init=RPY_INIT, valid_blocks=0, wide_from_device_euler_init=WIDE_FROM_DEVICE_EULER_INIT, height_init=HEIGHT_INIT, smooth_from=None):
        self.rpy = RPY_INIT.copy() if not np.isfinite(rpy_init).all() else np.array(rpy_init, float)
        self.height = HEIGHT_INIT.copy() if (not np.isfinite(height_init).all() or len(height_init) != 1) else np.array(height_init, float)
        self.wide_from_device_euler = WIDE_FROM_DEVICE_EULER_INIT.copy() if (not np.isfinite(wide_from_device_euler_init).all() or len(wide_from_device_euler_init) != 3) \
            else np.array(wide_from_device_euler_init, float)
        self.valid_blocks = 0 if (not np.isfinite(valid_blocks) or valid_blocks < 0) else valid_blocks
        self.rpys = np.tile(self.rpy, (self.INPUTS_WANTED, 1))
        self.wide_from_device_eulers = np.tile(self.wide_from_device_euler, (self.INPUTS_WANTED, 1))
        self.heights = np.tile(self.height, (self.INPUTS_WANTED, 1))
        self.idx = 0; self.block_idx = 0; self.v_ego = 0.0
        if smooth_from is None: self.old_rpy = RPY_INIT; self.old_rpy_weight = 0.0
        else: self.old_rpy = smooth_from; self.old_rpy_weight = 1.0

    def get_valid_idxs(self):
        before_current = list(range(self.block_idx))
        after_current = list(range(min(self.valid_blocks, self.block_idx + 1), self.valid_blocks))
        return before_current + after_current

    def update_status(self):
        valid_idxs = self.get_valid_idxs()
        if valid_idxs:
            self.wide_from_device_euler = np.mean(self.wide_from_device_eulers[valid_idxs], axis=0)
            self.height = np.mean(self.heights[valid_idxs], axis=0)
            rpys = self.rpys[valid_idxs]
            self.rpy = np.mean(rpys, axis=0)
            self.calib_spread = np.abs(np.max(rpys, axis=0) - np.min(rpys, axis=0))
        else:
            self.calib_spread = np.zeros(3)
        if self.valid_blocks < self.INPUTS_NEEDED:
            self.cal_status = RECAL if self.cal_status == RECAL else UNCAL
        elif self.is_calibration_valid(self.rpy):
            self.cal_status = CAL
        else:
            self.cal_status = INVALID
        spread_too_high = self.calib_spread[1] > self.MAX_ALLOWED_PITCH_SPREAD or self.calib_spread[2] > self.MAX_ALLOWED_YAW_SPREAD
        if spread_too_high and self.cal_status == CAL:
            self.reset(self.rpys[self.block_idx - 1], smooth_from=self.rpy)
            self.cal_status = RECAL

    def handle_v_ego(self, v_ego): self.v_ego = v_ego

    def get_smooth_rpy(self):
        if self.old_rpy_weight > 0: return self.old_rpy_weight * self.old_rpy + (1.0 - self.old_rpy_weight) * self.rpy
        return self.rpy

    def gates(self, trans, rot, trans_std, road_transform_trans_std):
        """The individual conditions of handle_cam_odom, for statistics: returns (speed_ok, yawrate_ok, angle_certain, height_certain)."""
        speed_ok = (self.v_ego > self.MIN_SPEED_FILTER) and (trans[0] > self.MIN_SPEED_FILTER)
        yaw_ok = abs(rot[2]) < self.MAX_YAW_RATE_FILTER
        rpy_certain = np.arctan2(trans_std[1], trans[0]) < self.MAX_VEL_ANGLE_STD
        height_certain = road_transform_trans_std[2] < self.MAX_HEIGHT_STD if len(road_transform_trans_std) == 3 else True
        return bool(speed_ok), bool(yaw_ok), bool(rpy_certain), bool(height_certain)

    def handle_cam_odom(self, trans, rot, wide_from_device_euler, trans_std, road_transform_trans, road_transform_trans_std):
        self.old_rpy_weight = max(0.0, self.old_rpy_weight - 1 / self.SMOOTH_CYCLES)
        straight_and_fast = ((self.v_ego > self.MIN_SPEED_FILTER) and (trans[0] > self.MIN_SPEED_FILTER) and (abs(rot[2]) < self.MAX_YAW_RATE_FILTER))
        rpy_certain = np.arctan2(trans_std[1], trans[0]) < self.MAX_VEL_ANGLE_STD
        height_certain = road_transform_trans_std[2] < self.MAX_HEIGHT_STD if len(road_transform_trans_std) == 3 else True
        certain_if_calib = (rpy_certain and height_certain) or (self.valid_blocks < self.INPUTS_NEEDED)
        if not (straight_and_fast and certain_if_calib):
            return None
        observed_rpy = np.array([0, -np.arctan2(trans[2], trans[0]), np.arctan2(trans[1], trans[0])])
        new_rpy = euler_from_rot(rot_from_euler(self.get_smooth_rpy()).dot(rot_from_euler(observed_rpy)))
        new_rpy = self.sanity_clip(new_rpy)
        new_wide = np.array(wide_from_device_euler) if len(wide_from_device_euler) == 3 else WIDE_FROM_DEVICE_EULER_INIT
        new_height = np.array([road_transform_trans[2]]) if len(road_transform_trans) == 3 else HEIGHT_INIT
        bs = float(self.BLOCK_SIZE)
        self.rpys[self.block_idx] = moving_avg_with_linear_decay(self.rpys[self.block_idx], new_rpy, self.idx, bs)
        self.wide_from_device_eulers[self.block_idx] = moving_avg_with_linear_decay(self.wide_from_device_eulers[self.block_idx], new_wide, self.idx, bs)
        self.heights[self.block_idx] = moving_avg_with_linear_decay(self.heights[self.block_idx], new_height, self.idx, bs)
        self.idx = (self.idx + 1) % self.BLOCK_SIZE
        if self.idx == 0 and not self.seeded:
            self.seeded = True
            self.rpy = self.rpys[0].copy()
        elif self.idx == 0:
            self.block_idx += 1
            self.valid_blocks = max(self.block_idx, self.valid_blocks)
            self.block_idx = self.block_idx % self.INPUTS_WANTED
        self.update_status()
        return new_rpy

    def cal_perc(self):   # get_msg: calPerc
        return min(100 * ((self.seeded + self.valid_blocks) * self.BLOCK_SIZE + self.idx) // ((1 + self.INPUTS_NEEDED) * self.BLOCK_SIZE), 100)

def observe(rpy_current, trans, rot=None):
    """The observation of handle_cam_odom without any gating: observed direction of travel (roll, pitch, yaw) in the frame of the warped image,
    and new_rpy = euler(R(rpy_current) R(observed)) after sanity_clip is NOT applied (use Calibrator.sanity_clip if needed)."""
    obs = np.array([0, -np.arctan2(trans[2], trans[0]), np.arctan2(trans[1], trans[0])])
    return obs, euler_from_rot(rot_from_euler(rpy_current).dot(rot_from_euler(obs)))

# ------------------------------------------------------------------------------------------------------------------------------------
# clip data
LOG = pickle.load(open(f"{CLIP}/log.pkl", "rb"))
META = np.load(f"{CLIP}/real_meta.npz"); V_EGO = META["v_ego"].astype(np.float64); RPY0 = META["rpy"].astype(np.float32); FRAME_ID = META["frame_id"]
W, H, F_NARROW, F_WIDE, CFG_KEY = rc.camera_config(LOG["dev"], LOG["sensor"], LOG.get("init", "unknown"))
FSIZE = W * H * 3 // 2
MICI = (LOG["dev"] == "mici") or (LOG["dev"] in ("unknown", None) and LOG.get("init") == "mici")
K_NARROW = rc.intrinsics(W, H, F_NARROW); K_WIDE = rc.intrinsics(W, H, F_WIDE)

def load_cal_log():
    p = f"{CDIR}/cal_log.pkl"
    if not os.path.exists(p):
        raise SystemExit("run extract_cal_log.py first")
    return pickle.load(open(p, "rb"))
CALLOG = load_cal_log()
_t_nar = np.array([x[0] for x in LOG["nar"]], np.float64)[:NF]
_ext = CALLOG["ext"] or CALLOG["live"]
_ext_t = np.array([e["t"] for e in _ext], np.float64)
_ext_i = np.clip(np.searchsorted(_ext_t, _t_nar, side="right") - 1, 0, len(_ext) - 1)
RPY_LOG = np.array([_ext[i]["rpy"] for i in _ext_i], np.float64)                 # [NF,3] logged calibration valid at each frame (device's own)
WIDE_LOG = np.array([_ext[i]["wide"] if len(_ext[i]["wide"]) == 3 else [np.nan] * 3 for i in _ext_i], np.float64)
HEIGHT_LOG = np.array([_ext[i]["height"][0] if len(_ext[i]["height"]) == 1 else np.nan for i in _ext_i], np.float64)
assert np.abs(RPY_LOG[0] - RPY0).max() < 1e-6, ("logged rpy differs from real_meta rpy", RPY_LOG[0], RPY0)
_odo = {o["fid"]: o for o in CALLOG["odo"]}
ODO_IDX = [i for i in range(NF) if int(FRAME_ID[i]) in _odo]
def odo_array(key):   # device's own cameraOdometry field per video frame (frames without a message are nan)
    n = len(_odo[int(FRAME_ID[ODO_IDX[0]])][key]); a = np.full((NF, n), np.nan)
    for i in ODO_IDX: a[i] = _odo[int(FRAME_ID[i])][key]
    return a

# ------------------------------------------------------------------------------------------------------------------------------------
# C warp
_build = os.path.join(CDIR, "build"); os.makedirs(_build, exist_ok=True)
_tmp_so = f"{_build}/libopcore.{os.getpid()}.so"     # per-process file + rename: several worker processes start at the same time
subprocess.run(["cc", "-std=c99", "-O2", "-shared", "-fPIC", "-o", _tmp_so, f"{ROOT}/core/opcore.c", "-lm"], check=True); os.replace(_tmp_so, f"{_build}/libopcore.so")
_lib = ctypes.CDLL(f"{_build}/libopcore.so"); f32p = ctypes.POINTER(ctypes.c_float)
_lib.op_warp_matrix.argtypes = [f32p, f32p, ctypes.c_int, f32p]
_lib.op_frame_to_model_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, f32p]
IMG_LEN = 6 * 128 * 256
def c_matrix(K, rpy, big):
    k = np.ascontiguousarray(K, np.float32); r = np.ascontiguousarray(rpy, np.float32); m = np.zeros(9, np.float32)
    _lib.op_warp_matrix(k.ctypes.data_as(f32p), r.ctypes.data_as(f32p), int(big), m.ctypes.data_as(f32p)); return m
def c_warp(nv12, m):
    out = np.zeros(IMG_LEN, np.float32); fr = np.ascontiguousarray(nv12)
    _lib.op_frame_to_model_input(fr.ctypes.data, W, fr.ctypes.data + W * H, W, W, H, m.ctypes.data_as(f32p), out.ctypes.data_as(f32p))
    return out.reshape(6, 128, 256)

def nv12_path(cam):
    n = "fe"[cam] + "_nv12.bin"
    for d in (CDIR, CLIP):
        if os.path.exists(f"{d}/{n}"): return f"{d}/{n}"
    raise FileNotFoundError(f"decoded frames {n} not found in {CDIR} or {CLIP} (see decode_clip.sh)")
_raw = {}
def src_frame(cam, i):
    if cam not in _raw: _raw[cam] = np.memmap(nv12_path(cam), np.uint8, "r")
    return np.asarray(_raw[cam][i * FSIZE:(i + 1) * FSIZE])

def warp_pair(i, rpy, _cache={}):
    """[2,6,128,256] uint8 model input for video frame i with ONE angle triple for both cameras, as modeld.py does (narrow: bigmodel_frame False, wide: True)."""
    key = tuple(np.asarray(rpy, np.float32).tolist())
    if key not in _cache:
        if len(_cache) > 64: _cache.clear()
        _cache[key] = (c_matrix(K_NARROW, rpy, 0), c_matrix(K_WIDE, rpy, 1))
    mn, mw = _cache[key]
    return np.stack([c_warp(src_frame(0, i), mn), c_warp(src_frame(1, i), mw)]).astype(np.uint8)

# ------------------------------------------------------------------------------------------------------------------------------------
# model
import onnxruntime as ort
_sess = None
def session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("CAL_THREADS", 2))
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _sess

class Runner:
    """float32 model with its recurrent state carried, desire zeros, traffic convention [1,0], action_t [0.2,0.5] (as build_reference.py)."""
    def __init__(self):
        self.s = session(); self.outs = [o.name for o in self.s.get_outputs()]
        self.st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
        self.desire = np.zeros(8, np.float32); self.tc = np.array([[1, 0]], np.float32); self.at = np.array([[0.2, 0.5]], np.float32)
    def step(self, frames):
        o = dict(zip(self.outs, self.s.run(self.outs, {"new_img": frames, "desire": self.desire, "traffic_convention": self.tc, "action_t": self.at, **self.st})))
        self.st = {k: o["next_" + k] for k in self.st}
        return o["outputs"][0].astype(np.float32)

def safe_exp(x): return np.exp(np.clip(x, -np.inf, 11))
def split_pose(row):
    """Parser.parse_mdn on the pose / wide_from_device_euler / road_transform slices (parse_model_outputs.py), mean and std (std = safe_exp(raw))."""
    pose = row[87:99]; wide = row[99:105]; rt = row[105:117]
    return dict(trans=pose[:3], rot=pose[3:6], trans_std=safe_exp(pose[6:9]), rot_std=safe_exp(pose[9:12]), wide=wide[:3], wide_std=safe_exp(wide[3:6]),
                rt=rt[:3], rt_std=safe_exp(rt[6:9]))

# ------------------------------------------------------------------------------------------------------------------------------------
STEP_KEYS = ("rpy_used", "cal_rpy", "new_rpy", "status", "valid_blocks", "idx", "gated", "cal_perc", "seeded")
def replay(init_rpy, n_loops=1, closed=True, consts=None, frames_fn=None, publish_every=5, progress=None, keep_outputs=False):
    """Replay the clip n_loops times in a row (the model state and the calibrator are carried across the seam).
    closed=True: the warp uses the angles the Calibrator has published (every 5th camera-odometry message, i.e. 4 Hz, as calibrationd.main does;
      modeld picks the new angles up before its next frame); the Calibrator is fed with the model's own pose output and carState.vEgo.
    closed=False: the warp uses init_rpy for every frame (the Calibrator is run on the side, only to record what it would have observed).
    frames_fn(k, i, rpy) -> [2,6,128,256] uint8 model input; default warps the decoded frames with rpy.
    Returns dict of per-step arrays."""
    consts = consts or {}
    cal = Calibrator(init_rpy, mici=MICI, **consts)
    run = Runner(); N = n_loops * NF
    rec = {k: [] for k in STEP_KEYS}; rows = []; pose = []
    rpy_used = np.array(init_rpy, np.float64)
    t0 = time.time()
    for k in range(N):
        i = k % NF
        fr = frames_fn(k, i, rpy_used) if frames_fn else warp_pair(i, rpy_used)
        row = run.step(fr); rows.append(row)
        p = split_pose(row); pose.append(row[87:117])
        cal.handle_v_ego(V_EGO[i])
        used = rpy_used.copy()
        if closed:
            new = cal.handle_cam_odom(p["trans"], p["rot"], p["wide"], p["trans_std"], p["rt"], p["rt_std"])
            if k % publish_every == 0: rpy_used = np.array(cal.get_smooth_rpy(), np.float64)
        else:   # fixed warp angles: no feedback; record what calibrationd's observation would be relative to the angles used (same gates as an uncalibrated Calibrator)
            sp, yw, _, _ = cal.gates(p["trans"], p["rot"], p["trans_std"], p["rt_std"])
            new = cal.sanity_clip(observe(used, p["trans"])[1]) if (sp and yw) else None
        for key, v in zip(STEP_KEYS, (used, cal.rpy.copy(), np.full(3, np.nan) if new is None else new, cal.cal_status, cal.valid_blocks, cal.idx, new is not None, cal.cal_perc(), cal.seeded)):
            rec[key].append(v)
        if progress and k % 600 == 599: progress(k, time.time() - t0)
    out = {k: np.array(v) for k, v in rec.items()}
    out["pose30"] = np.array(pose, np.float32)
    outputs = np.array(rows, np.float32)
    q = rc.driving_quantities(outputs, np.tile(V_EGO, n_loops))
    for k, v in q.items(): out["q_" + k] = np.asarray(v)
    if keep_outputs: out["outputs"] = outputs
    return out
