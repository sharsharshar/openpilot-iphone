"""Shared helpers for the real-footage experiments: paths, reference loading, driving-relevant metrics."""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# Working directory for the video, decoded frames and reference (about 2 GB). Git-ignored by default.
SCRATCH = os.environ.get("RF_SCRATCH", os.path.join(ROOT, "models", "real_footage"))
sys.path.insert(0, os.path.join(ROOT, "core", "tests", "oracle"))
import oracle                                   # noqa: E402
from upstream_constants import ModelConstants   # noqa: E402

IDX_2S = int(np.argmin(np.abs(np.array(ModelConstants.T_IDXS) - 2.0)))    # T_IDXS[14] = 1.91 s
IDX_10S = 32                                                              # T_IDXS[32] = 10 s
LAT_T, LONG_T = 0.2, 0.5

def driving_quantities(outputs, v_ego):
    """outputs [N,2576], v_ego [N] -> dict of per-step arrays, using oracle parse + get_action (state carried like modeld)."""
    n = len(outputs); prev = (0.0, 0.0)
    q = {k: [] for k in ("curv", "acc", "x2", "y2", "x10", "y10", "lane", "lead", "v0")}
    for i in range(n):
        o = oracle.parse(outputs[i])
        c, a, _ = oracle.get_action(o, prev, LAT_T, LONG_T, float(v_ego[i])); prev = (c, a)
        plan = o["plan"][0]
        pos = plan[:, 0:3]
        q["curv"].append(c); q["acc"].append(a)
        q["x2"].append(pos[IDX_2S, 0]); q["y2"].append(pos[IDX_2S, 1])
        q["x10"].append(pos[IDX_10S, 0]); q["y10"].append(pos[IDX_10S, 1])
        q["lane"].append(o["lane_lines_prob"][0, 1::2]); q["lead"].append(o["lead_prob"][0, 0])
        q["v0"].append(plan[0, 3])    # Plan.VELOCITY x at t=0
    return {k: np.array(v) for k, v in q.items()}

METRICS = [("curvature (1/m)", "curv"), ("accel (m/s^2)", "acc"), ("plan x @1.9s (m)", "x2"), ("plan y @1.9s (m)", "y2"),
           ("plan x @10s (m)", "x10"), ("plan y @10s (m)", "y10"), ("lane-line prob", "lane"), ("lead prob", "lead")]

def driving_diff(qa, qb, sl):
    """mean and worst |a-b| over steps in slice sl for each driving metric."""
    r = {}
    for name, k in METRICS:
        d = np.abs(qa[k][sl] - qb[k][sl]); r[k] = (float(d.mean()), float(d.max()))
    return r

def raw_diff(a, b, sl):
    d = np.abs(a[sl] - b[sl]); return float(d.max()), float(d.mean(1).max()), float(d.mean())

def load_ref(path):
    d = np.load(path); return {k: d[k] for k in ("frames", "desire", "traffic_convention", "action_t", "outputs")}
