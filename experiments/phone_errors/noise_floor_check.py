"""Independent check of the noise floor: how far do the driving outputs move for input changes nobody could see?
Uses only the stored model inputs (no warp): +-1 grey level of random noise on every sample, and a 1-sample horizontal
shift of every channel (2 pixels at full resolution, about 0.13 degrees of yaw for the narrow view).
Usage: RF_SCRATCH=<dir with real_reference_fp32.npz, real_meta.npz> python noise_floor_check.py
"""
import os, sys
import numpy as np, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "real_footage"))
import rf_common as rf

d = {k: v for k, v in np.load(os.path.join(rf.SCRATCH, "real_reference_fp32.npz")).items()}
v_ego = np.load(os.path.join(rf.SCRATCH, "real_meta.npz"))["v_ego"]
T = {"tensor(uint8)": np.uint8, "tensor(float16)": np.float16, "tensor(float)": np.float32}
sess = ort.InferenceSession(os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx"), providers=["CPUExecutionProvider"])
ins = {i.name: i for i in sess.get_inputs()}; outs = [o.name for o in sess.get_outputs()]

def run(frames):
    st = {k: np.zeros(v.shape, dtype=T[v.type]) for k, v in ins.items()}; res = []
    for i in range(len(frames)):
        st["new_img"] = frames[i]
        for k in ("traffic_convention", "action_t", "desire"): st[k] = d[k][i].reshape(st[k].shape).astype(st[k].dtype)
        r = dict(zip(outs, sess.run(outs, st)))
        for k in outs:
            if k.startswith("next_"): st[k[5:]] = r[k]
        res.append(r["outputs"].astype(np.float32).ravel())
    return np.array(res)

fr = d["frames"]; base = run(fr)
print("baseline vs stored reference, max diff:", np.abs(base - d["outputs"]).max())
qb = rf.driving_quantities(base, v_ego)
rng = np.random.default_rng(0)
cases = {"+-1 grey level random noise": np.clip(fr.astype(np.int16) + rng.integers(-1, 2, fr.shape, dtype=np.int8), 0, 255).astype(np.uint8),
         "1-sample horizontal shift": np.roll(fr, 1, axis=-1)}
for name, f in cases.items():
    q = rf.driving_quantities(run(f), v_ego); print(name)
    for k in qb:
        a = np.abs(np.asarray(q[k], float) - np.asarray(qb[k], float))[100:]
        print("   %-6s mean %.6f worst %.6f" % (k, a.mean(), a.max()))
