"""RUN ON A MAC (Core ML cannot execute on Linux). NOT YET TESTED - written without a Mac.
Runs the converted LEAN package (img_stack input, no frame queue; see make_lean.py) over the saved 40-step
sequence, carrying the two small state queues from one run to the next and building img_stack from a numpy frame
history exactly as the app does (verify_lean.NumpyHistory), and compares against the reference from the original
ONNX model.

Usage: python verify_coreml_mac.py <model.mlpackage> <reference.npz> [ALL|CPU_ONLY|CPU_AND_GPU|CPU_AND_NE]
Needs: pip install coremltools numpy
"""
import os, sys, time, numpy as np, coremltools as ct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lean import NumpyHistory

units = getattr(ct.ComputeUnit, sys.argv[3] if len(sys.argv) > 3 else "ALL")
ml = ct.models.MLModel(sys.argv[1], compute_units=units)
ref = np.load(sys.argv[2])
hist = NumpyHistory()
state = {"state_desire_q": np.zeros((100, 1, 8), np.float32),
         "state_feat_q": np.zeros((96, 1, 512), np.float32)}
got, ts = [], []
for t in range(len(ref["frames"])):
    hist.push(ref["frames"][t])
    feed = {"img_stack": hist.stack(), "desire": ref["desire"][t],
            "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **state}
    t0 = time.perf_counter(); r = ml.predict(feed); ts.append(time.perf_counter() - t0)
    got.append(np.asarray(r["outputs"], np.float32)[0])
    state = {k: np.asarray(r["next_" + k], np.float32) for k in state}   # float16 out -> float32 in
got = np.stack(got); d = np.abs(got - ref["outputs"])
print(f"compute units {units}: median {np.median(ts[3:]) * 1e3:.1f} ms per run (includes Python overhead)")
print(f"all finite: {np.isfinite(got).all()}")
print(f"max abs diff: step 0 {d[0].max():.4f}, step 10 {d[10].max():.4f}, last {d[-1].max():.4f}, overall {d.max():.4f}")
print(f"mean abs diff {d.mean():.5f}   (float16-vs-float32 rounding alone gave: overall max 1.71, mean 0.010)")
i = np.unravel_index(d.argmax(), d.shape); print(f"worst: step {i[0]}, output index {i[1]}: reference {ref['outputs'][i]:.3f}, got {got[i]:.3f}")
