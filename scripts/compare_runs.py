"""Run a sequence through an ONNX model with onnxruntime and compare against the saved reference.
Usage: python -I compare_runs.py <model.onnx> <reference.npz> <label>"""
import sys, numpy as np, onnxruntime as ort
sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
ref = np.load(sys.argv[2]); outs = [o.name for o in sess.get_outputs()]
T = {"tensor(uint8)": np.uint8, "tensor(float)": np.float32, "tensor(float16)": np.float16}
state = {i.name: np.zeros(i.shape, T[i.type]) for i in sess.get_inputs() if i.name.startswith("state_")}
got = []
import time; ts = []
for t in range(len(ref["frames"])):
    feed = {"new_img": ref["frames"][t], "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **state}
    t0 = time.perf_counter(); r = dict(zip(outs, sess.run(outs, feed))); ts.append(time.perf_counter() - t0)
    got.append(r["outputs"][0].astype(np.float32)); state = {k: r["next_" + k] for k in state}
got = np.stack(got); d = np.abs(got - ref["outputs"])
print(f"[{sys.argv[3]}] median {np.median(ts[3:])*1e3:.1f} ms/run")
print(f"  max abs diff: step0 {d[0].max():.4f}, step10 {d[10].max():.4f}, last {d[-1].max():.4f}, overall {d.max():.4f}; mean abs diff {d.mean():.5f}")
i = np.unravel_index(d.argmax(), d.shape); print(f"  worst element: step {i[0]} index {i[1]}: reference {ref['outputs'][i]:.3f} vs {got[i]:.3f}")
rel = d / (np.abs(ref["outputs"]) + 1.0); print(f"  max diff relative to (|ref|+1): {rel.max():.4f}; 99.9th percentile abs diff {np.percentile(d, 99.9):.4f}")
np.save(sys.argv[1].replace(".onnx", "_outputs.npy"), got)
