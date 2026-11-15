"""Prove on Linux that the lean model (frame queue removed, see make_lean.py) is the same model.

Usage: python -I verify_lean.py [lean_fp32.onnx] [reference.npz]

Keeps a 5-frame history per camera in plain numpy exactly as the app does (core/opcore.c op_hist_*), builds
img_stack each step, runs the lean float32 model with onnxruntime carrying only state_desire_q / state_feat_q,
and compares with the reference outputs of the ORIGINAL model (which carried the frame queue itself).
If models/driving_supercombo_fp32.onnx exists it is also run side by side to compare bit for bit.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

class NumpyHistory:
    """The last 5 model images per camera (uint8, zeros until filled). Frame layout [2 cams][6][128][256]."""
    def __init__(self):
        self.q = np.zeros((5, 2, 6, 128, 256), np.uint8)
    def push(self, frame):
        self.q = np.concatenate([self.q[1:], np.asarray(frame, np.uint8)[None]], axis=0)
    def stack(self):
        """[1, 24, 128, 256] float32: cam0 (frame 4 steps ago, newest), cam1 (same). Raw 0..255 values."""
        parts = [self.q[k, c] for c in (0, 1) for k in (0, 4)]
        return np.concatenate(parts, axis=0)[None].astype(np.float32)

def main():
    import onnxruntime as ort
    lean = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "driving_supercombo_lean_fp32.onnx")
    refp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "reference", "driving_reference_fp32.npz")
    ref = np.load(refp)
    sess = ort.InferenceSession(lean, providers=["CPUExecutionProvider"])
    print("lean inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
    outs = [o.name for o in sess.get_outputs()]; print("lean outputs:", outs)
    full = os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx")
    fsess = ort.InferenceSession(full, providers=["CPUExecutionProvider"]) if os.path.exists(full) else None
    hist = NumpyHistory()
    st = {"state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    fst = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), **{k: v.copy() for k, v in st.items()}}
    diffs, bitdiffs = [], []
    for t in range(len(ref["frames"])):
        hist.push(ref["frames"][t])
        feed = {"img_stack": hist.stack(), "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t],
                "action_t": ref["action_t"][t], **st}
        res = dict(zip(outs, sess.run(outs, feed)))
        st = {k: res["next_" + k] for k in st}
        diffs.append(float(np.abs(res["outputs"][0] - ref["outputs"][t]).max()))
        if fsess is not None:
            ff = {"new_img": ref["frames"][t], "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t],
                  "action_t": ref["action_t"][t], **fst}
            fo = [o.name for o in fsess.get_outputs()]; fr = dict(zip(fo, fsess.run(fo, ff)))
            fst = {k: fr["next_" + k] for k in fst}
            bitdiffs.append(float(np.abs(res["outputs"] - fr["outputs"]).max()))
    d = np.array(diffs)
    print(f"steps {len(d)}; max |lean - reference| per step, first 6: {np.array2string(d[:6], precision=2)}")
    print(f"early steps (history partly zeros, 0-3): max {d[:4].max():.3e}; later (4-39): max {d[4:].max():.3e}")
    print(f"overall max abs diff vs reference: {d.max():.3e}  (limit 1e-4)")
    if bitdiffs:
        print(f"lean vs original fp32 model run side by side: max abs diff {max(bitdiffs):.3e}")
    ok = d.max() < 1e-4
    print("VERIFY_LEAN", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
