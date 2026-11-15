"""Accuracy cost of compressing the driving model's weights, emulated exactly on Linux.

Weight-only compression is a function of the weights alone, so replacing each weight tensor by its
quantise-then-dequantise version in the float32 ONNX model reproduces what a compressed model computes
(apart from float16 rounding at run time). Runs the 40-step reference with state carried.

Usage: python -I quantize_study.py <driving_supercombo_fp32.onnx> <driving_reference_fp32.npz>
"""
import sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

model_path, ref_path = sys.argv[1], sys.argv[2]
ref = np.load(ref_path)
base = onnx.load(model_path)
used_as_weight = {n.input[1] for n in base.graph.node if n.op_type in ("Conv", "Gemm", "MatMul") and len(n.input) > 1}

def int8_per_channel(w):      # symmetric linear, one scale per output channel (axis 0)
    flat = w.reshape(w.shape[0], -1); s = np.abs(flat).max(1, keepdims=True) / 127.0; s[s == 0] = 1
    return (np.clip(np.round(flat / s), -127, 127) * s).reshape(w.shape)
def int8_per_tensor(w):
    s = np.abs(w).max() / 127.0 or 1.0; return np.clip(np.round(w / s), -127, 127) * s
def palettize(bits):
    def f(w, iters=12):
        x = w.ravel(); k = 2 ** bits
        c = np.quantile(x, (np.arange(k) + 0.5) / k)
        for _ in range(iters):
            edges = (c[1:] + c[:-1]) / 2; idx = np.searchsorted(edges, x)
            sums = np.bincount(idx, weights=x, minlength=k); cnt = np.bincount(idx, minlength=k)
            c = np.where(cnt > 0, sums / np.maximum(cnt, 1), c); c.sort()
        edges = (c[1:] + c[:-1]) / 2; return c[np.searchsorted(edges, x)].reshape(w.shape)
    return f
def fp16(w): return w.astype(np.float16).astype(np.float32)

def run(fn):
    m = onnx.ModelProto(); m.CopyFrom(base); n_t = n_el = 0
    for i, init in enumerate(m.graph.initializer):
        if init.name in used_as_weight and init.data_type == onnx.TensorProto.FLOAT:
            w = numpy_helper.to_array(init)
            if w.size < 1024: continue
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(fn(w).astype(np.float32), init.name)); n_t += 1; n_el += w.size
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]); outs = [o.name for o in sess.get_outputs()]
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    res = []
    for t in range(len(ref["frames"])):
        o = dict(zip(outs, sess.run(outs, {"new_img": ref["frames"][t], "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **st})))
        res.append(o["outputs"][0]); st = {k: o["next_" + k] for k in st}
    d = np.abs(np.stack(res) - ref["outputs"]); return n_t, n_el, d.max(), d.mean(1).max(), d.mean()

cases = [("float16 weights (what ships today)", fp16, 16), ("8-bit linear, per output channel", int8_per_channel, 8), ("8-bit linear, per tensor", int8_per_tensor, 8),
         ("8-bit lookup table (256 values per tensor)", palettize(8), 8), ("6-bit lookup table (64 values)", palettize(6), 6), ("4-bit lookup table (16 values)", palettize(4), 4)]
print(f"{'weight format':44} {'bits':>4} {'tensors':>7} {'weights':>10} | {'worst diff':>10} {'worst step mean':>15} {'overall mean':>12}")
for label, fn, bits in cases:
    n_t, n_el, mx, wm, om = run(fn)
    print(f"{label:44} {bits:4d} {n_t:7d} {n_el:10d} | {mx:10.3f} {wm:15.4f} {om:12.5f}", flush=True)
