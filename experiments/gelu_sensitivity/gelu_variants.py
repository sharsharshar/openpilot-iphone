"""How sensitive is the driving model to the way GELU is computed?

Replaces all 20 Gelu nodes (tanh variant) in the float32 ONNX model with alternatives built only from
operations every accelerator has, runs the 40-step reference sequence with state carried, and reports the
difference from the reference outputs.

Usage: python -I gelu_variants.py <driving_supercombo_fp32.onnx> <driving_reference_fp32.npz>
"""
import sys, math
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

model_path, ref_path = sys.argv[1], sys.argv[2]
ref = np.load(ref_path)

def gelu_tanh(x): return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def const(name, arr): return numpy_helper.from_array(np.asarray(arr, np.float32), name)

def variant_model(kind, **kw):
    m = onnx.load(model_path); g = m.graph
    new_nodes, inits = [], []
    for n in g.node:
        if n.op_type != "Gelu":
            new_nodes.append(n); continue
        x, y, p = n.input[0], n.output[0], n.name.replace("/", "_") + "_"
        if kind == "tanh":           # unchanged
            new_nodes.append(n)
        elif kind == "exact":        # erf form
            new_nodes.append(helper.make_node("Gelu", [x], [y], approximate="none"))
        elif kind == "sigmoid":      # x * sigmoid(1.702 x)
            inits.append(const(p + "k", 1.702))
            new_nodes += [helper.make_node("Mul", [x, p + "k"], [p + "a"]), helper.make_node("Sigmoid", [p + "a"], [p + "b"]),
                          helper.make_node("Mul", [x, p + "b"], [y])]
        elif kind == "pwl":          # piecewise linear: g(t0) + sum_k c_k * relu(x - t_k)
            K, lo, hi = kw["segments"], kw.get("lo", -6.0), kw.get("hi", 6.0)
            t = np.linspace(lo, hi, K + 1); gv = gelu_tanh(t)
            slopes = np.diff(gv) / np.diff(t)
            c = np.concatenate([[slopes[0]], np.diff(slopes), [1.0 - slopes[-1]]])   # slope 0 left of lo, 1 right of hi
            inits += [const(p + "t", t), const(p + "c", c), const(p + "g0", gv[0]), numpy_helper.from_array(np.array([-1], np.int64), p + "ax")]
            new_nodes += [helper.make_node("Unsqueeze", [x, p + "ax"], [p + "u"]), helper.make_node("Sub", [p + "u", p + "t"], [p + "d"]),
                          helper.make_node("Relu", [p + "d"], [p + "r"]), helper.make_node("Mul", [p + "r", p + "c"], [p + "m"]),
                          helper.make_node("ReduceSum", [p + "m", p + "ax"], [p + "s"], keepdims=0), helper.make_node("Add", [p + "s", p + "g0"], [y])]
    del g.node[:]; g.node.extend(new_nodes); g.initializer.extend(inits); del g.value_info[:]
    return m.SerializeToString()

def run(model_bytes):
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"]); outs = [o.name for o in sess.get_outputs()]
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    res = []
    for t in range(len(ref["frames"])):
        o = dict(zip(outs, sess.run(outs, {"new_img": ref["frames"][t], "desire": ref["desire"][t], "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t], **st})))
        res.append(o["outputs"][0]); st = {k: o["next_" + k] for k in st}
    d = np.abs(np.stack(res) - ref["outputs"]); return d.max(), d.mean(1).max(), d.mean()

xs = np.linspace(-8, 8, 200001)
def fn_err(f): e = np.abs(f(xs) - gelu_tanh(xs)); return e.max()
def pwl_fn(K, lo=-6.0, hi=6.0):
    t = np.linspace(lo, hi, K + 1); gv = gelu_tanh(t)
    return lambda x: np.where(x < lo, gv[0], np.where(x > hi, gv[-1] + (x - hi), np.interp(x, t, gv)))
from math import erf
exact = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / math.sqrt(2))))
cases = [("tanh form (as shipped)", "tanh", {}, lambda x: gelu_tanh(x)), ("exact erf form", "exact", {}, exact),
         ("x * sigmoid(1.702 x)", "sigmoid", {}, lambda x: x / (1 + np.exp(-1.702 * x)))]
cases += [(f"piecewise linear, {K} segments", "pwl", {"segments": K}, pwl_fn(K)) for K in (8, 16, 32, 64)]
print(f"{'GELU implementation':32} {'max fn error':>12} | model outputs vs reference: {'worst':>8} {'worst step mean':>16} {'overall mean':>13}")
for label, kind, kw, f in cases:
    mx, wm, om = run(variant_model(kind, **kw))
    print(f"{label:32} {fn_err(f):12.5f} | {'':28} {mx:8.3f} {wm:16.4f} {om:13.5f}")
