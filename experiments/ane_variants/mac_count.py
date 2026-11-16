"""Multiply-accumulates and weight sizes of the lean model by part (Conv / Gemm / MatMul), to put the variants in proportion.
Usage: python -I mac_count.py <model.onnx>"""
import sys
import numpy as np, onnx
from onnx import shape_inference, numpy_helper
m = shape_inference.infer_shapes(onnx.load(sys.argv[1])); g = m.graph
S = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input) + list(g.output)}
I = {i.name: numpy_helper.to_array(i).shape for i in g.initializer}
tot = {}
def add(k, macs, params):
    a = tot.setdefault(k, [0, 0, 0]); a[0] += macs; a[1] += params; a[2] += 1
for n in g.node:
    if n.op_type == "Conv":
        w = I[n.input[1]]; o = S[n.output[0]]
        grp = [a.i for a in n.attribute if a.name == "group"][0]
        kind = "Conv depthwise/grouped" if grp > 1 else ("Conv 1x1" if w[2] * w[3] == 1 else "Conv dense kxk")
        add(kind, int(np.prod(o)) * w[1] * w[2] * w[3], int(np.prod(w)))
    elif n.op_type == "Gemm":
        w = I[n.input[1]]; add("Gemm (vector x matrix)", w[0] * w[1], int(np.prod(w)))
    elif n.op_type == "MatMul" and n.input[1] in I:
        w = I[n.input[1]]; rows = int(np.prod(S[n.input[0]][:-1])); add("MatMul with weights (token linears)", rows * w[0] * w[1], int(np.prod(w)))
    elif n.op_type == "MatMul":
        a, b = S[n.input[0]], S[n.input[1]]; add("MatMul activation x activation", int(np.prod(a[:-1])) * a[-1] * b[-1], 0)
T = sum(v[0] for v in tot.values()); P = sum(v[1] for v in tot.values())
print(f"{'part':40s} {'nodes':>5s} {'MMACs':>9s} {'share':>7s} {'M params':>9s}")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:40s} {v[2]:5d} {v[0] / 1e6:9.1f} {100 * v[0] / T:6.2f}% {v[1] / 1e6:9.2f}")
print(f"{'total':40s} {'':5s} {T / 1e6:9.1f} {'':7s} {P / 1e6:9.2f}")
