"""Write a float32 copy of the float16 ONNX model (weights and internal casts), so we can
measure how much of any disagreement is just float16 rounding.
Usage: python -I make_fp32.py <in.onnx> <out.onnx>"""
import sys, numpy as np, onnx
from onnx import numpy_helper, TensorProto
m = onnx.load(sys.argv[1]); g = m.graph
n_w = n_c = 0
for i, init in enumerate(g.initializer):
    if init.data_type == TensorProto.FLOAT16:
        g.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name)); n_w += 1
for node in g.node:
    if node.op_type == "Cast":
        for a in node.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16: a.i = TensorProto.FLOAT; n_c += 1
del g.value_info[:]
for o in g.output:
    if o.type.tensor_type.elem_type == TensorProto.FLOAT16: o.type.tensor_type.elem_type = TensorProto.FLOAT
onnx.checker.check_model(m); onnx.save(m, sys.argv[2])
print(f"converted {n_w} weight tensors and {n_c} casts to float32 -> {sys.argv[2]}")
