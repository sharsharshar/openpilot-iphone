import sys, numpy as np, onnx
from onnx import shape_inference, helper
m = shape_inference.infer_shapes(onnx.load(sys.argv[1]))
g = m.graph
prod = {o: (i, n.op_type) for i, n in enumerate(g.node) for o in n.output}
rows = []
for v in list(g.input) + list(g.value_info) + list(g.output):
    t = v.type.tensor_type
    dims = [d.dim_value for d in t.shape.dim]
    if not dims or any(d <= 0 for d in dims): continue
    npdt = helper.tensor_dtype_to_np_dtype(t.elem_type)
    idx, op = prod.get(v.name, (-1, "MODEL INPUT"))
    rows.append((int(np.prod(dims)) * npdt.itemsize, str(npdt), dims, op, idx, v.name))
rows = sorted(set(map(lambda r: (r[0], r[1], tuple(r[2]), r[3], r[4], r[5]), rows)), reverse=True)
print("total tensors with known shape:", len(rows), "of nodes:", len(g.node))
for r in rows[:14]: print(f"{r[0]:>9} bytes  {r[1]:8} {str(list(r[2])):24} {r[3]:12} node#{r[4]:<4} {r[5][:40]}")
convs = [r for r in rows if r[3] == "Conv"]
print("largest Conv outputs:")
for r in convs[:4]: print(f"{r[0]:>9} bytes  {r[1]:8} {str(list(r[2])):24} node#{r[4]}")
