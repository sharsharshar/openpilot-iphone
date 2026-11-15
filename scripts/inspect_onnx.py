import sys, collections
import onnx
from onnx import shape_inference, numpy_helper

def dims(vi):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]

def dt(vi):
    return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)

for path in sys.argv[1:]:
    m = onnx.load(path)
    g = m.graph
    print("=" * 70); print(path)
    print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import], "ir:", m.ir_version, "producer:", m.producer_name, m.producer_version)
    init = {i.name for i in g.initializer}
    print("INPUTS:")
    for i in g.input:
        if i.name not in init: print("  ", i.name, dt(i), dims(i))
    print("OUTPUTS:")
    for o in g.output: print("  ", o.name, dt(o), dims(o))
    nparams = sum(int(__import__("numpy").prod(i.dims)) for i in g.initializer)
    print("params:", nparams, "initializer dtypes:", collections.Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in g.initializer))
    c = collections.Counter(n.op_type for n in g.node)
    print("nodes:", len(g.node)); print("op histogram:", c.most_common())
    for p in m.metadata_props:
        print("meta:", p.key, (p.value[:80] + "...") if len(p.value) > 80 else p.value)
    try:
        mi = shape_inference.infer_shapes(m)
        shapes = {v.name: dims(v) for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
        import numpy as np
        def size(name):
            s = shapes.get(name)
            if not s or any(not isinstance(x, int) or x <= 0 for x in s): return None
            return int(np.prod(s))
        print("first 25 nodes (op, output shape, elements):")
        for n in mi.graph.node[:25]:
            print("  ", n.op_type, shapes.get(n.output[0]), size(n.output[0]))
        sized = [(size(n.output[0]), n.op_type, shapes.get(n.output[0]), idx) for idx, n in enumerate(mi.graph.node) if size(n.output[0])]
        print("largest 8 activations:")
        for s in sorted(sized, reverse=True)[:8]: print("  ", s)
        # conv stats
        convs = [n for n in mi.graph.node if n.op_type == "Conv"]
        ks = collections.Counter()
        for n in convs:
            a = {x.name: x for x in n.attribute}
            k = tuple(a["kernel_shape"].ints) if "kernel_shape" in a else None
            grp = a["group"].i if "group" in a else 1
            ks[(k, "depthwise/grouped" if grp > 1 else "regular")] += 1
        print("conv kinds:", ks.most_common())
        # where do attention-like ops sit
        idxs = [i for i, n in enumerate(mi.graph.node) if n.op_type in ("MatMul", "Softmax", "LayerNormalization", "Gemm", "GRU", "LSTM", "Einsum", "Attention")]
        if idxs: print("matmul/softmax/layernorm/gemm/rnn node index range:", min(idxs), max(idxs), "count", len(idxs))
    except Exception as e:
        print("shape inference failed:", repr(e))
