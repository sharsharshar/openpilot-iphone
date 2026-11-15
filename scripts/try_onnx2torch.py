"""Step 1 of the PyTorch route: can onnx2torch rebuild the ONNX graph as a torch module,
and does that module reproduce the reference outputs?
Usage: python -I try_onnx2torch.py <model.onnx> <reference.npz>"""
import sys, collections, traceback
import numpy as np, onnx, torch
import onnx2torch_shim  # noqa: fills opset-20 gaps
from onnx2torch import convert
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY as REG

m = onnx.load(sys.argv[1])
opset = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0]
ops = collections.Counter(n.op_type for n in m.graph.node)
try:
    tm = convert(m).eval()
    print("onnx2torch convert: OK")
except Exception as e:
    print("onnx2torch convert FAILED:", type(e).__name__, str(e)[:400]); sys.exit(1)

ref = np.load(sys.argv[2])
names = [i.name for i in m.graph.input if i.name not in {x.name for x in m.graph.initializer}]
print("input order:", names)
state = {"state_img_q": torch.zeros(2,5,6,128,256, dtype=torch.uint8), "state_desire_q": torch.zeros(100,1,8), "state_feat_q": torch.zeros(96,1,512)}
worst = 0.0
try:
    with torch.no_grad():
        for t in range(len(ref["frames"])):
            feed = {"new_img": torch.from_numpy(ref["frames"][t]), "desire": torch.from_numpy(ref["desire"][t]),
                    "traffic_convention": torch.from_numpy(ref["traffic_convention"][t]), "action_t": torch.from_numpy(ref["action_t"][t]), **state}
            out = tm(*[feed[n] for n in names])
            o, state = out[0], {"state_img_q": out[1], "state_desire_q": out[2], "state_feat_q": out[3]}
            worst = max(worst, float(np.abs(o.float().numpy()[0] - ref["outputs"][t]).max()))
    print(f"torch module ran {len(ref['frames'])} steps; max abs diff vs ONNX reference: {worst:.5f}")
    torch.save(tm, sys.argv[1].replace(".onnx", "_torch.pt"))
except Exception as e:
    traceback.print_exc(limit=3); print("torch run FAILED:", type(e).__name__, str(e)[:400])
