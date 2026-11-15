"""Step 2 of the PyTorch route: trace the rebuilt torch module and hand it to coremltools.
Usage: python try_coreml.py <model_fp32.onnx> <out.mlpackage> [uint8|float]"""
import sys, warnings, traceback, collections
warnings.filterwarnings("ignore")
import numpy as np, onnx, torch
import onnx2torch_shim  # noqa
from onnx2torch import convert
import coremltools as ct

img_mode = sys.argv[3] if len(sys.argv) > 3 else "uint8"
tm = convert(onnx.load(sys.argv[1])).eval()
img_t = torch.uint8 if img_mode == "uint8" else torch.float32
ex = (torch.zeros(2,6,128,256, dtype=img_t), torch.zeros(8), torch.zeros(1,2), torch.zeros(1,2),
      torch.zeros(2,5,6,128,256, dtype=img_t), torch.zeros(100,1,8), torch.zeros(96,1,512))
names = ["new_img", "desire", "traffic_convention", "action_t", "state_img_q", "state_desire_q", "state_feat_q"]
with torch.no_grad():
    traced = torch.jit.trace(tm, ex, check_trace=False)
print("torch.jit.trace: OK")
dt = {"uint8": None, "float": np.float32}
inputs = [ct.TensorType(name=n, shape=tuple(e.shape), dtype=(np.float32 if e.dtype != torch.uint8 else (np.float32 if img_mode == "float" else np.uint8)))
          for n, e in zip(names, ex)]
try:
    out_names = ["outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q"]
    import os
    pipeline = ct.PassPipeline.DEFAULT
    if os.environ.get("GELU_EXPLICIT") == "1":   # stop coremltools from recognising the formula and fusing it back into a gelu op
        pipeline.remove_passes({"common::fuse_gelu_tanh_approximation", "common::fuse_gelu_exact"})
    ml = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=n) for n in out_names], pass_pipeline=pipeline, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                    compute_precision=ct.precision.FLOAT16)
    ml.save(sys.argv[2]); print("coremltools convert: OK ->", sys.argv[2])
    spec = ml.get_spec()
    print("inputs:", [(i.name, ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(i.type.multiArrayType.dataType), list(i.type.multiArrayType.shape)) for i in spec.description.input])
    print("outputs:", [(o.name, ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(o.type.multiArrayType.dataType), list(o.type.multiArrayType.shape)) for o in spec.description.output])
    ops = collections.Counter(op.type for fn in spec.mlProgram.functions.values() for blk in fn.block_specializations.values() for op in blk.operations)
    print("Core ML ops:", sorted(ops.items(), key=lambda x: -x[1]))
except Exception as e:
    tb = traceback.format_exc().strip().splitlines()
    print("coremltools convert FAILED:", type(e).__name__); print("  ", str(e)[:700]); print("  at:", [l.strip() for l in tb if "coremltools" in l][-3:])
