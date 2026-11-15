"""ONNX graph surgery: remove the frame queue from the driving model.

Usage: python -I make_lean.py [in.onnx] [out_dir]
  in.onnx  default models/driving_supercombo.onnx
  out_dir  default models/   -> driving_supercombo_lean.onnx (float16) and driving_supercombo_lean_fp32.onnx

What the original graph does with its two image inputs (checked below from the actual constants, not assumed):
  next_state_img_q = concat(state_img_q[:, 1:], new_img[:, None])           # drop oldest, append newest -> 5 frames
  for each camera c in (0, 1):  cam_c = next_state_img_q[c][0::4]            # frames 0 and 4 of the 5 = 4 steps ago, and now
                                 -> reshape [1, 12, 128, 256]                 # channels 0-5 = old frame, 6-11 = new frame
  x = concat(cam_0, cam_1, axis=1)                                            # [1, 24, 128, 256]
  x = (cast_to_model_dtype(x) - 127.5) / 63.75                                # then the convolutional network
So the network input is a pure function of the newest 5 frames of each camera. The lean model's new input is
`img_stack` [1, 24, 128, 256] float32 holding that concat, in raw 0..255 pixel values. The cast to the model dtype
and the Sub/Div normalisation stay INSIDE the model: pixel values are integers 0..255, which float32 and float16
both represent exactly, so the cast is lossless and the result is bit-for-bit what the original computed. No
normalisation is done by the caller. Everything downstream, and the two small queues, are untouched.
"""
import os, subprocess, sys
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "driving_supercombo.onnx")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models")

m = onnx.load(src); g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
by_out = {o: n for n in g.node for o in n.output}

def const(name): return init[name].tolist()
def attr(n, k): return helper.get_attribute_value(next(a for a in n.attribute if a.name == k))

# ---- 1. walk the graph back from the network's first op and derive what is selected --------------------------
sub = by_out["model/sub"]; cat = by_out[sub.input[0]]
assert sub.op_type == "Sub" and cat.op_type == "Concat" and attr(cat, "axis") == 1, "unexpected normalisation head"
assert np.all(init[sub.input[1]] == 127.5) and np.all(init[by_out["model/div"].input[1]] == 63.75), "unexpected mean/std"
removed = {cat.output[0]}
cams = []
for cast_out in cat.input:                       # concat order = camera order in the stack
    cast = by_out[cast_out]; assert cast.op_type == "Cast"
    rs = by_out[cast.input[0]]; assert rs.op_type == "Reshape" and const(rs.input[1]) == [1, 12, 128, 256]
    sl = by_out[rs.input[0]]; assert sl.op_type == "Slice"
    starts, ends, axes, steps = (const(i) for i in sl.input[1:5])
    assert axes == [0] and starts == [0] and steps == [4], (starts, ends, axes, steps)
    ga = by_out[sl.input[0]]; assert ga.op_type == "Gather" and attr(ga, "axis") == 0
    cams.append(int(init[ga.input[1]]))
    assert ga.input[0] == "next_state_img_q"
    removed |= {cast_out, cast.input[0], sl.input[0], rs.input[0], ga.output[0]}
assert cams == [0, 1], cams
frames = list(range(0, 5, 4)); assert frames == [0, 4]
q = by_out["next_state_img_q"]; assert q.op_type == "Concat" and attr(q, "axis") == 1
sl1 = by_out[q.input[0]]; un = by_out[q.input[1]]
assert sl1.input[0] == "state_img_q" and const(sl1.input[1]) == [1] and const(sl1.input[3]) == [1] and const(sl1.input[4]) == [1]
assert un.input[0] == "new_img" and const(un.input[1]) == [1]
removed |= {"slice_1", "unsqueeze", "next_state_img_q"}
users = {n.output[0] for n in g.node if any(i in removed for i in n.input)} - removed
assert users == {"model/sub"}, users   # nothing else reads the frame queue
print("derived: queue = drop oldest + append new_img (axis 1 of [2,5,6,128,256]); per camera frames", frames,
      "of the 5 (0 = 4 steps ago, 4 = newest); channels = cam0[old 6, new 6], cam1[old 6, new 6]; x = (cast(x)-127.5)/63.75")

# ---- 2. rebuild -------------------------------------------------------------------------------------------------
cast_to = next(attr(by_out[c], "to") for c in cat.input)      # model dtype: 10 (fp16)
nodes = []
for n in g.node:
    if n.output[0] in removed:
        if n.output[0] == cat.output[0]:
            nodes.append(helper.make_node("Cast", ["img_stack"], [cat.output[0]], name="img_stack_cast", to=cast_to))
        continue
    nodes.append(n)
del g.node[:]; g.node.extend(nodes)
keep_in = [i for i in g.input if i.name not in ("new_img", "state_img_q")]
del g.input[:]
g.input.append(helper.make_tensor_value_info("img_stack", TensorProto.FLOAT, [1, 24, 128, 256]))
g.input.extend(keep_in)
keep_out = [o for o in g.output if o.name != "next_state_img_q"]
del g.output[:]; g.output.extend(keep_out)
vi = [v for v in g.value_info if v.name not in removed]
del g.value_info[:]; g.value_info.extend(vi)
used = {i for n in g.node for i in n.input}
unused = [i for i in g.initializer if i.name not in used]
for i in unused: g.initializer.remove(i)
onnx.checker.check_model(m)
os.makedirs(out_dir, exist_ok=True)
p16 = os.path.join(out_dir, "driving_supercombo_lean.onnx"); p32 = os.path.join(out_dir, "driving_supercombo_lean_fp32.onnx")
onnx.save(m, p16)
print("inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("outputs:", [o.name for o in g.output])
print(f"removed {len(removed)} tensors' producers, {len(unused)} unused initializers; saved {p16}")
subprocess.run([sys.executable, "-I", os.path.join(HERE, "make_fp32.py"), p16, p32], check=True)
