"""Variant B's fold, isolated: the stem conv output for one real frame stack against a float64 computation of the ORIGINAL
(normalise, zero-pad the normalised image, conv). Compares the original float32 op sequence, the shipped fold (Pad with 127.5,
then conv with folded weights) and the naive fold (zero-pad the raw image), which is wrong at the border.
Usage: python -I check_stem_fold.py <variants_dir>"""
import os, sys, json
import numpy as np, onnx, torch
import torch.nn.functional as F
from onnx import numpy_helper
vd = sys.argv[1]; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fr = np.load(os.path.join(ROOT, "reference", "driving_reference_fp32.npz"))["frames"][20]
x = np.concatenate([fr[0], fr[0], fr[1], fr[1]], 0)[None].astype(np.float32)
I = lambda p: {i.name: numpy_helper.to_array(i) for i in onnx.load(p).graph.initializer}
nm = "model/vision_model.vision._en.stem.0.reparam_conv"
a, b = I(os.path.join(vd, "base.onnx")), I(os.path.join(vd, "B.onnx"))
w, bias, w2, b2 = a[nm + ".weight"], a[nm + ".bias"], b[nm + ".weight/folded"], b[nm + ".bias/folded"]
xt = torch.tensor(x, dtype=torch.float64)
truth = F.conv2d((xt - 127.5) / 63.75, torch.tensor(w, dtype=torch.float64), torch.tensor(bias, dtype=torch.float64), stride=2, padding=1)
cands = {"original, float32 (cast, sub, div, conv)": F.conv2d((torch.tensor(x) - 127.5) / 63.75, torch.tensor(w), torch.tensor(bias), stride=2, padding=1),
         "B: Pad(127.5) + folded conv, float32": F.conv2d(F.pad(torch.tensor(x), (1, 1, 1, 1), value=127.5), torch.tensor(w2), torch.tensor(b2), stride=2),
         "naive fold with zero padding (NOT used)": F.conv2d(torch.tensor(x), torch.tensor(w2), torch.tensor(b2), stride=2, padding=1)}
print(f"stem conv output, |max| = {truth.abs().max().item():.2f}; max abs error against the float64 original:")
out = {}
for k, v in cands.items():
    d = (v.double() - truth).abs()
    border = max(d[..., 0, :].max().item(), d[..., :, 0].max().item(), d[..., -1, :].max().item(), d[..., :, -1].max().item())
    print(f"  {k:48s} all {d.max().item():.3e}   border rows/cols {border:.3e}")
    out[k] = {"max": d.max().item(), "border": border}
json.dump(out, open(os.path.join(vd, "stem_fold.json"), "w"), indent=1)
