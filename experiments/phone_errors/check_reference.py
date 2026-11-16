"""Check 1: our own harness reproduces real_reference_fp32.npz (warped frames and model outputs) exactly. Also prints the yardstick."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, numpy as np
from pe_common import *
ref = np.load(f"{SCRATCH}/real_reference_fp32.npz")
t0 = time.time(); fr = baseline_inputs(); print("warped 1200 frames in %.1fs" % (time.time() - t0))
print("warped frames identical to stored frames:", np.array_equal(fr, ref["frames"]), " max abs diff", int(np.abs(fr.astype(int) - ref["frames"].astype(int)).max()))
t0 = time.time(); out = run_model(fr); print("model run %.1fs" % (time.time() - t0))
d = np.abs(out - ref["outputs"]); print("outputs vs stored reference: max abs diff %g, mean %g" % (d.max(), d.mean()))
out2 = run_model(fr); print("second run vs first: max abs diff %g" % np.abs(out2 - out).max())
q = quantities(out); qr = quantities(ref["outputs"]); print("driving quantities identical:", all(np.array_equal(q[k], qr[k]) for k in q))
dv = vs_device(q); print("yardstick (baseline vs device modelV2, warm):", {k: round(v, 5) for k, v in dv.items()})
np.save(f"{OUT}/baseline_outputs.npy", out)
