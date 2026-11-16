"""Run one of the existing experiment scripts unchanged on a large npz.
Those scripts index ref["frames"][t] inside their step loop; on an npz that re-reads (and decompresses) the whole
frames array every step (1.2 GB x steps). This wrapper makes np.load return a plain dict of in-memory arrays and
then runs the script as __main__ with its normal arguments.
Usage: python -I run_existing.py <script.py> <model.onnx> <reference.npz>
"""
import sys, runpy
import numpy as np
_load = np.load
def load(path, *a, **k):
    z = _load(path, *a, **k)
    return {key: z[key] for key in z.files} if hasattr(z, "files") else z
np.load = load
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
