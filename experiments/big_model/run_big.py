#!/usr/bin/env python3
"""Attempt to load the big pickle with real buffers on a CPU backend, as openpilot's load_oob does (allowlisted unpickler),
then lower/compile and call it. Records exactly where this fails.
Usage: DEV="CPU:CLANG:x86_64,x86-64-v3" python -I run_big.py big_driving_tinygrad.pkl"""
import sys, os, struct, pathlib, traceback, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tinygrad import Tensor, Device
from load_safe import SafeUnpickler
import io
path = sys.argv[1]
print("Device.DEFAULT =", Device.DEFAULT)
with open(path, 'rb') as f:
  n = struct.unpack('<q', f.read(8))[0]; ops = f.read(n); off0 = f.tell()
buffers = Tensor(pathlib.Path(path))[off0:].to(Device.DEFAULT).realize().uop.buffer   # same as helpers.load_pickle
print("weights copied to", Device.DEFAULT, "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
u = SafeUnpickler(io.BytesIO(ops)); u.persistent_load = lambda pid: buffers.view(*pid)
try:
  j = u.load(); print("unpickled OK; input_specs devices:", {v[2] for v in j['input_specs'].values()})
  from tinygrad.engine.realize import lower_and_compile
  lin = lower_and_compile(j['run'].captured._linear); print("lower_and_compile OK")
  j['run'].captured._linear = lin
  ins = {k: Tensor(np.zeros(s, dtype=d), device=Device.DEFAULT).realize() for k, (s, d, _) in j['input_specs'].items()}
  outs = {k: Tensor(np.zeros(s, dtype=d), device=Device.DEFAULT).realize() for k, (s, d, _) in j['output_specs'].items()}
  j['run'](output_buffers=outs, **ins); print("RAN", outs['outputs'].numpy()[:5])
except BaseException:
  traceback.print_exc()
