#!/usr/bin/env python3
"""Small model via the ONNX route on the pinned tinygrad (CPU): kernels/step, GlobalCounters ops+mem per step, peak RSS, median ms.
Usage: DEV="CPU:CLANG:x86_64,x86-64-v3" python -I onnx_small_counters.py driving_supercombo.onnx [steps]"""
import sys, time, statistics, resource, numpy as np
from tinygrad import Tensor, TinyJit, GlobalCounters, Device
from tinygrad.nn.onnx import OnnxRunner
path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
runner = OnnxRunner(path)
specs = {k: (v.shape, v.dtype) for k, v in runner.graph_inputs.items()}
for k, (s, d) in specs.items(): print(f"  input {k} {s} {d}")
rng = np.random.default_rng(0)
def mk(shape, dt):
  if 'u8' in str(dt): return Tensor(rng.integers(0, 255, shape, dtype=np.uint8)).realize()
  return Tensor((rng.standard_normal(shape) * 0.1).astype(np.float32)).realize()
inp = {k: mk(s, d) for k, (s, d) in specs.items()}
@TinyJit
def step(**kw):
  out = runner(kw)
  return tuple(v.contiguous() for v in out.values())
times, ops, mem, krn = [], [], [], []
for t in range(N):
  Device[Device.DEFAULT].synchronize()
  GlobalCounters.reset(); s = time.perf_counter()
  res = step(**inp); [r.realize() for r in res]
  Device[Device.DEFAULT].synchronize(); times.append(time.perf_counter() - s)
  ops.append(GlobalCounters.global_ops); mem.append(GlobalCounters.global_mem); krn.append(GlobalCounters.kernel_count)
  print(f"step {t} {times[-1]*1e3:.1f} ms ops {ops[-1]} mem {mem[-1]} kernels {krn[-1]}", flush=True)
fin = all(np.isfinite(r.numpy().astype(np.float32)).all() for r in res)
calls = len(step.captured._linear.src) if hasattr(step.captured, '_linear') else None
print(f"captured linear call count: {calls}")
print(f"steps measured after 2 warm-up: {N-2}; median ms {statistics.median(times[2:])*1e3:.1f}; ops/step {ops[-1]}; mem/step {mem[-1]}; kernels/step {krn[-1]}; finite {fin}")
print(f"peak RSS MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}")
