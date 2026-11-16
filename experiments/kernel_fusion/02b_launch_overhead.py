"""Fixed cost of one kernel launch inside a TinyJit replay (tinygrad 0.14.0).

Builds a jitted function that runs K tiny, independent kernels (each one a separate output buffer of 16 floats; one ADD with a
distinct constant per kernel so none can be merged), replays it, and fits  time = a + b*K  over several K. b is the cost per
launch when the arithmetic is ~zero. Also reports how many kernels the jit actually contains, so a fused result would show.
Usage: DEV=... taskset -c 3 python -I 02b_launch_overhead.py   (any device; on Metal it measures command-buffer encode + dispatch)
"""
import statistics, time, sys, os
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.uop.ops import Ops

dev = Device.DEFAULT
print(f"device={dev} DEV={os.environ.get('DEV','')}")
def build(K):
  @TinyJit
  def f(x):
    outs = [(x + float(i + 1)).contiguous() for i in range(K)]   # K independent elementwise kernels of 16 elements
    return tuple(outs)
  return f

xs = Tensor(np.arange(16, dtype=np.float32)).realize()
res = []
for K in (10, 50, 100, 200, 400):
  f = build(K)
  for _ in range(3): [o.realize() for o in f(xs)]
  nk = sum(1 for c in f.captured.linear.src if c.src[0].op is Ops.PROGRAM)
  ts = []
  for _ in range(60):
    Device[dev].synchronize(); t = time.perf_counter(); o = f(xs); Device[dev].synchronize(); ts.append(time.perf_counter() - t)
  res.append((K, nk, statistics.median(ts) * 1e6, min(ts) * 1e6))
  print(f"K={K:4d} kernels in jit={nk:4d}  step median {res[-1][2]:9.1f} us  min {res[-1][3]:9.1f} us  per kernel (median) {res[-1][2]/max(nk,1):7.2f} us", flush=True)
x = np.array([r[1] for r in res], float)
for idx, label in ((2, "median"), (3, "min")):
  y = np.array([r[idx] for r in res]); b, a = np.polyfit(x, y, 1)
  print(f"fit on {label}: fixed {a:.1f} us + {b:.2f} us per launch")
