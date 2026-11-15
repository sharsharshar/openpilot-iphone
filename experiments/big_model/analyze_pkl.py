#!/usr/bin/env python3
"""Structural analysis of an out-of-band openpilot tinygrad pickle through an allowlisted Unpickler (load_safe.py).
Out-of-band buffers are replaced by (nbytes, offset) placeholders, so no weight data is read and no device is opened.
Run with the tinygrad commit that openpilot pins, e.g.:  python -I analyze_pkl.py big_driving_tinygrad.pkl"""
import sys, re, os, collections, base64, pickle, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_safe import load_opcodes, Placeholder

ansi = re.compile(r'\x1b\[[0-9;]*m')
path = sys.argv[1]
ids = []
def pl(pid): ids.append(pid); return Placeholder(*pid)
j = load_opcodes(path, pl)
print("keys:", list(j))
for k in ('input_specs', 'output_specs'):
  print(k)
  for n, (shape, dt, dev) in j[k].items(): print(f"  {n:22s} {str(shape):22s} {dt:8s} {dev}")
md = j['metadata']['metadata']
print("metadata keys:", list(md), "| model_checkpoint:", md.get('model_checkpoint'))
class SliceOnly(pickle.Unpickler):  # output_slices is a base64 pickle of {name: slice}; builtins.slice only
  def find_class(self, m, n):
    if (m, n) == ('builtins', 'slice'): return slice
    raise pickle.UnpicklingError(f"{m}.{n}")
sl = SliceOnly(io.BytesIO(base64.b64decode(md['output_slices']))).load()
print("output_slices:")
for n, s in sl.items(): print(f"  {n:28s} {s.start}:{s.stop}" + (f"  ({s.stop-s.start})" if s.stop is not None else "  (open-ended)"))

lin = j['run'].captured._linear
topo = list(lin.toposort())
print("uops:", len(topo), dict(collections.Counter(u.op.name for u in topo).most_common(8)))
print("persistent-id buffers:", len(ids), "total bytes", sum(i[0] for i in ids))
bufs = [u for u in topo if u.op.name == 'BUFFER']
by = collections.defaultdict(lambda: [0, 0, 0])
for b in bufs:
  n = b.src[0].arg
  by[str(b.dtype)][0] += 1; by[str(b.dtype)][1] += n; by[str(b.dtype)][2] += n * b.dtype.itemsize
print("BUFFER uops (= out-of-band buffers) by dtype: count, elements, bytes")
for d, (c, e, nb) in by.items(): print(f"  {d:14s} {c:5d} {e:14d} {nb:14d}")
hc = [u for u in topo if u.op.name == 'CALL' and type(u.arg.aux).__name__ == 'HCQInfo']
assert len(hc) == 1
a = hc[0].arg.aux
print("HCQInfo device:", a.device, "| kernel launches:", len(a.kernels), "| estimates:", a.estimates)
names = [ansi.sub('', k[1]) for k in a.kernels]
est = [k[2] for k in a.kernels]
print("sum of per-kernel estimates: ops", sum(e.ops for e in est), "lds", sum(e.lds for e in est), "mem", sum(e.mem for e in est))
print("distinct kernel names:", len(set(names)))
kinds = collections.Counter(n.split('_')[0] for n in names)
print("by prefix (E=elementwise, r=reduce/matmul/conv):", dict(kinds))
ops_by = collections.Counter()
for n, e in zip(names, est): ops_by[n.split('_')[0]] += e.ops
print("ops by prefix:", dict(ops_by))
print("top 15 launches by ops:")
for n, e in sorted(zip(names, est), key=lambda x: -x[1].ops)[:15]: print(f"  {e.ops/1e9:8.3f} Gop  mem {e.mem/1e6:9.2f} MB  {n}")
tot = sum(e.ops for e in est)
srt = sorted((e.ops for e in est), reverse=True)
print("ops in top 10/50 launches:", sum(srt[:10])/tot, sum(srt[:50])/tot)
