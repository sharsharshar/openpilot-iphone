"""Step 5: distribution of tinygrad's static per-launch estimates (ops, lds, mem) in the big and small openpilot pickles.
Uses the allowlisted loader of experiments/big_model (nothing is executed, no weights are read).
Run with the tinygrad commit openpilot pins (the 'tgv' venv of the big_model experiment):
  python -I 06_big_distribution.py big.pkl driving_tinygrad_small.pkl > data/big_small_launches.txt
"""
import sys, os, re, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "big_model"))
from load_safe import load_opcodes, Placeholder
ansi = re.compile(r'\x1b\[[0-9;]*m')
def launches(path):
  j = load_opcodes(path, lambda pid: Placeholder(*pid))
  a = [u for u in j['run'].captured._linear.toposort() if u.op.name == 'CALL' and type(u.arg.aux).__name__ == 'HCQInfo'][0].arg.aux
  return [(ansi.sub('', k[1]), k[2].ops, k[2].lds, k[2].mem) for k in a.kernels]
for path in sys.argv[1:]:
  L = launches(path); tot = sum(l[1] for l in L); ops = sorted((l[1] for l in L), reverse=True)
  print(f"\n### {os.path.basename(path)}: {len(L)} launches, {tot/1e9:.3f} Gop, lds {sum(l[2] for l in L)/1e9:.2f} GB, mem {sum(l[3] for l in L)/1e6:.1f} MB, distinct names {len({l[0] for l in L})}")
  for k in (1, 10, 25, 50, 100, 200): print(f"top {k:3d} launches: {100*sum(ops[:k])/tot:5.1f}% of ops")
  print("launches needed for 50% / 90% / 99% of ops:", *[next(i + 1 for i in range(len(ops)) if sum(ops[:i + 1]) >= f * tot) for f in (.5, .9, .99)])
  print("ops per launch: median %.1f Mop, mean %.1f Mop, max %.1f Mop" % (ops[len(ops)//2]/1e6, tot/len(ops)/1e6, ops[0]/1e6))
  for lo, hi in ((0, 1e4), (1e4, 1e6), (1e6, 1e7), (1e7, 1e8), (1e8, 1e9), (1e9, 1e12)):
    s = [l for l in L if lo <= l[1] < hi]
    print(f"  launches with {lo:.0e}-{hi:.0e} ops: {len(s):4d}  {100*sum(l[1] for l in s)/tot:5.1f}% of ops")
  agg = collections.defaultdict(lambda: [0, 0]);
  for l in L: agg[l[0]][0] += 1; agg[l[0]][1] += l[1]
  print("heaviest kernel names (count, total Gop):")
  for n, (c, o) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:8]: print(f"  {n:46s} x{c:3d}  {o/1e9:7.2f} Gop")
  # arithmetic intensity: ops per byte of loads+stores
  print(f"ops / (lds bytes) overall: {tot/sum(l[2] for l in L):.1f}; launches with ops/lds < 2: {sum(1 for l in L if l[2] and l[1]/l[2] < 2)}")
