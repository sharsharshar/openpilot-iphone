"""Aggregate sweep JSONL files (from 03_sweep.sh / 03b_variants.sh) into a markdown table: per label, minimum and median over passes."""
import json, sys, collections
rows = collections.OrderedDict()
for f in sys.argv[1:]:
  for l in open(f):
    d = json.loads(l); rows.setdefault(d["label"], []).append(d)
print("| config | kernels | distinct | ops G | worst max diff | worst mean diff | step CPU ms (min of runs / median of runs) | step wall ms (min / median) | runs |\n|---|---|---|---|---|---|---|---|---|")
for k, v in rows.items():
  d = v[0]
  cmin = min(x["step_cpu_ms_min"] for x in v); cmed = sorted(x["step_cpu_ms_median"] for x in v)[len(v) // 2]
  wmin = min(x["step_ms_min"] for x in v); wmed = sorted(x["step_ms_median"] for x in v)[len(v) // 2]
  bad = " **WRONG OUTPUT**" if d["worst_max"] > 0.01 else ""
  print(f"| {k}{bad} | {d['kernels']} | {d['distinct']} | {d['ops_G']:.3f} | {max(x['worst_max'] for x in v):.2g} | {max(x['worst_mean'] for x in v):.2g} | {cmin:.0f} / {cmed:.0f} | {wmin:.0f} / {wmed:.0f} | {len(v)} |")
