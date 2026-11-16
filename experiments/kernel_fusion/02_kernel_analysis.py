"""Step 2: analyse a per-launch CSV written by tg_harness.py --per-kernel --map-weights.

Maps each tinygrad launch to a model stage / layer through the ONNX initializers it reads (matched by buffer contents in the
harness), then prints: top 15 launches, time and ops per stage, per kernel class, efficiency (GOPS) distribution, time by launch
size bucket, and a least-squares fit of time against ops.
Usage: python -I 02_kernel_analysis.py KERNELS.csv [--col seq_median_us] [--launch-us 15]
"""
import argparse, collections, csv, os, re, statistics
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--col", default="seq_min_us", help="time column: t_min_us / t_median_us (launch alone, repeated) or seq_min_us / seq_median_us (in step order)")
ap.add_argument("--launch-us", type=float, default=15.0, help="fixed cost per jitted launch (from 02b_launch_overhead.py)")
ap.add_argument("--out", default=None, help="write the labelled per-launch table here")
a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
for r in rows:
  for k in ("ops", "lds_bytes", "mem_bytes"): r[k] = int(r[k])
  for k in r:
    if k.endswith("_us"): r[k] = float(r[k])
  r["t"] = r[a.col]

def label(r):
  w = r["weights"].replace("model/", "")
  m = re.search(r"_en\.stages\.(\d)\.(downsample\.proj\.\d|blocks\.\d)\.([a-z_.]+?)\.(?:reparam_conv\.|conv\.|)", w)
  if "_en.stem" in w: return "encoder stem", re.search(r"stem\.(\d)", w).group(0)
  if "_en.stages" in w:
    st = re.search(r"stages\.(\d)", w).group(1); blk = re.search(r"(downsample\.proj\.\d|blocks\.\d)", w).group(1)
    part = ("mlp.fc1" if "mlp.fc1" in w else "mlp.fc2+scale+res" if "mlp.fc2" in w else "mlp.dw7x7" if "mlp.conv" in w else "token_mixer" if "token_mixer" in w
            else "downsample")
    return f"encoder stage {st}", f"{blk} {part}"
  if "_en.final_conv" in w or "_en.head" in w: return "encoder final conv+SE+pool+head", w.split("_en.")[1].split(";")[0]
  if "vision_model.vision._mean" in w: return "input normalisation / image prep", "mean,std"
  if "vision_model.policy" in w or "vision_model.no_bottleneck_policy" in w or w.startswith("summarizer") or ";summarizer" in w:
    return "per-frame heads (policy, no_bottleneck, summarizer)", w.split(";")[0].replace("vision_model.", "")
  if "p_policy_model.temporal_summarizer" in w: return "temporal transformer", w.split(";")[0].replace("p_policy_model.temporal_summarizer.", "")
  if "p_policy_model.temporal_hydra" in w: return "temporal heads", w.split(";")[0].replace("p_policy_model.temporal_hydra.", "")
  return "no weights (state/glue/softmax/etc.)", r["name"]

for r in rows:
  r["stage"], r["layer"] = label(r)
  n = r["name"]; w = r["weights"]
  enc = "_en." in w
  if n.startswith("E_"): r["cls"] = "elementwise E_*"
  elif enc and n.endswith("_7_7"): r["cls"] = "conv depthwise 7x7 (+fused epilogue)"
  elif enc and "stem.0" in w: r["cls"] = "conv dense 3x3 (stem.0)"
  elif enc and n.endswith("_3_3"): r["cls"] = "conv depthwise 3x3 (+fused epilogue)"
  elif enc: r["cls"] = "conv 1x1 (fc1, fc2, downsample, stem.2)"
  elif "p_policy_model.temporal_summarizer" in w and ("c_attn" in w or "c_proj" in w or "c_fc" in w or "_feats_encode" in w): r["cls"] = "temporal matmul (token-batched)"
  elif w: r["cls"] = "linear layers, M=1 (matvec)"
  else: r["cls"] = "other reduce r_* (norm/softmax/pool/etc.)"
  r["gops"] = r["ops"] / r["t"] / 1e3 if r["t"] > 0 else 0.0

tot = sum(r["t"] for r in rows) / 1e3; totops = sum(r["ops"] for r in rows)
print(f"{len(rows)} launches, time column {a.col}: sum {tot:.1f} ms, ops {totops/1e9:.3f} G, overall {totops/1e9/(tot/1e3):.1f} GOPS")

print("\n### Top 15 launches by time\n| # | launch | stage / layer | ops (M) | us | share | GOPS |\n|---|---|---|---|---|---|---|")
for i, r in enumerate(sorted(rows, key=lambda r: -r["t"])[:15]):
  print(f"| {i+1} | #{r['i']} {r['name']} | {r['stage']} / {r['layer']} | {r['ops']/1e6:.1f} | {r['t']:.0f} | {100*r['t']/1e3/tot:.1f}% | {r['gops']:.1f} |")

def group(key, title):
  g = collections.OrderedDict()
  for r in rows:
    x = g.setdefault(r[key], [0, 0, 0.0]); x[0] += 1; x[1] += r["ops"]; x[2] += r["t"]
  print(f"\n### {title}\n| {key} | launches | ops (M) | ops share | time ms | time share | GOPS |\n|---|---|---|---|---|---|---|")
  for k, (n, o, t) in sorted(g.items(), key=lambda kv: -kv[1][2]):
    print(f"| {k} | {n} | {o/1e6:.1f} | {100*o/totops:.1f}% | {t/1e3:.2f} | {100*t/1e3/tot:.1f}% | {o/t/1e3:.1f} |")
group("stage", "By stage"); group("cls", "By kernel class")

print("\n### Launch size buckets (time column)\n| launch time | launches | time ms | time share | fixed-cost share if each launch costs "
      f"{a.launch_us:.0f} us extra |\n|---|---|---|---|---|")
for lo, hi in ((0, 10), (10, 50), (50, 200), (200, 1000), (1000, 1e9)):
  s = [r for r in rows if lo <= r["t"] < hi]; t = sum(r["t"] for r in s) / 1e3
  print(f"| {lo}-{hi if hi<1e9 else 'inf'} us | {len(s)} | {t:.2f} | {100*t/tot:.1f}% | {100*len(s)*a.launch_us/1e3/(t+len(s)*a.launch_us/1e3) if s else 0:.0f}% |")
print(f"\nlaunch cost {a.launch_us:.0f} us x {len(rows)} = {len(rows)*a.launch_us/1e3:.2f} ms = {100*len(rows)*a.launch_us/1e3/(tot+len(rows)*a.launch_us/1e3):.1f}% of (kernel time + launch cost)")

# fit time = a + ops/rate, over launches with ops > 0, ordinary least squares and a robust version excluding the matvec class
x = np.array([r["ops"] for r in rows if r["ops"] > 0], float); y = np.array([r["t"] for r in rows if r["ops"] > 0], float)
for name, sel in (("all launches with ops>0", np.ones(len(x), bool)),):
  b, c = np.polyfit(x[sel], y[sel], 1)
  print(f"\nOLS fit, {name}: time = {c:.1f} us + {b*1e3:.3f} ns/op  (= {1/b/1e3:.1f} GOPS asymptotic); R^2 = {1-np.sum((y-(c+b*x))**2)/np.sum((y-y.mean())**2):.3f}")
conv = [r for r in rows if r["cls"].startswith("conv 1x1") or r["cls"].startswith("conv dense")]
xc = np.array([r["ops"] for r in conv], float); yc = np.array([r["t"] for r in conv], float)
b, c = np.polyfit(xc, yc, 1); print(f"OLS fit, dense + 1x1 conv launches only (n={len(conv)}): time = {c:.1f} us + {b*1e3:.3f} ns/op ({1/b/1e3:.1f} GOPS)")
tiny = [r for r in rows if r["ops"] < 3000]
print(f"launches with < 3000 ops (n={len(tiny)}): median time {statistics.median(r['t'] for r in tiny):.2f} us, min {min(r['t'] for r in tiny):.2f} us "
      "(this is the harness cost of one direct call: ctypes + timer, NOT the jitted launch cost)")
g = np.array([r["gops"] for r in rows if r["ops"] > 100000]);
print(f"GOPS over launches with > 0.1 Mop: min {g.min():.1f}, median {np.median(g):.1f}, max {g.max():.1f}")
if a.out:
  with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["i", "name", "stage", "layer", "cls", "ops", "lds_bytes", "mem_bytes", "t_min_us", "t_median_us", "seq_min_us", "seq_median_us", "gops", "weights"], extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
