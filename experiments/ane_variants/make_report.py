"""Collect verify.json, coreml.json, aneforge.json (all in the variants dir) into one markdown table: summary.md.
Usage: python -I make_report.py <variants_dir>"""
import json, os, sys
vd = sys.argv[1]
L = lambda n: json.load(open(os.path.join(vd, n))) if os.path.exists(os.path.join(vd, n)) else {}
ver, cml, af = L("verify.json"), L("coreml.json"), L("aneforge.json")
order = [k for k in ["base", "A", "B", "C", "D", "E", "E_fold", "ALL_DROPIN_noB", "ALL_DROPIN", "ALL_noB", "ALL"] if k in cml or k in ver]
b = cml.get("base", {})

def iface(r):
    return (sorted((n, t, tuple(s)) for n, t, s in r["inputs"]), sorted((n, t, tuple(s)) for n, t, s in r["outputs"]))

rows = ["| variant | max diff synthetic (40 steps) | max diff real (600 steps) | MIL ops total / without const | vs lean | cast | ANEForge A13/A14/A15/A16 native, rejected | drop-in for the app |", "|---|---|---|---|---|---|---|---|"]
for k in order:
    v, c, a = ver.get(k, {}), cml.get(k, {}), af.get(k, {})
    f = lambda d: f"{d['max_abs']:.2e}" if d else "-"
    if "ops" in c:
        tot = f"{c['total_ops']} / {c['non_const_ops']}"
        vs = f"{c['non_const_ops'] - b['non_const_ops']:+d}" if b else "-"
        cast = c["ops"].get("cast", 0)
        drop = "yes" if iface(c) == iface(b) else "no"
    else:
        tot = vs = cast = "-"; drop = "?"
    if a:
        o = a["outputs"].get("outputs", {})
        if o.get("import") == "ok":
            ane = " / ".join(f"{o[x]['native']}, {o[x]['reject']}" for x in ("A13", "A14", "A15", "A16") if x in o)
            allout = all(e["import"] == "ok" for e in a["outputs"].values())
            ane = ane + ("" if allout else " (import fails for another output)")
        else:
            ane = "import fails: " + o.get("import", "?")[:60]
    else:
        ane = "-"
    rows.append(f"| {k} | {f(v.get('synthetic'))} | {f(v.get('real'))} | {tot} | {vs} | {cast} | {ane} | {drop} |")
txt = "\n".join(rows) + "\n"
open(os.path.join(vd, "summary.md"), "w").write(txt)
print(txt)
