"""ANEForge static checks of every variant (importer + preflight for each family it knows). Runs on Linux; never touches an ANE.

Usage: python -I aneforge_check.py <ANEForge clone> <variants_dir> <names,...|all> [--json out.json]

For each ONNX variant and for EACH graph output separately (the importer reads only graph.output[0]; a copy of the model with
just that output is imported): the importer's exception if any; for the first output also `preflight` for the families ANEForge
offers (A13, A14, A15, A16; its docs fold every A17/A18 chip into A16 and know no A19), with native / decompose / reject / oversize
counts and the names of any flagged op, and the op histogram of ANEForge's own MIL lowering (no compile).
Preflight is a table lookup plus dimension caps, not a compiler run: "native" means "not in the floor table and within the caps".
"""
import copy, json, os, re, sys, warnings
from collections import Counter

AF_DIR, VDIR, NAMES = sys.argv[1], sys.argv[2], sys.argv[3]
JOUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
tmp = os.path.join(os.path.dirname(os.path.abspath(JOUT)) if JOUT else ".", "aneforge-cache")
os.environ.setdefault("ANEFORGE_CACHE_DIR", os.path.abspath(tmp))
sys.path.insert(0, os.path.abspath(AF_DIR))
import onnx
from onnx import shape_inference
import aneforge as af
from aneforge import _targets as tg
from aneforge import onnx as afonnx
from aneforge.graph import input as af_input, Tensor


def prune_to(m, out_name):
    """Copy of model m keeping only the nodes the given output depends on, and only that output."""
    mm = onnx.ModelProto(); mm.CopyFrom(m)
    prod = {o: n for n in mm.graph.node for o in n.output}
    need, stack = set(), [out_name]
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is not None and id(n) not in need:
            need.add(id(n)); stack.extend(x for x in n.input if x)
    keep = [n for n in mm.graph.node if id(n) in need]
    del mm.graph.node[:]; mm.graph.node.extend(keep)
    o = [x for x in mm.graph.output if x.name == out_name][0]
    del mm.graph.output[:]; mm.graph.output.append(o)
    return mm


def failing_nodes(m):
    """Each node imported on its own (fresh inputs of the inferred static shape), as scripts in experiments/aneforge/run_preflight.py:
    returns [(index, op, node name, error)] for the nodes the importer rejects."""
    mi = shape_inference.infer_shapes(m); g = mi.graph
    shp = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in list(g.value_info) + list(g.input) + list(g.output)}
    inits = afonnx._inits(g)
    bad = []
    for idx, n in enumerate(g.node):
        vals = dict(inits)
        for nm in n.input:
            if nm and nm not in inits:
                vals[nm] = af_input(shp[nm])
        try:
            if n.op_type not in afonnx._ONNX:
                raise NotImplementedError(f"ONNX op '{n.op_type}' not supported")
            ins = [vals.get(x) if x else None for x in n.input]
            afonnx._ONNX[n.op_type](n, ins, afonnx._attrs(n), inits)
        except Exception as e:
            bad.append((idx, n.op_type, n.name, f"{type(e).__name__}: {str(e)[:120]}"))
    return bad

FAMS = {n: getattr(tg.Family, n) for n in ("A13", "A14", "A15", "A16")}
names = sorted(f[:-5] for f in os.listdir(VDIR) if f.endswith(".onnx")) if NAMES == "all" else NAMES.split(",")
res = json.load(open(JOUT)) if JOUT and os.path.exists(JOUT) else {}
print("ANEForge", af.__version__, "families offered:", list(FAMS))
for nm in names:
    m = onnx.load(os.path.join(VDIR, nm + ".onnx"))
    init_names = {i.name for i in m.graph.initializer}
    r = {"onnx_nodes": len(m.graph.node), "outputs": {},
         # static scan for the node kinds that stopped the importer on the lean model (independent of which error is raised first)
         "known_blockers": {"GatherND": sum(n.op_type == "GatherND" for n in m.graph.node),
                            "Where": sum(n.op_type == "Where" for n in m.graph.node),
                            "Concat_with_constant_operand": sum(n.op_type == "Concat" and any(x in init_names for x in n.input) for n in m.graph.node)}}
    print(nm, "known blockers:", r["known_blockers"])
    for k, o in enumerate(m.graph.output):
        mm = prune_to(m, o.name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ins, out = af.onnx_to_tensor(mm)
            ent = {"import": "ok"}
            if k == 0:
                for fn, f in FAMS.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rep = tg.preflight(out, f)
                    ent[fn] = {"native": len(rep.native), "decompose": len(rep.decompose), "reject": len(rep.reject), "oversize": len(rep.oversize),
                               "flagged_ops": sorted({x.op for x in rep.decompose + rep.reject + rep.oversize})}
                try:
                    from aneforge import _compile as C
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        d = C._lower_fused_to_dir(out, None, int8=False)
                    mil = open(os.path.join(str(d), "model.mil")).read()
                    h = Counter(x for x in re.findall(r"=\s+([a-z_0-9]+)\(", mil) if x not in ("string", "uint64", "bool", "fp16", "int32"))
                    ent["aneforge_mil_ops"] = {"total": sum(h.values()), "hist": dict(h.most_common())}
                except Exception as e:
                    ent["aneforge_mil_ops"] = f"{type(e).__name__}: {str(e)[:200]}"
        except Exception as e:
            ent = {"import": f"{type(e).__name__}: {str(e)[:300]}"}
            try:
                ent["failing_nodes"] = failing_nodes(mm)
            except Exception as e2:
                ent["failing_nodes"] = f"per-node scan failed: {type(e2).__name__}: {str(e2)[:150]}"
        r["outputs"][o.name] = ent
        line = f"{nm:11s} output {o.name:22s} import {ent['import'][:110]}"
        if "A16" in ent:
            line += "  " + "  ".join(f"{fn}: native {ent[fn]['native']} rej {ent[fn]['reject']} dec {ent[fn]['decompose']} over {ent[fn]['oversize']}" for fn in FAMS)
        print(line, flush=True)
        for fnode in ent.get("failing_nodes", []) if isinstance(ent.get("failing_nodes"), list) else []:
            print("      not importable alone:", fnode, flush=True)
    res[nm] = r
    if JOUT:
        json.dump(res, open(JOUT, "w"), indent=1)
