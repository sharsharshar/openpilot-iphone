"""Build the ANE-oriented variants of the lean driving model as ONNX graphs.

Usage: python -I make_variants.py [lean_fp32.onnx] [out_dir]
Writes <out_dir>/<name>.onnx for every variant in VARIANTS (default models/variants/) and variants.json (interface + rewrite log).
"""
import json, os, sys
import onnx

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gb import GB
import rewrites as R


def build(name, gb):
    """Apply the rewrites of variant `name` (a GB on the lean graph)."""
    if name == "base":
        return gb
    if name == "A":
        return R.rw_A(gb)
    if name == "B":
        return R.rw_B(gb)
    if name == "C":                      # fully connected layers + transformer as 1x1 convs on channel-first tensors
        R.rw_tokens_from_chain_cf(gb)
        R.rw_transformer(gb, prune=False, cf=True)
        R.rw_C_fc(gb)
        return R.rw_cleanup_reshapes(gb)
    if name == "D":                      # queues in the app (time-major layouts)
        return R.rw_D(gb, cf=False)
    if name in ("E", "E_fold"):          # other exact simplifications (token-major); E_fold adds the two weight folds
        R.rw_E_simple(gb, folds=(name == "E_fold"))
        R.rw_tokens_shortcut(gb, cf=False)
        return R.rw_transformer(gb, prune=True, cf=False)
    if name in ("ALL_DROPIN", "ALL_DROPIN_noB", "ALL", "ALL_noB"):
        # ALL_DROPIN = A+B+C+E (same interface as the lean package); ALL = A+B+C+D+E (queues in the app, channel-first inputs);
        # *_noB leave the input normalisation alone, so that every weight stays an exact float16 number (see fp16_exactness.py)
        d = name.startswith("ALL") and not name.startswith("ALL_DROPIN")
        R.rw_A(gb)
        if not name.endswith("_noB"):
            R.rw_B(gb)
        if d:
            R.rw_D(gb, cf=True)
        R.rw_E_simple(gb, folds=False)
        if not d:
            R.rw_tokens_shortcut(gb, cf=True)
        R.rw_transformer(gb, prune=True, cf=True)
        R.rw_C_fc(gb)
        return R.rw_cleanup_reshapes(gb)
    raise KeyError(name)

VARIANTS = ["base", "A", "B", "C", "D", "E", "E_fold", "ALL_DROPIN_noB", "ALL_DROPIN", "ALL_noB", "ALL"]    # base = the lean model, copied unchanged


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "driving_supercombo_lean_fp32.onnx")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models", "variants")
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else VARIANTS
    os.makedirs(out, exist_ok=True)
    model = onnx.load(src)
    meta = {}
    for name in only:
        if name == "base":
            import shutil
            shutil.copyfile(src, os.path.join(out, "base.onnx"))
            m = model
            meta["base"] = {"nodes": len(m.graph.node), "ops": {}, "log": ["unchanged copy of the lean model"],
                            "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
                            "outputs": [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}
            print(f"base: copy of {src}")
            continue
        gb = build(name, GB(model))
        m = gb.finalize()
        path = os.path.join(out, f"{name}.onnx")
        onnx.save(m, path)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        meta[name] = {"nodes": len(m.graph.node), "ops": ops, "log": gb.log,
                      "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
                      "outputs": [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}
        print(f"{name}: {len(m.graph.node)} ONNX nodes -> {path}")
    j = os.path.join(out, "variants.json")
    old = json.load(open(j)) if os.path.exists(j) else {}
    old.update(meta)
    json.dump(old, open(j, "w"), indent=1)

if __name__ == "__main__":
    main()
