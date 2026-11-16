"""ONNX variant -> Core ML package with the project's existing route (onnx2torch + shim -> torch.jit.trace -> coremltools mlprogram,
iOS18, float16 compute), then count the MIL operations.

Usage: python -I convert_variants.py <variants_dir> <names,...|all> [--io16] [--json out.json]
  --io16  declare float16 inputs instead of float32 (writes <name>_io16.mlpackage). That is an interface change.
          (Outputs are float16 in every package, the shipped one included: coremltools' default for float16 compute.)
Also usable on the shipped package: python -I convert_variants.py --count <package.mlpackage>
Runs on Linux (conversion works there; prediction does not).
"""
import collections, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)                                  # shim_extra imports scripts/onnx2torch_shim.py (not modified)
import numpy as np


def describe(spec):
    import coremltools as ct
    AT = ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType
    io = lambda f: [(x.name, AT.Name(x.type.multiArrayType.dataType), list(x.type.multiArrayType.shape)) for x in f]
    ops = collections.Counter(op.type for fn in spec.mlProgram.functions.values() for blk in fn.block_specializations.values() for op in blk.operations)
    nc = sum(v for k, v in ops.items() if k != "const")
    return {"inputs": io(spec.description.input), "outputs": io(spec.description.output), "ops": dict(ops), "total_ops": sum(ops.values()), "non_const_ops": nc}


def convert(onnx_path, out_path, io16=False):
    import onnx, torch
    import shim_extra  # (scripts/onnx2torch_shim.py + converters for Pad and rsqrt)
    from onnx2torch import convert as o2t
    import coremltools as ct
    m = onnx.load(onnx_path)
    nf = shim_extra.fuse_rsqrt(m)
    if nf:
        print(f"   fused {nf} Sqrt+Reciprocal pairs into rsqrt for tracing")
    tm = o2t(m).eval()
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    ex = tuple(torch.zeros(*s) for _, s in ins)
    with torch.no_grad():
        traced = torch.jit.trace(tm, ex, check_trace=False)
    # the torch module that coremltools will convert, run in float32 against the reference for a few steps (checks the onnx2torch
    # step, the Pad / rsqrt converters and the tracing; it does NOT check coremltools' MIL program, which cannot run on Linux)
    import verify_variants as V
    ref = np.load(os.path.join(ROOT, "reference", "driving_reference_fp32.npz"))
    ref = {k: ref[k] for k in ("frames", "desire", "traffic_convention", "action_t", "outputs")}
    ref["files"] = list(ref)
    class _R(dict):
        files = property(lambda s: [])
    ref = _R(ref)
    nsteps = 3
    stacks = V.precompute_stacks(ref["frames"][:nsteps])
    names = [n for n, _ in ins]
    def tpred(feed):
        with torch.no_grad():
            o = tm(*[torch.tensor(feed[n]) for n in names])
        return [t.numpy() for t in o] if isinstance(o, (tuple, list)) else [o.numpy()]
    tcheck = V.run_variant(None, ref, stacks, nsteps, "torch-f32", predict=tpred, in_shapes={n: s for n, s in ins}, outs=[o.name for o in m.graph.output])
    dt = np.float16 if io16 else np.float32
    t0 = time.time()
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=tuple(s), dtype=dt) for n, s in ins],
                    outputs=[ct.TensorType(name=o.name) for o in m.graph.output],     # as scripts/try_coreml.py: output dtype left to coremltools (float16)
                    convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
    ml.save(out_path)
    r = describe(ml.get_spec())
    r["convert_seconds"] = round(time.time() - t0, 1)
    r["torch_module_float32_check"] = {"steps": nsteps, "max_abs_vs_reference": tcheck["max_abs"]}
    return r


def main():
    a = sys.argv[1:]
    if a and a[0] == "--count":
        import coremltools as ct
        r = describe(ct.models.MLModel(a[1], skip_model_load=True).get_spec())
        print(json.dumps(r, indent=1)); return
    io16 = "--io16" in a
    jout = a[a.index("--json") + 1] if "--json" in a else None
    a = [x for i, x in enumerate(a) if x not in ("--io16", "--json") and (i == 0 or a[i - 1] != "--json")]
    vdir, names = a[0], a[1]
    names = sorted(f[:-5] for f in os.listdir(vdir) if f.endswith(".onnx")) if names == "all" else names.split(",")
    res = json.load(open(jout)) if jout and os.path.exists(jout) else {}
    for nm in names:
        key = nm + ("_io16" if io16 else "")
        out = os.path.join(vdir, key + ".mlpackage")
        print(f"== {key}", flush=True)
        try:
            r = convert(os.path.join(vdir, nm + ".onnx"), out, io16)
            res[key] = r
            print(f"   ok: {r['total_ops']} ops ({r['non_const_ops']} excluding const), {r['convert_seconds']}s", flush=True)
            print("   ", sorted(((k, v) for k, v in r["ops"].items() if k != "const"), key=lambda x: -x[1]), flush=True)
        except Exception as e:
            import traceback
            res[key] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            print("   FAILED", type(e).__name__, str(e)[:500]); traceback.print_exc(limit=3)
        if jout:
            json.dump(res, open(jout, "w"), indent=1)

if __name__ == "__main__":
    main()
