"""Verify variants with onnxruntime (float32, CPU) against the reference outputs of the original model.

Usage: python -I verify_variants.py <variants_dir> <names,comma,separated|all> [--steps N] [--real <real_reference_fp32.npz>] [--json out.json]

The app side is emulated exactly as the app would do it: a 5-frame history per camera builds `img_stack`; the recurrent state
(desire queue [100,1,8], feature queue [96,1,512]) is carried either by the model (lean interface: next_state_* outputs fed back)
or, for variants D / ALL / ..., by this script (it builds the model's `desire_q` / `feat_hist` inputs from its own rings and
takes the new feature row from outputs[:, 1064:1576]).
Reports max and mean |outputs - reference| over all steps, for the 40-step synthetic reference (includes a desire pulse)
and, if given, the 600-step real-footage reference. Also compares the final carried state with the synthetic reference's.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))


class History:
    """Last 5 model images per camera as uint8; img_stack = [cam0 old, cam0 new, cam1 old, cam1 new] (see scripts/verify_lean.py)."""
    def __init__(self):
        self.q = np.zeros((5, 2, 6, 128, 256), np.uint8)
    def push(self, frame):
        self.q[:-1] = self.q[1:].copy(); self.q[-1] = frame
    def stack(self):
        return np.concatenate([self.q[k, c] for c in (0, 1) for k in (0, 4)], axis=0)       # [24,128,256] uint8


def precompute_stacks(frames):
    h, out = History(), np.empty((len(frames), 24, 128, 256), np.uint8)
    for t in range(len(frames)):
        h.push(frames[t]); out[t] = h.stack()
    return out


def run_variant(path, ref, stacks, steps, label, predict=None, in_shapes=None, outs=None):
    """predict(feed) -> list of output arrays in graph-output order; default: onnxruntime float32 CPU on `path`."""
    if predict is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        in_shapes = {i.name: i.shape for i in sess.get_inputs()}
        outs = [o.name for o in sess.get_outputs()]
        predict = lambda feed: sess.run(outs, feed)
    lean_iface = "state_desire_q" in in_shapes
    dq = np.zeros((100, 1, 8), np.float32); fq = np.zeros((96, 1, 512), np.float32)
    diffs, means = [], []
    t0 = time.time()
    for t in range(steps):
        feed = {"img_stack": stacks[t][None].astype(np.float32), "traffic_convention": ref["traffic_convention"][t], "action_t": ref["action_t"][t]}
        desire = ref["desire"][t].astype(np.float32)
        if lean_iface:
            feed.update(desire=desire, state_desire_q=dq, state_feat_q=fq)
        else:
            nxt = np.concatenate([dq[1:], desire.reshape(1, 1, 8)], 0)                       # the queue after this step's desire is appended
            if in_shapes["desire_q"] == [1, 25, 4, 8]:
                feed["desire_q"] = nxt.reshape(1, 25, 4, 8)
            else:
                assert in_shapes["desire_q"] == [1, 200, 4, 1]
                feed["desire_q"] = np.ascontiguousarray(nxt.reshape(25, 4, 8).transpose(0, 2, 1)).reshape(1, 200, 4, 1)   # (g, c, k)
            rows = fq[64:96:4, 0, :]                                                         # [8,512]
            if in_shapes["feat_hist"] == [1, 8, 512]:
                feed["feat_hist"] = rows.reshape(1, 8, 512)
            else:
                assert in_shapes["feat_hist"] == [1, 512, 1, 8]
                feed["feat_hist"] = np.ascontiguousarray(rows.T).reshape(1, 512, 1, 8)
        res = dict(zip(outs, predict(feed)))
        o = res["outputs"]
        if lean_iface:
            dq, fq = res["next_state_desire_q"], res["next_state_feat_q"]
        else:
            dq = nxt
            fq = np.concatenate([fq[1:], o[:, 1064:1576].reshape(1, 1, 512)], 0)
        d = np.abs(o[0] - ref["outputs"][t])
        diffs.append(float(d.max())); means.append(float(d.mean()))
    d = np.array(diffs)
    r = {"steps": steps, "max_abs": float(d.max()), "mean_abs": float(np.mean(means)), "worst_step": int(d.argmax()),
         "seconds": round(time.time() - t0, 1), "per_step_max": [float(f"{x:.3e}") for x in d]}
    if "final_state_desire_q" in ref.files and steps == len(ref["outputs"]):
        r["final_desire_q_maxdiff"] = float(np.abs(dq - ref["final_state_desire_q"]).max())
        r["final_feat_q_maxdiff"] = float(np.abs(fq - ref["final_state_feat_q"]).max())
    print(f"  {label:10s} steps {steps:4d}  max|d| {r['max_abs']:.3e} (step {r['worst_step']})  mean|d| {r['mean_abs']:.3e}  {r['seconds']}s", flush=True)
    return r


def main():
    a = [x for x in sys.argv[1:]]
    def opt(k, default=None):
        if k in a:
            i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
        return default
    steps_lim = opt("--steps"); real = opt("--real"); jout = opt("--json")
    vdir, names = a[0], a[1]
    names = sorted(f[:-5] for f in os.listdir(vdir) if f.endswith(".onnx")) if names == "all" else names.split(",")
    syn = np.load(os.path.join(ROOT, "reference", "driving_reference_fp32.npz"))
    syn = {k: syn[k] for k in syn.files}
    class D(dict):
        files = property(lambda s: list(s.keys()))
    syn = D(syn)
    syn_stacks = precompute_stacks(syn["frames"])
    data = [("synthetic", syn, syn_stacks)]
    if real:
        t0 = time.time(); rr = np.load(real); rr = D({k: rr[k] for k in ("frames", "desire", "traffic_convention", "action_t", "outputs")})
        data.append(("real", rr, precompute_stacks(rr["frames"]))); del rr["frames"]
        print(f"loaded real reference ({len(rr['outputs'])} steps) in {time.time() - t0:.0f}s")
    results = json.load(open(jout)) if jout and os.path.exists(jout) else {}
    for nm in names:
        path = nm if nm.endswith(".onnx") else os.path.join(vdir, nm + ".onnx")
        print(nm, flush=True)
        results[nm] = {}
        for label, ref, stacks in data:
            n = len(ref["outputs"]) if steps_lim is None else min(int(steps_lim), len(ref["outputs"]))
            results[nm][label] = run_variant(path, ref, stacks, n, label)
        if jout:
            json.dump(results, open(jout, "w"), indent=1)

if __name__ == "__main__":
    main()
