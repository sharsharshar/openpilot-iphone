"""Run the original ONNX driving model on CPU and save inputs + outputs as a reference.

Usage: python3 -I reference_run.py <model.onnx> <out.npz> [steps]

Inputs are synthetic (seeded), so this checks "does a port compute the same numbers",
not "does it drive well". State queues are carried from one run to the next.
"""
import sys, time
import numpy as np
import onnxruntime as ort

model_path, out_path = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40

so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
NP = {"tensor(uint8)": np.uint8, "tensor(float)": np.float32, "tensor(float16)": np.float16}
ins = {i.name: (tuple(i.shape), NP[i.type]) for i in sess.get_inputs()}
outs = [o.name for o in sess.get_outputs()]
state_names = [n for n in ins if f"next_{n}" in outs]
print("inputs:", ins); print("outputs:", outs); print("state:", state_names)

def make_inputs(seed, steps):
    """A moving synthetic scene: smooth gradients that drift, plus a little noise."""
    rng = np.random.default_rng(seed)
    frames = []
    yy, xx = np.mgrid[0:128, 0:256].astype(np.float32)
    for t in range(steps):
        base = 128 + 60 * np.sin((xx + 3 * t) / 23.0) + 40 * np.cos((yy - 2 * t) / 17.0)
        f = np.stack([[base + 10 * c + 5 * cam for c in range(6)] for cam in range(2)])
        f += rng.normal(0, 4, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    desire = np.zeros((steps, 8), np.float32)
    desire[steps // 2, 3] = 1.0                      # one desire pulse mid-sequence
    tc = np.tile(np.array([[1, 0]], np.float32), (steps, 1, 1))
    at = np.tile(np.array([[0.2, 0.5]], np.float32), (steps, 1, 1))
    return np.stack(frames), desire, tc, at

def run(frames, desire, tc, at, carry_state=True):
    state = {n: np.zeros(ins[n][0], ins[n][1]) for n in state_names}
    out_all, times = [], []
    for t in range(len(frames)):
        feed = {"new_img": frames[t], "desire": desire[t], "traffic_convention": tc[t], "action_t": at[t], **state}
        t0 = time.perf_counter()
        res = dict(zip(outs, sess.run(outs, feed)))
        times.append(time.perf_counter() - t0)
        out_all.append(res["outputs"][0].astype(np.float32))
        if carry_state:
            state = {n: res[f"next_{n}"] for n in state_names}
    return np.stack(out_all), state, np.array(times)

frames, desire, tc, at = make_inputs(0, steps)
ref, final_state, times = run(frames, desire, tc, at, carry_state=True)
print(f"ran {steps} steps; output shape {ref.shape}; all finite: {np.isfinite(ref).all()}")
print(f"cpu time per run: median {np.median(times[3:])*1e3:.1f} ms, max {times[3:].max()*1e3:.1f} ms (4 threads)")

# determinism: same inputs twice must give identical outputs
ref2, _, _ = run(frames, desire, tc, at, carry_state=True)
print("deterministic (max abs diff between two identical runs):", float(np.abs(ref - ref2).max()))

# check the image queue really is "drop oldest, append newest"
q = final_state["state_img_q"]
print("image queue == last 5 frames fed in:", bool((q == np.stack(frames[-5:], axis=1)).all()))

# the silent failure: forget to carry state (zeros every run)
nostate, _, _ = run(frames, desire, tc, at, carry_state=False)
d = np.abs(ref - nostate)
print(f"state forgotten: all finite {np.isfinite(nostate).all()}, "
      f"max abs diff vs reference at last step {d[-1].max():.3f}, mean abs diff {d[-1].mean():.4f}, "
      f"reference mean abs value {np.abs(ref[-1]).mean():.4f}")

np.savez_compressed(out_path, frames=frames, desire=desire, traffic_convention=tc, action_t=at, outputs=ref,
                    **{f"final_{k}": v for k, v in final_state.items()})
print("saved", out_path)
