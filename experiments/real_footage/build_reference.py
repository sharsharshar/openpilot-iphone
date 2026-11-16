"""Build a driving reference from real footage (openpilot model-replay segment) and sanity-check it.

Needs in $RF_SCRATCH: log.pkl (extract_log.py), f_nv12.bin / e_nv12.bin (narrow / wide, raw NV12 1344x760, first NFRAMES frames:
  ffmpeg -i fcamera.hevc -frames:v 600 -pix_fmt nv12 -f rawvideo f_nv12.bin ; same for ecamera.hevc -> e_nv12.bin).
Usage: python -I build_reference.py     (python from the project venv; tinygrad on CPU is used for the warp oracle)
"""
import os, sys, ctypes, subprocess, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DEV", "CPU")
import numpy as np
import onnxruntime as ort
from rf_common import *            # noqa
from PIL import Image

W, H, NF = 1344, 760, 600
FSIZE = W * H * 3 // 2
START = 0
# openpilot camera.py: ("mici","os04c10") -> _os_config: narrow 1344x760 f=1522*3/4, wide f=567/4*3
K_NARROW = np.array([[1522.0 * 3 / 4, 0, W / 2], [0, 1522.0 * 3 / 4, H / 2], [0, 0, 1]])
K_WIDE = np.array([[567.0 / 4 * 3, 0, W / 2], [0, 567.0 / 4 * 3, H / 2], [0, 0, 1]])

d = pickle.load(open(f"{SCRATCH}/log.pkl", "rb"))
print("device", d["dev"], "sensor narrow/wide", d["sensor"], d["wsensor"])
nar = np.array([x[1] for x in d["nar"]]); wide = np.array([x[1] for x in d["wide"]])
print("frame ids equal for all 1200:", np.array_equal(nar, wide), "range", nar[0], nar[-1])
sof = np.array([x[2] for x in d["nar"]]) - np.array([x[2] for x in d["wide"]])
print("narrow-wide start-of-frame difference: %.3f..%.3f ms" % (sof.min() / 1e6, sof.max() / 1e6))
t_nar = np.array([x[0] for x in d["nar"]], np.float64)
cs = np.array(d["cs"], np.float64)
vego_all = np.interp(t_nar, cs[:, 0], cs[:, 1])
ext_t = np.array([e[0] for e in d["ext"]], np.float64); ext_rpy = np.array([e[1] for e in d["ext"]])
sel = slice(START, START + NF)
print("vEgo over the stretch: min %.2f max %.2f mean %.2f m/s" % (vego_all[sel].min(), vego_all[sel].max(), vego_all[sel].mean()))
# calibration valid at the first used frame (what modeld would hold), and its spread over the stretch
i0 = max(0, int(np.searchsorted(ext_t, t_nar[START], side="right")) - 1)
rpy = ext_rpy[i0].astype(np.float32)
inr = (ext_t >= t_nar[START]) & (ext_t <= t_nar[START + NF - 1])
print("rpyCalib used:", rpy, " (log message %d, status %s); spread over stretch (max-min): %s" % (i0, d["ext"][i0][3], np.ptp(ext_rpy[inr], axis=0)))

# ---- C core as shared library
build = f"{SCRATCH}/build"; os.makedirs(build, exist_ok=True)
subprocess.run(["cc", "-std=c99", "-O2", "-shared", "-fPIC", "-o", f"{build}/libopcore.so", f"{ROOT}/core/opcore.c", "-lm"], check=True)
lib = ctypes.CDLL(f"{build}/libopcore.so")
f32p = ctypes.POINTER(ctypes.c_float)
lib.op_warp_matrix.argtypes = [f32p, f32p, ctypes.c_int, f32p]
lib.op_frame_to_model_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, f32p]
def c_matrix(K, big):
    k = np.ascontiguousarray(K, np.float32); r = np.ascontiguousarray(rpy, np.float32); m = np.zeros(9, np.float32)
    lib.op_warp_matrix(k.ctypes.data_as(f32p), r.ctypes.data_as(f32p), big, m.ctypes.data_as(f32p)); return m
def c_warp(frame, m):
    out = np.zeros(OP_IMG_LEN, np.float32); fr = np.ascontiguousarray(frame)
    lib.op_frame_to_model_input(fr.ctypes.data, W, fr.ctypes.data + W * H, W, W, H, m.ctypes.data_as(f32p), out.ctypes.data_as(f32p))
    return out.reshape(6, 128, 256)
OP_IMG_LEN = 6 * 128 * 256
M_c = [c_matrix(K_NARROW, 0), c_matrix(K_WIDE, 1)]
M_o = [oracle.get_warp_matrix(rpy.astype(np.float64), K, big).astype(np.float32) for K, big in ((K_NARROW, False), (K_WIDE, True))]
print("matrix C vs oracle (float32 cast): max abs diff narrow %.3e wide %.3e" % tuple(np.abs(M_c[i] - M_o[i].ravel()).max() for i in range(2)))
print("narrow M:\n", M_o[0]); print("wide M:\n", M_o[1])

# ---- warp all frames
raw = [np.memmap(f"{SCRATCH}/{n}_nv12.bin", np.uint8, "r") for n in ("f", "e")]
frames = np.zeros((NF, 2, 6, 128, 256), np.uint8); nonint = 0; t0 = time.time()
for i in range(NF):
    for c in range(2):
        out = c_warp(raw[c][(START + i) * FSIZE:(START + i + 1) * FSIZE], M_c[c])
        nonint += int((out != np.round(out)).sum()); frames[i, c] = out.astype(np.uint8)
print("warped %d x 2 frames in %.1fs; non-integer samples: %d" % (NF, time.time() - t0, nonint))

# ---- C vs tinygrad oracle on real frames
for c, name in ((0, "narrow"), (1, "wide")):
    for i in (0, NF // 2, NF - 1):
        fl = np.asarray(raw[c][(START + i) * FSIZE:(START + i + 1) * FSIZE])
        ref = oracle.tinygrad_warp(fl, M_c[c].reshape(3, 3), W, H, W, H, H // 2).astype(np.float32)
        mine = c_warp(fl, M_c[c]); bad = int((ref != mine).sum())
        print("oracle check %s frame %d: mismatching samples %d / %d (max |diff| %g); bitwise identical: %s" % (name, START + i, bad, ref.size, np.abs(ref - mine).max(), bad == 0))

# ---- PNGs
def yuv6_to_rgb(x):
    x = x.astype(np.float32); Y = np.zeros((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = x[0], x[1], x[2], x[3]
    U = np.kron(x[4], np.ones((2, 2), np.float32)) - 128; V = np.kron(x[5], np.ones((2, 2), np.float32)) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)
os.makedirs(f"{SCRATCH}/png", exist_ok=True)
for i in (0, NF // 2, NF - 1):
    img = np.concatenate([yuv6_to_rgb(frames[i, 0]), yuv6_to_rgb(frames[i, 1])], 1)
    # model-frame horizon reference: y = MEDMODEL_CY(47.6)*? shown as a thin line for narrow centre column
    Image.fromarray(img).save(f"{SCRATCH}/png/warped_{START + i}.png")
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(np.asarray(raw[c][(START + i) * FSIZE:(START + i) * FSIZE + W * H]).reshape(H, W)).resize((672, 380))) for c in range(2)], 1)).save(f"{SCRATCH}/png/rawY_{START + i}.png")
print("model-frame principal point for the narrow warp: (256, 47.6) in the 512x256 frame (medmodel_intrinsics)")

# ---- run fp32 model, state from zeros
desire = np.zeros((NF, 8), np.float32); tc = np.tile(np.array([[1, 0]], np.float32), (NF, 1, 1)); at = np.tile(np.array([[0.2, 0.5]], np.float32), (NF, 1, 1))
def run_model(path, frames, desire, tc, at, threads=4):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); outs = [o.name for o in sess.get_outputs()]
    st = {"state_img_q": np.zeros((2, 5, 6, 128, 256), np.uint8), "state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}
    res = []
    for t in range(len(frames)):
        o = dict(zip(outs, sess.run(outs, {"new_img": frames[t], "desire": desire[t], "traffic_convention": tc[t], "action_t": at[t], **st})))
        res.append(o["outputs"][0].astype(np.float32)); st = {k: o["next_" + k] for k in st}
    return np.stack(res)
t0 = time.time(); outputs = run_model(f"{ROOT}/models/driving_supercombo_fp32.onnx", frames, desire, tc, at)
print("fp32 model: %d steps in %.1fs, finite: %s" % (NF, time.time() - t0, np.isfinite(outputs).all()))
np.savez_compressed(f"{SCRATCH}/real_reference_fp32.npz", frames=frames, desire=desire, traffic_convention=tc, action_t=at, outputs=outputs)
np.savez(f"{SCRATCH}/real_meta.npz", v_ego=vego_all[sel], frame_id=nar[sel], rpy=rpy)

# ---- 40-step fixtures from a warm part: expected outputs from zero state over exactly these 40 frames
F0 = 300
fx = frames[F0:F0 + 40]
fx_out = run_model(f"{ROOT}/models/driving_supercombo_fp32.onnx", fx, desire[:40], tc[:40], at[:40])
fx.tofile(f"{SCRATCH}/frames_u8.bin"); fx_out.astype("<f4").tofile(f"{SCRATCH}/outputs_f32.bin")
print("fixtures: frames %d..%d; sizes %d / %d bytes; zero-state vs long-run outputs on those steps: mean abs diff %.4f (step 0) %.4f (step 39)" % (
    START + F0, START + F0 + 39, os.path.getsize(f"{SCRATCH}/frames_u8.bin"), os.path.getsize(f"{SCRATCH}/outputs_f32.bin"),
    np.abs(fx_out[0] - outputs[F0]).mean(), np.abs(fx_out[39] - outputs[F0 + 39]).mean()))

# ---- sanity: is it seeing a road?
syn = np.load(f"{ROOT}/reference/driving_reference_fp32.npz")["outputs"]
vs = vego_all[sel]
qr = driving_quantities(outputs, vs); qs = driving_quantities(syn, np.full(len(syn), 15.0))
print("\n== lane-line probs [outer-left, inner-left, inner-right, outer-right], mean/min/max over steps")
for name, q, sl in (("real warm(100:)", qr, slice(100, None)), ("real all", qr, slice(None)), ("synthetic all 40", qs, slice(None))):
    print(" %-16s mean %s min %s max %s" % (name, np.round(q["lane"][sl].mean(0), 3), np.round(q["lane"][sl].min(0), 3), np.round(q["lane"][sl].max(0), 3)))
print("lead prob: real warm mean %.3f max %.3f | synthetic mean %.3f max %.3f" % (qr["lead"][100:].mean(), qr["lead"][100:].max(), qs["lead"].mean(), qs["lead"].max()))
for a, b in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)):
    print(" steps %3d-%3d: vEgo %.2f  plan v(t=0) %.2f  lane p %s  lead p %.2f  plan x@10s %.1f  y@10s %.2f  curv %.5f" % (
        a, b, vs[a:b].mean(), qr["v0"][a:b].mean(), np.round(qr["lane"][a:b].mean(0), 2), qr["lead"][a:b].mean(), qr["x10"][a:b].mean(), qr["y10"][a:b].mean(), qr["curv"][a:b].mean()))
w = slice(100, None); dv = qr["v0"][w] - vs[w]
print("plan v(t=0) minus vEgo (warm): mean %.3f  mean abs %.3f  RMS %.3f  max abs %.3f; correlation %.3f" % (dv.mean(), np.abs(dv).mean(), np.sqrt((dv ** 2).mean()), np.abs(dv).max(), np.corrcoef(qr["v0"][w], vs[w])[0, 1]))
dvf = qr["v0"][:100] - vs[:100]; print("   first 100 steps: mean abs %.3f" % np.abs(dvf).mean())
print("synthetic: plan v(t=0) mean %.3f range %.3f..%.3f; plan x@10s mean %.1f" % (qs["v0"].mean(), qs["v0"].min(), qs["v0"].max(), qs["x10"].mean()))

# device's own modelV2 (different model version; loose comparison), aligned by frame id
mm = {m["fid"]: m for m in d["model"]}; ok = [i for i in range(NF) if int(nar[START + i]) in mm]
dev = lambda f: np.array([f(mm[int(nar[START + i])]) for i in ok])
dv0 = dev(lambda m: m["vx"][0]); dcurv = dev(lambda m: m["curv"]); dacc = dev(lambda m: m["acc"]); dy10 = dev(lambda m: m["py"][32]); dx10 = dev(lambda m: m["px"][32])
dlp = dev(lambda m: np.array(m["lp"])[1:3]); dlead = dev(lambda m: m["lead"][0])
ok = np.array(ok); wk = ok >= 100
c = lambda a, b: np.corrcoef(a, b)[0, 1]
print("\n== vs device's own modelV2 (frames matched by frameId: %d of %d, warm steps only)" % (wk.sum(), NF - 100))
print(" speed at t=0: device-vs-ours mean abs diff %.3f m/s, corr %.3f; device vs vEgo mean abs %.3f" % (np.abs(dv0[wk] - qr["v0"][ok][wk]).mean(), c(dv0[wk], qr["v0"][ok][wk]), np.abs(dv0[wk] - vs[ok][wk]).mean()))
print(" plan x@10s: device mean %.1f ours %.1f, corr %.3f" % (dx10[wk].mean(), qr["x10"][ok][wk].mean(), c(dx10[wk], qr["x10"][ok][wk])))
print(" plan y@10s: mean abs diff %.3f m (ours mean abs %.3f, device mean abs %.3f), corr %.3f" % (np.abs(dy10[wk] - qr["y10"][ok][wk]).mean(), np.abs(qr["y10"][ok][wk]).mean(), np.abs(dy10[wk]).mean(), c(dy10[wk], qr["y10"][ok][wk])))
print(" desired curvature: mean abs diff %.5f 1/m (device mean abs %.5f, ours %.5f), corr %.3f" % (np.abs(dcurv[wk] - qr["curv"][ok][wk]).mean(), np.abs(dcurv[wk]).mean(), np.abs(qr["curv"][ok][wk]).mean(), c(dcurv[wk], qr["curv"][ok][wk])))
print(" desired accel: mean abs diff %.3f m/s^2 (device mean abs %.3f, ours %.3f), corr %.3f" % (np.abs(dacc[wk] - qr["acc"][ok][wk]).mean(), np.abs(dacc[wk]).mean(), np.abs(qr["acc"][ok][wk]).mean(), c(dacc[wk], qr["acc"][ok][wk])))
print(" inner lane-line probs: device mean %s ours %s" % (np.round(dlp[wk].mean(0), 3), np.round(qr["lane"][ok][wk][:, 1:3].mean(0), 3)))
print(" lead prob: device mean %.3f ours %.3f; corr %.3f" % (dlead[wk].mean(), qr["lead"][ok][wk].mean(), c(dlead[wk], qr["lead"][ok][wk])))
