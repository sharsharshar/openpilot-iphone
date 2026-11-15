"""Verifies core/opcore.c against the upstream-derived oracle. Run via core/run_tests.sh (python -I)."""
import os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "oracle"))
import numpy as np
import oracle

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DRIVER = os.environ.get("OP_DRIVER", os.path.join(HERE, "build", "test_driver"))
TMP = tempfile.mkdtemp(prefix="opcore_")
FAILS = []
RESULT_FLOATS = 3 + 99 + 99 + 264 + 4 + 132 + 4

def check(name, ok, detail):
  print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)
  if not ok: FAILS.append(name)

def run(*args): subprocess.run([DRIVER, *map(str, args)], check=True)
def tmp(n): return os.path.join(TMP, n)

# ---------------------------------------------------------------- 1. parser
def parse_cases(rows, params):
  """rows [N,2576], params [N,3] (v_ego, lat_t, long_t); NaN v_ego = reset prev. Returns C results [N,RF], oracle [N,RF]."""
  rows = np.ascontiguousarray(rows, np.float32); params = np.ascontiguousarray(params, np.float32)
  rows.tofile(tmp("rows.bin")); params.tofile(tmp("params.bin"))
  run("parse", tmp("rows.bin"), tmp("params.bin"), tmp("res.bin"))
  c = np.fromfile(tmp("res.bin"), np.float32).reshape(len(rows), RESULT_FLOATS)
  ref = []; prev = (0.0, 0.0)
  for r, (v, lt, gt) in zip(rows, params):
    if np.isnan(v): prev = (0.0, 0.0); v = 0.0
    outs = oracle.parse(r)
    curv, acc, stop = oracle.get_action(outs, prev, float(lt), float(gt), float(v))
    prev = (curv, acc)
    f = oracle.result_fields(outs, curv, acc, stop)
    ref.append(np.concatenate([[f["desired_curvature"], f["desired_accel"], f["should_stop"]], f["plan_pos"].ravel(), f["plan_vel"].ravel(),
                               f["lane_lines"].ravel(), f["lane_line_probs"].ravel(), f["road_edges"].ravel(),
                               [f["lead_prob"]], f["lead_xyv"]]))
  return c, np.array(ref, np.float64)

SEGS = [("action(curv,acc,stop)", 0, 3), ("plan_pos", 3, 102), ("plan_vel", 102, 201), ("lane_lines", 201, 465),
        ("lane_line_probs", 465, 469), ("road_edges", 469, 601), ("lead", 601, 605)]
def relerr(c, ref, atol=1e-6): return np.abs(c - ref) / (atol + np.abs(ref))

def report_parse(name, c, ref):
  e = relerr(c, ref)
  worst = {s: float(e[:, a:b].max()) for s, a, b in SEGS}
  stop_ok = np.array_equal(c[:, 2], ref[:, 2])
  w = max(worst.values())
  check(name, w < 1e-4 and stop_ok, "worst rel err %.2e (%s), should_stop identical=%s" % (w, max(worst, key=worst.get), stop_ok))

def test_parse():
  d = np.load(os.path.join(ROOT, "reference", "driving_reference_fp32.npz"))
  out = d["outputs"].astype(np.float32); at = d["action_t"][:, 0, :]
  n = len(out)
  # (a) reference action_t, realistic speeds
  params = np.stack([np.linspace(5, 25, n), at[:, 0], at[:, 1]], 1)
  c, ref = parse_cases(out, params); report_parse("parse ref40 v=5..25 action_t=[0.2,0.5]", c, ref)
  # (b) sweep v_ego across clamp/threshold regions incl. negative, <0.3, 0.3 edge, 1.0 edge, and action_t on both sides of 0.3
  vs = np.array([-2, 0, 0.1, 0.29, 0.3, 0.31, 0.99, 1.0, 1.01, 3, 12, 40])
  params = np.stack([vs[np.arange(n) % len(vs)], np.where(np.arange(n) % 3 == 0, 0.1, at[:, 0] + 0.3 * (np.arange(n) % 2)),
                     np.where(np.arange(n) % 4 == 0, 0.15, 0.5 + 0.7 * (np.arange(n) % 3 == 1))], 1)
  params[0, 0] = np.nan; params[0, 0] = np.nan
  c, ref = parse_cases(out, params); report_parse("parse ref40 v sweep, action_t on both sides of 0.3", c, ref)
  # (c) stopped sequence: exercises should_stop and prev-carry of curvature when v_ego<=0.3
  params = np.stack([np.where(np.arange(n) < 20, 0.0, 6.0), at[:, 0], at[:, 1]], 1); params[0, 0] = np.nan
  c, ref = parse_cases(out, params); report_parse("parse ref40 stopped then moving (prev carry)", c, ref)
  # (d) saturating logits: sigmoid clamp at exp(11)
  rng = np.random.default_rng(1); rows = out[:8].copy()
  rows[:, 645:653] = rng.choice([-60, -12, -11, -10.9, 0, 11, 60], size=(8, 8)); rows[:, 1061:1064] = rng.choice([-80, 80, 0.5], size=(8, 3))
  params = np.stack([np.full(8, 10.0), np.full(8, 0.2), np.full(8, 0.5)], 1); params[0, 0] = np.nan
  c, ref = parse_cases(rows, params); report_parse("parse saturating sigmoid logits", c, ref)
  # (e) the app fixtures (all reference rows, v_ego 0, action_t [0.2,0.5])
  fx = np.fromfile(os.path.join(ROOT, "fixtures", "outputs_f32.bin"), "<f4").reshape(-1, 2576)
  params = np.tile([0.0, 0.2, 0.5], (len(fx), 1)); params[0, 0] = np.nan
  c, ref = parse_cases(fx, params); report_parse("parse fixtures/outputs_f32.bin (v_ego=0)", c, ref)
  check("fixture outputs == all %d reference rows" % len(fx), len(fx) == 40 and np.array_equal(fx, out[:len(fx)]), "bitwise")
  return d

# ---------------------------------------------------------------- 2. warp matrix
def test_matrix():
  rng = np.random.default_rng(2); cases = []
  Ks = [np.array([[2648., 0, 964.], [0, 2648., 604.], [0, 0, 1]]),          # tici narrow, 1928x1208
        np.array([[567., 0, 964.], [0, 567., 604.], [0, 0, 1]]),            # tici wide
        np.array([[1140., 0, 960.], [0, 1140., 540.], [0, 0, 1]]),          # 1920x1080-ish
        np.array([[1522 * .75, 0, 672.], [0, 1522 * .75, 380.], [0, 0, 1]]),
        np.array([[1500.5, 0, 639.5], [0, 1498.25, 359.5], [0, 0, 1]])]      # slightly non-square
  rpys = [[0, 0, 0], [0.01, -0.02, 0.03], [-0.05, 0.1, -0.15], [0.3, -0.2, 0.25]] + [rng.uniform(-0.2, 0.2, 3).tolist() for _ in range(6)]
  buf = b""; exp = []
  for K in Ks:
    for r in rpys:
      for big in (0, 1):
        K32 = K.astype(np.float32); r32 = np.array(r, np.float32)
        buf += K32.tobytes() + r32.tobytes() + np.int32(big).tobytes()
        exp.append(oracle.get_warp_matrix(r32.astype(np.float64), K32.astype(np.float64), bool(big)))
  open(tmp("mat_in.bin"), "wb").write(buf)
  run("matrix", tmp("mat_in.bin"), tmp("mat_out.bin"))
  c = np.fromfile(tmp("mat_out.bin"), np.float32).reshape(-1, 3, 3).astype(np.float64); exp = np.array(exp)
  # elementwise error relative to the matrix's largest entry (small entries are float32 cancellation noise)
  e = np.abs(c - exp).max(axis=(1, 2)) / np.abs(exp).max(axis=(1, 2))
  # also check the effect on pixels: where do the 4 model corners land (px)
  corners = np.array([[0, 0, 1], [511, 0, 1], [0, 255, 1], [511, 255, 1.]]).T
  def proj(M): p = M @ corners; return p[:2] / p[2]
  pe = max(np.abs(proj(c[i]) - proj(exp[i])).max() for i in range(len(c)))
  check("op_warp_matrix vs get_warp_matrix (%d cases, both big_model)" % len(c), e.max() < 1e-6 and pe < 1e-2,
        "worst err relative to max|M| %.2e, worst corner projection error %.2e px" % (e.max(), pe))

# ---------------------------------------------------------------- 3. frame warp
def nv12_buffer(rng, w, h, stride, y_height, uv_height, kind="noise"):
  buf = rng.integers(0, 256, stride * (y_height + uv_height) + 4096, dtype=np.uint8)   # padding bytes are random garbage
  if kind == "smooth":
    yy, xx = np.mgrid[0:h, 0:w]
    buf[:stride * y_height].reshape(y_height, stride)[:h, :w] = (128 + 100 * np.sin(xx / 37.0) * np.cos(yy / 23.0)).astype(np.uint8)
  return buf

def c_warp(buf, w, h, stride, y_height, M):
  buf.tofile(tmp("frame.bin")); np.asarray(M, np.float32).tofile(tmp("m.bin"))
  run("warp", tmp("frame.bin"), w, h, stride, y_height, tmp("m.bin"), tmp("warp_out.bin"))
  return np.fromfile(tmp("warp_out.bin"), np.float32).reshape(6, 128, 256)

def test_warp():
  rng = np.random.default_rng(3)
  worst_frac = 0.0; total = 0
  for (w, h, stride, yh, uvh) in [(1920, 1080, 2048, 1088, 544), (1280, 720, 1408, 736, 368)]:
    K = np.array([[1100., 0, w / 2], [0, 1100., h / 2], [0, 0, 1]])
    mats = {"narrow rpy0": oracle.get_warp_matrix([0, 0, 0], K, False),
            "wide rpy": oracle.get_warp_matrix([0.02, -0.03, 0.04], K * [[0.5, 1, 1], [1, 0.5, 1], [1, 1, 1]], True),
            "big tilt": oracle.get_warp_matrix([0.1, 0.15, -0.2], K, False),
            "half-scale (exact .5 ties)": np.diag([0.5, 0.5, 1.0]) + np.array([[0, 0, 0.25], [0, 0, 0.75], [0, 0, 0]]),
            "2x-zoom off-frame (clamp)": np.array([[8., 0, -300], [0, 8., -700], [0, 0, 1]]),
            "perspective + beyond frame": np.array([[3., 0.1, 1500], [0.05, 3., 900], [-1e-4, 2e-4, 1.0]])}
    for kind in ("noise", "smooth"):
      buf = nv12_buffer(rng, w, h, stride, yh, uvh, kind)
      for mn, M in mats.items():
        M32 = np.asarray(M, np.float32)
        c = c_warp(buf, w, h, stride, yh, M32)
        r = oracle.tinygrad_warp(buf, M32, w, h, stride, yh, uvh).astype(np.float32)
        bad = (c != r); frac = bad.mean(); total += c.size
        worst_frac = max(worst_frac, frac)
        maxd = float(np.abs(c - r).max())
        perch = bad.reshape(6, -1).mean(1)
        ok = frac <= 1e-4 and c.min() >= 0 and c.max() <= 255
        check("op_frame_to_model_input %dx%d stride %d %s/%s" % (w, h, stride, kind, mn), ok,
              "mismatching samples %.3e (per channel max %.1e), max |diff| %g" % (frac, perch.max(), maxd))
  return worst_frac

# ---------------------------------------------------------------- 4. end-to-end
def test_e2e():
  import onnxruntime as ort
  w, h, stride, yh, uvh = 1920, 1080, 2048, 1088, 544
  rng = np.random.default_rng(4)
  frames = [nv12_buffer(rng, w, h, stride, yh, uvh, "smooth") for _ in range(2)]
  for f in frames:                                    # add colour structure to the chroma planes and some noise on luma
    f[stride * yh: stride * (yh + uvh)] = rng.integers(60, 200, stride * uvh, dtype=np.uint8)
  K_n = np.array([[2000., 0, w / 2], [0, 2000., h / 2], [0, 0, 1]], np.float32)
  K_w = np.array([[900., 0, w / 2], [0, 900., h / 2], [0, 0, 1]], np.float32)
  rpy = np.array([0.005, -0.01, 0.02], np.float32)
  Mc = [np.zeros(9, np.float32), np.zeros(9, np.float32)]
  op_c = []
  for i, (K, big) in enumerate(((K_n, 0), (K_w, 1))):
    open(tmp("mat_in.bin"), "wb").write(K.tobytes() + rpy.tobytes() + np.int32(big).tobytes())
    run("matrix", tmp("mat_in.bin"), tmp("mat_out.bin"))
    Mc[i] = np.fromfile(tmp("mat_out.bin"), np.float32)
    op_c.append(c_warp(frames[i], w, h, stride, yh, Mc[i]))
  Mo = [oracle.get_warp_matrix(rpy.astype(np.float64), K.astype(np.float64), big).astype(np.float32) for K, big in ((K_n, False), (K_w, True))]
  op_o = [oracle.tinygrad_warp(frames[i], Mo[i], w, h, stride, yh, uvh).astype(np.float32) for i in range(2)]
  wd = max(float(np.abs(a - b).max()) for a, b in zip(op_c, op_o))
  frac = max(float((a != b).mean()) for a, b in zip(op_c, op_o))
  print("  e2e: warp input difference C vs tinygrad: max |diff| %g, mismatching fraction %.2e" % (wd, frac))
  sess = ort.InferenceSession(os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx"), providers=["CPUExecutionProvider"])
  ins = {i.name: i for i in sess.get_inputs()}
  def feed(new_img):
    f = {"new_img": np.stack(new_img).round().astype(np.uint8), "desire": np.zeros(8, np.float32),
         "traffic_convention": np.array([[1, 0]], np.float32), "action_t": np.array([[0.2, 0.5]], np.float32)}
    for n, i in ins.items():
      if n not in f: f[n] = np.zeros(i.shape, np.uint8 if "uint8" in i.type else np.float32)
    return f
  out_c = sess.run(["outputs"], feed(op_c))[0][0].astype(np.float32)
  out_o = sess.run(["outputs"], feed(op_o))[0][0].astype(np.float32)
  check("e2e model outputs finite", bool(np.isfinite(out_c).all() and np.isfinite(out_o).all()), "")
  raw = float(np.abs(out_c - out_o).max())
  print("  e2e: raw model output max |C-oracle| = %.3e (output magnitude max %.1f)" % (raw, np.abs(out_o).max()))
  params = np.array([[np.nan, 0.2, 0.5]], np.float32); params[0, 0] = np.nan
  # v_ego 8 m/s through parser, both paths from their own model outputs
  params = np.array([[8.0, 0.2, 0.5]], np.float32)
  c, _ = parse_cases(out_c[None], params)
  _, ref = parse_cases(out_o[None], params)     # oracle parse (the C result of this call is ignored)
  e = relerr(c, ref)
  # parser alone is verified to 1e-4; here the inputs differ only by warp sampling differences, so use a looser bound
  worst = max(float(e[:, a:b].max()) for s, a, b in SEGS)
  check("e2e synthetic NV12 -> C warp x2 -> onnxruntime -> C parse vs full-oracle path",
        worst < 1e-3 if frac < 1e-4 else False,
        "worst rel err %.2e; warp mismatching fraction %.2e; raw output max abs diff %.2e" % (worst, frac, raw))
  # model sensitivity sanity: output must depend on the image (guards against a trivially-equal comparison)
  out_z = sess.run(["outputs"], feed([np.zeros_like(op_c[0])] * 2))[0][0]
  sens = float(np.abs(out_z - out_c).max())
  check("e2e sanity: model output responds to image content", sens > 100 * max(raw, 1e-6), "|out(real)-out(black)| %.3e vs C/oracle diff %.3e" % (sens, raw))

# ---------------------------------------------------------------- 5. fixtures replay (what the iOS self-test does)
def test_fixtures():
  import onnxruntime as ort
  fr = np.fromfile(os.path.join(ROOT, "fixtures", "frames_u8.bin"), np.uint8).reshape(-1, 2, 6, 128, 256)
  ex = np.fromfile(os.path.join(ROOT, "fixtures", "outputs_f32.bin"), "<f4").reshape(-1, 2576)
  assert len(fr) == len(ex) == 40
  sess = ort.InferenceSession(os.path.join(ROOT, "models", "driving_supercombo_fp32.onnx"), providers=["CPUExecutionProvider"])
  ins = {i.name: i for i in sess.get_inputs()}; outs = [o.name for o in sess.get_outputs()]
  st = {n: np.zeros(i.shape, np.uint8 if "uint8" in i.type else np.float32) for n, i in ins.items() if "next_" + n in outs}
  worst = 0.0
  for t in range(len(fr)):
    des = np.zeros(8, np.float32)
    if t == 20: des[3] = 1.0   # the one desire pulse; SelfTest.swift feeds the same
    f = {"new_img": fr[t], "desire": des, "traffic_convention": np.array([[1, 0]], np.float32),
         "action_t": np.array([[0.2, 0.5]], np.float32), **st}
    res = dict(zip(outs, sess.run(outs, f)))
    st = {n: res["next_" + n] for n in st}
    worst = max(worst, float(np.abs(res["outputs"][0] - ex[t]).max()))
  check("fixtures replay through onnxruntime fp32 (state from zeros)", worst < 1e-4, "max abs diff vs outputs_f32.bin %.2e" % worst)

# ---------------------------------------------------------------- 6. frame history (lean model input)
def test_history():
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  from verify_lean import NumpyHistory
  fr = np.fromfile(os.path.join(ROOT, "fixtures", "frames_u8.bin"), np.uint8).reshape(-1, 2, 6, 128, 256)
  n = len(fr)
  for mode in ("u8", "f32"):
    for reset_at in (-1, 17):
      run("hist", os.path.join(ROOT, "fixtures", "frames_u8.bin"), mode, tmp("stack.bin"), reset_at)
      c = np.fromfile(tmp("stack.bin"), np.float32).reshape(n, 1, 24, 128, 256)
      h = NumpyHistory(); bad = []
      for t in range(n):
        if t == reset_at: h = NumpyHistory()
        h.push(fr[t])
        if c[t].tobytes() != h.stack().tobytes(): bad.append(t)
      check("history %s reset_at=%d: img_stack bitwise equal to numpy over %d steps (incl. zero-filled start and ring wrap)" % (mode, reset_at, n),
            not bad, "mismatching steps %s" % bad[:5])
  # the first stack really is: cam0 old (zeros), cam0 new, cam1 old (zeros), cam1 new
  run("hist", os.path.join(ROOT, "fixtures", "frames_u8.bin"), "u8", tmp("stack.bin"), -1)
  c = np.fromfile(tmp("stack.bin"), np.float32).reshape(n, 24, 128, 256)
  ok = (not c[0, 0:6].any() and np.array_equal(c[0, 6:12], fr[0, 0]) and not c[0, 12:18].any() and np.array_equal(c[0, 18:24], fr[0, 1])
        and np.array_equal(c[9, 0:6], fr[5, 0]) and np.array_equal(c[9, 6:12], fr[9, 0]) and np.array_equal(c[9, 12:18], fr[5, 1]))
  check("history layout: [cam0 4-steps-ago, cam0 newest, cam1 4-steps-ago, cam1 newest]", ok, "step 0 and step 9 spot check")
  # C-built stacks -> lean fp32 model -> the fixture outputs the app's self-test compares against
  import onnxruntime as ort
  lean = os.path.join(ROOT, "models", "driving_supercombo_lean_fp32.onnx")
  if not os.path.exists(lean):
    check("history -> lean model replay", False, "models/driving_supercombo_lean_fp32.onnx missing (run scripts/make_lean.py)"); return
  ex = np.fromfile(os.path.join(ROOT, "fixtures", "outputs_f32.bin"), "<f4").reshape(-1, 2576)
  sess = ort.InferenceSession(lean, providers=["CPUExecutionProvider"]); outs = [o.name for o in sess.get_outputs()]
  st = {"state_desire_q": np.zeros((100, 1, 8), np.float32), "state_feat_q": np.zeros((96, 1, 512), np.float32)}; worst = 0.0
  for t in range(n):
    des = np.zeros(8, np.float32)
    if t == 20: des[3] = 1.0
    res = dict(zip(outs, sess.run(outs, {"img_stack": c[t][None], "desire": des, "traffic_convention": np.array([[1, 0]], np.float32),
                                         "action_t": np.array([[0.2, 0.5]], np.float32), **st})))
    st = {k: res["next_" + k] for k in st}
    worst = max(worst, float(np.abs(res["outputs"][0] - ex[t]).max()))
  check("fixtures replay: C history -> lean fp32 model vs outputs_f32.bin", worst < 1e-4, "max abs diff %.2e" % worst)

if __name__ == "__main__":
  t0 = time.time()
  test_parse(); test_matrix(); test_warp(); test_e2e(); test_fixtures(); test_history()
  print("\n%s (%.0fs)" % ("ALL PASSED" if not FAILS else "FAILED: " + ", ".join(FAILS), time.time() - t0))
  sys.exit(1 if FAILS else 0)
