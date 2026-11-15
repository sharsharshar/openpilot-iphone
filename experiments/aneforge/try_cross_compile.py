"""Try ANEForge cross-compilation of the openpilot driving model for an iPhone-class ANE family. Runs on macOS.

Usage: python3 try_cross_compile.py <aneforge_clone> <cut_model.onnx> <out_dir> [arch=h16s]

<cut_model.onnx> is the adapted sub-graph written by run_preflight.py (driving_cut_adapted.onnx).
Candidates, smallest first: "stem" (first 2 convs + Gelu) and "full" (the whole adapted sub-graph).
For each candidate, each stage runs in its own child process with a timeout, so a hang or a crash
only loses that stage. Every result is printed as an ANEFORGE-prefixed line. Exit status is always 0.
Stages: import, preflight, cross_compile_check(out, arch), af.compile(out, target=arch).
"""
import json, os, subprocess, sys, time, traceback

STAGE_TIMEOUT = int(os.environ.get("ANEFORGE_STAGE_TIMEOUT", "600"))
CANDIDATES = {"stem": ("model/div", "model/gelu_2"), "full": None}

def P(*a): print("ANEFORGE", *a, flush=True)

def build(af_dir, onnx_path, out_dir, cand):
  sys.path.insert(0, os.path.abspath(af_dir))
  os.environ.setdefault("ANEFORGE_CACHE_DIR", os.path.join(os.path.abspath(out_dir), "aneforge-cache"))
  import onnx, onnx.utils
  import aneforge as af
  m = onnx.load(onnx_path)
  if CANDIDATES[cand] is not None:
    a, b = CANDIDATES[cand]
    path = os.path.join(out_dir, f"cand_{cand}.onnx")
    onnx.utils.extract_model(onnx_path, path, [a], [b])
    m = onnx.load(path)
  ins, out = af.onnx_to_tensor(m)
  return af, ins, out

def child(stage, af_dir, onnx_path, out_dir, cand, arch):
  import warnings
  warnings.simplefilter("always")
  af, ins, out = build(af_dir, onnx_path, out_dir, cand)
  if stage == "import":
    print("RESULT ok inputs", [t.shape for t in ins], "output", out.shape)
  elif stage == "preflight":
    from aneforge import _targets as tg
    rep = tg.preflight(out, tg.family_of_arch(arch))
    print("RESULT", "ok" if rep.ok else "NOT-OK", "native", len(rep.native), "decompose", len(rep.decompose),
          "reject", sorted({r.op for r in rep.reject}), "oversize", [(r.op, r.shape) for r in rep.oversize])
  elif stage == "cross_compile_check":
    from aneforge._compile import cross_compile_check
    print("RESULT", cross_compile_check(out, arch))
  elif stage == "compile":
    t = time.time(); mdl = af.compile(out, target=arch)
    print("RESULT compiled", type(mdl).__name__, f"{time.time()-t:.1f}s")

def run_stage(stage, cand, af_dir, onnx_path, out_dir, arch):
  cmd = [sys.executable, os.path.abspath(__file__), "--child", stage, af_dir, onnx_path, out_dir, cand, arch]
  t = time.time()
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=STAGE_TIMEOUT)
  except subprocess.TimeoutExpired as e:
    P(cand, stage, f"TIMEOUT after {STAGE_TIMEOUT}s"); return {"status": "timeout"}
  dt = time.time() - t
  res_lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
  tail = (r.stderr or "").strip().splitlines()
  if r.returncode == 0 and res_lines:
    P(cand, stage, "ok" if "NOT-OK" not in res_lines[-1] and "False" not in res_lines[-1] else "FAIL", res_lines[-1][7:], f"({dt:.1f}s)")
    for l in tail:
      if "Warning" in l: P(cand, stage, "warning:", l[:400])
    return {"status": "ran", "result": res_lines[-1], "stderr_tail": tail[-8:]}
  last = tail[-1] if tail else ""
  P(cand, stage, f"EXCEPTION/CRASH rc={r.returncode}", last[:600])
  for l in tail[-12:]: P(cand, stage, "stderr:", l[:300])
  return {"status": "failed", "rc": r.returncode, "stderr_tail": tail[-25:]}

def main():
  if len(sys.argv) > 1 and sys.argv[1] == "--child":
    _, _, stage, af_dir, onnx_path, out_dir, cand, arch = sys.argv
    child(stage, af_dir, onnx_path, out_dir, cand, arch); return
  af_dir, onnx_path, out_dir = sys.argv[1:4]
  arch = sys.argv[4] if len(sys.argv) > 4 else "h16s"
  os.makedirs(out_dir, exist_ok=True)
  P("host", sys.platform, os.uname().machine, "python", sys.version.split()[0], "arch", arch)
  results = {}
  for cand in CANDIDATES:
    results[cand] = {}
    for stage in ("import", "preflight", "cross_compile_check", "compile"):
      results[cand][stage] = run_stage(stage, cand, af_dir, onnx_path, out_dir, arch)
      if stage == "import" and results[cand][stage]["status"] != "ran": break   # nothing later can run
  json.dump(results, open(os.path.join(out_dir, "cross_compile.json"), "w"), indent=1)
  P("done; summary", {c: {s: v["status"] for s, v in r.items()} for c, r in results.items()})

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--child":
    main()          # a child must fail loudly: the parent records the traceback tail and return code
  else:
    try: main()
    except BaseException:
      P("driver exception", traceback.format_exc().replace("\n", " | "))
