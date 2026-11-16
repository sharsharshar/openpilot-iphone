# openpilot on an iPhone (prototype)

A small iOS app that runs comma.ai's openpilot driving model (converted to Core ML) on an iPhone.
It feeds camera frames through a C port of the preprocessing (`core/opcore.c`), runs the model
with its recurrent state carried from step to step, parses the 2576 outputs, and draws a top-down
view of the predicted path, lane lines and road edges, with timing numbers.

## Layout

- `coreml/` Core ML package of the model, `reference/` reference outputs, `fixtures/` 40 recorded steps.
- `core/` C preprocessing and output parsing. `ios/` the SwiftUI app and `project.yml` (XcodeGen).
- `.github/workflows/ios.yml` builds the app in an iPhone simulator and runs a self-test
  (`-selftest` launch argument: replays the fixtures and compares against the expected outputs).

## Status

Verified on GitHub's hosted Macs (see the Actions tab):

- The Core ML package matches the reference outputs on macOS 15 (40 steps, worst difference 2.39, mean 0.017).
- The app builds with Xcode 16.4 and runs in an iPhone simulator. On iOS 18.6 the self-test passes and the
  live path (model with state, C output parsing, plot) produces the expected curvature and acceleration.
- **Known open issue:** on the iOS 26.2 simulator the same model gives systematically different numbers
  (worst difference 5.9, mean 0.057, far end of the plan consistently low) and the self-test fails. The cause
  is not known. The CI job for iOS 26 reports this without failing the workflow.

Not verified anywhere yet: a real iPhone, the camera path (frame rate, intrinsics, orientation), the warp on
real camera frames, and anything on the Neural Engine or GPU.

## Lean model

The original model carries its last five camera images as an input and an output (`state_img_q` / `next_state_img_q`,
[2,5,6,128,256]) and only uses two of them: per camera, the image from 4 steps ago and the newest, stacked as channels
(cam 0 old 0-5, cam 0 new 6-11, cam 1 old 12-17, cam 1 new 18-23), then `(x - 127.5) / 63.75`. `scripts/make_lean.py` derives
that from the graph's constants and removes the queue by ONNX surgery: the model now takes `img_stack` [1,24,128,256]
float32 (raw 0..255; the cast and the normalisation stay inside the model, which is exact because the values are integers)
and carries only `state_desire_q` and `state_feat_q`. The history lives in C (`op_hist_*` in `core/opcore.c`, caller-owned
memory, uint8, ring buffer) and is filled into the model's input array each step. `coreml/driving_supercombo.mlpackage` is now
the lean package (same name); it has not been run anywhere yet (Core ML runs only on a Mac).

Equivalence, measured on Linux (`scripts/verify_lean.py`, onnxruntime float32, 40 reference steps including the first
steps with a zero-filled history): max abs difference to `reference/driving_reference_fp32.npz` is 0.0 (bit-identical), also
against the original model run side by side; the C history reproduces the numpy history bitwise (`core/tests`). Wrong frame
selection (frame 1 instead of 0) gives a difference of 20.7, so the check has teeth.

Bytes per frame in the app (not counting what Core ML does internally): before, 1,572,864 B copied into `new_img`, 7,864,320 B
written converting `next_state_img_q` float16 to float32 (3,932,160 B read), and Core ML took a 7,864,320 B `state_img_q` input:
9.4 MB written by the app per frame. After, 393,216 B pushed into the ring (from fixtures; from the warp's floats it reads 1.57 MB
and writes 0.39 MB) plus 3,145,728 B written building `img_stack` (786,432 B read): 3.5 MB written, no image state output.
Per-stage timings (`build_ms`, `predict_ms`, `carry_ms`, `parse_ms`, and `warp_ms` with the camera) are appended to the
`SELFTEST step` and `LIVE frames` lines.

## Experiments

- `experiments/ane_probe/`: drives the Neural Engine through Apple's private classes, bypassing Core ML. Also built
  into the app ("Run ANE probe" button, or the `-aneprobe` launch flag). Results so far: on a GitHub Mac (a virtual
  machine) the classes exist but the ANE compiler refuses to start ("Host too old, ANE hardware not available");
  in the iOS simulators the framework does not exist at all. It has not yet run on a real iPhone, which is the
  only place it can give an answer. Private API: for your own device only, never the App Store.
- `experiments/tinygrad_metal/`: the driving model through tinygrad (see `RESULTS.md`). On a GitHub Mac's virtual
  GPU via Metal: matches the reference, 161 kernel launches per inference from 78 distinct kernels, 51 ms per run
  with the time spread evenly across kernels. Timings on a virtual GPU do not predict a phone.

- `experiments/real_footage/`: the accuracy experiments repeated on 30 s of real road video (openpilot's own
  model-replay segment, both cameras, logged calibration and speed). The warp matches openpilot's on real frames,
  and the model's curvature output correlates 0.998 with what the comma device computed on that drive. float16 and
  the exact GELU change driving outputs negligibly; 8-bit weights and cruder GELU forms do not. The video and
  frames are downloaded at run time; only a 2-second, model-input-sized excerpt is committed, as the
  self-test's second fixture set (`fixtures/README.md`). See `RESULTS.md`.

## Simulator caveat

The simulator has no Neural Engine and no camera. It runs the model on the Mac CPU (and maybe GPU)
and plays the fixture frames instead of camera input, so the speed it reports says nothing about a
real phone. Use it only for correctness.

## Building for a real iPhone (on a Mac)

1. `brew install xcodegen`
2. `cd ios && xcodegen generate`
3. `open OpenpilotPhone.xcodeproj`
4. Select the target, then Signing & Capabilities: set your Team (change the bundle id if needed).
5. Pick your phone and Run. Allow camera access. Choose compute units in the app and watch ms per run.

## What this deliberately does not do

- No calibration: mounting angles are zero, intrinsics come from the camera format's field of view.
- One camera: the same frame feeds both the narrow and wide model inputs.
- Fixed speed (15 m/s) and constant desire, traffic convention and action times.
- No connection to a car, no control output.
- No driver monitoring.
