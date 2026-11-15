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
