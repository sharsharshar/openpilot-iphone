# Sensitivity experiments on real driving footage

All numbers below were measured by the scripts in this directory (logs in the scratch directory). Nothing here is a claim about
how the model behaves in a car; it is the same offline replay as the synthetic experiments, but on road video.

## Data
- Source: openpilot's model-replay test segment, route `98395b7c5b27882e|0000002b--2686b5a2d0`, segment 1
  (`fcamera.hevc` narrow road camera, `ecamera.hevc` wide road camera, `rlog.zst`), from the public `openpilotci` blob store.
- Both videos: HEVC, 1344x760, yuv420p, 20 fps (`r_frame_rate` 20/1), 1200 frames each (`ffprobe -count_frames`). Decoded with
  `ffmpeg -pix_fmt nv12 -f rawvideo` (first 600 frames per camera, 0.92 GB each; kept outside the repo).
- Log parsed with pycapnp + zstandard (`extract_log.py`; schemas `log.capnp`, `custom.capnp`, `deprecated.capnp` from openpilot master, `car.capnp` from the opendbc repo).
- Device and sensor from the log: `deviceState.deviceType = mici`, `narrowRoadCameraState.sensor = wideRoadCameraState.sensor = os04c10`.
  openpilot `camera.py` `DEVICE_CAMERAS[("mici","os04c10")]` is `_os_config`: narrow 1344x760 focal 1522*3/4 = 1141.5 px, wide 1344x760 focal 567/4*3 = 425.25 px, principal point at the frame centre (672, 380). Those are the intrinsics used.
- Calibration: `extrinsicsCalibration.rpyCalib` (status `calibrated`, 100 %) = [-1.72e-05, -0.029515, -0.006385] rad, the message valid at the first frame used; it changes by at most 7e-5 rad over the stretch, so it is held constant. The same angles are used for both cameras, as `modeld.py` does.
- Frame alignment: narrow and wide `frameId` are identical for all 1200 frames (1203..2402) and start-of-frame times differ by at most 0.02 ms. Video frame i has frameId 1203+i.
- Stretch used: video frames 0..599 (30 s, continuous, 20 Hz). `carState.vEgo` (interpolated to the camera-state message time) is 11.8 to 22.2 m/s, mean 17.3. Night-time highway with a lead car, cones and a curve.
- Model inputs: `new_img[0]` = narrow camera warped with `op_warp_matrix(K_narrow, rpyCalib, 0)`, `new_img[1]` = wide camera with `op_warp_matrix(K_wide, rpyCalib, 1)`, via `op_frame_to_model_input` (NV12 stride 1344, height 760). `desire` zeros, `traffic_convention` [1,0], `action_t` [0.2,0.5], state from zeros, float32 model.
- Reference: `real_reference_fp32.npz` (keys `frames` uint8 [600,2,6,128,256], `desire`, `traffic_convention`, `action_t`, `outputs` float32 [600,2576]) plus `real_meta.npz` (v_ego, frame ids, rpy), in the scratch directory.
- Warm steps = steps 100..599 (500 steps). "all" = all 600 (real) or all 40 (synthetic; there is no warm-only for 40 steps).

## Warp check
- `op_warp_matrix` (C) vs the oracle `get_warp_matrix`, float32: max abs difference 0 for both cameras.
- C warp vs tinygrad oracle on real frames (frames 0, 300, 599 of each camera, 196608 samples each): 5 of 6 bitwise identical; narrow frame 300 differs in 1 sample by 1 grey level (a rounding tie on a float32 coordinate; not further investigated). Also every one of the 600x2 warped images has only integer values.
- Images (scratch `png/warped_{0,300,599}.png`, narrow left and wide right, 6-channel YUV converted back to RGB, BT.601; and `png/rawY_*.png` for the raw luma): the road is centred, lane markings converge on a vanishing point close to the model-frame principal point. In the narrow image the horizon / vanishing point is at about y = 50-60 of 256 and x = 250-290 of 512 (openpilot's model frame puts it at (256, 47.6)); in the wide image it is at about y = 150-160 (model frame: 151.8) and x = 250. The lead car's tail-lights sit on the horizon near the centre, cones and dashed lane lines are visible and straight where the road is straight. The wide image shows a wider field of view, no bonnet. The footage is dark (night) with strong motion blur on near objects. I judged this by eye; I did not measure the vanishing point numerically.

## Is the model seeing a road? (float32 model, state from zeros)

### Statistics (from `build_reference.py`; log `build_reference.log` in scratch)
Lane-line probabilities (outer-left, inner-left, inner-right, outer-right), mean over steps:
- real, warm: 0.034, 0.852, 0.942, 0.234 (inner lines min 0.255 / 0.751, max 0.993 / 0.991)
- real, all 600: 0.049, 0.806, 0.935, 0.248
- synthetic, 40 steps: 0.013, 0.047, 0.019, 0.002 (max 0.254 on any line)

Plan speed at t=0 against logged vEgo (warm steps): mean difference +0.34 m/s, mean abs 0.51, RMS 0.62, max abs 1.59, correlation 0.987. First 100 steps: mean abs 0.68. Synthetic plan speed at t=0 averages 4.2 m/s (range 0.96..7.49); the synthetic input has no speed.

Lead probability: real warm mean 0.895 (max 0.999); synthetic mean 0.306 (max 0.595).

Evolution (mean per 100 steps, vEgo / plan v(t=0)): 21.6/21.0, 21.0/21.3, 18.5/19.0, 16.0/16.4, 12.5/13.4, 13.9/13.5 m/s. Lane probabilities stay high on the inner lines throughout (0.58 in the first 100 steps while the state warms up, then 0.67 to 0.95). Lead probability 0.96, 1.00, 1.00, 1.00, 0.94, 0.54 per 100 steps. Mean desired curvature goes -0.0009, -0.0006, -0.0025, +0.0002, +0.0055, +0.0074 1/m (positive curvature is left in openpilot's convention; the road bends that way after about step 400; plan y at 10 s goes from about -6/-32 m to +49 m; sign convention as in openpilot).

Against the device's own `modelV2` (different model version, matched by frameId, 500 warm steps; loose comparison only):
- speed at t=0: mean abs difference 0.43 m/s, correlation 0.994 (device vs vEgo: 0.39 m/s mean abs)
- plan x at 10 s: 144.5 m (device) vs 144.1 m (ours), correlation 0.950
- plan y at 10 s: mean abs difference 3.9 m (our mean abs 23.3 m, device 22.3 m), correlation 0.986
- desired curvature: mean abs difference 0.00026 1/m (typical magnitude 0.0033), correlation 0.998
- desired acceleration: mean abs difference 0.14 m/s^2 (typical magnitude 0.31), correlation 0.850
- inner lane-line probabilities: device 0.847 / 0.928, ours 0.852 / 0.942; lead probability device 0.924, ours 0.895 (correlation 0.951)
These device-vs-ours differences are a yardstick: two different model versions fed (nearly) the same video differ by about this much.

## Scale for the driving-relevant columns
- openpilot's `model_replay.py` compares replay output with a stored reference. On a PC it ignores `position`, lane lines, road edges, leads and path coefficients and compares the rest with tolerance 0.3 (`compare_logs`: `|a-b| <= max(0.3, 0.3*max(|a|,|b|))`), which includes `desiredCurvature` and `desiredAcceleration`; on the device the tolerance is the log-compare EPSILON (effectively exact). So upstream's own PC tolerance is very loose, and its exact on-device check is a regression test, not a driving requirement.
- Curvature to lateral acceleration: a curvature error of d (1/m) at speed v is a lateral acceleration error of v^2 d. At 17 m/s, 0.001 1/m is 0.29 m/s^2; at 22 m/s, 0.48 m/s^2. Acceleration is already in m/s^2.
- I did not find an openpilot number that says what plan-position or lane-probability error is acceptable, so those columns are reported without a threshold.

## Float16 model vs float32 reference
The raw numbers are for `models/driving_supercombo.onnx` run on the same frames, state from zeros, compared with the float32 reference.
Note on the synthetic baseline: the task text and earlier docs give worst 1.71 / mean 0.010 for float16 on synthetic frames. That is the difference between the stored `reference/driving_reference.npz` (float16 outputs generated earlier) and the float32 reference; I reproduced it (1.714 / 0.0254 worst step mean / 0.01025). A fresh float16 run on this machine (onnxruntime 1.32.0) gives 0.351 / 0.0057 / 0.00318 against the float32 reference, and differs from the stored float16 reference by 1.625 / 0.0219 / 0.0096. So the float16 CPU result is not reproducible between runs of different environments, and the synthetic comparison below uses the fresh run. I did not find out which part of the environment changed.

## Reading the tables
Real warm = steps 100..599; synthetic = all 40 steps with a fixed speed of 15 m/s for the action calculation (the synthetic data has no speed). Differences are absolute values; "driving" tables show mean / worst over the steps. "plan x/y" are forward and lateral position in metres at T_IDXS[14] = 1.91 s and T_IDXS[32] = 10 s. Lane-line probability difference is taken per line and averaged over the 4 lines. The "float16 weights (what ships today)" row in the compression table is 0 by construction: the float32 model is an exact upcast of the float16 weights.
The raw "real all steps" columns equal what the unmodified `gelu_variants.py` and `quantize_study.py` print on the real reference (checked: the logs `existing_gelu.log` and `existing_quant.log` match, run through `run_existing.py`, which only turns the npz into in-memory arrays because those scripts re-read the whole 1.2 GB frames array at every step when given an npz).


### Float16 model vs float32 reference: raw output differences (all 2576 outputs)

| variant | real warm: worst / worst-step-mean / mean | real all steps | synthetic (40 steps) |
|---|---|---|---|
| float16 model (original ONNX) | 0.179 / 0.0025 / 0.00098 | 0.216 / 0.0033 / 0.00098 | 0.351 / 0.0057 / 0.00318 |

### Float16 model vs float32 reference: driving-relevant differences, mean / worst over steps (real = warm steps 100-599; synthetic = all 40, speed assumed 15 m/s)

| variant | data | curvature (1/m) | accel (m/s^2) | plan x @1.9s (m) | plan y @1.9s (m) | plan x @10s (m) | plan y @10s (m) | lane-line prob | lead prob |
|---|---|---|---|---|---|---|---|---|---|
| float16 model (original ONNX) | real | 1e-06 / 1.5e-05 | 0.0042 / 0.018 | 0.0062 / 0.019 | 0.00029 / 0.0027 | 0.026 / 0.072 | 0.0078 / 0.051 | 0.00025 / 0.0023 | 3.8e-05 / 0.00093 |
| float16 model (original ONNX) | synthetic | 1.5e-06 / 5.7e-06 | 0.001 / 0.0022 | 0.021 / 0.046 | 0.00047 / 0.0018 | 0.093 / 0.22 | 0.0053 / 0.016 | 7.2e-05 / 0.0013 | 0.00082 / 0.0024 |

### GELU variants (float32 model): raw output differences (all 2576 outputs)

| variant | real warm: worst / worst-step-mean / mean | real all steps | synthetic (40 steps) |
|---|---|---|---|
| tanh form (as shipped) | 0.000 / 0.0000 / 0.00000 | 0.000 / 0.0000 / 0.00000 | 0.000 / 0.0000 / 0.00000 |
| exact erf form | 1.438 / 0.0162 / 0.00438 | 1.438 / 0.0162 / 0.00421 | 3.498 / 0.0492 / 0.02235 |
| x * sigmoid(1.702 x) | 71.878 / 0.7897 / 0.26480 | 71.878 / 0.7897 / 0.24649 | 132.152 / 2.2474 / 1.27011 |
| piecewise linear, 8 segments | 191.214 / 4.9058 / 2.71129 | 191.214 / 4.9058 / 2.65623 | 294.453 / 4.6469 / 3.61326 |
| piecewise linear, 16 segments | 91.414 / 2.2163 / 0.82730 | 91.414 / 2.2163 / 0.77570 | 189.540 / 2.5848 / 1.58891 |
| piecewise linear, 32 segments | 65.827 / 0.7836 / 0.24067 | 65.827 / 0.7836 / 0.23079 | 135.559 / 2.0200 / 0.82674 |
| piecewise linear, 64 segments | 22.391 / 0.2402 / 0.07010 | 22.391 / 0.2402 / 0.06622 | 45.073 / 0.6699 / 0.25558 |

### GELU variants (float32 model): driving-relevant differences, mean / worst over steps (real = warm steps 100-599; synthetic = all 40, speed assumed 15 m/s)

| variant | data | curvature (1/m) | accel (m/s^2) | plan x @1.9s (m) | plan y @1.9s (m) | plan x @10s (m) | plan y @10s (m) | lane-line prob | lead prob |
|---|---|---|---|---|---|---|---|---|---|
| tanh form (as shipped) | real | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| tanh form (as shipped) | synthetic | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| exact erf form | real | 2.3e-06 / 1.7e-05 | 0.002 / 0.0056 | 0.011 / 0.048 | 0.0014 / 0.0087 | 0.12 / 0.66 | 0.066 / 0.35 | 0.00075 / 0.014 | 0.00034 / 0.0094 |
| exact erf form | synthetic | 1.6e-05 / 3.8e-05 | 0.0036 / 0.0076 | 0.08 / 0.26 | 0.0038 / 0.0081 | 0.4 / 1.1 | 0.027 / 0.073 | 0.00026 / 0.0027 | 0.0028 / 0.008 |
| x * sigmoid(1.702 x) | real | 0.00014 / 0.0012 | 0.094 / 0.61 | 0.93 / 3.6 | 0.098 / 0.66 | 7.3 / 56 | 3.9 / 22 | 0.037 / 0.38 | 0.017 / 0.36 |
| x * sigmoid(1.702 x) | synthetic | 0.00069 / 0.0023 | 0.076 / 0.25 | 11 / 24 | 0.18 / 0.37 | 56 / 1.2e+02 | 1.9 / 5.5 | 0.045 / 0.2 | 0.14 / 0.31 |
| piecewise linear, 8 segments | real | 0.0056 / 0.013 | 0.3 / 1.4 | 5.8 / 12 | 2.3 / 5 | 42 / 98 | 37 / 82 | 0.21 / 0.78 | 0.89 / 1 |
| piecewise linear, 8 segments | synthetic | 0.0056 / 0.0089 | 0.17 / 0.36 | 27 / 39 | 2 / 3.4 | 1.3e+02 / 1.9e+02 | 14 / 29 | 0.27 / 0.81 | 0.16 / 0.47 |
| piecewise linear, 16 segments | real | 0.00035 / 0.0024 | 0.19 / 1 | 2.8 / 7.5 | 0.14 / 0.9 | 22 / 91 | 8.1 / 31 | 0.15 / 0.47 | 0.14 / 0.49 |
| piecewise linear, 16 segments | synthetic | 0.00066 / 0.0012 | 0.14 / 0.22 | 13 / 37 | 0.25 / 0.68 | 63 / 1.9e+02 | 3.6 / 9.9 | 0.11 / 0.33 | 0.12 / 0.43 |
| piecewise linear, 32 segments | real | 0.00013 / 0.00056 | 0.071 / 0.39 | 0.73 / 1.8 | 0.068 / 0.24 | 5.2 / 29 | 2.5 / 17 | 0.035 / 0.44 | 0.019 / 0.26 |
| piecewise linear, 32 segments | synthetic | 0.00036 / 0.0011 | 0.078 / 0.28 | 4.3 / 11 | 0.13 / 0.38 | 20 / 56 | 1.2 / 3.5 | 0.032 / 0.19 | 0.068 / 0.19 |
| piecewise linear, 64 segments | real | 4e-05 / 0.00025 | 0.024 / 0.1 | 0.15 / 0.64 | 0.019 / 0.12 | 1.6 / 12 | 0.65 / 4.4 | 0.013 / 0.16 | 0.0041 / 0.087 |
| piecewise linear, 64 segments | synthetic | 0.00012 / 0.00052 | 0.03 / 0.082 | 0.89 / 2.9 | 0.034 / 0.13 | 4.4 / 14 | 0.21 / 0.75 | 0.0048 / 0.05 | 0.028 / 0.086 |

### Weight compression (float32 model, weights replaced): raw output differences (all 2576 outputs)

| variant | real warm: worst / worst-step-mean / mean | real all steps | synthetic (40 steps) |
|---|---|---|---|
| float16 weights (what ships today) | 0.000 / 0.0000 / 0.00000 | 0.000 / 0.0000 / 0.00000 | 0.000 / 0.0000 / 0.00000 |
| 8-bit linear, per output channel | 72.610 / 0.6056 / 0.13609 | 72.610 / 0.6056 / 0.12901 | 75.860 / 1.1822 / 0.69808 |
| 8-bit linear, per tensor | 177.093 / 4.1258 / 2.09063 | 177.093 / 4.1258 / 1.98945 | 263.918 / 3.9920 / 3.13864 |
| 8-bit lookup table (256 values per tensor) | 210.124 / 3.8395 / 1.56232 | 243.617 / 3.8395 / 1.54159 | 138.923 / 2.9935 / 1.42448 |
| 6-bit lookup table (64 values) | 240.245 / 5.1016 / 3.28074 | 248.418 / 5.1016 / 3.27792 | 110.477 / 1.9719 / 1.32202 |
| 4-bit lookup table (16 values) | 169.198 / 4.4828 / 3.10374 | 198.577 / 4.4828 / 3.14147 | 175.943 / 2.4572 / 2.06762 |

### Weight compression (float32 model, weights replaced): driving-relevant differences, mean / worst over steps (real = warm steps 100-599; synthetic = all 40, speed assumed 15 m/s)

| variant | data | curvature (1/m) | accel (m/s^2) | plan x @1.9s (m) | plan y @1.9s (m) | plan x @10s (m) | plan y @10s (m) | lane-line prob | lead prob |
|---|---|---|---|---|---|---|---|---|---|
| float16 weights (what ships today) | real | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| float16 weights (what ships today) | synthetic | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 8-bit linear, per output channel | real | 6.8e-05 / 0.00044 | 0.15 / 0.27 | 0.26 / 1.5 | 0.022 / 0.14 | 4.6 / 33 | 1.7 / 12 | 0.02 / 0.21 | 0.0092 / 0.15 |
| 8-bit linear, per output channel | synthetic | 0.00022 / 0.00073 | 0.053 / 0.088 | 1.5 / 4.2 | 0.068 / 0.23 | 6.7 / 19 | 0.91 / 4.2 | 0.0061 / 0.077 | 0.077 / 0.23 |
| 8-bit linear, per tensor | real | 0.0022 / 0.0065 | 3.6 / 4.9 | 12 / 23 | 1.2 / 3.4 | 90 / 1.8e+02 | 28 / 73 | 0.49 / 0.98 | 0.16 / 0.82 |
| 8-bit linear, per tensor | synthetic | 0.0037 / 0.0061 | 2.7 / 4.3 | 31 / 48 | 1.6 / 2.9 | 1.5e+02 / 2.4e+02 | 31 / 40 | 0.085 / 0.29 | 0.14 / 0.51 |
| 8-bit lookup table (256 values per tensor) | real | 0.0014 / 0.0079 | 0.32 / 1.3 | 9.6 / 36 | 0.71 / 3 | 54 / 1.8e+02 | 15 / 54 | 0.22 / 0.96 | 0.35 / 0.7 |
| 8-bit lookup table (256 values per tensor) | synthetic | 0.0011 / 0.0034 | 0.86 / 1.3 | 8.4 / 15 | 0.42 / 1.4 | 54 / 91 | 1.6 / 6.3 | 0.074 / 0.23 | 0.25 / 0.53 |
| 6-bit lookup table (64 values) | real | 0.0033 / 0.014 | 0.42 / 1.5 | 30 / 41 | 1.3 / 4.7 | 1.3e+02 / 2e+02 | 23 / 72 | 0.49 / 0.96 | 0.33 / 0.69 |
| 6-bit lookup table (64 values) | synthetic | 0.0019 / 0.0042 | 1 / 1.6 | 4.7 / 10 | 0.61 / 1.6 | 32 / 63 | 10 / 23 | 0.019 / 0.25 | 0.14 / 0.35 |
| 4-bit lookup table (16 values) | real | 0.0042 / 0.018 | 0.38 / 0.91 | 23 / 35 | 1.5 / 5.4 | 92 / 1.6e+02 | 25 / 70 | 0.42 / 0.97 | 0.31 / 0.68 |
| 4-bit lookup table (16 values) | synthetic | 0.0029 / 0.0055 | 1.3 / 1.9 | 7.7 / 15 | 0.64 / 1.6 | 41 / 83 | 9.7 / 18 | 0.25 / 0.8 | 0.27 / 0.6 |

## Observations (read off the tables above)
- Float16 model: real warm worst raw difference 0.179, mean 0.00098; driving-relevant: curvature 1e-6 mean / 1.5e-5 worst 1/m, acceleration 0.004 / 0.018 m/s^2, plan position at 10 s 0.03 / 0.07 m forward.
- GELU: the exact (erf) form gives raw worst 1.44 / mean 0.0044 on real against 3.50 / 0.022 on synthetic; driving: curvature 2e-6 mean / 1.7e-5 worst, acceleration 0.002 / 0.006, plan x at 10 s 0.12 / 0.66 m, lane probability 0.0008 / 0.014. The sigmoid approximation gives raw 71.9 / 0.265 (synthetic 132 / 1.27); driving: curvature 1.4e-4 mean / 1.2e-3 worst, acceleration 0.09 / 0.61, plan x at 10 s 7.3 / 56 m, lane probability 0.037 / 0.38. Piecewise-linear GELU improves with segment count (64 segments: curvature 4e-5 / 2.5e-4, acceleration 0.024 / 0.10, plan y at 10 s 0.65 / 4.4 m, lane probability 0.013 / 0.16; 8 segments is badly off: curvature 0.0056 mean, lead probability off by 0.89 on average).
- Weight compression: 8-bit per output channel gives raw 72.6 / 0.129 on real (synthetic 75.9 / 0.70); driving: curvature 6.8e-5 mean / 4.4e-4 worst, acceleration 0.15 / 0.27, plan x at 10 s 4.6 / 33 m, lane probability 0.020 / 0.21. 8-bit per tensor and the 8/6/4-bit lookup tables are much worse (curvature mean 1.4e-3 to 4.2e-3 1/m, lane-probability mean 0.22 to 0.49; plan x at 10 s mean 54 to 130 m). Per-tensor 8-bit acceleration mean 3.6 m/s^2.
- Raw (absolute) sensitivity on real footage is lower than on synthetic for float16 (about 2x worst, 3x mean), the GELU variants (1.5x to 2.4x worst, 1.3x to 5x mean) and 8-bit per channel (similar worst, 5x lower mean). It is not lower for the lookup tables: 6-bit and 4-bit have a larger mean raw difference on real (3.28, 3.14) than on synthetic (1.32, 2.07), and 8-bit LUT is about the same (1.54 vs 1.42). So on real footage the model is still highly sensitive to coarse weight compression and to a crude GELU, and the sensitivity to GELU form and float16 is smaller than the synthetic numbers suggested but not absent.
- Yardstick from the device comparison (two model versions on the same video): curvature 2.6e-4 mean abs, acceleration 0.14 m/s^2, plan y at 10 s 3.9 m. Variants with mean differences well below these: float16, exact GELU. Variants at or above them: sigmoid GELU (accel worst 0.61), 8-bit per channel (acceleration mean 0.15, plan x at 10 s up to 33 m), 64-segment PWL (y at 10 s worst 4.4 m), and everything else.

## Fixtures (scratch only, not in the repo)
`frames_u8.bin` (40 x [2,6,128,256] uint8, 15,728,640 bytes) and `outputs_f32.bin` (40 x 2576 float32 little-endian, 412,160 bytes) in the scratch directory: real frames 300..339 (video frames, warm region), expected outputs recomputed with the float32 model from zero state over exactly those 40 frames. Step 0 of these differs from the long-run output at the same frame (mean abs 1.49) because the state is cold; step 39 matches the long run to 0.0000 (mean abs, 4 decimals) - an observation, see Not checked.

## Not checked
- Whether the HEVC-decoded frames equal what the camera ISP gave modeld on the car (the video is lossy; the on-device model input is from raw NV12). Colour range handling of ffmpeg's decode was not inspected.
- The vanishing point in the warped images was judged by eye, not measured.
- One mismatching sample (of 196,608) in the narrow frame 300 warp was not traced to its cause.
- Only one segment (30 s, 600 frames used, night-time highway) was used; results for daylight, city, low speed or other cameras are unknown. The first 100 steps are cold-start and are excluded from "warm".
- No closed-loop driving, no steering/longitudinal controller, no comparison on an iPhone, Core ML or iOS (only the onnxruntime CPU emulations of the existing experiments).
- "What matters for driving" thresholds are not established; the only external numbers are upstream's replay tolerance and the device-vs-ours yardstick. Lane-line, plan-position and lead errors have no justified threshold here.
- The float16 baseline discrepancy (1.71 stored vs 0.35 fresh on synthetic) was not diagnosed.
- Quantisation variants are weight-only emulations in float32 as in `quantize_study.py` (no activation quantisation, no real Core ML/ANE kernels).
- Speed used for synthetic action calculations is an assumed 15 m/s.
- Fixture step-39 agreement with the long run was observed (0.0000 mean abs) but not tested for being exact.

## Files
`extract_log.py` (log to pickle), `build_reference.py` (decode, warp, oracle check, PNGs, reference, fixtures, sanity statistics), `rf_common.py`, `sensitivity.py` (float16, GELU and compression on real and synthetic; `report` prints the tables), `run_existing.py` (runs the existing scripts unchanged on the big npz).
