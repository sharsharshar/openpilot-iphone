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

## More clips

The question: do the conclusions from the one night-highway clip hold on other footage? Five more segments were found, a reference was built for each
with the same pipeline (generalised, see Files), the same sanity checks were run, and the same sensitivity study was repeated on each. All numbers below were
measured by the scripts in this directory; the per-clip data (video, logs, references, `sens_*.json`) lives in a scratch directory outside the repo.

### Where the segments came from
- The openpilot process-replay list (`selfdrive/test/process_replay/test_processes.py`, copy in the scratch directory) names 17 real-car segments (and a set of regenerated `regen...` ones). I sent HTTP HEAD requests for
  `fcamera.hevc`, `ecamera.hevc`, `rlog.zst`, `rlog.bz2` (and `qlog.zst`) for each under `https://commadataci.blob.core.windows.net/openpilotci/<dongle>/<route>/<segment>/`.
  - Single road camera only (`ecamera.hevc` is 404; 2021 recordings, probably comma two or early tici, not checked; logs are `rlog.bz2`): HYUNDAI Sonata, TOYOTA Prius, TOYOTA RAV4, HONDA Civic, HONDA Accord, CHRYSLER Pacifica, NISSAN X-Trail, VOLKSWAGEN Golf. Not usable.
  - Both cameras and a log: HYUNDAI2 (Kia EV6), TOYOTA3 (Corolla TSS2), RAM, SUBARU, GM (Bolt EUV), MAZDA, FORD, RIVIAN, TESLA (Model Y). Logs are `rlog.bz2` for the 2021 to 2023 recordings (Python's `bz2`; `extract_log.py` now accepts both) and `rlog.zst` for TOYOTA3, RIVIAN and TESLA.
  - Not built although they qualify: SUBARU (daylight overcast highway at 32 m/s; sensor field unknown = AR0231), FORD (dusk/night, 6 to 16 m/s then 30 m/s), RIVIAN (night highway, 25 to 28 m/s; about the same condition as the first clip) and MAZDA (narrow has 1201 frames and the narrow/wide frame ids are offset by one; upstream's own comment calls its sensor timing bad; not used). I stopped at five clips for time and disk.
  - Five of the `regen...` segments in the list (2024/2025) that I probed have `rlog.zst` only, no video. The `tools/replay` demo route `a2a0ccea32023010|2023-07-27--13-01-19` (segments 0 and 1): all five file names are 404 in `openpilotci`.
- Hugging Face: `commaai/commaCarSegments` (the dataset upstream's tooling refers to; manifest 9.4 MB) contains `rlog.zst` only, by its README structure, so there is no video. I did not look at other comma datasets (comma2k19 is known to have one road camera on a different device; commaVQ is tokenised video), so "no other source" is not established.
- The GitHub API is blocked for the commaai organisation; not used. No host other than the ones named in the task was needed, and none was refused.

### Clips
Each is the first 600 frames (30 s, 20 Hz) of a 1200-frame (60 s) segment; narrow and wide video both have 1200 frames and identical frame ids (ffprobe `-count_frames`, `frameId` from the log). Steps 100..599 are "warm", as before.
Intrinsics are from openpilot's `DEVICE_CAMERAS` (`rf_common.camera_config`): `_os_config` (1344x760, narrow f 1141.5, wide f 425.25) for the mici/os04c10 clips, `_ar_ox_config` (1928x1208, narrow f 2648, wide f 567) for tici/tizi with ar0231/ox03c10. Principal point at the frame centre. Calibration: `extrinsicsCalibration.rpyCalib` where the log has it; the four older logs have only `liveCalibration.rpyCalib`, used the same way (status reads `uncalibrated` with `calPerc` 100 in those logs, as parsed with the master schema; the angles are steady, spread over the 30 s at most 1.4e-5 to 1e-4 rad). One angle triple for both cameras, as before.

| name | route | device / sensor | video | vEgo over the 600 frames (min / mean / max, m/s) | conditions (from the frames and the speed trace) |
|---|---|---|---|---|---|
| original | `98395b7c5b27882e\|0000002b--2686b5a2d0` seg 1 | mici / os04c10 | 1344x760 | 11.8 / 17.3 / 22.2 | night highway, lead car, cones, a curve |
| tesla | `aebd8f1d4ea16066\|000002d8--5850d3ea5c` seg 2 (Tesla Model Y) | mici / os04c10 (same device and sensor as the original) | 1344x760 | 1.3 / 15.4 / 27.5 | daylight; decelerates from 27 m/s to about 1.7 m/s on an arterial road towards a junction with traffic lights (visible in frame 300), lead vehicle at the end |
| gm | `376bf99325883932\|2022-10-27--13-41-22` seg 1 (Chevrolet Bolt EUV) | tici (initData) / ox03c10 | 1928x1208 | 3.3 / 6.4 / 9.9 | daylight residential streets, double yellow line, no right lane line, sharp turns (|curvature| up to 0.06 1/m), stop line |
| toyota3 | `8011d605be1cbb77\|000000cc--8e8d8ec716` seg 6 (Corolla TSS2) | tizi / ox03c10 | 1928x1208 | 30.1 / 30.3 / 30.5 | dusk/daylight freeway at constant 30 m/s, adjacent traffic, gentle curve at the start, an exit ramp visible in frame 300 |
| ram | `17fc16d840fe9d21\|2023-04-26--13-28-44` seg 5 (Ram 1500) | tici (initData) / ox03c10 | 1928x1208 | 5.6 / 17.5 / 24.8 | daylight highway, then a slow turn at about 7 m/s (curvature up to 0.05 1/m), then back to 15 m/s; lead vehicle part of the time |
| hyundai2 | `d545129f3ca90f28\|2022-11-07--20-43-08` seg 3 (Kia EV6) | tici (initData) / ar0231 | 1928x1208 | 6.7 / 11.8 / 13.2 | night, lit urban road, double yellow on the left only, lead car throughout; a different sensor (AR0231) |

Device/sensor/stats were read from the logs. "tici (initData)": `deviceState.deviceType` is unknown in those older logs, so `initData.deviceType` was used (the config is the same for tici, tizi and unknown with these sensors).
The sensor comes from `cameraState.sensor` in each log (gm, ram: ox03c10; hyundai2: ar0231).

### Sanity results per clip (float32 model, state from zeros; `build_reference.py`, logs `build_reference.log` in each clip directory)
Warps were judged by eye from `png/warped_{0,300,599}.png` (narrow left, wide right), and plan speed and the device's own `modelV2` were compared numerically. None of the five clips failed.

| clip | warp seen by eye | C matrix vs oracle | C warp vs tinygrad oracle (6 frames, 196,608 samples each) | inner lane-line prob (L / R), warm mean | lead prob, warm mean | plan v(t=0) minus vEgo, warm: mean abs / corr | vs device modelV2 (500 warm steps) |
|---|---|---|---|---|---|---|---|
| original | road centred, VP near model centre | 0 | 5 of 6 identical, 1 sample off by 1 | 0.852 / 0.942 | 0.895 | 0.51 / 0.987 | speed 0.43 m/s corr 0.994; y@10s 3.9 m corr 0.986; curvature corr 0.998 |
| tesla | road centred; lane edge, dashes, junction vanish at horizon in both views | 0 | 2 of 6 identical, the other 4 differ in 1 sample (grey-level differences 11, 9, 1, 3) | 0.829 / 0.801 | 0.746 | 0.39 / 0.999 | speed 0.16 m/s corr 1.000; y@10s 0.46 m corr 0.991; curvature 1.5e-4 corr 0.971; accel 0.04 corr 0.996 |
| gm | double yellow converges at the horizon in frame 0; frame 300 is mid-turn facing a wall, frame 599 at a stop line with the cross street visible; horizon plausible in both views | 0 | 5 of 6 identical, 1 sample off by 4 | 0.277 / 0.059 (weak lines: device model 0.345 / 0.042) | 0.076 | 0.49 / 0.964 | speed 0.42 m/s corr 0.966; y@10s 9.8 m corr 0.916; lead corr 0.48; no `modelV2.action` in this log |
| toyota3 | freeway lines converge at the horizon, lead cars and gantry sit on it | 0 | 3 of 6 identical (all wide), narrow off by 2 to 3 samples (diff up to 4) | 0.976 / 0.918 | 0.013 | 0.19 / not defined (speed constant; corr -0.09) | speed 0.60 m/s; y@10s 1.1 m corr 0.994; curvature 5e-5 corr 0.991 |
| ram | lane lines and lead truck as expected; bonnet is outside the cropped model view | 0 | 1 of 6 identical, the others off by 1 to 3 samples (diff up to 3) | 0.661 / 0.677 | 0.327 | 0.56 / 0.996 | speed 1.7 m/s corr 0.993 (device vs vEgo 2.1 m/s); y@10s 7.5 m corr 0.787; no `modelV2.action` |
| hyundai2 | double yellow and lead tail-lights at the horizon, dark; the bottom rows of the narrow image repeat the last source row (the warp samples below the frame edge) | 0 | 0 of 6 identical, 1 to 4 samples off (diff up to 4) | 0.934 / 0.109 (device model 0.967 / 0.192) | 0.980 | 0.30 / 0.979 | speed 0.73 m/s corr 0.947; y@10s 0.43 m corr 0.94; no `modelV2.action` |

All 600 x 2 warped images are integer-valued in every clip. The scripts print the oracle mismatches (a rounding tie on a float32 coordinate is the likely cause, as for the original clip; not traced).
Where the original clip's statistics are quoted they are the ones in the section above; re-running the generalised scripts on the first clip reproduced `real_reference_fp32.npz`, `real_meta.npz`, `log.pkl` fields and the fixtures bit for bit (np.array_equal / cmp).

### Sensitivity results across clips
Same variants and metrics as above, computed by the unchanged `sensitivity.py` with `RF_SCRATCH` pointing at each clip directory (it runs the original ONNX float16 model and the variants against that clip's float32 reference, and again against the synthetic reference, which gave the same synthetic numbers every time).
`compare_clips.py` prints these tables. The PWL 8 and 16 segment and the 8/6/4-bit lookup-table rows are in each clip's `sens_*.json`; only a subset is in the tables below (PWL 8/16 are worse than 32 on every clip).
Column "original" is the first clip. 500 warm steps per clip.

##### Mean over the 500 warm steps of |variant - float32 reference|

**raw output difference over all 2576 outputs**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.00098 | 0.0008 | 0.0013 | 0.0011 | 0.001 | 0.00069 |
| exact erf form | 0.0044 | 0.0036 | 0.0067 | 0.0024 | 0.0046 | 0.0021 |
| x * sigmoid(1.702 x) | 0.26 | 0.23 | 0.5 | 0.14 | 0.31 | 0.15 |
| piecewise linear, 64 segments | 0.07 | 0.051 | 0.11 | 0.039 | 0.066 | 0.046 |
| piecewise linear, 32 segments | 0.24 | 0.17 | 0.41 | 0.16 | 0.24 | 0.14 |
| 8-bit linear, per output channel | 0.14 | 0.095 | 0.16 | 0.06 | 0.13 | 0.065 |
| 8-bit linear, per tensor | 2.1 | 2.1 | 3.1 | 2.1 | 1.9 | 2.3 |
| 8-bit lookup table (256 values per tensor) | 1.6 | 0.89 | 1.7 | 0.71 | 1.2 | 0.84 |
| 6-bit lookup table (64 values) | 3.3 | 2.5 | 2.8 | 5.2 | 3.3 | 2.4 |
| 4-bit lookup table (16 values) | 3.1 | 3.3 | 3.7 | 3.6 | 3.2 | 2.3 |

**desired curvature (1/m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 1e-06 | 7.4e-07 | 8e-06 | 1.3e-07 | 4.1e-06 | 6.4e-07 |
| exact erf form | 2.3e-06 | 3.2e-06 | 2.6e-05 | 5.5e-07 | 1e-05 | 2.5e-06 |
| x * sigmoid(1.702 x) | 0.00014 | 0.0002 | 0.0028 | 2.4e-05 | 0.0011 | 0.00015 |
| piecewise linear, 64 segments | 4e-05 | 3.4e-05 | 0.00042 | 8.9e-06 | 0.0002 | 3.7e-05 |
| piecewise linear, 32 segments | 0.00013 | 9.4e-05 | 0.0018 | 2.5e-05 | 0.00077 | 0.00011 |
| 8-bit linear, per output channel | 6.8e-05 | 0.00013 | 0.00055 | 1.5e-05 | 0.00029 | 5.2e-05 |
| 8-bit linear, per tensor | 0.0022 | 0.0036 | 0.023 | 0.001 | 0.0098 | 0.002 |
| 8-bit lookup table (256 values per tensor) | 0.0014 | 0.00037 | 0.02 | 7.8e-05 | 0.0071 | 0.00076 |
| 6-bit lookup table (64 values) | 0.0033 | 0.00088 | 0.024 | 0.00031 | 0.012 | 0.0013 |
| 4-bit lookup table (16 values) | 0.0042 | 0.021 | 0.036 | 0.0031 | 0.02 | 0.0019 |

**desired acceleration (m/s^2)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.0042 | 0.0029 | 0.0019 | 0.0055 | 0.0041 | 0.003 |
| exact erf form | 0.002 | 0.0024 | 0.0018 | 0.001 | 0.0026 | 0.00089 |
| x * sigmoid(1.702 x) | 0.094 | 0.16 | 0.12 | 0.023 | 0.17 | 0.044 |
| piecewise linear, 64 segments | 0.024 | 0.028 | 0.031 | 0.0054 | 0.022 | 0.023 |
| piecewise linear, 32 segments | 0.071 | 0.095 | 0.12 | 0.011 | 0.092 | 0.052 |
| 8-bit linear, per output channel | 0.15 | 0.2 | 0.23 | 0.27 | 0.2 | 0.19 |
| 8-bit linear, per tensor | 3.6 | 3.3 | 2.6 | 2.7 | 3.2 | 3.2 |
| 8-bit lookup table (256 values per tensor) | 0.32 | 0.5 | 0.57 | 0.39 | 0.53 | 0.3 |
| 6-bit lookup table (64 values) | 0.42 | 1.1 | 0.65 | 0.061 | 1.3 | 0.67 |
| 4-bit lookup table (16 values) | 0.38 | 0.32 | 1.2 | 0.94 | 1.2 | 0.5 |

**plan x at 10 s (m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.026 | 0.018 | 0.014 | 0.062 | 0.025 | 0.017 |
| exact erf form | 0.12 | 0.091 | 0.067 | 0.062 | 0.092 | 0.078 |
| x * sigmoid(1.702 x) | 7.3 | 6.3 | 4.4 | 3.4 | 6.5 | 5.4 |
| piecewise linear, 64 segments | 1.6 | 1.1 | 1.2 | 0.89 | 0.96 | 1.5 |
| piecewise linear, 32 segments | 5.2 | 4.4 | 4.2 | 4 | 4.4 | 3.3 |
| 8-bit linear, per output channel | 4.6 | 2.1 | 0.97 | 0.68 | 2.3 | 3.1 |
| 8-bit linear, per tensor | 90 | 1.4e+02 | 1.4e+02 | 41 | 94 | 1.5e+02 |
| 8-bit lookup table (256 values per tensor) | 54 | 48 | 21 | 9 | 43 | 37 |
| 6-bit lookup table (64 values) | 1.3e+02 | 73 | 41 | 2.9e+02 | 1.1e+02 | 73 |
| 4-bit lookup table (16 values) | 92 | 83 | 73 | 1.5e+02 | 45 | 48 |

**plan y at 10 s (m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.0078 | 0.002 | 0.0085 | 0.0045 | 0.0082 | 0.00048 |
| exact erf form | 0.066 | 0.01 | 0.043 | 0.013 | 0.046 | 0.0019 |
| x * sigmoid(1.702 x) | 3.9 | 0.49 | 3.9 | 0.59 | 3.7 | 0.16 |
| piecewise linear, 64 segments | 0.65 | 0.16 | 0.65 | 0.12 | 0.51 | 0.044 |
| piecewise linear, 32 segments | 2.5 | 0.6 | 3.2 | 0.39 | 2 | 0.13 |
| 8-bit linear, per output channel | 1.7 | 0.54 | 1.2 | 0.36 | 1.5 | 0.19 |
| 8-bit linear, per tensor | 28 | 16 | 32 | 27 | 24 | 21 |
| 8-bit lookup table (256 values per tensor) | 15 | 1.9 | 17 | 4.2 | 13 | 0.7 |
| 6-bit lookup table (64 values) | 23 | 5.5 | 21 | 10 | 17 | 1.2 |
| 4-bit lookup table (16 values) | 25 | 32 | 38 | 23 | 33 | 3.2 |

**lane-line probability**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.00025 | 0.00017 | 8e-05 | 0.00011 | 0.00011 | 0.00017 |
| exact erf form | 0.00075 | 0.00065 | 0.00029 | 0.00073 | 0.00047 | 0.00038 |
| x * sigmoid(1.702 x) | 0.037 | 0.037 | 0.014 | 0.045 | 0.029 | 0.016 |
| piecewise linear, 64 segments | 0.013 | 0.009 | 0.0076 | 0.0085 | 0.0084 | 0.0081 |
| piecewise linear, 32 segments | 0.035 | 0.031 | 0.022 | 0.038 | 0.032 | 0.027 |
| 8-bit linear, per output channel | 0.02 | 0.016 | 0.0068 | 0.011 | 0.012 | 0.011 |
| 8-bit linear, per tensor | 0.49 | 0.44 | 0.098 | 0.68 | 0.41 | 0.28 |
| 8-bit lookup table (256 values per tensor) | 0.22 | 0.18 | 0.12 | 0.23 | 0.099 | 0.28 |
| 6-bit lookup table (64 values) | 0.49 | 0.49 | 0.086 | 0.68 | 0.41 | 0.28 |
| 4-bit lookup table (16 values) | 0.42 | 0.18 | 0.35 | 0.25 | 0.28 | 0.27 |

**lead probability**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 3.8e-05 | 0.0001 | 9.1e-05 | 2.6e-05 | 0.00011 | 1.6e-05 |
| exact erf form | 0.00034 | 0.00067 | 0.00045 | 8.7e-05 | 0.00078 | 7.1e-05 |
| x * sigmoid(1.702 x) | 0.017 | 0.036 | 0.021 | 0.0044 | 0.04 | 0.0078 |
| piecewise linear, 64 segments | 0.0041 | 0.0088 | 0.011 | 0.0017 | 0.012 | 0.0016 |
| piecewise linear, 32 segments | 0.019 | 0.029 | 0.042 | 0.0064 | 0.049 | 0.009 |
| 8-bit linear, per output channel | 0.0092 | 0.035 | 0.013 | 0.0039 | 0.021 | 0.0026 |
| 8-bit linear, per tensor | 0.16 | 0.22 | 0.61 | 0.77 | 0.46 | 0.09 |
| 8-bit lookup table (256 values per tensor) | 0.35 | 0.17 | 0.12 | 0.08 | 0.14 | 0.29 |
| 6-bit lookup table (64 values) | 0.33 | 0.45 | 0.39 | 0.46 | 0.43 | 0.53 |
| 4-bit lookup table (16 values) | 0.31 | 0.47 | 0.32 | 0.37 | 0.31 | 0.28 |

##### Worst single step (maximum over the 500 warm steps)

**raw output difference, worst single output of the 2576**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.179 | 0.273 | 0.424 | 0.168 | 0.385 | 0.104 |
| exact erf form | 1.44 | 2.75 | 6.31 | 0.323 | 2.22 | 0.78 |
| x * sigmoid(1.702 x) | 71.9 | 103 | 230 | 19.5 | 89.2 | 31 |
| piecewise linear, 64 segments | 22.4 | 25.9 | 94.7 | 7.05 | 22.2 | 14.2 |
| piecewise linear, 32 segments | 65.8 | 70.4 | 173 | 23.1 | 59.7 | 39.2 |
| 8-bit linear, per output channel | 72.6 | 30.9 | 81.8 | 7.1 | 87.5 | 18.5 |
| 8-bit linear, per tensor | 177 | 237 | 350 | 170 | 217 | 195 |
| 8-bit lookup table (256 values per tensor) | 210 | 130 | 370 | 148 | 222 | 135 |
| 6-bit lookup table (64 values) | 240 | 296 | 288 | 414 | 263 | 156 |
| 4-bit lookup table (16 values) | 169 | 277 | 265 | 263 | 304 | 162 |

**desired curvature (1/m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 1.5e-05 | 5e-06 | 8.3e-05 | 1.2e-06 | 8.8e-05 | 6.7e-06 |
| exact erf form | 1.7e-05 | 2e-05 | 0.00028 | 1.8e-06 | 0.00028 | 2.7e-05 |
| x * sigmoid(1.702 x) | 0.0012 | 0.0014 | 0.054 | 9e-05 | 0.027 | 0.0011 |
| piecewise linear, 64 segments | 0.00025 | 0.00022 | 0.005 | 2.7e-05 | 0.0052 | 0.00023 |
| piecewise linear, 32 segments | 0.00056 | 0.00064 | 0.014 | 8.8e-05 | 0.031 | 0.00073 |
| 8-bit linear, per output channel | 0.00044 | 0.001 | 0.0056 | 4.5e-05 | 0.0046 | 0.00045 |
| 8-bit linear, per tensor | 0.0065 | 0.017 | 0.11 | 0.0022 | 0.095 | 0.0066 |
| 8-bit lookup table (256 values per tensor) | 0.0079 | 0.0035 | 0.12 | 0.00043 | 0.078 | 0.0051 |
| 6-bit lookup table (64 values) | 0.014 | 0.0029 | 0.11 | 0.0012 | 0.1 | 0.0067 |
| 4-bit lookup table (16 values) | 0.018 | 0.098 | 0.12 | 0.0046 | 0.12 | 0.0088 |

**desired acceleration (m/s^2)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.018 | 0.014 | 0.0086 | 0.021 | 0.018 | 0.011 |
| exact erf form | 0.0056 | 0.0078 | 0.0092 | 0.0021 | 0.0066 | 0.0033 |
| x * sigmoid(1.702 x) | 0.61 | 0.62 | 0.66 | 0.07 | 0.88 | 0.22 |
| piecewise linear, 64 segments | 0.1 | 0.1 | 0.12 | 0.017 | 0.075 | 0.13 |
| piecewise linear, 32 segments | 0.39 | 0.36 | 0.46 | 0.037 | 0.63 | 0.29 |
| 8-bit linear, per output channel | 0.27 | 0.39 | 0.43 | 0.32 | 0.32 | 0.33 |
| 8-bit linear, per tensor | 4.9 | 4.6 | 4.9 | 3 | 6.2 | 3.8 |
| 8-bit lookup table (256 values per tensor) | 1.3 | 1.4 | 2 | 0.57 | 1.9 | 0.74 |
| 6-bit lookup table (64 values) | 1.5 | 1.9 | 1.7 | 0.2 | 2.8 | 1.6 |
| 4-bit lookup table (16 values) | 0.91 | 1.1 | 3 | 1.5 | 3.3 | 1.4 |

**plan x at 10 s (m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.072 | 0.11 | 0.05 | 0.13 | 0.087 | 0.067 |
| exact erf form | 0.66 | 0.75 | 0.66 | 0.24 | 0.47 | 0.52 |
| x * sigmoid(1.702 x) | 56 | 44 | 27 | 11 | 22 | 24 |
| piecewise linear, 64 segments | 12 | 6.3 | 7.5 | 3.5 | 6 | 11 |
| piecewise linear, 32 segments | 29 | 25 | 28 | 12 | 26 | 19 |
| 8-bit linear, per output channel | 33 | 15 | 6.5 | 2.5 | 9.8 | 18 |
| 8-bit linear, per tensor | 1.8e+02 | 2.3e+02 | 2.3e+02 | 76 | 2e+02 | 1.9e+02 |
| 8-bit lookup table (256 values per tensor) | 1.8e+02 | 1.1e+02 | 55 | 24 | 1e+02 | 1.2e+02 |
| 6-bit lookup table (64 values) | 2e+02 | 2.2e+02 | 86 | 3e+02 | 2.1e+02 | 1.2e+02 |
| 4-bit lookup table (16 values) | 1.6e+02 | 1.7e+02 | 2e+02 | 2.4e+02 | 1.9e+02 | 1e+02 |

**plan y at 10 s (m)**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.051 | 0.017 | 0.078 | 0.022 | 0.13 | 0.0038 |
| exact erf form | 0.35 | 0.093 | 0.32 | 0.066 | 0.53 | 0.017 |
| x * sigmoid(1.702 x) | 22 | 3.6 | 35 | 4.3 | 48 | 1 |
| piecewise linear, 64 segments | 4.4 | 1.5 | 6.9 | 0.95 | 8.5 | 0.38 |
| piecewise linear, 32 segments | 17 | 4.8 | 18 | 4.2 | 37 | 0.94 |
| 8-bit linear, per output channel | 12 | 4.5 | 13 | 2.1 | 21 | 0.76 |
| 8-bit linear, per tensor | 73 | 34 | 1.1e+02 | 69 | 95 | 31 |
| 8-bit lookup table (256 values per tensor) | 54 | 17 | 81 | 27 | 92 | 4 |
| 6-bit lookup table (64 values) | 72 | 21 | 77 | 45 | 93 | 5.5 |
| 4-bit lookup table (16 values) | 70 | 66 | 1.2e+02 | 66 | 1.2e+02 | 20 |

**lane-line probability**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.0023 | 0.0027 | 0.0011 | 0.0022 | 0.0014 | 0.0015 |
| exact erf form | 0.014 | 0.0091 | 0.011 | 0.0072 | 0.0077 | 0.0039 |
| x * sigmoid(1.702 x) | 0.38 | 0.39 | 0.32 | 0.33 | 0.31 | 0.18 |
| piecewise linear, 64 segments | 0.16 | 0.12 | 0.23 | 0.089 | 0.11 | 0.073 |
| piecewise linear, 32 segments | 0.44 | 0.26 | 0.48 | 0.31 | 0.26 | 0.21 |
| 8-bit linear, per output channel | 0.21 | 0.25 | 0.13 | 0.098 | 0.2 | 0.17 |
| 8-bit linear, per tensor | 0.98 | 0.97 | 0.89 | 0.98 | 0.98 | 0.98 |
| 8-bit lookup table (256 values per tensor) | 0.96 | 0.71 | 0.86 | 0.69 | 0.6 | 0.92 |
| 6-bit lookup table (64 values) | 0.96 | 0.97 | 0.87 | 0.96 | 0.95 | 0.96 |
| 4-bit lookup table (16 values) | 0.97 | 0.69 | 0.96 | 0.77 | 0.91 | 0.97 |

**lead probability**

| variant | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| float16 model (original ONNX) | 0.00093 | 0.00068 | 0.00087 | 0.00019 | 0.0014 | 0.00045 |
| exact erf form | 0.0094 | 0.0047 | 0.0035 | 0.00099 | 0.0093 | 0.0023 |
| x * sigmoid(1.702 x) | 0.36 | 0.24 | 0.18 | 0.059 | 0.34 | 0.18 |
| piecewise linear, 64 segments | 0.087 | 0.078 | 0.054 | 0.031 | 0.11 | 0.033 |
| piecewise linear, 32 segments | 0.26 | 0.2 | 0.2 | 0.071 | 0.32 | 0.15 |
| 8-bit linear, per output channel | 0.15 | 0.28 | 0.1 | 0.025 | 0.2 | 0.054 |
| 8-bit linear, per tensor | 0.82 | 0.77 | 0.84 | 0.91 | 0.92 | 0.21 |
| 8-bit lookup table (256 values per tensor) | 0.7 | 0.56 | 0.44 | 0.16 | 0.65 | 0.71 |
| 6-bit lookup table (64 values) | 0.69 | 0.78 | 0.73 | 0.64 | 0.78 | 0.73 |
| 4-bit lookup table (16 values) | 0.68 | 0.84 | 0.74 | 0.62 | 0.69 | 0.71 |


### What the tables say
- **Float16 model: negligible on every clip.** Worst step over the six clips: curvature at most 8.8e-5 1/m (gm, ram) with a mean of at most 8e-6; acceleration at most 0.021 m/s^2; plan x at 10 s at most 0.13 m; lane-line probability at most 0.0027; lead probability at most 0.0014. Raw worst output difference 0.10 to 0.42 (original 0.18), mean 0.0007 to 0.0013.
- **GELU exact (erf): negligible on every clip.** Worst step: curvature at most 2.8e-4 1/m (gm and ram, the turning clips; about 0.01 m/s^2 of lateral acceleration at gm's 6 m/s, 0.06 at ram's 15 m/s), acceleration at most 0.009 m/s^2, plan x at 10 s at most 0.75 m, lane probability at most 0.014, lead probability at most 0.009. The raw worst difference is 0.32 to 6.3 (original 1.44; gm is the largest) and the raw mean 0.002 to 0.007; the raw means are about 3 to 5 times float16's but the driving-relevant ones stay far below the device-vs-ours yardstick on the clips that have one.
- **GELU sigmoid: far from negligible on five of six clips.** Mean curvature difference 1.4e-4 (original), 2.0e-4 (tesla), 2.8e-3 (gm), 1.1e-3 (ram), 1.5e-4 (hyundai2), 2.4e-5 (toyota3); acceleration mean 0.02 to 0.17 m/s^2 and worst 0.07 to 0.88; plan x at 10 s mean 3.4 to 7.3 m, worst 11 to 56 m; lane-line probability worst 0.18 to 0.39; lead probability worst 0.06 to 0.36. On the two turning clips the worst-step curvature error is 0.054 (gm) and 0.027 (ram) 1/m. The exception is the constant-speed freeway clip toyota3, where curvature and acceleration errors are small (worst 9e-5 and 0.07) although the lane-line probability still moves by up to 0.33 at the worst step. So the sigmoid form "too far" claim holds, with the size depending on what the road is doing.
- **8-bit linear per output channel: well above float16 and erf on every clip, size depends on the clip and on the metric.** Raw mean 0.06 to 0.16. Acceleration mean 0.15 to 0.27 m/s^2 (worst 0.27 to 0.43) on all six, larger than the device-vs-ours yardstick (0.04 to 0.14) where there is one; curvature mean 1.5e-5 (toyota3) to 5.5e-4 (gm), at or below the yardstick (5e-5 to 2.6e-4) on the three clips that have one (original 6.8e-5, tesla 1.3e-4, toyota3 1.5e-5), and 5.5e-4 / 2.9e-4 on gm / ram where there is no yardstick; plan x at 10 s mean 0.7 to 4.6 m (worst 2.5 to 33 m); lane-line probability worst 0.10 to 0.25, lead worst up to 0.28. Whether that is "too far" depends on a threshold I do not have; it is tens of times the float16 differences (and about 30 times erf's) on every clip. It is smaller than sigmoid GELU in position and lane terms and larger in acceleration terms.
- **64-segment piecewise-linear GELU** (not mentioned in the task list but in the original study): mean curvature 9e-6 to 4.2e-4, acceleration mean 0.005 to 0.031, plan x at 10 s mean 0.9 to 1.6 m (worst 3.5 to 12 m), lane probability worst 0.07 to 0.23; between erf and sigmoid on every clip, as before.
- **Lookup tables and 8-bit per tensor: broken on every clip.** All of the 8-, 6- and 4-bit lookup tables and per-tensor 8-bit give a raw mean of 0.7 to 5.2 (float16: 0.001), acceleration mean 0.06 to 1.3 m/s^2, plan x at 10 s mean 9 to 290 m, lane-line probability mean 0.08 to 0.68 and worst 0.6 to 0.97, lead probability worst 0.16 to 0.92. The weakest case is the 8-bit lookup table (256 values per tensor) on the freeway clip toyota3 (curvature 7.8e-5, acceleration 0.39, plan x at 10 s 9 m, lane probability 0.23) and, for curvature only, on tesla (3.7e-4) and hyundai2 (7.6e-4); on the turning clips gm and ram its mean curvature error is 0.02 and 0.007 1/m. The ordering seen on the original clip (float16 < erf << PWL-64 < 8-bit per channel < PWL-32 ~ sigmoid << 8-bit LUT < per-tensor 8-bit ~ 6/4-bit LUT, by raw mean) is the same on all six clips by raw mean, except that on gm the 6-bit LUT (2.8) is just below per-tensor 8-bit (3.1).
- **Errors are larger on the clips with turns.** For sigmoid GELU, PWL, 8-bit per channel and the LUTs, the mean curvature difference on gm and ram is about 5 to 20 times that on the original, tesla and hyundai2 clips (more against toyota3). The first clip had one gentle curve; it understated this. Lateral acceleration equivalent: 1.1e-3 1/m at 17 m/s is 0.3 m/s^2.
- **Raw sensitivity on the original clip was not an outlier.** The raw mean differences of the other clips are within about 0.4x to 2x of the original for the same variant (sigmoid: 0.14 to 0.50 against 0.26), gm being the highest and toyota3/hyundai2 the lowest.

### Not new here, restated for the cross-clip view
- The device-vs-ours yardstick only exists for three clips with a device `modelV2.action` (original, tesla, toyota3: curvature mean abs difference 2.6e-4, 1.5e-4, 5.0e-5 1/m; acceleration 0.14, 0.04, 0.12 m/s^2, where toyota3's device acceleration is exactly 0 in the log).
- Speed plan vs vEgo: mean abs 0.19 to 0.56 m/s over the warm steps (original 0.51), correlation 0.96 to 0.999 where defined.

## Fixtures (scratch only, not in the repo)
`frames_u8.bin` (40 x [2,6,128,256] uint8, 15,728,640 bytes) and `outputs_f32.bin` (40 x 2576 float32 little-endian, 412,160 bytes) in the scratch directory: real frames 300..339 (video frames, warm region), expected outputs recomputed with the float32 model from zero state over exactly those 40 frames. Step 0 of these differs from the long-run output at the same frame (mean abs 1.49) because the state is cold; step 39 matches the long run to 0.0000 (mean abs, 4 decimals) - an observation, see Not checked.

## Not checked
- Whether the HEVC-decoded frames equal what the camera ISP gave modeld on the car (the video is lossy; the on-device model input is from raw NV12). Colour range handling of ffmpeg's decode was not inspected.
- The vanishing point in the warped images was judged by eye, not measured.
- One mismatching sample (of 196,608) in the narrow frame 300 warp was not traced to its cause.
- (Superseded for the tested variants by "More clips".) The original sections above use only one segment (30 s, 600 frames used, night-time highway). The first 100 steps are cold-start and are excluded from "warm".
- No closed-loop driving, no steering/longitudinal controller, no comparison on an iPhone, Core ML or iOS (only the onnxruntime CPU emulations of the existing experiments).
- "What matters for driving" thresholds are not established; the only external numbers are upstream's replay tolerance and the device-vs-ours yardstick. Lane-line, plan-position and lead errors have no justified threshold here.
- The float16 baseline discrepancy (1.71 stored vs 0.35 fresh on synthetic) was not diagnosed.
- Quantisation variants are weight-only emulations in float32 as in `quantize_study.py` (no activation quantisation, no real Core ML/ANE kernels).
- Speed used for synthetic action calculations is an assumed 15 m/s.
- Fixture step-39 agreement with the long run was observed (0.0000 mean abs) but not tested for being exact.
- More clips (see "More clips"): each is the first 30 s of a 60 s segment; the second half was not used. Six clips in total (two in daylight on streets or ramps, two highways, two at night, 3 devices/sensors: os04c10, ox03c10, ar0231), not a survey: no rain, snow, tunnel, glare, left-hand traffic, or fisheye-heavy wide-camera scenes were looked at.
- For the four older logs the calibration comes from `liveCalibration.rpyCalib` with status `uncalibrated` (calPerc 100) as parsed with the master schema; I did not check how those logs were produced. The warps look right by eye, but the vanishing points were not measured in the new clips either.
- Wide-camera focal length for tici/tizi (567 px) is flagged in openpilot's `camera.py` as "probably wrong? magnification is not consistent across frame"; the wide warps look plausible but were not checked quantitatively. The same angles are used for narrow and wide, as in `modeld.py`.
- The bottom rows of the narrow warp of the AR0231/OX03C10 clips (hyundai2 shows it clearly) repeat the last source row because the warp samples beyond the frame; this mirrors the clamping in the C core and the tinygrad oracle but was not compared to what the device's warp does there.
- Up to 4 samples per 196,608 differ between the C warp and the tinygrad oracle on single frames of the new clips (grey-level differences up to 11); not traced.
- The device `modelV2` of gm, ram and hyundai2 has no `action` field (older log), so the curvature/acceleration yardstick is missing for them; the speed/position comparison against an older device model is only a loose cross-check (ram: device vs vEgo is 2.1 m/s off, ours 0.56).
- The sensitivity runs in this section used 2 onnxruntime threads and ran several at a time; the numbers do not depend on that (CPU float32), but only the original clip's run was checked against the previous values, bit for bit for the reference, not re-run for the sensitivity JSON.
- SUBARU, FORD, RIVIAN (all with video and a log) were not processed; MAZDA has a frame-id offset between cameras.

## Files
`extract_log.py` (log to pickle; `rlog.zst` or `rlog.bz2`; `liveCalibration` fallback), `build_reference.py` (decode, warp, oracle check, PNGs, reference, fixtures, sanity statistics; frame size and intrinsics from the device and sensor in the log, `RF_START`, `RF_NF`, `RF_FIXTURES`), `rf_common.py` (adds `camera_config`, `intrinsics`; the other names are unchanged), `run_clip.sh` (one clip: log, decode, reference, delete the NV12 files), `compare_clips.py` (cross-clip tables), `sensitivity.py` (float16, GELU and compression on real and synthetic; `report` prints the tables), `run_existing.py` (runs the existing scripts unchanged on the big npz).
