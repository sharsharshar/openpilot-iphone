# Phone-style input errors: how much do the model's outputs move? (open-loop replay on one real clip)

Everything here was produced by the scripts in this directory, run in this session on the 30 s night-highway clip described in
`../real_footage/RESULTS.md` (600 steps, warm steps = 100..599, float32 onnxruntime model, state carried, desire zeros). Nothing here is
a claim about driving; it measures how far the model's outputs move when its inputs are changed in ways an iPhone could change them.
No design recommendation is made.

## Limits that apply to every number
- **Open loop.** The car in the video does not react to the perturbed outputs. A constant bias (for example from a yaw error) shows up here as a
  constant offset in curvature and plan position (see the "signed mean" column); in a real car it would act on the car's path. I did not model that.
- One clip, one sensor (`os04c10`, device `mici`), night, highway, lead car, one curve, speeds 11.8-22.2 m/s (mean 16.4 during warm steps). Other conditions are unknown.
- "Change" is always perturbed minus unperturbed run on the same frames (mean and worst of the absolute difference over the 500 warm steps; lane probability is averaged over the 4 lines and worst over any line). That is not an error against truth: there is no ground truth here.
- Lateral acceleration change = v_ego(t)^2 x curvature difference(t), with the logged v_ego.
- The yardstick (re-measured here, `check_reference.log`, matches `real_footage/RESULTS.md`): the comma device's own modelV2 (different model version) vs our unperturbed run differs by
  curvature 0.00026 1/m, acceleration 0.14 m/s^2, plan y at 10 s 3.9 m (plan x at 10 s 9.9 m, inner lane probs 0.048, lead prob 0.033), mean absolute, warm steps.
- **Noise floor.** Tiny perturbations (0.1 degree pitch, 0.1 degree yaw, +0.5% focal; table 0) already change the outputs by about: curvature 5e-5 to 1e-4 1/m, acceleration 0.01-0.06 m/s^2,
  plan y at 10 s 0.9-1.7 m, plan x at 10 s 2-4.4 m, worst-case plan y at 10 s 9-11 m. Differences of that size in the tables below are not distinguishable from this floor.
  Results are not strictly monotonic in the size of the perturbation (e.g. pitch +2 vs +5 for acceleration), which is another sign that one clip is a noisy sample.

## Checks that were run
1. **Harness reproduces the reference exactly** (`check_reference.py`, log `check_reference.log`): my own warp of all 1200 frames equals `real_reference_fp32.npz` frames bit for bit (max difference 0);
   my model run gives max absolute output difference 0 (mean 0) against the stored outputs for all 600 steps; a second run is also identical; the driving quantities are identical.
2. **Images look as intended** (`check_images.py`; PNGs in `$RF_SCRATCH/phone_errors/png/`, not in the repo; I looked at both). Measured image shift in the narrow model view, from pixel content, on frames 100/300/500:
   pitch +2 deg moves content 32 px down in the 256-row model luma frame for all three frames (the matrix predicts 31.8 = 910 x tan 2 deg); pitch -2 deg moves it 32 px up; yaw +2 deg moves it 32 px left (predicted 31.8); roll +2 deg moves it about 1 px (predicted under 1 px at the source centre).
   (The other component printed by the script in each row is an unconstrained search and is noise.) In `calib_narrow_view.png` the lead car's tail-lights sit on the red principal-point row for the baseline, below it for +2 deg pitch, above it for -2 deg. `onecamera_views.png` shows the four one-camera variants.
3. **Perturbed runs compared with the device's modelV2** on the same frames: second table of each section (first row is the unperturbed yardstick).

## Headline: largest and smallest effects, relative to the yardstick
Mean absolute change vs the unperturbed run (curvature 1/m / lateral acceleration m/s^2 / desired acceleration m/s^2 / plan y at 10 s m). Yardstick for comparison: 0.00026 / (0.00026 x 16.4^2 = 0.07) / 0.14 / 3.9.

| case | curvature | lat. accel | accel | plan y 10 s | plan x 10 s | reading |
|---|---|---|---|---|---|---|
| yaw error -5 deg | 0.0056 | 1.5 (worst 2.9) | 0.46 | 19 | 17 | curvature offset is nearly constant (signed mean 0.0056 = the mean), 21x the yardstick |
| model run only every other frame (outputs held) | 0.0027 | 0.57 (worst 1.9) | 0.63 | 25 | 120 | part of this is stale outputs, not model damage (see below) |
| pitch +5 deg | 0.0029 | 0.59 (worst 1.5) | 0.39 | 12 | 34 | |
| iOS recipe on the WIDE camera frames | 0.0021 | 0.42 (worst 1.6) | 1.0 | 11 | 120 | wrong K (63 deg assumed, true 116 deg) |
| yaw error +/-1 deg | 0.00065 / 0.00083 | 0.15 / 0.21 | 0.06 / 0.09 | 2.2 / 2.2 | 3 / 4 | 2.5-3x yardstick curvature; a constant offset of opposite sign to the yaw offset |
| no calibration (rpy = 0, as the iOS app today) | 0.00032 | 0.076 (worst 0.35) | 0.49 | 4.5 | 25 | acceleration 3.5x and plan x 2.5x the yardstick; curvature about 1.2x |
| pitch +0.5 to +2 deg | 0.0001-0.00026 | 0.03-0.06 | 0.5-0.7 | 4.2-6.1 | 25-35 | acceleration and forward plan move a lot even for +0.5 deg; -0.5 to -2 deg moves acceleration much less (0.12-0.17) |
| one camera, iOS recipe on the NARROW frames | 0.00036 | 0.089 (worst 0.54) | 0.47 | 4.1 | 27 | about the same as "no calibration" alone |
| focal length both cameras +/-10% | 0.00026 / 0.00032 | 0.06 / 0.07 | 0.09 / 0.14 | 2.3 / 3.2 | 20 / 22 | curvature about the yardstick; plan x 2x the yardstick |
| focal length +/-2% | 8e-5 | 0.02 | 0.03 | 1.4-1.6 | 4.6-4.9 | at the noise floor |
| roll +/-2 deg | 0.00013 | 0.03 | 0.05 | 2.0 | 4.5 | small; roll +/-5 deg: 0.0003 |
| image: half / quarter resolution, gain 0.7 / 1.3, 3x3 blur | 4e-5 to 1.1e-4 | 0.01-0.03 | 0.03-0.08 | 0.7-2.1 | 2-4.4 | at or near the noise floor |
| wide camera lagging by 1 or 2 frames | 1e-5 | 0.002 | 0.002-0.003 | 0.1 | 0.3-0.4 | far below the noise floor |
| wide focal length wrong by up to 10% | 1.2e-5 to 1.5e-5 | 0.003 | 0.006 | 0.15-0.18 | 0.5 | far below the noise floor |
| both inputs from the NARROW camera (true focal, logged rpy) | 2e-5 | 0.004 | 0.017 | 0.3 | 1.2 | far below the noise floor |

Verified by running, from these tables:
- On this clip the outputs are almost insensitive to everything done to the wide input alone (focal error, 1-2 frame lag, replacing it with narrow-camera pixels) and clearly sensitive to the same things done to the narrow input (focal error: narrow-only equals both; feeding the narrow view from the wide camera, `one_wide`, changes curvature 0.00029, acceleration 0.27 m/s^2, plan x 23 m).
  Why the model behaves like this is not established here.
- Against the device's modelV2 (second tables), these stay about as close as the unperturbed run (curvature within 0.00032, acceleration within 0.15, plan y at 10 s within 4.9 m): focal errors up to 5%, roll up to 2 deg, all image differences, wide lag, `one_narrow`, `drop2_repeat`. These move noticeably away: yaw offsets of 1 deg or more (curvature 0.00072 to 0.0055, 3-21x), positive pitch offsets (acceleration 0.33-0.74 vs 0.14, plan x at 10 s 26-36 m vs 9.9), no calibration (curvature 0.00044, acceleration 0.5, plan x 26 m), `ios_widesrc` (curvature 0.0022, acceleration 1.0, plan x 120 m), and the skip-timing cases.

## Notes per group
- **Calibration** (offsets added to the logged rpy = [roll, pitch, yaw] = [-1.7e-05, -0.0295, -0.0064] rad, same angles for both cameras as modeld does). Yaw gives a clean, nearly constant signed curvature bias with sign opposite to the offset (about 0.0004 per 0.5 deg, 0.0007-0.0008 per deg up to 2 deg, then 0.002 to 0.0056 at 5 deg) and a plan-y shift of the same sign pattern (-11 m / +18 m mean at 10 s for +5 / -5 deg).
  Pitch mostly moves desired acceleration and plan forward position, in an asymmetric way. Roll is the least sensitive axis. "No calibration" (rpy = 0 corresponds to removing 1.7 deg of pitch and 0.37 deg of yaw) lands in the same regime as pitch errors of 1-2 deg.
- **Focal length.** Assumed focal = true x (1 +/- pct). Narrow-only and both-cameras are practically identical; wide-only is negligible. Effects grow slowly with the error (curvature 8e-5 at 2%, 1.5e-4 at 5%, 2.6e-4 to 3.2e-4 at 10%); plan x at 10 s is the most affected (20-22 m at 10%).
- **One camera feeding both inputs.**
  (a) `one_wide`: both inputs warped from the wide frames with the true wide focal and logged rpy. (b) `one_narrow`: both from the narrow frames. Out of bounds: `op_frame_to_model_input` clamps each source coordinate to the frame (`clampi`, nearest neighbour), so pixels the camera never saw would be replicated edge pixels. For this pair of cameras it never happens: the narrow camera (1344 px, f 1141.5, horizontal FOV 60 deg) covers x 15..1307 and y 35..676 of its frame for the wide model view (0 of 131072 luma samples outside; measured with the exact matrix, `oob.json`), because the wide model view's FOV (about 60 deg horizontally) is about the same as this narrow camera's. So (b) is not a case of invented pixels here. It would be for a camera with a smaller FOV: at 1344x760 pixels and rpy 0, the fraction of the wide model view outside the source is 0.53 at 40 deg horizontal FOV, 0.27 at 50 deg, 0 at 63 deg and wider (table at the end).
  (c) iOS app today (read from `ios/OpenpilotPhone/Pipeline.swift` `warp(_:)`, `FrameSource.swift`): one `CVPixelBuffer` (NV12 full range, session preset 1280x720) per frame; intrinsics f = (w/2)/tan(hfov/2), c = frame centre, with hfov = `AVCaptureDevice.Format.videoFieldOfView` (fallback 63 deg); rpy = [0,0,0]; the same K gives `op_warp_matrix(K, 0, big=0)` for input 0 and `op_warp_matrix(K, 0, big=1)` for input 1; both are warped from that single frame by `op_frame_to_model_input`; the model's v_ego is a fixed 15 m/s there (not reproduced: I use the logged v_ego so numbers stay comparable). Reproduced as `ios_narrowsrc` (our narrow frames, f for 63 deg = 1097 px, rpy 0) and, as a stress case, `ios_widesrc` (wide frames with the same f = 1097, which is wrong by a factor of 2.6). `ios_narrowsrc_truef` keeps rpy 0 but uses the true narrow focal.
  Not reproduced: 1280x720 resolution and aspect (our frames are 1344x760), full-range vs video-range luma, portrait/landscape orientation handling, the iPhone's real field of view and distortion, rolling shutter, frame dropping when the model is busy. The narrow frames of this clip are a stand-in for an iPhone main camera only because their FOV (60 deg) is close to 63 deg.
- **Timing.** `repeat`: at every 5th (or 2nd) step the new input is replaced by the previous step's input and the model is run anyway. `skip`: the model is not run at that step, the previous outputs are held, and the model's state does not advance; the held outputs are compared with the unperturbed output of the same step, so the difference includes plain staleness (the car moved on) and is not purely a change in what the model computes. The `skip` rows at every 2nd frame effectively mean a 10 Hz model; they are large (plan x at 10 s 120 m, plan y 25 m). The reason `drop5_repeat` shows a larger curvature change (3.6e-4) than `drop2_repeat` (6e-5) was not investigated.
  Wide lag: wide input at step t comes from frame t-1 or t-2 (frame 0 repeated at the start); negligible.
- **Image differences** (applied to both cameras' source frames): half / quarter resolution = box average of Y, U and V planes before the warp with the focal length scaled; gain = luma x 0.7 or x 1.3, clipped, chroma untouched; blur = 3x3 box on luma only, chroma untouched. All near the noise floor on this clip. Night footage is dark, so gain 1.3 and 0.7 are small absolute changes in grey level; a daylight test could differ.

## Not checked
- Closed-loop effects of any kind; any controller; any threshold for what is acceptable (I did not find one for plan position, lane or lead probability).
- Any other clip, daylight, city, low speed, other sensors; run-to-run spread over different clips or start frames (only the "noise floor" controls speak to sensitivity).
- Combinations of errors (for example no calibration together with a wrong focal length together with one camera, beyond the single `ios_*` cases), and errors that differ between the two cameras' calibration.
- Lens distortion, rolling shutter, orientation/aspect differences, exposure and white balance behaviour of an iPhone, motion blur differences, HEVC-vs-ISP differences (see `real_footage/RESULTS.md`).
- Why the wide input matters so little here; why pitch +0.5 to +2 deg changes acceleration and plan x much more than negative pitch; why drop5_repeat > drop2_repeat. Not investigated.
- The float16 / Core ML model: all runs are the float32 ONNX model on CPU.
- The `skip` timing cases do not model the iOS app's real behaviour (it drops frames that arrive while it is busy, and the model then sees the next frame with advanced time).
- Whether the model's state warm-up (first 100 steps) matters for the perturbed runs: all use state from zeros like the reference.
- Tables show 2 significant digits; full precision JSON is in `$RF_SCRATCH/phone_errors/results/`.

## Files and how to run
`pe_common.py` (harness), `check_reference.py`, `check_images.py`, `run_all.py` (groups `calib focal onecam timing image oob control`), `report.py` (prints the tables below).
Set `RF_SCRATCH` to the data directory; outputs go to `$RF_SCRATCH/phone_errors/`. Run with the project python and `-I`, e.g. `RF_SCRATCH=... python -I check_reference.py` (it must run first: it writes `baseline_outputs.npy`), then `run_all.py <groups>`, then `report.py`.
The 62 perturbed runs took roughly an hour or more on 4 cores with two processes in parallel (not timed precisely).

# Full tables (generated by `report.py`)

### 1. Calibration error (offsets added to the logged rpy, both cameras; rpy = [roll, pitch, yaw]) (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| cal_zero | 0.00032 / 0.0018 | 0.076 / 0.35 | 0.49 / 0.95 | 0.095 / 0.58 | 4.5 / 23 | 25 / 75 | 0.054 / 0.42 | 0.015 / 0.29 | -0.00019, 0.15 |
| cal_pitch_+0.5 | 0.00012 / 0.00081 | 0.027 / 0.15 | 0.52 / 1.3 | 0.08 / 0.34 | 5.6 / 25 | 25 / 80 | 0.033 / 0.34 | 0.0091 / 0.12 | 1.3e-05, 1.1 |
| cal_pitch_-0.5 | 0.0001 / 0.00074 | 0.026 / 0.15 | 0.15 / 0.59 | 0.043 / 0.27 | 2.4 / 15 | 11 / 48 | 0.023 / 0.21 | 0.015 / 0.36 | -3.7e-05, 0.91 |
| cal_pitch_+1 | 0.00019 / 0.001 | 0.043 / 0.2 | 0.74 / 1.4 | 0.099 / 0.41 | 6.1 / 24 | 35 / 89 | 0.051 / 0.36 | 0.012 / 0.22 | 7.7e-05, 1 |
| cal_pitch_-1 | 0.00012 / 0.00089 | 0.029 / 0.16 | 0.12 / 0.63 | 0.051 / 0.33 | 2.8 / 17 | 14 / 81 | 0.025 / 0.31 | 0.013 / 0.28 | -4e-05, 1.2 |
| cal_pitch_+2 | 0.00026 / 0.0024 | 0.059 / 0.48 | 0.52 / 1 | 0.094 / 0.69 | 4.2 / 20 | 25 / 65 | 0.056 / 0.37 | 0.013 / 0.29 | 0.00018, 0.44 |
| cal_pitch_-2 | 0.00014 / 0.0009 | 0.033 / 0.18 | 0.17 / 0.74 | 0.087 / 0.65 | 3.4 / 21 | 12 / 86 | 0.025 / 0.3 | 0.013 / 0.34 | -3.2e-05, 0.64 |
| cal_pitch_+5 | 0.0029 / 0.01 | 0.59 / 1.5 | 0.39 / 1.1 | 0.89 / 3.1 | 12 / 47 | 34 / 62 | 0.19 / 0.77 | 0.062 / 0.58 | 0.0029, 8.4 |
| cal_pitch_-5 | 0.00024 / 0.0015 | 0.056 / 0.29 | 0.25 / 0.91 | 0.18 / 1.2 | 11 / 46 | 20 / 94 | 0.055 / 0.4 | 0.26 / 0.93 | 0.00017, 2.3 |
| cal_yaw_+0.5 | 0.00041 / 0.0013 | 0.099 / 0.27 | 0.041 / 0.18 | 0.13 / 0.36 | 1.4 / 17 | 3 / 28 | 0.022 / 0.26 | 0.0079 / 0.14 | -0.00041, -0.65 |
| cal_yaw_-0.5 | 0.00045 / 0.0012 | 0.12 / 0.27 | 0.043 / 0.16 | 0.15 / 0.34 | 1.7 / 17 | 2.8 / 23 | 0.017 / 0.27 | 0.0078 / 0.17 | 0.00045, 0.96 |
| cal_yaw_+1 | 0.00065 / 0.0026 | 0.15 / 0.51 | 0.057 / 0.28 | 0.21 / 0.5 | 2.2 / 15 | 3.2 / 33 | 0.034 / 0.37 | 0.0097 / 0.24 | -0.00065, -1.7 |
| cal_yaw_-1 | 0.00083 / 0.0024 | 0.21 / 0.58 | 0.089 / 0.25 | 0.26 / 0.55 | 2.2 / 15 | 4.3 / 18 | 0.03 / 0.39 | 0.0092 / 0.2 | 0.00083, 1.3 |
| cal_yaw_+2 | 0.00077 / 0.0039 | 0.19 / 0.76 | 0.06 / 0.29 | 0.32 / 0.8 | 4.3 / 17 | 4.4 / 23 | 0.049 / 0.52 | 0.014 / 0.28 | -0.00077, -3.9 |
| cal_yaw_-2 | 0.0014 / 0.0035 | 0.34 / 0.75 | 0.099 / 0.41 | 0.44 / 0.83 | 3.3 / 22 | 5.5 / 20 | 0.045 / 0.41 | 0.015 / 0.15 | 0.0014, 2.5 |
| cal_yaw_+5 | 0.0021 / 0.0053 | 0.64 / 1.8 | 0.29 / 0.99 | 1.1 / 2.9 | 12 / 51 | 12 / 55 | 0.085 / 0.53 | 0.042 / 0.33 | -0.002, -11 |
| cal_yaw_-5 | 0.0056 / 0.0088 | 1.5 / 2.9 | 0.46 / 1.4 | 2 / 4 | 19 / 48 | 17 / 66 | 0.12 / 0.54 | 0.08 / 0.86 | 0.0056, 18 |
| cal_roll_+0.5 | 5.8e-05 / 0.00044 | 0.013 / 0.085 | 0.022 / 0.16 | 0.025 / 0.16 | 1.2 / 15 | 2.3 / 22 | 0.013 / 0.17 | 0.0078 / 0.2 | 3e-05, 0.066 |
| cal_roll_-0.5 | 5.5e-05 / 0.0005 | 0.012 / 0.074 | 0.02 / 0.098 | 0.024 / 0.15 | 1 / 7.3 | 2 / 15 | 0.012 / 0.21 | 0.007 / 0.14 | -1.9e-05, 0.06 |
| cal_roll_+1 | 8.6e-05 / 0.00068 | 0.019 / 0.12 | 0.03 / 0.21 | 0.035 / 0.28 | 1.5 / 13 | 3.1 / 25 | 0.017 / 0.25 | 0.0094 / 0.25 | 5.8e-05, 0.027 |
| cal_roll_-1 | 7.8e-05 / 0.00049 | 0.017 / 0.092 | 0.031 / 0.14 | 0.03 / 0.16 | 1.3 / 9.9 | 2.8 / 26 | 0.017 / 0.22 | 0.0084 / 0.18 | -5.3e-05, 0.29 |
| cal_roll_+2 | 0.00013 / 0.00078 | 0.03 / 0.15 | 0.054 / 0.3 | 0.05 / 0.35 | 2 / 11 | 4.5 / 44 | 0.025 / 0.34 | 0.012 / 0.28 | 0.00011, -0.35 |
| cal_roll_-2 | 0.00013 / 0.00079 | 0.029 / 0.12 | 0.043 / 0.22 | 0.045 / 0.29 | 1.8 / 12 | 4.2 / 27 | 0.025 / 0.29 | 0.01 / 0.21 | -0.00011, 0.51 |
| cal_roll_+5 | 0.00032 / 0.0013 | 0.075 / 0.22 | 0.086 / 0.52 | 0.11 / 0.47 | 2.9 / 20 | 8.2 / 50 | 0.047 / 0.42 | 0.019 / 0.41 | 0.00031, -0.34 |
| cal_roll_-5 | 0.00031 / 0.0015 | 0.072 / 0.24 | 0.07 / 0.31 | 0.097 / 0.55 | 2.5 / 14 | 7.8 / 27 | 0.051 / 0.46 | 0.016 / 0.36 | -0.0003, 0.7 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| cal_zero | 0.00044 | 0.5 | 5.1 | 26 | 0.063 | 0.033 |
| cal_pitch_+0.5 | 0.00029 | 0.53 | 5.5 | 27 | 0.059 | 0.035 |
| cal_pitch_-0.5 | 0.00028 | 0.16 | 4.5 | 13 | 0.048 | 0.027 |
| cal_pitch_+1 | 0.00031 | 0.74 | 6.2 | 36 | 0.072 | 0.035 |
| cal_pitch_-1 | 0.00028 | 0.14 | 4.7 | 16 | 0.051 | 0.028 |
| cal_pitch_+2 | 0.00036 | 0.52 | 5.2 | 26 | 0.061 | 0.032 |
| cal_pitch_-2 | 0.00029 | 0.17 | 5.2 | 14 | 0.053 | 0.03 |
| cal_pitch_+5 | 0.0029 | 0.33 | 13 | 30 | 0.27 | 0.077 |
| cal_pitch_-5 | 0.0003 | 0.25 | 11 | 23 | 0.068 | 0.27 |
| cal_yaw_+0.5 | 0.00055 | 0.14 | 4.4 | 9.6 | 0.045 | 0.036 |
| cal_yaw_-0.5 | 0.00041 | 0.14 | 3.4 | 9.6 | 0.049 | 0.034 |
| cal_yaw_+1 | 0.00077 | 0.16 | 5.1 | 10 | 0.046 | 0.037 |
| cal_yaw_-1 | 0.00072 | 0.13 | 3.5 | 9.3 | 0.049 | 0.034 |
| cal_yaw_+2 | 0.00089 | 0.14 | 6.7 | 9.9 | 0.056 | 0.041 |
| cal_yaw_-2 | 0.0013 | 0.094 | 4.1 | 8.8 | 0.052 | 0.037 |
| cal_yaw_+5 | 0.0022 | 0.28 | 14 | 12 | 0.11 | 0.056 |
| cal_yaw_-5 | 0.0055 | 0.45 | 18 | 15 | 0.11 | 0.085 |
| cal_roll_+0.5 | 0.00026 | 0.14 | 3.8 | 9.9 | 0.049 | 0.035 |
| cal_roll_-0.5 | 0.00027 | 0.14 | 3.8 | 9.8 | 0.049 | 0.035 |
| cal_roll_+1 | 0.00026 | 0.13 | 3.7 | 10 | 0.046 | 0.033 |
| cal_roll_-1 | 0.00029 | 0.15 | 3.7 | 10 | 0.051 | 0.036 |
| cal_roll_+2 | 0.00026 | 0.12 | 3.6 | 10 | 0.046 | 0.034 |
| cal_roll_-2 | 0.00032 | 0.15 | 3.8 | 10 | 0.053 | 0.035 |
| cal_roll_+5 | 0.00031 | 0.12 | 3.8 | 12 | 0.05 | 0.04 |
| cal_roll_-5 | 0.00048 | 0.15 | 3.7 | 12 | 0.074 | 0.035 |

### 0. Control: tiny perturbations (noise floor) (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 6.2e-05 / 0.00045 | 0.014 / 0.067 | 0.06 / 0.33 | 0.031 / 0.18 | 1.7 / 9.8 | 4.4 / 37 | 0.015 / 0.18 | 0.0087 / 0.24 | 6.8e-06, 0.24 |
| ctl_yaw_+0.1 | 0.0001 / 0.0004 | 0.026 / 0.081 | 0.014 / 0.075 | 0.042 / 0.14 | 1.1 / 9.1 | 2 / 13 | 0.011 / 0.16 | 0.0069 / 0.17 | -9.6e-05, -0.19 |
| ctl_focal_+0.5 | 4.7e-05 / 0.00026 | 0.01 / 0.053 | 0.02 / 0.19 | 0.021 / 0.14 | 0.9 / 11 | 2.4 / 25 | 0.0098 / 0.13 | 0.0067 / 0.13 | -8.1e-07, 0.13 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| ctl_pitch_+0.1 | 0.00027 | 0.16 | 3.7 | 11 | 0.049 | 0.035 |
| ctl_yaw_+0.1 | 0.0003 | 0.14 | 4.1 | 9.7 | 0.047 | 0.034 |
| ctl_focal_+0.5 | 0.00027 | 0.14 | 3.9 | 10 | 0.048 | 0.035 |

### 2. Focal length error (assumed = true x (1 +/- pct)) (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| foc_both_+2 | 8.2e-05 / 0.0007 | 0.018 / 0.14 | 0.029 / 0.19 | 0.031 / 0.24 | 1.4 / 13 | 4.9 / 26 | 0.016 / 0.18 | 0.0092 / 0.22 | -3.1e-05, 0.082 |
| foc_both_-2 | 8.3e-05 / 0.00092 | 0.019 / 0.18 | 0.031 / 0.15 | 0.031 / 0.16 | 1.6 / 15 | 4.6 / 24 | 0.016 / 0.23 | 0.0098 / 0.4 | 3.1e-05, 0.0074 |
| foc_both_+5 | 0.00015 / 0.00099 | 0.033 / 0.2 | 0.046 / 0.19 | 0.041 / 0.27 | 1.8 / 16 | 10 / 39 | 0.022 / 0.27 | 0.014 / 0.34 | -7.5e-05, 0.23 |
| foc_both_-5 | 0.00017 / 0.0011 | 0.038 / 0.19 | 0.067 / 0.25 | 0.046 / 0.3 | 2.3 / 13 | 8.6 / 26 | 0.026 / 0.35 | 0.012 / 0.31 | 8.7e-05, -0.042 |
| foc_both_+10 | 0.00026 / 0.0016 | 0.057 / 0.32 | 0.086 / 0.42 | 0.052 / 0.32 | 2.3 / 16 | 20 / 53 | 0.03 / 0.41 | 0.014 / 0.28 | -0.00015, 0.36 |
| foc_both_-10 | 0.00032 / 0.0016 | 0.073 / 0.28 | 0.14 / 0.69 | 0.062 / 0.48 | 3.2 / 15 | 22 / 66 | 0.038 / 0.43 | 0.014 / 0.28 | 0.00014, 0.49 |
| foc_narrow_+2 | 8.1e-05 / 0.00067 | 0.018 / 0.13 | 0.029 / 0.18 | 0.031 / 0.25 | 1.3 / 13 | 4.9 / 25 | 0.015 / 0.17 | 0.0091 / 0.22 | -3e-05, 0.077 |
| foc_narrow_-2 | 8.3e-05 / 0.00094 | 0.019 / 0.18 | 0.032 / 0.15 | 0.031 / 0.17 | 1.5 / 14 | 4.6 / 22 | 0.016 / 0.23 | 0.0096 / 0.39 | 3.1e-05, 0.0065 |
| foc_narrow_+5 | 0.00015 / 0.00096 | 0.032 / 0.19 | 0.046 / 0.2 | 0.041 / 0.31 | 1.8 / 16 | 10 / 38 | 0.022 / 0.27 | 0.013 / 0.33 | -7.6e-05, 0.2 |
| foc_narrow_-5 | 0.00017 / 0.00099 | 0.037 / 0.19 | 0.067 / 0.25 | 0.046 / 0.28 | 2.2 / 14 | 8.5 / 26 | 0.025 / 0.38 | 0.012 / 0.31 | 8.8e-05, -0.033 |
| foc_narrow_+10 | 0.00026 / 0.0016 | 0.056 / 0.31 | 0.084 / 0.41 | 0.052 / 0.32 | 2.2 / 16 | 20 / 52 | 0.031 / 0.41 | 0.014 / 0.25 | -0.00015, 0.3 |
| foc_narrow_-10 | 0.00032 / 0.0015 | 0.072 / 0.27 | 0.14 / 0.69 | 0.063 / 0.47 | 3.2 / 15 | 22 / 66 | 0.037 / 0.46 | 0.014 / 0.3 | 0.00014, 0.49 |
| foc_wide_+2 | 7.3e-06 / 7.4e-05 | 0.0015 / 0.013 | 0.002 / 0.013 | 0.0031 / 0.031 | 0.088 / 1.3 | 0.25 / 4.6 | 0.0021 / 0.034 | 0.00081 / 0.027 | -1.2e-06, 0.013 |
| foc_wide_-2 | 7.4e-06 / 7.2e-05 | 0.0016 / 0.014 | 0.0028 / 0.019 | 0.0032 / 0.025 | 0.099 / 2.4 | 0.28 / 3.3 | 0.0019 / 0.03 | 0.00093 / 0.038 | 2.5e-07, -0.0014 |
| foc_wide_+5 | 9.8e-06 / 9e-05 | 0.0021 / 0.017 | 0.0033 / 0.023 | 0.004 / 0.036 | 0.12 / 1.1 | 0.36 / 5.5 | 0.0028 / 0.038 | 0.0013 / 0.046 | -1.6e-06, 0.0062 |
| foc_wide_-5 | 9.7e-06 / 0.00011 | 0.002 / 0.022 | 0.0037 / 0.022 | 0.0037 / 0.032 | 0.13 / 1.3 | 0.38 / 4.6 | 0.0029 / 0.044 | 0.0013 / 0.038 | 8.4e-07, 0.006 |
| foc_wide_+10 | 1.2e-05 / 8.8e-05 | 0.0026 / 0.017 | 0.0064 / 0.043 | 0.0049 / 0.048 | 0.15 / 1.3 | 0.54 / 4.7 | 0.004 / 0.06 | 0.0019 / 0.054 | -1e-06, 0.022 |
| foc_wide_-10 | 1.5e-05 / 0.00012 | 0.0032 / 0.024 | 0.006 / 0.031 | 0.0055 / 0.043 | 0.18 / 2.7 | 0.52 / 5.6 | 0.0041 / 0.044 | 0.0019 / 0.048 | 3.3e-06, -0.024 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| foc_both_+2 | 0.00028 | 0.14 | 4 | 12 | 0.048 | 0.033 |
| foc_both_-2 | 0.00026 | 0.13 | 3.7 | 9.5 | 0.05 | 0.034 |
| foc_both_+5 | 0.00031 | 0.15 | 4.4 | 16 | 0.047 | 0.034 |
| foc_both_-5 | 0.00026 | 0.13 | 3.6 | 11 | 0.053 | 0.035 |
| foc_both_+10 | 0.00039 | 0.18 | 4.8 | 25 | 0.047 | 0.033 |
| foc_both_-10 | 0.00032 | 0.14 | 3.8 | 20 | 0.058 | 0.033 |
| foc_narrow_+2 | 0.00028 | 0.14 | 4 | 12 | 0.048 | 0.033 |
| foc_narrow_-2 | 0.00026 | 0.13 | 3.7 | 9.4 | 0.05 | 0.034 |
| foc_narrow_+5 | 0.00031 | 0.15 | 4.3 | 16 | 0.047 | 0.034 |
| foc_narrow_-5 | 0.00026 | 0.13 | 3.7 | 11 | 0.053 | 0.035 |
| foc_narrow_+10 | 0.00039 | 0.18 | 4.7 | 25 | 0.046 | 0.033 |
| foc_narrow_-10 | 0.00032 | 0.15 | 3.8 | 20 | 0.057 | 0.033 |
| foc_wide_+2 | 0.00026 | 0.14 | 3.9 | 9.8 | 0.048 | 0.033 |
| foc_wide_-2 | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| foc_wide_+5 | 0.00026 | 0.14 | 3.9 | 9.8 | 0.048 | 0.033 |
| foc_wide_-5 | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| foc_wide_+10 | 0.00026 | 0.14 | 3.9 | 9.8 | 0.048 | 0.033 |
| foc_wide_-10 | 0.00026 | 0.14 | 3.9 | 10 | 0.048 | 0.034 |

### 3. One physical camera feeding both inputs (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| one_wide | 0.00029 / 0.002 | 0.065 / 0.39 | 0.27 / 0.68 | 0.11 / 0.62 | 4.5 / 23 | 23 / 63 | 0.051 / 0.43 | 0.02 / 0.51 | 5.7e-06, -0.43 |
| one_narrow | 2e-05 / 0.00015 | 0.0044 / 0.028 | 0.017 / 0.084 | 0.011 / 0.072 | 0.32 / 2.9 | 1.2 / 8.4 | 0.0062 / 0.092 | 0.002 / 0.034 | 5.4e-06, 0.042 |
| ios_narrowsrc | 0.00036 / 0.0028 | 0.089 / 0.54 | 0.47 / 0.95 | 0.092 / 0.72 | 4.1 / 22 | 27 / 72 | 0.059 / 0.41 | 0.015 / 0.36 | -0.00011, -0.11 |
| ios_widesrc | 0.0021 / 0.0085 | 0.42 / 1.6 | 1 / 2 | 0.48 / 1.6 | 11 / 43 | 1.2e+02 / 2e+02 | 0.14 / 0.63 | 0.18 / 0.73 | -0.0014, -1.9 |
| ios_narrowsrc_truef | 0.00033 / 0.002 | 0.078 / 0.38 | 0.54 / 1.1 | 0.1 / 0.63 | 4.6 / 23 | 26 / 76 | 0.054 / 0.42 | 0.015 / 0.34 | -0.00018, 0.12 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| one_wide | 0.00035 | 0.3 | 5.3 | 23 | 0.062 | 0.028 |
| one_narrow | 0.00025 | 0.14 | 3.9 | 10 | 0.047 | 0.034 |
| ios_narrowsrc | 0.00045 | 0.48 | 5 | 28 | 0.073 | 0.038 |
| ios_widesrc | 0.0022 | 1 | 10 | 1.2e+02 | 0.21 | 0.2 |
| ios_narrowsrc_truef | 0.00044 | 0.55 | 5.2 | 27 | 0.065 | 0.035 |

### 4. Timing (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| drop5_repeat | 0.00036 / 0.0039 | 0.077 / 0.72 | 0.066 / 0.24 | 0.18 / 1.5 | 2.7 / 28 | 11 / 71 | 0.013 / 0.35 | 0.0057 / 0.19 | -1.7e-05, -0.66 |
| drop2_repeat | 6.2e-05 / 0.00092 | 0.014 / 0.14 | 0.017 / 0.11 | 0.025 / 0.33 | 0.89 / 15 | 1.8 / 22 | 0.013 / 0.29 | 0.0067 / 0.39 | -8.3e-06, 0.14 |
| drop5_skip | 0.00092 / 0.0039 | 0.19 / 0.67 | 0.21 / 0.75 | 0.64 / 2.3 | 8.1 / 32 | 35 / 82 | 0.038 / 0.5 | 0.017 / 0.32 | 0.00054, 1.7 |
| drop2_skip | 0.0027 / 0.011 | 0.57 / 1.9 | 0.63 / 1.5 | 2.1 / 6.3 | 25 / 71 | 1.2e+02 / 1.9e+02 | 0.096 / 0.74 | 0.053 / 0.41 | 0.0016, 11 |
| lag_wide1 | 9.3e-06 / 9.3e-05 | 0.002 / 0.017 | 0.0019 / 0.016 | 0.0037 / 0.04 | 0.11 / 1.1 | 0.33 / 4.5 | 0.0029 / 0.064 | 0.0013 / 0.03 | -1.1e-06, -0.0022 |
| lag_wide2 | 1.2e-05 / 9.5e-05 | 0.0026 / 0.017 | 0.003 / 0.027 | 0.0048 / 0.043 | 0.14 / 1.8 | 0.43 / 5.2 | 0.004 / 0.053 | 0.0016 / 0.042 | -5.4e-07, 0.00069 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| drop5_repeat | 0.00051 | 0.16 | 5.1 | 16 | 0.049 | 0.036 |
| drop2_repeat | 0.00029 | 0.14 | 4 | 10 | 0.049 | 0.033 |
| drop5_skip | 0.00084 | 0.21 | 9.6 | 36 | 0.053 | 0.04 |
| drop2_skip | 0.0027 | 0.56 | 26 | 1.2e+02 | 0.12 | 0.07 |
| lag_wide1 | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| lag_wide2 | 0.00026 | 0.14 | 3.9 | 9.8 | 0.048 | 0.033 |

### 5. Image differences (perturbed minus unperturbed, mean / worst of |difference| over warm steps 100..599)

| case | curvature 1/m | lat. accel m/s^2 | desired accel m/s^2 | plan y @1.9s m | plan y @10s m | plan x @10s m | lane prob | lead prob | signed mean: curv 1/m, y@10s m |
|---|---|---|---|---|---|---|---|---|---|
| img_down2 | 5.8e-05 / 0.0004 | 0.014 / 0.076 | 0.036 / 0.19 | 0.027 / 0.16 | 1.3 / 10 | 3 / 31 | 0.014 / 0.19 | 0.0086 / 0.29 | -1.6e-05, -0.3 |
| img_down4 | 0.00011 / 0.00094 | 0.026 / 0.19 | 0.076 / 0.4 | 0.049 / 0.26 | 2.1 / 19 | 4.4 / 37 | 0.037 / 0.44 | 0.017 / 0.4 | -3.6e-05, -0.84 |
| img_gain0.7 | 4.2e-05 / 0.00032 | 0.0095 / 0.063 | 0.036 / 0.2 | 0.021 / 0.17 | 0.87 / 5.2 | 3.2 / 19 | 0.013 / 0.25 | 0.0052 / 0.12 | 2.3e-06, 0.015 |
| img_gain1.3 | 3.6e-05 / 0.00026 | 0.0081 / 0.049 | 0.028 / 0.15 | 0.02 / 0.17 | 0.74 / 6.5 | 2.5 / 20 | 0.011 / 0.14 | 0.0043 / 0.14 | 4.5e-06, -0.052 |
| img_blur3 | 4.8e-05 / 0.00047 | 0.01 / 0.09 | 0.053 / 0.18 | 0.025 / 0.21 | 1 / 8.2 | 2.2 / 14 | 0.011 / 0.14 | 0.0057 / 0.11 | 2.1e-05, 0.11 |

Same cases against the comma device's own modelV2 (mean abs difference, warm steps; first row = unperturbed run, the yardstick)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed (yardstick) | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| img_down2 | 0.00027 | 0.13 | 4.3 | 9.3 | 0.047 | 0.035 |
| img_down4 | 0.00028 | 0.14 | 4.9 | 9 | 0.058 | 0.033 |
| img_gain0.7 | 0.00027 | 0.13 | 4 | 11 | 0.046 | 0.034 |
| img_gain1.3 | 0.00026 | 0.14 | 4.1 | 9.7 | 0.048 | 0.032 |
| img_blur3 | 0.00024 | 0.15 | 4.1 | 10 | 0.045 | 0.034 |

### Experiment descriptions

- `cal_pitch_+0.5`: pitch offset +0.5 deg added to the logged calibration
- `cal_pitch_+1`: pitch offset +1 deg added to the logged calibration
- `cal_pitch_+2`: pitch offset +2 deg added to the logged calibration
- `cal_pitch_+5`: pitch offset +5 deg added to the logged calibration
- `cal_pitch_-0.5`: pitch offset -0.5 deg added to the logged calibration
- `cal_pitch_-1`: pitch offset -1 deg added to the logged calibration
- `cal_pitch_-2`: pitch offset -2 deg added to the logged calibration
- `cal_pitch_-5`: pitch offset -5 deg added to the logged calibration
- `cal_roll_+0.5`: roll offset +0.5 deg added to the logged calibration
- `cal_roll_+1`: roll offset +1 deg added to the logged calibration
- `cal_roll_+2`: roll offset +2 deg added to the logged calibration
- `cal_roll_+5`: roll offset +5 deg added to the logged calibration
- `cal_roll_-0.5`: roll offset -0.5 deg added to the logged calibration
- `cal_roll_-1`: roll offset -1 deg added to the logged calibration
- `cal_roll_-2`: roll offset -2 deg added to the logged calibration
- `cal_roll_-5`: roll offset -5 deg added to the logged calibration
- `cal_yaw_+0.5`: yaw offset +0.5 deg added to the logged calibration
- `cal_yaw_+1`: yaw offset +1 deg added to the logged calibration
- `cal_yaw_+2`: yaw offset +2 deg added to the logged calibration
- `cal_yaw_+5`: yaw offset +5 deg added to the logged calibration
- `cal_yaw_-0.5`: yaw offset -0.5 deg added to the logged calibration
- `cal_yaw_-1`: yaw offset -1 deg added to the logged calibration
- `cal_yaw_-2`: yaw offset -2 deg added to the logged calibration
- `cal_yaw_-5`: yaw offset -5 deg added to the logged calibration
- `cal_zero`: rpy = [0,0,0] (no calibration, as the iOS app does today)
- `ctl_focal_+0.5`: CONTROL: both focal lengths +0.5%
- `ctl_pitch_+0.1`: CONTROL: pitch offset +0.1 deg
- `ctl_yaw_+0.1`: CONTROL: yaw offset +0.1 deg
- `drop2_repeat`: every 2th frame dropped, previous model input repeated (model runs every step)
- `drop2_skip`: every 2th frame dropped, model not run at that step, previous outputs held
- `drop5_repeat`: every 5th frame dropped, previous model input repeated (model runs every step)
- `drop5_skip`: every 5th frame dropped, model not run at that step, previous outputs held
- `foc_both_+10`: assumed focal length both +10% (true value x 1.10)
- `foc_both_+2`: assumed focal length both +2% (true value x 1.02)
- `foc_both_+5`: assumed focal length both +5% (true value x 1.05)
- `foc_both_-10`: assumed focal length both -10% (true value x 0.90)
- `foc_both_-2`: assumed focal length both -2% (true value x 0.98)
- `foc_both_-5`: assumed focal length both -5% (true value x 0.95)
- `foc_narrow_+10`: assumed focal length narrow +10% (true value x 1.10)
- `foc_narrow_+2`: assumed focal length narrow +2% (true value x 1.02)
- `foc_narrow_+5`: assumed focal length narrow +5% (true value x 1.05)
- `foc_narrow_-10`: assumed focal length narrow -10% (true value x 0.90)
- `foc_narrow_-2`: assumed focal length narrow -2% (true value x 0.98)
- `foc_narrow_-5`: assumed focal length narrow -5% (true value x 0.95)
- `foc_wide_+10`: assumed focal length wide +10% (true value x 1.10)
- `foc_wide_+2`: assumed focal length wide +2% (true value x 1.02)
- `foc_wide_+5`: assumed focal length wide +5% (true value x 1.05)
- `foc_wide_-10`: assumed focal length wide -10% (true value x 0.90)
- `foc_wide_-2`: assumed focal length wide -2% (true value x 0.98)
- `foc_wide_-5`: assumed focal length wide -5% (true value x 0.95)
- `img_blur3`: 3x3 box blur of the source luma, chroma unchanged
- `img_down2`: source frames box-downscaled to 1/2 (672x380) before the warp (focal scaled)
- `img_down4`: source frames box-downscaled to 1/4 (336x190) before the warp (focal scaled)
- `img_gain0.7`: luma multiplied by 0.7 (clipped to 0..255), chroma unchanged
- `img_gain1.3`: luma multiplied by 1.3 (clipped to 0..255), chroma unchanged
- `ios_narrowsrc`: iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the NARROW camera frames
- `ios_narrowsrc_truef`: iOS-app recipe but with the true narrow focal length (rpy 0, same narrow frame in both inputs)
- `ios_widesrc`: iOS-app recipe (K from 63 deg hfov, rpy 0, same frame in both inputs) on the WIDE camera frames (true wide hfov is 116 deg, so K is wrong)
- `lag_wide1`: wide camera input lags the narrow by 1 frame(s)
- `lag_wide2`: wide camera input lags the narrow by 2 frame(s)
- `one_narrow`: both inputs from the NARROW camera frames, true narrow focal and logged rpy
- `one_wide`: both inputs from the WIDE camera frames, true wide focal and logged rpy

### Fraction of luma samples whose source coordinate is outside the source frame (clamped to the edge pixel by the C code)

| case | outside | by x | by y |
|---|---|---|---|
| baseline_narrow_view | 0.000 | 0.000 | 0.000 |
| baseline_wide_view | 0.000 | 0.000 | 0.000 |
| narrowcam_in_wide_view | 0.000 | 0.000 | 0.000 |
| widecam_in_narrow_view | 0.000 | 0.000 | 0.000 |
| ios_f63_rpy0_narrow_view | 0.000 | 0.000 | 0.000 |
| ios_f63_rpy0_wide_view | 0.000 | 0.000 | 0.000 |
| hfov40_rpy0_wide_view | 0.528 | 0.354 | 0.270 |
| hfov50_rpy0_wide_view | 0.274 | 0.170 | 0.125 |
| hfov63_rpy0_wide_view | 0.000 | 0.000 | 0.000 |
| hfov78_rpy0_wide_view | 0.000 | 0.000 | 0.000 |
| hfov90_rpy0_wide_view | 0.000 | 0.000 | 0.000 |

speed used for lateral acceleration: v_ego warm mean 16.4 m/s, range 11.8..22.2

## Independent noise-floor check (added after review)
`noise_floor_check.py` perturbs the stored model inputs directly, with no warp involved, and measures the change on
warm steps (mean / worst):

| input change | curvature (1/m) | acceleration (m/s^2) | plan x at 10 s (m) | plan y at 10 s (m) | lane-line prob |
|---|---|---|---|---|---|
| +-1 grey level of random noise on every sample | 0.000047 / 0.00034 | 0.014 / 0.095 | 1.3 / 15.8 | 0.77 / 5.3 | 0.011 / 0.16 |
| every channel shifted by 1 sample (about 0.13 deg of yaw) | 0.00013 / 0.00070 | 0.022 / 0.18 | 2.2 / 22.9 | 1.1 / 9.0 | 0.014 / 0.17 |

This agrees with the control runs above: input changes far too small to see move the driving outputs by about this
much. Differences of this size or smaller, from any cause, are inside the model's own sensitivity to its input on this clip.
