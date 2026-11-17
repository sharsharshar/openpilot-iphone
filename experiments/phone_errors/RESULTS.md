# Phone-style input errors: how much do the model's outputs move? (open-loop replay on one real clip)

Everything here was produced by the scripts in this directory, run in this session on the 30 s night-highway clip described in
`../real_footage/RESULTS.md` (600 steps, warm steps = 100..599, float32 onnxruntime model, state carried, desire zeros). Nothing here is
a claim about driving; it measures how far the model's outputs move when its inputs are changed in ways an iPhone could change them.
No design recommendation is made.
The same study (reduced set of cases) was repeated on five more clips, two of them with real turns: see "More clips" at the end of this file. Where it says "one clip" below, that is the original clip.

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

# More clips: the reduced study repeated on five more clips

Question: do the findings from the one night-highway clip hold on other footage, above all on the two clips with real turns (`gm`, `ram`)? Everything here was produced by the scripts in this directory, run in this session; no design recommendation is made. All the limits listed at the top of this file apply (open loop: a constant bias shows up as a constant offset and no feedback is modelled; "change" is perturbed minus unperturbed run on the same frames; warm steps 100..599; float32 ONNX model on CPU, state carried, desire zeros). **None of these clips comes from an iPhone camera.**

## What was run
- Clips: the five clips of `../real_footage/RESULTS.md` "More clips" (`tesla`, `gm`, `toyota3`, `ram`, `hyundai2`; first 600 frames of each, re-decoded with `ffmpeg -frames:v 600 -pix_fmt nv12` from `fcamera.hevc` / `ecamera.hevc`, deleted after each clip) and the original clip (`original`), re-run with the same reduced set so that the cross-clip tables are like for like. Frame size and focal lengths come from the clip's own log through `rf_common.camera_config` (mici/os04c10: 1344x760, f 1141.5 / 425.25; tici/tizi with ox03c10 or ar0231: 1928x1208, f 2648 / 567).
- 33 cases per clip (`run_all.py more`; names are the row names in the tables):
  - noise floor of the clip: `ctl_noise1` (+-1 grey level of random noise on every model-input sample, seed 0, exactly the draw of `noise_floor_check.py`), `ctl_pitch_+0.1`, `ctl_yaw_+0.1` (0.1 degree added to the logged calibration);
  - calibration: `cal_zero` (rpy = 0, as the iOS app does today), yaw +-1 and +-2 degrees, pitch +-1 and +-2 degrees, roll +2 degrees (offsets added to the logged rpy, same angles for both cameras);
  - focal length 5 percent (assumed = true x 1.05 / 0.95): `foc_both_*`, `foc_narrow_*`, `foc_wide_*`;
  - how much each input is used: `wide_lag2` / `narrow_lag2` (that input comes from 2 frames earlier, frame 0 repeated at the start), `wide_frozen` / `narrow_frozen` (that input's first frame repeated for the whole clip), `wide_grey` / `narrow_grey` (that input replaced by a constant image, all six channels 128), and the input built from the other camera's pixels: `one_narrow` = the wide input built from narrow-camera pixels (and, as before, the narrow input also from the narrow camera, so both inputs come from the narrow camera), `one_wide` = the narrow input built from wide-camera pixels (both inputs from the wide camera). These are the same runs as "one physical camera feeding both inputs, both ways round", so each is listed once;
  - the iOS app's current recipe, as described in the section "One camera feeding both inputs (c)" above (K from a 63 degree horizontal FOV, rpy 0, one frame in both inputs): `ios_narrowsrc` (narrow-camera frames), `ios_widesrc` (wide-camera frames), and `ios_narrowsrc_truef` (rpy 0 but the true narrow focal length, which separates "no calibration" from "wrong K");
  - timing: `drop5_repeat` (every 5th frame dropped, previous model input repeated), `drop2_skip` (model run only every other frame, previous outputs held, state not advanced; includes plain staleness);
  - image: `img_down2` (source frames box-averaged to half resolution before the warp).
- Metrics per case, as in the first study: mean and worst |change| over the 500 warm steps of desired curvature, desired acceleration, plan x and y at 10 s, lane-line probability (mean over the 4 lines; worst over any line) and lead probability. The turn split uses the UNPERTURBED run's desired curvature: "turn steps" = warm steps with |curvature| > 0.01 1/m (my choice of threshold), "other steps" = the rest. Per-step values of every run are kept (`more/<case>.npz` in the scratch directory) and all tables are computed from them by `more_report.py`.
- Noise floor: the first row group of every table (`ctl_*`). "x floor" in the summary tables = mean change of the case divided by mean change of `ctl_noise1` in the same clip on the same steps. The other two controls (0.1 degree of pitch or yaw) are larger than `ctl_noise1` in curvature on all clips (1.3 to 9 times, `ctl_yaw_+0.1` row of the summary tables), so a ratio of 1 to 3 should be read as "inside the floor".

## Checks run for this study
1. **Harness reproduces the stored reference exactly, on every clip** (`check_reference.py`; logs `check_reference.log` in each clip's `phone_errors/`): my own warp of all 1200 frames (re-decoded from the video) equals the stored `real_reference_fp32.npz` frames, max difference 0; my model run equals the stored outputs, max difference 0 (mean 0); a second run is identical; the driving quantities are identical. All six clips. The generalised scripts on the original clip reproduce its earlier results exactly: 26 cases that the first study also ran (`cal_*`, `foc_*`, `ctl_pitch/yaw`, `one_*`, `ios_*`, `lag_wide2`, `drop5_repeat`, `drop2_skip`, `img_down2`) were re-run into a separate directory and their result JSON (`vs_base` and `vs_device`) equals the first study's JSON exactly (`==` on every float), and `ctl_noise1` gives the numbers of `noise_floor_check.py` (curvature 4.7e-5, acceleration 0.014, plan x 1.3, plan y 0.77, lane 0.011).
2. **Images look as intended** (`check_images.py`; PNGs in `<clip>/phone_errors/png/`, not in the repo; I looked at `more_views_a.png` / `more_views_b.png` for `gm`, `hyundai2`, `toyota3`, `tesla`, and `more_views_b.png` for `ram`). In every clip the measured image shift for pitch +-2 degrees is 32 rows in the narrow model view (predicted 31.8 to 32.2) and for yaw +2 degrees 32 columns (predicted 31.8 to 31.9); roll +2 degrees a predicted shift under 4 px against a noisy measurement within 7 px (same remark as before: the other component printed by the script in each row is an unconstrained search and is noise). In the PNGs (rows of `more_views_a`: baseline; wide view from narrow-camera pixels; narrow view from wide-camera pixels; `more_views_b`: iOS recipe on narrow frames; iOS recipe on wide frames; narrow frozen with wide live; narrow live with wide frozen) on the tici/tizi clips:
   - the wide view built from the narrow camera shows the real scene only in a central band (about half the width and half the height) and edge pixels smeared outwards everywhere else; the script's measured fraction of the 512x256 view outside the source frame is 0.54 (gm), 0.59 (ram), 0.49 (toyota3), 0.47 (hyundai2); on the mici clips it is 0.000 (original) and 0.005 (tesla);
   - the narrow view built from wide-camera pixels is a blurry, 1.6x upsampled version of the scene (source pixels per model pixel 0.62, against 2.9 for the real narrow camera), with no replicated pixels;
   - the iOS recipe on wide-camera frames is the fisheye-looking whole wide image including the dashboard (K wrong by a factor of 2.6 on mici and 2.8 on tici/tizi); on narrow-camera frames of a tici the image is magnified by about 1.7 (K for 63 degrees applied to a 40 degree camera);
   - the frozen and grey inputs are what they say (checked in the code path: `np.repeat(x[:1])`, `np.full(128)`; the frozen frame is frame 0, visible in `more_views_b` row 3 and 4 as a different scene from the live half).
   - `hyundai2`: the baseline narrow view already has 5.5 percent of its samples outside the source (the bottom rows repeat the last source row, as noted in `real_footage/RESULTS.md`).
3. **Built-from-the-other-camera views disagree geometrically with the real ones.** `check_images.py` aligns the view built from the other camera with the real view of the same frame (brute-force integer shift of gradient-magnitude images, central crop, frames 100/300/500; correlation 0.46 to 0.92; the sign convention of the printed shift is not interpreted here, only its size). One model pixel is 0.063 degrees in the narrow view (focal 910) and 0.126 degrees in the wide view (focal 455).

   | clip | narrow view from wide-camera pixels vs the real narrow view: (dy, dx) in model rows, columns | wide view from narrow-camera pixels vs the real wide view: (dy, dx) |
   |---|---|---|
   | original | (12, -2), (14, 0), (14, 0) | (-6, 0), (-6, 0), (-6, 0) |
   | tesla | (-24, -18), (-24, -18), (-22, -20) | (12, 8), (12, 10), (12, 8) |
   | gm | (-32, 20), (-32, 14), (-32, 16) | (20, -6), (22, -6), (20, -8) |
   | toyota3 | (-4, -26), (-4, -22), (-2, -20) | (2, 6), (4, 8), (2, 4) |
   | ram | (-22, 12), (-22, 14), (-16, 4) | (10, -6), (12, -6), (8, -2) |
   | hyundai2 | (18, -28), (16, -26), (18, -26) | (-10, 14), (-8, 14), (-8, 12) |

   So with the one shared logged calibration (as modeld applies it), the same scene point lands up to 32 rows (about 2 degrees of pitch) and 28 columns (about 1.8 degrees of yaw) away in the narrow view, depending on the clip, even for two clips from the same device type (original and tesla are both mici/os04c10 and differ). A pure resolution difference would give (0, 0). I did not establish whether this comes from different camera mounting angles, the wide lens not being a pinhole (`camera.py` flags its focal length as uncertain), parallax, or the search itself (correlation as low as 0.46 on `gm`).

## Clips (what the numbers below refer to)
Warm steps only. "Turn steps": |reference curvature| > 0.01 1/m. The `original` clip has such steps too, but only a gentle curve (curvature 0.010 to 0.014 1/m at 12 to 14 m/s), not a turn. Calibration is the logged `rpyCalib` (roll, pitch, yaw in degrees; the angles `cal_zero` removes).

| clip | device / sensor, frame | v_ego warm min / mean / max (m/s) | logged rpy (deg) | steps with abs curvature > 0.01 (of 500), v_ego there, max abs curvature | noise floor, mean change of `ctl_noise1`: curvature 1/m, accel m/s^2, plan x m, plan y m, lane, lead | device `modelV2` available for curvature and acceleration | yardstick: unperturbed run vs device, plan y @10 s / plan x @10 s |
|---|---|---|---|---|---|---|---|
| original | mici / os04c10, 1344x760 | 11.8 / 16.4 / 22.2 | -0.00, -1.69, -0.37 | 77, 11.8-14.0 m/s, 0.0138 | 4.7e-5, 0.014, 1.3, 0.77, 0.011, 0.0032 | yes | 3.9 / 9.9 m |
| tesla | mici / os04c10, 1344x760 | 1.3 / 13.1 / 25.4 | -0.01, -1.44, 1.45 | 0 | 2.6e-5, 0.013, 0.96, 0.14, 0.0055, 0.0082 | yes | 0.46 / 2.7 m |
| gm | tici / ox03c10, 1928x1208 | 4.2 / 6.8 / 9.9 | -0.01, 3.03, 1.22 | 205, 4.2-6.2 m/s, 0.108 | 1.9e-4, 0.0081, 0.52, 0.45, 0.0028, 0.0043 | no | 9.8 / 9.8 m |
| toyota3 | tizi / ox03c10, 1928x1208 | 30.2 / 30.4 / 30.5 | 0.00, 0.88, -0.44 | 0 | 4.0e-6, 0.0023, 0.42, 0.12, 0.0051, 0.0012 | yes | 1.1 / 7.5 m |
| ram | tici / ox03c10, 1928x1208 | 5.6 / 16.1 / 24.3 | 0.00, 5.61, -0.66 | 114, 5.6-12.8 m/s, 0.103 | 6.6e-5, 0.015, 0.86, 0.42, 0.0034, 0.0061 | no | 7.5 / 16.4 m |
| hyundai2 | tici / ar0231, 1928x1208 | 6.7 / 11.6 / 13.2 | 0.00, -0.84, -0.02 | 0 | 2.4e-5, 0.010, 1.0, 0.026, 0.0026, 0.00074 | no | 0.43 / 18.3 m |

(Floors restricted to turn steps: original 1.7e-4 1/m, gm 3.0e-4, ram 2.4e-4 for curvature, against 2.5e-5, 1.1e-4, 1.6e-5 on the other steps of the same clips: the model's output is itself less stable in a turn, so a change in a turn must be compared with the turn floor.) For `gm`, `ram` and `hyundai2` the device's log has no `modelV2.action`, so distance to the device is reported for plan position and lane / lead probability only.

## Answers (read against each clip's own noise floor; tables further down)

### 1. Does the wide input matter on the clips with turns (`gm`, `ram`)?
- **Lag by 2 frames and a 5 percent focal error on the wide input alone: no.** Curvature change is 0.85 to 1.02 times the floor on `gm` and 1.05 to 1.37 on `ram` (all warm steps), and 0.77 to 0.87 (`gm`) and 1.02 to 1.43 (`ram`) on turn steps only; acceleration 0.5 to 1.0 times (all warm steps) and 0.7 to 1.07 (turn steps). On all six clips these three cases stay between 0.2 and 1.4 times the floor in every metric (curvature, acceleration, plan x and y at 10 s, lane, lead; all warm steps), and they move the distance to the device's `modelV2` by no more than the noise-only run does (per-clip device tables).
- **Removing the wide input (frozen first frame, or constant grey): yes, measurably, but small next to the narrow input.** On the turn steps of `gm` / `ram` the curvature change is 0.0018 / 0.0008 1/m (frozen) and 0.0020 / 0.0016 1/m (grey), i.e. 5.9 / 3.3 and 6.6 / 6.6 times the turn floor (3.0e-4 / 2.4e-4 1/m). The mean |reference curvature| on those steps is 0.056 / 0.048 1/m, so this is 2 to 4 percent of the curvature the model outputs in the turn. Plan y at 10 s moves 2.5 / 2.0 m (frozen) and 2.8 / 3.1 m (grey) against a floor of 0.6 m; desired acceleration 0.064 / 0.052 and 0.077 / 0.12 m/s^2 against 0.010 / 0.017. For scale, on the same steps: yaw +1 degree gives 0.0017 / 0.0017 1/m, a 5 percent focal error on both cameras 0.0024 / 0.0019 1/m, pitch +-1 degree 0.0014 to 0.0015 (`gm`) and about 0.0012 to 0.0014 (`ram`); removing the NARROW input gives 0.056 / 0.049 1/m, i.e. the model then outputs essentially zero curvature in the turn (signed change -99 to -101 percent). So the wide input is worth about as much curvature as a 1 degree yaw error or a 5 percent focal error in the turns, and about 30 times less than the narrow input.
- **No consistent amplification of the dependence on the wide input in turns, measured against the floor.** The same case divided by the floor on the same steps gives about the same multiple in turn steps and other steps for wide_grey (`gm` 6.6 vs 6.3, `ram` 6.6 vs 5.7) and for wide_frozen on `ram` (3.3 vs 3.4), but twice as much in the turn for wide_frozen on `gm` (5.9 vs 2.9). The absolute change is larger in the turn, although the absolute change is larger in the turn because the floor is (wide_grey curvature: `gm` 2.0e-3 in turns vs 7.1e-4 elsewhere, `ram` 1.6e-3 vs 9.0e-5). The sign of the change is not systematic: mean signed change along the turn direction is -2 percent (`gm`) and 0 percent (`ram`) of the turn curvature for both wide_grey and wide_frozen (table "Signed change in curvature on turn steps").
- **Wide input built from narrow-camera pixels** (`one_narrow`; on the tici clips 54 percent (`gm`) and 59 percent (`ram`) of the wide view is edge-replicated): on turn steps 3.8 / 3.7 times the floor in curvature and 5.0 / 5.2 in acceleration, i.e. about 0.6 of the effect of removing the wide input altogether.
- On the original clip the same removal costs 1.0 (frozen) and 1.8 (grey) times the floor in curvature, and 2.1 and 4.8 in acceleration; in curvature it is the clip with the weakest dependence on the wide input of the six. On its gentle-curve steps (77) the curvature change is 0.8 / 0.8 times the floor.

### 2. One camera feeding both inputs, against the noise floor and the calibration errors
Mean change in desired curvature as a multiple of the clip's own floor (all warm steps; the last two columns are turn steps only). Full tables, with acceleration, in the cross-clip summary below.

| case | original | tesla | gm | toyota3 | ram | hyundai2 | gm turn | ram turn |
|---|---|---|---|---|---|---|---|---|
| `one_narrow` (both inputs from the narrow camera) | 0.43 | 2.6 | 4.6 | 1.1 | 3.5 | 2.1 | 3.8 | 3.7 |
| `one_wide` (both inputs from the wide camera) | 6.1 | 53 | 14 | 41 | 19 | 35 | 12 | 16 |
| `ios_narrowsrc` (iOS recipe, narrow frames) | 7.7 | 117 | 73 | 98 | 77 | 28 | 97 | 71 |
| `ios_narrowsrc_truef` (same, true focal) | 6.9 | 116 | 13 | 26 | 19 | 17 | 9.2 | 17 |
| `cal_zero` (no calibration) | 6.7 | 118 | 13 | 26 | 19.5 | 16.5 | 9.3 | 17 |
| yaw +-1 degree | 14 / 18 | 69 / 72 | 8.6 / 10.5 | 36 / 66 | 17 / 18 | 34 / 69 | 5.5 / 5.2 | 7.0 / 6.2 |
| yaw +-2 degrees | 16 / 30 | 108 / 162 | 10 / 19 | 72 / 120 | 21 / 30 | 39 / 122 | 10 / 9.3 | 12 / 11.5 |
| pitch +-1 degree | 4.0 / 2.6 | 11 / 7.5 | 5.1 / 5.2 | 15 / 6.7 | 7.3 / 6.0 | 20 / 10 | 4.7 / 5.0 | 5.1 / 5.7 |
| focal +-5 percent, both cameras | 3.2 / 3.7 | 7.4 / 6.9 | 6.8 / 6.9 | 6.5 / 7.7 | 7.7 / 7.9 | 4.7 / 5.7 | 8.1 / 8.4 | 8.0 / 8.2 |
| `ctl_yaw_+0.1` (0.1 degree) | 2.2 | 7.4 | 2.0 | 9.1 | 2.8 | 6.2 | 1.7 | 1.8 |

- **`one_narrow` is small**: 0.4 to 4.6 times the floor (for comparison, a 1 degree pitch error is 2.6 to 20 times and a 5 percent focal error on the wide camera alone 0.2 to 1.4 times). On the turn steps it is about the size of a 1 degree pitch error (3.8 / 3.7 against 4.7 to 5.7) and below a 1 degree yaw error (5.2 to 7.0) and below "no calibration" (9.3 / 17). This holds also where half the wide view is replicated edge pixels (tici/tizi clips), and it is at the floor (0.43 and 1.1) on the original and `toyota3`.
- **`one_wide` is as large as the calibration errors**: 6 to 53 times the floor, between 0.45 and 2.2 times `cal_zero` clip by clip, of the order of a 1 to 2 degree yaw error (on `tesla` its 53 is below the 69 of a 1 degree yaw error). The narrow view built from wide-camera pixels is misaligned with the real narrow view by up to about 2 degrees (check 3 above), which is the size of the calibration errors in this table; whether that misalignment, the resolution loss or the lens distortion causes the effect was not tested (inference only).
- **`ios_narrowsrc_truef` equals `cal_zero` to within 4 percent in curvature on every clip, and to within 11 percent in acceleration** (curvature 6.9 / 6.7, 116 / 118, 13 / 13.2, 26 / 25.7, 19.2 / 19.5, 16.9 / 16.5 times the floor; also on the turn steps 9.2 / 9.3 and 17.1 / 17.2). With the true focal length, feeding one narrow frame to both inputs (with half of the wide view replicated edge pixels on tici/tizi) costs nothing measurable beyond not having a calibration. The iOS app's assumed 63 degree K is harmless on mici (about 4 percent off the true focal length of the 61 degree narrow camera: `ios_narrowsrc` 7.7 vs 6.9 and 117 vs 116) and costly on tici/tizi (K 1.7 times too small for the 40 degree camera: 28 to 98 times the floor, 1.7 to 5.6 times `cal_zero`).
- The size of the "no calibration" error depends on the logged angles of the clip (6.7 to 118 times the floor in curvature; it removes 5.6 degrees of pitch in `ram`, 3.0 in `gm`, 1.45 degrees of yaw in `tesla`), so the calibration-error rows should be compared clip by clip, not across clips.

### 3. What happened to the first study's findings
| finding from the original clip | all six clips | verdict |
|---|---|---|
| Lag, or a focal error up to 10 percent (here: 5 percent), of the wide input alone is far below the noise floor | 0.2 to 1.4 times the floor in every metric on all six clips (acceleration 0.2 to 1.1 times) | holds on all |
| Nothing done to the wide input matters | true for lag and focal error; removing it (frozen or grey): 1.0 / 1.8 times the floor in curvature on the original, 2.0 to 4.9 (frozen) and 5.8 to 8.7 (grey) on the other five, and 4.4 to 7.4 times in acceleration on all six | holds for lag and focal error only; reversed for removal on five of six clips (small in size, 2 to 4 percent of the turn curvature in `gm` and `ram`) |
| The narrow input is what the model uses | `narrow_frozen` / `narrow_grey` 33 to 182 times the floor in curvature, `narrow_lag2` 4.3 to 12; narrow-only focal error equals both-camera focal error (`foc_narrow_*` = `foc_both_*` within 2 percent) on every clip | holds on all |
| Both inputs from the narrow camera (`one_narrow`) is at the floor | 0.43 (original), 1.1 (`toyota3`), 2.1 (`hyundai2`), 2.6 (`tesla`), 3.5 (`ram`), 4.6 (`gm`) | holds on 2 of 6; small (below 5 times) on all |
| Both inputs from the wide camera (`one_wide`) is about the size of no calibration | 0.45 to 2.2 times `cal_zero` | holds as an order of magnitude only |
| The iOS recipe on narrow frames is about the same as no calibration | equal to within 15 percent on the two mici clips; 1.7 to 5.6 times `cal_zero` on the four tici/tizi clips (wrong K) | holds on mici, not on tici/tizi |
| A yaw offset gives a nearly constant curvature offset of opposite sign | mean signed curvature change for yaw +2 / -2 degrees: -7.7e-4 / +1.4e-3 (original), -2.8e-3 / +4.2e-3 (tesla), -1.8e-3 / +3.5e-3 (gm), -2.9e-4 / +4.8e-4 (toyota3), -1.3e-3 / +2.0e-3 (ram), -8.1e-4 / +2.9e-3 (hyundai2); the negative offset is 1.5 to 3.7 times larger on every clip | holds on all |
| Positive pitch moves desired acceleration much more than negative pitch | acceleration change pitch +1 / -1 degree, times the floor: 52 / 8.3, 46 / 12, 16.5 / 21, 306 / 29, 37 / 6.0, 28 / 11 (original, tesla, gm, toyota3, ram, hyundai2) | holds on five of six; not on `gm` (low speed, 4 to 10 m/s) |
| Half-resolution source is at the floor | 1.2 to 3.8 times the floor in curvature, 1.8 to 3.2 in acceleration; 0.6 to 1.8 in curvature on turn steps | holds (small), not exactly at 1 |
| Skipping the model on every other frame is large | 23 to 80 times the floor in curvature on all six; `drop5_repeat` 4.0 to 15 | holds on all |
| Roll 2 degrees is the least sensitive axis | 2.7 to 10.5 times the floor in curvature | holds (small) |

Distance to the device's own `modelV2` (tables per clip): `wide_lag2` and `foc_wide_*` leave it unchanged; `wide_grey` / `wide_frozen` change it by at most about 20 percent of its unperturbed value on `original`, `toyota3`, `hyundai2` and `ram`, and more on `tesla` (plan y 0.46 -> 0.69 / 0.56 m, plan x 2.7 -> 3.8 / 4.6 m, acceleration 0.043 -> 0.073 / 0.058) and `gm` (lane 0.064 -> 0.080, lead 0.052 -> 0.035 for grey); `cal_zero` moves plan x at 10 s away on every clip (e.g. 9.9 -> 26 m, 2.7 -> 15 m, 7.5 -> 17 m); `one_wide` moves plan y at 10 s away from the device on all clips except `ram` (where it moves closer, 7.5 -> 6.0 m).

### 4. Which cases are a fair stand-in for one iPhone main camera (about 69 degrees wide)? (inference, not tested on an iPhone)
The wide model view is about 59 degrees wide, so a 69 degree camera, correctly modelled, would supply real pixels for both views.
- **mici clips (`original`, `tesla`): the narrow camera (about 61 degrees) is a fair stand-in for its geometry.** `one_narrow` (0 and 0.5 percent of the wide view outside the source), `ios_narrowsrc_truef` and `ios_narrowsrc` (63 degree K, 4 percent off the true focal length) are the nearest stand-ins. They differ from an iPhone in resolution, lens, exposure, rolling shutter and the 8 degree difference in field of view.
- **tici / tizi clips (`gm`, `toyota3`, `ram`, `hyundai2`): `one_narrow` and `ios_narrowsrc_truef` are NOT fair stand-ins**, because 47 to 59 percent of the wide view is replicated edge pixels that an iPhone would not have (what they say is about a degraded wide view, not a real one); `ios_narrowsrc` is not either, its large numbers come from a K that is 1.7 times too small for this camera (shown by `ios_narrowsrc_truef` being much smaller).
- **`one_wide` (tici/tizi and mici) is the only case where both views are built from real pixels covering the whole view**, but it is not a clean stand-in either: the narrow view is built at 0.62 source pixel per model pixel (blurrier than an iPhone's), the wide lens is not a pinhole and `camera.py` flags its focal length as uncertain, and the measured camera-to-camera misalignment (check 3, up to about 2 degrees) is an error an iPhone camera would not have. I would read its size (comparable to a 1 to 2 degree yaw error) as probably an overestimate of what a single correctly modelled iPhone camera would show, but this was not tested.
- No case here has a camera that is both 69 degrees wide and sharp; none says anything about an iPhone's lens distortion, rolling shutter, orientation handling or auto exposure.

## Cross-clip summary tables

Curvature and desired acceleration, mean change as a multiple of the clip's own noise floor (the first row of each table is the floor in absolute units). Columns marked "turn steps" use only steps with |reference curvature| > 0.01 1/m, and their floor is computed on those steps.

**curvature 1/m**, mean |change| / mean |change| of `ctl_noise1` (floor in the first row, absolute)

| case | original | tesla | gm | toyota3 | ram | hyundai2 | original turn steps | gm turn steps | ram turn steps |
|---|---|---|---|---|---|---|---|---|---|
| floor (ctl_noise1), absolute | 4.7e-05 | 2.6e-05 | 0.00019 | 4e-06 | 6.6e-05 | 2.4e-05 | 0.00017 | 0.0003 | 0.00024 |
| ctl_yaw_+0.1 | 2.16 | 7.44 | 1.95 | 9.11 | 2.81 | 6.2 | 0.723 | 1.65 | 1.82 |
| cal_zero | 6.68 | 118 | 13.2 | 25.7 | 19.5 | 16.5 | 2.86 | 9.32 | 17.2 |
| cal_yaw_+1 | 13.7 | 68.9 | 8.61 | 35.6 | 17.2 | 33.9 | 5.41 | 5.54 | 7.02 |
| cal_yaw_-1 | 17.5 | 71.9 | 10.5 | 65.9 | 17.5 | 68.6 | 5.77 | 5.15 | 6.23 |
| cal_yaw_+2 | 16.3 | 108 | 10.1 | 72.4 | 21.3 | 38.7 | 6 | 10.2 | 11.8 |
| cal_yaw_-2 | 29.8 | 162 | 18.5 | 120 | 30.3 | 122 | 11.7 | 9.25 | 11.5 |
| cal_pitch_+1 | 3.98 | 10.7 | 5.07 | 15.2 | 7.28 | 20.4 | 1.79 | 4.68 | 5.09 |
| cal_pitch_-1 | 2.61 | 7.45 | 5.22 | 6.67 | 5.97 | 10.2 | 1.25 | 4.98 | 5.71 |
| cal_pitch_+2 | 5.59 | 15.3 | 6.07 | 15.4 | 9.63 | 16 | 4.17 | 5.77 | 7.37 |
| cal_pitch_-2 | 2.89 | 8.77 | 6.58 | 4.92 | 9.83 | 9.96 | 1.43 | 5.9 | 9.83 |
| cal_roll_+2 | 2.73 | 10.5 | 5.82 | 6.33 | 8.54 | 5.24 | 1.47 | 5.96 | 8.26 |
| foc_both_+5 | 3.16 | 7.38 | 6.82 | 6.52 | 7.67 | 4.67 | 2.43 | 8.1 | 7.95 |
| foc_both_-5 | 3.65 | 6.85 | 6.87 | 7.65 | 7.88 | 5.65 | 2.63 | 8.38 | 8.16 |
| foc_wide_+5 | 0.208 | 0.784 | 0.849 | 0.396 | 1.05 | 0.86 | 0.153 | 0.78 | 1.02 |
| foc_wide_-5 | 0.205 | 0.752 | 1.02 | 0.393 | 1.2 | 0.861 | 0.138 | 0.874 | 1.18 |
| wide_lag2 | 0.262 | 0.611 | 0.929 | 0.448 | 1.37 | 0.992 | 0.174 | 0.77 | 1.43 |
| wide_frozen | 1.01 | 3.73 | 4.85 | 2.02 | 3.33 | 3.45 | 0.798 | 5.91 | 3.31 |
| wide_grey | 1.77 | 8.65 | 6.52 | 5.95 | 6.45 | 5.82 | 0.77 | 6.62 | 6.63 |
| one_narrow | 0.428 | 2.62 | 4.64 | 1.1 | 3.51 | 2.11 | 0.228 | 3.81 | 3.66 |
| narrow_lag2 | 4.25 | 5.92 | 9.94 | 5.76 | 11.4 | 12.3 | 1.76 | 11.7 | 11.4 |
| narrow_frozen | 69.6 | 32.6 | 128 | 79.5 | 180 | 53.2 | 71.1 | 186 | 203 |
| narrow_grey | 71 | 35.8 | 127 | 95.1 | 182 | 67.6 | 72.5 | 185 | 204 |
| one_wide | 6.09 | 53.4 | 13.8 | 40.7 | 18.8 | 35.4 | 4.12 | 12.1 | 15.5 |
| ios_narrowsrc | 7.71 | 117 | 73.4 | 98.2 | 76.9 | 27.9 | 3.96 | 97.1 | 71.4 |
| ios_narrowsrc_truef | 6.91 | 116 | 13 | 26 | 19.2 | 16.9 | 2.93 | 9.2 | 17.1 |
| drop5_repeat | 7.57 | 3.96 | 12 | 9.02 | 15.2 | 6.08 | 7.2 | 16.6 | 16.6 |
| drop2_skip | 58.2 | 25.1 | 69.3 | 59.2 | 80 | 23.2 | 50.7 | 93.2 | 78 |
| img_down2 | 1.24 | 3.84 | 1.81 | 3.04 | 1.98 | 2.43 | 0.623 | 1.59 | 1.82 |

**desired accel m/s^2**, mean |change| / mean |change| of `ctl_noise1` (floor in the first row, absolute)

| case | original | tesla | gm | toyota3 | ram | hyundai2 | original turn steps | gm turn steps | ram turn steps |
|---|---|---|---|---|---|---|---|---|---|
| floor (ctl_noise1), absolute | 0.014 | 0.013 | 0.0081 | 0.0023 | 0.015 | 0.01 | 0.017 | 0.01 | 0.017 |
| ctl_yaw_+0.1 | 1.02 | 1.22 | 1.09 | 3.37 | 1.17 | 2.01 | 0.556 | 0.65 | 0.707 |
| cal_zero | 34.3 | 24.7 | 13.5 | 29.9 | 15.9 | 20 | 9.54 | 10.1 | 17 |
| cal_yaw_+1 | 3.97 | 3.18 | 6.81 | 19.6 | 4.78 | 6 | 1.58 | 3.87 | 1.86 |
| cal_yaw_-1 | 6.23 | 3.91 | 4.21 | 22.5 | 11.6 | 9.54 | 4.01 | 2.83 | 4.34 |
| cal_yaw_+2 | 4.18 | 5.8 | 9.94 | 20.5 | 6.71 | 5.41 | 4.8 | 6.71 | 3.56 |
| cal_yaw_-2 | 6.92 | 8.88 | 6.98 | 23.9 | 9.41 | 13.1 | 5.02 | 4.96 | 8.68 |
| cal_pitch_+1 | 51.8 | 46.3 | 16.5 | 306 | 37.2 | 28.1 | 14.3 | 7.47 | 10.1 |
| cal_pitch_-1 | 8.28 | 11.8 | 21.1 | 29.1 | 6.03 | 11.3 | 7.97 | 8.42 | 4.51 |
| cal_pitch_+2 | 36.2 | 45.2 | 25.1 | 212 | 49.6 | 28.3 | 10.3 | 12.9 | 20.9 |
| cal_pitch_-2 | 11.7 | 10.9 | 20.3 | 19.5 | 9.24 | 12.1 | 14 | 8.6 | 9.31 |
| cal_roll_+2 | 3.8 | 2.68 | 7.5 | 8.17 | 5.33 | 2.89 | 3.58 | 7.6 | 11 |
| foc_both_+5 | 3.25 | 3.94 | 5.77 | 6.28 | 10.8 | 3.17 | 1.89 | 5.63 | 5.77 |
| foc_both_-5 | 4.68 | 2.59 | 5.98 | 5.43 | 10.4 | 5.13 | 3.02 | 4.88 | 5.94 |
| foc_wide_+5 | 0.23 | 0.306 | 0.88 | 0.533 | 0.606 | 0.96 | 0.297 | 0.821 | 1.07 |
| foc_wide_-5 | 0.259 | 0.255 | 0.965 | 0.457 | 0.525 | 0.602 | 0.284 | 0.881 | 0.736 |
| wide_lag2 | 0.209 | 0.299 | 0.74 | 0.431 | 0.484 | 0.666 | 0.159 | 0.812 | 0.959 |
| wide_frozen | 2.1 | 2.4 | 5.44 | 5.01 | 2.37 | 6.79 | 3.76 | 6.16 | 3.13 |
| wide_grey | 4.75 | 5.05 | 7.37 | 4.41 | 4.6 | 5.61 | 5.37 | 7.42 | 7.09 |
| one_narrow | 1.22 | 2.06 | 4.74 | 2.67 | 2.47 | 3.23 | 0.655 | 5.05 | 5.24 |
| narrow_lag2 | 2.4 | 1.93 | 5.47 | 5.86 | 2.52 | 3.37 | 2.24 | 5.63 | 4.15 |
| narrow_frozen | 21.8 | 75.8 | 78.2 | 47.6 | 78 | 53.3 | 5.64 | 68.4 | 81.3 |
| narrow_grey | 27.7 | 79.5 | 73.6 | 40.9 | 77.9 | 80 | 8.24 | 53.6 | 82.3 |
| one_wide | 18.8 | 20 | 20.7 | 35.4 | 12.5 | 27.7 | 9.16 | 13.6 | 8.04 |
| ios_narrowsrc | 32.8 | 20.4 | 37.5 | 50.2 | 43.8 | 13.2 | 9.11 | 31.5 | 59 |
| ios_narrowsrc_truef | 37.6 | 22.8 | 12 | 29.8 | 15.7 | 20.5 | 9.88 | 9.88 | 16.8 |
| drop5_repeat | 4.62 | 3.38 | 5.42 | 7.3 | 7.34 | 6.38 | 2.17 | 5.1 | 7.14 |
| drop2_skip | 43.9 | 77.4 | 134 | 58.7 | 25.5 | 69.9 | 32.6 | 52.9 | 20 |
| img_down2 | 2.5 | 3.23 | 1.99 | 1.87 | 1.78 | 2.79 | 2.08 | 1.78 | 1.19 |

All metrics (generated by `more_report.py`):

### Cross-clip summary as multiples of each clip's noise floor, all warm steps

Cell = mean |change| of the case divided by mean |change| of `ctl_noise1` (+-1 grey level of random noise) in the same clip and on the same steps. Above 1 means larger than the clip's own noise floor.


**curvature 1/m** (floor, absolute: original 4.7e-05, tesla 2.6e-05, gm 0.00019, toyota3 4e-06, ram 6.6e-05, hyundai2 2.4e-05)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 1.31 | 3.13 | 2.25 | 3.32 | 2.74 | 3.42 |
| ctl_yaw_+0.1 | 2.16 | 7.44 | 1.95 | 9.11 | 2.81 | 6.2 |
| cal_zero | 6.68 | 118 | 13.2 | 25.7 | 19.5 | 16.5 |
| cal_yaw_+1 | 13.7 | 68.9 | 8.61 | 35.6 | 17.2 | 33.9 |
| cal_yaw_-1 | 17.5 | 71.9 | 10.5 | 65.9 | 17.5 | 68.6 |
| cal_yaw_+2 | 16.3 | 108 | 10.1 | 72.4 | 21.3 | 38.7 |
| cal_yaw_-2 | 29.8 | 162 | 18.5 | 120 | 30.3 | 122 |
| cal_pitch_+1 | 3.98 | 10.7 | 5.07 | 15.2 | 7.28 | 20.4 |
| cal_pitch_-1 | 2.61 | 7.45 | 5.22 | 6.67 | 5.97 | 10.2 |
| cal_pitch_+2 | 5.59 | 15.3 | 6.07 | 15.4 | 9.63 | 16 |
| cal_pitch_-2 | 2.89 | 8.77 | 6.58 | 4.92 | 9.83 | 9.96 |
| cal_roll_+2 | 2.73 | 10.5 | 5.82 | 6.33 | 8.54 | 5.24 |
| foc_both_+5 | 3.16 | 7.38 | 6.82 | 6.52 | 7.67 | 4.67 |
| foc_both_-5 | 3.65 | 6.85 | 6.87 | 7.65 | 7.88 | 5.65 |
| foc_narrow_+5 | 3.15 | 7.43 | 6.64 | 6.46 | 7.22 | 4.74 |
| foc_narrow_-5 | 3.63 | 6.76 | 7.03 | 7.62 | 7.53 | 5.57 |
| foc_wide_+5 | 0.208 | 0.784 | 0.849 | 0.396 | 1.05 | 0.86 |
| foc_wide_-5 | 0.205 | 0.752 | 1.02 | 0.393 | 1.2 | 0.861 |
| wide_lag2 | 0.262 | 0.611 | 0.929 | 0.448 | 1.37 | 0.992 |
| wide_frozen | 1.01 | 3.73 | 4.85 | 2.02 | 3.33 | 3.45 |
| wide_grey | 1.77 | 8.65 | 6.52 | 5.95 | 6.45 | 5.82 |
| one_narrow | 0.428 | 2.62 | 4.64 | 1.1 | 3.51 | 2.11 |
| narrow_lag2 | 4.25 | 5.92 | 9.94 | 5.76 | 11.4 | 12.3 |
| narrow_frozen | 69.6 | 32.6 | 128 | 79.5 | 180 | 53.2 |
| narrow_grey | 71 | 35.8 | 127 | 95.1 | 182 | 67.6 |
| one_wide | 6.09 | 53.4 | 13.8 | 40.7 | 18.8 | 35.4 |
| ios_narrowsrc | 7.71 | 117 | 73.4 | 98.2 | 76.9 | 27.9 |
| ios_widesrc | 44.1 | 25.3 | 77.9 | 70.2 | 130 | 37.9 |
| ios_narrowsrc_truef | 6.91 | 116 | 13 | 26 | 19.2 | 16.9 |
| drop5_repeat | 7.57 | 3.96 | 12 | 9.02 | 15.2 | 6.08 |
| drop2_skip | 58.2 | 25.1 | 69.3 | 59.2 | 80 | 23.2 |
| img_down2 | 1.24 | 3.84 | 1.81 | 3.04 | 1.98 | 2.43 |

**desired accel m/s^2** (floor, absolute: original 0.014, tesla 0.013, gm 0.0081, toyota3 0.0023, ram 0.015, hyundai2 0.01)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 4.19 | 4.33 | 2.79 | 9.64 | 2.17 | 2.35 |
| ctl_yaw_+0.1 | 1.02 | 1.22 | 1.09 | 3.37 | 1.17 | 2.01 |
| cal_zero | 34.3 | 24.7 | 13.5 | 29.9 | 15.9 | 20 |
| cal_yaw_+1 | 3.97 | 3.18 | 6.81 | 19.6 | 4.78 | 6 |
| cal_yaw_-1 | 6.23 | 3.91 | 4.21 | 22.5 | 11.6 | 9.54 |
| cal_yaw_+2 | 4.18 | 5.8 | 9.94 | 20.5 | 6.71 | 5.41 |
| cal_yaw_-2 | 6.92 | 8.88 | 6.98 | 23.9 | 9.41 | 13.1 |
| cal_pitch_+1 | 51.8 | 46.3 | 16.5 | 306 | 37.2 | 28.1 |
| cal_pitch_-1 | 8.28 | 11.8 | 21.1 | 29.1 | 6.03 | 11.3 |
| cal_pitch_+2 | 36.2 | 45.2 | 25.1 | 212 | 49.6 | 28.3 |
| cal_pitch_-2 | 11.7 | 10.9 | 20.3 | 19.5 | 9.24 | 12.1 |
| cal_roll_+2 | 3.8 | 2.68 | 7.5 | 8.17 | 5.33 | 2.89 |
| foc_both_+5 | 3.25 | 3.94 | 5.77 | 6.28 | 10.8 | 3.17 |
| foc_both_-5 | 4.68 | 2.59 | 5.98 | 5.43 | 10.4 | 5.13 |
| foc_narrow_+5 | 3.2 | 4.05 | 5.73 | 6.26 | 10.7 | 2.99 |
| foc_narrow_-5 | 4.69 | 2.52 | 6.3 | 5.16 | 10.3 | 5.05 |
| foc_wide_+5 | 0.23 | 0.306 | 0.88 | 0.533 | 0.606 | 0.96 |
| foc_wide_-5 | 0.259 | 0.255 | 0.965 | 0.457 | 0.525 | 0.602 |
| wide_lag2 | 0.209 | 0.299 | 0.74 | 0.431 | 0.484 | 0.666 |
| wide_frozen | 2.1 | 2.4 | 5.44 | 5.01 | 2.37 | 6.79 |
| wide_grey | 4.75 | 5.05 | 7.37 | 4.41 | 4.6 | 5.61 |
| one_narrow | 1.22 | 2.06 | 4.74 | 2.67 | 2.47 | 3.23 |
| narrow_lag2 | 2.4 | 1.93 | 5.47 | 5.86 | 2.52 | 3.37 |
| narrow_frozen | 21.8 | 75.8 | 78.2 | 47.6 | 78 | 53.3 |
| narrow_grey | 27.7 | 79.5 | 73.6 | 40.9 | 77.9 | 80 |
| one_wide | 18.8 | 20 | 20.7 | 35.4 | 12.5 | 27.7 |
| ios_narrowsrc | 32.8 | 20.4 | 37.5 | 50.2 | 43.8 | 13.2 |
| ios_widesrc | 72.8 | 38.7 | 61.7 | 21.4 | 39 | 54.5 |
| ios_narrowsrc_truef | 37.6 | 22.8 | 12 | 29.8 | 15.7 | 20.5 |
| drop5_repeat | 4.62 | 3.38 | 5.42 | 7.3 | 7.34 | 6.38 |
| drop2_skip | 43.9 | 77.4 | 134 | 58.7 | 25.5 | 69.9 |
| img_down2 | 2.5 | 3.23 | 1.99 | 1.87 | 1.78 | 2.79 |

**plan x @10s m** (floor, absolute: original 1.3, tesla 0.96, gm 0.52, toyota3 0.42, ram 0.86, hyundai2 1)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 3.31 | 2.85 | 2.34 | 9.39 | 3.12 | 2.92 |
| ctl_yaw_+0.1 | 1.46 | 1.61 | 1.72 | 1.64 | 2.36 | 2.73 |
| cal_zero | 18.9 | 14.7 | 13.7 | 22.8 | 17 | 8.16 |
| cal_yaw_+1 | 2.41 | 2.34 | 4.12 | 7.74 | 7.92 | 4.01 |
| cal_yaw_-1 | 3.17 | 3.3 | 4.59 | 7.68 | 11.7 | 5.36 |
| cal_yaw_+2 | 3.27 | 3.14 | 5.19 | 8.78 | 8.15 | 3.76 |
| cal_yaw_-2 | 4.07 | 2.97 | 8.67 | 6.21 | 10.8 | 8.02 |
| cal_pitch_+1 | 26.3 | 20.3 | 6.74 | 97.1 | 25.4 | 10.1 |
| cal_pitch_-1 | 10.3 | 4.77 | 6.06 | 42.9 | 7.94 | 7.49 |
| cal_pitch_+2 | 18.5 | 20.5 | 9.84 | 57.7 | 30.9 | 7.18 |
| cal_pitch_-2 | 8.77 | 6.52 | 10.1 | 8.59 | 8.23 | 8.62 |
| cal_roll_+2 | 3.32 | 2.84 | 5.48 | 4.51 | 4.82 | 2.92 |
| foc_both_+5 | 7.77 | 4.72 | 4.02 | 23 | 5.35 | 5.73 |
| foc_both_-5 | 6.42 | 5.95 | 3.94 | 20.8 | 4.94 | 4.22 |
| foc_narrow_+5 | 7.73 | 4.81 | 3.96 | 23 | 5.3 | 5.23 |
| foc_narrow_-5 | 6.34 | 5.91 | 3.89 | 21.2 | 4.77 | 4.14 |
| foc_wide_+5 | 0.269 | 0.364 | 0.792 | 0.37 | 0.768 | 1.21 |
| foc_wide_-5 | 0.28 | 0.317 | 0.862 | 0.396 | 0.782 | 0.954 |
| wide_lag2 | 0.323 | 0.39 | 0.833 | 0.397 | 0.781 | 1.22 |
| wide_frozen | 1.62 | 3.2 | 3.16 | 3.79 | 2.52 | 5.66 |
| wide_grey | 2.31 | 3.03 | 3.62 | 3.64 | 4.16 | 5.4 |
| one_narrow | 0.905 | 1.47 | 3.53 | 1.84 | 1.57 | 2.12 |
| narrow_lag2 | 3.52 | 3.26 | 4.64 | 3.93 | 4.55 | 5.89 |
| narrow_frozen | 105 | 79.6 | 92 | 724 | 136 | 78.7 |
| narrow_grey | 84.1 | 48.5 | 84.2 | 559 | 83 | 29.5 |
| one_wide | 17 | 12.6 | 11.3 | 36.3 | 18 | 10.1 |
| ios_narrowsrc | 20.1 | 16.6 | 23.5 | 266 | 54.5 | 33.3 |
| ios_widesrc | 86.3 | 157 | 91.3 | 289 | 142 | 71.8 |
| ios_narrowsrc_truef | 19.3 | 15.4 | 12.5 | 25.8 | 16 | 8.46 |
| drop5_repeat | 7.9 | 4.83 | 5.02 | 36.4 | 9.59 | 7.39 |
| drop2_skip | 86.3 | 88.3 | 37.8 | 287 | 146 | 80.4 |
| img_down2 | 2.22 | 2.58 | 1.95 | 2.49 | 2.43 | 2.33 |

**plan y @10s m** (floor, absolute: original 0.77, tesla 0.14, gm 0.45, toyota3 0.12, ram 0.42, hyundai2 0.026)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 2.21 | 3.51 | 2.19 | 5.43 | 2.4 | 3.55 |
| ctl_yaw_+0.1 | 1.42 | 1.85 | 1.49 | 4.45 | 2.02 | 4.59 |
| cal_zero | 5.83 | 6.15 | 8.32 | 11.8 | 23 | 8.72 |
| cal_yaw_+1 | 2.84 | 4.57 | 3.99 | 14.1 | 5.36 | 22.4 |
| cal_yaw_-1 | 2.88 | 5.74 | 3.86 | 20.4 | 4.49 | 32.6 |
| cal_yaw_+2 | 5.61 | 5.99 | 6.73 | 31.2 | 8.11 | 36 |
| cal_yaw_-2 | 4.3 | 15.1 | 6.28 | 53.8 | 7.53 | 51.3 |
| cal_pitch_+1 | 7.87 | 17.4 | 4.49 | 12.9 | 9.59 | 10.1 |
| cal_pitch_-1 | 3.58 | 4.08 | 4.73 | 8.16 | 4.81 | 6.43 |
| cal_pitch_+2 | 5.42 | 13.8 | 5.6 | 15.2 | 13.2 | 11 |
| cal_pitch_-2 | 4.44 | 6.47 | 5.67 | 6.72 | 8.37 | 8.23 |
| cal_roll_+2 | 2.56 | 3.86 | 6.14 | 5.18 | 4.09 | 4.44 |
| foc_both_+5 | 2.32 | 2.96 | 3.83 | 4.71 | 4.38 | 4.5 |
| foc_both_-5 | 2.94 | 3.38 | 3.51 | 5.24 | 4.45 | 4.7 |
| foc_narrow_+5 | 2.27 | 2.96 | 4 | 4.67 | 4.18 | 4.38 |
| foc_narrow_-5 | 2.9 | 3.31 | 3.76 | 5.1 | 4.42 | 4.69 |
| foc_wide_+5 | 0.156 | 0.261 | 0.882 | 0.319 | 0.94 | 1.01 |
| foc_wide_-5 | 0.169 | 0.287 | 0.961 | 0.297 | 1.04 | 0.851 |
| wide_lag2 | 0.185 | 0.337 | 0.82 | 0.307 | 1.08 | 1.05 |
| wide_frozen | 0.706 | 1.69 | 3.66 | 1.49 | 2.58 | 3.9 |
| wide_grey | 1.72 | 3.92 | 5.06 | 4.28 | 2.86 | 4.91 |
| one_narrow | 0.418 | 0.927 | 4.22 | 1.38 | 1.69 | 2.09 |
| narrow_lag2 | 3.05 | 3.57 | 5.46 | 6.06 | 4.66 | 10.2 |
| narrow_frozen | 30.2 | 36.6 | 47.4 | 83 | 39.9 | 46.7 |
| narrow_grey | 30.5 | 39.5 | 47.8 | 86.3 | 40.3 | 83.4 |
| one_wide | 5.85 | 9.26 | 7.25 | 23.6 | 9.03 | 41.7 |
| ios_narrowsrc | 5.31 | 6.61 | 20.4 | 15.8 | 29.3 | 15.7 |
| ios_widesrc | 14 | 15 | 21.5 | 47.5 | 32.1 | 40.3 |
| ios_narrowsrc_truef | 5.91 | 5.96 | 7.78 | 11.9 | 21.6 | 8.58 |
| drop5_repeat | 3.53 | 4.15 | 4.11 | 9.26 | 5.73 | 5.77 |
| drop2_skip | 32.5 | 66.4 | 30.1 | 74.6 | 72.3 | 40.1 |
| img_down2 | 1.73 | 2.57 | 1.57 | 3.53 | 1.88 | 2.51 |

**lane prob** (floor, absolute: original 0.011, tesla 0.0055, gm 0.0028, toyota3 0.0051, ram 0.0034, hyundai2 0.0026)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 1.4 | 3.11 | 2.97 | 2.87 | 3.12 | 3.63 |
| ctl_yaw_+0.1 | 1.06 | 2.33 | 2.39 | 2.04 | 2.25 | 2.95 |
| cal_zero | 5.02 | 10.6 | 12.7 | 4.66 | 14.8 | 11.4 |
| cal_yaw_+1 | 3.14 | 5.16 | 8 | 6.91 | 9.54 | 6.63 |
| cal_yaw_-1 | 2.76 | 5.5 | 6.55 | 4.79 | 7.88 | 7.79 |
| cal_yaw_+2 | 4.55 | 7.57 | 12.6 | 8.99 | 11.6 | 8.61 |
| cal_yaw_-2 | 4.16 | 8.63 | 9.17 | 5.89 | 12.4 | 16.2 |
| cal_pitch_+1 | 4.78 | 10 | 8.72 | 10.7 | 13.1 | 12.5 |
| cal_pitch_-1 | 2.37 | 6.36 | 6.25 | 4.83 | 5.76 | 10.5 |
| cal_pitch_+2 | 5.24 | 11.5 | 10.7 | 6.96 | 18.8 | 10.7 |
| cal_pitch_-2 | 2.34 | 7.79 | 7.2 | 7.08 | 6.99 | 14.7 |
| cal_roll_+2 | 2.35 | 6.19 | 6.33 | 4.99 | 6.9 | 6.57 |
| foc_both_+5 | 2.08 | 4.93 | 4.52 | 2.7 | 5.19 | 4.66 |
| foc_both_-5 | 2.38 | 5.25 | 4.22 | 3.73 | 5.66 | 5.06 |
| foc_narrow_+5 | 2.09 | 4.79 | 4.55 | 2.67 | 5.14 | 4.53 |
| foc_narrow_-5 | 2.38 | 5.15 | 4.12 | 3.64 | 5.63 | 4.97 |
| foc_wide_+5 | 0.261 | 0.826 | 0.882 | 0.42 | 0.99 | 1.05 |
| foc_wide_-5 | 0.266 | 0.841 | 1.03 | 0.438 | 1.02 | 1.15 |
| wide_lag2 | 0.374 | 0.732 | 0.901 | 0.482 | 1.02 | 1.33 |
| wide_frozen | 0.953 | 3.67 | 4.27 | 3.74 | 2.82 | 4.21 |
| wide_grey | 2.2 | 4.64 | 8.02 | 3.75 | 4.87 | 5.98 |
| one_narrow | 0.576 | 2.48 | 3.12 | 1.44 | 2.17 | 2.21 |
| narrow_lag2 | 3.25 | 4.6 | 4.83 | 4.03 | 6.8 | 8.64 |
| narrow_frozen | 15.7 | 44.5 | 32 | 103 | 100 | 69.9 |
| narrow_grey | 28 | 45.3 | 81.7 | 67.7 | 79.1 | 62.9 |
| one_wide | 4.77 | 9.53 | 16.3 | 10.4 | 16.1 | 26.5 |
| ios_narrowsrc | 5.54 | 12.4 | 19.1 | 23.9 | 56.8 | 30.6 |
| ios_widesrc | 13.4 | 20 | 34.7 | 30.4 | 59.3 | 63.7 |
| ios_narrowsrc_truef | 5.08 | 11.8 | 12.9 | 4.12 | 14.7 | 11.2 |
| drop5_repeat | 1.23 | 2.82 | 2.58 | 2.36 | 3.91 | 3.18 |
| drop2_skip | 8.98 | 18 | 15.8 | 20.5 | 31.1 | 19.9 |
| img_down2 | 1.3 | 3.61 | 2.07 | 1.78 | 2.29 | 2.73 |

**lead prob** (floor, absolute: original 0.0032, tesla 0.0082, gm 0.0043, toyota3 0.0012, ram 0.0061, hyundai2 0.00074)

| case | original | tesla | gm | toyota3 | ram | hyundai2 |
|---|---|---|---|---|---|---|
| ctl_pitch_+0.1 | 2.78 | 4.47 | 2.68 | 4.3 | 3.29 | 4.07 |
| ctl_yaw_+0.1 | 2.2 | 3.98 | 1.82 | 2.74 | 2.81 | 3.06 |
| cal_zero | 4.71 | 9.84 | 8.55 | 8.21 | 50 | 11 |
| cal_yaw_+1 | 3.07 | 4.48 | 4.95 | 11.6 | 10 | 6.85 |
| cal_yaw_-1 | 2.91 | 5.53 | 4.85 | 10.3 | 14.8 | 7.91 |
| cal_yaw_+2 | 4.43 | 3.7 | 7.19 | 13.1 | 12 | 9.11 |
| cal_yaw_-2 | 4.64 | 5.55 | 11.4 | 26.1 | 14.1 | 18.7 |
| cal_pitch_+1 | 3.74 | 6.5 | 17.7 | 25.3 | 9.1 | 12.7 |
| cal_pitch_-1 | 4.1 | 5.39 | 5.71 | 9.94 | 6.42 | 9.92 |
| cal_pitch_+2 | 4.05 | 5.6 | 25.4 | 16.3 | 10.8 | 10.5 |
| cal_pitch_-2 | 4.21 | 7.37 | 7.8 | 10.2 | 8.08 | 11.4 |
| cal_roll_+2 | 3.66 | 5.73 | 5.11 | 5.51 | 5.35 | 4.6 |
| foc_both_+5 | 4.32 | 5.99 | 4.55 | 3.15 | 5.44 | 4.66 |
| foc_both_-5 | 3.96 | 6.6 | 3.72 | 3.71 | 5.88 | 5.33 |
| foc_narrow_+5 | 4.28 | 6.08 | 4.4 | 3.17 | 5.43 | 4.62 |
| foc_narrow_-5 | 3.94 | 6.61 | 3.72 | 3.77 | 5.65 | 5.15 |
| foc_wide_+5 | 0.404 | 0.47 | 0.982 | 0.336 | 0.924 | 0.904 |
| foc_wide_-5 | 0.422 | 0.52 | 0.989 | 0.368 | 0.847 | 0.916 |
| wide_lag2 | 0.504 | 0.46 | 0.982 | 0.385 | 1.06 | 1.02 |
| wide_frozen | 1.7 | 2.05 | 5.2 | 1.58 | 2.79 | 3.6 |
| wide_grey | 3.18 | 6.18 | 6.11 | 1.91 | 6.82 | 7.87 |
| one_narrow | 0.63 | 1.14 | 3.96 | 0.757 | 2.14 | 2.34 |
| narrow_lag2 | 5.46 | 6.34 | 4.93 | 4.45 | 6.82 | 11.5 |
| narrow_frozen | 247 | 77 | 99.7 | 9.95 | 62.4 | 782 |
| narrow_grey | 258 | 79.6 | 14.8 | 50.8 | 44.2 | 1.22e+03 |
| one_wide | 6.5 | 8.29 | 10.2 | 26.3 | 11.9 | 105 |
| ios_narrowsrc | 4.92 | 9.69 | 14.1 | 212 | 52.4 | 10.4 |
| ios_widesrc | 58.5 | 33.4 | 55 | 16.5 | 69.1 | 360 |
| ios_narrowsrc_truef | 4.75 | 9.83 | 8.57 | 7.88 | 50.7 | 13.2 |
| drop5_repeat | 1.81 | 1.83 | 2.6 | 2.49 | 3.72 | 4.52 |
| drop2_skip | 17 | 12.9 | 13.2 | 12.7 | 14.9 | 36.7 |
| img_down2 | 2.72 | 4.81 | 1.91 | 3.35 | 2.66 | 2.59 |

### Signed change in curvature on turn steps (sign of the reference curvature x (perturbed - reference), mean; in brackets as a fraction of mean |reference curvature| on those steps)

Positive: perturbed run commands more curvature in the direction of the turn; negative: less.

| case | original | gm | ram |
|---|---|---|---|
| ctl_noise1 | 0.00017 (+1%) | 7.4e-05 (+0%) | -2.7e-05 (-0%) |
| ctl_pitch_+0.1 | 1.3e-05 (+0%) | -7.3e-05 (-0%) | -0.00013 (-0%) |
| ctl_yaw_+0.1 | -0.00011 (-1%) | -0.00013 (-0%) | -0.0002 (-0%) |
| cal_zero | 0.00014 (+1%) | 0.0005 (+1%) | 0.0026 (+6%) |
| cal_yaw_+1 | -0.00091 (-8%) | -0.00028 (-0%) | -0.0014 (-3%) |
| cal_yaw_-1 | 0.00098 (+8%) | 0.0002 (+0%) | 0.0013 (+3%) |
| cal_yaw_+2 | -0.001 (-8%) | -0.00075 (-1%) | -0.0024 (-5%) |
| cal_yaw_-2 | 0.002 (+16%) | 0.00017 (+0%) | 0.0025 (+5%) |
| cal_pitch_+1 | 0.00022 (+2%) | -0.00073 (-1%) | -0.00031 (-1%) |
| cal_pitch_-1 | -8.4e-05 (-1%) | 0.00019 (+0%) | 0.00036 (+1%) |
| cal_pitch_+2 | 0.00061 (+5%) | 0.00022 (+0%) | -0.00083 (-2%) |
| cal_pitch_-2 | 9.3e-05 (+1%) | 0.00066 (+1%) | -0.00027 (-1%) |
| cal_roll_+2 | 0.00023 (+2%) | -0.0011 (-2%) | -0.0015 (-3%) |
| foc_both_+5 | -0.00041 (-3%) | -0.0022 (-4%) | -0.0017 (-4%) |
| foc_both_-5 | 0.00044 (+4%) | 0.0021 (+4%) | 0.0018 (+4%) |
| foc_narrow_+5 | -0.00041 (-3%) | -0.0022 (-4%) | -0.0015 (-3%) |
| foc_narrow_-5 | 0.00044 (+4%) | 0.0021 (+4%) | 0.0017 (+4%) |
| foc_wide_+5 | -5.4e-06 (-0%) | -3.7e-05 (-0%) | -0.00016 (-0%) |
| foc_wide_-5 | 7.5e-06 (+0%) | -2.4e-05 (-0%) | 0.00013 (+0%) |
| wide_lag2 | 7.9e-07 (+0%) | 2.5e-05 (+0%) | -9.7e-05 (-0%) |
| wide_frozen | 0.00011 (+1%) | -0.0014 (-2%) | 3.4e-05 (+0%) |
| wide_grey | 2.4e-05 (+0%) | -0.0012 (-2%) | -0.00017 (-0%) |
| one_narrow | 2.7e-05 (+0%) | 0.00022 (+0%) | -0.00029 (-1%) |
| narrow_lag2 | -8.3e-05 (-1%) | -0.00024 (-0%) | 0.00021 (+0%) |
| narrow_frozen | -0.012 (-100%) | -0.056 (-100%) | -0.048 (-100%) |
| narrow_grey | -0.012 (-102%) | -0.056 (-99%) | -0.048 (-101%) |
| one_wide | 0.00066 (+5%) | 0.0026 (+5%) | 0.0035 (+7%) |
| ios_narrowsrc | 0.00055 (+5%) | 0.026 (+46%) | 0.017 (+35%) |
| ios_widesrc | -0.0073 (-60%) | -0.033 (-59%) | -0.036 (-74%) |
| ios_narrowsrc_truef | 0.00021 (+2%) | 0.0009 (+2%) | 0.0028 (+6%) |
| drop5_repeat | 4.1e-05 (+0%) | -0.00035 (-1%) | -0.00035 (-1%) |
| drop2_skip | 0.0086 (+71%) | 0.022 (+40%) | 0.018 (+38%) |
| img_down2 | -5.7e-06 (-0%) | -3.6e-05 (-0%) | 7.7e-05 (+0%) |

### Cross-clip summary as multiples of the noise floor, turn steps only (|reference curvature| > 0.01)

Cell = mean |change| of the case divided by mean |change| of `ctl_noise1` (+-1 grey level of random noise) in the same clip and on the same steps. Above 1 means larger than the clip's own noise floor.


**curvature 1/m** (floor, absolute: original 0.00017, gm 0.0003, ram 0.00024)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 0.696 | 2.22 | 2.59 |
| ctl_yaw_+0.1 | 0.723 | 1.65 | 1.82 |
| cal_zero | 2.86 | 9.32 | 17.2 |
| cal_yaw_+1 | 5.41 | 5.54 | 7.02 |
| cal_yaw_-1 | 5.77 | 5.15 | 6.23 |
| cal_yaw_+2 | 6 | 10.2 | 11.8 |
| cal_yaw_-2 | 11.7 | 9.25 | 11.5 |
| cal_pitch_+1 | 1.79 | 4.68 | 5.09 |
| cal_pitch_-1 | 1.25 | 4.98 | 5.71 |
| cal_pitch_+2 | 4.17 | 5.77 | 7.37 |
| cal_pitch_-2 | 1.43 | 5.9 | 9.83 |
| cal_roll_+2 | 1.47 | 5.96 | 8.26 |
| foc_both_+5 | 2.43 | 8.1 | 7.95 |
| foc_both_-5 | 2.63 | 8.38 | 8.16 |
| foc_narrow_+5 | 2.47 | 7.96 | 7.45 |
| foc_narrow_-5 | 2.62 | 8.36 | 7.74 |
| foc_wide_+5 | 0.153 | 0.78 | 1.02 |
| foc_wide_-5 | 0.138 | 0.874 | 1.18 |
| wide_lag2 | 0.174 | 0.77 | 1.43 |
| wide_frozen | 0.798 | 5.91 | 3.31 |
| wide_grey | 0.77 | 6.62 | 6.63 |
| one_narrow | 0.228 | 3.81 | 3.66 |
| narrow_lag2 | 1.76 | 11.7 | 11.4 |
| narrow_frozen | 71.1 | 186 | 203 |
| narrow_grey | 72.5 | 185 | 204 |
| one_wide | 4.12 | 12.1 | 15.5 |
| ios_narrowsrc | 3.96 | 97.1 | 71.4 |
| ios_widesrc | 43 | 109 | 150 |
| ios_narrowsrc_truef | 2.93 | 9.2 | 17.1 |
| drop5_repeat | 7.2 | 16.6 | 16.6 |
| drop2_skip | 50.7 | 93.2 | 78 |
| img_down2 | 0.623 | 1.59 | 1.82 |

**desired accel m/s^2** (floor, absolute: original 0.017, gm 0.01, ram 0.017)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 1.4 | 1.73 | 1.41 |
| ctl_yaw_+0.1 | 0.556 | 0.65 | 0.707 |
| cal_zero | 9.54 | 10.1 | 17 |
| cal_yaw_+1 | 1.58 | 3.87 | 1.86 |
| cal_yaw_-1 | 4.01 | 2.83 | 4.34 |
| cal_yaw_+2 | 4.8 | 6.71 | 3.56 |
| cal_yaw_-2 | 5.02 | 4.96 | 8.68 |
| cal_pitch_+1 | 14.3 | 7.47 | 10.1 |
| cal_pitch_-1 | 7.97 | 8.42 | 4.51 |
| cal_pitch_+2 | 10.3 | 12.9 | 20.9 |
| cal_pitch_-2 | 14 | 8.6 | 9.31 |
| cal_roll_+2 | 3.58 | 7.6 | 11 |
| foc_both_+5 | 1.89 | 5.63 | 5.77 |
| foc_both_-5 | 3.02 | 4.88 | 5.94 |
| foc_narrow_+5 | 1.79 | 5.69 | 4.99 |
| foc_narrow_-5 | 2.69 | 5.23 | 5.91 |
| foc_wide_+5 | 0.297 | 0.821 | 1.07 |
| foc_wide_-5 | 0.284 | 0.881 | 0.736 |
| wide_lag2 | 0.159 | 0.812 | 0.959 |
| wide_frozen | 3.76 | 6.16 | 3.13 |
| wide_grey | 5.37 | 7.42 | 7.09 |
| one_narrow | 0.655 | 5.05 | 5.24 |
| narrow_lag2 | 2.24 | 5.63 | 4.15 |
| narrow_frozen | 5.64 | 68.4 | 81.3 |
| narrow_grey | 8.24 | 53.6 | 82.3 |
| one_wide | 9.16 | 13.6 | 8.04 |
| ios_narrowsrc | 9.11 | 31.5 | 59 |
| ios_widesrc | 70.8 | 62.9 | 33.8 |
| ios_narrowsrc_truef | 9.88 | 9.88 | 16.8 |
| drop5_repeat | 2.17 | 5.1 | 7.14 |
| drop2_skip | 32.6 | 52.9 | 20 |
| img_down2 | 2.08 | 1.78 | 1.19 |

**plan x @10s m** (floor, absolute: original 2.1, gm 0.67, ram 0.52)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 1.21 | 2.15 | 3.31 |
| ctl_yaw_+0.1 | 0.863 | 1.52 | 1.89 |
| cal_zero | 3.3 | 14.3 | 14.9 |
| cal_yaw_+1 | 2.36 | 3.25 | 3.86 |
| cal_yaw_-1 | 2.56 | 3.24 | 4.14 |
| cal_yaw_+2 | 3.78 | 4.7 | 7.49 |
| cal_yaw_-2 | 3.31 | 4.94 | 7.01 |
| cal_pitch_+1 | 4.71 | 5 | 10.9 |
| cal_pitch_-1 | 4.51 | 4.07 | 6.16 |
| cal_pitch_+2 | 3.55 | 7.3 | 22.2 |
| cal_pitch_-2 | 7.88 | 10.2 | 16.6 |
| cal_roll_+2 | 2.54 | 5.09 | 8.73 |
| foc_both_+5 | 4.28 | 4.07 | 4.87 |
| foc_both_-5 | 3.16 | 3.5 | 4.51 |
| foc_narrow_+5 | 4.25 | 4.03 | 4.32 |
| foc_narrow_-5 | 3.08 | 3.42 | 4.17 |
| foc_wide_+5 | 0.227 | 0.724 | 1.15 |
| foc_wide_-5 | 0.223 | 0.868 | 1.54 |
| wide_lag2 | 0.276 | 0.869 | 1.55 |
| wide_frozen | 2.23 | 3.28 | 2.98 |
| wide_grey | 2.06 | 3.95 | 6.36 |
| one_narrow | 0.428 | 4.05 | 3.11 |
| narrow_lag2 | 2.17 | 4.86 | 6.49 |
| narrow_frozen | 51.2 | 72.3 | 117 |
| narrow_grey | 43.5 | 64.8 | 74.5 |
| one_wide | 4.3 | 9.26 | 20.4 |
| ios_narrowsrc | 4.26 | 14.7 | 28.7 |
| ios_widesrc | 41.9 | 75.4 | 72.1 |
| ios_narrowsrc_truef | 3.26 | 13.1 | 14.6 |
| drop5_repeat | 3.22 | 5.22 | 6.85 |
| drop2_skip | 27.2 | 38.6 | 77.5 |
| img_down2 | 1.29 | 1.83 | 2.13 |

**plan y @10s m** (floor, absolute: original 1.9, gm 0.61, ram 0.6)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 0.994 | 1.85 | 2.53 |
| ctl_yaw_+0.1 | 0.822 | 1.19 | 2.09 |
| cal_zero | 2.19 | 8.23 | 23.9 |
| cal_yaw_+1 | 1.47 | 3.29 | 3.63 |
| cal_yaw_-1 | 1.87 | 3.49 | 3.51 |
| cal_yaw_+2 | 1.84 | 5.58 | 7.1 |
| cal_yaw_-2 | 1.9 | 5.66 | 6.36 |
| cal_pitch_+1 | 1.64 | 4.17 | 7.41 |
| cal_pitch_-1 | 3.56 | 3.84 | 5.42 |
| cal_pitch_+2 | 1.79 | 4.45 | 15.3 |
| cal_pitch_-2 | 3.18 | 6 | 15.4 |
| cal_roll_+2 | 2.09 | 4.69 | 6.53 |
| foc_both_+5 | 1.55 | 3.82 | 4.16 |
| foc_both_-5 | 2.3 | 3.6 | 4.12 |
| foc_narrow_+5 | 1.47 | 3.97 | 3.89 |
| foc_narrow_-5 | 2.24 | 3.85 | 4.04 |
| foc_wide_+5 | 0.141 | 0.735 | 1.37 |
| foc_wide_-5 | 0.17 | 0.842 | 1.75 |
| wide_lag2 | 0.18 | 0.648 | 2.1 |
| wide_frozen | 0.955 | 4.03 | 3.25 |
| wide_grey | 1.12 | 4.56 | 5.04 |
| one_narrow | 0.269 | 3.68 | 2.85 |
| narrow_lag2 | 2.18 | 6.02 | 7.28 |
| narrow_frozen | 25.8 | 72.8 | 76.1 |
| narrow_grey | 26 | 73 | 76.9 |
| one_wide | 2.81 | 6.05 | 11.5 |
| ios_narrowsrc | 2.16 | 25.1 | 29.6 |
| ios_widesrc | 8.54 | 29 | 52 |
| ios_narrowsrc_truef | 2.1 | 8.11 | 22.1 |
| drop5_repeat | 2.46 | 4.57 | 5.36 |
| drop2_skip | 22.5 | 26.6 | 43.3 |
| img_down2 | 0.871 | 1.3 | 2.02 |

**lane prob** (floor, absolute: original 0.0075, gm 0.00073, ram 0.0014)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 1.26 | 2.38 | 2.71 |
| ctl_yaw_+0.1 | 0.93 | 1.86 | 1.73 |
| cal_zero | 5.54 | 8.48 | 22.1 |
| cal_yaw_+1 | 2.41 | 4.39 | 3.58 |
| cal_yaw_-1 | 2.45 | 3.97 | 2.84 |
| cal_yaw_+2 | 3.77 | 8.98 | 6.02 |
| cal_yaw_-2 | 6.15 | 6.21 | 4.98 |
| cal_pitch_+1 | 5.2 | 4.73 | 4.5 |
| cal_pitch_-1 | 2.85 | 5.14 | 4.62 |
| cal_pitch_+2 | 6 | 10.8 | 7.15 |
| cal_pitch_-2 | 2.83 | 7.29 | 7.44 |
| cal_roll_+2 | 2.75 | 5.27 | 5.13 |
| foc_both_+5 | 2.92 | 5.27 | 4.95 |
| foc_both_-5 | 3.48 | 4.49 | 4.07 |
| foc_narrow_+5 | 2.99 | 5.06 | 5.02 |
| foc_narrow_-5 | 3.39 | 4.37 | 4.05 |
| foc_wide_+5 | 0.359 | 0.835 | 0.736 |
| foc_wide_-5 | 0.385 | 1.06 | 1.11 |
| wide_lag2 | 0.436 | 0.998 | 1.13 |
| wide_frozen | 1.49 | 3.61 | 2.49 |
| wide_grey | 2.17 | 6.5 | 3.84 |
| one_narrow | 0.55 | 3.43 | 2.54 |
| narrow_lag2 | 3.04 | 6.17 | 6.64 |
| narrow_frozen | 20.7 | 77.2 | 245 |
| narrow_grey | 37 | 287 | 118 |
| one_wide | 6.81 | 9.01 | 9.65 |
| ios_narrowsrc | 7.51 | 23.6 | 13 |
| ios_widesrc | 27.4 | 98.6 | 24.1 |
| ios_narrowsrc_truef | 5.38 | 8.5 | 20.1 |
| drop5_repeat | 1.58 | 2.47 | 2.79 |
| drop2_skip | 24.2 | 18.6 | 11.8 |
| img_down2 | 1.1 | 1.72 | 1.75 |

**lead prob** (floor, absolute: original 0.0058, gm 0.0044, ram 0.0079)

| case | original | gm | ram |
|---|---|---|---|
| ctl_pitch_+0.1 | 2.35 | 2.35 | 2.08 |
| ctl_yaw_+0.1 | 1.98 | 1.52 | 1.58 |
| cal_zero | 5.76 | 7.2 | 9.9 |
| cal_yaw_+1 | 3.21 | 3.14 | 4.31 |
| cal_yaw_-1 | 2.69 | 3.52 | 3.59 |
| cal_yaw_+2 | 4.97 | 5.26 | 6.1 |
| cal_yaw_-2 | 7.05 | 5.36 | 6.84 |
| cal_pitch_+1 | 3.4 | 9.95 | 5.38 |
| cal_pitch_-1 | 4.79 | 4.71 | 4.05 |
| cal_pitch_+2 | 5.14 | 20.2 | 10.3 |
| cal_pitch_-2 | 4.09 | 5.76 | 7.64 |
| cal_roll_+2 | 4.24 | 5.05 | 4.69 |
| foc_both_+5 | 4.61 | 3.94 | 4.1 |
| foc_both_-5 | 4.72 | 3.83 | 3.8 |
| foc_narrow_+5 | 4.62 | 3.78 | 4.14 |
| foc_narrow_-5 | 4.66 | 3.85 | 3.56 |
| foc_wide_+5 | 0.581 | 1.08 | 0.986 |
| foc_wide_-5 | 0.482 | 1.1 | 0.935 |
| wide_lag2 | 0.604 | 1.04 | 1.4 |
| wide_frozen | 1.85 | 7.7 | 3.25 |
| wide_grey | 4.09 | 7.66 | 5.29 |
| one_narrow | 0.732 | 4.84 | 2.45 |
| narrow_lag2 | 5.41 | 5.18 | 5.25 |
| narrow_frozen | 134 | 91.1 | 54.1 |
| narrow_grey | 143 | 11.6 | 7.64 |
| one_wide | 6.01 | 8.74 | 5.87 |
| ios_narrowsrc | 5 | 9.14 | 12 |
| ios_widesrc | 37.8 | 82.2 | 74.6 |
| ios_narrowsrc_truef | 5.62 | 7.4 | 9.93 |
| drop5_repeat | 2.52 | 2.37 | 2.5 |
| drop2_skip | 28.9 | 11.7 | 7.56 |
| img_down2 | 2.71 | 1.77 | 1.57 |

## Per-clip tables

Each table: mean / worst |change| of the perturbed run against the unperturbed run over the warm steps (the first rows are the noise floor of the clip). Case names are described in "What was run". The device tables give the mean absolute distance to the comma device's own `modelV2` (first row: unperturbed run), for a subset of the cases (`--short-dev`).

### Clip original

unperturbed |curvature| warm: mean 0.0033 max 0.0138 1/m; steps with |curvature| > 0.01: 77 of 500 (v_ego there 11.8..14.0 m/s); v_ego warm 11.8..22.2 mean 16.4

#### original: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 4.7e-05 / 0.00034 | 0.014 / 0.095 | 1.3 / 16 | 0.77 / 5.3 | 0.011 / 0.16 | 0.0032 / 0.1 |
| ctl_pitch_+0.1 | 6.2e-05 / 0.00045 | 0.06 / 0.33 | 4.4 / 37 | 1.7 / 9.8 | 0.015 / 0.18 | 0.0087 / 0.24 |
| ctl_yaw_+0.1 | 0.0001 / 0.0004 | 0.014 / 0.075 | 2 / 13 | 1.1 / 9.1 | 0.011 / 0.16 | 0.0069 / 0.17 |
| cal_zero | 0.00032 / 0.0018 | 0.49 / 0.95 | 25 / 75 | 4.5 / 23 | 0.054 / 0.42 | 0.015 / 0.29 |
| cal_yaw_+1 | 0.00065 / 0.0026 | 0.057 / 0.28 | 3.2 / 33 | 2.2 / 15 | 0.034 / 0.37 | 0.0097 / 0.24 |
| cal_yaw_-1 | 0.00083 / 0.0024 | 0.089 / 0.25 | 4.3 / 18 | 2.2 / 15 | 0.03 / 0.39 | 0.0092 / 0.2 |
| cal_yaw_+2 | 0.00077 / 0.0039 | 0.06 / 0.29 | 4.4 / 23 | 4.3 / 17 | 0.049 / 0.52 | 0.014 / 0.28 |
| cal_yaw_-2 | 0.0014 / 0.0035 | 0.099 / 0.41 | 5.5 / 20 | 3.3 / 22 | 0.045 / 0.41 | 0.015 / 0.15 |
| cal_pitch_+1 | 0.00019 / 0.001 | 0.74 / 1.4 | 35 / 89 | 6.1 / 24 | 0.051 / 0.36 | 0.012 / 0.22 |
| cal_pitch_-1 | 0.00012 / 0.00089 | 0.12 / 0.63 | 14 / 81 | 2.8 / 17 | 0.025 / 0.31 | 0.013 / 0.28 |
| cal_pitch_+2 | 0.00026 / 0.0024 | 0.52 / 1 | 25 / 65 | 4.2 / 20 | 0.056 / 0.37 | 0.013 / 0.29 |
| cal_pitch_-2 | 0.00014 / 0.0009 | 0.17 / 0.74 | 12 / 86 | 3.4 / 21 | 0.025 / 0.3 | 0.013 / 0.34 |
| cal_roll_+2 | 0.00013 / 0.00078 | 0.054 / 0.3 | 4.5 / 44 | 2 / 11 | 0.025 / 0.34 | 0.012 / 0.28 |
| foc_both_+5 | 0.00015 / 0.00099 | 0.046 / 0.19 | 10 / 39 | 1.8 / 16 | 0.022 / 0.27 | 0.014 / 0.34 |
| foc_both_-5 | 0.00017 / 0.0011 | 0.067 / 0.25 | 8.6 / 26 | 2.3 / 13 | 0.026 / 0.35 | 0.012 / 0.31 |
| foc_narrow_+5 | 0.00015 / 0.00096 | 0.046 / 0.2 | 10 / 38 | 1.8 / 16 | 0.022 / 0.27 | 0.013 / 0.33 |
| foc_narrow_-5 | 0.00017 / 0.00099 | 0.067 / 0.25 | 8.5 / 26 | 2.2 / 14 | 0.025 / 0.38 | 0.012 / 0.31 |
| foc_wide_+5 | 9.8e-06 / 9e-05 | 0.0033 / 0.023 | 0.36 / 5.5 | 0.12 / 1.1 | 0.0028 / 0.038 | 0.0013 / 0.046 |
| foc_wide_-5 | 9.7e-06 / 0.00011 | 0.0037 / 0.022 | 0.38 / 4.6 | 0.13 / 1.3 | 0.0029 / 0.044 | 0.0013 / 0.038 |
| wide_lag2 | 1.2e-05 / 9.5e-05 | 0.003 / 0.027 | 0.43 / 5.2 | 0.14 / 1.8 | 0.004 / 0.053 | 0.0016 / 0.042 |
| wide_frozen | 4.8e-05 / 0.00047 | 0.03 / 0.18 | 2.2 / 20 | 0.54 / 6 | 0.01 / 0.11 | 0.0054 / 0.14 |
| wide_grey | 8.3e-05 / 0.00052 | 0.068 / 0.31 | 3.1 / 23 | 1.3 / 7.9 | 0.024 / 0.34 | 0.01 / 0.15 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 2e-05 / 0.00015 | 0.017 / 0.084 | 1.2 / 8.4 | 0.32 / 2.9 | 0.0062 / 0.092 | 0.002 / 0.034 |
| narrow_lag2 | 0.0002 / 0.0016 | 0.034 / 0.22 | 4.7 / 36 | 2.4 / 23 | 0.035 / 0.42 | 0.017 / 0.52 |
| narrow_frozen | 0.0033 / 0.014 | 0.31 / 1.4 | 1.4e+02 / 2e+02 | 23 / 72 | 0.17 / 0.55 | 0.78 / 0.9 |
| narrow_grey | 0.0034 / 0.014 | 0.39 / 1.6 | 1.1e+02 / 1.9e+02 | 24 / 73 | 0.3 / 0.82 | 0.81 / 0.93 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.00029 / 0.002 | 0.27 / 0.68 | 23 / 63 | 4.5 / 23 | 0.051 / 0.43 | 0.02 / 0.51 |
| ios_narrowsrc | 0.00036 / 0.0028 | 0.47 / 0.95 | 27 / 72 | 4.1 / 22 | 0.059 / 0.41 | 0.015 / 0.36 |
| ios_widesrc | 0.0021 / 0.0085 | 1 / 2 | 1.2e+02 / 2e+02 | 11 / 43 | 0.14 / 0.63 | 0.18 / 0.73 |
| ios_narrowsrc_truef | 0.00033 / 0.002 | 0.54 / 1.1 | 26 / 76 | 4.6 / 23 | 0.054 / 0.42 | 0.015 / 0.34 |
| drop5_repeat | 0.00036 / 0.0039 | 0.066 / 0.24 | 11 / 71 | 2.7 / 28 | 0.013 / 0.35 | 0.0057 / 0.19 |
| drop2_skip | 0.0027 / 0.011 | 0.63 / 1.5 | 1.2e+02 / 1.9e+02 | 25 / 71 | 0.096 / 0.74 | 0.053 / 0.41 |
| img_down2 | 5.8e-05 / 0.0004 | 0.036 / 0.19 | 3 / 31 | 1.3 / 10 | 0.014 / 0.19 | 0.0086 / 0.29 |

#### original: steps with |reference curvature| > 0.01 (77 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 0.00017 / 0.00034 | 0.017 / 0.087 | 2.1 / 11 | 1.9 / 5.3 | 0.0075 / 0.042 | 0.0058 / 0.027 |
| ctl_pitch_+0.1 | 0.00012 / 0.00038 | 0.023 / 0.052 | 2.6 / 15 | 1.9 / 7.2 | 0.0095 / 0.1 | 0.014 / 0.062 |
| ctl_yaw_+0.1 | 0.00012 / 0.00035 | 0.0093 / 0.053 | 1.8 / 13 | 1.6 / 9.1 | 0.007 / 0.051 | 0.011 / 0.15 |
| cal_zero | 0.00048 / 0.0015 | 0.16 / 0.39 | 7 / 23 | 4.2 / 14 | 0.042 / 0.21 | 0.033 / 0.2 |
| cal_yaw_+1 | 0.00092 / 0.0013 | 0.026 / 0.13 | 5 / 16 | 2.8 / 15 | 0.018 / 0.14 | 0.019 / 0.18 |
| cal_yaw_-1 | 0.00098 / 0.0018 | 0.067 / 0.15 | 5.5 / 14 | 3.6 / 15 | 0.018 / 0.092 | 0.016 / 0.053 |
| cal_yaw_+2 | 0.001 / 0.0022 | 0.08 / 0.16 | 8 / 21 | 3.6 / 17 | 0.028 / 0.26 | 0.029 / 0.2 |
| cal_yaw_-2 | 0.002 / 0.003 | 0.084 / 0.21 | 7.1 / 19 | 3.7 / 22 | 0.046 / 0.19 | 0.041 / 0.14 |
| cal_pitch_+1 | 0.0003 / 0.001 | 0.24 / 0.47 | 10 / 32 | 3.2 / 13 | 0.039 / 0.17 | 0.02 / 0.12 |
| cal_pitch_-1 | 0.00021 / 0.00089 | 0.13 / 0.24 | 9.6 / 34 | 6.9 / 17 | 0.021 / 0.14 | 0.028 / 0.14 |
| cal_pitch_+2 | 0.00071 / 0.0021 | 0.17 / 0.34 | 7.5 / 17 | 3.5 / 15 | 0.045 / 0.19 | 0.03 / 0.15 |
| cal_pitch_-2 | 0.00024 / 0.0009 | 0.24 / 0.36 | 17 / 40 | 6.1 / 17 | 0.021 / 0.14 | 0.024 / 0.34 |
| cal_roll_+2 | 0.00025 / 0.00078 | 0.06 / 0.25 | 5.4 / 29 | 4 / 11 | 0.021 / 0.16 | 0.024 / 0.24 |
| foc_both_+5 | 0.00041 / 0.00084 | 0.032 / 0.091 | 9.1 / 25 | 3 / 16 | 0.022 / 0.19 | 0.027 / 0.14 |
| foc_both_-5 | 0.00045 / 0.0011 | 0.051 / 0.11 | 6.7 / 22 | 4.4 / 13 | 0.026 / 0.15 | 0.027 / 0.14 |
| foc_narrow_+5 | 0.00042 / 0.00083 | 0.03 / 0.086 | 9 / 25 | 2.8 / 16 | 0.022 / 0.21 | 0.027 / 0.17 |
| foc_narrow_-5 | 0.00044 / 0.00099 | 0.045 / 0.1 | 6.6 / 23 | 4.3 / 14 | 0.025 / 0.14 | 0.027 / 0.14 |
| foc_wide_+5 | 2.6e-05 / 9e-05 | 0.005 / 0.016 | 0.48 / 4.3 | 0.27 / 0.91 | 0.0027 / 0.035 | 0.0034 / 0.046 |
| foc_wide_-5 | 2.3e-05 / 0.0001 | 0.0048 / 0.0099 | 0.48 / 4.4 | 0.33 / 1.3 | 0.0029 / 0.044 | 0.0028 / 0.038 |
| wide_lag2 | 3e-05 / 9.5e-05 | 0.0027 / 0.011 | 0.59 / 3.5 | 0.35 / 1.8 | 0.0033 / 0.036 | 0.0035 / 0.042 |
| wide_frozen | 0.00014 / 0.00047 | 0.063 / 0.18 | 4.7 / 20 | 1.8 / 6 | 0.011 / 0.062 | 0.011 / 0.081 |
| wide_grey | 0.00013 / 0.00043 | 0.09 / 0.27 | 4.4 / 23 | 2.2 / 7.1 | 0.016 / 0.14 | 0.024 / 0.067 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 3.9e-05 / 0.00015 | 0.011 / 0.038 | 0.91 / 8.4 | 0.52 / 2.9 | 0.0041 / 0.034 | 0.0042 / 0.026 |
| narrow_lag2 | 0.0003 / 0.00086 | 0.038 / 0.14 | 4.6 / 22 | 4.2 / 23 | 0.023 / 0.19 | 0.031 / 0.17 |
| narrow_frozen | 0.012 / 0.014 | 0.095 / 0.25 | 1.1e+02 / 1.3e+02 | 50 / 72 | 0.16 / 0.34 | 0.77 / 0.88 |
| narrow_grey | 0.012 / 0.014 | 0.14 / 0.3 | 93 / 1.2e+02 | 50 / 73 | 0.28 / 0.78 | 0.82 / 0.92 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0007 / 0.002 | 0.15 / 0.25 | 9.2 / 28 | 5.4 / 23 | 0.051 / 0.26 | 0.035 / 0.23 |
| ios_narrowsrc | 0.00067 / 0.0021 | 0.15 / 0.34 | 9.1 / 26 | 4.2 / 18 | 0.056 / 0.26 | 0.029 / 0.23 |
| ios_widesrc | 0.0073 / 0.0085 | 1.2 / 1.5 | 89 / 1.3e+02 | 16 / 30 | 0.21 / 0.57 | 0.22 / 0.63 |
| ios_narrowsrc_truef | 0.0005 / 0.0017 | 0.17 / 0.4 | 6.9 / 22 | 4.1 / 13 | 0.04 / 0.2 | 0.032 / 0.18 |
| drop5_repeat | 0.0012 / 0.0039 | 0.036 / 0.11 | 6.8 / 47 | 4.8 / 24 | 0.012 / 0.15 | 0.015 / 0.11 |
| drop2_skip | 0.0086 / 0.011 | 0.55 / 0.82 | 58 / 98 | 44 / 71 | 0.18 / 0.49 | 0.17 / 0.41 |
| img_down2 | 0.00011 / 0.0004 | 0.035 / 0.086 | 2.8 / 19 | 1.7 / 5.7 | 0.0082 / 0.07 | 0.016 / 0.12 |

#### original: all other warm steps (423 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 2.5e-05 / 0.00021 | 0.014 / 0.095 | 1.2 / 16 | 0.56 / 3.9 | 0.011 / 0.16 | 0.0027 / 0.1 |
| ctl_pitch_+0.1 | 5.2e-05 / 0.00045 | 0.066 / 0.33 | 4.8 / 37 | 1.7 / 9.8 | 0.016 / 0.18 | 0.0079 / 0.24 |
| ctl_yaw_+0.1 | 9.8e-05 / 0.0004 | 0.015 / 0.075 | 2 / 12 | 1 / 6 | 0.012 / 0.16 | 0.0061 / 0.17 |
| cal_zero | 0.00028 / 0.0018 | 0.55 / 0.95 | 29 / 75 | 4.5 / 23 | 0.056 / 0.42 | 0.011 / 0.29 |
| cal_yaw_+1 | 0.0006 / 0.0026 | 0.062 / 0.28 | 2.9 / 33 | 2.1 / 9.4 | 0.036 / 0.37 | 0.0081 / 0.24 |
| cal_yaw_-1 | 0.0008 / 0.0024 | 0.093 / 0.25 | 4 / 18 | 2 / 11 | 0.032 / 0.39 | 0.008 / 0.2 |
| cal_yaw_+2 | 0.00072 / 0.0039 | 0.056 / 0.29 | 3.7 / 23 | 4.5 / 15 | 0.052 / 0.52 | 0.011 / 0.28 |
| cal_yaw_-2 | 0.0013 / 0.0035 | 0.1 / 0.41 | 5.2 / 20 | 3.3 / 16 | 0.044 / 0.41 | 0.0099 / 0.15 |
| cal_pitch_+1 | 0.00017 / 0.00096 | 0.83 / 1.4 | 40 / 89 | 6.6 / 24 | 0.053 / 0.36 | 0.01 / 0.22 |
| cal_pitch_-1 | 0.00011 / 0.00073 | 0.12 / 0.63 | 15 / 81 | 2 / 12 | 0.026 / 0.31 | 0.01 / 0.28 |
| cal_pitch_+2 | 0.00018 / 0.0024 | 0.58 / 1 | 28 / 65 | 4.3 / 20 | 0.058 / 0.37 | 0.0097 / 0.29 |
| cal_pitch_-2 | 0.00012 / 0.00077 | 0.15 / 0.74 | 11 / 86 | 2.9 / 21 | 0.026 / 0.3 | 0.011 / 0.26 |
| cal_roll_+2 | 0.00011 / 0.0007 | 0.053 / 0.3 | 4.3 / 44 | 1.6 / 9.4 | 0.026 / 0.34 | 0.0092 / 0.28 |
| foc_both_+5 | 0.0001 / 0.00099 | 0.049 / 0.19 | 11 / 39 | 1.6 / 15 | 0.022 / 0.27 | 0.011 / 0.34 |
| foc_both_-5 | 0.00012 / 0.00092 | 0.07 / 0.25 | 9 / 26 | 1.9 / 12 | 0.025 / 0.35 | 0.0098 / 0.31 |
| foc_narrow_+5 | 0.0001 / 0.00096 | 0.048 / 0.2 | 11 / 38 | 1.6 / 15 | 0.022 / 0.27 | 0.011 / 0.33 |
| foc_narrow_-5 | 0.00012 / 0.00091 | 0.071 / 0.25 | 8.9 / 26 | 1.9 / 11 | 0.025 / 0.38 | 0.0098 / 0.31 |
| foc_wide_+5 | 6.9e-06 / 8.5e-05 | 0.003 / 0.023 | 0.34 / 5.5 | 0.092 / 1.1 | 0.0028 / 0.038 | 0.00089 / 0.023 |
| foc_wide_-5 | 7.2e-06 / 0.00011 | 0.0035 / 0.022 | 0.36 / 4.6 | 0.094 / 0.73 | 0.0028 / 0.032 | 0.0011 / 0.038 |
| wide_lag2 | 9.3e-06 / 8.8e-05 | 0.003 / 0.027 | 0.41 / 5.2 | 0.11 / 0.89 | 0.0041 / 0.053 | 0.0012 / 0.035 |
| wide_frozen | 3.2e-05 / 0.00029 | 0.024 / 0.17 | 1.7 / 12 | 0.31 / 2.4 | 0.01 / 0.11 | 0.0044 / 0.14 |
| wide_grey | 7.5e-05 / 0.00052 | 0.064 / 0.31 | 2.9 / 21 | 1.2 / 7.9 | 0.025 / 0.34 | 0.0076 / 0.15 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 1.7e-05 / 0.00014 | 0.019 / 0.084 | 1.3 / 7.8 | 0.29 / 2.4 | 0.0065 / 0.092 | 0.0016 / 0.034 |
| narrow_lag2 | 0.00018 / 0.0016 | 0.034 / 0.22 | 4.8 / 36 | 2 / 10 | 0.037 / 0.42 | 0.015 / 0.52 |
| narrow_frozen | 0.0017 / 0.01 | 0.35 / 1.4 | 1.5e+02 / 2e+02 | 18 / 63 | 0.17 / 0.55 | 0.78 / 0.9 |
| narrow_grey | 0.0017 / 0.01 | 0.44 / 1.6 | 1.2e+02 / 1.9e+02 | 19 / 63 | 0.3 / 0.82 | 0.81 / 0.93 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.00021 / 0.002 | 0.29 / 0.68 | 25 / 63 | 4.3 / 17 | 0.051 / 0.43 | 0.018 / 0.51 |
| ios_narrowsrc | 0.00031 / 0.0028 | 0.52 / 0.95 | 30 / 72 | 4.1 / 22 | 0.06 / 0.41 | 0.013 / 0.36 |
| ios_widesrc | 0.0011 / 0.0068 | 1 / 2 | 1.2e+02 / 2e+02 | 9.7 / 43 | 0.13 / 0.63 | 0.18 / 0.73 |
| ios_narrowsrc_truef | 0.00029 / 0.002 | 0.6 / 1.1 | 29 / 76 | 4.7 / 23 | 0.057 / 0.42 | 0.012 / 0.34 |
| drop5_repeat | 0.0002 / 0.0029 | 0.071 / 0.24 | 11 / 71 | 2.4 / 28 | 0.013 / 0.35 | 0.0041 / 0.19 |
| drop2_skip | 0.0017 / 0.0095 | 0.64 / 1.5 | 1.3e+02 / 1.9e+02 | 22 / 64 | 0.081 / 0.74 | 0.033 / 0.37 |
| img_down2 | 5e-05 / 0.00026 | 0.036 / 0.19 | 3 / 31 | 1.3 / 10 | 0.015 / 0.19 | 0.0073 / 0.29 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | 0.00026 | 0.14 | 3.9 | 9.9 | 0.048 | 0.033 |
| ctl_noise1 | 0.00025 | 0.14 | 4.1 | 9.8 | 0.051 | 0.034 |
| cal_zero | 0.00044 | 0.5 | 5.1 | 26 | 0.063 | 0.033 |
| cal_yaw_+2 | 0.00089 | 0.14 | 6.7 | 9.9 | 0.056 | 0.041 |
| cal_yaw_-2 | 0.0013 | 0.094 | 4.1 | 8.8 | 0.052 | 0.037 |
| cal_pitch_+2 | 0.00036 | 0.52 | 5.2 | 26 | 0.061 | 0.032 |
| cal_pitch_-2 | 0.00029 | 0.17 | 5.2 | 14 | 0.053 | 0.03 |
| foc_both_+5 | 0.00031 | 0.15 | 4.4 | 16 | 0.047 | 0.034 |
| wide_lag2 | 0.00026 | 0.14 | 3.9 | 9.8 | 0.048 | 0.033 |
| wide_frozen | 0.00027 | 0.14 | 4.1 | 9.7 | 0.049 | 0.036 |
| wide_grey | 0.00027 | 0.13 | 4.1 | 9.2 | 0.047 | 0.038 |
| one_narrow | 0.00025 | 0.14 | 3.9 | 10 | 0.047 | 0.034 |
| narrow_grey | 0.0034 | 0.39 | 23 | 1.1e+02 | 0.46 | 0.84 |
| one_wide | 0.00035 | 0.3 | 5.3 | 23 | 0.062 | 0.028 |
| ios_narrowsrc | 0.00045 | 0.48 | 5 | 28 | 0.073 | 0.038 |
| drop5_repeat | 0.00051 | 0.16 | 5.1 | 16 | 0.049 | 0.036 |
| drop2_skip | 0.0027 | 0.56 | 26 | 1.2e+02 | 0.12 | 0.07 |
| img_down2 | 0.00027 | 0.13 | 4.3 | 9.3 | 0.047 | 0.035 |

### Clip tesla

unperturbed |curvature| warm: mean 0.0008 max 0.0031 1/m; steps with |curvature| > 0.01: 0 of 500 (v_ego there - m/s); v_ego warm 1.3..25.4 mean 13.1

#### tesla: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 2.6e-05 / 0.00021 | 0.013 / 0.05 | 0.96 / 9.1 | 0.14 / 1.4 | 0.0055 / 0.069 | 0.0082 / 0.079 |
| ctl_pitch_+0.1 | 8.1e-05 / 0.00085 | 0.058 / 0.16 | 2.7 / 20 | 0.51 / 6.4 | 0.017 / 0.27 | 0.037 / 0.44 |
| ctl_yaw_+0.1 | 0.00019 / 0.00092 | 0.016 / 0.055 | 1.5 / 14 | 0.27 / 2.8 | 0.013 / 0.19 | 0.033 / 0.3 |
| cal_zero | 0.003 / 0.0096 | 0.33 / 0.59 | 14 / 62 | 0.89 / 7 | 0.058 / 0.55 | 0.081 / 0.61 |
| cal_yaw_+1 | 0.0018 / 0.0055 | 0.042 / 0.17 | 2.2 / 11 | 0.66 / 3.6 | 0.028 / 0.39 | 0.037 / 0.32 |
| cal_yaw_-1 | 0.0019 / 0.0061 | 0.052 / 0.2 | 3.2 / 25 | 0.83 / 4.3 | 0.03 / 0.37 | 0.046 / 0.34 |
| cal_yaw_+2 | 0.0028 / 0.013 | 0.077 / 0.26 | 3 / 19 | 0.87 / 4.2 | 0.042 / 0.47 | 0.03 / 0.26 |
| cal_yaw_-2 | 0.0042 / 0.016 | 0.12 / 0.42 | 2.8 / 21 | 2.2 / 8.2 | 0.048 / 0.54 | 0.046 / 0.29 |
| cal_pitch_+1 | 0.00028 / 0.0011 | 0.62 / 1.2 | 19 / 78 | 2.5 / 11 | 0.055 / 0.44 | 0.054 / 0.37 |
| cal_pitch_-1 | 0.00019 / 0.002 | 0.16 / 0.36 | 4.6 / 21 | 0.59 / 6.4 | 0.035 / 0.38 | 0.044 / 0.35 |
| cal_pitch_+2 | 0.0004 / 0.0024 | 0.6 / 1.1 | 20 / 62 | 2 / 12 | 0.063 / 0.43 | 0.046 / 0.3 |
| cal_pitch_-2 | 0.00023 / 0.0016 | 0.15 / 0.48 | 6.2 / 46 | 0.94 / 9.8 | 0.043 / 0.47 | 0.061 / 0.56 |
| cal_roll_+2 | 0.00027 / 0.0015 | 0.036 / 0.14 | 2.7 / 19 | 0.56 / 6 | 0.034 / 0.48 | 0.047 / 0.41 |
| foc_both_+5 | 0.00019 / 0.0012 | 0.052 / 0.3 | 4.5 / 30 | 0.43 / 5.1 | 0.027 / 0.35 | 0.049 / 0.44 |
| foc_both_-5 | 0.00018 / 0.0011 | 0.034 / 0.19 | 5.7 / 35 | 0.49 / 6.3 | 0.029 / 0.32 | 0.054 / 0.39 |
| foc_narrow_+5 | 0.00019 / 0.0012 | 0.054 / 0.3 | 4.6 / 30 | 0.43 / 4.9 | 0.026 / 0.34 | 0.05 / 0.44 |
| foc_narrow_-5 | 0.00017 / 0.00092 | 0.034 / 0.19 | 5.6 / 35 | 0.48 / 6.2 | 0.028 / 0.32 | 0.054 / 0.39 |
| foc_wide_+5 | 2e-05 / 0.00024 | 0.0041 / 0.016 | 0.35 / 2.8 | 0.038 / 0.47 | 0.0046 / 0.08 | 0.0039 / 0.041 |
| foc_wide_-5 | 1.9e-05 / 0.0002 | 0.0034 / 0.013 | 0.3 / 3.3 | 0.042 / 0.81 | 0.0046 / 0.12 | 0.0043 / 0.042 |
| wide_lag2 | 1.6e-05 / 0.00019 | 0.004 / 0.023 | 0.37 / 6.1 | 0.049 / 0.79 | 0.004 / 0.15 | 0.0038 / 0.05 |
| wide_frozen | 9.6e-05 / 0.00082 | 0.032 / 0.12 | 3.1 / 12 | 0.24 / 1.8 | 0.02 / 0.3 | 0.017 / 0.13 |
| wide_grey | 0.00022 / 0.0013 | 0.067 / 0.21 | 2.9 / 18 | 0.57 / 4.6 | 0.026 / 0.27 | 0.051 / 0.38 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 6.8e-05 / 0.00062 | 0.027 / 0.16 | 1.4 / 7.9 | 0.13 / 1.1 | 0.014 / 0.17 | 0.0094 / 0.074 |
| narrow_lag2 | 0.00015 / 0.0012 | 0.026 / 0.11 | 3.1 / 26 | 0.52 / 6.1 | 0.025 / 0.34 | 0.052 / 0.4 |
| narrow_frozen | 0.00084 / 0.0031 | 1 / 1.7 | 76 / 2.2e+02 | 5.3 / 21 | 0.25 / 0.63 | 0.63 / 0.95 |
| narrow_grey | 0.00092 / 0.0031 | 1.1 / 1.8 | 46 / 1.5e+02 | 5.7 / 21 | 0.25 / 0.83 | 0.66 / 0.96 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0014 / 0.0055 | 0.27 / 0.87 | 12 / 67 | 1.3 / 11 | 0.053 / 0.48 | 0.068 / 0.64 |
| ios_narrowsrc | 0.003 / 0.009 | 0.27 / 0.54 | 16 / 71 | 0.96 / 7.9 | 0.069 / 0.7 | 0.08 / 0.57 |
| ios_widesrc | 0.00065 / 0.0032 | 0.52 / 1.4 | 1.5e+02 / 3.1e+02 | 2.2 / 15 | 0.11 / 0.71 | 0.27 / 0.85 |
| ios_narrowsrc_truef | 0.003 / 0.0091 | 0.3 / 0.53 | 15 / 62 | 0.86 / 6.6 | 0.065 / 0.63 | 0.081 / 0.63 |
| drop5_repeat | 0.0001 / 0.00074 | 0.045 / 0.21 | 4.6 / 46 | 0.6 / 9.2 | 0.016 / 0.5 | 0.015 / 0.28 |
| drop2_skip | 0.00065 / 0.0024 | 1 / 2.8 | 84 / 2.1e+02 | 9.6 / 56 | 0.099 / 0.65 | 0.11 / 0.69 |
| img_down2 | 9.9e-05 / 0.00086 | 0.043 / 0.19 | 2.5 / 18 | 0.37 / 5.7 | 0.02 / 0.24 | 0.04 / 0.33 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | 0.00015 | 0.043 | 0.46 | 2.7 | 0.025 | 0.048 |
| ctl_noise1 | 0.00016 | 0.035 | 0.45 | 2.5 | 0.024 | 0.047 |
| cal_zero | 0.0031 | 0.36 | 0.94 | 15 | 0.07 | 0.096 |
| cal_yaw_+2 | 0.0028 | 0.055 | 1 | 3.5 | 0.045 | 0.05 |
| cal_yaw_-2 | 0.0042 | 0.092 | 2.1 | 3.7 | 0.064 | 0.065 |
| cal_pitch_+2 | 0.0004 | 0.63 | 2.1 | 20 | 0.069 | 0.056 |
| cal_pitch_-2 | 0.00025 | 0.16 | 0.97 | 6.4 | 0.053 | 0.07 |
| foc_both_+5 | 0.0002 | 0.061 | 0.57 | 4.1 | 0.035 | 0.058 |
| wide_lag2 | 0.00016 | 0.043 | 0.47 | 2.7 | 0.025 | 0.048 |
| wide_frozen | 0.00019 | 0.058 | 0.56 | 4.6 | 0.034 | 0.054 |
| wide_grey | 0.00025 | 0.073 | 0.69 | 3.8 | 0.033 | 0.052 |
| one_narrow | 0.00015 | 0.042 | 0.48 | 3.2 | 0.026 | 0.05 |
| narrow_grey | 0.00088 | 1 | 5.8 | 47 | 0.35 | 0.63 |
| one_wide | 0.0014 | 0.26 | 1.3 | 12 | 0.057 | 0.068 |
| ios_narrowsrc | 0.003 | 0.3 | 0.96 | 17 | 0.09 | 0.092 |
| drop5_repeat | 0.0002 | 0.05 | 0.83 | 5.8 | 0.033 | 0.051 |
| drop2_skip | 0.00069 | 1.1 | 9.5 | 83 | 0.12 | 0.095 |
| img_down2 | 0.00017 | 0.045 | 0.51 | 2.9 | 0.031 | 0.057 |

### Clip gm

unperturbed |curvature| warm: mean 0.0243 max 0.1076 1/m; steps with |curvature| > 0.01: 205 of 500 (v_ego there 4.2..6.2 m/s); v_ego warm 4.2..9.9 mean 6.8

#### gm: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 0.00019 / 0.0028 | 0.0081 / 0.036 | 0.52 / 4.7 | 0.45 / 3.5 | 0.0028 / 0.09 | 0.0043 / 0.043 |
| ctl_pitch_+0.1 | 0.00043 / 0.0064 | 0.023 / 0.076 | 1.2 / 12 | 0.98 / 9.8 | 0.0083 / 0.22 | 0.011 / 0.13 |
| ctl_yaw_+0.1 | 0.00037 / 0.0038 | 0.0088 / 0.047 | 0.89 / 6.1 | 0.67 / 7.6 | 0.0067 / 0.21 | 0.0078 / 0.081 |
| cal_zero | 0.0025 / 0.018 | 0.11 / 0.47 | 7.1 / 34 | 3.7 / 30 | 0.035 / 0.54 | 0.036 / 0.27 |
| cal_yaw_+1 | 0.0016 / 0.011 | 0.055 / 0.25 | 2.1 / 19 | 1.8 / 16 | 0.022 / 0.53 | 0.021 / 0.32 |
| cal_yaw_-1 | 0.002 / 0.0091 | 0.034 / 0.21 | 2.4 / 23 | 1.7 / 13 | 0.018 / 0.32 | 0.021 / 0.19 |
| cal_yaw_+2 | 0.0019 / 0.018 | 0.08 / 0.39 | 2.7 / 29 | 3 / 26 | 0.035 / 0.61 | 0.031 / 0.29 |
| cal_yaw_-2 | 0.0035 / 0.012 | 0.056 / 0.28 | 4.5 / 27 | 2.8 / 17 | 0.026 / 0.45 | 0.049 / 0.43 |
| cal_pitch_+1 | 0.00096 / 0.012 | 0.13 / 0.41 | 3.5 / 16 | 2 / 20 | 0.024 / 0.44 | 0.075 / 0.33 |
| cal_pitch_-1 | 0.00099 / 0.014 | 0.17 / 0.47 | 3.1 / 23 | 2.1 / 18 | 0.017 / 0.27 | 0.024 / 0.21 |
| cal_pitch_+2 | 0.0012 / 0.012 | 0.2 / 0.5 | 5.1 / 20 | 2.5 / 21 | 0.03 / 0.54 | 0.11 / 0.45 |
| cal_pitch_-2 | 0.0012 / 0.014 | 0.16 / 0.47 | 5.2 / 22 | 2.5 / 21 | 0.02 / 0.3 | 0.033 / 0.22 |
| cal_roll_+2 | 0.0011 / 0.013 | 0.061 / 0.31 | 2.8 / 21 | 2.7 / 25 | 0.018 / 0.37 | 0.022 / 0.27 |
| foc_both_+5 | 0.0013 / 0.011 | 0.047 / 0.19 | 2.1 / 12 | 1.7 / 16 | 0.013 / 0.26 | 0.019 / 0.24 |
| foc_both_-5 | 0.0013 / 0.012 | 0.048 / 0.2 | 2 / 18 | 1.6 / 12 | 0.012 / 0.24 | 0.016 / 0.15 |
| foc_narrow_+5 | 0.0013 / 0.0077 | 0.046 / 0.19 | 2 / 13 | 1.8 / 16 | 0.013 / 0.27 | 0.019 / 0.22 |
| foc_narrow_-5 | 0.0013 / 0.011 | 0.051 / 0.22 | 2 / 20 | 1.7 / 15 | 0.011 / 0.27 | 0.016 / 0.14 |
| foc_wide_+5 | 0.00016 / 0.0022 | 0.0071 / 0.034 | 0.41 / 5 | 0.39 / 4.7 | 0.0025 / 0.08 | 0.0042 / 0.047 |
| foc_wide_-5 | 0.00019 / 0.0041 | 0.0078 / 0.04 | 0.45 / 5.3 | 0.43 / 6.3 | 0.0029 / 0.1 | 0.0042 / 0.049 |
| wide_lag2 | 0.00018 / 0.0049 | 0.006 / 0.056 | 0.43 / 6.7 | 0.37 / 6.8 | 0.0025 / 0.077 | 0.0042 / 0.053 |
| wide_frozen | 0.00092 / 0.01 | 0.044 / 0.26 | 1.6 / 12 | 1.6 / 16 | 0.012 / 0.24 | 0.022 / 0.24 |
| wide_grey | 0.0012 / 0.015 | 0.06 / 0.23 | 1.9 / 11 | 2.3 / 18 | 0.022 / 0.37 | 0.026 / 0.19 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 0.00088 / 0.014 | 0.038 / 0.15 | 1.8 / 12 | 1.9 / 20 | 0.0087 / 0.14 | 0.017 / 0.17 |
| narrow_lag2 | 0.0019 / 0.023 | 0.044 / 0.19 | 2.4 / 25 | 2.4 / 28 | 0.013 / 0.31 | 0.021 / 0.23 |
| narrow_frozen | 0.024 / 0.11 | 0.63 / 1.6 | 48 / 89 | 21 / 77 | 0.089 / 0.79 | 0.42 / 0.6 |
| narrow_grey | 0.024 / 0.11 | 0.6 / 1.8 | 43 / 89 | 21 / 78 | 0.23 / 0.81 | 0.063 / 0.25 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0026 / 0.0097 | 0.17 / 0.57 | 5.8 / 30 | 3.2 / 35 | 0.045 / 0.7 | 0.044 / 0.27 |
| ios_narrowsrc | 0.014 / 0.07 | 0.3 / 1.1 | 12 / 41 | 9.1 / 69 | 0.053 / 0.73 | 0.06 / 0.32 |
| ios_widesrc | 0.015 / 0.07 | 0.5 / 1.1 | 47 / 97 | 9.6 / 65 | 0.097 / 0.78 | 0.23 / 0.89 |
| ios_narrowsrc_truef | 0.0025 / 0.019 | 0.097 / 0.41 | 6.4 / 37 | 3.5 / 24 | 0.036 / 0.55 | 0.036 / 0.27 |
| drop5_repeat | 0.0023 / 0.037 | 0.044 / 0.18 | 2.6 / 27 | 1.8 / 36 | 0.0072 / 0.38 | 0.011 / 0.18 |
| drop2_skip | 0.013 / 0.075 | 1.1 / 2.6 | 20 / 79 | 13 / 81 | 0.044 / 0.74 | 0.056 / 0.28 |
| img_down2 | 0.00034 / 0.0054 | 0.016 / 0.086 | 1 / 7.6 | 0.7 / 8.7 | 0.0058 / 0.13 | 0.0081 / 0.094 |

#### gm: steps with |reference curvature| > 0.01 (205 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 0.0003 / 0.0021 | 0.01 / 0.036 | 0.67 / 3.1 | 0.61 / 3.5 | 0.00073 / 0.03 | 0.0044 / 0.033 |
| ctl_pitch_+0.1 | 0.00067 / 0.0064 | 0.018 / 0.076 | 1.4 / 12 | 1.1 / 7.9 | 0.0017 / 0.06 | 0.01 / 0.092 |
| ctl_yaw_+0.1 | 0.0005 / 0.0027 | 0.0068 / 0.044 | 1 / 6.1 | 0.73 / 4.5 | 0.0014 / 0.065 | 0.0067 / 0.042 |
| cal_zero | 0.0028 / 0.014 | 0.1 / 0.47 | 9.6 / 34 | 5.1 / 24 | 0.0062 / 0.16 | 0.032 / 0.22 |
| cal_yaw_+1 | 0.0017 / 0.011 | 0.04 / 0.25 | 2.2 / 19 | 2 / 15 | 0.0032 / 0.08 | 0.014 / 0.14 |
| cal_yaw_-1 | 0.0016 / 0.0091 | 0.03 / 0.17 | 2.2 / 23 | 2.1 / 13 | 0.0029 / 0.082 | 0.016 / 0.14 |
| cal_yaw_+2 | 0.0031 / 0.018 | 0.07 / 0.39 | 3.2 / 29 | 3.4 / 26 | 0.0065 / 0.18 | 0.023 / 0.2 |
| cal_yaw_-2 | 0.0028 / 0.01 | 0.052 / 0.26 | 3.3 / 27 | 3.5 / 17 | 0.0045 / 0.16 | 0.024 / 0.19 |
| cal_pitch_+1 | 0.0014 / 0.012 | 0.078 / 0.27 | 3.4 / 15 | 2.6 / 20 | 0.0034 / 0.11 | 0.044 / 0.2 |
| cal_pitch_-1 | 0.0015 / 0.0063 | 0.088 / 0.3 | 2.7 / 21 | 2.4 / 13 | 0.0037 / 0.12 | 0.021 / 0.1 |
| cal_pitch_+2 | 0.0017 / 0.0094 | 0.13 / 0.44 | 4.9 / 19 | 2.7 / 21 | 0.0078 / 0.3 | 0.089 / 0.45 |
| cal_pitch_-2 | 0.0018 / 0.0085 | 0.09 / 0.36 | 6.8 / 22 | 3.7 / 21 | 0.0053 / 0.19 | 0.025 / 0.13 |
| cal_roll_+2 | 0.0018 / 0.013 | 0.079 / 0.31 | 3.4 / 21 | 2.9 / 25 | 0.0038 / 0.09 | 0.022 / 0.12 |
| foc_both_+5 | 0.0024 / 0.0077 | 0.059 / 0.19 | 2.7 / 12 | 2.3 / 16 | 0.0038 / 0.099 | 0.017 / 0.1 |
| foc_both_-5 | 0.0025 / 0.012 | 0.051 / 0.2 | 2.3 / 18 | 2.2 / 12 | 0.0033 / 0.1 | 0.017 / 0.1 |
| foc_narrow_+5 | 0.0024 / 0.0077 | 0.059 / 0.19 | 2.7 / 13 | 2.4 / 16 | 0.0037 / 0.082 | 0.017 / 0.096 |
| foc_narrow_-5 | 0.0025 / 0.011 | 0.055 / 0.22 | 2.3 / 20 | 2.4 / 15 | 0.0032 / 0.12 | 0.017 / 0.11 |
| foc_wide_+5 | 0.00023 / 0.0019 | 0.0086 / 0.034 | 0.49 / 5 | 0.45 / 4.7 | 0.00061 / 0.02 | 0.0048 / 0.028 |
| foc_wide_-5 | 0.00026 / 0.0041 | 0.0092 / 0.04 | 0.58 / 5.3 | 0.52 / 6.3 | 0.00077 / 0.027 | 0.0048 / 0.049 |
| wide_lag2 | 0.00023 / 0.0033 | 0.0085 / 0.052 | 0.58 / 6.7 | 0.4 / 3.4 | 0.00073 / 0.028 | 0.0046 / 0.03 |
| wide_frozen | 0.0018 / 0.0074 | 0.064 / 0.26 | 2.2 / 11 | 2.5 / 16 | 0.0026 / 0.11 | 0.034 / 0.15 |
| wide_grey | 0.002 / 0.015 | 0.077 / 0.23 | 2.6 / 11 | 2.8 / 18 | 0.0047 / 0.15 | 0.034 / 0.15 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 0.0011 / 0.012 | 0.053 / 0.15 | 2.7 / 10 | 2.3 / 19 | 0.0025 / 0.074 | 0.021 / 0.12 |
| narrow_lag2 | 0.0035 / 0.023 | 0.059 / 0.19 | 3.3 / 25 | 3.7 / 28 | 0.0045 / 0.18 | 0.023 / 0.14 |
| narrow_frozen | 0.056 / 0.11 | 0.71 / 1.6 | 48 / 86 | 45 / 77 | 0.056 / 0.43 | 0.4 / 0.57 |
| narrow_grey | 0.056 / 0.11 | 0.56 / 1.8 | 43 / 89 | 45 / 78 | 0.21 / 0.74 | 0.051 / 0.22 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0036 / 0.0097 | 0.14 / 0.48 | 6.2 / 30 | 3.7 / 26 | 0.0065 / 0.31 | 0.039 / 0.22 |
| ios_narrowsrc | 0.029 / 0.07 | 0.33 / 1.1 | 9.8 / 38 | 15 / 57 | 0.017 / 0.44 | 0.04 / 0.21 |
| ios_widesrc | 0.033 / 0.07 | 0.66 / 1.1 | 51 / 97 | 18 / 65 | 0.072 / 0.48 | 0.36 / 0.89 |
| ios_narrowsrc_truef | 0.0028 / 0.014 | 0.1 / 0.41 | 8.8 / 37 | 5 / 23 | 0.0062 / 0.16 | 0.033 / 0.22 |
| drop5_repeat | 0.005 / 0.037 | 0.053 / 0.18 | 3.5 / 27 | 2.8 / 36 | 0.0018 / 0.11 | 0.01 / 0.16 |
| drop2_skip | 0.028 / 0.075 | 0.55 / 1.7 | 26 / 74 | 16 / 66 | 0.013 / 0.44 | 0.052 / 0.28 |
| img_down2 | 0.00048 / 0.0046 | 0.019 / 0.084 | 1.2 / 7.6 | 0.8 / 4.3 | 0.0012 / 0.049 | 0.0078 / 0.055 |

#### gm: all other warm steps (295 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 0.00011 / 0.0028 | 0.0065 / 0.032 | 0.41 / 4.7 | 0.33 / 3.1 | 0.0042 / 0.09 | 0.0041 / 0.043 |
| ctl_pitch_+0.1 | 0.00026 / 0.0051 | 0.026 / 0.062 | 1 / 7.8 | 0.87 / 9.8 | 0.013 / 0.22 | 0.012 / 0.13 |
| ctl_yaw_+0.1 | 0.00028 / 0.0038 | 0.01 / 0.047 | 0.8 / 4.7 | 0.62 / 7.6 | 0.01 / 0.21 | 0.0085 / 0.081 |
| cal_zero | 0.0023 / 0.018 | 0.11 / 0.47 | 5.3 / 29 | 2.8 / 30 | 0.056 / 0.54 | 0.04 / 0.27 |
| cal_yaw_+1 | 0.0016 / 0.0052 | 0.065 / 0.25 | 2.1 / 7.2 | 1.6 / 16 | 0.036 / 0.53 | 0.026 / 0.32 |
| cal_yaw_-1 | 0.0023 / 0.0072 | 0.037 / 0.21 | 2.5 / 8.8 | 1.4 / 12 | 0.029 / 0.32 | 0.024 / 0.19 |
| cal_yaw_+2 | 0.0011 / 0.0075 | 0.088 / 0.39 | 2.3 / 9.2 | 2.7 / 21 | 0.055 / 0.61 | 0.036 / 0.29 |
| cal_yaw_-2 | 0.004 / 0.012 | 0.06 / 0.28 | 5.3 / 16 | 2.3 / 16 | 0.04 / 0.45 | 0.066 / 0.43 |
| cal_pitch_+1 | 0.00065 / 0.0096 | 0.17 / 0.41 | 3.6 / 16 | 1.6 / 13 | 0.039 / 0.44 | 0.097 / 0.33 |
| cal_pitch_-1 | 0.00063 / 0.014 | 0.23 / 0.47 | 3.4 / 23 | 1.9 / 18 | 0.027 / 0.27 | 0.027 / 0.21 |
| cal_pitch_+2 | 0.00074 / 0.012 | 0.25 / 0.5 | 5.2 / 20 | 2.3 / 20 | 0.045 / 0.54 | 0.12 / 0.45 |
| cal_pitch_-2 | 0.00088 / 0.014 | 0.22 / 0.47 | 4.1 / 22 | 1.7 / 17 | 0.03 / 0.3 | 0.039 / 0.22 |
| cal_roll_+2 | 0.00062 / 0.0098 | 0.048 / 0.19 | 2.4 / 16 | 2.6 / 23 | 0.027 / 0.37 | 0.021 / 0.27 |
| foc_both_+5 | 0.0005 / 0.011 | 0.038 / 0.13 | 1.6 / 11 | 1.3 / 12 | 0.019 / 0.26 | 0.021 / 0.24 |
| foc_both_-5 | 0.00045 / 0.0069 | 0.047 / 0.15 | 1.8 / 14 | 1.1 / 12 | 0.018 / 0.24 | 0.015 / 0.15 |
| foc_narrow_+5 | 0.00047 / 0.0077 | 0.037 / 0.12 | 1.6 / 10 | 1.3 / 12 | 0.019 / 0.27 | 0.02 / 0.22 |
| foc_narrow_-5 | 0.00051 / 0.0077 | 0.048 / 0.19 | 1.8 / 16 | 1.2 / 14 | 0.017 / 0.27 | 0.015 / 0.14 |
| foc_wide_+5 | 0.00011 / 0.0022 | 0.0061 / 0.028 | 0.36 / 4 | 0.35 / 4.3 | 0.0037 / 0.08 | 0.0038 / 0.047 |
| foc_wide_-5 | 0.00014 / 0.0037 | 0.0068 / 0.034 | 0.35 / 3.6 | 0.37 / 5.7 | 0.0043 / 0.1 | 0.0038 / 0.037 |
| wide_lag2 | 0.00014 / 0.0049 | 0.0043 / 0.056 | 0.32 / 4 | 0.34 / 6.8 | 0.0038 / 0.077 | 0.0039 / 0.053 |
| wide_frozen | 0.00032 / 0.01 | 0.03 / 0.13 | 1.2 / 12 | 1 / 15 | 0.018 / 0.24 | 0.014 / 0.24 |
| wide_grey | 0.00071 / 0.013 | 0.047 / 0.21 | 1.3 / 9.6 | 1.9 / 18 | 0.035 / 0.37 | 0.021 / 0.19 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 0.00069 / 0.014 | 0.028 / 0.15 | 1.2 / 12 | 1.6 / 20 | 0.013 / 0.14 | 0.014 / 0.17 |
| narrow_lag2 | 0.00074 / 0.0092 | 0.034 / 0.11 | 1.8 / 10 | 1.6 / 17 | 0.02 / 0.31 | 0.02 / 0.23 |
| narrow_frozen | 0.0022 / 0.01 | 0.57 / 1.5 | 47 / 89 | 4.8 / 42 | 0.11 / 0.79 | 0.44 / 0.6 |
| narrow_grey | 0.0023 / 0.011 | 0.62 / 1.6 | 44 / 87 | 5 / 43 | 0.24 / 0.81 | 0.071 / 0.25 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0019 / 0.0092 | 0.18 / 0.57 | 5.6 / 23 | 2.9 / 35 | 0.072 / 0.7 | 0.047 / 0.27 |
| ios_narrowsrc | 0.0033 / 0.025 | 0.29 / 0.78 | 14 / 41 | 4.8 / 69 | 0.078 / 0.73 | 0.074 / 0.32 |
| ios_widesrc | 0.0022 / 0.0096 | 0.39 / 1 | 45 / 88 | 3.9 / 30 | 0.11 / 0.78 | 0.14 / 0.87 |
| ios_narrowsrc_truef | 0.0023 / 0.019 | 0.093 / 0.34 | 4.8 / 17 | 2.4 / 24 | 0.056 / 0.55 | 0.039 / 0.27 |
| drop5_repeat | 0.00038 / 0.0076 | 0.037 / 0.17 | 2 / 25 | 1.2 / 21 | 0.011 / 0.38 | 0.012 / 0.18 |
| drop2_skip | 0.0027 / 0.043 | 1.5 / 2.6 | 15 / 79 | 11 / 81 | 0.065 / 0.74 | 0.06 / 0.28 |
| img_down2 | 0.00025 / 0.0054 | 0.014 / 0.086 | 0.86 / 6.3 | 0.63 / 8.7 | 0.0089 / 0.13 | 0.0083 / 0.094 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps; this log has no modelV2.action, so curvature and acceleration are not comparable)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | n/a | n/a | 9.8 | 9.8 | 0.064 | 0.052 |
| ctl_noise1 | n/a | n/a | 9.9 | 9.7 | 0.065 | 0.05 |
| cal_zero | n/a | n/a | 10 | 12 | 0.081 | 0.051 |
| cal_yaw_+2 | n/a | n/a | 11 | 10 | 0.06 | 0.068 |
| cal_yaw_-2 | n/a | n/a | 10 | 11 | 0.082 | 0.088 |
| cal_pitch_+2 | n/a | n/a | 11 | 10 | 0.091 | 0.16 |
| cal_pitch_-2 | n/a | n/a | 9.9 | 11 | 0.067 | 0.05 |
| foc_both_+5 | n/a | n/a | 9.8 | 9.7 | 0.067 | 0.06 |
| wide_lag2 | n/a | n/a | 9.8 | 9.8 | 0.065 | 0.051 |
| wide_frozen | n/a | n/a | 10 | 10 | 0.067 | 0.062 |
| wide_grey | n/a | n/a | 11 | 9.9 | 0.08 | 0.035 |
| one_narrow | n/a | n/a | 10 | 9.4 | 0.06 | 0.049 |
| narrow_grey | n/a | n/a | 25 | 45 | 0.34 | 0.068 |
| one_wide | n/a | n/a | 10 | 12 | 0.11 | 0.045 |
| ios_narrowsrc | n/a | n/a | 16 | 15 | 0.13 | 0.062 |
| drop5_repeat | n/a | n/a | 10 | 10 | 0.067 | 0.053 |
| drop2_skip | n/a | n/a | 19 | 22 | 0.084 | 0.049 |
| img_down2 | n/a | n/a | 9.8 | 9.7 | 0.065 | 0.049 |

### Clip toyota3

unperturbed |curvature| warm: mean 0.0003 max 0.0012 1/m; steps with |curvature| > 0.01: 0 of 500 (v_ego there - m/s); v_ego warm 30.2..30.5 mean 30.4

#### toyota3: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 4e-06 / 1.9e-05 | 0.0023 / 0.0091 | 0.42 / 1.7 | 0.12 / 1.2 | 0.0051 / 0.07 | 0.0012 / 0.033 |
| ctl_pitch_+0.1 | 1.3e-05 / 6.1e-05 | 0.022 / 0.074 | 3.9 / 9.5 | 0.65 / 6.7 | 0.015 / 0.16 | 0.0051 / 0.045 |
| ctl_yaw_+0.1 | 3.6e-05 / 0.0001 | 0.0078 / 0.032 | 0.68 / 3.4 | 0.54 / 3.4 | 0.01 / 0.13 | 0.0032 / 0.029 |
| cal_zero | 0.0001 / 0.00027 | 0.069 / 0.17 | 9.5 / 26 | 1.4 / 7.2 | 0.024 / 0.21 | 0.0096 / 0.12 |
| cal_yaw_+1 | 0.00014 / 0.00044 | 0.045 / 0.12 | 3.2 / 8.7 | 1.7 / 6 | 0.035 / 0.26 | 0.014 / 0.1 |
| cal_yaw_-1 | 0.00026 / 0.00062 | 0.052 / 0.14 | 3.2 / 14 | 2.5 / 11 | 0.024 / 0.34 | 0.012 / 0.12 |
| cal_yaw_+2 | 0.00029 / 0.0008 | 0.047 / 0.16 | 3.6 / 12 | 3.8 / 13 | 0.046 / 0.35 | 0.015 / 0.11 |
| cal_yaw_-2 | 0.00048 / 0.0011 | 0.055 / 0.13 | 2.6 / 9.8 | 6.5 / 20 | 0.03 / 0.36 | 0.031 / 0.14 |
| cal_pitch_+1 | 6.1e-05 / 0.00016 | 0.71 / 1.4 | 40 / 88 | 1.6 / 11 | 0.055 / 0.48 | 0.03 / 0.3 |
| cal_pitch_-1 | 2.7e-05 / 7.3e-05 | 0.067 / 0.23 | 18 / 34 | 0.98 / 6.8 | 0.025 / 0.2 | 0.012 / 0.096 |
| cal_pitch_+2 | 6.2e-05 / 0.00018 | 0.49 / 0.87 | 24 / 48 | 1.8 / 9 | 0.035 / 0.31 | 0.019 / 0.23 |
| cal_pitch_-2 | 2e-05 / 8.9e-05 | 0.045 / 0.14 | 3.6 / 20 | 0.81 / 6.1 | 0.036 / 0.36 | 0.012 / 0.15 |
| cal_roll_+2 | 2.5e-05 / 8.3e-05 | 0.019 / 0.054 | 1.9 / 7.6 | 0.62 / 5.4 | 0.025 / 0.3 | 0.0065 / 0.083 |
| foc_both_+5 | 2.6e-05 / 0.00011 | 0.014 / 0.043 | 9.6 / 14 | 0.57 / 7.2 | 0.014 / 0.22 | 0.0037 / 0.043 |
| foc_both_-5 | 3.1e-05 / 0.00015 | 0.013 / 0.041 | 8.7 / 15 | 0.63 / 4.5 | 0.019 / 0.2 | 0.0044 / 0.047 |
| foc_narrow_+5 | 2.6e-05 / 0.00011 | 0.014 / 0.043 | 9.6 / 14 | 0.56 / 7.2 | 0.014 / 0.21 | 0.0037 / 0.046 |
| foc_narrow_-5 | 3e-05 / 0.00015 | 0.012 / 0.041 | 8.8 / 15 | 0.62 / 4.6 | 0.019 / 0.2 | 0.0044 / 0.051 |
| foc_wide_+5 | 1.6e-06 / 1.2e-05 | 0.0012 / 0.0042 | 0.15 / 0.82 | 0.038 / 0.26 | 0.0021 / 0.036 | 0.00039 / 0.004 |
| foc_wide_-5 | 1.6e-06 / 1.1e-05 | 0.0011 / 0.0041 | 0.16 / 1.2 | 0.036 / 0.29 | 0.0022 / 0.079 | 0.00043 / 0.0058 |
| wide_lag2 | 1.8e-06 / 1.2e-05 | 0.001 / 0.0043 | 0.17 / 0.93 | 0.037 / 0.45 | 0.0025 / 0.037 | 0.00045 / 0.0072 |
| wide_frozen | 8.1e-06 / 3.9e-05 | 0.012 / 0.038 | 1.6 / 5.3 | 0.18 / 0.86 | 0.019 / 0.21 | 0.0019 / 0.024 |
| wide_grey | 2.4e-05 / 6.9e-05 | 0.01 / 0.034 | 1.5 / 4.6 | 0.52 / 1.8 | 0.019 / 0.19 | 0.0022 / 0.023 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 4.4e-06 / 2.1e-05 | 0.0062 / 0.022 | 0.76 / 2.6 | 0.17 / 1.1 | 0.0073 / 0.11 | 0.00089 / 0.0089 |
| narrow_lag2 | 2.3e-05 / 0.00011 | 0.014 / 0.056 | 1.6 / 12 | 0.73 / 5.8 | 0.021 / 0.26 | 0.0052 / 0.064 |
| narrow_frozen | 0.00032 / 0.0012 | 0.11 / 0.25 | 3e+02 / 3.1e+02 | 10 / 45 | 0.52 / 0.77 | 0.012 / 0.061 |
| narrow_grey | 0.00038 / 0.0012 | 0.094 / 0.3 | 2.3e+02 / 3e+02 | 10 / 45 | 0.34 / 0.78 | 0.06 / 0.11 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.00016 / 0.00044 | 0.082 / 0.19 | 15 / 30 | 2.8 / 11 | 0.053 / 0.46 | 0.031 / 0.27 |
| ios_narrowsrc | 0.00039 / 0.0011 | 0.12 / 0.33 | 1.1e+02 / 1.3e+02 | 1.9 / 11 | 0.12 / 0.57 | 0.25 / 0.78 |
| ios_widesrc | 0.00028 / 0.00091 | 0.049 / 0.21 | 1.2e+02 / 1.5e+02 | 5.7 / 32 | 0.15 / 0.68 | 0.019 / 0.17 |
| ios_narrowsrc_truef | 0.0001 / 0.00029 | 0.069 / 0.16 | 11 / 27 | 1.4 / 7.2 | 0.021 / 0.25 | 0.0093 / 0.11 |
| drop5_repeat | 3.6e-05 / 0.00035 | 0.017 / 0.048 | 15 / 76 | 1.1 / 14 | 0.012 / 0.28 | 0.0029 / 0.066 |
| drop2_skip | 0.00024 / 0.0012 | 0.14 / 0.34 | 1.2e+02 / 1.7e+02 | 9 / 37 | 0.1 / 0.64 | 0.015 / 0.2 |
| img_down2 | 1.2e-05 / 5.5e-05 | 0.0043 / 0.017 | 1 / 4.1 | 0.42 / 3.6 | 0.0091 / 0.12 | 0.0039 / 0.06 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | 4.9e-05 | 0.12 | 1.1 | 7.5 | 0.018 | 0.017 |
| ctl_noise1 | 4.9e-05 | 0.12 | 1.1 | 7.2 | 0.018 | 0.017 |
| cal_zero | 0.0001 | 0.1 | 1.8 | 17 | 0.022 | 0.016 |
| cal_yaw_+2 | 0.00028 | 0.09 | 4.3 | 4.8 | 0.031 | 0.017 |
| cal_yaw_-2 | 0.00049 | 0.1 | 6.1 | 7.2 | 0.024 | 0.021 |
| cal_pitch_+2 | 7.7e-05 | 0.38 | 1.8 | 17 | 0.03 | 0.017 |
| cal_pitch_-2 | 5.7e-05 | 0.091 | 1.4 | 11 | 0.023 | 0.018 |
| foc_both_+5 | 5.7e-05 | 0.11 | 1.1 | 17 | 0.019 | 0.018 |
| wide_lag2 | 4.9e-05 | 0.12 | 1.1 | 7.5 | 0.018 | 0.017 |
| wide_frozen | 5.1e-05 | 0.12 | 1.1 | 9 | 0.021 | 0.017 |
| wide_grey | 5.3e-05 | 0.12 | 0.94 | 6.2 | 0.019 | 0.017 |
| one_narrow | 4.9e-05 | 0.12 | 1 | 8.1 | 0.018 | 0.017 |
| narrow_grey | 0.00039 | 0.11 | 11 | 2.2e+02 | 0.32 | 0.051 |
| one_wide | 0.00015 | 0.062 | 3.1 | 8.2 | 0.044 | 0.028 |
| ios_narrowsrc | 0.00038 | 0.16 | 2 | 1e+02 | 0.1 | 0.24 |
| drop5_repeat | 6.9e-05 | 0.12 | 1.8 | 19 | 0.021 | 0.017 |
| drop2_skip | 0.00024 | 0.086 | 8.9 | 1.3e+02 | 0.051 | 0.016 |
| img_down2 | 4.9e-05 | 0.12 | 1.2 | 8.2 | 0.019 | 0.018 |

### Clip ram

unperturbed |curvature| warm: mean 0.0119 max 0.1025 1/m; steps with |curvature| > 0.01: 114 of 500 (v_ego there 5.6..12.8 m/s); v_ego warm 5.6..24.3 mean 16.1

#### ram: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 6.6e-05 / 0.0018 | 0.015 / 0.039 | 0.86 / 7.3 | 0.42 / 6.5 | 0.0034 / 0.067 | 0.0061 / 0.051 |
| ctl_pitch_+0.1 | 0.00018 / 0.0028 | 0.033 / 0.16 | 2.7 / 13 | 1 / 18 | 0.011 / 0.15 | 0.02 / 0.29 |
| ctl_yaw_+0.1 | 0.00019 / 0.0021 | 0.018 / 0.083 | 2 / 12 | 0.85 / 12 | 0.0076 / 0.11 | 0.017 / 0.18 |
| cal_zero | 0.0013 / 0.024 | 0.24 / 0.72 | 15 / 76 | 9.6 / 92 | 0.05 / 0.71 | 0.31 / 0.82 |
| cal_yaw_+1 | 0.0011 / 0.0048 | 0.074 / 0.27 | 6.8 / 27 | 2.2 / 14 | 0.032 / 0.42 | 0.061 / 0.34 |
| cal_yaw_-1 | 0.0012 / 0.0043 | 0.18 / 0.54 | 10 / 47 | 1.9 / 19 | 0.027 / 0.34 | 0.091 / 0.46 |
| cal_yaw_+2 | 0.0014 / 0.0073 | 0.1 / 0.34 | 7 / 31 | 3.4 / 38 | 0.039 / 0.47 | 0.073 / 0.41 |
| cal_yaw_-2 | 0.002 / 0.01 | 0.14 / 0.48 | 9.2 / 40 | 3.2 / 27 | 0.042 / 0.35 | 0.086 / 0.41 |
| cal_pitch_+1 | 0.00048 / 0.007 | 0.57 / 1.3 | 22 / 68 | 4 / 23 | 0.045 / 0.38 | 0.056 / 0.33 |
| cal_pitch_-1 | 0.0004 / 0.011 | 0.093 / 0.23 | 6.8 / 30 | 2 / 20 | 0.019 / 0.28 | 0.039 / 0.33 |
| cal_pitch_+2 | 0.00064 / 0.0076 | 0.76 / 1.4 | 27 / 71 | 5.5 / 37 | 0.063 / 0.42 | 0.066 / 0.37 |
| cal_pitch_-2 | 0.00065 / 0.007 | 0.14 / 0.46 | 7.1 / 30 | 3.5 / 56 | 0.024 / 0.29 | 0.049 / 0.35 |
| cal_roll_+2 | 0.00057 / 0.0077 | 0.082 / 0.36 | 4.1 / 18 | 1.7 / 41 | 0.023 / 0.27 | 0.033 / 0.3 |
| foc_both_+5 | 0.00051 / 0.0076 | 0.17 / 0.46 | 4.6 / 21 | 1.8 / 23 | 0.018 / 0.22 | 0.033 / 0.32 |
| foc_both_-5 | 0.00052 / 0.01 | 0.16 / 0.4 | 4.2 / 38 | 1.9 / 18 | 0.019 / 0.25 | 0.036 / 0.49 |
| foc_narrow_+5 | 0.00048 / 0.0068 | 0.17 / 0.47 | 4.5 / 21 | 1.8 / 20 | 0.017 / 0.21 | 0.033 / 0.29 |
| foc_narrow_-5 | 0.0005 / 0.011 | 0.16 / 0.41 | 4.1 / 35 | 1.9 / 24 | 0.019 / 0.21 | 0.034 / 0.47 |
| foc_wide_+5 | 6.9e-05 / 0.0019 | 0.0093 / 0.13 | 0.66 / 11 | 0.39 / 8.7 | 0.0034 / 0.15 | 0.0056 / 0.086 |
| foc_wide_-5 | 7.9e-05 / 0.0022 | 0.0081 / 0.076 | 0.67 / 11 | 0.43 / 18 | 0.0034 / 0.075 | 0.0052 / 0.091 |
| wide_lag2 | 9.1e-05 / 0.003 | 0.0075 / 0.063 | 0.67 / 13 | 0.45 / 9.2 | 0.0035 / 0.11 | 0.0064 / 0.077 |
| wide_frozen | 0.00022 / 0.0039 | 0.037 / 0.22 | 2.2 / 13 | 1.1 / 15 | 0.0095 / 0.14 | 0.017 / 0.23 |
| wide_grey | 0.00043 / 0.0079 | 0.071 / 0.31 | 3.6 / 22 | 1.2 / 29 | 0.016 / 0.29 | 0.042 / 0.34 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 0.00023 / 0.0075 | 0.038 / 0.24 | 1.4 / 15 | 0.71 / 13 | 0.0074 / 0.12 | 0.013 / 0.13 |
| narrow_lag2 | 0.00076 / 0.0096 | 0.039 / 0.19 | 3.9 / 19 | 2 / 31 | 0.023 / 0.37 | 0.042 / 0.5 |
| narrow_frozen | 0.012 / 0.1 | 1.2 / 2.7 | 1.2e+02 / 2.1e+02 | 17 / 92 | 0.34 / 0.87 | 0.38 / 0.81 |
| narrow_grey | 0.012 / 0.1 | 1.2 / 2.9 | 71 / 1.6e+02 | 17 / 93 | 0.27 / 0.84 | 0.27 / 0.93 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0012 / 0.015 | 0.19 / 0.54 | 15 / 57 | 3.8 / 54 | 0.054 / 0.4 | 0.073 / 0.54 |
| ios_narrowsrc | 0.0051 / 0.047 | 0.67 / 2.3 | 47 / 1e+02 | 12 / 1.1e+02 | 0.19 / 0.86 | 0.32 / 0.79 |
| ios_widesrc | 0.0087 / 0.085 | 0.6 / 1.5 | 1.2e+02 / 2.7e+02 | 13 / 87 | 0.2 / 0.84 | 0.42 / 0.97 |
| ios_narrowsrc_truef | 0.0013 / 0.023 | 0.24 / 0.69 | 14 / 69 | 9.1 / 78 | 0.05 / 0.62 | 0.31 / 0.82 |
| drop5_repeat | 0.001 / 0.027 | 0.11 / 0.54 | 8.2 / 64 | 2.4 / 40 | 0.013 / 0.4 | 0.023 / 0.33 |
| drop2_skip | 0.0053 / 0.048 | 0.39 / 1.2 | 1.3e+02 / 2.4e+02 | 30 / 1.2e+02 | 0.11 / 0.68 | 0.091 / 0.56 |
| img_down2 | 0.00013 / 0.0031 | 0.027 / 0.098 | 2.1 / 12 | 0.79 / 9.6 | 0.0077 / 0.11 | 0.016 / 0.18 |

#### ram: steps with |reference curvature| > 0.01 (114 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 0.00024 / 0.0018 | 0.017 / 0.039 | 0.52 / 2 | 0.6 / 6.5 | 0.0014 / 0.033 | 0.0079 / 0.051 |
| ctl_pitch_+0.1 | 0.00061 / 0.0028 | 0.024 / 0.1 | 1.7 / 10 | 1.5 / 18 | 0.0039 / 0.15 | 0.016 / 0.11 |
| ctl_yaw_+0.1 | 0.00043 / 0.0021 | 0.012 / 0.057 | 0.98 / 5.5 | 1.3 / 12 | 0.0025 / 0.073 | 0.012 / 0.098 |
| cal_zero | 0.0041 / 0.024 | 0.29 / 0.61 | 7.7 / 40 | 14 / 92 | 0.032 / 0.71 | 0.078 / 0.53 |
| cal_yaw_+1 | 0.0017 / 0.0048 | 0.031 / 0.13 | 2 / 12 | 2.2 / 14 | 0.0051 / 0.092 | 0.034 / 0.21 |
| cal_yaw_-1 | 0.0015 / 0.0043 | 0.073 / 0.27 | 2.1 / 13 | 2.1 / 19 | 0.0041 / 0.1 | 0.028 / 0.3 |
| cal_yaw_+2 | 0.0028 / 0.0073 | 0.06 / 0.19 | 3.9 / 15 | 4.3 / 38 | 0.0086 / 0.14 | 0.048 / 0.26 |
| cal_yaw_-2 | 0.0027 / 0.01 | 0.15 / 0.48 | 3.6 / 13 | 3.8 / 27 | 0.0071 / 0.23 | 0.054 / 0.34 |
| cal_pitch_+1 | 0.0012 / 0.007 | 0.17 / 0.46 | 5.6 / 17 | 4.5 / 23 | 0.0065 / 0.1 | 0.042 / 0.19 |
| cal_pitch_-1 | 0.0014 / 0.011 | 0.076 / 0.19 | 3.2 / 10 | 3.3 / 20 | 0.0066 / 0.13 | 0.032 / 0.21 |
| cal_pitch_+2 | 0.0017 / 0.0076 | 0.35 / 0.91 | 11 / 29 | 9.3 / 37 | 0.01 / 0.16 | 0.081 / 0.29 |
| cal_pitch_-2 | 0.0023 / 0.007 | 0.16 / 0.4 | 8.6 / 26 | 9.3 / 56 | 0.011 / 0.13 | 0.06 / 0.3 |
| cal_roll_+2 | 0.002 / 0.0077 | 0.18 / 0.36 | 4.5 / 11 | 4 / 41 | 0.0074 / 0.11 | 0.037 / 0.21 |
| foc_both_+5 | 0.0019 / 0.0076 | 0.097 / 0.3 | 2.5 / 16 | 2.5 / 23 | 0.0071 / 0.15 | 0.032 / 0.18 |
| foc_both_-5 | 0.0019 / 0.01 | 0.1 / 0.3 | 2.3 / 11 | 2.5 / 18 | 0.0058 / 0.14 | 0.03 / 0.17 |
| foc_narrow_+5 | 0.0018 / 0.0068 | 0.084 / 0.25 | 2.2 / 13 | 2.4 / 20 | 0.0072 / 0.15 | 0.033 / 0.2 |
| foc_narrow_-5 | 0.0018 / 0.011 | 0.099 / 0.31 | 2.2 / 11 | 2.4 / 24 | 0.0058 / 0.12 | 0.028 / 0.16 |
| foc_wide_+5 | 0.00024 / 0.0019 | 0.018 / 0.1 | 0.59 / 3.5 | 0.83 / 8.3 | 0.0011 / 0.037 | 0.0078 / 0.086 |
| foc_wide_-5 | 0.00028 / 0.0022 | 0.012 / 0.074 | 0.8 / 9.8 | 1.1 / 18 | 0.0016 / 0.075 | 0.0074 / 0.091 |
| wide_lag2 | 0.00034 / 0.003 | 0.016 / 0.058 | 0.8 / 11 | 1.3 / 9.2 | 0.0016 / 0.042 | 0.011 / 0.077 |
| wide_frozen | 0.00078 / 0.0039 | 0.052 / 0.2 | 1.5 / 6 | 2 / 15 | 0.0036 / 0.071 | 0.026 / 0.14 |
| wide_grey | 0.0016 / 0.0079 | 0.12 / 0.28 | 3.3 / 13 | 3.1 / 29 | 0.0055 / 0.1 | 0.042 / 0.21 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 0.00087 / 0.0075 | 0.088 / 0.24 | 1.6 / 6.7 | 1.7 / 13 | 0.0036 / 0.089 | 0.019 / 0.11 |
| narrow_lag2 | 0.0027 / 0.0096 | 0.07 / 0.19 | 3.4 / 14 | 4.4 / 31 | 0.0095 / 0.37 | 0.041 / 0.27 |
| narrow_frozen | 0.048 / 0.1 | 1.4 / 2.7 | 61 / 1.3e+02 | 46 / 92 | 0.35 / 0.68 | 0.43 / 0.81 |
| narrow_grey | 0.048 / 0.1 | 1.4 / 2.9 | 38 / 1.3e+02 | 46 / 93 | 0.17 / 0.75 | 0.06 / 0.31 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.0037 / 0.015 | 0.13 / 0.38 | 11 / 30 | 7 / 54 | 0.014 / 0.21 | 0.046 / 0.31 |
| ios_narrowsrc | 0.017 / 0.047 | 0.99 / 2.3 | 15 / 49 | 18 / 91 | 0.019 / 0.25 | 0.095 / 0.56 |
| ios_widesrc | 0.036 / 0.085 | 0.57 / 1.5 | 37 / 92 | 31 / 87 | 0.035 / 0.41 | 0.59 / 0.93 |
| ios_narrowsrc_truef | 0.004 / 0.023 | 0.28 / 0.69 | 7.5 / 28 | 13 / 78 | 0.029 / 0.62 | 0.078 / 0.32 |
| drop5_repeat | 0.0039 / 0.027 | 0.12 / 0.54 | 3.5 / 34 | 3.2 / 28 | 0.004 / 0.22 | 0.02 / 0.22 |
| drop2_skip | 0.018 / 0.048 | 0.34 / 1.2 | 40 / 1.5e+02 | 26 / 85 | 0.017 / 0.19 | 0.06 / 0.26 |
| img_down2 | 0.00043 / 0.0031 | 0.02 / 0.076 | 1.1 / 4.7 | 1.2 / 9.6 | 0.0025 / 0.083 | 0.012 / 0.11 |

#### ram: all other warm steps (386 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 1.6e-05 / 0.00018 | 0.015 / 0.039 | 0.96 / 7.3 | 0.36 / 2.9 | 0.004 / 0.067 | 0.0056 / 0.046 |
| ctl_pitch_+0.1 | 5.4e-05 / 0.0013 | 0.036 / 0.16 | 3 / 13 | 0.85 / 6.9 | 0.013 / 0.13 | 0.021 / 0.29 |
| ctl_yaw_+0.1 | 0.00011 / 0.00067 | 0.02 / 0.083 | 2.3 / 12 | 0.72 / 4.9 | 0.0092 / 0.11 | 0.019 / 0.18 |
| cal_zero | 0.00047 / 0.0026 | 0.23 / 0.72 | 17 / 76 | 8.2 / 42 | 0.056 / 0.56 | 0.37 / 0.82 |
| cal_yaw_+1 | 0.00098 / 0.0031 | 0.086 / 0.27 | 8.2 / 27 | 2.3 / 9.5 | 0.04 / 0.42 | 0.069 / 0.34 |
| cal_yaw_-1 | 0.0011 / 0.0038 | 0.21 / 0.54 | 12 / 47 | 1.8 / 11 | 0.033 / 0.34 | 0.11 / 0.46 |
| cal_yaw_+2 | 0.001 / 0.005 | 0.12 / 0.34 | 7.9 / 31 | 3.1 / 26 | 0.048 / 0.47 | 0.081 / 0.41 |
| cal_yaw_-2 | 0.0018 / 0.0057 | 0.14 / 0.47 | 11 / 40 | 3 / 15 | 0.052 / 0.35 | 0.096 / 0.41 |
| cal_pitch_+1 | 0.00027 / 0.0022 | 0.69 / 1.3 | 27 / 68 | 3.9 / 20 | 0.056 / 0.38 | 0.059 / 0.33 |
| cal_pitch_-1 | 0.00011 / 0.001 | 0.098 / 0.23 | 7.9 / 30 | 1.6 / 12 | 0.023 / 0.28 | 0.041 / 0.33 |
| cal_pitch_+2 | 0.00031 / 0.0028 | 0.89 / 1.4 | 31 / 71 | 4.4 / 33 | 0.079 / 0.42 | 0.061 / 0.37 |
| cal_pitch_-2 | 0.00016 / 0.0014 | 0.14 / 0.46 | 6.6 / 30 | 1.8 / 19 | 0.027 / 0.29 | 0.046 / 0.35 |
| cal_roll_+2 | 0.00016 / 0.0022 | 0.052 / 0.18 | 4 / 18 | 1.1 / 15 | 0.028 / 0.27 | 0.031 / 0.3 |
| foc_both_+5 | 0.0001 / 0.0011 | 0.19 / 0.46 | 5.2 / 21 | 1.6 / 13 | 0.021 / 0.22 | 0.033 / 0.32 |
| foc_both_-5 | 0.00011 / 0.0016 | 0.18 / 0.4 | 4.8 / 38 | 1.7 / 8.8 | 0.023 / 0.25 | 0.038 / 0.49 |
| foc_narrow_+5 | 9.9e-05 / 0.00095 | 0.19 / 0.47 | 5.2 / 21 | 1.6 / 12 | 0.02 / 0.21 | 0.033 / 0.29 |
| foc_narrow_-5 | 0.00011 / 0.0017 | 0.18 / 0.41 | 4.7 / 35 | 1.7 / 8.7 | 0.023 / 0.21 | 0.036 / 0.47 |
| foc_wide_+5 | 1.9e-05 / 0.00073 | 0.0068 / 0.13 | 0.68 / 11 | 0.27 / 8.7 | 0.004 / 0.15 | 0.005 / 0.059 |
| foc_wide_-5 | 2e-05 / 0.001 | 0.0068 / 0.076 | 0.63 / 11 | 0.25 / 9.7 | 0.004 / 0.065 | 0.0045 / 0.041 |
| wide_lag2 | 1.8e-05 / 0.00094 | 0.0049 / 0.063 | 0.63 / 13 | 0.21 / 8.5 | 0.004 / 0.11 | 0.0051 / 0.055 |
| wide_frozen | 5.4e-05 / 0.0017 | 0.032 / 0.22 | 2.3 / 13 | 0.82 / 8.3 | 0.011 / 0.14 | 0.015 / 0.23 |
| wide_grey | 9e-05 / 0.0018 | 0.057 / 0.31 | 3.7 / 22 | 0.65 / 10 | 0.02 / 0.29 | 0.042 / 0.34 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 4.6e-05 / 0.0011 | 0.023 / 0.22 | 1.3 / 15 | 0.41 / 8.5 | 0.0085 / 0.12 | 0.011 / 0.13 |
| narrow_lag2 | 0.00019 / 0.0017 | 0.03 / 0.15 | 4.1 / 19 | 1.2 / 8.3 | 0.027 / 0.32 | 0.042 / 0.5 |
| narrow_frozen | 0.0013 / 0.0099 | 1.2 / 2.3 | 1.3e+02 / 2.1e+02 | 8.1 / 48 | 0.34 / 0.87 | 0.37 / 0.66 |
| narrow_grey | 0.0013 / 0.0099 | 1.1 / 2.4 | 81 / 1.6e+02 | 8.1 / 48 | 0.3 / 0.84 | 0.33 / 0.93 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.00053 / 0.0045 | 0.21 / 0.54 | 17 / 57 | 2.8 / 51 | 0.066 / 0.4 | 0.081 / 0.54 |
| ios_narrowsrc | 0.0016 / 0.019 | 0.58 / 1.8 | 56 / 1e+02 | 11 / 1.1e+02 | 0.24 / 0.86 | 0.39 / 0.79 |
| ios_widesrc | 0.00069 / 0.0056 | 0.61 / 1.5 | 1.5e+02 / 2.7e+02 | 8.1 / 69 | 0.25 / 0.84 | 0.37 / 0.97 |
| ios_narrowsrc_truef | 0.00045 / 0.0023 | 0.23 / 0.67 | 16 / 69 | 7.8 / 40 | 0.056 / 0.56 | 0.38 / 0.82 |
| drop5_repeat | 0.00014 / 0.0022 | 0.11 / 0.41 | 9.6 / 64 | 2.2 / 40 | 0.016 / 0.4 | 0.024 / 0.33 |
| drop2_skip | 0.0014 / 0.014 | 0.41 / 1.1 | 1.5e+02 / 2.4e+02 | 32 / 1.2e+02 | 0.13 / 0.68 | 0.1 / 0.56 |
| img_down2 | 4.3e-05 / 0.00099 | 0.03 / 0.098 | 2.4 / 12 | 0.66 / 9.3 | 0.0093 / 0.11 | 0.017 / 0.18 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps; this log has no modelV2.action, so curvature and acceleration are not comparable)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | n/a | n/a | 7.5 | 16 | 0.054 | 0.093 |
| ctl_noise1 | n/a | n/a | 7.7 | 17 | 0.053 | 0.092 |
| cal_zero | n/a | n/a | 11 | 22 | 0.06 | 0.34 |
| cal_yaw_+2 | n/a | n/a | 8 | 20 | 0.051 | 0.13 |
| cal_yaw_-2 | n/a | n/a | 8.8 | 13 | 0.066 | 0.13 |
| cal_pitch_+2 | n/a | n/a | 7.8 | 16 | 0.082 | 0.09 |
| cal_pitch_-2 | n/a | n/a | 6.6 | 16 | 0.06 | 0.1 |
| foc_both_+5 | n/a | n/a | 7.1 | 17 | 0.059 | 0.094 |
| wide_lag2 | n/a | n/a | 7.8 | 16 | 0.054 | 0.093 |
| wide_frozen | n/a | n/a | 7.6 | 18 | 0.051 | 0.089 |
| wide_grey | n/a | n/a | 7 | 17 | 0.057 | 0.074 |
| one_narrow | n/a | n/a | 7.6 | 16 | 0.053 | 0.095 |
| narrow_grey | n/a | n/a | 17 | 62 | 0.31 | 0.26 |
| one_wide | n/a | n/a | 6 | 12 | 0.087 | 0.086 |
| ios_narrowsrc | n/a | n/a | 11 | 37 | 0.24 | 0.35 |
| drop5_repeat | n/a | n/a | 8.2 | 18 | 0.057 | 0.096 |
| drop2_skip | n/a | n/a | 31 | 1.4e+02 | 0.12 | 0.12 |
| img_down2 | n/a | n/a | 7.7 | 17 | 0.052 | 0.09 |

### Clip hyundai2

unperturbed |curvature| warm: mean 0.0013 max 0.0067 1/m; steps with |curvature| > 0.01: 0 of 500 (v_ego there - m/s); v_ego warm 6.7..13.2 mean 11.6

#### hyundai2: change vs unperturbed run, all warm steps (mean / worst) (500 steps)

| case | curvature 1/m | desired accel m/s^2 | plan x @10s m | plan y @10s m | lane prob | lead prob |
|---|---|---|---|---|---|---|
| ctl_noise1 | 2.4e-05 / 0.00026 | 0.01 / 0.052 | 1 / 5.5 | 0.026 / 0.17 | 0.0026 / 0.086 | 0.00074 / 0.023 |
| ctl_pitch_+0.1 | 8.2e-05 / 0.00052 | 0.024 / 0.13 | 3 / 19 | 0.092 / 0.54 | 0.0093 / 0.14 | 0.003 / 0.073 |
| ctl_yaw_+0.1 | 0.00015 / 0.00052 | 0.02 / 0.079 | 2.8 / 16 | 0.12 / 0.61 | 0.0076 / 0.14 | 0.0023 / 0.045 |
| cal_zero | 0.0004 / 0.0024 | 0.2 / 0.38 | 8.4 / 32 | 0.23 / 1.5 | 0.029 / 0.29 | 0.0081 / 0.14 |
| cal_yaw_+1 | 0.00082 / 0.003 | 0.06 / 0.19 | 4.1 / 20 | 0.58 / 2.4 | 0.017 / 0.42 | 0.0051 / 0.058 |
| cal_yaw_-1 | 0.0017 / 0.004 | 0.096 / 0.21 | 5.5 / 21 | 0.84 / 2.5 | 0.02 / 0.25 | 0.0058 / 0.29 |
| cal_yaw_+2 | 0.00093 / 0.0045 | 0.054 / 0.14 | 3.9 / 18 | 0.93 / 3.7 | 0.022 / 0.35 | 0.0067 / 0.1 |
| cal_yaw_-2 | 0.0029 / 0.0066 | 0.13 / 0.37 | 8.3 / 31 | 1.3 / 4.3 | 0.042 / 0.47 | 0.014 / 0.25 |
| cal_pitch_+1 | 0.00049 / 0.0022 | 0.28 / 0.51 | 10 / 32 | 0.26 / 1.8 | 0.032 / 0.31 | 0.0094 / 0.23 |
| cal_pitch_-1 | 0.00024 / 0.0013 | 0.11 / 0.36 | 7.7 / 29 | 0.17 / 1.1 | 0.027 / 0.42 | 0.0073 / 0.16 |
| cal_pitch_+2 | 0.00039 / 0.0021 | 0.28 / 0.64 | 7.4 / 38 | 0.28 / 1.6 | 0.027 / 0.38 | 0.0078 / 0.28 |
| cal_pitch_-2 | 0.00024 / 0.0015 | 0.12 / 0.52 | 8.9 / 42 | 0.21 / 1.6 | 0.038 / 0.4 | 0.0084 / 0.13 |
| cal_roll_+2 | 0.00013 / 0.00088 | 0.029 / 0.17 | 3 / 18 | 0.11 / 1.1 | 0.017 / 0.24 | 0.0034 / 0.063 |
| foc_both_+5 | 0.00011 / 0.00077 | 0.032 / 0.11 | 5.9 / 33 | 0.12 / 1 | 0.012 / 0.17 | 0.0034 / 0.096 |
| foc_both_-5 | 0.00014 / 0.0011 | 0.052 / 0.14 | 4.3 / 21 | 0.12 / 0.97 | 0.013 / 0.17 | 0.0039 / 0.11 |
| foc_narrow_+5 | 0.00011 / 0.00085 | 0.03 / 0.13 | 5.4 / 26 | 0.11 / 1 | 0.012 / 0.16 | 0.0034 / 0.071 |
| foc_narrow_-5 | 0.00013 / 0.0011 | 0.051 / 0.14 | 4.3 / 21 | 0.12 / 1 | 0.013 / 0.17 | 0.0038 / 0.11 |
| foc_wide_+5 | 2.1e-05 / 0.00015 | 0.0096 / 0.054 | 1.2 / 16 | 0.026 / 0.51 | 0.0027 / 0.05 | 0.00067 / 0.029 |
| foc_wide_-5 | 2.1e-05 / 0.00013 | 0.006 / 0.044 | 0.98 / 8.2 | 0.022 / 0.21 | 0.003 / 0.072 | 0.00068 / 0.014 |
| wide_lag2 | 2.4e-05 / 0.00026 | 0.0067 / 0.051 | 1.3 / 16 | 0.027 / 0.55 | 0.0034 / 0.055 | 0.00075 / 0.036 |
| wide_frozen | 8.3e-05 / 0.00099 | 0.068 / 0.17 | 5.8 / 24 | 0.1 / 0.74 | 0.011 / 0.18 | 0.0027 / 0.14 |
| wide_grey | 0.00014 / 0.00072 | 0.056 / 0.25 | 5.6 / 19 | 0.13 / 0.88 | 0.015 / 0.3 | 0.0058 / 0.081 |
| one_narrow (wide input from narrow-camera pixels; both inputs narrow cam) | 5.1e-05 / 0.00038 | 0.032 / 0.18 | 2.2 / 21 | 0.054 / 0.52 | 0.0057 / 0.065 | 0.0017 / 0.048 |
| narrow_lag2 | 0.0003 / 0.0021 | 0.034 / 0.2 | 6.1 / 32 | 0.26 / 2.1 | 0.022 / 0.41 | 0.0085 / 0.2 |
| narrow_frozen | 0.0013 / 0.0067 | 0.53 / 1.5 | 81 / 1.2e+02 | 1.2 / 5.3 | 0.18 / 0.58 | 0.58 / 0.74 |
| narrow_grey | 0.0016 / 0.0072 | 0.8 / 1.8 | 30 / 77 | 2.2 / 6.7 | 0.16 / 0.49 | 0.9 / 0.94 |
| one_wide (narrow input from wide-camera pixels; both inputs wide cam) | 0.00085 / 0.0037 | 0.28 / 0.64 | 10 / 40 | 1.1 / 7.9 | 0.068 / 0.56 | 0.078 / 0.77 |
| ios_narrowsrc | 0.00067 / 0.003 | 0.13 / 0.39 | 34 / 62 | 0.41 / 2.3 | 0.079 / 0.83 | 0.0077 / 0.16 |
| ios_widesrc | 0.00091 / 0.0052 | 0.55 / 1.3 | 74 / 1.5e+02 | 1 / 3.7 | 0.16 / 0.63 | 0.27 / 0.88 |
| ios_narrowsrc_truef | 0.00041 / 0.0024 | 0.21 / 0.4 | 8.7 / 31 | 0.22 / 1.5 | 0.029 / 0.26 | 0.0097 / 0.19 |
| drop5_repeat | 0.00015 / 0.0013 | 0.064 / 0.27 | 7.6 / 49 | 0.15 / 2.1 | 0.0082 / 0.5 | 0.0033 / 0.15 |
| drop2_skip | 0.00056 / 0.003 | 0.7 / 1.3 | 83 / 1.5e+02 | 1 / 7.4 | 0.051 / 0.8 | 0.027 / 0.27 |
| img_down2 | 5.9e-05 / 0.00037 | 0.028 / 0.13 | 2.4 / 22 | 0.065 / 0.73 | 0.007 / 0.15 | 0.0019 / 0.059 |

#### Distance to the device's own modelV2 (mean abs difference, warm steps; this log has no modelV2.action, so curvature and acceleration are not comparable)

| case | curvature | accel | plan y @10s | plan x @10s | inner lane probs | lead prob |
|---|---|---|---|---|---|---|
| unperturbed | n/a | n/a | 0.43 | 18 | 0.068 | 0.018 |
| ctl_noise1 | n/a | n/a | 0.43 | 18 | 0.068 | 0.019 |
| cal_zero | n/a | n/a | 0.45 | 24 | 0.089 | 0.025 |
| cal_yaw_+2 | n/a | n/a | 1.2 | 20 | 0.074 | 0.02 |
| cal_yaw_-2 | n/a | n/a | 1.2 | 23 | 0.077 | 0.031 |
| cal_pitch_+2 | n/a | n/a | 0.49 | 23 | 0.086 | 0.025 |
| cal_pitch_-2 | n/a | n/a | 0.45 | 19 | 0.068 | 0.019 |
| foc_both_+5 | n/a | n/a | 0.43 | 16 | 0.067 | 0.019 |
| wide_lag2 | n/a | n/a | 0.43 | 18 | 0.068 | 0.018 |
| wide_frozen | n/a | n/a | 0.44 | 15 | 0.069 | 0.02 |
| wide_grey | n/a | n/a | 0.45 | 21 | 0.071 | 0.024 |
| one_narrow | n/a | n/a | 0.43 | 17 | 0.068 | 0.019 |
| narrow_grey | n/a | n/a | 2.4 | 25 | 0.18 | 0.92 |
| one_wide | n/a | n/a | 1.2 | 18 | 0.085 | 0.095 |
| ios_narrowsrc | n/a | n/a | 0.61 | 48 | 0.16 | 0.019 |
| drop5_repeat | n/a | n/a | 0.48 | 23 | 0.07 | 0.02 |
| drop2_skip | n/a | n/a | 1.2 | 69 | 0.1 | 0.043 |
| img_down2 | n/a | n/a | 0.43 | 18 | 0.069 | 0.019 |

## Not checked (additions for the "More clips" study)
- Closed-loop effects, thresholds for what is acceptable, and the whole original "not checked" list above still apply.
- Only the reduced set (33 cases) was run on the other clips: no focal errors of 2 or 10 percent, no roll -2 or +-5 degrees, no pitch +-0.5 or +-5 degrees, no wide focal error beyond 5 percent, no `lag_wide1`, no `drop2_repeat` / `drop5_skip`, no `img_down4`, gain or blur cases. The first study's tables for the original clip are unchanged.
- Six clips, 30 s each, one instance of each condition: two with real turns (`gm`: 4 to 6 m/s, `ram`: 6 to 13 m/s, both daylight, both tici with ox03c10), the original's gentle curve, no clip with a fast turn or a highway curve at speed. The turn threshold of 0.01 1/m is my choice and its effect on the split was not examined. Turn steps are adjacent in time, so the 205 and 114 steps are far fewer independent samples than steps.
- The floor is one random draw (seed 0) of +-1 grey level on the model inputs, per clip, plus two 0.1 degree rotations; its own run-to-run spread was not measured, and a ratio to it near 1 to 3 is not distinguishable from 1 (the 0.1 degree controls are 1.3 to 9 times `ctl_noise1` in curvature).
- Why removing the wide input matters on five clips and almost not on the original clip, and why `one_wide` is as large as a calibration error: not established. In particular I did not test whether the measured camera-to-camera misalignment (check 3), the 1.6x upsampling of the narrow view, or the wide lens's non-pinhole geometry is responsible, for example by compensating the measured shift or by degrading the narrow camera to the same resolution. The shift measurement is a brute-force integer search with correlation 0.46 to 0.92 on three frames per clip; its sign was not interpreted, and parallax and the wide camera's focal-length uncertainty were not separated from a true difference in camera pointing.
- The wide-camera-based cases use the pinhole intrinsics of `DEVICE_CAMERAS` (f 567 for the ar0231/ox03c10 wide camera of the tici/tizi, f 425.25 for the mici); the wide cameras are not ideal pinholes and the real distortion was not modelled.
- Which of these cases an iPhone would reproduce (section 4 above) is inference from the fields of view only. No iPhone footage; nothing about the iPhone's lens distortion, rolling shutter, portrait / landscape handling, exposure or frame dropping.
- The `drop2_skip` case is the same simplification as before (outputs held, state not advanced; staleness included in the change).
- Distance to the device's `modelV2` for curvature and acceleration is not available for `gm`, `ram`, `hyundai2` (no `modelV2.action` in those logs).
- Decoded frames were produced with the installed `ffmpeg`; the check that the warp of the re-decoded frames equals the stored reference frames passed on every clip (max difference 0), so the decoder output matches the one used for the references.

## Files and how to run (additions)
`pe_common.py` now takes the frame size and focal lengths from the clip's log (same values as before for the original clip), finds decoded frames in `$RF_SCRATCH` (as before) or `$RF_SCRATCH/phone_errors/frames`, and writes to `$PE_OUT` if set (default `$RF_SCRATCH/phone_errors`). `run_all.py more [case-name-prefix ...]` runs the 33-case set (results `more/<case>.json` and per-step `more/<case>.npz`, plus `more/_base.npz`); `more_report.py NAME=DIR ... [--short-dev] [--headline] [--abs]` prints the tables of this section; `check_images.py` additionally writes `more_views_a.png`, `more_views_b.png` and prints the outside-the-source fractions and the camera-to-camera shifts; `run_clip_pe.sh <clip_dir> <python>` does one clip end to end (decode 600 frames per camera to `phone_errors/frames`, reference check, images, `oob`, the `more` group in two processes, then deletes the decoded frames; peak extra disk about 2.1 GB per camera at 1928x1208). One clip took 10 to 25 minutes on 4 cores shared with other jobs.

