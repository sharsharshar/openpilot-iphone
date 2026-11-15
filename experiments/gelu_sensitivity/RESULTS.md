# GELU sensitivity of the small driving model

All 20 `Gelu` nodes in `driving_supercombo.onnx` use the tanh formula. `gelu_variants.py` swaps all of them
for another implementation in the float32 copy of the model, replays the 40-step reference with state
carried (onnxruntime, CPU), and compares the 2576 outputs with the reference.

Command: `python -I gelu_variants.py ../../models/driving_supercombo_fp32.onnx ../../reference/driving_reference_fp32.npz`
(the float32 copy is made by `scripts/make_fp32.py`).

| GELU implementation | max error of the function itself | worst output diff | worst per-step mean diff | overall mean diff |
|---|---|---|---|---|
| tanh formula (as shipped) | 0 | 0 | 0 | 0 |
| exact erf formula | 0.00047 | 3.498 | 0.0492 | 0.02235 |
| x * sigmoid(1.702 x) | 0.02066 | 132.152 | 2.2474 | 1.27011 |
| piecewise linear, 8 segments on [-6, 6] | 0.12447 | 294.453 | 4.6469 | 3.61326 |
| piecewise linear, 16 segments | 0.04778 | 189.540 | 2.5848 | 1.58891 |
| piecewise linear, 32 segments | 0.01346 | 135.559 | 2.0200 | 0.82674 |
| piecewise linear, 64 segments | 0.00347 | 45.073 | 0.6699 | 0.25558 |

For scale: float16 rounding alone gives worst 1.71, mean 0.010; Core ML on macOS 15 gives worst 2.39, mean 0.017;
Core ML on the iOS 26.2 simulator gives worst 5.9, mean 0.057.

Caveats: the reference frames are synthetic, not road footage. The piecewise-linear knots are uniform, not optimised.

## Is Core ML's built-in GELU the cause of the iOS 26 offset? No.

Branch `exp/gelu-explicit` carries a Core ML package converted with `GELU_EXPLICIT=1` (each GELU written out as
mul/add/tanh; the package has 0 `gelu` ops and 20 `tanh` ops). The app's normal CI ran against it (run 43692148001).

| Where | Built-in GELU (main): worst / mean | GELU written out: worst / mean | Self-test, written out |
|---|---|---|---|
| macOS 15, Python | 2.39 / 0.0174 | 2.98 / 0.0163 | n/a |
| iOS 18.6 simulator | 2.39 / 0.0175 | 2.98 / 0.0164 | FAIL (one step's mean was 0.056, limit 0.05) |
| iOS 26.2 simulator | 5.90 / 0.0569 | 6.41 / 0.0549 | FAIL |

Writing GELU out changes nothing material on iOS 26, so the offset there does not come from Core ML's GELU.
On iOS 18 the written-out form is about as accurate overall but trips the per-step limit once, so `main` keeps
the built-in GELU.
