# Weight compression: accuracy cost on the small driving model

`quantize_study.py` replaces every Conv/Gemm/MatMul weight tensor (135 tensors, 29.9M weights) in the float32
ONNX model with its quantise-then-dequantise version, replays the 40-step reference with state carried
(onnxruntime, CPU), and compares the 2576 outputs with the reference. No fine-tuning.

| Weight format | Bits | Worst output diff | Worst per-step mean | Overall mean |
|---|---|---|---|---|
| float16 (what ships today) | 16 | 0.000 | 0.0000 | 0.00000 |
| 8-bit linear, per output channel | 8 | 75.860 | 1.1822 | 0.69808 |
| 8-bit linear, per tensor | 8 | 263.918 | 3.9920 | 3.13864 |
| 8-bit lookup table, 256 values per tensor | 8 | 138.923 | 2.9935 | 1.42448 |
| 6-bit lookup table, 64 values | 6 | 110.477 | 1.9719 | 1.32202 |
| 4-bit lookup table, 16 values | 4 | 175.943 | 2.4572 | 2.06762 |

The app's self-test limit is a per-step mean under 0.05 and a worst difference under 3.0. Every compressed
variant misses it by more than an order of magnitude.

Caveats: the reference frames are synthetic patterns, not road footage, and the model may be less sensitive on
real images. The lookup tables come from a simple 1-D k-means (12 iterations), not a tuned method, which is why
the 8-bit table is not better than the 6-bit one. No calibration data or fine-tuning was used.
