# ANE-oriented variants of the lean driving model

Eleven ONNX graphs derived from the lean float32 model (`models/driving_supercombo_lean_fp32.onnx`, the model the app ships as
`coreml/driving_supercombo.mlpackage`), each converted to a Core ML package with the project's existing route, so that someone
with an iPhone or a Mac can time them against each other. **Nothing here measures speed, and nothing ran on a Neural Engine or in
Core ML**: what was checked is (a) the ONNX graphs against both references with onnxruntime in float32, (b) the structure of the
resulting MIL programs, (c) ANEForge's static importer and preflight. No variant is claimed to be faster.

Reproduce everything from `models/driving_supercombo_fp32.onnx` with one command (about 45 minutes on 4 cores):

    PY=<python with numpy onnx onnxruntime torch onnx2torch coremltools> REAL_REF=<real_reference_fp32.npz> ANEFORGE=<ANEForge clone> \
      sh experiments/ane_variants/run_all.sh

(`REAL_REF` and `ANEFORGE` are optional; without them the 600-step real check, the precision controls and the ANEForge step are skipped.)
Outputs go to `models/variants/` (git-ignored): `*.onnx`, `*.mlpackage` (plus `*_io16.mlpackage`), `verify.json`, `coreml.json`,
`aneforge.json`, `controls.json`, `fp16_exactness.json`, `summary.md`. Nothing outside `experiments/ane_variants/` and `models/variants/`
was touched; `coreml/driving_supercombo.mlpackage`, the app, CI, `fixtures/`, `core/` and `scripts/` are unchanged (the scripts
there are imported or re-run, not edited). Nothing was committed.

## Result table

All numbers below were produced by `run_all.sh` in this session (`models/variants/summary.md` is the generated version).

- "max diff" is the largest |outputs - reference outputs| over all steps and all 2576 outputs, onnxruntime float32 CPU, recurrent state carried
  (by the model, or by the script when the variant moves the queues to the app). Synthetic reference: 40 steps with a desire pulse.
  Real reference: 600 steps of road footage. Output values reach 260, so 1e-3 is about 4e-6 relative.
- "MIL ops" are the operations of the converted Core ML program, counted from the package's protobuf: total including `const`, and without `const`.
  The lean package as shipped and the same ONNX converted again here have identical counts (1029 / 326), so the counts are comparable.
- ANEForge columns: native / rejected for A13, A14, A15, A16 (the only families it offers; its docs fold every A17/A18 chip into A16 and it
  knows no A19). All variants that import report 0 rejected, 0 decomposed, 0 oversize for all four; the native count was identical for the four families.

| variant | what it is | max diff synthetic | max diff real | MIL ops total / without const | vs lean (without const) | ANEForge imports whole model | ANEForge native / rejected (A13=A14=A15=A16) | drop-in for the app |
|---|---|---|---|---|---|---|---|---|
| base | lean model as shipped | 0 | 0 | 1029 / 326 | 0 | no: GatherND, Where, Concat with a constant | - | yes |
| A | the three non-importable nodes replaced | 0 | 0 | 1027 / 326 | 0 | **yes** (all 3 outputs) | 554 / 0 | yes |
| B | input normalisation folded into the stem conv | 3.97e-04 | 1.16e-03 | 1029 / 325 | -1 | no (same 3 nodes) | - | yes |
| C | every Gemm and the transformer as 1x1 convs, channel-first | 1.37e-04 | 1.37e-04 | 1440 / 323 | -3 | no (GatherND, Concat) | - | yes |
| D | queues kept by the app | 0 | 0 | 985 / 311 | -15 | no (Where, Concat) | - | **no** (inputs and outputs change) |
| E | other exact simplifications (no weight folds) | 5.34e-05 | 7.63e-05 | 996 / 315 | -11 | no (Concat) | - | yes |
| E_fold | E plus two weight folds | 6.10e-05 | 1.83e-04 | 992 / 313 | -13 | no (Concat) | - | yes |
| ALL_DROPIN_noB | A + C + E | 1.60e-04 | 1.53e-04 | 1415 / 314 | -12 | **yes** | 547 / 0 | yes |
| ALL_DROPIN | A + B + C + E | 3.74e-04 | 1.16e-03 | 1415 / 313 | -13 | **yes** | 551 / 0 | yes |
| ALL_noB | A + C + D + E | 1.60e-04 | 1.53e-04 | 1378 / 300 | -26 | **yes** | 531 / 0 | **no** (as D) |
| ALL | A + B + C + D + E | 3.74e-04 | 1.16e-03 | 1378 / 299 | -27 | **yes** | 535 / 0 | **no** (as D) |

Float16 packages with float16 inputs (`*_io16.mlpackage`, made for base, ALL_DROPIN_noB and ALL_noB): the 5 (or 4) `cast` ops that the converter
puts at the float32 inputs disappear (326 -> 321 without const for base; 314 -> 309; 300 -> 296). That changes the interface: the app would have
to hand over Float16 multi-arrays. Core ML's outputs are float16 in every package, the shipped one included.

How to read the diffs: the lean model gives 0 against the reference (bit-identical). A, D and the pure data-movement parts give exactly 0.
Everything else sums in a different order or folds a constant, which changes float32 rounding. The yardstick for that is a control: the lean model with
every stem-conv weight multiplied by (1 + u * 2^-24), u uniform in [-1, 1] (a different but equally valid rounding), drifts from the reference by
2.98e-04 / 4.88e-04 (synthetic / real); the same on all weights: 5.15e-04 / 6.87e-04. The onnx2torch module that coremltools converts, run in
float32 on 3 steps, differs from the reference by 1.8e-04 for the unchanged lean model, and by 1.0e-04 to 2.4e-04 for the variants. So every diff
in the table is of the size that re-rounding alone produces; the B-containing variants are at the top of that range (1.16e-3 at the worst step of 600, on an output of value 124;
the control's worst was 6.9e-4, one random draw each). It is above the roughly 1e-5 expected in the brief, even at 40 steps, but the control (every stem weight re-rounded by one float32 ulp or less) already gives 3.0e-4 at 40 steps, so 1e-5 was too optimistic
for this network (60+ layers, outputs up to 260, recurrent state). I did not accept the numbers as noise without that control, and I did not find a bug behind them (the variants that only move data are exactly 0, and the stem fold was checked in isolation, below).

## The float16 caveat (important, found late)

The shipped model's weights are float16 numbers stored as float32, and the Core ML package stores float16 weights, so any rewrite that
keeps every weight bit-for-bit loses nothing in conversion. A rewrite that **folds a constant into weights** produces float32 numbers that are not float16
numbers; float16 storage rounds them. `fp16_exactness.py` counts this per variant (of 30.0 million weight elements):

| variant | weights not exactly float16 |
|---|---|
| base, A, C, D, E, ALL_DROPIN_noB, ALL_noB | 0 |
| B, ALL_DROPIN, ALL | 13,845 (the stem conv: w/63.75 and the new bias) |
| E_fold | 3,124,219 (10.4%: 12 layer-scale folds and 2 output-scale folds) |

`precision_controls.py` rounds each variant's weights to float16 (activations stay float32) and runs the 40 and 600 reference steps:

| variant (weights rounded to float16) | max diff synthetic | mean diff synthetic | max diff real | mean diff real |
|---|---|---|---|---|
| ALL_DROPIN_noB | 1.60e-04 | 1.1e-06 | 1.53e-04 | 7.0e-07 (unchanged from float32: nothing to round) |
| B | 0.511 | 3.2e-03 | 0.694 | 7.9e-04 |
| ALL_DROPIN (= ALL) | 0.511 | 3.2e-03 | 0.694 | 7.9e-04 |
| E_fold | 0.329 | 3.1e-03 | 0.208 | 6.1e-04 |

So folding the normalisation into the first conv (B) is exact in float32 but **not free in a float16 package**: it adds float16 rounding of 13.8k stem weights,
worth a worst-case 0.5 to 0.7 and a mean of 8e-4 to 3e-3 on outputs that reach 260. For scale (README and `experiments/gelu_sensitivity`): float16 alone
already gives worst 1.7 / mean 0.010 in onnxruntime and worst 2.4 / mean 0.017 in Core ML on macOS 15 (synthetic frames). So B adds less than the float16
noise that is there anyway, but it is a real change, which is why the `_noB` variants exist (they are the ones whose float16 package is, weight for weight,
the same network as the shipped one), and why the layer-scale and output-scale folds are in a separate `E_fold` instead of in E and ALL.
One more fold was tried and removed for the same reason: folding the attention score scale 0.125 into the query weights is exact in float32 (power of two)
but pushes 1091 weights into float16's subnormal range, so the scale stays a `Mul` on the scores.
This is an emulation of weight storage only. It does not model float16 activations, and Core ML's float16 behaviour on a device is not something I could run.

## What each variant does

Everything is an ONNX rewrite of the lean graph (`rewrites.py`, driven by `make_variants.py`; each rewrite asserts the structure it expects).

**A. Replacements for what ANEForge's importer cannot take.** The lean graph has 3 such nodes (of the 5 in the earlier analysis of the full model, the 2 stepped
Slices belonged to the image-queue bookkeeping that `make_lean.py` removed; the remaining stride-4 Slice on the 96-row feature queue imports fine because 96 is divisible by 4).
(1) `GatherND` with constant indices -9..-1 on 25 rows -> `Slice[16:25]`: pure data movement, bit-identical. (2) `Where(causal mask, -inf, scores)` -> `Add(scores, bias)`,
bias -60000 where masked, 0 elsewhere; exact because x + 0 = x and exp(-60000 - max) is exactly 0 in float32 and float16, as long as no row is fully masked (asserted; the
diagonal is never masked). -60000 instead of -inf so float16 hardware never sees an infinity. (3) `Concat` with a constant all-zero `[1,2]` operand -> the preceding `Gemm`
(the desire-state head) gets two extra output rows with weight 0 and bias 0, which output 0*x + 0 = 0. Verified 0.0 on both references, and ANEForge imports all three outputs.

**B. Normalisation folded into the stem conv.** `Cast -> Sub(127.5) -> Div(63.75) -> Conv(3x3, stride 2, pad 1)` becomes `Pad(127.5) -> Conv(pad 0)` on the raw
0..255 input with weights w/63.75 and bias b - sum(w * 127.5/63.75). A plain fold with zero padding is wrong at the image border (the original pads the *normalised* image with 0,
which is raw 127.5): measured on a real frame (`check_stem_fold.py`, against a float64 evaluation of the original) the naive fold is off by 57.8 at the border, the
original float32 sequence by 5.5e-5, the shipped fold by 1.3e-4 (max |output| 94). The input stays raw 0..255 float32, so it is drop-in. In the MIL program the
`sub` is gone and a `pad` is added: 325 vs 326 ops, not a reduction in any meaningful sense. The `cast` ops are not the model's: they are the converter's float32-input casts.

**C. Fully connected layers as 1x1 convolutions on channel-first tensors.** All 82 `Gemm` become `Conv` 1x1 on `[1,C,1,1]`; the vector path between them runs in 4D, so the
`Reshape` before the first Gemm and the `Unsqueeze`s disappear, and one `Reshape` returns `outputs` to `[1,2576]`. The temporal transformer (9 tokens x 512) is rebuilt from the same weights
as Apple's (B, C, 1, S) form: tokens `[1,512,1,9]`, every linear a 1x1 conv, LayerNorm written out over channels (mean, subtract, square, mean, add eps, rsqrt, multiply, then
x*gamma+beta; coremltools fuses the `rsqrt` form into a single `layer_norm` op, the `sqrt`+`div` form it does not, so `convert_variants.py` fuses `Sqrt`+`Reciprocal` into `torch.rsqrt` at trace time only),
the 8 heads a reshape of the channel axis, a masked softmax. MIL: 144 `conv` and 0 `linear`; `layer_norm` 2. Non-const ops 323, 3 fewer than lean; reshape 12, transpose 4, slice 6 (lean: 7, 6, 9 + 3 squeeze + 3 expand_dims). Drop-in.

**D. Queues kept by the app.** What the model does with them (read from the graph, and the variant is bit-identical on both references, which confirms it):
- `state_feat_q [96,1,512]`: it reads rows 0,4,8,...,92 (stride-4 slice, 24 rows), keeps the last 8 of those (rows 64,68,...,92), appends the new feature vector
  (the 512 values that are also `outputs[:, 1064:1576]`, the last piece of the vision-output concat) and uses these 9 tokens. It writes `concat(q[1:], new)` back. Rows 0..63 are never read.
- `state_desire_q [100,1,8]`: it writes `concat(q[1:], desire)`, views that as `[25,4,1,8]` and takes the max over each group of 4, giving 25 x 8 values.
So the model needs only 8 feature rows and the already-shifted desire queue as inputs, and returns only `outputs`.
D's interface (names/shapes, float32): `img_stack [1,24,128,256]` as before; `desire_q [1,25,4,8]` = the app's `[100,1,8]` ring *after* it appended this step's desire (same bytes, new shape);
`traffic_convention [1,2]`, `action_t [1,2]` as before (`action_t` is unused by the model in every variant); `feat_hist [1,8,512]` = rows 64,68,...,92 of the app's 96-row ring, oldest first.
Output: `outputs [1,2576]` only. After predicting, the app drops row 0 of the feature ring and appends `outputs[1064:1576]`. `desire` is no longer an input.
ALL and ALL_noB use channel-first layouts instead: `desire_q [1,200,4,1]` with element (g,c,k) at (g*8+c)*4+k = ring[4g+k][c] (one `ReduceMax` over H straight into the first 1x1 conv; the app writes a
transposed copy of 800 floats), and `feat_hist [1,512,1,8]` with element (c,j) at c*8+j = ring[64+4j][c] (the app writes a transposed copy of 4096 floats). `verify_variants.py` is the reference implementation of
this app side. A cheaper option not built: the history rows only enter through the per-token `feats_encode` MLP, so the app could cache the *encoded* rows and the model encode one row instead of nine.
It was left out because it changes what the carried state means.

**E. Other exact simplifications** (all bit-identical except where stated): remove the 6 float32->float32 `Cast`s and 3 `Identity`s; remove `Clip(min=0)` on a vector norm (a norm is >= 0); let the norm `Div`
broadcast instead of an `Expand`; `hidden` is exactly the last piece of the vision-output `Concat`, so the `Slice` goes, and the two nested `Concat`s that build `outputs` become one; the next-state feature row reuses `hidden`
(instead of slicing it out of `outputs` again); the 7-op detour that builds the 9 tokens from the queue (stride slice, transpose, cast, concat, transpose, GatherND, transpose) becomes `Slice(start 64, step 4)`, `Reshape`, `Concat`;
and the transformer keeps only what the final `Gather(index 8)` uses: only the last token's output is read and its causal-mask row has no masked entry, so queries, attention rows, `c_proj`, LayerNorm 2 and the MLP are computed for one token instead of nine
(keys and values still for all nine), the mask disappears, and q/k/v are three matmuls on split weights instead of one matmul + a 5D reshape/transpose/3 slices/3 squeezes. This last part changes float32 summation order (hence 5e-5 / 7.6e-5, not 0).
`E_fold` additionally folds the 12 layer-scale `Mul`s into their convs and the 2 output-scale `Mul`s into their Gemms (see the float16 caveat).
MIL result: 315 ops without const (-11); `E_fold` 313.

**ALL_DROPIN / ALL (+ `_noB`).** Composition of the above (order A, B, [D], E, token path, transformer, C; see `make_variants.py`), with E without its weight folds. MIL non-const ops 313 to 314 (drop-in) and 299 to 300 (with the queues in the app), compared with 326:
144 `conv`, 0 `linear`, 2 `layer_norm`, 3 or 4 `mul`. Layout-only operations (`reshape`, `transpose`, `squeeze`, `expand_dims`): lean 7 + 6 + 3 + 3 = 19; the drop-in
combined variants 13 reshape + 2 transpose = 15; the combined variants with the queues in the app 7 reshape and nothing else = 7.

## ANEForge (static only)

`aneforge_check.py` imports each variant once per graph output (a copy of the graph pruned to that output, because the importer reads only `graph.output[0]`), reports the importer's exception and the
per-node standalone failures, and runs `preflight` for the families it offers. ANEForge commit 34c18a7, Linux, no dylib, nothing compiled.

- base, B, C: `NotImplementedError: ONNX op 'GatherND' not supported` (B and base also have the `Where` and the constant `Concat`; the constant-`Concat` failure only shows when the other two are gone, e.g. in E:
  `AttributeError: 'numpy.ndarray' object has no attribute 'srcs'`). D: `TypeError: select expects three graph Tensors` (the `Where`). E, E_fold: the constant `Concat`.
- A, ALL_DROPIN(_noB), ALL(_noB): the whole model imports, for every output (`outputs` and, where present, `next_state_desire_q`, `next_state_feat_q`), with no workaround; preflight: A 554, ALL_DROPIN_noB 547, ALL_DROPIN 551,
  ALL_noB 531, ALL 535 native, 0 reject / decompose / oversize, the same for A13, A14, A15, A16. The counts differ because ANEForge decomposes the graph differently, so they say nothing about quality.
- ANEForge's docs say the ANE does not accept float32 or int32 program inputs and outputs (uint8, float16 accepted). The Core ML packages have float32 inputs, which Core ML converts (the cast ops above); the `_io16` packages show the
  program without them. I did not try uint8 inputs.
- "Native" means "not in ANEForge's floor table and within its dimension caps". It is not a compiler verdict, and importing says nothing about whether the imported graph computes the right numbers: that was checked on the ONNX graphs only.

## What Apple's guidance says, and how this model relates to it

Sources read (2026-11-16; the sentences below came back through a summarising fetch tool, so treat wording as close paraphrase and read the originals before quoting):

- "Deploying Transformers on the Apple Neural Engine": https://machinelearning.apple.com/research/neural-engine-transformers
- `apple/ml-ane-transformers`: https://github.com/apple/ml-ane-transformers, with `ane_transformers/reference/layer_norm.py` and `.../multihead_attention.py` (raw files on GitHub)
- coremltools guide: https://apple.github.io/coremltools/docs-guides/source/faqs.html and https://apple.github.io/coremltools/docs-guides/source/model-prediction.html

The article's four principles: (1) data format: 4D channels-first (B, C, 1, S) with the sequence on the last axis, `nn.Linear` swapped for `nn.Conv2d` (it does not say "1x1"; the reference code uses kernel size 1);
the last axis of an ANE buffer "is not packed; it must be contiguous and aligned to 64 bytes", so a singleton last axis is padded (32x at 16 bit, it says); (2) chunk large intermediates (attention split per head) for L2 residency;
(3) minimise memory copies: reshapes and transposes "are likely to trigger memory copies unless specifically handled" (their attention has one transpose, on the key, and uses `einsum('bchq,bkhc->bkhq')`);
(4) short sequences are bandwidth-bound: more batch, quantisation or pruning help. It gives no op-support table, no numeric tensor-size limits, no softmax-axis rule. The repo README adds that the optimised model has more operations (longer load/compile),
that 4 of 606 operations (embedding lookups) ran on the CPU, and a measured 2.84x for DistilBERT at sequence length 128, batch 1 on A14/M1 and later. `LayerNormANE` is mean, subtract, square, mean, add eps, `rsqrt`, multiply on BC1S, then
(x + bias) * weight; the model needs x * weight + bias, which is what the variants use. The coremltools pages say nothing about ANE-friendly graph forms; they say that with no restriction all compute units including the ANE may be used, that
flexible shapes can leave the ANE unless they are `EnumeratedShapes` (up to 128), and that the first load specialises for the device and caches it (a `FastPrediction` strategy trades a longer first load for faster predictions).
A secondary source (ExecuTorch's Core ML docs) says the ANE computes in float16 only and that `RangeDim` pushes a model to CPU/GPU; I did not find that in Apple's own text.

How the graph relates (from `mac_count.py` on the lean model: 847 million MACs, 29.9 million parameters):

| part | nodes | MMACs | share of MACs | M parameters |
|---|---|---|---|---|
| Conv 1x1 | 30 | 637.7 | 75.3% | 6.23 |
| Conv dense 3x3 (stem) | 1 | 113.2 | 13.4% | 0.01 |
| Conv depthwise / grouped | 29 | 43.2 | 5.1% | 0.22 |
| MatMul with weights (transformer linears, 9 tokens) | 6 | 33.0 | 3.9% | 3.67 |
| Gemm (one vector times a matrix) | 82 | 19.8 | 2.3% | 19.80 |

- The backbone (60 `Conv`, 94% of the MACs) is already in the recommended form: 4D NCHW tensors. Its last-axis widths run 256, 128, 64, 32, 16, 8 elements (512 B down to 16 B in float16), so the late stages
  ([1,512,4,8], [1,1024,4,8]) fall under the 64-byte padding rule (up to 4x on small tensors). No variant touches it; there is nothing to change without changing the network.
- The rest departs from the article: 82 vector-matrix `Gemm` (sequence length 1) and a 9-token transformer in token-major `[1,9,512]` with 5D reshapes. That part is 6% of the MACs but 78% of the parameters, so it is the
  bandwidth-bound case of principle 4 (weights streamed for one vector). The article's remedies (batching, smaller weights) do not apply here (batch 1; no quantisation by the rules of this task).
  C applies principle 1 to it; E and the ALL variants apply principle 3: layout-only operations in the converted program (`reshape`, `transpose`, `squeeze`, `expand_dims`) go from 19 (6 of them transposes) to 15 (2 transposes) in ALL_DROPIN_noB and to 7 (no transposes) in ALL_noB.
- With sequence length 1 the (B, C, 1, S) layout has a singleton last axis, the padded case the article warns about (a [1,1024,1,1] vector would occupy 1024 x 64 B). The article's layout is aimed at S > 1. Whether the ANE prefers a 1x1 conv to `linear`
  at S = 1 is not stated by any of these sources, and I have no measurement. That is the open question C answers on a device, not here.
- Not applied: per-head chunking (principle 2): the attention tensors are [1,8,9,9] or [1,8,1,9], a few hundred bytes.

## What failed, was dropped, or was not done

- Core ML prediction, the Neural Engine, speed: not available here. The torch module that coremltools converts was run in float32 for 3 steps per variant (a check on the onnx2torch step and on the Pad / rsqrt converters added in `shim_extra.py`);
  coremltools' own MIL program was not run, so whether it computes what the ONNX computes is unverified for every variant, the shipped one included.
- B was first built as a plain fold (zero padding): rejected as wrong at the border (above). B is exact in float32 but not in float16 weights (caveat above).
- Folding the 0.125 score scale into the query weights: removed (float16 subnormals). E5/E7 folds: kept only in `E_fold`.
- The 0.0 bit-identity of the lean model does not carry over to the transformer rewrite, the Gemm-to-Conv swap or B: they are at rounding level (diffs above, control above). Variants that only move or select data (A, D) are bit-identical.
- The reference's stored `final_state_feat_q` differs from the carried state by 1.9e-3 for the lean model too (it is stored at float16 precision), so that comparison is not reported as a variant property.
- Not built: caching encoded feature rows; Core ML stateful models (iOS 18 `MLState`) for the queues instead of D's inputs; uint8 image input (Core ML multi-arrays; not tried); per-head attention chunking; a layout that avoids the
  singleton last axis of the head vectors. `action_t` is an input that nothing reads, in every variant (kept for the interface).
- ANEForge has no A19 family; A16 is its ceiling and covers A17/A18 by its own claim. The iPhone 17 (A19) is therefore not a named target, and I did not verify that claim.
- ANEForge's own MIL lowering of the importable variants was computed (op counts in `aneforge.json`, 1.8k to 1.9k ops because it expands GELU and LayerNorm) and is not used for anything here; the Core ML packages come only from the onnx2torch + coremltools route.

## Files

`gb.py` (graph-edit helper), `rewrites.py` (all rewrites, with docstrings stating exactness), `make_variants.py`, `verify_variants.py` (onnxruntime float32, both references, app-side queue emulation), `convert_variants.py` (+ `shim_extra.py`: Pad and
rsqrt converters on top of `scripts/onnx2torch_shim.py`), `aneforge_check.py`, `precision_controls.py`, `fp16_exactness.py`, `check_stem_fold.py`, `mac_count.py`, `make_report.py`, `run_all.sh`.
