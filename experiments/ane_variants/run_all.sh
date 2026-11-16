#!/bin/sh
# One command: models/driving_supercombo_fp32.onnx -> lean model -> all variants -> verification -> Core ML packages and op counts
#              -> ANEForge static checks -> controls -> models/variants/summary.md
#
#   sh experiments/ane_variants/run_all.sh
#
# Environment (all optional):
#   PY          python with numpy, onnx, onnxruntime, torch, onnx2torch, coremltools (default: python3)
#   REAL_REF    real-footage reference npz (experiments/real_footage/build_reference.py makes it); without it only the 40-step
#               synthetic reference is checked and the precision controls are skipped
#   ANEFORGE    a clone of https://github.com/sbryngelson/ANEForge (commit 34c18a7); without it the ANEForge step is skipped
# Writes everything under models/variants/ (git-ignored). About 25 minutes on 4 cores with the real reference, 5 without.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
OUT=models/variants
A=experiments/ane_variants
mkdir -p $OUT
$PY -I scripts/make_lean.py models/driving_supercombo_fp32.onnx $OUT/_lean > $OUT/make_lean.log
$PY -I $A/make_variants.py $OUT/_lean/driving_supercombo_lean_fp32.onnx $OUT
rm -f $OUT/verify.json $OUT/coreml.json $OUT/aneforge.json
if [ -n "$REAL_REF" ]; then
  $PY -I $A/verify_variants.py $OUT all --real "$REAL_REF" --json $OUT/verify.json
else
  $PY -I $A/verify_variants.py $OUT all --json $OUT/verify.json
fi
$PY -I $A/convert_variants.py $OUT all --json $OUT/coreml.json 2>&1 | grep -v "passes/s\|ops/s"
$PY -I $A/convert_variants.py $OUT base,ALL_DROPIN,ALL --io16 --json $OUT/coreml.json 2>&1 | grep -v "passes/s\|ops/s"
$PY -I $A/check_stem_fold.py $OUT
if [ -n "$ANEFORGE" ]; then $PY -I $A/aneforge_check.py "$ANEFORGE" $OUT all --json $OUT/aneforge.json; fi
if [ -n "$REAL_REF" ]; then $PY -I $A/precision_controls.py $OUT "$REAL_REF" $OUT/controls.json; fi
$PY -I $A/make_report.py $OUT
