#!/bin/bash
# Step 3b/3c: graph-level variants (from 04_graph_rewrites.py) through tinygrad, plus a few option combinations and BEAM.
# Usage: PY=... VARDIR=dir_with_variants OUT=variants.jsonl PASSES=2 CORE=3 ./03b_variants.sh
PY=${PY:-python}; OUT=${OUT:-variants.jsonl}; PASSES=${PASSES:-2}; CORE=${CORE:-3}; STEPS=${STEPS:-24}; VARDIR=${VARDIR:?}
DEVBASE=${DEVBASE:-CPU:CLANG:x86_64,x86-64-v3}
HERE=$(cd "$(dirname "$0")" && pwd)
run() { local label=$1 model=$2; shift 2; local pin=(); [ -n "$CORE" ] && pin=(taskset -c "$CORE")
  echo "=== $label" >&2
  env DEV="$DEVBASE" "$@" "${pin[@]}" timeout ${TMO:-900} "$PY" -I "$HERE/tg_harness.py" --model "$model" --label "$label" --steps "$STEPS" --json "$OUT" 2>&1 | grep -v '^RESULT' | tail -2 >&2; }
for p in $(seq 1 "$PASSES"); do
  for v in lean lean_foldnorm lean_foldnorm_foldscale lean_foldnorm_foldscale_concat; do run "variant:$v" "$VARDIR/$v.onnx"; done
done
run "PCONTIG=2 MAX_KERNEL_BUFFERS=16" "$VARDIR/lean.onnx" PCONTIG=2 MAX_KERNEL_BUFFERS=16
run "PCONTIG=2 MAX_KERNEL_BUFFERS=8" "$VARDIR/lean.onnx" PCONTIG=2 MAX_KERNEL_BUFFERS=8
run "PCONTIG=2 SPLIT_REDUCEOP=0 MAX_KERNEL_BUFFERS=1000" "$VARDIR/lean.onnx" PCONTIG=2 MAX_KERNEL_BUFFERS=1000
run "BEAM=1 (lean)" "$VARDIR/lean.onnx" BEAM=1
