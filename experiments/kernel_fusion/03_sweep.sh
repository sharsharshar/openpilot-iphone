#!/bin/bash
# Step 3: sweep the tinygrad options that exist in 0.14.0 (see RESULTS.md for file:line of each) on the CPU backend, one pinned core.
# Every run checks all steps against reference/driving_reference_fp32.npz (worst_max / worst_mean in the JSON).
# Usage: PY=/path/to/python OUT=/path/to/sweep.jsonl PASSES=2 CORE=3 ./03_sweep.sh [extra config names...]
# Mac / Metal later: DEVBASE=METAL CORE= ./03_sweep.sh   (CORE empty = no taskset)
PY=${PY:-python}; OUT=${OUT:-sweep.jsonl}; PASSES=${PASSES:-2}; CORE=${CORE:-3}; STEPS=${STEPS:-24}
DEVBASE=${DEVBASE:-CPU:CLANG:x86_64,x86-64-v3}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$(dirname "$HERE")")
FP32=$ROOT/models/driving_supercombo_fp32.onnx; LEAN=$ROOT/models/driving_supercombo_lean_fp32.onnx
run() {  # label model env...
  local label=$1 model=$2; shift 2
  local pin=(); [ -n "$CORE" ] && pin=(taskset -c "$CORE")
  echo "=== $label" >&2
  env DEV="$DEVBASE" "$@" "${pin[@]}" "$PY" -I "$HERE/tg_harness.py" --model "$model" --label "$label" --steps "$STEPS" --json "$OUT" 2>&1 | grep -v '^RESULT' | tail -2 >&2 \
    || echo "FAILED $label" >&2
}
for p in $(seq 1 "$PASSES"); do
  run base            $FP32
  run PCONTIG=1       $FP32 PCONTIG=1
  run PCONTIG=2       $FP32 PCONTIG=2
  run PCONTIG=3       $FP32 PCONTIG=3
  run SPLIT_REDUCEOP=0 $FP32 SPLIT_REDUCEOP=0
  run NOOPT=1         $FP32 NOOPT=1
  run MAX_KERNEL_BUFFERS=4    $FP32 MAX_KERNEL_BUFFERS=4
  run MAX_KERNEL_BUFFERS=8    $FP32 MAX_KERNEL_BUFFERS=8
  run MAX_KERNEL_BUFFERS=1000 $FP32 MAX_KERNEL_BUFFERS=1000
  run WINO=1          $FP32 WINO=1
  run TRANSCENDENTAL=2 $FP32 TRANSCENDENTAL=2
  run lean            $LEAN
done
