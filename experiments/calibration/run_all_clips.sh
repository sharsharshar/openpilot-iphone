#!/bin/sh
# Drive the whole experiment, one clip at a time (disk: decoded frames of one clip at a time, deleted afterwards).
# Usage: run_all_clips.sh <python> <cereal_dir> <name>=<clip_dir>:<loops> ...
#   e.g. run_all_clips.sh $PY $SCRATCH/real/cereal original=$SCRATCH/real:4 gm=$SCRATCH/real_more/gm:8
# Needs ffmpeg for clips whose frames were deleted. Four worker processes with one onnxruntime thread each (results do not depend on the thread count:
# 100 steps with 1, 2 and 4 threads are bit-identical to the stored reference).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); PY=$1; CER=$2; shift 2
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; D=${rest%%:*}; L=${rest#*:}
  mkdir -p "$D/calibration"
  "$PY" -I "$HERE/extract_cal_log.py" "$D" "$CER" > "$D/calibration/extract.log" 2>&1
  CAL_CLIP_DIR=$D "$PY" -I "$HERE/obs_check.py" > "$D/calibration/obs_check.log" 2>&1
  mine=0
  if [ ! -f "$D/f_nv12.bin" ] && [ ! -f "$D/calibration/f_nv12.bin" ]; then "$HERE/decode_clip.sh" "$D" > "$D/calibration/decode.log" 2>&1; mine=1; fi
  for i in 0 1 2 3; do
    CAL_CLIP_DIR=$D CAL_LOOPS=$L CAL_THREADS=1 "$PY" -I "$HERE/run_clip.py" --part $i/4 > "$D/calibration/run_$i.log" 2>&1 &
  done
  wait
  [ $mine = 1 ] && rm -f "$D/calibration/f_nv12.bin" "$D/calibration/e_nv12.bin"
  echo "$name done"
done
