#!/bin/sh
# One clip of the "More clips" study. Usage: run_clip_pe.sh <clip_dir> <python>
# <clip_dir> holds fcamera.hevc, ecamera.hevc, log.pkl, real_reference_fp32.npz, real_meta.npz (made by experiments/real_footage/run_clip.sh).
# Decodes the first 600 frames of both cameras to NV12 into <clip_dir>/phone_errors/frames (about 2.1 GB per camera for 1928x1208),
# checks the harness against the stored reference, writes check PNGs, runs the `more` group (two processes), then deletes the decoded frames.
set -e
D=$1; PY=$2; HERE=$(cd "$(dirname "$0")" && pwd); OUT="$D/phone_errors"; mkdir -p "$OUT/frames"
unset PE_OUT      # the study writes into <clip_dir>/phone_errors
if [ ! -f "$D/f_nv12.bin" ]; then
  for p in f:fcamera e:ecamera; do
    [ -f "$OUT/frames/${p%%:*}_nv12.bin.done" ] && continue
    ffmpeg -loglevel error -y -i "$D/${p#*:}.hevc" -frames:v 600 -pix_fmt nv12 -f rawvideo "$OUT/frames/${p%%:*}_nv12.bin"; touch "$OUT/frames/${p%%:*}_nv12.bin.done"
  done
fi
export RF_SCRATCH="$D"
"$PY" -I "$HERE/check_reference.py" > "$OUT/check_reference.log" 2>&1
"$PY" -I "$HERE/check_images.py" > "$OUT/check_images.log" 2>&1
"$PY" -I "$HERE/run_all.py" oob > "$OUT/oob.log" 2>&1
"$PY" -I "$HERE/run_all.py" more zzz_none > /dev/null 2>&1          # writes more/_base.npz only
"$PY" -I "$HERE/run_all.py" more ctl_ cal_ foc_ > "$OUT/run_A.log" 2>&1 &
"$PY" -I "$HERE/run_all.py" more wide_ narrow_ one_ ios_ drop img_ > "$OUT/run_B.log" 2>&1 &
wait
rm -f "$OUT/frames/f_nv12.bin" "$OUT/frames/e_nv12.bin"
