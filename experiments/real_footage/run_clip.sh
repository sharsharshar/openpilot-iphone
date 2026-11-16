#!/bin/sh
# Build the float32 reference for one clip directory containing fcamera.hevc, ecamera.hevc and rlog.zst / rlog.bz2.
# Usage: run_clip.sh <clip_dir> <cereal_dir> <python>      (RF_NF / RF_START as in build_reference.py)
# Decodes the first RF_NF (600) frames of both cameras to NV12 in <clip_dir>, builds the reference, then deletes the NV12 files.
set -e
D=$1; CER=$2; PY=$3; HERE=$(cd "$(dirname "$0")" && pwd); NF=${RF_NF:-600}
"$PY" -I "$HERE/extract_log.py" "$D" "$CER" > "$D/extract_log.log"
for p in f:fcamera e:ecamera; do
  ffmpeg -loglevel error -y -i "$D/${p#*:}.hevc" -frames:v "$NF" -pix_fmt nv12 -f rawvideo "$D/${p%%:*}_nv12.bin"
done
RF_SCRATCH="$D" RF_FIXTURES=0 "$PY" -I "$HERE/build_reference.py" > "$D/build_reference.log" 2>&1
rm -f "$D/f_nv12.bin" "$D/e_nv12.bin"
