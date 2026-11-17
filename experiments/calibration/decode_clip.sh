#!/bin/sh
# Decode the first 600 frames of both cameras of one clip to NV12 in <clip_dir>/calibration/ (same ffmpeg command as experiments/real_footage/run_clip.sh).
# Usage: decode_clip.sh <clip_dir>        delete afterwards with: rm -f <clip_dir>/calibration/f_nv12.bin <clip_dir>/calibration/e_nv12.bin
set -e
D=$1; NF=${RF_NF:-600}; mkdir -p "$D/calibration"
for p in f:fcamera e:ecamera; do
  ffmpeg -loglevel error -y -i "$D/${p#*:}.hevc" -frames:v "$NF" -pix_fmt nv12 -f rawvideo "$D/calibration/${p%%:*}_nv12.bin"
done
ls -l "$D/calibration"/*_nv12.bin
