# Fixtures

Each set is 40 consecutive 20 Hz steps: model input frames (`[40][2][6][128][256]` uint8, camera 0 = narrow,
camera 1 = wide) and the outputs the float32 reference model gave for them starting from zero state
(`[40][2576]` float32, little-endian).

- `frames_u8.bin`, `outputs_f32.bin`: synthetic frames, with `desire[3]` pulsed at step 20. Checks the plumbing.
- `real_frames_u8.bin`, `real_outputs_f32.bin`: real road video, no desire pulse. The frames are video frames
  300..339 of segment 1 of route `98395b7c5b27882e|0000002b--2686b5a2d0`, the segment openpilot's own model-replay
  test uses, taken from comma's public `openpilotci` store and warped to model-input size with the C core and the
  calibration from that drive's log. The footage belongs to comma.ai; it is included here only as a small
  downsampled test input. Built by `experiments/real_footage/build_reference.py`.
