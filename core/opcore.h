/* opcore: portable C99 core for running comma.ai openpilot's driving model (warp + output parsing).
   Semantics follow openpilot (MIT): selfdrive/modeld/{parse_model_outputs,modeld,constants}.py,
   selfdrive/controls/lib/drive_helpers.py, common/transformations/{model,camera,orientation}.py and
   tinygrad examples/openpilot/compile_warp.py. See core/tests for the oracle. */
#ifndef OPCORE_H
#define OPCORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#define OP_IMG_C 6
#define OP_IMG_H 128
#define OP_IMG_W 256
#define OP_IMG_LEN (OP_IMG_C * OP_IMG_H * OP_IMG_W)
#define OP_OUT_LEN 2576
#define OP_IDX_N 33
#define OP_NUM_LANE_LINES 4
#define OP_NUM_ROAD_EDGES 2

/* 3x3 row-major matrix that maps a model-frame pixel (u, v, 1) to a camera-frame pixel (homogeneous),
   i.e. what openpilot passes to its warp. cam_intrinsics is the camera's 3x3 row-major K in pixels for the
   full camera frame. rpy_calib is device-from-calibrated roll, pitch, yaw in radians (all zero = perfectly
   mounted). big_model = 0 for the narrow "img" input, 1 for the wide "big_img" input. */
void op_warp_matrix(const float cam_intrinsics[9], const float rpy_calib[3], int big_model, float out_m[9]);

/* Warp one NV12 camera frame into one model image: out is OP_IMG_LEN floats, layout [6][128][256], values 0..255,
   channel order and sampling exactly as openpilot's warp + YUV420 packing. */
void op_frame_to_model_input(const uint8_t *y, int y_stride, const uint8_t *uv, int uv_stride,
                             int width, int height, const float m[9], float *out);

typedef struct {
  float desired_curvature;                 /* 1/m, as openpilot's get_action_from_model */
  float desired_accel;                     /* m/s^2 */
  int   should_stop;
  float plan_pos[OP_IDX_N][3];             /* x, y, z in metres at T_IDXS */
  float plan_vel[OP_IDX_N][3];
  float lane_lines[OP_NUM_LANE_LINES][OP_IDX_N][2];   /* y, z at X_IDXS */
  float lane_line_probs[OP_NUM_LANE_LINES];           /* 0..1 */
  float road_edges[OP_NUM_ROAD_EDGES][OP_IDX_N][2];
  float lead_prob;                         /* probability of a lead now, 0..1 */
  float lead_x, lead_y, lead_v;            /* first lead at t=0 */
} op_result_t;

/* Parse the raw 2576-float model output. prev may be NULL on the first call. */
void op_parse_outputs(const float *outputs, float v_ego, float lat_action_t, float long_action_t,
                      const op_result_t *prev, op_result_t *out);

/* ------------------------------------------------------------------ frame history for the lean model
   The lean model takes img_stack [1][24][128][256] float32 (raw 0..255 values): for camera 0 then camera 1,
   6 channels of the image from 4 steps ago followed by 6 channels of the newest image. The history keeps the last
   OP_HIST_LEN model images per camera as uint8 (the warp output is integer valued 0..255, so this is exact and
   4x smaller than float32). Caller owns the memory: pass a buffer of OP_HIST_BYTES bytes to op_hist_init.
   Before OP_HIST_LEN pushes the missing images read as zeros, like the original model's zero-initialised queue. */
#define OP_HIST_LEN 5
#define OP_HIST_CAMS 2
#define OP_HIST_BYTES ((size_t)OP_HIST_LEN * OP_HIST_CAMS * OP_IMG_LEN)
#define OP_STACK_LEN (OP_HIST_CAMS * 2 * OP_IMG_LEN)   /* 24 * 128 * 256 floats */

typedef struct {
  uint8_t *buf;   /* OP_HIST_BYTES bytes, slot-major: slot, camera, then [6][128][256]. Owned by the caller. */
  int head;       /* slot the next push writes to; also the slot of the oldest image kept */
} op_hist_t;

/* Attach the buffer and clear it to zeros. */
void op_hist_init(op_hist_t *h, uint8_t *buf);
/* Forget all images (zeros again). */
void op_hist_reset(op_hist_t *h);
/* Append one step: frame is [2][6][128][256] (camera 0 then 1), the layout of the model's old new_img input.
   The float version expects integer values 0..255 (anything else is rounded and clamped). */
void op_hist_push_u8(op_hist_t *h, const uint8_t *frame);
void op_hist_push_f32(op_hist_t *h, const float *frame);
/* Write img_stack (OP_STACK_LEN floats) from the current history. */
void op_hist_fill_stack(const op_hist_t *h, float *img_stack);

/* x positions (metres) for lane lines / road edges, and time stamps (seconds) for the plan. */
extern const float OP_X_IDXS[OP_IDX_N];
extern const float OP_T_IDXS[OP_IDX_N];

#ifdef __cplusplus
}
#endif
#endif
