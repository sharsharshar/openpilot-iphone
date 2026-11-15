/* opcore: portable C99 core for running comma.ai openpilot's driving model (warp + output parsing).
   Semantics follow openpilot (MIT): selfdrive/modeld/{parse_model_outputs,modeld,constants}.py,
   selfdrive/controls/lib/drive_helpers.py, common/transformations/{model,camera,orientation}.py and
   tinygrad examples/openpilot/compile_warp.py. See core/tests for the oracle. */
#ifndef OPCORE_H
#define OPCORE_H

#ifdef __cplusplus
extern "C" {
#endif

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

/* x positions (metres) for lane lines / road edges, and time stamps (seconds) for the plan. */
extern const float OP_X_IDXS[OP_IDX_N];
extern const float OP_T_IDXS[OP_IDX_N];

#ifdef __cplusplus
}
#endif
#endif
