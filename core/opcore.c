#include "opcore.h"
#include <math.h>
#include <string.h>

const float OP_T_IDXS[OP_IDX_N] = {
  0.0f, 0.009765625f, 0.0390625f, 0.087890625f, 0.15625f, 0.244140625f, 0.3515625f, 0.478515625f, 0.625f,
  0.791015625f, 0.9765625f, 1.181640625f, 1.40625f, 1.650390625f, 1.9140625f, 2.197265625f, 2.5f,
  2.822265625f, 3.1640625f, 3.525390625f, 3.90625f, 4.306640625f, 4.7265625f, 5.166015625f, 5.625f,
  6.103515625f, 6.6015625f, 7.119140625f, 7.65625f, 8.212890625f, 8.7890625f, 9.384765625f, 10.0f };
const float OP_X_IDXS[OP_IDX_N] = {
  0.0f, 0.1875f, 0.75f, 1.6875f, 3.0f, 4.6875f, 6.75f, 9.1875f, 12.0f, 15.1875f, 18.75f, 22.6875f, 27.0f,
  31.6875f, 36.75f, 42.1875f, 48.0f, 54.1875f, 60.75f, 67.6875f, 75.0f, 82.6875f, 90.75f, 99.1875f, 108.0f,
  117.1875f, 126.75f, 136.6875f, 147.0f, 157.6875f, 168.75f, 180.1875f, 192.0f };

/* ------------------------------------------------------------------ warp matrix */

/* Warp destination is the 512x256 luma plane (model_w x model_h in tinygrad); the packed model image is
   6 x 128 x 256. Intrinsics of the model frames: openpilot common/transformations/model.py
   (medmodel: f=910, cx=256, cy=47.6; sbigmodel: f=455, cx=256, cy=0.5*(256+47.6)). */
static void inv3(const double a[9], double o[9]) {
  double c00 = a[4]*a[8] - a[5]*a[7], c01 = a[5]*a[6] - a[3]*a[8], c02 = a[3]*a[7] - a[4]*a[6];
  double det = a[0]*c00 + a[1]*c01 + a[2]*c02;
  double id = 1.0 / det;
  o[0] = c00*id;                       o[1] = (a[2]*a[7] - a[1]*a[8])*id;  o[2] = (a[1]*a[5] - a[2]*a[4])*id;
  o[3] = c01*id;                       o[4] = (a[0]*a[8] - a[2]*a[6])*id;  o[5] = (a[2]*a[3] - a[0]*a[5])*id;
  o[6] = c02*id;                       o[7] = (a[1]*a[6] - a[0]*a[7])*id;  o[8] = (a[0]*a[4] - a[1]*a[3])*id;
}
static void mul3(const double a[9], const double b[9], double o[9]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      o[i*3 + j] = a[i*3]*b[j] + a[i*3 + 1]*b[3 + j] + a[i*3 + 2]*b[6 + j];
}

void op_warp_matrix(const float cam_intrinsics[9], const float rpy_calib[3], int big_model, float out_m[9]) {
  /* view_frame_from_device_frame = device_frame_from_view_frame^T */
  static const double V[9] = {0, 1, 0,  0, 0, 1,  1, 0, 0};
  const double f = big_model ? 455.0 : 910.0;
  const double cy = big_model ? 0.5 * (256.0 + 47.6) : 47.6;
  const double Km[9] = {f, 0, 256.0,  0, f, cy,  0, 0, 1};
  double KV[9], calib_from_model[9];
  mul3(Km, V, KV);                      /* model K @ view_from_calib(0,0,0)[:, :3] */
  inv3(KV, calib_from_model);

  /* rot_from_euler: Rz(yaw) @ Ry(pitch) @ Rx(roll) */
  double cx = cos((double)rpy_calib[0]), sx = sin((double)rpy_calib[0]);
  double cyy = cos((double)rpy_calib[1]), sy = sin((double)rpy_calib[1]);
  double cz = cos((double)rpy_calib[2]), sz = sin((double)rpy_calib[2]);
  const double Rx[9] = {1, 0, 0,  0, cx, -sx,  0, sx, cx};
  const double Ry[9] = {cyy, 0, sy,  0, 1, 0,  -sy, 0, cyy};
  const double Rz[9] = {cz, -sz, 0,  sz, cz, 0,  0, 0, 1};
  double t1[9], dev_from_calib[9], K[9], t2[9], cam_from_calib[9], m[9];
  mul3(Rz, Ry, t1);
  mul3(t1, Rx, dev_from_calib);
  for (int i = 0; i < 9; i++) K[i] = (double)cam_intrinsics[i];
  mul3(K, V, t2);
  mul3(t2, dev_from_calib, cam_from_calib);
  mul3(cam_from_calib, calib_from_model, m);
  for (int i = 0; i < 9; i++) out_m[i] = (float)m[i];
}

/* ------------------------------------------------------------------ frame warp */

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF /* match unfused float32 math of the reference */
#endif

/* Nearest-neighbour perspective sample, as tinygrad warp_perspective_tinygrad: integer destination
   coordinates (no +0.5 pixel-centre shift), float32 math, round-half-to-even, clamp to the source rectangle.
   Plane sampler: src(row, col) = base[row*stride + col*step]. */
static inline int clampi(float v, int hi) {
  float r = rintf(v);
  if (!(r > 0.0f)) return 0;            /* also maps NaN to 0 */
  if (r > (float)hi) return hi;
  return (int)r;
}

static void warp_plane(const uint8_t *base, int stride, int step, int src_w, int src_h,
                       const float M[9], int dst_w, int dst_h, float *out) {
  for (int yy = 0; yy < dst_h; yy++) {
    for (int xx = 0; xx < dst_w; xx++) {
      float fx = (float)xx, fy = (float)yy;
      float sx = M[0]*fx + M[1]*fy + M[2];
      float sy = M[3]*fx + M[4]*fy + M[5];
      float sw = M[6]*fx + M[7]*fy + M[8];
      sx = sx / sw; sy = sy / sw;
      int ix = clampi(sx, src_w - 1), iy = clampi(sy, src_h - 1);
      out[(size_t)yy * dst_w + xx] =
          (float)base[(size_t)iy * stride + (size_t)ix * step];
    }
  }
}

void op_frame_to_model_input(const uint8_t *y, int y_stride, const uint8_t *uv, int uv_stride,
                             int width, int height, const float m[9], float *out) {
  /* UV_SCALE @ M @ UV_SCALE_INV == elementwise scale (tinygrad frame_prepare_tinygrad) */
  const float mu[9] = {m[0], m[1], m[2]*0.5f,  m[3], m[4], m[5]*0.5f,  m[6]*2.0f, m[7]*2.0f, m[8]};
  const int plane = OP_IMG_H * OP_IMG_W;
  /* luma: 512x256 warp, deinterleaved into channels 0..3 as [r%2 + 2*(c%2)] -> (even/even, odd/even, even/odd, odd/odd) */
  for (int ch = 0; ch < 4; ch++) {
    int dr = ch & 1, dc = ch >> 1;
    float *o = out + ch * plane;
    /* dst luma coords: x = 2*c + dc, y = 2*r + dr */
    for (int r = 0; r < OP_IMG_H; r++) {
      for (int c = 0; c < OP_IMG_W; c++) {
        float fx = (float)(2*c + dc), fy = (float)(2*r + dr);
        float sx = m[0]*fx + m[1]*fy + m[2];
        float sy = m[3]*fx + m[4]*fy + m[5];
        float sw = m[6]*fx + m[7]*fy + m[8];
        sx = sx / sw; sy = sy / sw;
        int ix = clampi(sx, width - 1), iy = clampi(sy, height - 1);
        o[r*OP_IMG_W + c] = (float)y[(size_t)iy * y_stride + ix];
      }
    }
  }
  /* chroma: 256x128 warps of the deinterleaved U and V planes (size width/2 x height/2) */
  for (int ch = 0; ch < 2; ch++) {
    warp_plane(uv + ch, uv_stride, 2, width / 2, height / 2, mu, OP_IMG_W, OP_IMG_H,
               out + (4 + ch) * plane);
  }
}

/* ------------------------------------------------------------------ output parser */

/* upstream safe_exp clips the argument at 11; sigmoid(x) = 1 / (1 + safe_exp(-x)) */
static float op_sigmoid(float x) {
  double e = -(double)x;
  if (e > 11.0) e = 11.0;
  return (float)(1.0 / (1.0 + exp(e)));
}

static double interp1(double x, const float *xp, const float *fp, int n) { /* numpy.interp semantics */
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  int i = 0;
  while (i < n - 2 && x >= xp[i + 1]) i++;
  double x0 = xp[i], x1 = xp[i + 1];
  return fp[i] + (fp[i + 1] - fp[i]) * (x - x0) / (x1 - x0);
}

#define DT_MDL 0.05
#define LONG_SMOOTH_SECONDS 0.3
#define LAT_SMOOTH_SECONDS 0.0
#define MIN_LAT_CONTROL_SPEED 0.3
#define MIN_SPEED 1.0
#define MIN_STABLE_DELAY 0.3

static double smooth_value(double val, double prev, double tau) {
  double alpha = tau > 0.0 ? 1.0 - exp(-DT_MDL / tau) : 1.0;
  return alpha * val + (1.0 - alpha) * prev;
}

void op_parse_outputs(const float *o, float v_ego_in, float lat_t_in, float long_t_in,
                      const op_result_t *prev, op_result_t *out) {
  /* Copy prev first so that out == prev works. */
  const float prev_curv = prev ? prev->desired_curvature : 0.0f;
  const float prev_acc = prev ? prev->desired_accel : 0.0f;

  /* blocks (mean = first half of each mdn block): plan 1576 (33x15), lane_lines 117, road_edges 653,
     lead 917 (3 hyp x 6 t x 4: x,y,v,a), lane_lines_prob 645 (8: odd entries are the probs), lead_prob 1061 */
  const float *plan = o + 1576;
  float f_vs[OP_IDX_N], f_yaws[OP_IDX_N];
  for (int i = 0; i < OP_IDX_N; i++) {
    for (int k = 0; k < 3; k++) {
      out->plan_pos[i][k] = plan[i*15 + k];
      out->plan_vel[i][k] = plan[i*15 + 3 + k];
    }
    f_vs[i] = plan[i*15 + 3]; f_yaws[i] = plan[i*15 + 11];
  }
  for (int l = 0; l < OP_NUM_LANE_LINES; l++) {
    for (int i = 0; i < OP_IDX_N; i++) {
      out->lane_lines[l][i][0] = o[117 + (l*OP_IDX_N + i)*2];
      out->lane_lines[l][i][1] = o[117 + (l*OP_IDX_N + i)*2 + 1];
    }
    out->lane_line_probs[l] = op_sigmoid(o[645 + 2*l + 1]);
  }
  for (int e = 0; e < OP_NUM_ROAD_EDGES; e++)
    for (int i = 0; i < OP_IDX_N; i++) {
      out->road_edges[e][i][0] = o[653 + (e*OP_IDX_N + i)*2];
      out->road_edges[e][i][1] = o[653 + (e*OP_IDX_N + i)*2 + 1];
    }
  out->lead_prob = op_sigmoid(o[1061]);
  out->lead_x = o[917]; out->lead_y = o[918]; out->lead_v = o[919];

  /* modeld.py get_action_from_model (plan branch) */
  double v_ego = v_ego_in > 0.0f ? (double)v_ego_in : 0.0;   /* modeld: v_ego = max(vEgo, 0) */
  double lat_t = lat_t_in, long_t = long_t_in;
  double v_now = f_vs[0], a_now = plan[6];
  double v_target;
  if (long_t < MIN_STABLE_DELAY)
    v_target = v_now + (long_t / MIN_STABLE_DELAY) * (interp1(MIN_STABLE_DELAY, OP_T_IDXS, f_vs, OP_IDX_N) - v_now);
  else
    v_target = interp1(long_t, OP_T_IDXS, f_vs, OP_IDX_N);
  double a_target = 2.0 * (v_target - v_now) / long_t - a_now;

  double psi_target;
  if (lat_t < MIN_STABLE_DELAY)
    psi_target = (lat_t / MIN_STABLE_DELAY) * interp1(MIN_STABLE_DELAY, OP_T_IDXS, f_yaws, OP_IDX_N);
  else
    psi_target = interp1(lat_t, OP_T_IDXS, f_yaws, OP_IDX_N);
  double vc = v_ego > MIN_SPEED ? v_ego : MIN_SPEED;
  double psi_rate = plan[14];
  double curv = 2.0 * (psi_target / (vc * lat_t)) - psi_rate / vc;

  int stop = (v_ego < 0.3 && a_target < 0.1) ? 1 : 0;
  double acc = smooth_value(a_target, prev_acc, LONG_SMOOTH_SECONDS);
  if (v_ego > MIN_LAT_CONTROL_SPEED) curv = smooth_value(curv, prev_curv, LAT_SMOOTH_SECONDS);
  else curv = prev_curv;
  out->desired_curvature = (float)curv;
  out->desired_accel = (float)acc;
  out->should_stop = stop;
}
