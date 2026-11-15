/* Test driver: reads binary inputs written by run_tests.py, calls opcore, writes binary results.
   Usage:
     test_driver matrix  in.bin out.bin            in: N x (9 f32 K, 3 f32 rpy, 1 i32 big)  out: N x 9 f32
     test_driver parse   rows.bin params.bin out.bin   rows: N x 2576 f32; params: N x 3 f32 (v_ego, lat_t, long_t)
                                                       out: N x RESULT_FLOATS f32 (prev carried between rows)
     test_driver hist    frames_u8.bin u8|f32 out.bin reset_at   frame history -> img_stack per step
     test_driver warp    frame.bin w h stride m.bin out.bin   NV12, y plane stride==uv stride (uv at stride*y_height)
                         extra arg y_height after stride-> see main */
#include "../opcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RESULT_FLOATS (3 + 99 + 99 + 264 + 4 + 132 + 4)

static void *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  void *b = malloc(sz ? sz : 1);
  if (fread(b, 1, sz, f) != (size_t)sz) { perror("read"); exit(2); }
  fclose(f); *n = (size_t)sz; return b;
}
static void spit(const char *p, const void *b, size_t n) {
  FILE *f = fopen(p, "wb"); if (!f) { perror(p); exit(2); }
  if (fwrite(b, 1, n, f) != n) { perror("write"); exit(2); }
  fclose(f);
}
static void flatten(const op_result_t *r, float *o) {
  int k = 0;
  o[k++] = r->desired_curvature; o[k++] = r->desired_accel; o[k++] = (float)r->should_stop;
  memcpy(o + k, r->plan_pos, sizeof r->plan_pos); k += 99;
  memcpy(o + k, r->plan_vel, sizeof r->plan_vel); k += 99;
  memcpy(o + k, r->lane_lines, sizeof r->lane_lines); k += 264;
  memcpy(o + k, r->lane_line_probs, sizeof r->lane_line_probs); k += 4;
  memcpy(o + k, r->road_edges, sizeof r->road_edges); k += 132;
  o[k++] = r->lead_prob; o[k++] = r->lead_x; o[k++] = r->lead_y; o[k++] = r->lead_v;
}

int main(int argc, char **argv) {
  size_t n;
  if (argc >= 4 && !strcmp(argv[1], "matrix")) {
    unsigned char *in = slurp(argv[2], &n);
    size_t cnt = n / 52; float *out = malloc(cnt * 9 * sizeof(float));
    for (size_t i = 0; i < cnt; i++) {
      float K[9], rpy[3]; int big;
      memcpy(K, in + i*52, 36); memcpy(rpy, in + i*52 + 36, 12); memcpy(&big, in + i*52 + 48, 4);
      op_warp_matrix(K, rpy, big, out + i*9);
    }
    spit(argv[3], out, cnt * 9 * sizeof(float)); return 0;
  }
  if (argc >= 5 && !strcmp(argv[1], "parse")) {
    float *rows = slurp(argv[2], &n); size_t cnt = n / (OP_OUT_LEN * 4);
    float *par = slurp(argv[3], &n);
    float *out = malloc(cnt * RESULT_FLOATS * sizeof(float));
    op_result_t prev, cur; int have = 0;
    for (size_t i = 0; i < cnt; i++) {
      /* a v_ego/lat/long of NaN in params[0] row means "reset prev" (new sequence) */
      if (par[i*3] != par[i*3]) { have = 0; par[i*3] = 0.f; }
      op_parse_outputs(rows + i*OP_OUT_LEN, par[i*3], par[i*3 + 1], par[i*3 + 2], have ? &prev : NULL, &cur);
      flatten(&cur, out + i*RESULT_FLOATS);
      prev = cur; have = 1;
    }
    spit(argv[4], out, cnt * RESULT_FLOATS * sizeof(float)); return 0;
  }
  if (argc == 9 && !strcmp(argv[1], "warp")) {
    /* warp frame.bin width height stride y_height m.bin out.bin */
    uint8_t *fr = slurp(argv[2], &n);
    int w = atoi(argv[3]), h = atoi(argv[4]), stride = atoi(argv[5]), yh = atoi(argv[6]);
    float *m = slurp(argv[7], &n);
    float *out = malloc(OP_IMG_LEN * sizeof(float));
    op_frame_to_model_input(fr, stride, fr + (size_t)stride * yh, stride, w, h, m, out);
    spit(argv[8], out, OP_IMG_LEN * sizeof(float)); return 0;
  }
  if (argc == 6 && !strcmp(argv[1], "hist")) {
    /* hist frames_u8.bin u8|f32 out.bin reset_at: push each step (as uint8, or through the float path), write the
       img_stack after every push. reset_at >= 0 calls op_hist_reset just before that step's push. */
    uint8_t *fr = slurp(argv[2], &n); size_t steps = n / (2 * OP_IMG_LEN);
    float *out = malloc(steps * OP_STACK_LEN * sizeof(float));
    float *tmp = malloc(2 * OP_IMG_LEN * sizeof(float));
    uint8_t *mem = malloc(OP_HIST_BYTES); op_hist_t h; op_hist_init(&h, mem);
    int use_f32 = !strcmp(argv[3], "f32"), reset_at = atoi(argv[5]);
    for (size_t t = 0; t < steps; t++) {
      const uint8_t *f = fr + t * 2 * OP_IMG_LEN;
      if ((int)t == reset_at) op_hist_reset(&h);
      if (use_f32) { for (size_t i = 0; i < 2 * OP_IMG_LEN; i++) tmp[i] = (float)f[i]; op_hist_push_f32(&h, tmp); }
      else op_hist_push_u8(&h, f);
      op_hist_fill_stack(&h, out + t * OP_STACK_LEN);
    }
    spit(argv[4], out, steps * OP_STACK_LEN * sizeof(float));
    return 0;
  }
  fprintf(stderr, "bad usage\n"); return 1;
}
