// Hand-written kernels for the ConvFFN part of one encoder block of the driving model, single thread, AVX2+FMA.
//   Y = T + gamma * ( W2 * gelu_tanh( W1 * X + b1 ) + b2 )          (1x1 convs, CHW layout: X[C][P], pixel index contiguous)
// X is the output of the depthwise 7x7 conv, T the block's residual input (output of the token mixer).
// Three versions of the same arithmetic, to separate "fusion" from "kernel quality":
//   ffn_unfused   5 passes over memory:  gemm1 | bias+gelu | gemm2 | bias+scale+residual     (what a graph without fusion does)
//   ffn_epilogue  2 passes:  gemm1+bias+gelu | gemm2+bias+scale+residual                       (what tinygrad generates structurally)
//   ffn_fused     1 pass:    16-pixel tile, hidden tile kept in a stack buffer (L1), never written to the big hidden tensor
// build: clang -O3 -mavx2 -mfma -ffp-contract=fast -shared -fPIC fused_block.c -o fused_block.so
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define TP 16          // pixels per tile (two ymm vectors)
#define MR 4           // output channels per register block
#define MAXHD 1536

static inline __m256 exp256(__m256 x) {   // Cephes expf, relative error ~1e-7
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(87.0f));
  __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  __m256i n = _mm256_cvtps_epi32(fx);
  __m256 p2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(y, p2n);
}
// gelu(x) with the tanh approximation (the ONNX Gelu nodes of this model have approximate="tanh"):
//   0.5*x*(1+tanh(u)) = x / (1 + exp(-2u)),  u = sqrt(2/pi) * (x + 0.044715 x^3)
static inline __m256 gelu8(__m256 x) {
  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 u = _mm256_mul_ps(x, _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x2, _mm256_set1_ps(1.0f)));
  u = _mm256_mul_ps(u, _mm256_set1_ps(-2.0f * 0.7978845608028654f));
  return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), exp256(u)));
}

// out[n][p0..p0+15] for n in [0,N) from in[k][p0..] (row stride ldin), W[n][k]; epilogue selected by mode:
//  0: raw   1: gelu(acc + b)   2: res + g*(acc + b)
static inline void block16(int K, int N, const float *in, int ldin, const float *W, const float *b, const float *g,
                           const float *res, int ldres, float *out, int ldout, int mode) {
  for (int n = 0; n < N; n += MR) {
    __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
    const float *w0 = W + (size_t)n * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    for (int k = 0; k < K; k++) {
      __m256 x0 = _mm256_loadu_ps(in + (size_t)k * ldin), x1 = _mm256_loadu_ps(in + (size_t)k * ldin + 8);
      __m256 w;
      w = _mm256_broadcast_ss(w0 + k); a00 = _mm256_fmadd_ps(w, x0, a00); a01 = _mm256_fmadd_ps(w, x1, a01);
      w = _mm256_broadcast_ss(w1 + k); a10 = _mm256_fmadd_ps(w, x0, a10); a11 = _mm256_fmadd_ps(w, x1, a11);
      w = _mm256_broadcast_ss(w2 + k); a20 = _mm256_fmadd_ps(w, x0, a20); a21 = _mm256_fmadd_ps(w, x1, a21);
      w = _mm256_broadcast_ss(w3 + k); a30 = _mm256_fmadd_ps(w, x0, a30); a31 = _mm256_fmadd_ps(w, x1, a31);
    }
    __m256 acc[4][2] = {{a00, a01}, {a10, a11}, {a20, a21}, {a30, a31}};
    for (int j = 0; j < MR; j++) {
      float *o = out + (size_t)(n + j) * ldout;
      for (int h = 0; h < 2; h++) {
        __m256 v = acc[j][h];
        if (mode == 1) v = gelu8(_mm256_add_ps(v, _mm256_broadcast_ss(b + n + j)));
        else if (mode == 2) v = _mm256_fmadd_ps(_mm256_broadcast_ss(g + n + j), _mm256_add_ps(v, _mm256_broadcast_ss(b + n + j)),
                                               _mm256_loadu_ps(res + (size_t)(n + j) * ldres + h * 8));
        _mm256_storeu_ps(o + h * 8, v);
      }
    }
  }
}

// ---- one pass: pointwise conv (+epilogue) over the whole image, P multiple of 16 ----
static void gemm_pass(int P, int K, int N, const float *X, const float *W, const float *b, const float *g, const float *res, float *Y, int mode) {
  for (int p0 = 0; p0 < P; p0 += TP) block16(K, N, X + p0, P, W, b, g, res ? res + p0 : 0, P, Y + p0, P, mode);
}
static void bias_gelu_pass(int P, int N, const float *b, float *H) {
  for (int n = 0; n < N; n++) { __m256 bb = _mm256_set1_ps(b[n]);
    for (int p = 0; p < P; p += 8) _mm256_storeu_ps(H + (size_t)n * P + p, gelu8(_mm256_add_ps(_mm256_loadu_ps(H + (size_t)n * P + p), bb))); }
}
static void bias_scale_res_pass(int P, int N, const float *b, const float *g, const float *T, const float *Z, float *Y) {
  for (int n = 0; n < N; n++) { __m256 bb = _mm256_set1_ps(b[n]), gg = _mm256_set1_ps(g[n]);
    for (int p = 0; p < P; p += 8)
      _mm256_storeu_ps(Y + (size_t)n * P + p, _mm256_fmadd_ps(gg, _mm256_add_ps(_mm256_loadu_ps(Z + (size_t)n * P + p), bb), _mm256_loadu_ps(T + (size_t)n * P + p))); }
}

// scratch: Hd*P floats (hidden) and C*P floats (second gemm output) for the unfused version
void ffn_unfused(int P, int C, int Hd, const float *X, const float *T, const float *W1, const float *b1, const float *W2, const float *b2,
                 const float *g, float *Y, float *H, float *Z) {
  gemm_pass(P, C, Hd, X, W1, 0, 0, 0, H, 0);
  bias_gelu_pass(P, Hd, b1, H);
  gemm_pass(P, Hd, C, H, W2, 0, 0, 0, Z, 0);
  bias_scale_res_pass(P, C, b2, g, T, Z, Y);
}
void ffn_epilogue(int P, int C, int Hd, const float *X, const float *T, const float *W1, const float *b1, const float *W2, const float *b2,
                  const float *g, float *Y, float *H) {
  gemm_pass(P, C, Hd, X, W1, b1, 0, 0, H, 1);
  gemm_pass(P, Hd, C, H, W2, b2, g, T, Y, 2);
}
void ffn_fused(int P, int C, int Hd, const float *X, const float *T, const float *W1, const float *b1, const float *W2, const float *b2,
               const float *g, float *Y) {
  _Alignas(32) float Ht[MAXHD * TP];
  for (int p0 = 0; p0 < P; p0 += TP) {
    block16(C, Hd, X + p0, P, W1, b1, 0, 0, 0, Ht, TP, 1);             // hidden tile [Hd][16] stays in L1
    block16(Hd, C, Ht, TP, W2, b2, g, T + p0, P, Y + p0, P, 2);       // consumed immediately
  }
}
