// ops.h: numeric kernels (ROADMAP §8). CONTRACT HEADER: signatures are
// shared by the CPU implementation (src/ops/*_cpu.cpp) and the future CUDA
// implementation (Phase B). Change only through the DECISIONS.md process.
//
// Rules for every function (CLAUDE.md §4.2):
//   - free functions in tslm::ops taking raw pointers plus dimensions;
//   - pointers may be device pointers in CUDA builds, so callers never
//     dereference them on the host and kernels never allocate, throw, or
//     keep global state;
//   - *_forward OVERWRITES its outputs; *_backward (Phase 3) ACCUMULATES (+=);
//   - output buffers must not overlap input buffers;
//   - layouts are row-major (§6.2). Sizes that can exceed 2^31 are size_t.
//
// This file grows task by task. Declared so far: P2.1-P2.3 forward ops.
#pragma once

#include <cstddef>
#include <cstdint>

#include "tslm/common.h"

namespace tslm {
namespace ops {

// Embedding: out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:]
//   out [B,T,C], tokens [B,T] (each in [0, V), validated by the caller),
//   wte [V,C], wpe [>=T, C].
void embedding_forward(real_t* out, const int32_t* tokens, const real_t* wte, const real_t* wpe,
                       int B, int T, int C);

// LayerNorm over the last dim of N rows (eps 1e-5, biased variance):
//   mean[n] = sum_c inp[n,c] / C
//   rstd[n] = 1 / sqrt(sum_c (inp[n,c] - mean[n])^2 / C + 1e-5)
//   out[n,c] = (inp[n,c] - mean[n]) * rstd[n] * gamma[c] + beta[c]
//   out, inp [N,C]; mean, rstd [N] (saved for backward); gamma, beta [C].
void layernorm_forward(real_t* out, real_t* mean, real_t* rstd, const real_t* inp,
                       const real_t* gamma, const real_t* beta, int N, int C);

// GELU, tanh approximation, elementwise over n values:
//   out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_forward(real_t* out, const real_t* inp, size_t n);

// Residual add, elementwise over n values: out = a + b.
void residual_forward(real_t* out, const real_t* a, const real_t* b, size_t n);

// Linear: out[N,OC] = inp[N,C] * W[OC,C]^T (+ bias[OC], nullable)
//   out[n,o] = sum_c inp[n,c] * W[o,c] + (bias ? bias[o] : 0)
void matmul_forward(real_t* out, const real_t* inp, const real_t* W, const real_t* bias, int N, int C,
                    int OC);

// Causal multi-head attention on fused qkv[B,T,3C] -> out[B,T,C]; att[B,NH,T,T] saved.
//   hs = C / NH; for token (b,t): Q = qkv[b,t,0:C], K = qkv[b,t,C:2C], V = qkv[b,t,2C:3C],
//   head h occupies [h*hs, (h+1)*hs) inside each third.
//   att[b,h,t,t2] = softmax_{t2<=t}( Q_h(t) . K_h(t2) / sqrt(hs) ), and exactly 0 for t2 > t
//   out[b,t,h*hs+i] = sum_{t2<=t} att[b,h,t,t2] * V_h(t2)[i]
void attention_forward(real_t* out, real_t* att, const real_t* qkv, int B, int T, int C, int NH);

}  // namespace ops
}  // namespace tslm
