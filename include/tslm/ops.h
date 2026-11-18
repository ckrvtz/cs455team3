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
// This file grows task by task. Declared so far: P2.1-P2.4 forward ops, P3.1-P3.4 backward ops.
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
// dwte[tokens[b,t],:] += dout[b,t,:];  dwpe[t,:] += dout[b,t,:]  (repeated tokens accumulate)
void embedding_backward(real_t* dwte, real_t* dwpe, const real_t* dout, const int32_t* tokens, int B, int T,
                        int C);

// LayerNorm over the last dim of N rows (eps 1e-5, biased variance):
//   mean[n] = sum_c inp[n,c] / C
//   rstd[n] = 1 / sqrt(sum_c (inp[n,c] - mean[n])^2 / C + 1e-5)
//   out[n,c] = (inp[n,c] - mean[n]) * rstd[n] * gamma[c] + beta[c]
//   out, inp [N,C]; mean, rstd [N] (saved for backward); gamma, beta [C].
void layernorm_forward(real_t* out, real_t* mean, real_t* rstd, const real_t* inp,
                       const real_t* gamma, const real_t* beta, int N, int C);
// With xhat = (inp - mean) * rstd and g = dout * gamma (per row, means over C):
//   dinp  += rstd * (g - mean(g) - xhat * mean(g * xhat))
//   dgamma += sum_n dout * xhat;   dbeta += sum_n dout
// mean/rstd are the values saved by layernorm_forward.
void layernorm_backward(real_t* dinp, real_t* dgamma, real_t* dbeta, const real_t* dout, const real_t* inp,
                        const real_t* gamma, const real_t* mean, const real_t* rstd, int N, int C);

// GELU, tanh approximation, elementwise over n values:
//   out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_forward(real_t* out, const real_t* inp, size_t n);
// dinp += dout * d/dx[gelu(x)] (exact derivative of the tanh form above)
void gelu_backward(real_t* dinp, const real_t* inp, const real_t* dout, size_t n);

// Residual add, elementwise over n values: out = a + b.
void residual_forward(real_t* out, const real_t* a, const real_t* b, size_t n);
// da += dout; db += dout
void residual_backward(real_t* da, real_t* db, const real_t* dout, size_t n);

// Linear: out[N,OC] = inp[N,C] * W[OC,C]^T (+ bias[OC], nullable)
//   out[n,o] = sum_c inp[n,c] * W[o,c] + (bias ? bias[o] : 0)
void matmul_forward(real_t* out, const real_t* inp, const real_t* W, const real_t* bias, int N, int C,
                    int OC);
// dinp += dout*W ; dW += dout^T*inp ; dbias += colsum(dout)  (dbias nullable)
//   dinp[n,c] += sum_o dout[n,o] * W[o,c];  dW[o,c] += sum_n dout[n,o] * inp[n,c];  dbias[o] += sum_n dout[n,o]
void matmul_backward(real_t* dinp, real_t* dW, real_t* dbias, const real_t* dout, const real_t* inp,
                     const real_t* W, int N, int C, int OC);

// Causal multi-head attention on fused qkv[B,T,3C] -> out[B,T,C]; att[B,NH,T,T] saved.
//   hs = C / NH; for token (b,t): Q = qkv[b,t,0:C], K = qkv[b,t,C:2C], V = qkv[b,t,2C:3C],
//   head h occupies [h*hs, (h+1)*hs) inside each third.
//   att[b,h,t,t2] = softmax_{t2<=t}( Q_h(t) . K_h(t2) / sqrt(hs) ), and exactly 0 for t2 > t
//   out[b,t,h*hs+i] = sum_{t2<=t} att[b,h,t,t2] * V_h(t2)[i]
void attention_forward(real_t* out, real_t* att, const real_t* qkv, int B, int T, int C, int NH);
// Backward through attention_forward, ACCUMULATING into dqkv [B,T,3C] (same layout as qkv).
// att is the forward's saved post-softmax att. scratch: B*NH*T*T reals of workspace,
// contents undefined on return (serial keeps datt/dpre there; CUDA batched-GEMM uses it too).
//   datt = dY V^T;  dV += att^T dY;  dpre = att * (datt - rowsum(att * datt))
//   dQ += dpre K / sqrt(hs);  dK += dpre^T Q / sqrt(hs)   (causal entries only)
void attention_backward(real_t* dqkv, real_t* scratch, const real_t* dout, const real_t* qkv, const real_t* att, int B,
                        int T, int C, int NH);

// Softmax + cross entropy over N rows of V logits.
//   probs[n,v] = exp(logits[n,v] - m_n) / sum_v' exp(logits[n,v'] - m_n),  m_n = max_v logits[n,v]
//   losses[n]  = -log probs[n, targets[n]]   (targets[n] in [0, V), validated by the caller)
// Per-position losses only; the host reduces them with sum_reduce.
void softmax_xent_forward(real_t* probs, real_t* losses, const real_t* logits, const int32_t* targets, int N,
                          int V);
// Gradient of the MEAN loss (1/N) sum_n losses[n] w.r.t. logits, ACCUMULATED:
//   dlogits[n,v] += (probs[n,v] - 1[v == targets[n]]) / N
void softmax_xent_backward(real_t* dlogits, const real_t* probs, const int32_t* targets, int N, int V);

// Sum of n values accumulated in double, returned to the host (used for the mean loss).
double sum_reduce(const real_t* x, size_t n);

}  // namespace ops
}  // namespace tslm
