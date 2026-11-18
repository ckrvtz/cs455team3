// attention_cpu.cpp: causal multi-head attention forward and backward, CPU (ROADMAP §8, §6.2, §6.3).
//
// Shapes (row-major): qkv [B,T,3C] (Q | K | V thirds; head h at
// [h*hs, (h+1)*hs) inside each third, hs = C/NH), att [B,NH,T,T], out [B,T,C].
// Math, for each (b, h) and query position t:
//   s[t2]   = Q_h(t) . K_h(t2) / sqrt(hs)             for t2 <= t (causal)
//   att[t2] = exp(s[t2] - max s) / sum_{t2'<=t} exp(s[t2'] - max s)
//   att[t2] = 0                                      for t2 > t
//   out[t, h*hs + i] = sum_{t2<=t} att[t2] * V_h(t2)[i]
// Semantics: OVERWRITES out and every element of att, including the masked
// upper triangle (exact zeros), so the saved att is fully defined.
// Pre-softmax scores are not kept: stage 1 writes them into att, and
// stage 2 replaces them in place (backward needs only att, §6.2).
//
// Structure: three explicit stages per (b, h), each over all rows before
// the next starts:
//   stage 1 scores (Q K^T * scale), stage 2 row softmax, stage 3 att * V.
// This mirrors the Phase B CUDA design (batched GEMM, masked softmax,
// batched GEMM).
//
// SERIAL COMPUTES ONLY THE CAUSAL LOWER TRIANGLE: stages 1-3 visit t2 <= t
// only, about half the work of a full T x T tile. A CUDA kernel may compute
// the full tile and mask it; benchmark comparisons must keep this in mind
// (see docs/PROFILE.md).
//
// Numerics: the row max is subtracted before exp. Each row's exp sum is
// accumulated in double, and each probability is formed as e / sum in
// double, then rounded once. Score dot products use LANES independent
// partial sums (as in matmul_cpu.cpp), so they vectorize without
// -ffast-math. Stage 3 accumulates whole V rows in order of t2 (an axpy per
// key, vectorized across the head dimension).
//
// BACKWARD (ACCUMULATES into dqkv, same layout as qkv). With dY = dout's
// head slice, A = att, scale = 1/sqrt(hs), per (b, h), causal entries only:
//   stage A  datt[t,t2] = dY(t) . V(t2)                  -> scratch
//            dV(t2)    += sum_{t >= t2} A[t,t2] dY(t)
//   stage B  dpre[t,t2] = A[t,t2] (datt[t,t2] - sum_{t2'<=t} A[t,t2'] datt[t,t2'])
//            (softmax backward, in place in scratch; the row sum in double)
//   stage C  dQ(t)  += scale sum_{t2<=t} dpre[t,t2] K(t2)
//            dK(t2) += scale sum_{t>=t2} dpre[t,t2] Q(t)
// Same three-stage, lower-triangle-only structure as forward (CUDA: two
// batched GEMMs around a softmax-backward kernel). Only att is needed from
// the forward pass; pre-softmax scores are never recomputed. The scratch
// region for (b, h) is the T x T block at the same offset as att's. Only
// its lower triangle is written and read, so its prior contents never
// matter, and they are undefined on return.
// Vectorization: datt uses lane_dot. dV, dQ and dK are axpys over the head
// dimension (independent elements). Each gradient element is summed in
// ascending t (dV, dK) or t2 (dQ) order, so results are deterministic.
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

namespace {

constexpr size_t LANES = 32 / sizeof(real_t);

// Dot product of two contiguous vectors with LANES partial sums.
inline real_t lane_dot(const real_t* a, const real_t* b, size_t n) {
    real_t acc[LANES] = {};
    const size_t nv = n / LANES * LANES;
    for (size_t k = 0; k < nv; k += LANES)
        for (size_t l = 0; l < LANES; ++l) acc[l] += a[k + l] * b[k + l];
    real_t s = 0;
    for (size_t l = 0; l < LANES; ++l) s += acc[l];
    for (size_t k = nv; k < n; ++k) s += a[k] * b[k];
    return s;
}

}  // namespace

void attention_forward(real_t* out, real_t* att, const real_t* qkv, int B, int T, int C, int NH) {
    const size_t Ts = static_cast<size_t>(T);
    const size_t Cs = static_cast<size_t>(C);
    const size_t C3 = 3 * Cs;
    const size_t hs = Cs / static_cast<size_t>(NH);
    const real_t scale = static_cast<real_t>(1.0 / std::sqrt(static_cast<double>(hs)));

    for (int b = 0; b < B; ++b) {
        const real_t* qkv_b = qkv + static_cast<size_t>(b) * Ts * C3;
        real_t* out_b = out + static_cast<size_t>(b) * Ts * Cs;
        for (int h = 0; h < NH; ++h) {
            const size_t hoff = static_cast<size_t>(h) * hs;
            real_t* A = att + (static_cast<size_t>(b) * static_cast<size_t>(NH) + static_cast<size_t>(h)) * Ts * Ts;

            // Stage 1: scaled scores, lower triangle; upper triangle zeroed.
            for (size_t t = 0; t < Ts; ++t) {
                const real_t* q = qkv_b + t * C3 + hoff;
                real_t* row = A + t * Ts;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t* k = qkv_b + t2 * C3 + Cs + hoff;
                    row[t2] = lane_dot(q, k, hs) * scale;
                }
                for (size_t t2 = t + 1; t2 < Ts; ++t2) row[t2] = 0;
            }

            // Stage 2: row softmax over t2 <= t (max-subtracted, double sum).
            for (size_t t = 0; t < Ts; ++t) {
                real_t* row = A + t * Ts;
                real_t mx = row[0];
                for (size_t t2 = 1; t2 <= t; ++t2) mx = row[t2] > mx ? row[t2] : mx;
                double sum = 0.0;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t e = std::exp(row[t2] - mx);
                    row[t2] = e;
                    sum += static_cast<double>(e);
                }
                for (size_t t2 = 0; t2 <= t; ++t2)
                    row[t2] = static_cast<real_t>(static_cast<double>(row[t2]) / sum);
            }

            // Stage 3: out = att * V, causal rows only.
            for (size_t t = 0; t < Ts; ++t) {
                const real_t* row = A + t * Ts;
                real_t* o = out_b + t * Cs + hoff;
                for (size_t i = 0; i < hs; ++i) o[i] = 0;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t a = row[t2];
                    const real_t* v = qkv_b + t2 * C3 + 2 * Cs + hoff;
                    for (size_t i = 0; i < hs; ++i) o[i] += a * v[i];
                }
            }
        }
    }
}

void attention_backward(real_t* dqkv, real_t* scratch, const real_t* dout, const real_t* qkv, const real_t* att, int B,
                        int T, int C, int NH) {
    const size_t Ts = static_cast<size_t>(T);
    const size_t Cs = static_cast<size_t>(C);
    const size_t C3 = 3 * Cs;
    const size_t hs = Cs / static_cast<size_t>(NH);
    const real_t scale = static_cast<real_t>(1.0 / std::sqrt(static_cast<double>(hs)));

    for (int b = 0; b < B; ++b) {
        const real_t* qkv_b = qkv + static_cast<size_t>(b) * Ts * C3;
        real_t* dqkv_b = dqkv + static_cast<size_t>(b) * Ts * C3;
        const real_t* dout_b = dout + static_cast<size_t>(b) * Ts * Cs;
        for (int h = 0; h < NH; ++h) {
            const size_t hoff = static_cast<size_t>(h) * hs;
            const size_t blk = (static_cast<size_t>(b) * static_cast<size_t>(NH) + static_cast<size_t>(h)) * Ts * Ts;
            const real_t* A = att + blk;
            real_t* D = scratch + blk;  // datt, then dpre (lower triangle only)

            // Stage A: datt = dY V^T; dV += A^T dY.
            for (size_t t = 0; t < Ts; ++t) {
                const real_t* dy = dout_b + t * Cs + hoff;
                const real_t* arow = A + t * Ts;
                real_t* drow = D + t * Ts;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t* v = qkv_b + t2 * C3 + 2 * Cs + hoff;
                    real_t* dv = dqkv_b + t2 * C3 + 2 * Cs + hoff;
                    drow[t2] = lane_dot(dy, v, hs);
                    const real_t a = arow[t2];
                    for (size_t i = 0; i < hs; ++i) dv[i] += a * dy[i];
                }
            }

            // Stage B: softmax backward in place, dpre = A * (datt - sum(A * datt)).
            for (size_t t = 0; t < Ts; ++t) {
                const real_t* arow = A + t * Ts;
                real_t* drow = D + t * Ts;
                double dotsum = 0.0;
                for (size_t t2 = 0; t2 <= t; ++t2)
                    dotsum += static_cast<double>(arow[t2]) * static_cast<double>(drow[t2]);
                const real_t ds = static_cast<real_t>(dotsum);
                for (size_t t2 = 0; t2 <= t; ++t2) drow[t2] = arow[t2] * (drow[t2] - ds);
            }

            // Stage C: dQ += scale dpre K; dK += scale dpre^T Q.
            for (size_t t = 0; t < Ts; ++t) {
                const real_t* drow = D + t * Ts;
                const real_t* q = qkv_b + t * C3 + hoff;
                real_t* dq = dqkv_b + t * C3 + hoff;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t g = drow[t2] * scale;
                    const real_t* k = qkv_b + t2 * C3 + Cs + hoff;
                    real_t* dk = dqkv_b + t2 * C3 + Cs + hoff;
                    for (size_t i = 0; i < hs; ++i) {
                        dq[i] += g * k[i];
                        dk[i] += g * q[i];
                    }
                }
            }
        }
    }
}

}  // namespace ops
}  // namespace tslm
