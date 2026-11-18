// layernorm_cpu.cpp: LayerNorm forward over the last dimension, CPU (ROADMAP §8, §6.3).
//
// Shapes (row-major): inp, out [N,C]; gamma, beta [C]; mean, rstd [N].
// Math (eps = 1e-5, biased variance):
//   mean[n] = (1/C) * sum_c x[n,c]
//   var[n]  = (1/C) * sum_c (x[n,c] - mean[n])^2
//   rstd[n] = 1 / sqrt(var[n] + eps)
//   out[n,c] = (x[n,c] - mean[n]) * rstd[n] * gamma[c] + beta[c]
// Semantics: forward OVERWRITES out, mean and rstd. mean/rstd are saved for
// layernorm_backward (P3.1).
// Numerics: two passes (mean first, then the centered sum of squares) with
// both sums accumulated in double. This avoids the cancellation of the
// one-pass E[x^2] - E[x]^2 form and costs little, since the op is
// memory-bound. mean and rstd are rounded to real_t once, and the
// normalization uses the rounded values, so forward and backward see
// exactly the same saved statistics.
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

namespace {
constexpr double LN_EPS = 1e-5;
}

void layernorm_forward(real_t* out, real_t* mean, real_t* rstd, const real_t* inp,
                       const real_t* gamma, const real_t* beta, int N, int C) {
    const size_t Cs = static_cast<size_t>(C);
    for (int n = 0; n < N; ++n) {
        const real_t* x = inp + static_cast<size_t>(n) * Cs;
        real_t* o = out + static_cast<size_t>(n) * Cs;

        double sum = 0.0;
        for (size_t c = 0; c < Cs; ++c) sum += x[c];
        const double m = sum / static_cast<double>(C);

        double sq = 0.0;
        for (size_t c = 0; c < Cs; ++c) {
            const double d = static_cast<double>(x[c]) - m;
            sq += d * d;
        }
        const double var = sq / static_cast<double>(C);
        const real_t mr = static_cast<real_t>(m);
        const real_t rs = static_cast<real_t>(1.0 / std::sqrt(var + LN_EPS));

        for (size_t c = 0; c < Cs; ++c) o[c] = (x[c] - mr) * rs * gamma[c] + beta[c];
        mean[n] = mr;
        rstd[n] = rs;
    }
}

}  // namespace ops
}  // namespace tslm
