// layernorm_cpu.cpp: LayerNorm forward and backward over the last dimension, CPU (ROADMAP §8, §6.3).
//
// Shapes (row-major): inp, out [N,C]; gamma, beta [C]; mean, rstd [N].
// Math (eps = 1e-5, biased variance):
//   mean[n] = (1/C) * sum_c x[n,c]
//   var[n]  = (1/C) * sum_c (x[n,c] - mean[n])^2
//   rstd[n] = 1 / sqrt(var[n] + eps)
//   out[n,c] = (x[n,c] - mean[n]) * rstd[n] * gamma[c] + beta[c]
// Semantics: forward OVERWRITES out, mean and rstd. mean/rstd are saved for
// layernorm_backward.
//
// Backward (ACCUMULATES, +=), with xhat = (x - mean) * rstd and
// g = dout * gamma, derived from out = xhat * gamma + beta:
//   dinp[n,c]  += rstd * (g - mean_c(g) - xhat * mean_c(g * xhat))
//   dgamma[c]  += sum_n dout[n,c] * xhat[n,c]
//   dbeta[c]   += sum_n dout[n,c]
// The two row means are accumulated in double. xhat is recomputed from the
// saved (rounded) mean/rstd, exactly as the forward pass used them.
// dgamma/dbeta are summed over rows in row order, in real_t (they are
// slices of the flat grads buffer; the op cannot allocate a double scratch).
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

void layernorm_backward(real_t* dinp, real_t* dgamma, real_t* dbeta, const real_t* dout, const real_t* inp,
                        const real_t* gamma, const real_t* mean, const real_t* rstd, int N, int C) {
    const size_t Cs = static_cast<size_t>(C);
    const double invC = 1.0 / static_cast<double>(C);
    for (int n = 0; n < N; ++n) {
        const real_t* x = inp + static_cast<size_t>(n) * Cs;
        const real_t* d = dout + static_cast<size_t>(n) * Cs;
        real_t* dx = dinp + static_cast<size_t>(n) * Cs;
        const real_t mr = mean[n];
        const real_t rs = rstd[n];

        // Row means of g = dout*gamma and g*xhat, accumulated in double.
        double sum_g = 0.0, sum_gx = 0.0;
        for (size_t c = 0; c < Cs; ++c) {
            const real_t xhat = (x[c] - mr) * rs;
            const real_t g = d[c] * gamma[c];
            sum_g += static_cast<double>(g);
            sum_gx += static_cast<double>(g) * static_cast<double>(xhat);
        }
        const real_t mean_g = static_cast<real_t>(sum_g * invC);
        const real_t mean_gx = static_cast<real_t>(sum_gx * invC);

        for (size_t c = 0; c < Cs; ++c) {
            const real_t xhat = (x[c] - mr) * rs;
            const real_t g = d[c] * gamma[c];
            dx[c] += rs * (g - mean_g - xhat * mean_gx);
            dgamma[c] += d[c] * xhat;
            dbeta[c] += d[c];
        }
    }
}

}  // namespace ops
}  // namespace tslm
