// optim_cpu.cpp: optimizer and gradient-clipping ops, CPU (ROADMAP §8, §2.3).
//
// sumsq_reduce(x, n) = sum_i x[i]^2, accumulated in double in index order
//   (deterministic). The global gradient norm is sqrt(sumsq_reduce(grads,
//   num_params_padded)); padding is always 0, so it adds nothing.
// scale_inplace(x, s, n): x[i] *= s. Used for clipping (s = clip / norm).
// adamw_step: decoupled AdamW (Loshchilov & Hutter), per element, in place:
//   m = b1 m + (1 - b1) g
//   v = b2 v + (1 - b2) g^2
//   m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)          (t = step, 1-based)
//   p -= lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )
//   Algebraically identical to PyTorch's AdamW, which first scales p by
//   (1 - lr wd), then subtracts lr m_hat / (sqrt(v_hat) + eps). Both use the
//   p from before the step in the decay term.
// Semantics: adamw_step and scale_inplace modify their arrays in place. The
// bias corrections 1/(1 - b^t) are computed once per call in double, then
// rounded to real_t. Every element is independent, so the loops vectorize
// (sqrt and division are plain IEEE operations, allowed without
// -ffast-math).
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

double sumsq_reduce(const real_t* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i]);
        s += v * v;
    }
    return s;
}

void scale_inplace(real_t* x, real_t s, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] *= s;
}

void adamw_step(real_t* p, const real_t* g, real_t* m, real_t* v, size_t n, real_t lr, real_t beta1, real_t beta2,
                real_t eps, real_t weight_decay, int step) {
    const double t = static_cast<double>(step);
    const real_t inv_bc1 = static_cast<real_t>(1.0 / (1.0 - std::pow(static_cast<double>(beta1), t)));
    const real_t inv_bc2 = static_cast<real_t>(1.0 / (1.0 - std::pow(static_cast<double>(beta2), t)));
    const real_t one = static_cast<real_t>(1);
    const real_t b1c = one - beta1, b2c = one - beta2;
    for (size_t i = 0; i < n; ++i) {
        const real_t gi = g[i];
        const real_t mi = beta1 * m[i] + b1c * gi;
        const real_t vi = beta2 * v[i] + b2c * gi * gi;
        m[i] = mi;
        v[i] = vi;
        const real_t m_hat = mi * inv_bc1;
        const real_t v_hat = vi * inv_bc2;
        p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
    }
}

}  // namespace ops
}  // namespace tslm
