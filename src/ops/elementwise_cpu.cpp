// elementwise_cpu.cpp: GELU and residual-add forward, CPU (ROADMAP §8, §6.3).
//
// Shapes: flat arrays of n elements (any tensor shape, contiguous).
// GELU (tanh approximation, as in GPT-2):
//   out[i] = 0.5 * x * (1 + tanh(k0 * (x + k1 * x^3))),  k0 = sqrt(2/pi), k1 = 0.044715
// Residual: out[i] = a[i] + b[i]
// Semantics: forward OVERWRITES out. Computed in real_t (float in release
// builds, double in the gradcheck build). std::tanh saturates cleanly to
// +-1 for large |x|, so no input clamping is needed.
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

namespace {
// sqrt(2/pi) to double precision; converted to real_t at use.
constexpr double GELU_K0 = 0.7978845608028654;
constexpr double GELU_K1 = 0.044715;
}  // namespace

void gelu_forward(real_t* out, const real_t* inp, size_t n) {
    const real_t k0 = static_cast<real_t>(GELU_K0);
    const real_t k1 = static_cast<real_t>(GELU_K1);
    const real_t half = static_cast<real_t>(0.5);
    const real_t one = static_cast<real_t>(1);
    for (size_t i = 0; i < n; ++i) {
        const real_t x = inp[i];
        const real_t u = k0 * (x + k1 * x * x * x);
        out[i] = half * x * (one + std::tanh(u));
    }
}

void residual_forward(real_t* out, const real_t* a, const real_t* b, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

}  // namespace ops
}  // namespace tslm
