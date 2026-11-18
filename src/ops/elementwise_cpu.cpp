// elementwise_cpu.cpp: GELU and residual-add forward and backward, CPU (ROADMAP §8, §6.3).
//
// Shapes: flat arrays of n elements (any tensor shape, contiguous).
// GELU (tanh approximation, as in GPT-2):
//   out[i] = 0.5 * x * (1 + tanh(k0 * (x + k1 * x^3))),  k0 = sqrt(2/pi), k1 = 0.044715
// Residual: out[i] = a[i] + b[i]
// Semantics: forward OVERWRITES out. Computed in real_t (float in release
// builds, double in the gradcheck build).
//
// Evaluation form: the literal 1 + tanh(u) loses most of its digits to
// cancellation when tanh(u) is close to -1 (x below about -4). In double,
// that limited GELU and its derivative near x = -6 to about 1e-7 relative
// accuracy, which the finite-difference check exposed. The exact identity
// 1 + tanh(u) = 2 sigma(2u), with sigma(z) = 1/(1+exp(-z)), avoids it:
//   out = x * sigma(2u),  u = k0 (x + k1 x^3)
// This is the same function as the tanh form above (ROADMAP §6.3), only
// evaluated without cancellation. It never produces NaN: for large |u| the
// exp saturates to 0 or +inf and sigma becomes exactly 0 or 1.
//
// Backward (ACCUMULATES, +=), the exact derivative of that function:
//   s = sigma(2u) = (1 + tanh u)/2,  1 - s = sigma(-2u),  1 - tanh^2 u = 4 s (1 - s)
//   d gelu / dx = s + 2 x s (1 - s) k0 (1 + 3 k1 x^2)
//   dinp[i] += dout[i] * d gelu / dx(inp[i])
// Residual backward: da[i] += dout[i]; db[i] += dout[i].
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

namespace {
// sqrt(2/pi) to double precision; converted to real_t at use.
constexpr double GELU_K0 = 0.7978845608028654;
constexpr double GELU_K1 = 0.044715;
}  // namespace

// sigma(z) = 1 / (1 + exp(-z)). For large |z| the exp overflows to +inf or
// underflows to 0, and the result is exactly 0 or 1: never NaN.
inline real_t sigmoid(real_t z) { return static_cast<real_t>(1) / (static_cast<real_t>(1) + std::exp(-z)); }

void gelu_forward(real_t* out, const real_t* inp, size_t n) {
    const real_t k0 = static_cast<real_t>(GELU_K0);
    const real_t k1 = static_cast<real_t>(GELU_K1);
    const real_t two = static_cast<real_t>(2);
    for (size_t i = 0; i < n; ++i) {
        const real_t x = inp[i];
        const real_t u = k0 * (x + k1 * x * x * x);
        out[i] = x * sigmoid(two * u);  // == 0.5 x (1 + tanh u), without cancellation
    }
}

void gelu_backward(real_t* dinp, const real_t* inp, const real_t* dout, size_t n) {
    const real_t k0 = static_cast<real_t>(GELU_K0);
    const real_t k1 = static_cast<real_t>(GELU_K1);
    const real_t one = static_cast<real_t>(1);
    const real_t two = static_cast<real_t>(2);
    const real_t three = static_cast<real_t>(3);
    for (size_t i = 0; i < n; ++i) {
        const real_t x = inp[i];
        const real_t u = k0 * (x + k1 * x * x * x);
        const real_t s = sigmoid(two * u);    // (1 + tanh u) / 2
        const real_t sm = sigmoid(-two * u);  // 1 - s, computed without cancellation
        const real_t grad = s + two * x * s * sm * k0 * (one + three * k1 * x * x);
        dinp[i] += dout[i] * grad;
    }
}

void residual_forward(real_t* out, const real_t* a, const real_t* b, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void residual_backward(real_t* da, real_t* db, const real_t* dout, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

}  // namespace ops
}  // namespace tslm
