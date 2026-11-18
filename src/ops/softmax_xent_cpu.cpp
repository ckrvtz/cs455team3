// softmax_xent_cpu.cpp: softmax + cross-entropy forward and sum_reduce, CPU
// (ROADMAP §8, §6.3).
//
// Shapes (row-major): logits, probs [N,V]; targets [N] int32 in [0,V);
// losses [N].
// Math, per row n with m = max_v logits[n,v]:
//   e[v]      = exp(logits[n,v] - m)
//   S         = sum_v e[v]                          (accumulated in double)
//   probs[n,v] = e[v] / S                           (in double, rounded once)
//   losses[n] = log(S) - (logits[n,target] - m)     (= -log probs[n,target])
// The loss uses the log-sum-exp form rather than -log(probs[target]), so it
// stays finite and accurate even when probs[target] underflows to 0 in real_t.
// Semantics: OVERWRITES probs and losses. The mean loss is
// sum_reduce(losses, N) / N, computed by the caller.
//
// sum_reduce(x, n) = sum_i x[i], accumulated in double in index order, so
// the result is deterministic.
//
// Performance notes (docs/PROFILE.md): at V = 10,000 the exp and the
// double-precision row sum run as scalar code, because vectorizing either
// would need -ffast-math or a vector math library, which the serial
// baseline does not use. This op is memory/exp-bound and much cheaper than
// the head matmul that produces the logits.
#include "tslm/ops.h"

#include <cmath>

namespace tslm {
namespace ops {

void softmax_xent_forward(real_t* probs, real_t* losses, const real_t* logits, const int32_t* targets, int N,
                          int V) {
    const size_t Vs = static_cast<size_t>(V);
    for (int n = 0; n < N; ++n) {
        const real_t* l = logits + static_cast<size_t>(n) * Vs;
        real_t* p = probs + static_cast<size_t>(n) * Vs;

        real_t m = l[0];
        for (size_t v = 1; v < Vs; ++v) m = l[v] > m ? l[v] : m;

        double S = 0.0;
        for (size_t v = 0; v < Vs; ++v) {
            const real_t e = std::exp(l[v] - m);
            p[v] = e;  // stash e[v]; normalized below
            S += static_cast<double>(e);
        }
        for (size_t v = 0; v < Vs; ++v) p[v] = static_cast<real_t>(static_cast<double>(p[v]) / S);

        const size_t tgt = static_cast<size_t>(targets[n]);
        losses[n] = static_cast<real_t>(std::log(S) - (static_cast<double>(l[tgt]) - static_cast<double>(m)));
    }
}

double sum_reduce(const real_t* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(x[i]);
    return s;
}

}  // namespace ops
}  // namespace tslm
