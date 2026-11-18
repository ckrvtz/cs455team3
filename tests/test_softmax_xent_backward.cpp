// test_softmax_xent_backward.cpp: softmax_xent_backward (ROADMAP P3.4).
// DoD: finite-difference test.
//
// The function differentiated is the MEAN loss L = (1/N) sum_n losses[n]
// produced by softmax_xent_forward. Rows are independent: perturbing logit
// (n, v) changes only losses[n], so the finite difference evaluates only
// that row's term losses[n]/N. This is mathematically identical to
// differencing L, and keeps the other rows' rounding noise out.
//   REAL=double build: §10 criterion (eps 1e-5, rel err < 1e-6, floor 1e-8).
//   float build: smoke check (eps 1e-2, tol 2e-2, floor 1e-2).
#include "test.h"

#include "tslm/common.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tslm;

namespace {

constexpr double EPS = REAL_IS_DOUBLE ? 1e-5 : 1e-2;
constexpr double TOL = REAL_IS_DOUBLE ? 1e-6 : 2e-2;
constexpr double FLOOR = REAL_IS_DOUBLE ? 1e-8 : 1e-2;

struct Case {
    int N, V;
    std::vector<real_t> logits;
    std::vector<int32_t> targets;
};

Case make(int N, int V, uint64_t seed, double sd) {
    Rng rng(seed, 4);
    Case c{N, V, std::vector<real_t>(size_t(N) * V), std::vector<int32_t>(size_t(N))};
    for (auto& x : c.logits) x = static_cast<real_t>(rng.normal(0.0, sd));
    for (auto& t : c.targets) t = static_cast<int32_t>(rng.uniform_int(uint64_t(V)));
    return c;
}

std::vector<real_t> grad(const Case& c, std::vector<real_t>* probs_out = nullptr) {
    std::vector<real_t> probs(c.logits.size()), losses(size_t(c.N)), d(c.logits.size(), 0);
    ops::softmax_xent_forward(probs.data(), losses.data(), c.logits.data(), c.targets.data(), c.N, c.V);
    ops::softmax_xent_backward(d.data(), probs.data(), c.targets.data(), c.N, c.V);
    if (probs_out) *probs_out = probs;
    return d;
}

}  // namespace

TEST(finite_differences) {
    struct S { int N, V; double sd; };
    // Well-conditioned logits only. With very spread logits the row loss is
    // large (target improbable) while many gradients are tiny: the FD
    // rounding noise ~ u*|loss|/eps (~1e-11 at loss ~8, eps 1e-5) then exceeds
    // 1e-6 of those gradients in any precision. That regime is checked
    // exactly in wide_range_exact instead.
    const S shapes[] = {{1, 2, 1.0}, {6, 13, 1.0}, {5, 100, 1.5}, {3, 7, 0.1}, {4, 31, 1.0}};
    double worst = 0, worst_grad = 0, worst_loss = 0;
    int bad = 0;
    for (const S& s : shapes) {
        Case c = make(s.N, s.V, uint64_t(s.N * 1000 + s.V), s.sd);
        const std::vector<real_t> g = grad(c);
        std::vector<real_t> probs(size_t(s.V)), loss1(1);
        for (int n = 0; n < s.N; ++n) {
            real_t* row = &c.logits[size_t(n) * s.V];
            auto row_loss = [&] {  // losses[n] / N
                ops::softmax_xent_forward(probs.data(), loss1.data(), row, &c.targets[size_t(n)], 1, s.V);
                return double(loss1[0]) / s.N;
            };
            for (int v = 0; v < s.V; ++v) {
                const real_t orig = row[v];
                const real_t up = static_cast<real_t>(double(orig) + EPS), dn = static_cast<real_t>(double(orig) - EPS);
                row[v] = up;
                const double lp = row_loss();
                row[v] = dn;
                const double lm = row_loss();
                row[v] = orig;
                const double num = (lp - lm) / (double(up) - double(dn));
                const double a = g[size_t(n) * s.V + v];
                const double rel = std::fabs(a - num) / std::fmax(std::fmax(std::fabs(a), std::fabs(num)), FLOOR);
                if (rel > worst) {
                    worst = rel;
                    worst_grad = a;
                    worst_loss = row_loss() * s.N;
                }
                if (!(rel < TOL) && bad++ < 3) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf), "N=%d V=%d [%d,%d]: analytic %.10g numeric %.10g rel %.3g", s.N,
                                  s.V, n, v, a, num, rel);
                    test::report(__FILE__, __LINE__, buf);
                }
            }
        }
    }
    std::printf("    FD over all logits: worst rel err %.2e (tol %.0e), at gradient %.3e with row loss %.2f\n",
                worst, TOL, worst_grad, worst_loss);
    EXPECT_EQ(bad, 0);
}

// Widely spread logits (where finite differences cannot resolve the tiny
// gradients): compare with (p - onehot)/N computed independently in double
// from the logits, entry by entry. p itself is accurate to a few ulps of
// real_t, so the relative error is limited by real_t precision.
TEST(wide_range_exact) {
    const double tol = REAL_IS_DOUBLE ? 1e-12 : 1e-5;
    double worst = 0;
    for (double sd : {3.0, 6.0}) {
        const Case c = make(8, 31, uint64_t(sd * 10), sd);
        const std::vector<real_t> g = grad(c);
        for (int n = 0; n < c.N; ++n) {
            const real_t* l = &c.logits[size_t(n) * c.V];
            double m = l[0], S = 0;
            for (int v = 1; v < c.V; ++v) m = l[v] > m ? l[v] : m;
            for (int v = 0; v < c.V; ++v) S += std::exp(double(l[v]) - m);
            for (int v = 0; v < c.V; ++v) {
                const double p = std::exp(double(l[v]) - m) / S;
                const double want = (v == c.targets[size_t(n)] ? p - 1.0 : p) / c.N;
                const double got = g[size_t(n) * c.V + v];
                const double rel = std::fabs(got - want) / std::fmax(std::fabs(want), 1e-300);
                worst = rel > worst ? rel : worst;
                // Entries below the real_t subnormal range are allowed to flush to 0.
                if (!(rel < tol) && !(std::fabs(want) < 1e-36 && got == 0))
                    test::report(__FILE__, __LINE__, "wide-range gradient mismatch, sd " + std::to_string(sd));
            }
        }
    }
    std::printf("    wide-range logits (sd 3, 6): worst rel err %.2e (tol %.0e)\n", worst, tol);
}

// Hand case (values from the §6.3 formulas, computed independently in double).
TEST(hand_case) {
    const real_t logits[2 * 3] = {1, 2, 3,   1, 2, 3};
    const int32_t targets[2] = {2, 0};
    real_t probs[6], losses[2], d[6] = {0, 0, 0, 0, 0, 0};
    ops::softmax_xent_forward(probs, losses, logits, targets, 2, 3);
    ops::softmax_xent_backward(d, probs, targets, 2, 3);
    const double p0 = 0.09003057317038046, p1 = 0.24472847105479764, p2 = 0.6652409557748218;
    const double want[6] = {p0 / 2, p1 / 2, (p2 - 1) / 2,   (p0 - 1) / 2, p1 / 2, p2 / 2};  // N = 2
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(d[i], want[i], 1e-7);
}

// Each row of (p - onehot) sums to ~0; target entry < 0, others > 0.
TEST(row_properties) {
    const Case c = make(20, 50, 7, 2.0);
    const std::vector<real_t> g = grad(c);
    for (int n = 0; n < c.N; ++n) {
        double s = 0;
        for (int v = 0; v < c.V; ++v) {
            const real_t x = g[size_t(n) * c.V + v];
            s += x;
            if (v == c.targets[size_t(n)]) ASSERT_TRUE(x < 0);
            else ASSERT_TRUE(x >= 0);
        }
        EXPECT_NEAR(s, 0.0, 1e-7);
    }
}

// V = 10000 uniform logits: dlogits = (1/V - onehot) / N.
TEST(uniform_large_vocab) {
    const int N = 3, V = 10000;
    Case c{N, V, std::vector<real_t>(size_t(N) * V, real_t(0.25)), {5, 0, V - 1}};
    const std::vector<real_t> g = grad(c);
    for (int n = 0; n < N; ++n)
        for (int v = 0; v < V; ++v) {
            const double want = (1.0 / V - (v == c.targets[size_t(n)] ? 1.0 : 0.0)) / N;
            ASSERT_NEAR(g[size_t(n) * V + v], want, 1e-7 * std::fmax(1.0, std::fabs(want)) + 1e-12);
        }
}

// Target probability near 1: (p_t - 1)/N stays accurate (no cancellation).
TEST(confident_target_accuracy) {
    const real_t logits[3] = {0, -12, -12};  // p_target ~ 1 - 1.2e-5
    const int32_t target = 0;
    real_t probs[3], loss, d[3] = {0, 0, 0};
    ops::softmax_xent_forward(probs, &loss, logits, &target, 1, 3);
    ops::softmax_xent_backward(d, probs, &target, 1, 3);
    const double e = std::exp(-12.0), want = -2 * e / (1 + 2 * e);  // p_t - 1, exact
    // Relative accuracy of the small gradient is limited only by how
    // accurately p_t itself is stored (rounding of real_t near 1).
    const double tol = REAL_IS_DOUBLE ? 1e-9 : 1e-2;
    std::printf("    p_t - 1: got %.9e, exact %.9e\n", double(d[0]), want);
    EXPECT_NEAR(double(d[0]) / want, 1.0, tol);
}

TEST(accumulates_exactly) {
    const Case c = make(9, 40, 3, 1.0);
    std::vector<real_t> probs;
    const std::vector<real_t> once = grad(c, &probs);
    std::vector<real_t> twice = once;
    ops::softmax_xent_backward(twice.data(), probs.data(), c.targets.data(), c.N, c.V);
    // Each element is a single += per call, so x + x == 2x exactly.
    for (size_t i = 0; i < once.size(); ++i) ASSERT_EQ(twice[i], real_t(2) * once[i]);
}

TEST_MAIN()
