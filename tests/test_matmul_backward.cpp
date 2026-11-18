// test_matmul_backward.cpp: matmul_backward (ROADMAP P3.2).
// DoD: finite-difference test; verifies += by calling twice and checking 2x.
//
// 1. Finite differences (small odd shapes, every entry of inp, W, bias):
//    L = sum w * out (double), dL/dout = w, central differences of L.
//    REAL=double build: §10 criterion (eps 1e-5, rel err < 1e-6, floor 1e-8).
//    float build: smoke check only (eps 1e-2, tol 2e-2, floor 1e-2).
// 2. Blocked-path correctness on larger and block-boundary shapes (NB=128,
//    KC=256, OB=64, 4-row tiles) against a naive double reference, with the
//    rigorous order-independent bound |got - ref| <= (K + 2) u sum|terms|,
//    where K is the number of summed terms.
// 3. Accumulate semantics: second call doubles; prefilled buffers get
//    exactly the fresh gradient added; null dbias works.
#include "test.h"

#include "tslm/common.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace tslm;

namespace {

constexpr double EPS = REAL_IS_DOUBLE ? 1e-5 : 1e-2;
constexpr double TOL = REAL_IS_DOUBLE ? 1e-6 : 2e-2;
constexpr double FLOOR = REAL_IS_DOUBLE ? 1e-8 : 1e-2;

std::vector<real_t> randn(size_t n, Rng& rng) {
    std::vector<real_t> v(n);
    for (auto& x : v) x = static_cast<real_t>(rng.normal());
    return v;
}

struct Problem {
    int N, C, OC;
    std::vector<real_t> inp, W, bias, w;  // w = dL/dout
};

Problem make(int N, int C, int OC, uint64_t seed) {
    Rng rng(seed, 1);
    Problem p{N, C, OC, randn(size_t(N) * C, rng), randn(size_t(OC) * C, rng), randn(size_t(OC), rng),
              randn(size_t(N) * OC, rng)};
    return p;
}

double loss(Problem& p, bool with_bias) {
    std::vector<real_t> out(size_t(p.N) * p.OC);
    ops::matmul_forward(out.data(), p.inp.data(), p.W.data(), with_bias ? p.bias.data() : nullptr, p.N, p.C, p.OC);
    double s = 0;
    for (size_t i = 0; i < out.size(); ++i) s += double(p.w[i]) * double(out[i]);
    return s;
}

double fd_check(const char* what, Problem& pr, std::vector<real_t>& param, const std::vector<real_t>& g,
                bool with_bias) {
    double worst = 0;
    int bad = 0;
    for (size_t i = 0; i < param.size(); ++i) {
        const real_t orig = param[i];
        const real_t up = static_cast<real_t>(double(orig) + EPS), dn = static_cast<real_t>(double(orig) - EPS);
        param[i] = up;
        const double lp = loss(pr, with_bias);
        param[i] = dn;
        const double lm = loss(pr, with_bias);
        param[i] = orig;
        const double num = (lp - lm) / (double(up) - double(dn));
        const double a = g[i];
        const double rel = std::fabs(a - num) / std::fmax(std::fmax(std::fabs(a), std::fabs(num)), FLOOR);
        worst = rel > worst ? rel : worst;
        if (!(rel < TOL) && bad++ < 3) {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "%s[%zu] (N=%d C=%d OC=%d): analytic %.10g numeric %.10g rel %.3g", what,
                          i, pr.N, pr.C, pr.OC, a, num, rel);
            test::report(__FILE__, __LINE__, buf);
        }
    }
    return worst;
}

struct Grads {
    std::vector<real_t> dinp, dW, dbias;
};

Grads backward(const Problem& p, bool with_bias) {
    Grads g{std::vector<real_t>(p.inp.size(), 0), std::vector<real_t>(p.W.size(), 0),
            std::vector<real_t>(p.bias.size(), 0)};
    ops::matmul_backward(g.dinp.data(), g.dW.data(), with_bias ? g.dbias.data() : nullptr, p.w.data(), p.inp.data(),
                         p.W.data(), p.N, p.C, p.OC);
    return g;
}

}  // namespace

TEST(finite_differences_small_shapes) {
    struct S { int N, C, OC; };
    const S shapes[] = {{1, 1, 1}, {3, 5, 7}, {5, 3, 4}, {7, 9, 6}, {4, 4, 4}, {9, 2, 11}};
    double worst[3] = {0, 0, 0};
    for (const S& s : shapes) {
        for (int wb = 0; wb <= 1; ++wb) {
            Problem p = make(s.N, s.C, s.OC, uint64_t(s.N * 100 + s.C * 10 + s.OC));
            const Grads g = backward(p, wb);
            const double a = fd_check("dinp", p, p.inp, g.dinp, wb);
            const double b = fd_check("dW", p, p.W, g.dW, wb);
            worst[0] = a > worst[0] ? a : worst[0];
            worst[1] = b > worst[1] ? b : worst[1];
            if (wb) {
                const double c = fd_check("dbias", p, p.bias, g.dbias, wb);
                worst[2] = c > worst[2] ? c : worst[2];
            } else {
                for (real_t v : g.dbias) EXPECT_EQ(v, real_t(0));  // null dbias: untouched
            }
        }
    }
    std::printf("    FD worst rel err: dinp %.2e, dW %.2e, dbias %.2e (tol %.0e)\n", worst[0], worst[1], worst[2],
                TOL);
}

// Blocked-path correctness vs naive double reference on bigger shapes.
TEST(blocked_vs_reference) {
    struct S { int N, C, OC; };
    const S shapes[] = {
        {130, 257, 65}, {129, 300, 63}, {127, 255, 129}, {5, 513, 3}, {256, 64, 100},
        {1, 1024, 1024}, {1024, 7, 9}, {64, 100, 7}, {7, 64, 256}, {100, 1, 64},
    };
    const double u = std::numeric_limits<real_t>::epsilon() / 2.0;
    int bad = 0;
    for (const S& s : shapes) {
        const Problem p = make(s.N, s.C, s.OC, uint64_t(s.N + s.C + s.OC));
        const Grads g = backward(p, true);
        const size_t N = size_t(s.N), C = size_t(s.C), OC = size_t(s.OC);
        auto check = [&](const char* what, double got, double ref, double mag, size_t K) {
            if (!(std::fabs(got - ref) <= double(K + 2) * u * mag + 1e-30) && bad++ < 3) {
                char buf[200];
                std::snprintf(buf, sizeof(buf), "%s N=%d C=%d OC=%d: got %.9g ref %.9g", what, s.N, s.C, s.OC, got,
                              ref);
                test::report(__FILE__, __LINE__, buf);
            }
        };
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < C; ++c) {
                double ref = 0, mag = 0;
                for (size_t o = 0; o < OC; ++o) {
                    const double t = double(p.w[n * OC + o]) * double(p.W[o * C + c]);
                    ref += t;
                    mag += std::fabs(t);
                }
                check("dinp", g.dinp[n * C + c], ref, mag, OC);
            }
        for (size_t o = 0; o < OC; ++o) {
            for (size_t c = 0; c < C; ++c) {
                double ref = 0, mag = 0;
                for (size_t n = 0; n < N; ++n) {
                    const double t = double(p.w[n * OC + o]) * double(p.inp[n * C + c]);
                    ref += t;
                    mag += std::fabs(t);
                }
                check("dW", g.dW[o * C + c], ref, mag, N);
            }
            double ref = 0, mag = 0;
            for (size_t n = 0; n < N; ++n) {
                ref += p.w[n * OC + o];
                mag += std::fabs(double(p.w[n * OC + o]));
            }
            check("dbias", g.dbias[o], ref, mag, N);
        }
    }
    EXPECT_EQ(bad, 0);
}

// DoD: calling backward twice accumulates (2x); prefilled buffers get the
// fresh gradient added.
TEST(accumulates) {
    const Problem p = make(37, 70, 45, 9);
    const Grads once = backward(p, true);
    Grads twice = once;
    ops::matmul_backward(twice.dinp.data(), twice.dW.data(), twice.dbias.data(), p.w.data(), p.inp.data(), p.W.data(),
                         p.N, p.C, p.OC);
    auto doubled = [&](const char* what, const std::vector<real_t>& a, const std::vector<real_t>& b) {
        // b = a + a' where a' recomputes a in the same order from a different
        // starting value; equal to 2a up to rounding of the accumulation.
        for (size_t i = 0; i < a.size(); ++i) {
            const double want = 2.0 * double(a[i]);
            if (!(std::fabs(double(b[i]) - want) <= 1e-5 * std::fmax(1.0, std::fabs(want)))) {
                test::report(__FILE__, __LINE__, std::string(what) + ": second call did not double the gradient");
                return;
            }
        }
    };
    doubled("dinp", once.dinp, twice.dinp);
    doubled("dW", once.dW, twice.dW);
    doubled("dbias", once.dbias, twice.dbias);

    // Prefilled with a constant: result - constant matches the fresh gradient.
    Grads pre{std::vector<real_t>(p.inp.size(), real_t(3)), std::vector<real_t>(p.W.size(), real_t(3)),
              std::vector<real_t>(p.bias.size(), real_t(3))};
    ops::matmul_backward(pre.dinp.data(), pre.dW.data(), pre.dbias.data(), p.w.data(), p.inp.data(), p.W.data(), p.N,
                         p.C, p.OC);
    for (size_t i = 0; i < once.dW.size(); ++i)
        ASSERT_NEAR(double(pre.dW[i]) - 3.0, double(once.dW[i]), 1e-5 * std::fmax(1.0, std::fabs(double(once.dW[i]))));
}

// Exact integer case (every summation order is exact).
TEST(exact_integer_case) {
    const real_t inp[2 * 3] = {1, 2, 3, 4, 5, 6};
    const real_t W[2 * 3] = {1, 0, -1, 2, 1, 0};
    const real_t dout[2 * 2] = {1, 2, -1, 3};
    real_t dinp[6] = {0}, dW[6] = {0}, db[2] = {0};
    ops::matmul_backward(dinp, dW, db, dout, inp, W, 2, 3, 2);
    // dinp = dout * W: row0 = 1*[1,0,-1] + 2*[2,1,0] = [5,2,-1]; row1 = -1*[1,0,-1] + 3*[2,1,0] = [5,3,1]
    const real_t want_dinp[6] = {5, 2, -1, 5, 3, 1};
    // dW = dout^T * inp: row0 = 1*[1,2,3] - 1*[4,5,6] = [-3,-3,-3]; row1 = 2*[1,2,3] + 3*[4,5,6] = [14,19,24]
    const real_t want_dW[6] = {-3, -3, -3, 14, 19, 24};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(dinp[i], want_dinp[i]);
        EXPECT_EQ(dW[i], want_dW[i]);
    }
    EXPECT_EQ(db[0], real_t(0));
    EXPECT_EQ(db[1], real_t(5));
}

TEST_MAIN()
