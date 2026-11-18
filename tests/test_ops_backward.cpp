// test_ops_backward.cpp: P3.1 backward ops (embedding, layernorm, gelu, residual).
//
// Method: for a random weight vector w, define the scalar L = sum_i w[i]*out[i]
// (accumulated in double), so dL/dout = w. Each backward op is called with
// dout = w on zeroed gradient buffers, and every resulting gradient entry is
// compared with the central difference (L(p+eps) - L(p-eps)) / (2 eps).
// Relative error: |a - n| / max(|a|, |n|, FLOOR).
//   REAL=double build (the P3.1 DoD, ROADMAP §10 criterion):
//     eps = 1e-5, tolerance 1e-6, FLOOR = 1e-8.
//   float build (smoke check only; float central differences cannot reach
//     1e-6): eps = 1e-2, tolerance 2e-2, FLOOR = 1e-2.
// Also checks accumulate (+=) semantics: a second call on the same buffers
// doubles the gradient.
#include "test.h"

#include "tslm/common.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace tslm;

namespace {

constexpr double EPS = REAL_IS_DOUBLE ? 1e-5 : 1e-2;
constexpr double TOL = REAL_IS_DOUBLE ? 1e-6 : 2e-2;
constexpr double FLOOR = REAL_IS_DOUBLE ? 1e-8 : 1e-2;

std::vector<real_t> randn(size_t n, Rng& rng, double sd = 1.0) {
    std::vector<real_t> v(n);
    for (auto& x : v) x = static_cast<real_t>(rng.normal(0.0, sd));
    return v;
}

double dot(const std::vector<real_t>& w, const std::vector<real_t>& out) {
    double s = 0;
    for (size_t i = 0; i < w.size(); ++i) s += double(w[i]) * double(out[i]);
    return s;
}

// Central-difference check of analytic gradient g for every entry of p.
// loss() must recompute the forward pass from the current p.
double fd_check(const char* what, std::vector<real_t>& p, const std::vector<real_t>& g,
                const std::function<double(size_t)>& loss) {
    double worst = 0;
    int bad = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const real_t orig = p[i];
        p[i] = static_cast<real_t>(double(orig) + EPS);
        const double lp = loss(i);
        p[i] = static_cast<real_t>(double(orig) - EPS);
        const double lm = loss(i);
        p[i] = orig;
        // Use the actually-applied step (it is rounded in the float build).
        const double h = (double(static_cast<real_t>(double(orig) + EPS)) -
                          double(static_cast<real_t>(double(orig) - EPS)));
        const double num = (lp - lm) / h;
        const double a = g[i];
        const double denom = std::fmax(std::fmax(std::fabs(a), std::fabs(num)), FLOOR);
        const double rel = std::fabs(a - num) / denom;
        worst = rel > worst ? rel : worst;
        if (!(rel < TOL)) {
            if (bad < 3) {
                char buf[200];
                std::snprintf(buf, sizeof(buf), "%s[%zu]: analytic %.10g numeric %.10g rel %.3g", what, i, a, num, rel);
                test::report(__FILE__, __LINE__, buf);
            }
            ++bad;
        }
    }
    std::printf("    %-14s %4zu entries, worst rel err %.2e (tol %.0e)\n", what, p.size(), worst, TOL);
    return worst;
}

void expect_doubled(const char* what, const std::vector<real_t>& once, const std::vector<real_t>& twice) {
    for (size_t i = 0; i < once.size(); ++i) {
        const double want = 2.0 * double(once[i]);
        if (!(std::fabs(double(twice[i]) - want) <= 1e-6 * std::fmax(1.0, std::fabs(want)))) {
            test::report(__FILE__, __LINE__, std::string(what) + ": second backward call did not accumulate");
            return;
        }
    }
}

}  // namespace

TEST(embedding_backward_fd) {
    const int B = 2, T = 3, C = 4, V = 5;
    Rng rng(1, 0);
    std::vector<real_t> wte = randn(size_t(V) * C, rng), wpe = randn(size_t(T) * C, rng);
    const int32_t tok[B * T] = {1, 3, 1,  4, 1, 0};  // token 1 repeats; token 2 unused
    const std::vector<real_t> w = randn(size_t(B) * T * C, rng);
    std::vector<real_t> out(w.size());
    auto loss = [&](size_t) {
        ops::embedding_forward(out.data(), tok, wte.data(), wpe.data(), B, T, C);
        return dot(w, out);
    };
    std::vector<real_t> dwte(wte.size(), 0), dwpe(wpe.size(), 0);
    ops::embedding_backward(dwte.data(), dwpe.data(), w.data(), tok, B, T, C);
    fd_check("emb dwte", wte, dwte, loss);
    fd_check("emb dwpe", wpe, dwpe, loss);
    for (int c = 0; c < C; ++c) EXPECT_EQ(dwte[size_t(2) * C + c], real_t(0));  // unused token: no gradient

    std::vector<real_t> dwte2 = dwte, dwpe2 = dwpe;
    ops::embedding_backward(dwte2.data(), dwpe2.data(), w.data(), tok, B, T, C);
    expect_doubled("emb dwte", dwte, dwte2);
    expect_doubled("emb dwpe", dwpe, dwpe2);
}

TEST(layernorm_backward_fd) {
    const int N = 4, C = 8;
    Rng rng(2, 0);
    std::vector<real_t> x = randn(size_t(N) * C, rng, 2.0), gamma = randn(C, rng), beta = randn(C, rng);
    for (int c = 0; c < C; ++c) x[size_t(3) * C + c] = static_cast<real_t>(5.0 + 0.1 * c);  // offset, small spread
    const std::vector<real_t> w = randn(size_t(N) * C, rng);
    std::vector<real_t> out(w.size()), mean(N), rstd(N);
    auto loss = [&](size_t) {
        ops::layernorm_forward(out.data(), mean.data(), rstd.data(), x.data(), gamma.data(), beta.data(), N, C);
        return dot(w, out);
    };
    loss(0);  // saved mean/rstd at the base point
    std::vector<real_t> dx(x.size(), 0), dg(C, 0), db(C, 0);
    ops::layernorm_backward(dx.data(), dg.data(), db.data(), w.data(), x.data(), gamma.data(), mean.data(),
                            rstd.data(), N, C);
    const std::vector<real_t> m0 = mean, r0 = rstd;
    fd_check("ln dinp", x, dx, loss);
    fd_check("ln dgamma", gamma, dg, loss);
    fd_check("ln dbeta", beta, db, loss);

    std::vector<real_t> dx2 = dx, dg2 = dg, db2 = db;
    ops::layernorm_backward(dx2.data(), dg2.data(), db2.data(), w.data(), x.data(), gamma.data(), m0.data(),
                            r0.data(), N, C);
    expect_doubled("ln dinp", dx, dx2);
    expect_doubled("ln dgamma", dg, dg2);
    expect_doubled("ln dbeta", db, db2);
}

TEST(gelu_backward_fd) {
    const size_t n = 61;
    std::vector<real_t> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<real_t>(-6.0 + 0.2 * double(i) + 0.013);  // avoid exact 0
    Rng rng(3, 0);
    const std::vector<real_t> w = randn(n, rng);
    std::vector<real_t> out(n);
    // Elementwise op: perturbing x[i] changes only out[i], so L's dependence
    // on x[i] is exactly w[i]*out[i]. Evaluating only that term keeps the
    // other 60 terms' rounding noise (~1e-10 after dividing by 2 eps) out
    // of the tail entries, where gelu' is ~1e-9.
    auto loss = [&](size_t i) {
        ops::gelu_forward(out.data(), x.data(), n);
        return double(w[i]) * double(out[i]);
    };
    std::vector<real_t> dx(n, 0);
    ops::gelu_backward(dx.data(), x.data(), w.data(), n);
    fd_check("gelu dinp", x, dx, loss);

    // Hand values: gelu'(0) = 0.5; gelu'(x) -> 1 for large x, -> 0 for very negative x.
    const real_t pts[3] = {0, 20, -20};
    const real_t ones[3] = {1, 1, 1};
    real_t d[3] = {0, 0, 0};
    ops::gelu_backward(d, pts, ones, 3);
    EXPECT_NEAR(d[0], 0.5, 1e-7);
    EXPECT_NEAR(d[1], 1.0, 1e-7);
    EXPECT_NEAR(d[2], 0.0, 1e-7);

    std::vector<real_t> dx2 = dx;
    ops::gelu_backward(dx2.data(), x.data(), w.data(), n);
    expect_doubled("gelu dinp", dx, dx2);
}

TEST(residual_backward_fd) {
    const size_t n = 20;
    Rng rng(4, 0);
    std::vector<real_t> a = randn(n, rng), b = randn(n, rng);
    const std::vector<real_t> w = randn(n, rng);
    std::vector<real_t> out(n);
    auto loss = [&](size_t) {
        ops::residual_forward(out.data(), a.data(), b.data(), n);
        return dot(w, out);
    };
    std::vector<real_t> da(n, 0), db(n, 0);
    ops::residual_backward(da.data(), db.data(), w.data(), n);
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(da[i], w[i]);  // exactly dout
    fd_check("res da", a, da, loss);
    fd_check("res db", b, db, loss);
    std::vector<real_t> da2 = da, db2 = db;
    ops::residual_backward(da2.data(), db2.data(), w.data(), n);
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(da2[i], real_t(2) * w[i]);
}

TEST_MAIN()
