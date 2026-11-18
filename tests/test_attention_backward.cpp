// test_attention_backward.cpp: attention_backward (ROADMAP P3.3).
// DoD: finite-difference test; scratch parameter accepted.
//
// 1. Finite differences of L = sum w * out (dL/dout = w) w.r.t. every Q, K, V
//    entry, on small shapes covering head sizes below (hs = 4) and at or
//    above (hs = 8, 16) the lane width, T = 1, odd T, and B > 1.
//    REAL=double build: §10 criterion (eps 1e-5, rel err < 1e-6, floor 1e-8).
//    float build: smoke check (eps 1e-2, tol 2e-2, floor 1e-2).
// 2. Larger shapes vs an independent naive double reference (full T x T
//    softmax Jacobian), held to rel-L2 < 1e-5 per Q/K/V part.
// 3. Accumulate (+=): a second call doubles dqkv.
// 4. Scratch: pre-filling it with NaN changes nothing, so the op writes
//    every scratch entry it reads.
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

struct Dims {
    int B, T, C, NH;
    size_t qkv() const { return size_t(B) * T * 3 * C; }
    size_t out() const { return size_t(B) * T * C; }
    size_t att() const { return size_t(B) * NH * T * T; }
};

std::vector<real_t> randn(size_t n, uint64_t seed, double sd) {
    Rng rng(seed, 2);
    std::vector<real_t> v(n);
    for (auto& x : v) x = static_cast<real_t>(rng.normal(0.0, sd));
    return v;
}

struct Fwd {
    std::vector<real_t> out, att;
};

Fwd forward(const Dims& d, const std::vector<real_t>& qkv) {
    Fwd f{std::vector<real_t>(d.out()), std::vector<real_t>(d.att())};
    ops::attention_forward(f.out.data(), f.att.data(), qkv.data(), d.B, d.T, d.C, d.NH);
    return f;
}

std::vector<real_t> backward(const Dims& d, const std::vector<real_t>& qkv, const std::vector<real_t>& w,
                             real_t scratch_fill = 0) {
    const Fwd f = forward(d, qkv);
    std::vector<real_t> dqkv(d.qkv(), 0), scratch(d.att(), scratch_fill);
    ops::attention_backward(dqkv.data(), scratch.data(), w.data(), qkv.data(), f.att.data(), d.B, d.T, d.C, d.NH);
    return dqkv;
}

const char* part_name(size_t idx, const Dims& d) {
    const size_t c = idx % (3 * size_t(d.C));
    return c < size_t(d.C) ? "dQ" : (c < 2 * size_t(d.C) ? "dK" : "dV");
}

}  // namespace

TEST(finite_differences) {
    const Dims dims[] = {{1, 1, 8, 2}, {2, 5, 8, 2}, {1, 7, 32, 2}, {2, 4, 16, 2}, {1, 6, 24, 3}};
    double worst = 0, worst_grad = 0, worst_num = 0;
    int bad = 0;
    for (const Dims& d : dims) {
        std::vector<real_t> qkv = randn(d.qkv(), uint64_t(d.T * 7 + d.C), 1.0);
        const std::vector<real_t> w = randn(d.out(), uint64_t(d.T * 11 + d.C), 1.0);
        const std::vector<real_t> g = backward(d, qkv, w);
        auto loss = [&] {
            const Fwd f = forward(d, qkv);
            double s = 0;
            for (size_t i = 0; i < f.out.size(); ++i) s += double(w[i]) * double(f.out[i]);
            return s;
        };
        for (size_t i = 0; i < qkv.size(); ++i) {
            const real_t orig = qkv[i];
            const real_t up = static_cast<real_t>(double(orig) + EPS), dn = static_cast<real_t>(double(orig) - EPS);
            qkv[i] = up;
            const double lp = loss();
            qkv[i] = dn;
            const double lm = loss();
            qkv[i] = orig;
            const double num = (lp - lm) / (double(up) - double(dn));
            const double a = g[i];
            const double rel = std::fabs(a - num) / std::fmax(std::fmax(std::fabs(a), std::fabs(num)), FLOOR);
            if (rel > worst) {
                worst = rel;
                worst_grad = a;
                worst_num = num;
            }
            if (!(rel < TOL) && bad++ < 4) {
                char buf[220];
                std::snprintf(buf, sizeof(buf), "B=%d T=%d C=%d NH=%d %s[%zu]: analytic %.10g numeric %.10g rel %.3g",
                              d.B, d.T, d.C, d.NH, part_name(i, d), i, a, num, rel);
                test::report(__FILE__, __LINE__, buf);
            }
        }
    }
    std::printf("    FD over all Q/K/V entries: worst rel err %.2e (tol %.0e), at analytic %.3e vs numeric %.3e\n", worst,
                TOL, worst_grad, worst_num);
    EXPECT_EQ(bad, 0);
}

// Independent naive double reference: recompute att from qkv, then use the
// full softmax Jacobian J[t2][t3] = a[t2] (delta - a[t3]) explicitly.
TEST(vs_naive_reference) {
    const Dims dims[] = {{2, 40, 64, 4}, {1, 64, 128, 4}, {3, 17, 48, 3}};
    for (const Dims& d : dims) {
        const std::vector<real_t> qkv = randn(d.qkv(), uint64_t(d.T + d.C), 1.0);
        const std::vector<real_t> w = randn(d.out(), uint64_t(d.T * 3 + d.C), 1.0);
        const std::vector<real_t> g = backward(d, qkv, w);
        const size_t T = size_t(d.T), C = size_t(d.C), C3 = 3 * C, hs = C / size_t(d.NH);
        const double scale = 1.0 / std::sqrt(double(hs));
        std::vector<double> ref(d.qkv(), 0.0);
        for (int b = 0; b < d.B; ++b)
            for (int h = 0; h < d.NH; ++h)
                for (size_t t = 0; t < T; ++t) {
                    auto Q = [&](size_t tt, size_t i) { return double(qkv[(size_t(b) * T + tt) * C3 + h * hs + i]); };
                    auto K = [&](size_t tt, size_t i) { return double(qkv[(size_t(b) * T + tt) * C3 + C + h * hs + i]); };
                    auto V = [&](size_t tt, size_t i) { return double(qkv[(size_t(b) * T + tt) * C3 + 2 * C + h * hs + i]); };
                    std::vector<double> s(t + 1), a(t + 1), da(t + 1, 0.0);
                    double mx = -1e300, sum = 0;
                    for (size_t t2 = 0; t2 <= t; ++t2) {
                        double dot = 0;
                        for (size_t i = 0; i < hs; ++i) dot += Q(t, i) * K(t2, i);
                        s[t2] = dot * scale;
                        mx = s[t2] > mx ? s[t2] : mx;
                    }
                    for (size_t t2 = 0; t2 <= t; ++t2) sum += (a[t2] = std::exp(s[t2] - mx));
                    for (double& x : a) x /= sum;
                    const real_t* dy = &w[(size_t(b) * T + t) * C + h * hs];
                    for (size_t t2 = 0; t2 <= t; ++t2)
                        for (size_t i = 0; i < hs; ++i) {
                            da[t2] += double(dy[i]) * V(t2, i);
                            ref[(size_t(b) * T + t2) * C3 + 2 * C + h * hs + i] += a[t2] * double(dy[i]);
                        }
                    for (size_t t2 = 0; t2 <= t; ++t2) {
                        double ds = 0;  // d L / d s[t2] via the full Jacobian
                        for (size_t t3 = 0; t3 <= t; ++t3) ds += da[t3] * a[t3] * ((t2 == t3 ? 1.0 : 0.0) - a[t2]);
                        for (size_t i = 0; i < hs; ++i) {
                            ref[(size_t(b) * T + t) * C3 + h * hs + i] += ds * scale * K(t2, i);
                            ref[(size_t(b) * T + t2) * C3 + C + h * hs + i] += ds * scale * Q(t, i);
                        }
                    }
                }
        for (int part = 0; part < 3; ++part) {
            double num = 0, den = 0;
            for (size_t r = 0; r < size_t(d.B) * T; ++r)
                for (size_t c = 0; c < C; ++c) {
                    const size_t i = r * C3 + size_t(part) * C + c;
                    num += (double(g[i]) - ref[i]) * (double(g[i]) - ref[i]);
                    den += ref[i] * ref[i];
                }
            const double rel = std::sqrt(num / den);
            std::printf("    B=%d T=%d C=%d NH=%d %s rel-L2 vs reference %.2e\n", d.B, d.T, d.C, d.NH,
                        part == 0 ? "dQ" : (part == 1 ? "dK" : "dV"), rel);
            EXPECT_TRUE(rel < 1e-5);
        }
    }
}

TEST(accumulates_and_scratch_is_workspace) {
    const Dims d{2, 9, 32, 4};
    const std::vector<real_t> qkv = randn(d.qkv(), 5, 1.0), w = randn(d.out(), 6, 1.0);
    const std::vector<real_t> once = backward(d, qkv, w);

    // Scratch pre-filled with NaN: bit-identical result.
    const std::vector<real_t> nanfill = backward(d, qkv, w, std::numeric_limits<real_t>::quiet_NaN());
    bool same = true;
    for (size_t i = 0; i < once.size(); ++i) same = same && once[i] == nanfill[i];
    EXPECT_TRUE(same);

    // Second call on the same dqkv doubles it.
    const Fwd f = forward(d, qkv);
    std::vector<real_t> dqkv(d.qkv(), 0), scratch(d.att());
    for (int k = 0; k < 2; ++k)
        ops::attention_backward(dqkv.data(), scratch.data(), w.data(), qkv.data(), f.att.data(), d.B, d.T, d.C, d.NH);
    for (size_t i = 0; i < once.size(); ++i)
        ASSERT_NEAR(double(dqkv[i]), 2.0 * double(once[i]), 1e-5 * std::fmax(1.0, std::fabs(2.0 * double(once[i]))));

    // Position 0 attends only to itself (att = 1): its query gets no gradient.
    for (int b = 0; b < d.B; ++b)
        for (int c = 0; c < d.C; ++c) EXPECT_EQ(once[size_t(b) * d.T * 3 * d.C + size_t(c)], real_t(0));
}

TEST_MAIN()
