// test_attention.cpp: causal attention forward (ROADMAP P2.3).
// DoD: matches a naive reference; causality test passes; softmax rows sum to
// 1 within 1e-6.
//
// Reference: per (b, h, t), scores, softmax and att*V computed directly in
// double from the formulas (no staging, no triangle shortcut beyond the
// mask's definition). Tolerances for N(0,1) inputs with hs = 16..64:
// att |err| <= 1e-5 absolute (att in [0,1]), out |err| <= 1e-5 * (1 + sum
// att*|v|). Both are well above float rounding of a 64-term dot product
// followed by exp/normalize (observed maxima are printed), and far below
// any indexing/masking error (which is O(att) or O(|v|)).
#include "test.h"

#include "tslm/common.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace tslm;

namespace {

struct Dims {
    int B, T, C, NH;
};

std::vector<real_t> random_qkv(const Dims& d, uint64_t seed, double sd) {
    Rng rng(seed, 7);
    std::vector<real_t> qkv(size_t(d.B) * d.T * 3 * d.C);
    for (auto& v : qkv) v = static_cast<real_t>(rng.normal(0.0, sd));
    return qkv;
}

struct Result {
    std::vector<real_t> out, att;
};

Result run(const Dims& d, const std::vector<real_t>& qkv) {
    Result r;
    r.out.assign(size_t(d.B) * d.T * d.C, std::numeric_limits<real_t>::quiet_NaN());
    r.att.assign(size_t(d.B) * d.NH * d.T * d.T, std::numeric_limits<real_t>::quiet_NaN());
    ops::attention_forward(r.out.data(), r.att.data(), qkv.data(), d.B, d.T, d.C, d.NH);
    return r;
}

// Compare against the double reference; returns number of violations.
int compare_reference(const Dims& d, const std::vector<real_t>& qkv, const Result& r, double* max_att_err,
                      double* max_out_err) {
    const size_t T = size_t(d.T), C = size_t(d.C), hs = C / size_t(d.NH), C3 = 3 * C;
    const double scale = 1.0 / std::sqrt(double(hs));
    int bad = 0;
    std::vector<double> s(T), p(T);
    for (int b = 0; b < d.B; ++b)
        for (int h = 0; h < d.NH; ++h)
            for (size_t t = 0; t < T; ++t) {
                const real_t* q = &qkv[(size_t(b) * T + t) * C3 + size_t(h) * hs];
                double mx = -1e300;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    const real_t* k = &qkv[(size_t(b) * T + t2) * C3 + C + size_t(h) * hs];
                    double dot = 0;
                    for (size_t i = 0; i < hs; ++i) dot += double(q[i]) * double(k[i]);
                    s[t2] = dot * scale;
                    mx = s[t2] > mx ? s[t2] : mx;
                }
                double sum = 0;
                for (size_t t2 = 0; t2 <= t; ++t2) sum += (p[t2] = std::exp(s[t2] - mx));
                const real_t* arow = &r.att[((size_t(b) * d.NH + h) * T + t) * T];
                for (size_t t2 = 0; t2 < T; ++t2) {
                    const double want = t2 <= t ? p[t2] / sum : 0.0;
                    const double err = std::fabs(double(arow[t2]) - want);
                    *max_att_err = err > *max_att_err ? err : *max_att_err;
                    if (t2 > t && arow[t2] != real_t(0)) ++bad;  // mask must be exact 0
                    if (!(err <= 1e-5)) ++bad;
                }
                for (size_t i = 0; i < hs; ++i) {
                    double o = 0, mag = 0;
                    for (size_t t2 = 0; t2 <= t; ++t2) {
                        const double v = qkv[(size_t(b) * T + t2) * C3 + 2 * C + size_t(h) * hs + i];
                        o += p[t2] / sum * v;
                        mag += p[t2] / sum * std::fabs(v);
                    }
                    const double got = r.out[(size_t(b) * T + t) * C + size_t(h) * hs + i];
                    const double err = std::fabs(got - o);
                    *max_out_err = err > *max_out_err ? err : *max_out_err;
                    if (!(err <= 1e-5 * (1.0 + mag))) ++bad;
                }
            }
    return bad;
}

}  // namespace

TEST(matches_naive_reference) {
    const Dims dims[] = {
        {2, 16, 32, 2},  // gradcheck-like
        {1, 33, 48, 3},  // hs = 16, odd T
        {2, 64, 128, 4}, // small preset head size 32
        {1, 40, 256, 4}, // hs = 64
        {3, 1, 8, 2},    // T = 1
        {1, 7, 10, 2},   // hs = 5 (< LANES): scalar tail only
    };
    double max_att = 0, max_out = 0;
    int bad = 0;
    for (const Dims& d : dims) {
        const auto qkv = random_qkv(d, uint64_t(d.T * 31 + d.C), 1.0);
        bad += compare_reference(d, qkv, run(d, qkv), &max_att, &max_out);
    }
    std::printf("    max |att err| = %.3g, max |out err| = %.3g (tolerance 1e-5)\n", max_att, max_out);
    EXPECT_EQ(bad, 0);
}

TEST(softmax_rows_sum_to_one_and_mask_is_zero) {
    const Dims d{2, 50, 64, 4};
    const auto qkv = random_qkv(d, 5, 2.0);
    const Result r = run(d, qkv);
    double worst = 0;
    for (size_t row = 0; row < size_t(d.B) * d.NH * d.T; ++row) {
        const size_t t = row % size_t(d.T);
        double sum = 0;
        for (size_t t2 = 0; t2 < size_t(d.T); ++t2) {
            const real_t a = r.att[row * d.T + t2];
            if (t2 > t) ASSERT_EQ(a, real_t(0));
            else ASSERT_TRUE(a >= real_t(0));
            sum += a;
        }
        worst = std::fabs(sum - 1.0) > worst ? std::fabs(sum - 1.0) : worst;
    }
    std::printf("    max |row sum - 1| = %.3g\n", worst);
    EXPECT_TRUE(worst <= 1e-6);
}

// First position attends only to itself: att = 1 exactly, out = V(0) exactly.
TEST(first_position_exact) {
    const Dims d{2, 5, 12, 3};
    const auto qkv = random_qkv(d, 9, 1.0);
    const Result r = run(d, qkv);
    for (int b = 0; b < d.B; ++b) {
        for (int h = 0; h < d.NH; ++h) EXPECT_EQ(r.att[(size_t(b) * d.NH + h) * d.T * d.T], real_t(1));
        for (int c = 0; c < d.C; ++c)
            EXPECT_EQ(r.out[size_t(b) * d.T * d.C + c], qkv[size_t(b) * d.T * 3 * d.C + 2 * d.C + c]);
    }
}

// Perturbing every Q, K, V value at position p (batch row 0) must leave all
// outputs and att rows at positions < p bit-identical, and leave batch row 1
// completely unchanged.
TEST(causality) {
    const Dims d{2, 24, 32, 4};
    const auto base = random_qkv(d, 13, 1.0);
    const Result r0 = run(d, base);
    for (int p : {0, 1, 11, 23}) {
        auto pert = base;
        for (int c = 0; c < 3 * d.C; ++c) pert[(size_t(0) * d.T + p) * 3 * d.C + c] += real_t(0.75);
        const Result r1 = run(d, pert);
        for (int t = 0; t < d.T; ++t) {
            const bool before = t < p;
            for (int c = 0; c < d.C; ++c) {
                const size_t i0 = size_t(t) * d.C + c;                    // batch 0
                const size_t i1 = (size_t(d.T) + t) * d.C + c;            // batch 1
                if (before) ASSERT_EQ(r1.out[i0], r0.out[i0]);
                ASSERT_EQ(r1.out[i1], r0.out[i1]);
            }
            for (int h = 0; h < d.NH; ++h)
                for (int t2 = 0; t2 < d.T; ++t2) {
                    const size_t a0 = ((size_t(0) * d.NH + h) * d.T + t) * d.T + t2;
                    const size_t a1 = ((size_t(1) * d.NH + h) * d.T + t) * d.T + t2;
                    if (before) ASSERT_EQ(r1.att[a0], r0.att[a0]);
                    ASSERT_EQ(r1.att[a1], r0.att[a1]);
                }
        }
        // And position p itself does change (the perturbation is not a no-op).
        bool changed = false;
        for (int c = 0; c < d.C; ++c) changed = changed || r1.out[size_t(p) * d.C + c] != r0.out[size_t(p) * d.C + c];
        EXPECT_TRUE(changed);
    }
}

// Large logits: max subtraction keeps everything finite and normalized.
TEST(large_magnitude_stable) {
    const Dims d{1, 32, 32, 2};
    const auto qkv = random_qkv(d, 21, 40.0);  // scores ~ O(1e4)
    const Result r = run(d, qkv);
    for (real_t v : r.out) ASSERT_TRUE(std::isfinite(v));
    for (size_t row = 0; row < size_t(d.NH) * d.T; ++row) {
        double sum = 0;
        for (int t2 = 0; t2 < d.T; ++t2) {
            ASSERT_TRUE(std::isfinite(r.att[row * d.T + t2]));
            sum += r.att[row * d.T + t2];
        }
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_MAIN()
