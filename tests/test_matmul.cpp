// test_matmul.cpp: blocked matmul_forward vs a naive reference (ROADMAP P2.2).
//
// Reference: plain triple loop accumulated in double. Tolerance: a rigorous
// worst-case floating-point bound valid for ANY summation order,
//   |got - ref| <= gamma * (sum_c |inp[n,c] * W[o,c]| + |bias[o]|) + tiny,
//   gamma = (C + 2) * u,  u = eps(real_t) / 2,
// so the check holds whatever order the blocked kernel adds in, yet is
// orders of magnitude tighter than a wrong-index or missed-block bug.
// Shapes: N, C, OC drawn from {1, 7, 64, 100, 256, 1024} (DoD), plus
// shapes straddling the cache-block sizes (NB=128, KC=256, OB=64) and the
// 4x4 register tile.
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

struct Shape {
    int N, C, OC;
};

// Returns the number of out-of-bound elements (and reports the first few).
int check_shape(const Shape& s, bool with_bias, uint64_t seed) {
    const size_t N = size_t(s.N), C = size_t(s.C), OC = size_t(s.OC);
    Rng rng(seed, uint64_t(s.N) * 1000003u + uint64_t(s.C) * 1009u + uint64_t(s.OC));
    std::vector<real_t> inp(N * C), W(OC * C), bias(OC);
    for (auto& v : inp) v = static_cast<real_t>(rng.normal());
    for (auto& v : W) v = static_cast<real_t>(rng.normal(0.0, 0.5));
    for (auto& v : bias) v = static_cast<real_t>(rng.normal());
    std::vector<real_t> out(N * OC, std::numeric_limits<real_t>::quiet_NaN());  // must be overwritten
    ops::matmul_forward(out.data(), inp.data(), W.data(), with_bias ? bias.data() : nullptr, s.N, s.C, s.OC);

    const double u = std::numeric_limits<real_t>::epsilon() / 2.0;
    const double gamma = double(C + 2) * u;
    int bad = 0;
    for (size_t n = 0; n < N; ++n) {
        for (size_t o = 0; o < OC; ++o) {
            double ref = with_bias ? double(bias[o]) : 0.0;
            double mag = with_bias ? std::fabs(double(bias[o])) : 0.0;
            for (size_t c = 0; c < C; ++c) {
                const double p = double(inp[n * C + c]) * double(W[o * C + c]);
                ref += p;
                mag += std::fabs(p);
            }
            const double got = double(out[n * OC + o]);
            const double bound = gamma * mag + 1e-30;
            if (!(std::fabs(got - ref) <= bound)) {
                if (bad < 3) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf), "N=%d C=%d OC=%d bias=%d out[%zu,%zu]: got %.9g ref %.9g bound %.3g",
                                  s.N, s.C, s.OC, int(with_bias), n, o, got, ref, bound);
                    test::report(__FILE__, __LINE__, buf);
                }
                ++bad;
            }
        }
    }
    return bad;
}

}  // namespace

TEST(matmul_vs_reference_grid) {
    const int dims[] = {1, 7, 64, 100, 256};
    int shapes = 0, bad = 0;
    for (int n : dims)
        for (int c : dims)
            for (int oc : dims) {
                bad += check_shape({n, c, oc}, (n + c + oc) % 2 == 0, 1);
                ++shapes;
            }
    std::printf("    %d shapes from {1,7,64,100,256}^3: %d bad elements\n", shapes, bad);
    EXPECT_EQ(bad, 0);
}

TEST(matmul_vs_reference_with_1024) {
    const Shape shapes[] = {
        {1024, 64, 100}, {7, 1024, 256}, {100, 256, 1024}, {1, 1024, 1024},
        {64, 1024, 64},  {1024, 7, 1024}, {1024, 1, 7},
    };
    int bad = 0;
    for (const Shape& s : shapes) {
        bad += check_shape(s, true, 2);
        bad += check_shape(s, false, 3);
    }
    EXPECT_EQ(bad, 0);
}

// Shapes straddling NB=128, KC=256, OB=64 and the 4x4 tile, plus C tails
// shorter than one vector of lanes.
TEST(matmul_block_boundaries) {
    const Shape shapes[] = {
        {130, 257, 65}, {129, 300, 63}, {127, 255, 129}, {4, 3, 4}, {5, 9, 5},
        {3, 513, 2},    {131, 8, 67},   {2, 1, 3},       {256, 512, 128},
    };
    int bad = 0;
    for (const Shape& s : shapes) {
        bad += check_shape(s, true, 4);
        bad += check_shape(s, false, 5);
    }
    EXPECT_EQ(bad, 0);
}

// Exact small case: integers, so every summation order is exact.
TEST(matmul_exact_integer_case) {
    const real_t inp[2 * 3] = {1, 2, 3, 4, 5, 6};
    const real_t W[4 * 3] = {1, 0, 0,  0, 1, 0,  0, 0, 1,  1, -1, 2};
    const real_t bias[4] = {10, 20, 30, 40};
    real_t out[2 * 4];
    ops::matmul_forward(out, inp, W, bias, 2, 3, 4);
    const real_t want[8] = {11, 22, 33, 45,   14, 25, 36, 51};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(out[i], want[i]);
    ops::matmul_forward(out, inp, W, nullptr, 2, 3, 4);
    const real_t want0[8] = {1, 2, 3, 5,   4, 5, 6, 11};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(out[i], want0[i]);
}

// Same inputs -> bit-identical outputs (fixed summation order).
TEST(matmul_deterministic) {
    const int N = 37, C = 300, OC = 70;
    Rng rng(9);
    std::vector<real_t> inp(size_t(N) * C), W(size_t(OC) * C);
    for (auto& v : inp) v = static_cast<real_t>(rng.normal());
    for (auto& v : W) v = static_cast<real_t>(rng.normal());
    std::vector<real_t> a(size_t(N) * OC), b(size_t(N) * OC);
    ops::matmul_forward(a.data(), inp.data(), W.data(), nullptr, N, C, OC);
    ops::matmul_forward(b.data(), inp.data(), W.data(), nullptr, N, C, OC);
    EXPECT_TRUE(a == b);
}

TEST_MAIN()
