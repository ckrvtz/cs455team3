// test_softmax_xent.cpp: softmax_xent_forward and sum_reduce (ROADMAP P2.4).
// DoD: loss on uniform logits = ln(V). Expected values for the hand cases
// were computed independently in double precision from the §6.3 formulas.
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

std::vector<real_t> nans(size_t n) { return std::vector<real_t>(n, std::numeric_limits<real_t>::quiet_NaN()); }

// |got - want| <= 1e-6 * max(1, |want|)
bool close(double got, double want) {
    const double s = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    return std::fabs(got - want) <= 1e-6 * s;
}

}  // namespace

// DoD: uniform logits -> every loss = ln V, mean = ln V, probs = 1/V.
TEST(uniform_logits_give_ln_V) {
    for (int V : {7, 10000}) {
        const int N = 5;
        for (real_t value : {real_t(0), real_t(3.7), real_t(-120)}) {
            std::vector<real_t> logits(size_t(N) * V, value), probs = nans(size_t(N) * V), losses = nans(N);
            std::vector<int32_t> targets = {0, 1, V - 1, V / 2, 3};
            ops::softmax_xent_forward(probs.data(), losses.data(), logits.data(), targets.data(), N, V);
            const double lnV = std::log(double(V));
            for (int n = 0; n < N; ++n) EXPECT_TRUE(close(losses[size_t(n)], lnV));
            for (real_t p : probs) ASSERT_TRUE(close(p, 1.0 / V));
            const double mean = ops::sum_reduce(losses.data(), size_t(N)) / N;
            EXPECT_TRUE(close(mean, lnV));
            if (V == 10000 && value == real_t(0))
                std::printf("    V=10000 uniform: mean loss %.9f, ln(10000) = %.9f\n", mean, lnV);
        }
    }
}

TEST(hand_cases) {
    const real_t logits[2 * 3] = {1, 2, 3,   5, 5, 5};
    const int32_t targets[2] = {2, 0};
    std::vector<real_t> probs = nans(6), losses = nans(2);
    ops::softmax_xent_forward(probs.data(), losses.data(), logits, targets, 2, 3);
    const double want_p[6] = {0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
                              1.0 / 3, 1.0 / 3, 1.0 / 3};
    for (int i = 0; i < 6; ++i) EXPECT_TRUE(close(probs[size_t(i)], want_p[i]));
    EXPECT_TRUE(close(losses[0], 0.4076059644443804));
    EXPECT_TRUE(close(losses[1], std::log(3.0)));

    const real_t l2[4] = {0.5f, -1.25f, 2.0f, 0.0f};
    const int32_t t2 = 1;
    real_t p2[4], loss2;
    ops::softmax_xent_forward(p2, &loss2, l2, &t2, 1, 4);
    EXPECT_TRUE(close(loss2, 3.584498612603663));
    const double want2[4] = {0.1596935500321082, 0.0277505779326804, 0.7156968377823842, 0.09685903425282703};
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(close(p2[i], want2[i]));
}

// Random rows vs a double reference; rows sum to 1; loss = -log p[target].
TEST(random_rows_vs_reference) {
    const int N = 40, V = 1000;
    Rng rng(17, 0);
    std::vector<real_t> logits(size_t(N) * V);
    for (auto& v : logits) v = static_cast<real_t>(rng.normal(0.0, 3.0));
    std::vector<int32_t> targets(N);
    for (auto& t : targets) t = static_cast<int32_t>(rng.uniform_int(V));
    std::vector<real_t> probs = nans(logits.size()), losses = nans(N);
    ops::softmax_xent_forward(probs.data(), losses.data(), logits.data(), targets.data(), N, V);
    double worst_sum = 0;
    for (int n = 0; n < N; ++n) {
        const real_t* l = &logits[size_t(n) * V];
        double m = l[0];
        for (int v = 1; v < V; ++v) m = l[v] > m ? l[v] : m;
        double S = 0;
        for (int v = 0; v < V; ++v) S += std::exp(double(l[v]) - m);
        double rowsum = 0;
        for (int v = 0; v < V; ++v) {
            const double want = std::exp(double(l[v]) - m) / S;
            const double got = probs[size_t(n) * V + v];
            if (!(std::fabs(got - want) <= 1e-6 * (want > 1e-3 ? want : 1e-3)))
                test::report(__FILE__, __LINE__, "prob mismatch row " + std::to_string(n));
            rowsum += got;
        }
        worst_sum = std::fabs(rowsum - 1) > worst_sum ? std::fabs(rowsum - 1) : worst_sum;
        const double want_loss = std::log(S) - (double(l[targets[size_t(n)]]) - m);
        EXPECT_TRUE(close(losses[size_t(n)], want_loss));
    }
    EXPECT_TRUE(worst_sum <= 1e-6);
}

// Huge logits stay finite; a target whose probability underflows still gets
// a finite, correct loss (log-sum-exp form).
TEST(extreme_values) {
    const int V = 5;
    const real_t logits[2 * 5] = {1e4f, 1e4f - 1, -1e4f, 0, 1e4f,   0, 0, 0, 0, -1000};
    const int32_t targets[2] = {2, 4};
    std::vector<real_t> probs = nans(10), losses = nans(2);
    ops::softmax_xent_forward(probs.data(), losses.data(), logits, targets, 2, V);
    for (real_t p : probs) EXPECT_TRUE(std::isfinite(p));
    EXPECT_EQ(probs[2], real_t(0));  // e^-20000 underflows
    // Row 0: m = 1e4, S = 2 + e^-1 (+ ~0); loss = log(S) - (-1e4 - 1e4).
    EXPECT_TRUE(close(losses[0], std::log(2.0 + std::exp(-1.0)) + 2e4));
    // Row 1: target logit -1000: loss = log(4 + e^-1000) + 1000.
    EXPECT_TRUE(close(losses[1], std::log(4.0) + 1000.0));
    EXPECT_TRUE(std::isfinite(losses[0]) && std::isfinite(losses[1]));
}

TEST(sum_reduce_accumulates_in_double) {
    EXPECT_EQ(ops::sum_reduce(nullptr, 0), 0.0);
    const real_t ints[5] = {1, 2, 3, 4, -10};
    EXPECT_EQ(ops::sum_reduce(ints, 5), 0.0);
    // 1e6 copies of real_t(0.1): a real_t accumulator would drift by ~1%
    // in float; a double accumulator matches n * double(real_t(0.1)).
    const size_t n = 1000000;
    std::vector<real_t> x(n, real_t(0.1));
    const double want = double(n) * double(real_t(0.1));
    const double got = ops::sum_reduce(x.data(), n);
    EXPECT_NEAR(got, want, 1e-6);
    std::printf("    sum_reduce(1e6 x 0.1) = %.9f (exact for these inputs: %.9f)\n", got, want);
}

TEST_MAIN()
