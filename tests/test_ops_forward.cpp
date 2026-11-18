// test_ops_forward.cpp: P2.1 forward ops (embedding, layernorm, gelu, residual).
// DoD: hand-computed cases match to 1e-6. Expected values were worked out
// from the formulas in ROADMAP §6.3 in double precision, independently of
// this code, and are written out here. Tolerance: |got - want| <=
// 1e-6 * max(1, |want|), the same in the float and double builds.
// Outputs are pre-filled with NaN to prove forward ops overwrite.
#include "test.h"

#include "tslm/common.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace tslm;

namespace {

constexpr double TOL = 1e-6;

void expect_close(double got, double want, const char* what, size_t i) {
    const double scale = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    if (!(std::fabs(got - want) <= TOL * scale)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s[%zu]: got %.9g want %.9g", what, i, got, want);
        test::report(__FILE__, __LINE__, buf);
    }
}

std::vector<real_t> nan_vec(size_t n) {
    return std::vector<real_t>(n, std::numeric_limits<real_t>::quiet_NaN());
}

}  // namespace

TEST(embedding_hand_case) {
    // V=4, C=3, B=2, T=2; wpe has 3 rows (max_T=3), only 2 used.
    const real_t wte[4 * 3] = {0.1f, 0.2f, 0.3f,   1, 2, 3,   -1, -2, -3,   10, 20, 30};
    const real_t wpe[3 * 3] = {0.01f, 0.02f, 0.03f,   0.5f, 0.5f, 0.5f,   9, 9, 9};
    const int32_t tok[2 * 2] = {1, 3,  0, 1};
    std::vector<real_t> out = nan_vec(2 * 2 * 3);
    ops::embedding_forward(out.data(), tok, wte, wpe, 2, 2, 3);
    const double want[12] = {
        1.01, 2.02, 3.03,     // b0 t0: wte[1] + wpe[0]
        10.5, 20.5, 30.5,     // b0 t1: wte[3] + wpe[1]
        0.11, 0.22, 0.33,     // b1 t0: wte[0] + wpe[0]
        1.5, 2.5, 3.5,        // b1 t1: wte[1] + wpe[1]
    };
    for (size_t i = 0; i < 12; ++i) expect_close(out[i], want[i], "embedding", i);
}

TEST(layernorm_hand_cases) {
    const int N = 3, C = 4;
    const real_t inp[N * C] = {1, 2, 3, 4,   5, 5, 5, 5,   -2, 0.5f, 3, 10.25f};
    const real_t gamma[C] = {1, 2, 0.5f, -1};
    const real_t beta[C] = {0, 0.1f, 0.2f, 0.3f};
    std::vector<real_t> out = nan_vec(N * C), mean = nan_vec(N), rstd = nan_vec(N);
    ops::layernorm_forward(out.data(), mean.data(), rstd.data(), inp, gamma, beta, N, C);

    const double want_mean[N] = {2.5, 5.0, 2.9375};
    const double want_rstd[N] = {0.894423613312618, 316.2277660168379, 0.2184821598805085};
    const double want_out[N * C] = {
        -1.3416354199689269, -0.794423613312618, 0.4236059033281545, -1.0416354199689268,
        0.0, 0.1, 0.2, 0.3,  // constant row: var = 0, rstd = 1/sqrt(eps), out = beta
        -1.0787556644100107, -0.9651005294174789, 0.2068275674962659, -1.2976507941262185,
    };
    for (size_t n = 0; n < size_t(N); ++n) {
        expect_close(mean[n], want_mean[n], "ln mean", n);
        expect_close(rstd[n], want_rstd[n], "ln rstd", n);
    }
    for (size_t i = 0; i < size_t(N * C); ++i) expect_close(out[i], want_out[i], "ln out", i);
}

// Properties on random rows with gamma=1, beta=0: each output row has mean 0
// and variance var/(var+eps) (about 1). Covers larger C and N than the
// hand cases.
TEST(layernorm_random_row_properties) {
    const int N = 64, C = 256;
    Rng rng(11, 0);
    std::vector<real_t> inp(size_t(N) * C), gamma(C, real_t(1)), beta(C, real_t(0));
    for (size_t i = 0; i < inp.size(); ++i) inp[i] = static_cast<real_t>(rng.normal(3.0, 2.0));
    std::vector<real_t> out = nan_vec(inp.size()), mean = nan_vec(N), rstd = nan_vec(N);
    ops::layernorm_forward(out.data(), mean.data(), rstd.data(), inp.data(), gamma.data(), beta.data(), N, C);
    for (int n = 0; n < N; ++n) {
        double m = 0, v = 0, im = 0, iv = 0;
        for (int c = 0; c < C; ++c) {
            m += out[size_t(n) * C + c];
            im += inp[size_t(n) * C + c];
        }
        m /= C;
        im /= C;
        for (int c = 0; c < C; ++c) {
            const double d = out[size_t(n) * C + c] - m, e = inp[size_t(n) * C + c] - im;
            v += d * d;
            iv += e * e;
        }
        v /= C;
        iv /= C;
        expect_close(m, 0.0, "ln row mean", size_t(n));
        // Variance of a float-rounded output: allow a few float ulps (1e-5).
        if (!(std::fabs(v - iv / (iv + 1e-5)) <= 1e-5))
            test::report(__FILE__, __LINE__, "ln row variance off for row " + std::to_string(n));
        expect_close(mean[size_t(n)], im, "ln saved mean", size_t(n));
        expect_close(rstd[size_t(n)], 1.0 / std::sqrt(iv + 1e-5), "ln saved rstd", size_t(n));
    }
}

TEST(gelu_hand_cases) {
    const real_t x[10] = {-10, -3, -1, -0.5f, 0, 0.5f, 1, 2, 3, 10};
    const double want[10] = {
        -0.0, -0.0036373920817729943, -0.15880800939172324, -0.15428599017485606, 0.0,
        0.34571400982514394, 0.8411919906082768, 1.954597694087775, 2.996362607918227, 10.0,
    };
    std::vector<real_t> out = nan_vec(10);
    ops::gelu_forward(out.data(), x, 10);
    for (size_t i = 0; i < 10; ++i) expect_close(out[i], want[i], "gelu", i);
    // Large magnitudes saturate without NaN/Inf.
    const real_t big[2] = {real_t(1e4), real_t(-1e4)};
    real_t ob[2];
    ops::gelu_forward(ob, big, 2);
    EXPECT_TRUE(std::isfinite(ob[0]) && std::isfinite(ob[1]));
    expect_close(ob[0], 1e4, "gelu big", 0);
    expect_close(ob[1], 0.0, "gelu big", 1);
}

TEST(residual_hand_case) {
    const real_t a[5] = {1, -2, 0.5f, 1e3f, 0};
    const real_t b[5] = {2, 2, 0.25f, -1e3f, -0.125f};
    std::vector<real_t> out = nan_vec(5);
    ops::residual_forward(out.data(), a, b, 5);
    const double want[5] = {3, 0, 0.75, 0, -0.125};
    for (size_t i = 0; i < 5; ++i) expect_close(out[i], want[i], "residual", i);
    ops::residual_forward(out.data(), a, b, 0);  // n = 0 is a no-op
}

TEST_MAIN()
