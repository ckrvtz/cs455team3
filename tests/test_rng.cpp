// test_rng.cpp: rng.h (ROADMAP P0.3, §4.6).
// Known-answer values were recorded once and are committed here. They were
// cross-checked against an independent from-the-definition implementation
// (not committed), and the bare-algorithm cases match the published
// reference outputs of splitmix64 and xoshiro256**.
#include "test.h"

#include "tslm/rng.h"

#include <cmath>
#include <cstdint>
#include <vector>

using tslm::Rng;

TEST(splitmix64_known_answer) {
    uint64_t s = 0;
    EXPECT_EQ(tslm::splitmix64_next(&s), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(tslm::splitmix64_next(&s), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(tslm::splitmix64_next(&s), 0x06C45D188009454Full);
}

TEST(xoshiro256ss_raw_state_known_answer) {
    Rng r = Rng::from_state(1, 2, 3, 4);
    EXPECT_EQ(r.next_u64(), uint64_t{11520});
    EXPECT_EQ(r.next_u64(), uint64_t{0});
    EXPECT_EQ(r.next_u64(), uint64_t{1509978240});
    EXPECT_EQ(r.next_u64(), uint64_t{1215971899390074240});
}

// ROADMAP P0.3 DoD: first 5 outputs of a fixed seed.
TEST(seeded_known_answer_first_5) {
    const uint64_t want[5] = {0x044A17A0F866DC39ull, 0xB04857E345BCBDCFull, 0x981823853570D2EFull,
                              0x5B3D2425EF3637AEull, 0xCD182A56A0515778ull};
    Rng r(1337, 0);
    for (uint64_t w : want) EXPECT_EQ(r.next_u64(), w);

    const uint64_t want_s1[5] = {0xDE2314E8CD278A77ull, 0xC66B8286C608CFEDull, 0x135C484E3D9459ADull,
                                 0xBB2514D5ED6155F4ull, 0x6D926E3BDCF1D561ull};
    Rng r1(1337, 1);
    for (uint64_t w : want_s1) EXPECT_EQ(r1.next_u64(), w);

    Rng r0(0, 0);
    EXPECT_EQ(r0.next_u64(), 0xC78B4F5177DDB86Bull);
}

TEST(same_seed_same_stream_is_identical) {
    Rng a(42, 7), b(42, 7);
    for (int i = 0; i < 1000; ++i) ASSERT_EQ(a.next_u64(), b.next_u64());
}

TEST(streams_and_seeds_differ) {
    // Neighbouring streams/seeds must not produce overlapping prefixes.
    std::vector<uint64_t> firsts;
    for (uint64_t seed = 0; seed < 4; ++seed) {
        for (uint64_t stream = 0; stream < 128; ++stream) {
            Rng r(seed, stream);
            firsts.push_back(r.next_u64());
        }
    }
    int dup = 0;
    for (size_t i = 0; i < firsts.size(); ++i)
        for (size_t j = i + 1; j < firsts.size(); ++j) dup += firsts[i] == firsts[j];
    EXPECT_EQ(dup, 0);
}

TEST(uniform_range) {
    Rng r(1, 0);
    double lo = 1.0, hi = 0.0;
    for (int i = 0; i < 100000; ++i) {
        const double u = r.uniform();
        ASSERT_TRUE(u >= 0.0 && u < 1.0);
        lo = u < lo ? u : lo;
        hi = u > hi ? u : hi;
    }
    EXPECT_TRUE(lo < 1e-3);
    EXPECT_TRUE(hi > 1.0 - 1e-3);
}

TEST(uniform_int_bounds_and_balance) {
    Rng r(2, 0);
    const uint64_t n = 7;
    const int draws = 700000;
    int counts[7] = {0};
    for (int i = 0; i < draws; ++i) {
        const uint64_t k = r.uniform_int(n);
        ASSERT_TRUE(k < n);
        ++counts[k];
    }
    // Expected 100000 each; binomial sd ~ 293. Allow ~5 sd.
    for (int c : counts) EXPECT_NEAR(c, 100000, 1500);
    EXPECT_EQ(r.uniform_int(1), uint64_t{0});
}

// ROADMAP P0.3 DoD: normal() mean/std within 1% over 1e6 samples. For a
// zero-mean target, "within 1%" is taken relative to std: |mean| < 0.01.
TEST(normal_mean_std_1e6) {
    Rng r(1337, 3);
    const int n = 1000000;
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = r.normal();
        ASSERT_TRUE(std::isfinite(x));
        sum += x;
        sumsq += x * x;
    }
    const double mean = sum / n;
    const double sd = std::sqrt(sumsq / n - mean * mean);
    std::printf("    normal(): mean=%.6f std=%.6f over %d samples\n", mean, sd, n);
    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(sd, 1.0, 0.01);

    // Scaled form used by weight init: N(0, 0.02).
    Rng r2(1337, 4);
    double s2 = 0.0, ss2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = r2.normal(0.0, 0.02);
        s2 += x;
        ss2 += x * x;
    }
    const double m2 = s2 / n;
    EXPECT_NEAR(m2, 0.0, 0.01 * 0.02);
    EXPECT_NEAR(std::sqrt(ss2 / n - m2 * m2), 0.02, 0.01 * 0.02);
}

TEST(bad_arguments_abort) {
    std::string err;
    EXPECT_TRUE(tslm::test::dies_with_abort([] { Rng r(1); r.uniform_int(0); }, &err));
    EXPECT_TRUE(err.find("n must be >= 1") != std::string::npos);
    err.clear();
    EXPECT_TRUE(tslm::test::dies_with_abort([] { Rng::from_state(0, 0, 0, 0); }, &err));
    EXPECT_TRUE(err.find("all-zero") != std::string::npos);
}

TEST_MAIN()
