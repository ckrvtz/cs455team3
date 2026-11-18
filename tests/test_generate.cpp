// test_generate.cpp: sampling and the generation loop (ROADMAP P6.1).
// DoD part testable without a trained model: greedy output is reproducible.
//   sample_token: greedy argmax with lowest-id ties; <unk> excluded unless
//     allowed; top_k = 1 equals greedy; samples stay inside the top-k set;
//     sampled frequencies match softmax(logits / temperature); a seed
//     reproduces the draws; bad options abort.
//   generate_ids: greedy and seeded sampling are reproducible; each greedy
//     token equals the argmax of a direct model_forward over the sliding
//     window (context longer than max_seq_len included); EOS stops early and
//     is not returned; max_new bounds the output.
#include "test.h"

#include <cmath>
#include <vector>

#include "tslm/generate.h"
#include "tslm/model.h"
#include "tslm/rng.h"

using namespace tslm;

TEST(sample_greedy_ties_and_unk) {
    std::vector<real_t> l = {5, 1, 3, 3, 2};  // id 0 = <unk> has the largest logit
    SampleOptions o;
    o.greedy = true;
    EXPECT_EQ(sample_token(l.data(), 5, o, nullptr), 2);  // tie 2/3 -> lower id; <unk> skipped
    o.allow_unk = true;
    EXPECT_EQ(sample_token(l.data(), 5, o, nullptr), 0);
}

TEST(sample_top_k_and_distribution) {
    const int V = 6;
    std::vector<real_t> l = {9, 0.5, 2.0, 1.0, -1.0, 1.5};
    Rng rng(42, SAMPLING_STREAM);
    SampleOptions o;
    o.top_k = 1;
    for (int i = 0; i < 50; ++i) EXPECT_EQ(sample_token(l.data(), V, o, &rng), 2);  // == greedy (unk excluded)
    o.top_k = 3;  // allowed ids 1..5; top 3 = {2, 5, 3}
    for (int i = 0; i < 500; ++i) {
        const int s = sample_token(l.data(), V, o, &rng);
        EXPECT_TRUE(s == 2 || s == 5 || s == 3);
    }
    // Frequencies vs softmax(l / T) over ids 1..5, T = 0.7, 200k draws.
    o.top_k = 0;
    o.temperature = 0.7;
    const int N = 200000;
    std::vector<int> cnt(V, 0);
    for (int i = 0; i < N; ++i) cnt[sample_token(l.data(), V, o, &rng)]++;
    EXPECT_EQ(cnt[0], 0);
    double z = 0;
    for (int i = 1; i < V; ++i) z += std::exp(l[i] / 0.7);
    for (int i = 1; i < V; ++i) {
        const double p = std::exp(l[i] / 0.7) / z, f = double(cnt[i]) / N;
        // 5 standard errors of a binomial proportion.
        EXPECT_NEAR(f, p, 5 * std::sqrt(p * (1 - p) / N) + 1e-12);
    }
    // Same seed -> same draws.
    Rng a(9, SAMPLING_STREAM), b(9, SAMPLING_STREAM);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(sample_token(l.data(), V, o, &a), sample_token(l.data(), V, o, &b));
    std::string out;
    o.temperature = 0;
    EXPECT_TRUE(test::dies_with_abort([&] { sample_token(l.data(), V, o, &rng); }, &out));
}

namespace {
Model* tiny_model() {
    ModelConfig c;
    c.vocab_size = 48;
    c.max_seq_len = 8;
    c.n_layer = 2;
    c.n_embd = 16;
    c.n_head = 2;
    Model* m = model_create(c, 1, 8);
    model_init_weights(m, 21);
    // Standard init gives near-uniform logits; scale the weights so the
    // greedy choices are well separated.
    for (size_t i = 0; i < m->params_layout.num_params_padded; ++i) m->params[i] *= 20;
    return m;
}
}  // namespace

TEST(generate_reproducible_and_matches_forward) {
    Model* m = tiny_model();
    GenerateOptions g;
    g.sample.greedy = true;
    g.max_new = 12;
    g.eos_id = -1;
    const std::vector<int32_t> ctx = {1, 5, 9, 2, 33, 7, 8, 40, 11, 3};  // longer than max_seq_len = 8
    const std::vector<int32_t> a = generate_ids(m, ctx, g), b = generate_ids(m, ctx, g);
    EXPECT_EQ(a.size(), size_t(12));
    EXPECT_TRUE(a == b);
    // Each token = argmax (ids >= 1) of a direct forward over the last <= 8 tokens.
    std::vector<int32_t> seq = ctx;
    for (int32_t got : a) {
        const size_t T = std::min(seq.size(), size_t(8));
        model_forward(m, seq.data() + (seq.size() - T), nullptr, 1, int(T));
        const real_t* row = model_logits(m) + (T - 1) * 48;
        int best = 1;
        for (int i = 2; i < 48; ++i)
            if (row[i] > row[best]) best = i;
        EXPECT_EQ(got, best);
        seq.push_back(best);
    }
    // EOS stop: make the third generated token the EOS id.
    g.eos_id = a[2];
    const std::vector<int32_t> c = generate_ids(m, ctx, g);
    size_t first = 0;
    while (a[first] != a[2]) ++first;  // first occurrence of that id
    EXPECT_EQ(c.size(), first);
    EXPECT_TRUE(std::equal(c.begin(), c.end(), a.begin()));
    // Seeded sampling is reproducible; a different seed differs.
    g.sample.greedy = false;
    g.sample.temperature = 1.0;
    g.eos_id = -1;
    g.max_new = 30;
    const auto s1 = generate_ids(m, ctx, g), s2 = generate_ids(m, ctx, g);
    EXPECT_TRUE(s1 == s2);
    g.seed = 4242;
    EXPECT_TRUE(generate_ids(m, ctx, g) != s1);
    g.max_new = 0;
    EXPECT_TRUE(generate_ids(m, ctx, g).empty());
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { generate_ids(m, {1, 48}, g); }, &out));  // id out of range
    model_free(m);
}

TEST_MAIN()
