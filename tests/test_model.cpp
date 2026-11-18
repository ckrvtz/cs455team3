// test_model.cpp: model_create / model_init_weights / model_forward (ROADMAP P2.5).
// DoD: init loss within +-0.15 of ln(V) on real data; activation arena
// printed in MB (model_summary).
//
// The forward pass is checked against an independent, deliberately naive
// reference model written here: every step in double, straight from the
// §2.1 / §6.3 formulas, sharing no code with the ops. The tiny config is the
// §10 gradcheck model (L=2, C=16, NH=2, V=32, T=8, B=2).
#include "test.h"

#include "tslm/build_config.h"
#include "tslm/common.h"
#include "tslm/config.h"
#include "tslm/dataset.h"
#include "tslm/model.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tslm;

namespace {

ModelConfig tiny_cfg(int tie) {
    ModelConfig c;
    c.vocab_size = 32;
    c.max_seq_len = 8;
    c.n_layer = 2;
    c.n_embd = 16;
    c.n_head = 2;
    c.tie_embeddings = tie;
    return c;
}

ModelConfig preset(const char* path) {
    Config c;
    std::string err;
    CHECK(config_load_file(path, &c, &err), "%s", err.c_str());
    return c.model;
}

std::vector<int32_t> random_tokens(size_t n, int V, uint64_t seed) {
    Rng rng(seed, 3);
    std::vector<int32_t> t(n);
    for (auto& x : t) x = static_cast<int32_t>(rng.uniform_int(uint64_t(V)));
    return t;
}

// ---- naive double-precision reference model -------------------------------

using Vec = std::vector<double>;

struct RefParams {
    const Model* m;
    Vec get(int idx) const {
        const TensorDesc& t = m->params_layout[idx];
        return Vec(m->params + t.offset, m->params + t.offset + t.numel);
    }
};

Vec ref_layernorm(const Vec& x, const Vec& g, const Vec& b, int N, int C) {
    Vec y(x.size());
    for (int n = 0; n < N; ++n) {
        double mu = 0, var = 0;
        for (int c = 0; c < C; ++c) mu += x[size_t(n) * C + c];
        mu /= C;
        for (int c = 0; c < C; ++c) var += (x[size_t(n) * C + c] - mu) * (x[size_t(n) * C + c] - mu);
        var /= C;
        for (int c = 0; c < C; ++c)
            y[size_t(n) * C + c] = (x[size_t(n) * C + c] - mu) / std::sqrt(var + 1e-5) * g[size_t(c)] + b[size_t(c)];
    }
    return y;
}

// y[n,o] = sum_c x[n,c] W[o,c] + bias[o]
Vec ref_linear(const Vec& x, const Vec& W, const Vec* bias, int N, int C, int OC) {
    Vec y(size_t(N) * OC);
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < OC; ++o) {
            double s = bias ? (*bias)[size_t(o)] : 0.0;
            for (int c = 0; c < C; ++c) s += x[size_t(n) * C + c] * W[size_t(o) * C + c];
            y[size_t(n) * OC + o] = s;
        }
    return y;
}

Vec ref_attention(const Vec& qkv, int B, int T, int C, int NH) {
    const int hs = C / NH;
    Vec y(size_t(B) * T * C, 0.0);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h)
            for (int t = 0; t < T; ++t) {
                Vec w(size_t(t) + 1);
                double mx = -1e300, sum = 0;
                for (int t2 = 0; t2 <= t; ++t2) {
                    double d = 0;
                    for (int i = 0; i < hs; ++i)
                        d += qkv[(size_t(b) * T + t) * 3 * C + h * hs + i] *
                             qkv[(size_t(b) * T + t2) * 3 * C + C + h * hs + i];
                    w[size_t(t2)] = d / std::sqrt(double(hs));
                    mx = w[size_t(t2)] > mx ? w[size_t(t2)] : mx;
                }
                for (int t2 = 0; t2 <= t; ++t2) sum += (w[size_t(t2)] = std::exp(w[size_t(t2)] - mx));
                for (int t2 = 0; t2 <= t; ++t2)
                    for (int i = 0; i < hs; ++i)
                        y[(size_t(b) * T + t) * C + h * hs + i] +=
                            w[size_t(t2)] / sum * qkv[(size_t(b) * T + t2) * 3 * C + 2 * C + h * hs + i];
            }
    return y;
}

// Returns logits [N, V]; *loss = mean cross-entropy.
Vec ref_forward(const Model* m, const std::vector<int32_t>& tok, const std::vector<int32_t>& tgt, int B, int T,
                double* loss) {
    const ModelConfig& c = m->cfg;
    const ParamLayout& P = m->params_layout;
    const RefParams rp{m};
    const int C = c.n_embd, V = c.vocab_size, N = B * T;
    const Vec wte = rp.get(P.wte), wpe = rp.get(P.wpe);
    Vec x(size_t(N) * C);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < C; ++k) x[size_t(n) * C + k] = wte[size_t(tok[size_t(n)]) * C + k] + wpe[size_t(n % T) * C + k];
    for (int l = 0; l < c.n_layer; ++l) {
        const LayerParamIdx& p = P.layers[size_t(l)];
        const Vec qb = rp.get(p.qkv_b), pb = rp.get(p.proj_b), fb = rp.get(p.fc_b), fpb = rp.get(p.fcproj_b);
        Vec h = ref_layernorm(x, rp.get(p.ln1_g), rp.get(p.ln1_b), N, C);
        h = ref_attention(ref_linear(h, rp.get(p.qkv_w), &qb, N, C, 3 * C), B, T, C, c.n_head);
        h = ref_linear(h, rp.get(p.proj_w), &pb, N, C, C);
        for (size_t i = 0; i < x.size(); ++i) x[i] += h[i];
        h = ref_layernorm(x, rp.get(p.ln2_g), rp.get(p.ln2_b), N, C);
        h = ref_linear(h, rp.get(p.fc_w), &fb, N, C, 4 * C);
        for (double& u : h) u = 0.5 * u * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (u + 0.044715 * u * u * u)));
        h = ref_linear(h, rp.get(p.fcproj_w), &fpb, N, 4 * C, C);
        for (size_t i = 0; i < x.size(); ++i) x[i] += h[i];
    }
    x = ref_layernorm(x, rp.get(P.lnf_g), rp.get(P.lnf_b), N, C);
    const Vec logits = ref_linear(x, rp.get(P.head_w), nullptr, N, C, V);
    double total = 0;
    for (int n = 0; n < N; ++n) {
        double mx = -1e300, s = 0;
        for (int v = 0; v < V; ++v) mx = logits[size_t(n) * V + v] > mx ? logits[size_t(n) * V + v] : mx;
        for (int v = 0; v < V; ++v) s += std::exp(logits[size_t(n) * V + v] - mx);
        total += std::log(s) - (logits[size_t(n) * V + size_t(tgt[size_t(n)])] - mx);
    }
    *loss = total / N;
    return logits;
}

}  // namespace

TEST(create_layouts_and_summary) {
    for (const char* path : {"configs/smoke.cfg", "configs/small.cfg", "configs/base.cfg"}) {
        Config c;
        std::string err;
        ASSERT_TRUE(config_load_file(path, &c, &err));
        // Report the arena for the preset's real training capacity from the
        // layout alone (base would be ~1.5 GB if allocated); allocate a model
        // at a small capacity to check the accessors and summary.
        const ActLayout full = make_act_layout(c.model, c.train.batch_size, c.train.seq_len);
        std::printf("    %s: activation arena at B=%d T=%d would be %.1f MB\n", path, c.train.batch_size,
                    c.train.seq_len, double(full.num_elems * sizeof(real_t)) / (1024.0 * 1024.0));
        Model* m = model_create(c.model, 1, 8);
        std::printf("      %s\n", model_summary(m).c_str());
        EXPECT_EQ(model_act_bytes(m), m->acts_layout.num_elems * sizeof(real_t));
        EXPECT_EQ(m->params_layout.num_params_padded * sizeof(real_t), model_param_bytes(m));
        EXPECT_EQ(m->B_cap, 1);
        EXPECT_EQ(m->T_cap, 8);
        model_free(m);
    }
}

TEST(init_rules_and_statistics) {
    const ModelConfig cfg = preset("configs/small.cfg");
    Model* m = model_create(cfg, 1, 8);
    model_init_weights(m, 1337);
    const ParamLayout& L = m->params_layout;
    const double sd_resid = 0.02 / std::sqrt(2.0 * cfg.n_layer);
    for (size_t i = 0; i < L.tensors.size(); ++i) {
        const TensorDesc& t = L.tensors[i];
        const real_t* p = m->params + t.offset;
        const std::string& n = t.name;
        const bool bias = n.size() > 2 && n.compare(n.size() - 2, 2, "_b") == 0;
        const bool gain = n.size() > 2 && n.compare(n.size() - 2, 2, "_g") == 0;
        if (bias || gain) {
            const real_t want = gain ? real_t(1) : real_t(0);
            for (size_t k = 0; k < t.numel; ++k) ASSERT_EQ(p[k], want);
            continue;
        }
        double s = 0, ss = 0;
        for (size_t k = 0; k < t.numel; ++k) {
            s += p[k];
            ss += double(p[k]) * p[k];
        }
        const double mean = s / double(t.numel), sd = std::sqrt(ss / double(t.numel) - mean * mean);
        const bool resid = n.size() >= 6 && n.compare(n.size() - 6, 6, "proj_w") == 0;
        const double want_sd = resid ? sd_resid : 0.02;
        // 4-sigma sampling bounds for mean and std of numel normal samples.
        const double se = want_sd / std::sqrt(double(t.numel));
        if (!(std::fabs(mean) <= 4 * se) || !(std::fabs(sd - want_sd) <= 4 * want_sd / std::sqrt(2.0 * double(t.numel))))
            test::report(__FILE__, __LINE__, "init statistics off for " + n + ": mean " + std::to_string(mean) +
                                                 " sd " + std::to_string(sd) + " want sd " + std::to_string(want_sd));
    }
    // Padding is zero.
    size_t covered = 0;
    for (const TensorDesc& t : L.tensors) covered += t.numel;
    EXPECT_EQ(covered, L.num_params);
    // Tensor i uses Rng(seed, stream=i): first value of wte and of h1.fc_w.
    Rng r0(1337, uint64_t(L.wte));
    EXPECT_EQ(m->param(L.wte)[0], static_cast<real_t>(r0.normal(0.0, 0.02)));
    const int fc1 = L.layers[1].fc_w;
    Rng r1(1337, uint64_t(fc1));
    EXPECT_EQ(m->param(fc1)[0], static_cast<real_t>(r1.normal(0.0, 0.02)));
    model_free(m);
}

TEST(init_deterministic_and_seed_sensitive) {
    const ModelConfig cfg = tiny_cfg(0);
    Model* a = model_create(cfg, 1, 4);
    Model* b = model_create(cfg, 1, 4);
    Model* c = model_create(cfg, 1, 4);
    model_init_weights(a, 7);
    model_init_weights(b, 7);
    model_init_weights(c, 8);
    const size_t bytes = model_param_bytes(a);
    EXPECT_TRUE(std::memcmp(a->params, b->params, bytes) == 0);
    EXPECT_TRUE(std::memcmp(a->params, c->params, bytes) != 0);
    model_free(a);
    model_free(b);
    model_free(c);
}

// Wiring check against the naive reference: tied and untied.
TEST(forward_matches_naive_reference) {
    for (int tie = 1; tie >= 0; --tie) {
        const ModelConfig cfg = tiny_cfg(tie);
        const int B = 2, T = 8, V = cfg.vocab_size;
        Model* m = model_create(cfg, B, T);
        model_init_weights(m, 42);
        // Larger-than-default weights so every path contributes visibly.
        for (size_t i = 0; i < m->params_layout.num_params_padded; ++i) m->params[i] *= real_t(10);
        const auto tok = random_tokens(size_t(B) * T, V, 1), tgt = random_tokens(size_t(B) * T, V, 2);
        const double loss = model_forward(m, tok.data(), tgt.data(), B, T);
        double ref_loss = 0;
        const Vec ref = ref_forward(m, tok, tgt, B, T, &ref_loss);
        const real_t* logits = model_logits(m);
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double e = std::fabs(double(logits[i]) - ref[i]) / (1.0 + std::fabs(ref[i]));
            worst = e > worst ? e : worst;
        }
        std::printf("    tied=%d: loss %.7f ref %.7f, max rel logit err %.3g\n", tie, loss, ref_loss, worst);
        EXPECT_TRUE(worst <= 1e-4);
        EXPECT_NEAR(loss, ref_loss, 1e-4 * (1.0 + ref_loss));
        model_free(m);
    }
}

// Smaller runtime B, T than capacity gives bit-identical results to a model
// whose capacity is exactly that shape.
TEST(runtime_shape_below_capacity) {
    const ModelConfig cfg = tiny_cfg(1);
    Model* big = model_create(cfg, 3, 8);
    Model* exact = model_create(cfg, 2, 5);
    model_init_weights(big, 3);
    model_init_weights(exact, 3);
    const auto tok = random_tokens(10, cfg.vocab_size, 4), tgt = random_tokens(10, cfg.vocab_size, 5);
    const double l1 = model_forward(big, tok.data(), tgt.data(), 2, 5);
    const double l2 = model_forward(exact, tok.data(), tgt.data(), 2, 5);
    EXPECT_EQ(l1, l2);
    EXPECT_TRUE(std::memcmp(model_logits(big), model_logits(exact), 10 * size_t(cfg.vocab_size) * sizeof(real_t)) == 0);
    model_free(big);
    model_free(exact);
}

TEST(forward_without_targets) {
    const ModelConfig cfg = tiny_cfg(1);
    Model* m = model_create(cfg, 2, 8);
    model_init_weights(m, 9);
    const auto tok = random_tokens(16, cfg.vocab_size, 6), tgt = random_tokens(16, cfg.vocab_size, 7);
    model_forward(m, tok.data(), tgt.data(), 2, 8);
    const std::vector<real_t> with(model_logits(m), model_logits(m) + 16 * cfg.vocab_size);
    const double none = model_forward(m, tok.data(), nullptr, 2, 8);
    EXPECT_TRUE(std::isnan(none));
    EXPECT_TRUE(std::memcmp(with.data(), model_logits(m), with.size() * sizeof(real_t)) == 0);
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { model_forward(m, tok.data(), nullptr, 3, 8); }, &out));
    EXPECT_TRUE(out.find("outside capacity") != std::string::npos);
    model_free(m);
}

// At init the loss is close to ln V (logits are small): random tokens.
TEST(init_loss_random_tokens) {
    const ModelConfig cfg = preset("configs/smoke.cfg");
    const int B = 4, T = 32;
    Model* m = model_create(cfg, B, T);
    model_init_weights(m, 1337);
    const auto tok = random_tokens(size_t(B) * T, cfg.vocab_size, 8), tgt = random_tokens(size_t(B) * T, cfg.vocab_size, 9);
    const double loss = model_forward(m, tok.data(), tgt.data(), B, T);
    std::printf("    smoke init loss on random tokens: %.4f (ln V = %.4f)\n", loss, std::log(double(cfg.vocab_size)));
    EXPECT_NEAR(loss, std::log(double(cfg.vocab_size)), 0.15);
    model_free(m);
}

// ROADMAP §10 / P2.5 DoD: fresh `base` model on real data, loss within
// +-0.15 of ln(10000). Uses data/val.bin when prepare_data has been run.
TEST(init_loss_real_data_base_if_present) {
    FILE* f = std::fopen("data/val.bin", "rb");
    if (!f) {
        std::printf("    data/val.bin not present: real-data init-loss check not done\n");
        return;
    }
    std::fclose(f);
    const ModelConfig cfg = preset("configs/base.cfg");
    const bool debug = std::strstr(build_description(), "mode=debug") != nullptr;
    const int B = debug ? 1 : 4, T = 256, batches = debug ? 1 : 2;  // keep the sanitizer build fast
    DataLoader dl;
    std::string err;
    ASSERT_TRUE(dl.open("data/val.bin", B, T, 1337, 0, 1, &err));
    Model* m = model_create(cfg, B, T);
    std::printf("    %s\n", model_summary(m).c_str());
    model_init_weights(m, 1337);
    std::vector<int32_t> x(size_t(B) * T), y(size_t(B) * T);
    double sum = 0;
    for (int i = 0; i < batches; ++i) {
        dl.next_batch(x.data(), y.data());
        sum += model_forward(m, x.data(), y.data(), B, T);
    }
    const double loss = sum / batches;
    std::printf("    base init loss on data/val.bin (%d x %d x %d tokens): %.4f, ln(10000) = %.4f\n", batches, B, T,
                loss, std::log(10000.0));
    EXPECT_NEAR(loss, std::log(10000.0), 0.15);
    model_free(m);
}

TEST_MAIN()
