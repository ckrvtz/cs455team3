// test_model_backward.cpp: model_backward wiring (ROADMAP P3.5).
// DoD: no gradient on padding/unused entries; all-zero check on tensors
// (rows) that must be zero; tied-embedding gradient accumulates from both
// paths.
//
// Also a sampled finite-difference check of every parameter tensor (first,
// last and 3 random entries) with the 4th-order 5-point stencil
// (docs/DECISIONS.md 2026-11-18). This is a wiring check; the full gradcheck
// with reporting is P3.6.
//   REAL=double: h = 1e-3, rel err < 1e-6 (floor 1e-8), as §10.
//   float: smoke check, h = 1e-2, tol 2e-2 (floor 1e-2).
#include "test.h"

#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/model.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace tslm;

namespace {

constexpr double H = REAL_IS_DOUBLE ? 1e-3 : 1e-2;
constexpr double TOL = REAL_IS_DOUBLE ? 1e-6 : 2e-2;
constexpr double FLOOR = REAL_IS_DOUBLE ? 1e-8 : 1e-2;

ModelConfig cfg(int V, int maxT, int L, int C, int NH, int tie) {
    ModelConfig m;
    m.vocab_size = V;
    m.max_seq_len = maxT;
    m.n_layer = L;
    m.n_embd = C;
    m.n_head = NH;
    m.tie_embeddings = tie;
    return m;
}

std::vector<int32_t> tokens(size_t n, int vmax, uint64_t seed) {
    Rng rng(seed, 9);
    std::vector<int32_t> t(n);
    for (auto& x : t) x = static_cast<int32_t>(rng.uniform_int(uint64_t(vmax)));
    return t;
}

void scale_params(Model* m, double s) {
    for (size_t i = 0; i < m->params_layout.num_params_padded; ++i) m->params[i] *= static_cast<real_t>(s);
}

// Forward + backward from zeroed grads; returns the loss.
double fwd_bwd(Model* m, const std::vector<int32_t>& x, const std::vector<int32_t>& y, int B, int T) {
    dev_memset(m->grads, 0, model_param_bytes(m));
    const double loss = model_forward(m, x.data(), y.data(), B, T);
    model_backward(m);
    return loss;
}

}  // namespace

// Sampled 5-point finite differences over every parameter tensor.
TEST(sampled_finite_differences) {
    struct Case { int tie; double scale; };
    for (const Case cs : {Case{1, 1.0}, Case{0, 3.0}}) {
        const ModelConfig c = cfg(32, 8, 2, 16, 2, cs.tie);  // §10 gradcheck model
        const int B = 2, T = 8;
        Model* m = model_create(c, B, T);
        model_init_weights(m, 11);
        scale_params(m, cs.scale);
        const auto x = tokens(size_t(B) * T, c.vocab_size, 1), y = tokens(size_t(B) * T, c.vocab_size, 2);
        fwd_bwd(m, x, y, B, T);
        const std::vector<real_t> g(m->grads, m->grads + m->params_layout.num_params_padded);
        auto loss_at = [&](size_t i, double d) {
            const real_t orig = m->params[i];
            m->params[i] = static_cast<real_t>(double(orig) + d);
            const double l = model_forward(m, x.data(), y.data(), B, T);
            m->params[i] = orig;
            return l;
        };
        Rng pick(5, 0);
        double worst = 0;
        int bad = 0, checked = 0;
        for (const TensorDesc& t : m->params_layout.tensors) {
            std::set<size_t> idx = {0, t.numel - 1};
            while (idx.size() < 5 && idx.size() < t.numel) idx.insert(size_t(pick.uniform_int(t.numel)));
            // Key biases have a structurally zero gradient (softmax shift
            // invariance); FD cannot confirm "exactly 0" to the 1e-8 floor
            // (loss rounding alone is ~1e-15). They are checked directly in
            // structurally_zero_gradients instead.
            const bool is_qkv_b = t.name.size() >= 5 && t.name.compare(t.name.size() - 5, 5, "qkv_b") == 0;
            for (size_t k : idx) {
                if (is_qkv_b && k >= size_t(c.n_embd) && k < 2 * size_t(c.n_embd)) continue;
                const size_t i = t.offset + k;
                // 5-point stencil, differences first: subtracting nearly equal
                // losses is exact, so a loss that does not depend on the entry
                // gives exactly 0 (summing -L + 8L - 8L + L would not).
                const double num =
                    (8.0 * (loss_at(i, H) - loss_at(i, -H)) - (loss_at(i, 2 * H) - loss_at(i, -2 * H))) / (12 * H);
                const double a = g[i];
                const double rel = std::fabs(a - num) / std::fmax(std::fmax(std::fabs(a), std::fabs(num)), FLOOR);
                worst = rel > worst ? rel : worst;
                ++checked;
                if (!(rel < TOL) && bad++ < 4) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf), "tied=%d %s[%zu]: analytic %.10g numeric %.10g rel %.3g", cs.tie,
                                  t.name.c_str(), k, a, num, rel);
                    test::report(__FILE__, __LINE__, buf);
                }
            }
        }
        std::printf("    tied=%d scale=%.0f: %d sampled entries over %zu tensors, worst rel err %.2e (tol %.0e)\n",
                    cs.tie, cs.scale, checked, m->params_layout.tensors.size(), worst, TOL);
        model_free(m);
    }
}

// DoD: padding entries and unused rows get exactly zero gradient.
TEST(padding_and_unused_entries_are_zero) {
    // Odd sizes so the flat buffer has padding; T < max_seq_len so some wpe
    // rows are unused; tokens drawn from ids 0..9 only, so most wte rows are
    // unused by the embedding.
    const int B = 2, T = 5, used_vocab = 10;
    for (int tie = 0; tie <= 1; ++tie) {
        const ModelConfig c = cfg(37, 7, 2, 12, 3, tie);
        Model* m = model_create(c, B, T);
        model_init_weights(m, 3);
        const ParamLayout& L = m->params_layout;
        ASSERT_TRUE(L.num_params_padded > L.num_params);
        const auto x = tokens(size_t(B) * T, used_vocab, 4), y = tokens(size_t(B) * T, c.vocab_size, 5);
        fwd_bwd(m, x, y, B, T);
        const std::set<int32_t> present(x.begin(), x.end());

        // Padding: every element not covered by a tensor is exactly zero.
        std::vector<bool> covered(L.num_params_padded, false);
        for (const TensorDesc& t : L.tensors)
            for (size_t k = 0; k < t.numel; ++k) covered[t.offset + k] = true;
        int pad_nonzero = 0, pad_count = 0;
        for (size_t i = 0; i < L.num_params_padded; ++i)
            if (!covered[i]) {
                ++pad_count;
                pad_nonzero += m->grads[i] != real_t(0);
            }
        EXPECT_TRUE(pad_count > 0);
        EXPECT_EQ(pad_nonzero, 0);

        // wpe rows >= T are never used.
        const real_t* gwpe = m->grad(L.wpe);
        for (int t = T; t < c.max_seq_len; ++t)
            for (int k = 0; k < c.n_embd; ++k) ASSERT_EQ(gwpe[size_t(t) * c.n_embd + k], real_t(0));
        // Used wpe rows do get gradient.
        bool any = false;
        for (int k = 0; k < c.n_embd; ++k) any = any || gwpe[k] != real_t(0);
        EXPECT_TRUE(any);

        // wte rows of tokens absent from the batch: zero when untied (embedding
        // is the only path); nonzero when tied (the head reaches every row).
        const real_t* gwte = m->grad(L.wte);
        int absent_zero_rows = 0, absent_rows = 0;
        for (int v = 0; v < c.vocab_size; ++v) {
            if (present.count(v)) continue;
            ++absent_rows;
            bool zero = true;
            for (int k = 0; k < c.n_embd; ++k) zero = zero && gwte[size_t(v) * c.n_embd + k] == real_t(0);
            absent_zero_rows += zero;
        }
        if (tie) EXPECT_EQ(absent_zero_rows, 0);
        else EXPECT_EQ(absent_zero_rows, absent_rows);
        std::printf("    tied=%d: %d padding entries zero; %d unused wpe rows zero; %d/%d absent-token wte rows zero\n", tie,
                    pad_count, c.max_seq_len - T, absent_zero_rows, absent_rows);
        model_free(m);
    }
}

// DoD "tensors that must be zero": the key-bias slice of every qkv_b. Adding a
// constant vector to every key shifts all of a query's scores by the same
// amount, which softmax ignores, so dL/d(key bias) = 0 exactly. Its computed
// gradient must be negligible next to the query/value bias gradients of
// the same tensor.
TEST(structurally_zero_gradients) {
    const int B = 2, T = 8;
    for (int tie = 0; tie <= 1; ++tie) {
        const ModelConfig c = cfg(32, 8, 2, 16, 2, tie);
        Model* m = model_create(c, B, T);
        model_init_weights(m, 13);
        scale_params(m, 3.0);  // non-trivial attention
        const auto x = tokens(size_t(B) * T, 32, 14), y = tokens(size_t(B) * T, 32, 15);
        fwd_bwd(m, x, y, B, T);
        for (const LayerParamIdx& p : m->params_layout.layers) {
            const real_t* g = m->grad(p.qkv_b);
            double kmax = 0, qvmax = 0;
            for (int k = 0; k < 3 * c.n_embd; ++k) {
                const double v = std::fabs(double(g[k]));
                if (k >= c.n_embd && k < 2 * c.n_embd) kmax = v > kmax ? v : kmax;
                else qvmax = v > qvmax ? v : qvmax;
            }
            const double ratio = kmax / qvmax;
            std::printf("    tied=%d: max |d key-bias| = %.2e vs max |d q/v-bias| = %.2e (ratio %.1e)\n", tie, kmax,
                        qvmax, ratio);
            EXPECT_TRUE(qvmax > 0);
            EXPECT_TRUE(ratio < (REAL_IS_DOUBLE ? 1e-12 : 1e-5));
        }
        model_free(m);
    }
}

// Tied wte gradient == untied (wte gradient + head_w gradient) when head_w == wte.
TEST(tied_gradient_sums_both_paths) {
    const int B = 2, T = 6;
    Model* tied = model_create(cfg(32, 8, 2, 16, 2, 1), B, T);
    Model* untied = model_create(cfg(32, 8, 2, 16, 2, 0), B, T);
    model_init_weights(tied, 21);
    model_init_weights(untied, 21);  // same streams for the shared tensors
    const ParamLayout& Pt = tied->params_layout;
    const ParamLayout& Pu = untied->params_layout;
    // Make the untied model compute exactly the tied function: head_w := wte.
    const size_t wte_n = Pt[Pt.wte].numel;
    std::memcpy(untied->param(Pu.head_w), untied->param(Pu.wte), wte_n * sizeof(real_t));
    for (const TensorDesc& t : Pt.tensors)  // other tensors identical by construction
        ASSERT_TRUE(std::memcmp(tied->param(Pt.find(t.name)), untied->param(Pu.find(t.name)),
                                t.numel * sizeof(real_t)) == 0);
    const auto x = tokens(size_t(B) * T, 32, 6), y = tokens(size_t(B) * T, 32, 7);
    const double lt = fwd_bwd(tied, x, y, B, T), lu = fwd_bwd(untied, x, y, B, T);
    EXPECT_EQ(lt, lu);  // identical forward
    const real_t* gt = tied->grad(Pt.wte);
    const real_t* ge = untied->grad(Pu.wte);
    const real_t* gh = untied->grad(Pu.head_w);
    double worst = 0;
    for (size_t i = 0; i < wte_n; ++i) {
        const double want = double(ge[i]) + double(gh[i]);
        const double err = std::fabs(double(gt[i]) - want) / std::fmax(std::fabs(want), 1e-12);
        worst = err > worst ? err : worst;
    }
    std::printf("    tied wte grad vs untied (wte + head_w) grads: worst rel diff %.2e\n", worst);
    EXPECT_TRUE(worst < (REAL_IS_DOUBLE ? 1e-12 : 1e-5));
    // Every other tensor's gradient is bit-identical between the two models.
    for (const TensorDesc& t : Pt.tensors) {
        if (t.name == "wte") continue;
        ASSERT_TRUE(std::memcmp(tied->grad(Pt.find(t.name)), untied->grad(Pu.find(t.name)), t.numel * sizeof(real_t)) ==
                    0);
    }
    model_free(tied);
    model_free(untied);
}

TEST(accumulates_and_leaves_params_unchanged) {
    const int B = 2, T = 8;
    Model* m = model_create(cfg(32, 8, 2, 16, 2, 1), B, T);
    model_init_weights(m, 8);
    const auto x = tokens(size_t(B) * T, 32, 8), y = tokens(size_t(B) * T, 32, 9);
    const size_t n = m->params_layout.num_params_padded;
    const std::vector<real_t> p0(m->params, m->params + n);
    fwd_bwd(m, x, y, B, T);
    const std::vector<real_t> once(m->grads, m->grads + n);
    model_backward(m);  // no zeroing in between: gradients add up
    for (size_t i = 0; i < n; ++i)
        ASSERT_NEAR(double(m->grads[i]), 2.0 * double(once[i]), 1e-5 * std::fmax(1e-3, std::fabs(2.0 * double(once[i]))));
    EXPECT_TRUE(std::memcmp(p0.data(), m->params, n * sizeof(real_t)) == 0);
    model_free(m);
}

// Runtime batch smaller than capacity: backward uses the last forward's shape.
TEST(runtime_shape_below_capacity) {
    const ModelConfig c = cfg(32, 8, 2, 16, 2, 1);
    Model* big = model_create(c, 3, 8);
    Model* exact = model_create(c, 2, 5);
    model_init_weights(big, 4);
    model_init_weights(exact, 4);
    const auto x = tokens(10, 32, 10), y = tokens(10, 32, 11);
    fwd_bwd(big, x, y, 2, 5);
    fwd_bwd(exact, x, y, 2, 5);
    EXPECT_TRUE(std::memcmp(big->grads, exact->grads, model_param_bytes(big)) == 0);
    model_free(big);
    model_free(exact);
}

TEST(misuse_aborts) {
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            Model* m = model_create(cfg(32, 8, 1, 16, 2, 1), 1, 4);
            model_backward(m);
        },
        &out));
    EXPECT_TRUE(out.find("no forward pass") != std::string::npos);
    out.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            Model* m = model_create(cfg(32, 8, 1, 16, 2, 1), 1, 4);
            const int32_t x[4] = {1, 2, 3, 4};
            model_forward(m, x, nullptr, 1, 4);
            model_backward(m);
        },
        &out));
    EXPECT_TRUE(out.find("no targets") != std::string::npos);
}

TEST_MAIN()
