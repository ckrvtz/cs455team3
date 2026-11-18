// test_optim.cpp: LR schedule, AdamW, clipping and the decay mask (ROADMAP P4.1).
// DoD (unit part): schedule checked at warmup start/end, mid and final.
// The golden 5-step AdamW trajectory is in test_golden.cpp. Hand values
// were computed independently from the §2.3 / §8 formulas in double.
#include "test.h"

#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/model.h"
#include "tslm/ops.h"
#include "tslm/optim.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace tslm;

namespace {
TrainConfig sched(double lr, double frac, int W, int S) {
    TrainConfig t;
    t.lr = lr;
    t.min_lr_frac = frac;
    t.warmup_steps = W;
    t.max_steps = S;
    return t;
}
}  // namespace

TEST(lr_schedule_key_points) {
    const TrainConfig t = sched(1e-3, 0.1, 10, 110);
    EXPECT_NEAR(lr_at(t, 1), 1e-4, 1e-18);                    // warmup start
    EXPECT_NEAR(lr_at(t, 5), 5e-4, 1e-18);
    EXPECT_NEAR(lr_at(t, 10), 1e-3, 1e-18);                   // warmup end = peak
    EXPECT_NEAR(lr_at(t, 11), 0.0009997779521645793, 1e-18);  // cosine just started
    EXPECT_NEAR(lr_at(t, 35), 0.0008681980515339464, 1e-18);  // quarter of decay
    EXPECT_NEAR(lr_at(t, 60), 0.00055, 1e-18);                // mid-decay
    EXPECT_NEAR(lr_at(t, 110), 1e-4, 1e-18);                  // final = min_lr_frac * peak
    EXPECT_NEAR(lr_at(t, 200), 1e-4, 1e-18);                  // past max_steps: stays at floor
    // Monotone: up during warmup, then down.
    for (int s = 1; s < 10; ++s) EXPECT_TRUE(lr_at(t, s) < lr_at(t, s + 1));
    for (int s = 10; s < 110; ++s) EXPECT_TRUE(lr_at(t, s) >= lr_at(t, s + 1));
    // No warmup: cosine from step 1.
    const TrainConfig n = sched(1e-3, 0.1, 0, 100);
    EXPECT_NEAR(lr_at(n, 1), 0.0009997779521645793, 1e-18);
    EXPECT_NEAR(lr_at(n, 100), 1e-4, 1e-18);
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { lr_at(t, 0); }, &out));
}

TEST(adamw_step_hand_case) {
    // p=1, g=0.5 then -0.25; lr=0.1, b1=0.9, b2=0.95, eps=1e-8, wd=0.1.
    real_t p = 1, m = 0, v = 0;
    real_t g = real_t(0.5);
    ops::adamw_step(&p, &g, &m, &v, 1, real_t(0.1), real_t(0.9), real_t(0.95), real_t(1e-8), real_t(0.1), 1);
    const double tol = REAL_IS_DOUBLE ? 1e-15 : 1e-6;
    EXPECT_NEAR(p, 0.890000002, tol);
    EXPECT_NEAR(m, 0.05, tol);
    EXPECT_NEAR(v, 0.0125, tol);
    g = real_t(-0.25);
    ops::adamw_step(&p, &g, &m, &v, 1, real_t(0.1), real_t(0.9), real_t(0.95), real_t(1e-8), real_t(0.1), 2);
    EXPECT_NEAR(p, 0.8542630578558273, tol);
    EXPECT_NEAR(m, 0.02, tol);
    EXPECT_NEAR(v, 0.015, tol);
    // wd = 0 and g = 0: parameter unchanged; wd > 0 and g = 0: p *= (1 - lr wd).
    real_t q[2] = {real_t(0.7), real_t(0.7)}, zg[2] = {0, 0}, zm[2] = {0, 0}, zv[2] = {0, 0};
    ops::adamw_step(&q[0], &zg[0], &zm[0], &zv[0], 1, real_t(0.1), real_t(0.9), real_t(0.95), real_t(1e-8), 0, 1);
    ops::adamw_step(&q[1], &zg[1], &zm[1], &zv[1], 1, real_t(0.1), real_t(0.9), real_t(0.95), real_t(1e-8), real_t(0.1), 1);
    EXPECT_EQ(q[0], real_t(0.7));
    EXPECT_NEAR(q[1], 0.7 * (1 - 0.1 * 0.1), REAL_IS_DOUBLE ? 1e-15 : 1e-7);
}

TEST(sumsq_and_scale) {
    const real_t x[4] = {3, 4, 0, -12};
    EXPECT_EQ(ops::sumsq_reduce(x, 4), 169.0);
    EXPECT_EQ(ops::sumsq_reduce(x, 0), 0.0);
    real_t y[3] = {1, -2, 4};
    ops::scale_inplace(y, real_t(0.5), 3);
    EXPECT_EQ(y[0], real_t(0.5));
    EXPECT_EQ(y[1], real_t(-1));
    EXPECT_EQ(y[2], real_t(2));
}

namespace {
ModelConfig tiny() {
    ModelConfig c;
    c.vocab_size = 32;
    c.max_seq_len = 8;
    c.n_layer = 1;
    c.n_embd = 16;
    c.n_head = 2;
    return c;
}
void fill_grads(Model* m, double value) {
    std::vector<real_t> g(m->params_layout.num_params_padded, real_t(0));
    for (const TensorDesc& t : m->params_layout.tensors)
        for (size_t k = 0; k < t.numel; ++k) g[t.offset + k] = static_cast<real_t>(value);
    copy_h2d(m->grads, g.data(), g.size() * sizeof(real_t));
}
}  // namespace

TEST(clip_grad_norm_behaviour) {
    Model* m = model_create(tiny(), 1, 4);
    const double n = double(m->params_layout.num_params);
    fill_grads(m, 0.01);  // norm = 0.01 sqrt(n)
    const double want = 0.01 * std::sqrt(n);
    // Below the threshold: returned norm, grads untouched.
    const double r1 = clip_grad_norm(m, want * 2);
    EXPECT_NEAR(r1, want, 1e-6 * want);
    EXPECT_EQ(m->grads[0], real_t(0.01));
    // Above: returns the pre-clip norm, and afterwards the norm equals max_norm.
    const double r2 = clip_grad_norm(m, want / 4);
    EXPECT_NEAR(r2, want, 1e-6 * want);
    EXPECT_NEAR(std::sqrt(ops::sumsq_reduce(m->grads, m->params_layout.num_params_padded)), want / 4, 1e-5 * want);
    model_free(m);
}

// Decay mask from the layout: with zero gradients, decayed tensors shrink by
// (1 - lr wd) and non-decayed ones (biases, LayerNorm, wpe) do not move;
// padding is untouched.
TEST(adamw_update_applies_decay_mask) {
    Model* m = model_create(tiny(), 1, 4);
    model_init_weights(m, 5);
    const size_t np = m->params_layout.num_params_padded;
    std::vector<real_t> before(m->params, m->params + np);
    fill_grads(m, 0.0);
    TrainConfig tc;
    tc.weight_decay = 0.1;
    adamw_update(m, tc, 1, 0.01);
    int decayed = 0;
    for (const TensorDesc& t : m->params_layout.tensors) {
        for (size_t k = 0; k < t.numel; ++k) {
            const double b = before[t.offset + k], a = m->params[t.offset + k];
            if (t.decay) ASSERT_NEAR(a, b * (1 - 0.01 * 0.1), 1e-6 * std::fabs(b) + 1e-12);
            else ASSERT_EQ(a, b);
        }
        decayed += t.decay;
    }
    EXPECT_EQ(decayed, 1 + 4 * 1);  // wte + 4 matrices per layer
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { adamw_update(m, tc, 0, 0.01); }, &out));
    model_free(m);
}

TEST_MAIN()
