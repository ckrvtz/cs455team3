// test_golden.cpp: forward and backward vs float64 PyTorch golden files (ROADMAP §10, P2.6, P3.7).
// No Python at test time. For each case:
//   <case>_init.bin  written by bin/dump_init (config, seed, f32 init params, batch)
//   <case>_fwd.bin   written offline by dev_only/gen_golden.py (float64 PyTorch)
// Checks:
//   1. model_init_weights(seed) (times scale) reproduces the dumped f32
//      params bit-exactly in the float build, and to within float rounding
//      in the double build (init is frozen).
//   2. Forward with the dumped params: loss rel err < 1e-5 (§10 float32 vs
//      float64 criterion), per-position losses, and the logits' relative
//      L2 error, which is reported and held to < 1e-5 here too.
//   3. Backward: every parameter gradient vs PyTorch autograd (grad.<name>),
//      per-tensor relative L2 error < 1e-4 (§10 criterion).
//   4. AdamW: 5 training steps vs torch.optim.AdamW (P4.1), see check_adamw.
#include "test.h"

#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/model.h"
#include "tslm/optim.h"
#include "tslm/tensor_file.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include <unistd.h>  // truncate()

using namespace tslm;

namespace {

TensorFile load_or_die(const std::string& path) {
    TensorFile tf;
    std::string err;
    CHECK(load_tensor_file(path, &tf, &err), "%s", err.c_str());
    return tf;
}

struct Case {
    ModelConfig cfg;
    int B = 0, T = 0;
    uint64_t seed = 0;
    double scale = 1;
    TensorFile init, fwd;
};

Case load_case(const std::string& name) {
    Case c;
    c.init = load_or_die("tests/golden/" + name + "_init.bin");
    c.fwd = load_or_die("tests/golden/" + name + "_fwd.bin");
    const NamedTensor& conf = c.init.get("config");
    c.cfg.vocab_size = int(conf.i(0));
    c.cfg.max_seq_len = int(conf.i(1));
    c.cfg.n_layer = int(conf.i(2));
    c.cfg.n_embd = int(conf.i(3));
    c.cfg.n_head = int(conf.i(4));
    c.cfg.tie_embeddings = int(conf.i(5));
    c.B = int(conf.i(6));
    c.T = int(conf.i(7));
    c.seed = static_cast<uint64_t>(c.init.get("seed").i(0));
    c.scale = c.init.get("scale").f(0);
    return c;
}

void check_case(const std::string& name) {
    const Case c = load_case(name);
    Model* m = model_create(c.cfg, c.B, c.T);
    const ParamLayout& L = m->params_layout;

    // 1. Init reproducibility.
    model_init_weights(m, c.seed);
    std::vector<real_t> host(L.num_params_padded);
    copy_d2h(host.data(), m->params, host.size() * sizeof(real_t));
    int init_mismatch = 0;
    for (const TensorDesc& t : L.tensors) {
        const NamedTensor& g = c.init.get("param." + t.name);
        ASSERT_EQ(g.numel(), uint64_t(t.numel));
        for (size_t k = 0; k < t.numel; ++k) {
            const double mine = static_cast<float>(host[t.offset + k] * static_cast<real_t>(c.scale));
            const double want = g.f(k);
            const bool ok = REAL_IS_DOUBLE ? std::fabs(mine - want) <= 1e-6 * std::fabs(want) : mine == want;
            init_mismatch += !ok;
        }
    }
    EXPECT_EQ(init_mismatch, 0);

    // 2. Forward with exactly the dumped params.
    for (const TensorDesc& t : L.tensors) {
        const NamedTensor& g = c.init.get("param." + t.name);
        for (size_t k = 0; k < t.numel; ++k) host[t.offset + k] = static_cast<real_t>(g.f(k));
    }
    copy_h2d(m->params, host.data(), host.size() * sizeof(real_t));
    const size_t N = size_t(c.B) * size_t(c.T);
    std::vector<int32_t> tok(N), tgt(N);
    for (size_t i = 0; i < N; ++i) {
        tok[i] = static_cast<int32_t>(c.init.get("tokens").i(i));
        tgt[i] = static_cast<int32_t>(c.init.get("targets").i(i));
    }
    const double loss = model_forward(m, tok.data(), tgt.data(), c.B, c.T);
    const double gold = c.fwd.get("loss").f(0);
    const double loss_rel = std::fabs(loss - gold) / std::fabs(gold);

    const NamedTensor& gl = c.fwd.get("logits");
    ASSERT_EQ(gl.numel(), uint64_t(N) * uint64_t(c.cfg.vocab_size));
    std::vector<real_t> logits(gl.numel());
    copy_d2h(logits.data(), model_logits(m), logits.size() * sizeof(real_t));
    double num = 0, den = 0;
    for (uint64_t i = 0; i < gl.numel(); ++i) {
        const double d = double(logits[i]) - gl.f(i);
        num += d * d;
        den += gl.f(i) * gl.f(i);
    }
    const double logits_rel_l2 = std::sqrt(num / den);

    const NamedTensor& gpl = c.fwd.get("losses");
    std::vector<real_t> pos(N);
    copy_d2h(pos.data(), m->act(m->acts_layout.losses), N * sizeof(real_t));
    double worst_pos = 0;
    for (size_t i = 0; i < N; ++i) {
        const double e = std::fabs(double(pos[i]) - gpl.f(i)) / std::fabs(gpl.f(i));
        worst_pos = e > worst_pos ? e : worst_pos;
    }

    std::printf("    %-16s loss %.9f golden %.9f | loss rel err %.2e | per-pos max rel %.2e | logits rel-L2 %.2e\n",
                name.c_str(), loss, gold, loss_rel, worst_pos, logits_rel_l2);
    EXPECT_TRUE(loss_rel < 1e-5);
    EXPECT_TRUE(worst_pos < 1e-5);
    EXPECT_TRUE(logits_rel_l2 < 1e-5);

    // 3. Backward (P3.7): every parameter gradient vs PyTorch float64 autograd,
    //    per-tensor relative L2 error < 1e-4 (§10). Tied cases: wte's golden
    //    gradient includes both the embedding and the output-head paths.
    dev_memset(m->grads, 0, model_param_bytes(m));
    model_forward(m, tok.data(), tgt.data(), c.B, c.T);
    model_backward(m);
    std::vector<real_t> grads(L.num_params_padded);
    copy_d2h(grads.data(), m->grads, grads.size() * sizeof(real_t));
    double worst = 0;
    std::string worst_name;
    int tensors = 0;
    for (const TensorDesc& t : L.tensors) {
        const NamedTensor& gg = c.fwd.get("grad." + t.name);
        ASSERT_EQ(gg.numel(), uint64_t(t.numel));
        double num = 0, den = 0;
        for (size_t k = 0; k < t.numel; ++k) {
            const double ref = gg.f(k), d = double(grads[t.offset + k]) - ref;
            num += d * d;
            den += ref * ref;
        }
        ASSERT_TRUE(den > 0);
        const double rel = std::sqrt(num / den);
        if (rel > worst) {
            worst = rel;
            worst_name = t.name;
        }
        ++tensors;
        if (!(rel < 1e-4)) test::report(__FILE__, __LINE__, name + " grad." + t.name + " rel-L2 " + std::to_string(rel));
    }
    EXPECT_EQ(size_t(tensors), L.tensors.size());
    std::printf("    %-16s gradients: %d tensors, worst rel-L2 %.2e (%s), tol 1e-4\n", name.c_str(), tensors, worst,
                worst_name.c_str());
    model_free(m);
}

// 4. AdamW trajectory (P4.1): `steps` steps of forward -> backward ->
//    clip_grad_norm -> adamw_update on the fixed batch, from the dumped init
//    params, with the hyperparameters stored in <case>_adamw.bin. The final
//    params must match PyTorch float64 (torch.optim.AdamW) with per-tensor
//    rel-L2 < 1e-4 (§10), except the key-bias slice of qkv_b (see below).
//    Per-step loss (< 1e-5), pre-clip grad norm (< 1e-4) and LR are
//    compared too.
void check_adamw(const std::string& name) {
    const Case c = load_case(name);
    const TensorFile ad = load_or_die("tests/golden/" + name + "_adamw.bin");
    const NamedTensor& hp = ad.get("hparams");
    TrainConfig tc;
    tc.lr = hp.f(0);
    tc.min_lr_frac = hp.f(1);
    tc.warmup_steps = int(hp.f(2));
    tc.max_steps = int(hp.f(3));
    tc.beta1 = hp.f(4);
    tc.beta2 = hp.f(5);
    tc.adam_eps = hp.f(6);
    tc.weight_decay = hp.f(7);
    tc.grad_clip = hp.f(8);
    const int steps = int(hp.f(9));

    Model* m = model_create(c.cfg, c.B, c.T);
    const ParamLayout& L = m->params_layout;
    std::vector<real_t> host(L.num_params_padded, real_t(0));
    for (const TensorDesc& t : L.tensors) {
        const NamedTensor& g = c.init.get("param." + t.name);
        for (size_t k = 0; k < t.numel; ++k) host[t.offset + k] = static_cast<real_t>(g.f(k));
    }
    copy_h2d(m->params, host.data(), host.size() * sizeof(real_t));
    const size_t N = size_t(c.B) * size_t(c.T);
    std::vector<int32_t> tok(N), tgt(N);
    for (size_t i = 0; i < N; ++i) {
        tok[i] = static_cast<int32_t>(c.init.get("tokens").i(i));
        tgt[i] = static_cast<int32_t>(c.init.get("targets").i(i));
    }

    double worst_loss = 0, worst_norm = 0, worst_lr = 0;
    for (int s = 1; s <= steps; ++s) {
        dev_memset(m->grads, 0, model_param_bytes(m));
        const double loss = model_forward(m, tok.data(), tgt.data(), c.B, c.T);
        model_backward(m);
        const double norm = clip_grad_norm(m, tc.grad_clip);
        const double lr = lr_at(tc, s);
        adamw_update(m, tc, s, lr);
        const double gl = ad.get("step_loss").f(uint64_t(s - 1)), gn = ad.get("step_grad_norm").f(uint64_t(s - 1));
        const double glr = ad.get("step_lr").f(uint64_t(s - 1));
        worst_loss = std::fmax(worst_loss, std::fabs(loss - gl) / std::fabs(gl));
        worst_norm = std::fmax(worst_norm, std::fabs(norm - gn) / std::fabs(gn));
        worst_lr = std::fmax(worst_lr, std::fabs(lr - glr) / std::fabs(glr));
    }
    EXPECT_TRUE(worst_loss < 1e-5);
    EXPECT_TRUE(worst_norm < 1e-4);
    EXPECT_TRUE(worst_lr < 1e-14);  // same formula, double precision on both sides

    // Final params. Key-bias exception (docs/DECISIONS.md, structural-zero list):
    // the key slice [C,2C) of each qkv_b has an exactly zero true gradient,
    // so in float its gradient is rounding noise, which Adam's
    // m/(sqrt(v)+eps) turns into real steps, while float64 sees ~0. It has
    // no effect on the model's output (the step losses above must still
    // match). So qkv_b is compared on its query and value slices, and the
    // key slice is only held to the Adam step bound |value| <= steps * lr
    // (it starts at 0, and each Adam step moves an entry by at most about lr).
    copy_d2h(host.data(), m->params, host.size() * sizeof(real_t));
    const size_t C = size_t(c.cfg.n_embd);
    double worst = 0, worst_key = 0;
    std::string worst_name;
    for (const TensorDesc& t : L.tensors) {
        const NamedTensor& gp = ad.get("param." + t.name);
        const bool qkv_b = t.name.size() >= 5 && t.name.compare(t.name.size() - 5, 5, "qkv_b") == 0;
        double num = 0, den = 0;
        for (size_t k = 0; k < t.numel; ++k) {
            if (qkv_b && k >= C && k < 2 * C) {
                worst_key = std::fmax(worst_key, std::fabs(double(host[t.offset + k])));
                continue;
            }
            const double d = double(host[t.offset + k]) - gp.f(k);
            num += d * d;
            den += gp.f(k) * gp.f(k);
        }
        const double rel = std::sqrt(num / den);
        if (rel > worst) {
            worst = rel;
            worst_name = t.name;
        }
        if (!(rel < 1e-4)) test::report(__FILE__, __LINE__, name + " after AdamW param." + t.name + " rel-L2 " + std::to_string(rel));
    }
    const double key_bound = double(steps) * tc.lr;
    EXPECT_TRUE(worst_key <= key_bound);
    std::printf("    %-16s AdamW x%d: step loss rel %.1e, grad-norm rel %.1e, lr rel %.1e | params worst rel-L2 %.2e (%s), "
                "tol 1e-4 | key-bias max |value| %.1e (bound %.0e)\n",
                name.c_str(), steps, worst_loss, worst_norm, worst_lr, worst, worst_name.c_str(), worst_key, key_bound);
    model_free(m);
}

}  // namespace

TEST(tensor_file_round_trip_and_errors) {
    const char* tmp = std::getenv("TSLM_TEST_TMPDIR");
    const std::string path = std::string(tmp ? tmp : ".") + "/tf_roundtrip.bin";
    TensorFile tf;
    const float f[3] = {1.5f, -0.0f, 3.25e-7f};
    const double d[2] = {0.1, -1e300};
    const int32_t i32[4] = {0, -1, 2147483647, -2147483647 - 1};
    const int64_t i64[1] = {-1234567890123LL};
    tf.add_f32("a", {3}, f);
    tf.add_f64("b", {1, 2}, d);
    tf.add_i32("c", {2, 2}, i32);
    tf.add_i64("scalar", {}, i64);
    std::string err;
    ASSERT_TRUE(save_tensor_file(path, tf, &err));
    TensorFile back;
    ASSERT_TRUE(load_tensor_file(path, &back, &err));
    ASSERT_EQ(back.tensors.size(), size_t{4});
    EXPECT_EQ(back.get("a").f(2), double(3.25e-7f));
    EXPECT_TRUE(std::signbit(back.get("a").f(1)));
    EXPECT_EQ(back.get("b").f(1), -1e300);
    EXPECT_EQ(back.get("b").dims[1], uint64_t{2});
    EXPECT_EQ(back.get("c").i(3), int64_t(-2147483647 - 1));
    EXPECT_EQ(back.get("scalar").i(0), int64_t{-1234567890123LL});
    EXPECT_EQ(back.get("scalar").numel(), uint64_t{1});
    EXPECT_TRUE(back.find("nope") == nullptr);

    // Truncated file and bad magic are rejected.
    FILE* fp = std::fopen(path.c_str(), "r+b");
    ASSERT_TRUE(fp != nullptr);
    std::fseek(fp, 0, SEEK_END);
    const long size = std::ftell(fp);
    std::fclose(fp);
    ASSERT_TRUE(truncate(path.c_str(), size - 3) == 0);
    EXPECT_TRUE(!load_tensor_file(path, &back, &err));
    fp = std::fopen(path.c_str(), "r+b");
    std::fputc('X', fp);
    std::fclose(fp);
    EXPECT_TRUE(!load_tensor_file(path, &back, &err));
    EXPECT_TRUE(err.find("bad magic") != std::string::npos);
    std::remove(path.c_str());
}

TEST(golden_forward_tiny_tied) { check_case("tiny"); }

TEST(golden_forward_tiny_untied_x10) { check_case("tiny_untied_x10"); }

TEST(golden_adamw_tiny_tied) { check_adamw("tiny"); }

TEST(golden_adamw_tiny_untied_x10) { check_adamw("tiny_untied_x10"); }

TEST_MAIN()
