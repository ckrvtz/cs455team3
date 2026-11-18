// test_eval.cpp: validation pass and fixed validation batches (ROADMAP P4.4).
// DoD: val loss on a fresh model ≈ ln(V).
//   - fixed_batch: file order, rank slicing as in §4.5, loader state untouched,
//     out-of-range batch aborts
//   - evaluate = mean of model_forward over hand-built file-order batches
//   - evaluate is repeatable and leaves params / grads / m / v unchanged
//   - fresh model at V = 10000: val loss within 0.05 of ln(10000)
//   - train_run refuses val_batches larger than the validation file; training
//     columns do not depend on how often validation runs
#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/dataset.h"
#include "tslm/eval.h"
#include "tslm/model.h"
#include "tslm/token_file.h"
#include "tslm/train.h"

using namespace tslm;

namespace {

std::string tmp_dir() {
    const char* t = std::getenv("TSLM_TEST_TMPDIR");
    return std::string(t ? t : ".");
}

void ensure_comm() {
    static bool inited = false;
    if (!inited) {
        int argc = 0;
        char** argv = nullptr;
        comm_init(&argc, &argv);
        inited = true;
    }
}

uint32_t token_at(uint64_t p, uint32_t vocab) { return static_cast<uint32_t>((p * 7919 + p / 3) % vocab); }

// Token file whose token at position p is token_at(p, vocab).
std::string write_tokens(const std::string& name, uint64_t n, uint32_t vocab) {
    const std::string path = tmp_dir() + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f, "cannot write %s", path.c_str());
    TokenFileHeader h;
    h.vocab_size = vocab;
    h.num_tokens = n;
    unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, hdr);
    std::fwrite(hdr, 1, sizeof(hdr), f);
    for (uint64_t p = 0; p < n; ++p) {
        const uint32_t v = token_at(p, vocab);
        const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
        std::fwrite(b, 1, 2, f);
    }
    std::fclose(f);
    return path;
}

ModelConfig model_cfg(int V) {
    ModelConfig m;
    m.vocab_size = V;
    m.max_seq_len = 16;
    m.n_layer = 1;
    m.n_embd = 16;
    m.n_head = 2;
    return m;
}

}  // namespace

TEST(fixed_batch_file_order_and_ranks) {
    const int B = 2, T = 4;
    const std::string p = write_tokens("fixed.bin", 1 + T * 20, 50);  // 20 chunks
    DataLoader d0, d1, solo;
    std::string err;
    ASSERT_TRUE(d0.open(p, B, T, 1, 0, 2, &err));
    ASSERT_TRUE(d1.open(p, B, T, 1, 1, 2, &err));
    ASSERT_TRUE(solo.open(p, 2 * B, T, 1, 0, 1, &err));
    std::vector<int32_t> x0(B * T), y0(B * T), x1(B * T), y1(B * T), xs(2 * B * T), ys(2 * B * T);
    for (uint64_t idx = 0; idx < 5; ++idx) {  // 5 batches of gB = 4 chunks
        d0.fixed_batch(idx, x0.data(), y0.data());
        d1.fixed_batch(idx, x1.data(), y1.data());
        solo.fixed_batch(idx, xs.data(), ys.data());
        for (int r = 0; r < 2 * B; ++r) {
            const uint64_t chunk = idx * 4 + uint64_t(r);
            for (int t = 0; t < T; ++t) {
                EXPECT_EQ(uint32_t(xs[r * T + t]), token_at(chunk * T + t, 50));
                EXPECT_EQ(uint32_t(ys[r * T + t]), token_at(chunk * T + t + 1, 50));
                // Rank r' holds global rows [r'*B, (r'+1)*B).
                const std::vector<int32_t>& xr = r < B ? x0 : x1;
                EXPECT_EQ(xr[(r % B) * T + t], xs[r * T + t]);
            }
        }
    }
    // Shuffled training order is unaffected by fixed_batch calls.
    DataLoader fresh;
    ASSERT_TRUE(fresh.open(p, 2 * B, T, 1, 0, 1, &err));
    std::vector<int32_t> xa(2 * B * T), ya(2 * B * T), xb(2 * B * T), yb(2 * B * T);
    fresh.next_batch(xa.data(), ya.data());
    solo.next_batch(xb.data(), yb.data());
    EXPECT_TRUE(xa == xb && ya == yb);
    EXPECT_EQ(solo.state().step, uint64_t(1));
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { solo.fixed_batch(5, xs.data(), ys.data()); }, &out));
    EXPECT_TRUE(out.find("fixed_batch") != std::string::npos);
}

TEST(evaluate_matches_manual_mean_and_has_no_side_effects) {
    ensure_comm();
    const int B = 2, T = 8, V = 40, nb = 3;
    const std::string p = write_tokens("evalref.bin", 1 + T * 10, V);
    DataLoader val;
    std::string err;
    ASSERT_TRUE(val.open(p, B, T, 1, 0, 1, &err));
    Model* m = model_create(model_cfg(V), B, T);
    model_init_weights(m, 3);
    // Non-zero grads / m / v so "unchanged" means something.
    const size_t np = m->params_layout.num_params_padded, bytes = model_param_bytes(m);
    std::vector<real_t> fill(np);
    for (size_t i = 0; i < np; ++i) fill[i] = real_t(0.001 * double(i % 97));
    copy_h2d(m->grads, fill.data(), bytes);
    copy_h2d(m->m, fill.data(), bytes);
    copy_h2d(m->v, fill.data(), bytes);
    const std::vector<real_t> p0(m->params, m->params + np), g0(m->grads, m->grads + np), m0(m->m, m->m + np),
        v0(m->v, m->v + np);

    std::vector<int32_t> xh(B * T), yh(B * T);
    int32_t* xd = static_cast<int32_t*>(dev_malloc(B * T * sizeof(int32_t)));
    int32_t* yd = static_cast<int32_t*>(dev_malloc(B * T * sizeof(int32_t)));
    const double a = evaluate(m, val, nb, xh.data(), yh.data(), xd, yd);
    const double b = evaluate(m, val, nb, xh.data(), yh.data(), xd, yd);
    EXPECT_EQ(a, b);  // same fixed batches every time

    // Reference: batches built by hand from the file-order formula.
    double sum = 0.0;
    for (int i = 0; i < nb; ++i) {
        for (int r = 0; r < B; ++r)
            for (int t = 0; t < T; ++t) {
                const uint64_t pos = uint64_t(i * B + r) * T + uint64_t(t);
                xh[r * T + t] = int32_t(token_at(pos, V));
                yh[r * T + t] = int32_t(token_at(pos + 1, V));
            }
        sum += model_forward(m, xh.data(), yh.data(), B, T);
    }
    EXPECT_EQ(a, sum / nb);

    EXPECT_TRUE(std::memcmp(p0.data(), m->params, bytes) == 0);
    EXPECT_TRUE(std::memcmp(g0.data(), m->grads, bytes) == 0);
    EXPECT_TRUE(std::memcmp(m0.data(), m->m, bytes) == 0);
    EXPECT_TRUE(std::memcmp(v0.data(), m->v, bytes) == 0);
    dev_free(xd);
    dev_free(yd);
    model_free(m);
}

TEST(fresh_model_val_loss_is_ln_v) {
    ensure_comm();
    const int B = 4, T = 16, V = 10000;
    const std::string p = write_tokens("eval10k.bin", 1 + T * 64, V);
    DataLoader val;
    std::string err;
    ASSERT_TRUE(val.open(p, B, T, 1, 0, 1, &err));
    Model* m = model_create(model_cfg(V), B, T);
    std::vector<int32_t> xh(B * T), yh(B * T);
    int32_t* xd = static_cast<int32_t*>(dev_malloc(B * T * sizeof(int32_t)));
    int32_t* yd = static_cast<int32_t*>(dev_malloc(B * T * sizeof(int32_t)));
    for (uint64_t seed : {1ull, 2ull, 1337ull}) {
        model_init_weights(m, seed);
        const double v = evaluate(m, val, 16, xh.data(), yh.data(), xd, yd);
        std::printf("    V=10000 fresh model, seed %llu: val loss %.5f, ln V = %.5f, ppl %.1f\n",
                    static_cast<unsigned long long>(seed), v, std::log(double(V)), std::exp(v));
        EXPECT_NEAR(v, std::log(double(V)), 0.05);
    }
    dev_free(xd);
    dev_free(yd);
    model_free(m);
}

TEST(train_validation_config_checks_and_isolation) {
    ensure_comm();
    Config c;
    c.model = model_cfg(40);
    c.train.out_dir = tmp_dir() + "/evalruns";
    c.train.train_data = write_tokens("evaltrain.bin", 1 + 8 * 200, 40);
    c.train.val_data = write_tokens("evalval.bin", 1 + 8 * 10, 40);  // 10 chunks
    c.train.batch_size = 2;
    c.train.seq_len = 8;
    c.train.max_steps = 12;
    c.train.warmup_steps = 2;
    c.train.lr = 1e-2;
    c.train.log_every = 1;
    c.train.ckpt_every = 0;
    c.train.val_batches = 6;  // needs 12 chunks
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(c); }, &out));
    EXPECT_TRUE(out.find("val_batches=6") != std::string::npos);

    // Validation every step vs only at the end: identical training columns.
    c.train.val_batches = 5;
    c.train.run_name = "every";
    c.train.eval_every = 1;
    train_run(c);
    c.train.run_name = "end";
    c.train.eval_every = 1000;
    train_run(c);
    auto read = [](const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f, "cannot read %s", path.c_str());
        std::string s;
        char b[4096];
        size_t n;
        while ((n = std::fread(b, 1, sizeof(b), f)) > 0) s.append(b, n);
        std::fclose(f);
        return s;
    };
    // Compare step, tokens, train_loss, lr, grad_norm on every row.
    auto det = [](const std::string& log) {
        std::string o;
        int col = 0;
        std::string fld;
        for (char ch : log) {
            if (ch == ',' || ch == '\n') {
                if (col == 0 || col == 1 || col == 2 || col == 5 || col == 6) o += fld + "|";
                fld.clear();
                col = ch == '\n' ? 0 : col + 1;
                if (ch == '\n') o += "\n";
            } else {
                fld += ch;
            }
        }
        return o;
    };
    const std::string e = read(c.train.out_dir + "/every/log.csv"), f = read(c.train.out_dir + "/end/log.csv");
    EXPECT_EQ(det(e), det(f));
    EXPECT_TRUE(e != f);  // but the val columns differ (filled every step vs only at the end)
}

TEST_MAIN()
