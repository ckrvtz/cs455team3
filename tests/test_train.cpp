// test_train.cpp: training loop, log.csv and manifest (ROADMAP P4.2, §9).
// Runs a tiny model on a synthetic, learnable token file in TSLM_TEST_TMPDIR
// (no dataset needed) and checks:
//   - log.csv: exact frozen header; rows at step 1, every log_every, last step;
//     tokens_seen; empty val columns; finite values; phases <= step_ms
//   - manifest.txt: metadata lines present; reloading it as a config
//     reproduces the run's config exactly
//   - same seed => bit-identical loss / lr / grad_norm columns
//   - the loss falls on learnable data
//   - logged loss / grad_norm equal an independent reference loop written
//     from §9 (catches a missing grad zeroing or a reordered step)
//   - val_loss / val_ppl only on validation rows, ppl = exp(val)
//   - lr column = lr_at(step); non-finite loss / norm aborts; vocab mismatch aborts
#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/dataset.h"
#include "tslm/logging.h"
#include "tslm/optim.h"
#include "tslm/token_file.h"
#include "tslm/train.h"
#include "tslm/model.h"

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

// Periodic stream 0..P-1 repeated (P = 13 < V), so every next token is
// determined by the current one: a learnable task.
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
        const uint32_t v = static_cast<uint32_t>((p * 5) % 13);
        const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
        std::fwrite(b, 1, 2, f);
    }
    std::fclose(f);
    return path;
}

Config tiny_config(const std::string& run_name) {
    Config c;
    c.model.vocab_size = 32;
    c.model.max_seq_len = 16;
    c.model.n_layer = 1;
    c.model.n_embd = 16;
    c.model.n_head = 2;
    c.train.run_name = run_name;
    c.train.out_dir = tmp_dir() + "/runs";
    c.train.train_data = write_tokens("train_loop.bin", 1 + 8 * 400, 32);
    c.train.val_data = write_tokens("val_loop.bin", 1 + 8 * 40, 32);  // 40 chunks
    c.train.val_batches = 4;
    c.train.eval_every = 10;
    c.train.batch_size = 2;
    c.train.seq_len = 8;
    c.train.max_steps = 25;
    c.train.warmup_steps = 3;
    c.train.lr = 1e-2;
    c.train.log_every = 10;
    return c;
}

std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f, "cannot read %s", path.c_str());
    std::string s;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), f)) > 0) s.append(b, n);
    std::fclose(f);
    return s;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);
    return out;
}

// log.csv rows (without header), each split into its 14 fields.
std::vector<std::vector<std::string>> read_log(const std::string& path, std::string* header) {
    std::vector<std::string> lines = split(read_file(path), '\n');
    CHECK(!lines.empty() && lines.back().empty(), "%s must end with a newline", path.c_str());
    lines.pop_back();
    *header = lines.front();
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 1; i < lines.size(); ++i) rows.push_back(split(lines[i], ','));
    return rows;
}


// Reference: the §9 step written out directly (no logging, no timers).
// Returns the loss and pre-clip grad norm of every step.
void reference_run(const Config& cfg, std::vector<double>* loss, std::vector<double>* gnorm) {
    const TrainConfig& tc = cfg.train;
    DataLoader dl;
    std::string err;
    CHECK(dl.open(tc.train_data, tc.batch_size, tc.seq_len, tc.seed, 0, 1, &err), "%s", err.c_str());
    Model* m = model_create(cfg.model, tc.batch_size, tc.seq_len);
    model_init_weights(m, tc.seed);
    const size_t n = size_t(tc.batch_size) * tc.seq_len;
    std::vector<int32_t> x(n), y(n);
    for (int step = 1; step <= tc.max_steps; ++step) {
        dl.next_batch(x.data(), y.data());
        dev_memset(m->grads, 0, model_param_bytes(m));
        loss->push_back(model_forward(m, x.data(), y.data(), tc.batch_size, tc.seq_len));
        model_backward(m);
        gnorm->push_back(clip_grad_norm(m, tc.grad_clip));
        adamw_update(m, tc, step, lr_at(tc, step));
    }
    model_free(m);
}

}  // namespace

TEST(train_writes_log_and_manifest) {
    ensure_comm();
    const Config cfg = tiny_config("loop_a");
    const TrainSummary s = train_run(cfg);
    EXPECT_EQ(s.steps, 25);
    EXPECT_EQ(s.tokens_seen, uint64_t(25 * 2 * 8));
    EXPECT_TRUE(s.final_loss < 0.6 * s.first_loss);  // learnable data (measured 3.48 -> 1.85)
    EXPECT_NEAR(s.first_loss, std::log(32.0), 0.3);   // fresh model ~ ln(V)

    std::string header;
    const auto rows = read_log(run_dir(cfg) + "/log.csv", &header);
    EXPECT_EQ(header, std::string(LOG_CSV_HEADER));
    EXPECT_EQ(header, std::string("step,tokens_seen,train_loss,val_loss,val_ppl,lr,grad_norm,tok_per_s,step_ms,"
                                  "data_ms,fwd_ms,bwd_ms,comm_ms,opt_ms"));
    ASSERT_EQ(rows.size(), size_t(4));
    const int want_steps[4] = {1, 10, 20, 25};
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        ASSERT_EQ(r.size(), size_t(14));
        const int step = std::atoi(r[0].c_str());
        EXPECT_EQ(step, want_steps[i]);
        EXPECT_EQ(std::strtoull(r[1].c_str(), nullptr, 10), uint64_t(step) * 16);
        // Validation runs at multiples of eval_every (10) and at the last step.
        if (step % 10 == 0 || step == 25) {
            const double vl = std::strtod(r[3].c_str(), nullptr), vp = std::strtod(r[4].c_str(), nullptr);
            EXPECT_TRUE(std::isfinite(vl) && vl > 0.0);
            EXPECT_NEAR(vp, std::exp(vl), 1e-7 * vp);  // %.9g of each
        } else {
            EXPECT_TRUE(r[3].empty() && r[4].empty());
        }
        EXPECT_NEAR(std::strtod(r[5].c_str(), nullptr), lr_at(cfg.train, step), 1e-8 * lr_at(cfg.train, step));  // %.9g
        double phases = 0.0;
        for (int k = 9; k < 14; ++k) {
            const double v = std::strtod(r[k].c_str(), nullptr);
            EXPECT_TRUE(v >= 0.0);
            phases += v;
        }
        const double step_ms = std::strtod(r[8].c_str(), nullptr);
        EXPECT_TRUE(step_ms > 0.0 && phases <= step_ms + 0.01);  // phases partition the step
        EXPECT_TRUE(std::strtod(r[7].c_str(), nullptr) > 0.0);
        EXPECT_TRUE(std::isfinite(std::strtod(r[2].c_str(), nullptr)));
        EXPECT_TRUE(std::isfinite(std::strtod(r[6].c_str(), nullptr)));
    }
    EXPECT_NEAR(std::strtod(rows.back()[2].c_str(), nullptr), s.final_loss, 1e-6 * s.final_loss);
    EXPECT_NEAR(std::strtod(rows.back()[3].c_str(), nullptr), s.final_val_loss, 1e-6 * s.final_val_loss);
    EXPECT_TRUE(s.final_val_loss < 0.6 * std::log(32.0));  // val falls too (same periodic data)

    const std::string man = read_file(run_dir(cfg) + "/manifest.txt");
    for (const char* key : {"# date_utc: ", "# git: ", "# build: ", "# compiler: ", "# cxxflags: ", "# cpu: ",
                            "# cores_online: ", "# ram_mb: ", "# world_size: 1", "# train_tokens: 3201"})
        EXPECT_TRUE(man.find(key) != std::string::npos);
    Config reloaded;
    std::string err;
    ASSERT_TRUE(config_load_file(run_dir(cfg) + "/manifest.txt", &reloaded, &err));
    EXPECT_EQ(config_to_text(reloaded), config_to_text(cfg));
}

TEST(train_matches_reference_loop) {
    ensure_comm();
    Config cfg = tiny_config("ref");
    cfg.train.max_steps = 12;
    cfg.train.log_every = 4;
    train_run(cfg);
    std::vector<double> loss, gnorm;
    reference_run(cfg, &loss, &gnorm);
    std::string h;
    const auto rows = read_log(run_dir(cfg) + "/log.csv", &h);
    // Steps 1, 4, 8, 10 (validation), 12: the row after a validation still
    // matches the reference, so evaluating does not disturb training.
    ASSERT_EQ(rows.size(), size_t(5));
    char b[64];
    for (const auto& r : rows) {
        const int step = std::atoi(r[0].c_str());
        std::snprintf(b, sizeof(b), "%.9g", loss[step - 1]);
        EXPECT_EQ(r[2], std::string(b));
        std::snprintf(b, sizeof(b), "%.9g", gnorm[step - 1]);
        EXPECT_EQ(r[6], std::string(b));
    }
}

TEST(train_is_deterministic) {
    ensure_comm();
    const Config a = tiny_config("det_a"), b = tiny_config("det_b");
    const TrainSummary sa = train_run(a), sb = train_run(b);
    EXPECT_EQ(sa.final_loss, sb.final_loss);
    std::string h;
    const auto ra = read_log(run_dir(a) + "/log.csv", &h), rb = read_log(run_dir(b) + "/log.csv", &h);
    ASSERT_EQ(ra.size(), rb.size());
    for (size_t i = 0; i < ra.size(); ++i)
        for (int k : {0, 1, 2, 5, 6}) EXPECT_EQ(ra[i][k], rb[i][k]);  // step, tokens, loss, lr, grad_norm
    // A different seed changes the trajectory.
    Config c = tiny_config("det_c");
    c.train.seed = 7;
    EXPECT_TRUE(train_run(c).final_loss != sa.final_loss);
}

TEST(format_log_row_empty_val) {
    LogRow r;
    r.step = 3;
    r.tokens_seen = 48;
    r.train_loss = 1.5;
    r.lr = 1e-3;
    r.grad_norm = 0.25;
    r.tok_per_s = 100;
    r.step_ms = 2;
    EXPECT_EQ(format_log_row(r), std::string("3,48,1.5,,,0.001,0.25,100.0,2.000,0.000,0.000,0.000,0.000,0.000"));
    r.val_loss = 2.0;
    r.val_ppl = std::exp(2.0);
    EXPECT_EQ(format_log_row(r), std::string("3,48,1.5,2,7.3890561,0.001,0.25,100.0,2.000,0.000,0.000,0.000,0.000,0.000"));
}

TEST(non_finite_values_abort) {
    check_step_finite(1, 2.0, 0.5);  // finite: no abort
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([] { check_step_finite(4, std::nan(""), 1.0); }, &out));
    EXPECT_TRUE(out.find("step 4") != std::string::npos);
    EXPECT_TRUE(test::dies_with_abort([] { check_step_finite(5, 1.0, HUGE_VAL); }, &out));
    EXPECT_TRUE(test::dies_with_abort([] { check_step_finite(6, -HUGE_VAL, 1.0); }, &out));
}

TEST(vocab_mismatch_aborts) {
    ensure_comm();
    Config c = tiny_config("bad_vocab");
    c.model.vocab_size = 40;  // file says 32
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(c); }, &out));
    EXPECT_TRUE(out.find("vocab_size") != std::string::npos);
}

TEST_MAIN()
