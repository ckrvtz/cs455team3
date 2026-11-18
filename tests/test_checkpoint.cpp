// test_checkpoint.cpp: checkpoint format, validation and resume (ROADMAP
// P4.3, §9). DoD: a resumed run is bit-exact with an uninterrupted one.
//   - save/load round trip of params, m, v and every header field
//   - corrupt / truncated / mismatched files are refused with a message
//   - a checkpoint with the other element size (float <-> double) loads with
//     an element-wise conversion
//   - resume: run 20 steps with checkpoints at 10 and 20; resuming from
//     ckpt_10 (in a new directory, and in place) reproduces ckpt_20 byte for
//     byte and the same log.csv loss / lr / grad_norm values. The data file
//     has 15 steps per epoch, so the resumed segment crosses an epoch boundary.
//   - resume with a different seed / model, or with nothing left to run, aborts
#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tslm/backend.h"
#include "tslm/checkpoint.h"
#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/model.h"
#include "tslm/optim.h"
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

void write_file(const std::string& path, const std::string& s) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f, "cannot write %s", path.c_str());
    CHECK(std::fwrite(s.data(), 1, s.size(), f) == s.size(), "write failed");
    std::fclose(f);
}

bool exists(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}

// Learnable periodic stream, 1 + 8*30 tokens: 30 chunks of T=8 -> 15 steps
// per epoch at B=2.
std::string write_tokens(const std::string& name) {
    const std::string path = tmp_dir() + "/" + name;
    const uint64_t n = 1 + 8 * 30;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f, "cannot write %s", path.c_str());
    TokenFileHeader h;
    h.vocab_size = 32;
    h.num_tokens = n;
    unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, hdr);
    std::fwrite(hdr, 1, sizeof(hdr), f);
    for (uint64_t p = 0; p < n; ++p) {
        const unsigned char b[2] = {static_cast<unsigned char>((p * 5) % 13), 0};
        std::fwrite(b, 1, 2, f);
    }
    std::fclose(f);
    return path;
}

ModelConfig tiny_model() {
    ModelConfig m;
    m.vocab_size = 32;
    m.max_seq_len = 16;
    m.n_layer = 1;
    m.n_embd = 16;
    m.n_head = 2;
    return m;
}

Config tiny_config(const std::string& run_name) {
    Config c;
    c.model = tiny_model();
    c.train.run_name = run_name;
    c.train.out_dir = tmp_dir() + "/ckruns";
    c.train.train_data = write_tokens("ckpt_train.bin");
    c.train.val_data = write_tokens("ckpt_val.bin");
    c.train.val_batches = 4;
    c.train.batch_size = 2;
    c.train.seq_len = 8;
    c.train.max_steps = 20;
    c.train.warmup_steps = 3;
    c.train.lr = 1e-2;
    c.train.log_every = 5;
    c.train.ckpt_every = 10;
    return c;
}

// A model with non-trivial params, m and v (two optimizer steps on fake grads).
Model* trained_model() {
    Model* m = model_create(tiny_model(), 2, 8);
    model_init_weights(m, 11);
    const size_t np = m->params_layout.num_params_padded;
    std::vector<real_t> g(np, real_t(0));
    for (const TensorDesc& t : m->params_layout.tensors)
        for (size_t k = 0; k < t.numel; ++k) g[t.offset + k] = real_t(std::sin(0.37 * double(t.offset + k)));
    TrainConfig tc;
    for (int s = 1; s <= 2; ++s) {
        copy_h2d(m->grads, g.data(), np * sizeof(real_t));
        adamw_update(m, tc, s, 1e-2);
    }
    return m;
}

bool same_buffers(const Model* a, const Model* b) {
    const size_t bytes = model_param_bytes(a);
    return std::memcmp(a->params, b->params, bytes) == 0 && std::memcmp(a->m, b->m, bytes) == 0 &&
           std::memcmp(a->v, b->v, bytes) == 0;
}

std::string ck_path(const std::string& name) { return tmp_dir() + "/" + name; }

// log.csv rows reduced to the deterministic columns step, tokens_seen,
// train_loss, lr, grad_norm.
std::string det_columns(const std::string& log) {
    std::string out;
    size_t pos = 0;
    while (pos < log.size()) {
        size_t nl = log.find('\n', pos);
        if (nl == std::string::npos) nl = log.size();
        const std::string line = log.substr(pos, nl - pos);
        pos = nl + 1;
        int col = 0;
        std::string f;
        for (size_t i = 0; i <= line.size(); ++i) {
            if (i == line.size() || line[i] == ',') {
                if (col == 0 || col == 1 || col == 2 || col == 5 || col == 6) out += f + "|";
                f.clear();
                ++col;
            } else {
                f += line[i];
            }
        }
        out += "\n";
    }
    return out;
}

}  // namespace

TEST(save_load_round_trip) {
    Model* a = trained_model();
    CheckpointInfo w;
    w.step = 1234;
    w.tokens_seen = 98765;
    w.loader.epoch = 3;
    w.loader.step = 7;
    w.seed = 42;
    w.global_batch = 2;
    w.seq_len = 8;
    const std::string p = ck_path("rt.bin");
    checkpoint_save(p, a, w);
    EXPECT_TRUE(!exists(p + ".tmp"));
    const size_t np = a->params_layout.num_params_padded;
    EXPECT_EQ(read_file(p).size(), size_t(CKPT_HEADER_BYTES) + 3 * np * sizeof(real_t));

    Model* b = model_create(tiny_model(), 2, 8);
    CheckpointInfo r;
    std::string err;
    ASSERT_TRUE(checkpoint_load(p, b, &r, &err));
    EXPECT_TRUE(same_buffers(a, b));
    EXPECT_EQ(r.real_bytes, uint32_t(sizeof(real_t)));
    EXPECT_EQ(r.step, uint64_t(1234));
    EXPECT_EQ(r.tokens_seen, uint64_t(98765));
    EXPECT_EQ(r.loader.epoch, uint64_t(3));
    EXPECT_EQ(r.loader.step, uint64_t(7));
    EXPECT_EQ(r.seed, uint64_t(42));
    EXPECT_EQ(r.global_batch, uint32_t(2));
    EXPECT_EQ(r.seq_len, uint32_t(8));
    EXPECT_EQ(r.num_params, uint64_t(a->params_layout.num_params));
    EXPECT_EQ(r.num_params_padded, uint64_t(np));
    EXPECT_EQ(r.model.n_embd, 16);
    EXPECT_EQ(r.model.tie_embeddings, 1);
    // Saving again gives the same bytes (deterministic format).
    checkpoint_save(ck_path("rt2.bin"), b, w);
    EXPECT_TRUE(read_file(p) == read_file(ck_path("rt2.bin")));
    model_free(a);
    model_free(b);
}

TEST(bad_files_are_refused) {
    Model* a = trained_model();
    CheckpointInfo w;
    w.seed = 1;
    const std::string p = ck_path("good.bin");
    checkpoint_save(p, a, w);
    const std::string good = read_file(p);
    Model* b = model_create(tiny_model(), 2, 8);
    CheckpointInfo r;
    std::string err;

    std::string s = good;
    s[0] = 'X';
    write_file(ck_path("magic.bin"), s);
    EXPECT_TRUE(!checkpoint_load(ck_path("magic.bin"), b, &r, &err));
    EXPECT_TRUE(err.find("magic") != std::string::npos);

    write_file(ck_path("trunc.bin"), good.substr(0, good.size() - 4));
    EXPECT_TRUE(!checkpoint_load(ck_path("trunc.bin"), b, &r, &err));
    EXPECT_TRUE(err.find("truncated") != std::string::npos);

    write_file(ck_path("short.bin"), good.substr(0, 40));
    EXPECT_TRUE(!checkpoint_load(ck_path("short.bin"), b, &r, &err));

    s = good;
    s[CKPT_HEADER_BYTES + 100] ^= 1;  // flip one payload bit
    write_file(ck_path("flip.bin"), s);
    EXPECT_TRUE(!checkpoint_load(ck_path("flip.bin"), b, &r, &err));
    EXPECT_TRUE(err.find("checksum") != std::string::npos);

    s = good;
    s[120] = 1;  // reserved byte
    write_file(ck_path("resv.bin"), s);
    EXPECT_TRUE(!checkpoint_load(ck_path("resv.bin"), b, &r, &err));

    s = good;
    s[4] = 9;  // version
    write_file(ck_path("ver.bin"), s);
    EXPECT_TRUE(!checkpoint_load(ck_path("ver.bin"), b, &r, &err));
    EXPECT_TRUE(err.find("version") != std::string::npos);

    ModelConfig other = tiny_model();
    other.n_head = 4;
    Model* c = model_create(other, 2, 8);
    EXPECT_TRUE(!checkpoint_load(p, c, &r, &err));
    EXPECT_TRUE(err.find("model config differs") != std::string::npos);

    EXPECT_TRUE(!checkpoint_load(ck_path("missing.bin"), b, &r, &err));
    EXPECT_TRUE(checkpoint_load(p, b, &r, &err));  // the original still loads
    model_free(a);
    model_free(b);
    model_free(c);
}

// Build the same checkpoint with the other element size by hand and check it
// loads with an element-wise conversion.
TEST(other_precision_converts) {
    Model* a = trained_model();
    CheckpointInfo w;
    const std::string p = ck_path("prec.bin");
    checkpoint_save(p, a, w);
    const std::string good = read_file(p);
    const size_t n = 3 * a->params_layout.num_params_padded;
    const bool to_double = sizeof(real_t) == 4;
    std::string s = good.substr(0, CKPT_HEADER_BYTES);
    s[8] = to_double ? 8 : 4;
    for (size_t i = 0; i < n; ++i) {
        real_t x;
        std::memcpy(&x, good.data() + CKPT_HEADER_BYTES + i * sizeof(real_t), sizeof(real_t));
        if (to_double) {
            const double d = static_cast<double>(x);
            s.append(reinterpret_cast<const char*>(&d), 8);
        } else {
            const float f = static_cast<float>(x);
            s.append(reinterpret_cast<const char*>(&f), 4);
        }
    }
    uint64_t h = fnv1a64(FNV1A64_INIT, s.data() + CKPT_HEADER_BYTES, s.size() - CKPT_HEADER_BYTES);
    for (int i = 0; i < 8; ++i) s[104 + i] = static_cast<char>(h >> (8 * i));
    write_file(ck_path("prec_other.bin"), s);

    Model* b = model_create(tiny_model(), 2, 8);
    CheckpointInfo r;
    std::string err;
    ASSERT_TRUE(checkpoint_load(ck_path("prec_other.bin"), b, &r, &err));
    EXPECT_EQ(r.real_bytes, uint32_t(to_double ? 8 : 4));
    // float -> double -> float is exact; in the double build the values are
    // rounded to float, so compare against that rounding.
    const real_t* src[3] = {a->params, a->m, a->v};
    const real_t* dst[3] = {b->params, b->m, b->v};
    bool all = true;
    for (int k = 0; k < 3; ++k)
        for (size_t i = 0; i < a->params_layout.num_params_padded; ++i)
            all = all && dst[k][i] == static_cast<real_t>(static_cast<float>(src[k][i]));
    EXPECT_TRUE(all);
    model_free(a);
    model_free(b);
}

TEST(resume_is_bit_exact) {
    ensure_comm();
    const Config full = tiny_config("full");
    train_run(full);
    const std::string fdir = run_dir(full);
    for (const char* f : {"/ckpt_10.bin", "/ckpt_20.bin", "/ckpt_latest.bin", "/log.csv", "/manifest.txt"})
        EXPECT_TRUE(exists(fdir + f));
    EXPECT_TRUE(!exists(fdir + "/ckpt_20.bin.tmp"));
    EXPECT_TRUE(read_file(fdir + "/ckpt_20.bin") == read_file(fdir + "/ckpt_latest.bin"));
    const std::string ck20 = read_file(fdir + "/ckpt_20.bin");
    const std::string log_full = read_file(fdir + "/log.csv");

    // Resume into a new directory.
    Config r = tiny_config("resumed");
    r.train.resume = fdir + "/ckpt_10.bin";
    const TrainSummary s = train_run(r);
    EXPECT_EQ(s.start_step, 11);
    EXPECT_EQ(s.steps, 20);
    EXPECT_EQ(s.tokens_seen, uint64_t(20 * 16));
    EXPECT_TRUE(read_file(run_dir(r) + "/ckpt_20.bin") == ck20);
    // The new log holds only steps 15 and 20; they match the full run's rows.
    const std::string dr = det_columns(read_file(run_dir(r) + "/log.csv"));
    const std::string df = det_columns(log_full);
    EXPECT_TRUE(df.find(dr.substr(dr.find('\n') + 1)) != std::string::npos);

    // Resume in place: rows after step 10 are dropped and rewritten identically.
    Config ip = tiny_config("full");
    ip.train.resume = fdir + "/ckpt_10.bin";
    train_run(ip);
    EXPECT_TRUE(read_file(fdir + "/ckpt_20.bin") == ck20);
    EXPECT_EQ(det_columns(read_file(fdir + "/log.csv")), df);
    // The manifest records the resume and is still a valid config.
    Config reloaded;
    std::string err;
    ASSERT_TRUE(config_load_file(fdir + "/manifest.txt", &reloaded, &err));
    EXPECT_EQ(reloaded.train.resume, ip.train.resume);
}

TEST(resume_mismatches_abort) {
    ensure_comm();
    const Config base = tiny_config("mm_src");
    train_run(base);
    const std::string ck = run_dir(base) + "/ckpt_10.bin";
    std::string out;

    Config a = tiny_config("mm_seed");
    a.train.resume = ck;
    a.train.seed = 99;
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(a); }, &out));
    EXPECT_TRUE(out.find("seed") != std::string::npos);

    Config b = tiny_config("mm_model");
    b.train.resume = ck;
    b.model.n_embd = 32;
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(b); }, &out));
    EXPECT_TRUE(out.find("model config differs") != std::string::npos);

    Config c = tiny_config("mm_done");
    c.train.resume = run_dir(base) + "/ckpt_20.bin";
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(c); }, &out));
    EXPECT_TRUE(out.find("nothing to run") != std::string::npos);

    Config d = tiny_config("mm_batch");
    d.train.resume = ck;
    d.train.batch_size = 1;
    EXPECT_TRUE(test::dies_with_abort([&] { train_run(d); }, &out));
}

TEST_MAIN()
