// test_signoff.cpp: ROADMAP §10 sign-off layers for P5.1 (determinism,
// causality, resume) at the scale of the `smoke` preset, on the committed
// real-data fixture (tests/fixtures/). The fourth P5.1 layer, exact parameter
// counts for every preset (§2.2), is tests/test_layout.cpp
// (param_counts_match_roadmap). Smaller-scale versions of determinism and
// resume also live in test_train.cpp and test_checkpoint.cpp.
//
//   determinism  Same seed, same binary, 2 runs x 20 steps: every log.csv row
//                (log_every = 1) has bit-identical train_loss, lr, grad_norm
//                and val_loss, and the final params are identical (FNV-1a
//                checksum printed; whole final checkpoint byte-identical).
//                The second run happens in a forked child process whose
//                heap is first dirtied (dirty_heap): allocation addresses
//                then differ from the first run's, and reused small blocks
//                hold 0xA5 garbage instead of whatever the first run left,
//                so a dependence on allocation addresses or on reused
//                uninitialized heap memory shows up as a mismatch.
//   resume       20 steps = 10 + checkpoint + resume 10: identical final
//                checkpoint (params, m, v, step, loader state).
//   causality    Full model (2 layers, tied and untied): changing the token
//                at position t leaves the logits of every position < t
//                bit-identical, changes position t, and leaves the other
//                batch rows bit-identical.
//
// Debug (-O0 + sanitizers) builds shorten determinism to 2 x 6 steps and
// resume to 6 = 3 + 3 (a smoke step there takes ~1.2 s); causality is
// unchanged. Each test prints the sizes it ran.
#include "test.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/model.h"
#include "tslm/rng.h"
#include "tslm/token_file.h"
#include "tslm/train.h"

using namespace tslm;

namespace {

bool debug_build() { return std::strcmp(TSLM_BUILD_MODE, "debug") == 0; }

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

Config smoke_config(const std::string& run_name, int steps, int ckpt_every) {
    Config c;
    std::string err;
    CHECK(config_load_file("configs/smoke.cfg", &c, &err), "%s", err.c_str());
    c.train.train_data = "tests/fixtures/smoke_train.bin";
    c.train.val_data = "tests/fixtures/smoke_val.bin";
    c.train.out_dir = tmp_dir() + "/signoff";
    c.train.run_name = run_name;
    c.train.max_steps = steps;
    c.train.warmup_steps = steps / 4;
    c.train.log_every = 1;
    c.train.eval_every = steps / 2;
    c.train.val_batches = 4;
    c.train.ckpt_every = ckpt_every;
    return c;
}

// Shift the child's allocation addresses and leave garbage in freed small
// blocks: 2000 blocks of 16..4096 bytes filled with 0xA5, every other one
// freed (the rest stay allocated until the child exits).
void dirty_heap() {
    std::vector<void*> keep;
    for (int i = 0; i < 2000; ++i) {
        const size_t sz = 16 + size_t((i * 37) % 4081);
        void* p = std::malloc(sz);
        CHECK(p, "malloc failed");
        std::memset(p, 0xA5, sz);
        if (i % 2) std::free(p);
        else keep.push_back(p);
    }
    static std::vector<void*> leak;
    leak.swap(keep);
}

// Run train_run in a forked child with a dirtied heap; CHECKs that it
// exited cleanly.
void train_in_child(const Config& c) {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
        dirty_heap();
        train_run(c);
        std::fflush(stdout);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid, "waitpid failed");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child training run failed (status %d)", status);
}

// log.csv reduced to step, tokens_seen, train_loss, val_loss, val_ppl, lr,
// grad_norm (every column except the timings).
std::string numeric_columns(const std::string& log) {
    std::string out, fld;
    int col = 0;
    for (char ch : log) {
        if (ch == ',' || ch == '\n') {
            if (col <= 6) out += fld + "|";
            fld.clear();
            col = ch == '\n' ? 0 : col + 1;
            if (ch == '\n') out += "\n";
        } else {
            fld += ch;
        }
    }
    return out;
}

}  // namespace

TEST(determinism_two_runs) {
    ensure_comm();
    const int steps = debug_build() ? 6 : 20;
    std::printf("    determinism: smoke preset, 2 x %d steps (second run in a child process)\n", steps);
    const Config a = smoke_config("det_a", steps, 0), b = smoke_config("det_b", steps, 0);
    train_run(a);
    train_in_child(b);
    const std::string la = read_file(run_dir(a) + "/log.csv"), lb = read_file(run_dir(b) + "/log.csv");
    const std::string na = numeric_columns(la), nb = numeric_columns(lb);
    EXPECT_EQ(na, nb);
    // One row per step plus the header.
    size_t rows = 0;
    for (char ch : la) rows += ch == '\n';
    EXPECT_EQ(rows, size_t(steps + 1));

    const std::string ca = read_file(run_dir(a) + "/ckpt_latest.bin"),
                      cb = read_file(run_dir(b) + "/ckpt_latest.bin");
    EXPECT_TRUE(ca == cb);
    // Checksum of the final params alone (the first blob after the header).
    Model* m = model_create(a.model, 1, 1);
    const size_t pbytes = model_param_bytes(m);
    model_free(m);
    ASSERT_TRUE(ca.size() >= 128 + pbytes && cb.size() >= 128 + pbytes);
    const uint64_t ha = fnv1a64(FNV1A64_INIT, ca.data() + 128, pbytes);
    const uint64_t hb = fnv1a64(FNV1A64_INIT, cb.data() + 128, pbytes);
    std::printf("    final params FNV-1a: %016llx / %016llx\n", static_cast<unsigned long long>(ha),
                static_cast<unsigned long long>(hb));
    EXPECT_EQ(ha, hb);
}

TEST(resume_matches_uninterrupted) {
    ensure_comm();
    const int steps = debug_build() ? 6 : 20, half = steps / 2;
    std::printf("    resume: smoke preset, %d steps vs %d + checkpoint + %d\n", steps, half, steps - half);
    const Config full = smoke_config("res_full", steps, half);
    train_run(full);
    Config r = smoke_config("res_resumed", steps, half);
    char name[64];
    std::snprintf(name, sizeof(name), "/ckpt_%d.bin", half);
    r.train.resume = run_dir(full) + name;
    train_in_child(r);
    std::snprintf(name, sizeof(name), "/ckpt_%d.bin", steps);
    EXPECT_TRUE(read_file(run_dir(full) + name) == read_file(run_dir(r) + name));
    // The resumed log's rows equal the uninterrupted log's rows for those steps.
    const std::string nf = numeric_columns(read_file(run_dir(full) + "/log.csv"));
    std::string nr = numeric_columns(read_file(run_dir(r) + "/log.csv"));
    nr = nr.substr(nr.find('\n') + 1);  // drop the header line
    EXPECT_TRUE(!nr.empty() && nf.find(nr) != std::string::npos);
}

TEST(causality_full_model) {
    for (int tied : {1, 0}) {
        ModelConfig mc;
        mc.vocab_size = 64;
        mc.max_seq_len = 16;
        mc.n_layer = 2;
        mc.n_embd = 32;
        mc.n_head = 4;
        mc.tie_embeddings = tied;
        const int B = 2, T = 16, V = mc.vocab_size;
        Model* m = model_create(mc, B, T);
        model_init_weights(m, 99);
        Rng rng(5, 0);
        std::vector<int32_t> tok(B * T);
        for (int32_t& t : tok) t = static_cast<int32_t>(rng.uniform_int(uint64_t(V)));
        model_forward(m, tok.data(), nullptr, B, T);
        const size_t n = size_t(B) * T * V;
        const std::vector<real_t> base(model_logits(m), model_logits(m) + n);
        for (int t : {0, 1, 7, T - 1}) {
            std::vector<int32_t> pert = tok;
            pert[t] = (pert[t] + 1 + t) % V;  // batch row 0 only
            model_forward(m, pert.data(), nullptr, B, T);
            const real_t* lg = model_logits(m);
            const size_t row = size_t(V);
            // Row 0, positions < t: bit-identical.
            EXPECT_TRUE(t == 0 || std::memcmp(lg, base.data(), size_t(t) * row * sizeof(real_t)) == 0);
            // Row 0, position t: changed.
            EXPECT_TRUE(std::memcmp(lg + size_t(t) * row, base.data() + size_t(t) * row, row * sizeof(real_t)) != 0);
            // Batch row 1: entirely bit-identical.
            EXPECT_TRUE(std::memcmp(lg + size_t(T) * row, base.data() + size_t(T) * row,
                                    size_t(T) * row * sizeof(real_t)) == 0);
        }
        model_free(m);
    }
}

TEST_MAIN()
