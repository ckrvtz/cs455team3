// test_smoke.cpp: ROADMAP §10 "Overfit" and "Smoke training" rows (P4.5).
//
// Both run configs/smoke.cfg unchanged except for data paths, output dir and
// (overfit only) step count / cadence, on real TinyStories tokens from the
// committed fixture tests/fixtures/ (see tests/fixtures/README.md):
//   smoke_train.bin  first 99,958 train tokens (prepare_data --max_tokens 100000)
//   smoke_val.bin    first 4,097 tokens of the matching val.bin (= the 32 fixed
//                    validation batches of 4 x 32 the preset evaluates)
// The fixture replaces §10's "~1M-token subset" so make test needs no
// dataset (CLAUDE.md §6; docs/DECISIONS.md 2026-11-18, P4.5). 200 steps of
// B=4, T=32 read 25,600 tokens, well inside the 100k.
//
// Overfit: a token file holding exactly one global batch (the first 4 chunks
// of smoke_train.bin, written at test time), so every step sees the same
// batch. Criterion: train loss < 0.1 at step 300.
// Smoke: criteria: validation loss decreases at every evaluation (50, 100,
// 150, 200; "coarsely" = each may rise by at most 0.05 over the previous
// one), ends below 6.0, and the 200 steps take under 60 s. The time limit is
// checked in release builds only: sanitizer/-O0 builds are not the timed
// configuration (the time is printed in every build).
//
// Debug (-O0 + ASan/UBSan) builds: a step there takes ~1.2 s (measured: the
// 300-step overfit run took 377 s), so both tests run a 10-step version that
// exercises the same code paths (training, validation, checkpoint) under the
// sanitizers and checks only that the losses are finite. The §10 criteria
// are asserted in the release and REAL=double builds. Each test prints which
// mode it ran in.
#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/token_file.h"
#include "tslm/train.h"

using namespace tslm;

namespace {

bool debug_build() { return std::strcmp(TSLM_BUILD_MODE, "debug") == 0; }

// Sanitizer build: shorten a run to 10 steps with 2 validation batches.
void shorten_for_debug(Config* c) {
    c->train.max_steps = 10;
    c->train.warmup_steps = 2;
    c->train.log_every = 5;
    c->train.eval_every = 5;
    c->train.val_batches = 2;
    std::printf("    debug build: 10-step sanitizer run; the section 10 criteria are checked in release/double\n");
}

const char* const FIXTURE_TRAIN = "tests/fixtures/smoke_train.bin";
const char* const FIXTURE_VAL = "tests/fixtures/smoke_val.bin";

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

Config smoke_config(const std::string& run_name) {
    Config c;
    std::string err;
    CHECK(config_load_file("configs/smoke.cfg", &c, &err), "%s", err.c_str());
    c.train.train_data = FIXTURE_TRAIN;
    c.train.val_data = FIXTURE_VAL;
    c.train.out_dir = tmp_dir() + "/smokeruns";
    c.train.run_name = run_name;
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

// The first n_tokens of a token file as a new token file.
std::string head_tokens(const std::string& src, uint64_t n_tokens, const std::string& name) {
    const std::string in = read_file(src);
    TokenFileHeader h;
    std::string err;
    CHECK(in.size() >= TOKEN_FILE_HEADER_BYTES &&
              decode_token_header(reinterpret_cast<const unsigned char*>(in.data()), &h, &err),
          "%s: %s", src.c_str(), err.c_str());
    CHECK(h.num_tokens >= n_tokens, "%s has only %llu tokens", src.c_str(),
          static_cast<unsigned long long>(h.num_tokens));
    h.num_tokens = n_tokens;
    unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, hdr);
    const std::string path = tmp_dir() + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f, "cannot write %s", path.c_str());
    std::fwrite(hdr, 1, sizeof(hdr), f);
    std::fwrite(in.data() + TOKEN_FILE_HEADER_BYTES, 1, 2 * n_tokens, f);
    std::fclose(f);
    return path;
}

// (step, val_loss) for every log.csv row with a val_loss.
std::vector<std::pair<int, double>> val_rows(const std::string& log) {
    std::vector<std::pair<int, double>> out;
    size_t pos = log.find('\n') + 1;  // skip header
    while (pos < log.size()) {
        size_t nl = log.find('\n', pos);
        if (nl == std::string::npos) nl = log.size();
        const std::string line = log.substr(pos, nl - pos);
        pos = nl + 1;
        // fields: step, tokens_seen, train_loss, val_loss, ...
        const size_t c1 = line.find(','), c2 = line.find(',', c1 + 1), c3 = line.find(',', c2 + 1),
                     c4 = line.find(',', c3 + 1);
        const std::string v = line.substr(c3 + 1, c4 - c3 - 1);
        if (!v.empty()) out.emplace_back(std::atoi(line.c_str()), std::strtod(v.c_str(), nullptr));
    }
    return out;
}

}  // namespace

TEST(overfit_one_batch) {
    ensure_comm();
    Config c = smoke_config("overfit");
    // Exactly one global batch: B=4 chunks of T=32 -> 4*32 + 1 tokens.
    const uint64_t n = uint64_t(c.train.batch_size) * uint64_t(c.train.seq_len) + 1;
    c.train.train_data = head_tokens(FIXTURE_TRAIN, n, "overfit_batch.bin");
    c.train.max_steps = 300;
    c.train.log_every = 50;
    c.train.eval_every = 300;
    if (debug_build()) {
        shorten_for_debug(&c);
        const TrainSummary d = train_run(c);
        EXPECT_TRUE(std::isfinite(d.final_loss) && d.final_loss < d.first_loss);
        return;
    }
    const TrainSummary s = train_run(c);
    std::printf("    overfit: train loss %.4f -> %.4f after %d steps (%.1f s)\n", s.first_loss, s.final_loss,
                s.steps, s.wall_s);
    EXPECT_EQ(s.steps, 300);
    EXPECT_TRUE(s.final_loss < 0.1);
}

TEST(smoke_training) {
    ensure_comm();
    Config c = smoke_config("smoke");
    EXPECT_EQ(c.train.max_steps, 200);  // the preset is what §10 specifies
    if (debug_build()) {
        shorten_for_debug(&c);
        const TrainSummary d = train_run(c);
        EXPECT_TRUE(std::isfinite(d.final_loss) && std::isfinite(d.final_val_loss));
        return;
    }
    const TrainSummary s = train_run(c);
    const auto vals = val_rows(read_file(run_dir(c) + "/log.csv"));
    std::printf("    smoke: %d steps in %.1f s; val", s.steps, s.wall_s);
    for (const auto& v : vals) std::printf(" %d:%.4f", v.first, v.second);
    std::printf("\n");
    ASSERT_EQ(vals.size(), size_t(4));  // steps 50, 100, 150, 200
    for (size_t i = 1; i < vals.size(); ++i) EXPECT_TRUE(vals[i].second <= vals[i - 1].second + 0.05);
    EXPECT_TRUE(vals.back().second < 6.0);
    EXPECT_TRUE(vals.front().second - vals.back().second > 0.3);  // a real decrease, not noise
    if (std::strcmp(TSLM_BUILD_MODE, "release") == 0) EXPECT_TRUE(s.wall_s < 60.0);
}

TEST_MAIN()
