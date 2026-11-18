// test_bench.cpp: bench library (ROADMAP §12, P7.1).
//   - median_of: odd/even counts, order independence
//   - format_bench_row: exact frozen column order and number formats;
//     commas in string fields abort
//   - append_bench_row: creates the file with the exact §12 header, appends
//     without repeating it, refuses a file with a different first line
//   - default_machine_id has no commas; build_version is "serial" here
//   - run_bench on a tiny synthetic workload: steps - 3 measured steps,
//     phases sum to at most the mean step, tok_per_s = tokens / median,
//     prewarm reads exactly the file size; too few steps abort
#include "test.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tslm/bench.h"
#include "tslm/comm.h"
#include "tslm/token_file.h"

using namespace tslm;

namespace {
std::string tmp_dir() {
    const char* t = std::getenv("TSLM_TEST_TMPDIR");
    return std::string(t ? t : ".");
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
BenchRow sample_row() {
    BenchRow r;
    r.version = "serial";
    r.preset = "small";
    r.global_batch = 32;
    r.T = 256;
    r.steps = 50;
    r.ranks = 1;
    r.median_step_ms = 3800.12345;
    r.mean_step_ms = 3810.5;
    r.tok_per_s = 2155.66;
    r.data_ms = 0.04;
    r.fwd_ms = 1200;
    r.bwd_ms = 2500.25;
    r.comm_ms = 0;
    r.opt_ms = 50.5;
    r.peak_mem_MB = 1300.06;
    r.machine_id = "cpu x | 4 cores | 16 GB";
    r.git_hash = "abc123";
    return r;
}
}  // namespace

TEST(median_of_values) {
    EXPECT_EQ(median_of({3, 1, 2}), 2.0);
    EXPECT_EQ(median_of({4, 1, 3, 2}), 2.5);
    EXPECT_EQ(median_of({7}), 7.0);
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([] { median_of({}); }, &out));
}

TEST(row_format_and_header) {
    EXPECT_EQ(std::string(BENCH_CSV_HEADER),
              std::string("version,preset,global_batch,T,steps,ranks,median_step_ms,mean_step_ms,tok_per_s,data_ms,"
                          "fwd_ms,bwd_ms,comm_ms,opt_ms,peak_mem_MB,machine_id,git_hash"));
    EXPECT_EQ(format_bench_row(sample_row()),
              std::string("serial,small,32,256,50,1,3800.123,3810.500,2155.7,0.040,1200.000,2500.250,0.000,50.500,"
                          "1300.1,cpu x | 4 cores | 16 GB,abc123"));
    BenchRow bad = sample_row();
    bad.machine_id = "a,b";
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { format_bench_row(bad); }, &out));
    EXPECT_TRUE(default_machine_id().find(',') == std::string::npos);
    EXPECT_EQ(std::string(build_version()), std::string("serial"));
}

TEST(append_creates_and_validates) {
    const std::string p = tmp_dir() + "/bench_append.csv";
    std::remove(p.c_str());
    std::string err;
    ASSERT_TRUE(append_bench_row(p, sample_row(), &err));
    ASSERT_TRUE(append_bench_row(p, sample_row(), &err));
    const std::string row = format_bench_row(sample_row());
    EXPECT_EQ(read_file(p), std::string(BENCH_CSV_HEADER) + "\n" + row + "\n" + row + "\n");
    const std::string q = tmp_dir() + "/bench_bad.csv";
    std::FILE* f = std::fopen(q.c_str(), "w");
    std::fputs("step,loss\n", f);
    std::fclose(f);
    EXPECT_TRUE(!append_bench_row(q, sample_row(), &err));
    EXPECT_TRUE(err.find("refusing") != std::string::npos);
}

TEST(run_bench_tiny_workload) {
    int argc = 0;
    char** argv = nullptr;
    comm_init(&argc, &argv);
    const std::string data = tmp_dir() + "/bench_train.bin";
    const uint64_t n = 1 + 8 * 64;
    {
        std::FILE* f = std::fopen(data.c_str(), "wb");
        TokenFileHeader h;
        h.vocab_size = 32;
        h.num_tokens = n;
        unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
        encode_token_header(h, hdr);
        std::fwrite(hdr, 1, sizeof(hdr), f);
        for (uint64_t i = 0; i < n; ++i) {
            const unsigned char b[2] = {static_cast<unsigned char>(i % 31), 0};
            std::fwrite(b, 1, 2, f);
        }
        std::fclose(f);
    }
    Config c;
    c.model.vocab_size = 32;
    c.model.max_seq_len = 8;
    c.model.n_layer = 1;
    c.model.n_embd = 16;
    c.model.n_head = 2;
    c.train.run_name = "tiny";
    c.train.train_data = data;
    c.train.batch_size = 2;
    c.train.seq_len = 8;
    c.train.max_steps = 9;
    c.train.warmup_steps = 2;
    BenchOptions o;
    const BenchRow r = run_bench(c, &o);
    EXPECT_EQ(o.prewarm_bytes, uint64_t(TOKEN_FILE_HEADER_BYTES + 2 * n));
    EXPECT_EQ(o.measured_steps, 9 - BENCH_WARMUP_STEPS);
    EXPECT_EQ(r.preset, std::string("tiny"));
    EXPECT_EQ(r.global_batch, 2);
    EXPECT_EQ(r.T, 8);
    EXPECT_EQ(r.steps, 9);
    EXPECT_EQ(r.ranks, 1);
    EXPECT_TRUE(r.median_step_ms > 0 && r.mean_step_ms > 0);
    EXPECT_NEAR(r.tok_per_s, 16.0 / (r.median_step_ms / 1e3), 1e-9 * r.tok_per_s);
    const double phases = r.data_ms + r.fwd_ms + r.bwd_ms + r.comm_ms + r.opt_ms;
    EXPECT_TRUE(phases > 0 && phases <= r.mean_step_ms * (1 + 1e-9));
    EXPECT_TRUE(r.peak_mem_MB > 0);
    std::string out;
    c.train.max_steps = 3;  // nothing left after the warmup
    EXPECT_TRUE(test::dies_with_abort([&] { BenchOptions o2; run_bench(c, &o2); }, &out));
    EXPECT_TRUE(out.find("warmup") != std::string::npos);
}

TEST_MAIN()
