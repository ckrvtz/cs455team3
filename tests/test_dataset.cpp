// test_dataset.cpp: DataLoader (ROADMAP P1.4, §7.4, §4.5).
// Synthetic token files hold token[p] = p (vocab 65536), so every x/y value
// identifies its stream position and chunk. The epoch-permutation known
// answers were recorded once and cross-checked against an independent
// re-implementation of Rng + Fisher-Yates (not committed).
#include "test.h"

#include "tslm/common.h"
#include "tslm/dataset.h"
#include "tslm/token_file.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace tslm;

namespace {

std::string tmp_path(const std::string& name) {
    const char* t = std::getenv("TSLM_TEST_TMPDIR");
    return std::string(t ? t : ".") + "/" + name;
}

// Write a token file whose tokens are 0, 1, 2, ... (mod vocab).
std::string write_tokens(const std::string& name, uint64_t n, uint32_t vocab = 65536,
                         uint64_t header_tokens = UINT64_MAX) {
    const std::string path = tmp_path(name);
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f, "cannot write %s", path.c_str());
    TokenFileHeader h;
    h.vocab_size = vocab;
    h.num_tokens = header_tokens == UINT64_MAX ? n : header_tokens;
    unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, hdr);
    std::fwrite(hdr, 1, sizeof(hdr), f);
    for (uint64_t p = 0; p < n; ++p) {
        const uint32_t v = static_cast<uint32_t>(p % 65536);
        const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
        std::fwrite(b, 1, 2, f);
    }
    std::fclose(f);
    return path;
}

DataLoader* open_or_die(const std::string& path, int B, int T, uint64_t seed, int rank = 0,
                        int world = 1) {
    DataLoader* d = new DataLoader();
    std::string err;
    CHECK(d->open(path, B, T, seed, rank, world, &err), "%s", err.c_str());
    return d;
}

struct Batch {
    std::vector<int32_t> x, y;
};

Batch next(DataLoader& d, int B, int T) {
    Batch b;
    b.x.resize(size_t(B) * T);
    b.y.resize(size_t(B) * T);
    d.next_batch(b.x.data(), b.y.data());
    return b;
}

}  // namespace

TEST(epoch_permutation_known_answer) {
    uint32_t p[10];
    epoch_permutation(1337, 0, 10, p);
    const uint32_t want0[10] = {7, 3, 2, 1, 8, 4, 0, 5, 9, 6};
    for (int i = 0; i < 10; ++i) EXPECT_EQ(p[i], want0[i]);
    epoch_permutation(1337, 1, 10, p);
    const uint32_t want1[10] = {4, 1, 9, 3, 8, 7, 5, 2, 6, 0};
    for (int i = 0; i < 10; ++i) EXPECT_EQ(p[i], want1[i]);
    std::vector<uint32_t> q(1000);
    epoch_permutation(42, 0, 1000, q.data());
    const uint32_t want2[8] = {167, 60, 914, 170, 57, 131, 949, 891};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(q[size_t(i)], want2[i]);
    // It is a permutation.
    std::set<uint32_t> s(q.begin(), q.end());
    EXPECT_EQ(s.size(), size_t{1000});
    EXPECT_EQ(*s.rbegin(), 999u);
}

TEST(chunk_layout_and_counts) {
    const int T = 8, B = 2;
    // 1 + 8*13 + 5 tokens -> (N-1)/T = 13 chunks; 13 / 2 = 6 steps (1 dropped).
    const std::string path = write_tokens("chunks.bin", 1 + 8 * 13 + 5);
    DataLoader* d = open_or_die(path, B, T, 7);
    EXPECT_EQ(d->num_chunks(), uint64_t{13});
    EXPECT_EQ(d->steps_per_epoch(), uint64_t{6});
    EXPECT_EQ(d->vocab_size(), 65536u);
    std::vector<uint32_t> perm(13);
    epoch_permutation(7, 0, 13, perm.data());
    for (uint64_t s = 0; s < d->steps_per_epoch(); ++s) {
        const Batch b = next(*d, B, T);
        for (int r = 0; r < B; ++r) {
            const uint32_t chunk = perm[size_t(s) * B + size_t(r)];
            for (int t = 0; t < T; ++t) {
                // x = tokens [c*T, c*T+T), y = tokens [c*T+1, c*T+T]
                ASSERT_EQ(b.x[size_t(r) * T + size_t(t)], int32_t(chunk * T + t));
                ASSERT_EQ(b.y[size_t(r) * T + size_t(t)], int32_t(chunk * T + t + 1));
            }
        }
    }
    EXPECT_EQ(d->epoch(), uint64_t{0});
    next(*d, B, T);  // rolls into epoch 1
    EXPECT_EQ(d->epoch(), uint64_t{1});
    EXPECT_EQ(d->state().step, uint64_t{1});
    delete d;
}

TEST(no_duplicate_chunks_within_epoch) {
    const int T = 4, B = 3;
    const std::string path = write_tokens("dups.bin", 1 + 4 * 50);  // 50 chunks -> 16 steps
    DataLoader* d = open_or_die(path, B, T, 99);
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::set<int32_t> starts;
        for (uint64_t s = 0; s < d->steps_per_epoch(); ++s) {
            const Batch b = next(*d, B, T);
            for (int r = 0; r < B; ++r) starts.insert(b.x[size_t(r) * T] / T);
        }
        EXPECT_EQ(starts.size(), size_t(d->steps_per_epoch() * B));  // 48 distinct of 50
        EXPECT_EQ(d->epoch(), uint64_t(epoch));
    }
    delete d;
}

TEST(determinism_for_fixed_seed) {
    const int T = 6, B = 4;
    const std::string path = write_tokens("det.bin", 1 + 6 * 40);
    DataLoader* a = open_or_die(path, B, T, 1234);
    DataLoader* b = open_or_die(path, B, T, 1234);
    DataLoader* c = open_or_die(path, B, T, 1235);
    bool any_diff = false;
    for (int s = 0; s < 35; ++s) {  // spans several epochs (10 steps each)
        const Batch ba = next(*a, B, T), bb = next(*b, B, T), bc = next(*c, B, T);
        ASSERT_TRUE(ba.x == bb.x && ba.y == bb.y);
        any_diff = any_diff || ba.x != bc.x;
    }
    EXPECT_TRUE(any_diff);  // a different seed gives a different order
    delete a;
    delete b;
    delete c;
}

// ROADMAP §7.4: 4 loaders (rank r of 4, B=8) == one loader (B=32, rank 0 of 1).
TEST(sharding_equivalence_4x8_vs_1x32) {
    const int T = 5;
    const std::string path = write_tokens("shard.bin", 1 + 5 * 300);  // 300 chunks -> 9 steps
    DataLoader* serial = open_or_die(path, 32, T, 2024);
    DataLoader* ranks[4];
    for (int r = 0; r < 4; ++r) ranks[r] = open_or_die(path, 8, T, 2024, r, 4);
    EXPECT_EQ(serial->steps_per_epoch(), ranks[0]->steps_per_epoch());
    for (int s = 0; s < 25; ++s) {  // crosses two epoch boundaries
        const Batch full = next(*serial, 32, T);
        std::vector<int32_t> x, y;
        for (int r = 0; r < 4; ++r) {
            const Batch part = next(*ranks[r], 8, T);
            x.insert(x.end(), part.x.begin(), part.x.end());
            y.insert(y.end(), part.y.begin(), part.y.end());
            ASSERT_EQ(ranks[r]->state().epoch, serial->state().epoch);
            ASSERT_EQ(ranks[r]->state().step, serial->state().step);
        }
        ASSERT_TRUE(x == full.x);
        ASSERT_TRUE(y == full.y);
    }
    delete serial;
    for (DataLoader* d : ranks) delete d;
}

TEST(resume_reproduces_next_batches) {
    const int T = 7, B = 3;
    const std::string path = write_tokens("resume.bin", 1 + 7 * 31);  // 31 chunks -> 10 steps
    for (int k : {0, 4, 10, 13}) {  // includes exactly at, and past, an epoch end
        DataLoader* a = open_or_die(path, B, T, 5);
        for (int i = 0; i < k; ++i) next(*a, B, T);
        const LoaderState st = a->state();
        std::vector<Batch> want;
        for (int i = 0; i < 12; ++i) want.push_back(next(*a, B, T));

        DataLoader* b = open_or_die(path, B, T, 5);
        b->set_state(st);
        for (int i = 0; i < 12; ++i) {
            const Batch got = next(*b, B, T);
            ASSERT_TRUE(got.x == want[size_t(i)].x && got.y == want[size_t(i)].y);
        }
        delete a;
        delete b;
    }
}

TEST(open_errors) {
    DataLoader d;
    std::string err;
    EXPECT_TRUE(!d.open(tmp_path("missing.bin"), 2, 4, 0, 0, 1, &err));
    EXPECT_TRUE(err.find("cannot open") != std::string::npos);

    const std::string ok = write_tokens("ok.bin", 1 + 4 * 10);
    EXPECT_TRUE(!d.open(ok, 4, 4, 0, 0, 3, &err));  // gB=12 > 10 chunks
    EXPECT_TRUE(err.find("cannot fill") != std::string::npos);
    EXPECT_TRUE(!d.open(ok, 2, 4, 0, 1, 1, &err));  // rank out of range
    EXPECT_TRUE(!d.open(ok, 0, 4, 0, 0, 1, &err));

    const std::string bad_size = write_tokens("badsize.bin", 41, 65536, 50);  // header says 50
    EXPECT_TRUE(!d.open(bad_size, 2, 4, 0, 0, 1, &err));
    EXPECT_TRUE(err.find("does not match") != std::string::npos);

    const std::string tiny = tmp_path("tiny.bin");
    FILE* f = std::fopen(tiny.c_str(), "wb");
    std::fwrite("TSTK", 1, 4, f);
    std::fclose(f);
    EXPECT_TRUE(!d.open(tiny, 2, 4, 0, 0, 1, &err));

    // A token >= vocab_size is caught while copying.
    const std::string oob = write_tokens("oob.bin", 1 + 4 * 10, 20);  // tokens go up to 40
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(
        [&] {
            DataLoader e;
            std::string er;
            CHECK(e.open(oob, 10, 4, 0, 0, 1, &er), "%s", er.c_str());
            int32_t x[40], y[40];
            e.next_batch(x, y);
        },
        &out));
    EXPECT_TRUE(out.find(">= vocab_size") != std::string::npos);
}

// If prepare_data has been run, the real val.bin must load and yield
// in-range tokens; otherwise this check is skipped (synthetic tests above
// cover the loader).
TEST(real_val_bin_if_present) {
    FILE* f = std::fopen("data/val.bin", "rb");
    if (!f) {
        std::printf("    data/val.bin not present: real-data check not done\n");
        return;
    }
    std::fclose(f);
    DataLoader* d = open_or_die("data/val.bin", 4, 256, 1337);
    std::printf("    data/val.bin: %llu tokens, %llu chunks of 256, %llu steps/epoch at B=4\n",
                static_cast<unsigned long long>(d->num_tokens()),
                static_cast<unsigned long long>(d->num_chunks()),
                static_cast<unsigned long long>(d->steps_per_epoch()));
    const Batch b = next(*d, 4, 256);
    for (int32_t v : b.x) ASSERT_TRUE(v >= 0 && uint32_t(v) < d->vocab_size());
    delete d;
}

TEST_MAIN()
