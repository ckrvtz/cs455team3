// test_runtime.cpp: backend.h (CPU), comm.h (serial), timer.h (ROADMAP P0.3).
#include "test.h"

#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/common.h"
#include "tslm/timer.h"

#include <cstdint>
#include <cstring>
#include <string>

using namespace tslm;

// ---- backend ---------------------------------------------------------------

// Must stay the FIRST test: dev_init is only legal before any other dev_* call
// in the process (D-B8). Misuse is checked in forked children.
TEST(dev_init_and_peak_bytes) {
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([] { dev_init(-1); }, &out));
    EXPECT_TRUE(out.find("local_rank") != std::string::npos);
    dev_init(0);
    EXPECT_TRUE(test::dies_with_abort([] { dev_init(0); }, &out));
    EXPECT_TRUE(out.find("twice") != std::string::npos);

    const size_t p0 = dev_peak_bytes();
    void* a = dev_malloc(1000);
    void* b = dev_malloc(3000);
    EXPECT_EQ(dev_peak_bytes(), p0 + 4000);  // requested bytes, not rounded
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % DEV_ALIGN, uintptr_t(0));
    dev_free(b);
    void* c = dev_malloc(500);  // current = p0 + 1500 < peak
    EXPECT_EQ(dev_peak_bytes(), p0 + 4000);
    dev_free(a);
    dev_free(c);
    dev_free(nullptr);
    void* d = dev_malloc(5000);  // a new high-water mark
    EXPECT_EQ(dev_peak_bytes(), p0 + 5000);
    dev_free(d);
    // dev_init after any dev_* call is refused (fresh flag only matters in a child).
    EXPECT_TRUE(test::dies_with_abort([] { dev_init(1); }, &out));
}

TEST(dev_malloc_alignment_and_zero_size) {
    EXPECT_TRUE(dev_malloc(0) == nullptr);
    const size_t sizes[] = {1, 63, 64, 65, 1000, size_t(1) << 20};
    for (size_t n : sizes) {
        void* p = dev_malloc(n);
        ASSERT_TRUE(p != nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % DEV_ALIGN, uintptr_t{0});
        dev_memset(p, 0xAB, n);  // whole requested range is writable
        dev_free(p);
    }
    dev_free(nullptr);  // allowed
#if defined(TSLM_BACKEND_CUDA)
    EXPECT_TRUE(std::strstr(dev_name(), "(sm_") != nullptr);  // "<GPU> (sm_XY), driver ..., CUDA runtime ..."
#else
    EXPECT_TRUE(std::strcmp(dev_name(), "cpu") == 0);
#endif
}

TEST(memset_and_copies_round_trip) {
    const size_t n = 1000;
    real_t host_in[n], host_out[n];
    for (size_t i = 0; i < n; ++i) host_in[i] = static_cast<real_t>(i) * real_t(0.5) - real_t(3);

    real_t* a = static_cast<real_t*>(dev_malloc(n * sizeof(real_t)));
    real_t* b = static_cast<real_t*>(dev_malloc(n * sizeof(real_t)));
    copy_h2d(a, host_in, n * sizeof(real_t));
    copy_d2d(b, a, n * sizeof(real_t));
    copy_d2h(host_out, b, n * sizeof(real_t));
    dev_sync();
    EXPECT_TRUE(std::memcmp(host_in, host_out, sizeof(host_in)) == 0);

    dev_memset(b, 0, n * sizeof(real_t));
    copy_d2h(host_out, b, n * sizeof(real_t));
    bool all_zero = true;
    for (size_t i = 0; i < n; ++i) all_zero = all_zero && host_out[i] == real_t(0);
    EXPECT_TRUE(all_zero);

    copy_h2d(a, nullptr, 0);  // zero-byte copies are no-ops
    dev_free(a);
    dev_free(b);
}

TEST(backend_misuse_aborts) {
    std::string err;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            char* p = static_cast<char*>(dev_malloc(128));
            copy_d2d(p + 8, p, 64);  // overlapping
        },
        &err));
    EXPECT_TRUE(err.find("overlapping") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort([] { dev_memset(nullptr, 0, 4); }, &err));
    EXPECT_TRUE(err.find("null pointer") != std::string::npos);
}

// ---- comm (serial) -----------------------------------------------------------

TEST(comm_serial_lifecycle_and_noops) {
    // Run in a child so the process-wide comm state is fresh and finalize
    // does not affect other tests; the child aborts on any mismatch.
    std::string err;
    const bool aborted = test::dies_with_abort(
        [] {
            int argc = 1;
            char arg0[] = "prog";
            char* args[] = {arg0, nullptr};
            char** argv = args;
            comm_init(&argc, &argv);
            CHECK(comm_rank() == 0 && comm_size() == 1, "rank/size");
            real_t v[4] = {1, -2, 3.5, 0};
            const real_t orig[4] = {1, -2, 3.5, 0};
            comm_broadcast(v, 4, 0);
            comm_allreduce_avg(v, 4);
            for (int i = 0; i < 4; ++i) CHECK(v[i] == orig[i], "collective changed data");
            double d[2] = {1.25, -7.0};
            comm_allreduce_sum_double(d, 2);
            CHECK(d[0] == 1.25 && d[1] == -7.0, "sum_double changed data");
            comm_allreduce_avg(nullptr, 0);
            comm_barrier();
            comm_finalize();
            CHECK(argc == 1 && argv == args, "comm_init changed argv");
            CHECK(false, "reached end ok");  // marker: everything above passed
        },
        &err);
    EXPECT_TRUE(aborted);
    EXPECT_TRUE(err.find("reached end ok") != std::string::npos);
}

TEST(comm_misuse_aborts) {
    std::string err;
    EXPECT_TRUE(test::dies_with_abort([] { (void)comm_rank(); }, &err));
    EXPECT_TRUE(err.find("before comm_init") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            comm_init(nullptr, nullptr);
            comm_init(nullptr, nullptr);
        },
        &err));
    EXPECT_TRUE(err.find("called twice") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            comm_init(nullptr, nullptr);
            real_t x = 0;
            comm_broadcast(&x, 1, 1);
        },
        &err));
    EXPECT_TRUE(err.find("root 1 out of range") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            comm_init(nullptr, nullptr);
            comm_finalize();
            comm_barrier();
        },
        &err));
    EXPECT_TRUE(err.find("after comm_finalize") != std::string::npos);
}

// ---- timer ---------------------------------------------------------------------

static double spin_ms(double ms) {
    const double t0 = now_seconds();
    volatile double sink = 0.0;
    while ((now_seconds() - t0) * 1e3 < ms) sink = sink + 1.0;
    return sink;
}

TEST(phase_timer_accumulates) {
    PhaseTimer t;
    for (int i = 0; i < NUM_PHASES; ++i) EXPECT_EQ(t.ms(static_cast<Phase>(i)), 0.0);

    t.begin(Phase::Fwd);
    spin_ms(5.0);
    t.end(Phase::Fwd);
    t.begin(Phase::Fwd);
    spin_ms(5.0);
    t.end(Phase::Fwd);
    t.begin(Phase::Opt);
    t.end(Phase::Opt);

    // Lower bounds only: wall time can only exceed the spin time.
    EXPECT_TRUE(t.ms(Phase::Fwd) >= 10.0);
    EXPECT_TRUE(t.ms(Phase::Opt) >= 0.0);
    EXPECT_EQ(t.ms(Phase::Data), 0.0);
    EXPECT_NEAR(t.total_ms(), t.ms(Phase::Fwd) + t.ms(Phase::Opt), 1e-9);

    t.reset();
    EXPECT_EQ(t.total_ms(), 0.0);
    EXPECT_TRUE(std::strcmp(phase_name(Phase::Bwd), "bwd") == 0);
    EXPECT_TRUE(std::strcmp(phase_name(Phase::Comm), "comm") == 0);
}

TEST(phase_timer_misuse_aborts) {
    std::string err;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            PhaseTimer t;
            t.begin(Phase::Fwd);
            t.begin(Phase::Bwd);
        },
        &err));
    EXPECT_TRUE(err.find("still open") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            PhaseTimer t;
            t.end(Phase::Data);
        },
        &err));
    EXPECT_TRUE(err.find("<none>") != std::string::npos);
}

TEST_MAIN()
