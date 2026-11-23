// test_build.cpp: checks that the Makefile's BACKEND/COMM/REAL/MODE selection
// reaches the code consistently (ROADMAP P0.1).
#include "test.h"

#include "tslm/build_config.h"

#include <cstdio>
#include <cstring>
#include <string>

TEST(build_description_matches_constants) {
    const std::string desc = tslm::build_description();
    std::printf("    build: %s\n", desc.c_str());

    const std::string want_prefix = std::string("backend=") + tslm::BUILD_BACKEND +
                                    " comm=" + tslm::BUILD_COMM + " real=" + tslm::BUILD_REAL +
                                    " mode=";
    EXPECT_EQ(desc.compare(0, want_prefix.size(), want_prefix), 0);

    const std::string mode = desc.substr(desc.rfind('=') + 1);
    EXPECT_TRUE(mode == "release" || mode == "debug");
    EXPECT_EQ(tslm::REAL_IS_DOUBLE, std::strcmp(tslm::BUILD_REAL, "double") == 0);
}

// Phase B: CPU and CUDA backends exist; MPI (Phases C/D) still #errors at
// build time, so every build that compiles is serial.
TEST(phase_b_backends_serial_comm) {
    const std::string backend = tslm::BUILD_BACKEND;
    EXPECT_TRUE(backend == "cpu" || backend == "cuda");
#if defined(TSLM_BACKEND_CUDA)
    EXPECT_EQ(backend, std::string("cuda"));
#else
    EXPECT_EQ(backend, std::string("cpu"));
#endif
    EXPECT_EQ(std::string(tslm::BUILD_COMM), std::string("serial"));
}

TEST_MAIN()
