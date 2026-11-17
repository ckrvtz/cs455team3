// test_build.cpp: checks that the Makefile's BACKEND/COMM/REAL/MODE selection
// reaches the code consistently (ROADMAP P0.1). Uses a plain main() returning
// nonzero on failure; the shared tests/test.h harness arrives in P0.3.
#include "tslm/build_config.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_failures = 0;

static void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

int main() {
    const std::string desc = tslm::build_description();
    std::printf("build: %s\n", desc.c_str());

    const std::string want_prefix = std::string("backend=") + tslm::BUILD_BACKEND +
                                    " comm=" + tslm::BUILD_COMM + " real=" + tslm::BUILD_REAL +
                                    " mode=";
    expect(desc.compare(0, want_prefix.size(), want_prefix) == 0,
           "build_description() matches the constexpr build constants");

    const std::string mode = desc.substr(desc.rfind('=') + 1);
    expect(mode == "release" || mode == "debug", "mode is release or debug");

    expect(tslm::REAL_IS_DOUBLE == (std::strcmp(tslm::BUILD_REAL, "double") == 0),
           "REAL_IS_DOUBLE agrees with BUILD_REAL");

    // Phase A: only the serial CPU build exists (other combos #error at build time).
    expect(std::strcmp(tslm::BUILD_BACKEND, "cpu") == 0, "backend is cpu in Phase A");
    expect(std::strcmp(tslm::BUILD_COMM, "serial") == 0, "comm is serial in Phase A");

    std::printf("test_build: %s (%d failure%s)\n", g_failures ? "FAILED" : "ok", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
