// test.h: minimal bespoke unit-test harness (ROADMAP P0.3).
//
// Usage (one test executable per tests/test_*.cpp):
//   #include "test.h"
//   TEST(adds) { ASSERT_EQ(1 + 1, 2); EXPECT_NEAR(0.1 + 0.2, 0.3, 1e-12); }
//   TEST_MAIN()
//
// EXPECT_* records a failure and continues; ASSERT_* records a failure and
// returns from the current TEST body. Each failure prints file:line, the
// expression, and both values. TEST_MAIN runs every test in definition
// order, prints one line per test plus a summary, and exits nonzero if any
// test failed. `make test` relies on that exit code.
//
// Comparisons: *_EQ uses ==. *_NEAR passes iff |a - b| <= tol, so NaN always
// fails. *_TRUE takes a condition.
//
// dies_with_abort(fn, &stderr_text) runs fn in a forked child and reports
// whether it ended with SIGABRT (i.e. a CHECK fired), capturing its stderr.
// POSIX-only, as are all target platforms (Linux workstations and cluster).
#pragma once

#include <cmath>
#include <csignal>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace tslm {
namespace test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& current_failures() {
    static int n = 0;
    return n;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// Value formatting for failure messages.
template <typename T>
std::string show(const T& v) {
    char buf[64];
    if constexpr (std::is_same_v<T, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_enum_v<T>) {
        return show(static_cast<std::underlying_type_t<T>>(v));
    } else if constexpr (std::is_floating_point_v<T>) {
        std::snprintf(buf, sizeof(buf), "%.17g", static_cast<double>(v));
        return buf;
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
        return buf;
    } else if constexpr (std::is_integral_v<T>) {
        std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
        return buf;
    } else if constexpr (std::is_convertible_v<T, std::string>) {
        return "\"" + std::string(v) + "\"";
    } else if constexpr (std::is_pointer_v<T>) {
        std::snprintf(buf, sizeof(buf), "%p", static_cast<const void*>(v));
        return buf;
    } else {
        return "<unprintable>";
    }
}

inline void report(const char* file, int line, const std::string& msg) {
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++current_failures();
}

template <typename A, typename B>
bool check_eq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
    if (a == b) return true;
    report(file, line, std::string(ea) + " == " + eb + "  (" + show(a) + " vs " + show(b) + ")");
    return false;
}

inline bool check_near(double a, double b, double tol, const char* ea, const char* eb,
                       const char* file, int line) {
    if (std::fabs(a - b) <= tol) return true;
    report(file, line,
           std::string(ea) + " ~= " + eb + "  (" + show(a) + " vs " + show(b) +
               ", |diff|=" + show(std::fabs(a - b)) + ", tol=" + show(tol) + ")");
    return false;
}

inline bool check_true(bool c, const char* expr, const char* file, int line) {
    if (c) return true;
    report(file, line, std::string(expr) + " is false");
    return false;
}

template <typename Fn>
bool dies_with_abort(Fn fn, std::string* captured_stderr) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return false;
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        fn();
        _exit(0);  // fn returned normally: no abort
    }
    close(pipefd[1]);
    char buf[512];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        if (captured_stderr) captured_stderr->append(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

inline int run_all() {
    int failed_tests = 0;
    const auto& tests = registry();
    for (const TestCase& t : tests) {
        current_failures() = 0;
        t.fn();
        const bool ok = current_failures() == 0;
        if (!ok) ++failed_tests;
        std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", t.name);
        std::fflush(stdout);
    }
    std::printf("%zu tests, %d failed\n", tests.size(), failed_tests);
    return failed_tests == 0 ? 0 : 1;
}

}  // namespace test
}  // namespace tslm

#define TEST(name)                                                              \
    static void tslm_test_##name();                                             \
    static const ::tslm::test::Registrar tslm_test_registrar_##name(#name, tslm_test_##name); \
    static void tslm_test_##name()

#define TEST_MAIN() \
    int main() { return ::tslm::test::run_all(); }

#define EXPECT_EQ(a, b) ((void)::tslm::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__))
#define EXPECT_NEAR(a, b, tol) \
    ((void)::tslm::test::check_near((a), (b), (tol), #a, #b, __FILE__, __LINE__))
#define EXPECT_TRUE(c) ((void)::tslm::test::check_true(static_cast<bool>(c), #c, __FILE__, __LINE__))

#define ASSERT_EQ(a, b) \
    do { if (!::tslm::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)) return; } while (0)
#define ASSERT_NEAR(a, b, tol) \
    do { if (!::tslm::test::check_near((a), (b), (tol), #a, #b, __FILE__, __LINE__)) return; } while (0)
#define ASSERT_TRUE(c) \
    do { if (!::tslm::test::check_true(static_cast<bool>(c), #c, __FILE__, __LINE__)) return; } while (0)
