// check.h: fatal-error checks (CLAUDE.md §5 "Error handling").
//
// CHECK(cond, fmt, ...) evaluates `cond` once; if false it prints
//   CHECK failed: <cond>
//     at <file>:<line>
//     <printf-formatted message>
// to stderr and calls std::abort(). It is always on (release builds too).
// It is for programmer errors, I/O failures, and bad config, never for
// expected control flow. Host code only. ops:: kernels must not use it,
// because they may later run on the device.
//
// CUDA_CHECK / MPI_CHECK are stubs until Phase B / Phase C. Using either one
// in a build without that backend fails at compile time (static_assert), so
// a misplaced call can never compile into the serial build silently.
#pragma once

#include "tslm/build_config.h"

namespace tslm {

#if defined(__GNUC__)
[[noreturn]] void check_fail(const char* file, int line, const char* cond, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));
#else
[[noreturn]] void check_fail(const char* file, int line, const char* cond, const char* fmt, ...);
#endif

}  // namespace tslm

// The format string is part of __VA_ARGS__, so CHECK(c, "plain text") is
// valid ISO C++17 (no empty variadic pack).
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) ::tslm::check_fail(__FILE__, __LINE__, #cond, __VA_ARGS__); \
    } while (0)

#if defined(TSLM_BACKEND_CUDA)
#error "tslm: CUDA_CHECK is not implemented yet (Phase B)"
#else
#define CUDA_CHECK(call) static_assert(sizeof(#call) == 0, "CUDA_CHECK used in a non-CUDA build")
#endif

#if defined(TSLM_COMM_MPI)
#error "tslm: MPI_CHECK is not implemented yet (Phase C)"
#else
#define MPI_CHECK(call) static_assert(sizeof(#call) == 0, "MPI_CHECK used in a non-MPI build")
#endif
