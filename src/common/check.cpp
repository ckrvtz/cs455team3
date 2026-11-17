// check.cpp: failure path for CHECK (see include/tslm/check.h).
#include "tslm/check.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace tslm {

void check_fail(const char* file, int line, const char* cond, const char* fmt, ...) {
    // Flush stdout first so already-printed progress is not lost or
    // interleaved after the error on a shared terminal/log.
    std::fflush(stdout);
    std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n  ", cond, file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::abort();
}

}  // namespace tslm
