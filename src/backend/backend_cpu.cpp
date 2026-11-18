// backend_cpu.cpp: CPU implementation of backend.h (BACKEND=cpu).
// "Device" memory is ordinary host memory. Allocations use aligned_alloc with
// the size rounded up to a multiple of DEV_ALIGN, as aligned_alloc requires.
// Each block starts with a DEV_ALIGN-byte header holding the requested size,
// so dev_free can keep the byte counters behind dev_peak_bytes(); the pointer
// returned to the caller is block + DEV_ALIGN and therefore still aligned.
#include "tslm/backend.h"

#include "tslm/common.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace tslm {

namespace {

// Copies require non-overlapping regions (memcpy semantics, like cudaMemcpy).
void check_copy_args(void* dst, const void* src, size_t bytes, const char* what) {
    if (bytes == 0) return;
    CHECK(dst != nullptr && src != nullptr, "%s: null pointer with %zu bytes", what, bytes);
    const char* d = static_cast<const char*>(dst);
    const char* s = static_cast<const char*>(src);
    CHECK(d + bytes <= s || s + bytes <= d, "%s: overlapping regions (%zu bytes)", what, bytes);
}

size_t g_current_bytes = 0;  // requested bytes currently allocated
size_t g_peak_bytes = 0;
bool g_used = false;         // any dev_* call made (dev_init must come first)
bool g_initialised = false;  // dev_init called

void mark_used() { g_used = true; }

}  // namespace

void dev_init(int local_rank) {
    CHECK(local_rank >= 0, "dev_init: local_rank %d must be >= 0", local_rank);
    CHECK(!g_initialised, "dev_init called twice");
    CHECK(!g_used, "dev_init must be called before any other dev_* call");
    g_initialised = true;
}

size_t dev_peak_bytes() { return g_peak_bytes; }

void* dev_malloc(size_t bytes) {
    mark_used();
    if (bytes == 0) return nullptr;
    CHECK(bytes <= SIZE_MAX - 2 * DEV_ALIGN, "dev_malloc: size overflow (%zu bytes)", bytes);
    const size_t rounded = (bytes + DEV_ALIGN - 1) / DEV_ALIGN * DEV_ALIGN + DEV_ALIGN;
    char* base = static_cast<char*>(std::aligned_alloc(DEV_ALIGN, rounded));
    CHECK(base != nullptr, "dev_malloc: out of memory allocating %zu bytes", bytes);
    std::memcpy(base, &bytes, sizeof(bytes));
    g_current_bytes += bytes;
    if (g_current_bytes > g_peak_bytes) g_peak_bytes = g_current_bytes;
    return base + DEV_ALIGN;
}

void dev_free(void* p) {
    mark_used();
    if (!p) return;
    char* base = static_cast<char*>(p) - DEV_ALIGN;
    size_t bytes = 0;
    std::memcpy(&bytes, base, sizeof(bytes));
    CHECK(bytes <= g_current_bytes, "dev_free: corrupt block header or double free (%zu bytes recorded)", bytes);
    g_current_bytes -= bytes;
    std::free(base);
}

void dev_memset(void* p, int value, size_t bytes) {
    mark_used();
    if (bytes == 0) return;
    CHECK(p != nullptr, "dev_memset: null pointer with %zu bytes", bytes);
    std::memset(p, value, bytes);
}

void copy_h2d(void* dst_dev, const void* src_host, size_t bytes) {
    mark_used();
    check_copy_args(dst_dev, src_host, bytes, "copy_h2d");
    if (bytes) std::memcpy(dst_dev, src_host, bytes);
}

void copy_d2h(void* dst_host, const void* src_dev, size_t bytes) {
    mark_used();
    check_copy_args(dst_host, src_dev, bytes, "copy_d2h");
    if (bytes) std::memcpy(dst_host, src_dev, bytes);
}

void copy_d2d(void* dst_dev, const void* src_dev, size_t bytes) {
    mark_used();
    check_copy_args(dst_dev, src_dev, bytes, "copy_d2d");
    if (bytes) std::memcpy(dst_dev, src_dev, bytes);
}

void dev_sync() { mark_used(); }

const char* dev_name() { return "cpu"; }

}  // namespace tslm
