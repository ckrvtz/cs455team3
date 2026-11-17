// backend_cpu.cpp: CPU implementation of backend.h (BACKEND=cpu).
// "Device" memory is ordinary host memory. Allocations use aligned_alloc with
// the size rounded up to a multiple of DEV_ALIGN, as aligned_alloc requires.
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

}  // namespace

void* dev_malloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    CHECK(bytes <= SIZE_MAX - DEV_ALIGN, "dev_malloc: size overflow (%zu bytes)", bytes);
    const size_t rounded = (bytes + DEV_ALIGN - 1) / DEV_ALIGN * DEV_ALIGN;
    void* p = std::aligned_alloc(DEV_ALIGN, rounded);
    CHECK(p != nullptr, "dev_malloc: out of memory allocating %zu bytes", bytes);
    return p;
}

void dev_free(void* p) { std::free(p); }

void dev_memset(void* p, int value, size_t bytes) {
    if (bytes == 0) return;
    CHECK(p != nullptr, "dev_memset: null pointer with %zu bytes", bytes);
    std::memset(p, value, bytes);
}

void copy_h2d(void* dst_dev, const void* src_host, size_t bytes) {
    check_copy_args(dst_dev, src_host, bytes, "copy_h2d");
    if (bytes) std::memcpy(dst_dev, src_host, bytes);
}

void copy_d2h(void* dst_host, const void* src_dev, size_t bytes) {
    check_copy_args(dst_host, src_dev, bytes, "copy_d2h");
    if (bytes) std::memcpy(dst_host, src_dev, bytes);
}

void copy_d2d(void* dst_dev, const void* src_dev, size_t bytes) {
    check_copy_args(dst_dev, src_dev, bytes, "copy_d2d");
    if (bytes) std::memcpy(dst_dev, src_dev, bytes);
}

void dev_sync() {}

const char* dev_name() { return "cpu"; }

}  // namespace tslm
