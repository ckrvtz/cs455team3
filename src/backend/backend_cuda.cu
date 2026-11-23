// backend_cuda.cu: CUDA implementation of include/tslm/backend.h
// (BACKEND=cuda; ROADMAP_CUDA.md §4.1, §4.4, task B0.2). Same semantics as
// backend_cpu.cpp; the pointers it returns are device pointers.
//
//   dev_init(r)        cudaSetDevice(r % device_count), creates the CUDA
//                      context, the cuBLAS handle and the reduction scratch
//                      (cuda_internal.h). Optional: the first other dev_* call
//                      initialises device 0 the same way. If called, it must
//                      come first, once, with r >= 0 (as in the CPU backend).
//   dev_malloc         cudaMalloc (256-byte aligned, so DEV_ALIGN holds);
//                      the requested size is remembered in a host-side table
//                      for dev_peak_bytes() (allocations happen only at setup,
//                      never in the training loop, so the table costs nothing
//                      per step; it is only looked up, never iterated).
//   dev_free           cudaFree; aborts on a pointer dev_malloc did not return.
//   dev_memset         cudaMemset on the default stream.
//   copy_h2d/d2h/d2d   cudaMemcpy (h2d/d2h synchronous with the host; d2d is
//                      ordered on the default stream like every kernel).
//   dev_sync           cudaDeviceSynchronize.
//   dev_name           e.g. "Quadro M2000M (sm_50), driver 580.178.04 (CUDA
//                      13.0), CUDA runtime 12.9". The driver version string
//                      comes from /proc/driver/nvidia/version (NVML is not an
//                      approved dependency); "driver unknown" if that file is
//                      absent or has an unexpected format.
// Everything runs on the default stream, so all work is ordered (D-B3). The
// cuBLAS handle and the scratch live until process exit (destroying them in
// a static destructor could run after the CUDA runtime has shut down).
#include "tslm/backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include "cuda_internal.h"
#include "tslm/common.h"

namespace tslm {

namespace {

bool g_used = false;         // any dev_* call made (dev_init must come first)
bool g_initialised = false;  // device selected and resources created
int g_device = -1;
cublasHandle_t g_cublas = nullptr;
double* g_reduce_scratch = nullptr;
size_t g_current_bytes = 0;  // requested bytes currently allocated
size_t g_peak_bytes = 0;
std::unordered_map<const void*, size_t> g_sizes;  // dev_malloc pointer -> requested bytes
std::string g_name;

void init_device(int local_rank) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    CHECK(count > 0, "dev_init: no CUDA device found");
    g_device = local_rank % count;
    CUDA_CHECK(cudaSetDevice(g_device));
    CUDA_CHECK(cudaFree(nullptr));  // create the context now, not inside the first timed op
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    CUBLAS_CHECK(cublasSetStream(g_cublas, nullptr));                     // default stream
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));       // no TF32 (D-B4)
    CUBLAS_CHECK(cublasSetPointerMode(g_cublas, CUBLAS_POINTER_MODE_HOST));  // host alpha/beta
    CUDA_CHECK(cudaMalloc(&g_reduce_scratch, cuda::REDUCE_SCRATCH_DOUBLES * sizeof(double)));
    g_initialised = true;
}

// Every dev_* entry point: initialise device 0 if dev_init was not called.
void mark_used() {
    g_used = true;
    if (!g_initialised) init_device(0);
}

// "580.178.04" from "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  <date>".
std::string driver_version_string() {
    std::FILE* f = std::fopen("/proc/driver/nvidia/version", "r");
    if (!f) return "unknown";
    char line[512] = {0};
    const bool got = std::fgets(line, sizeof(line), f) != nullptr;
    std::fclose(f);
    if (!got) return "unknown";
    const char* key = "Kernel Module";
    const char* p = std::strstr(line, key);
    if (!p) return "unknown";
    p += std::strlen(key);
    while (*p == ' ') ++p;
    std::string v;
    while (*p && *p != ' ' && *p != '\n') v += *p++;
    return v.empty() ? "unknown" : v;
}

void check_copy_args(void* dst, const void* src, size_t bytes, const char* what) {
    if (bytes == 0) return;
    CHECK(dst != nullptr && src != nullptr, "%s: null pointer with %zu bytes", what, bytes);
}

}  // namespace

namespace cuda {

cublasHandle_t cublas_handle() {
    mark_used();
    return g_cublas;
}

double* reduce_scratch() {
    mark_used();
    return g_reduce_scratch;
}

}  // namespace cuda

void dev_init(int local_rank) {
    CHECK(local_rank >= 0, "dev_init: local_rank %d must be >= 0", local_rank);
    CHECK(!g_used, "dev_init must be called before any other dev_* call");
    CHECK(!g_initialised, "dev_init called twice");
    init_device(local_rank);
}

size_t dev_peak_bytes() { return g_peak_bytes; }

void* dev_malloc(size_t bytes) {
    mark_used();
    if (bytes == 0) return nullptr;
    void* p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);  // best effort, for the message only
        CHECK(false, "dev_malloc: cudaMalloc of %zu bytes failed: %s (device free %zu MiB of %zu MiB)", bytes,
              cudaGetErrorString(err), free_b >> 20, total_b >> 20);
    }
    CHECK(reinterpret_cast<uintptr_t>(p) % DEV_ALIGN == 0, "dev_malloc: cudaMalloc returned a pointer not aligned to %zu",
          DEV_ALIGN);
    g_sizes[p] = bytes;
    g_current_bytes += bytes;
    if (g_current_bytes > g_peak_bytes) g_peak_bytes = g_current_bytes;
    return p;
}

void dev_free(void* p) {
    mark_used();
    if (!p) return;
    const auto it = g_sizes.find(p);
    CHECK(it != g_sizes.end(), "dev_free: %p was not returned by dev_malloc, or was already freed", p);
    g_current_bytes -= it->second;
    g_sizes.erase(it);
    CUDA_CHECK(cudaFree(p));
}

void dev_memset(void* p, int value, size_t bytes) {
    mark_used();
    if (bytes == 0) return;
    CHECK(p != nullptr, "dev_memset: null pointer with %zu bytes", bytes);
    CUDA_CHECK(cudaMemset(p, value, bytes));
}

void copy_h2d(void* dst_dev, const void* src_host, size_t bytes) {
    mark_used();
    check_copy_args(dst_dev, src_host, bytes, "copy_h2d");
    if (bytes) CUDA_CHECK(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice));
}

void copy_d2h(void* dst_host, const void* src_dev, size_t bytes) {
    mark_used();
    check_copy_args(dst_host, src_dev, bytes, "copy_d2h");
    if (bytes) CUDA_CHECK(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost));
}

void copy_d2d(void* dst_dev, const void* src_dev, size_t bytes) {
    mark_used();
    check_copy_args(dst_dev, src_dev, bytes, "copy_d2d");
    if (bytes == 0) return;
    const char* d = static_cast<const char*>(dst_dev);
    const char* s = static_cast<const char*>(src_dev);
    CHECK(d + bytes <= s || s + bytes <= d, "copy_d2d: overlapping regions (%zu bytes)", bytes);
    CUDA_CHECK(cudaMemcpy(dst_dev, src_dev, bytes, cudaMemcpyDeviceToDevice));
}

void dev_sync() {
    mark_used();
    CUDA_CHECK(cudaDeviceSynchronize());
}

const char* dev_name() {
    mark_used();
    if (g_name.empty()) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, g_device));
        int rt = 0, drv = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&rt));
        CUDA_CHECK(cudaDriverGetVersion(&drv));
        char b[512];
        std::snprintf(b, sizeof(b), "%s (sm_%d%d), driver %s (CUDA %d.%d), CUDA runtime %d.%d", p.name, p.major,
                      p.minor, driver_version_string().c_str(), drv / 1000, (drv % 1000) / 10, rt / 1000,
                      (rt % 1000) / 10);
        g_name = b;
    }
    return g_name.c_str();
}

}  // namespace tslm
