// cuda_common.cuh: device-side helpers shared by the src/ops/*_cuda.cu files
// (ROADMAP_CUDA.md §4.1, §4.3; D-B11). Kernel-internal: never included by
// ops.h or by host .cpp code.
//
// B0.2 provides the launch check, ceil_div and the not-implemented abort used
// by the per-family stubs. The fixed-order warp/block reductions are added
// with their first user (B1.2).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "tslm/common.h"

namespace tslm {
namespace cuda {

// Check a kernel launch: always the launch error; in MODE=debug
// (TSLM_CUDA_DEBUG_SYNC, set by the Makefile) also synchronise, so an
// asynchronous fault is reported at the op that caused it.
#if defined(TSLM_CUDA_DEBUG_SYNC)
#define CUDA_LAUNCH_CHECK()                       \
    do {                                          \
        CUDA_CHECK(cudaGetLastError());           \
        CUDA_CHECK(cudaDeviceSynchronize());      \
    } while (0)
#else
#define CUDA_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())
#endif

// Number of blocks of `block` threads covering n items (n > 0).
inline size_t ceil_div(size_t n, size_t block) { return (n + block - 1) / block; }

// Abort for an ops:: entry point whose CUDA kernel is not written yet. Every
// stub calls it, so a CUDA binary stops loudly at the first unimplemented op
// (CLAUDE.md §10: no stub behind a passing test). Removed with the last stub.
[[noreturn]] inline void not_implemented(const char* file, int line, const char* op, const char* task) {
    ::tslm::check_fail(file, line, "implemented", "not implemented: %s (CUDA build; ROADMAP_CUDA.md task %s)", op,
                       task);
}
#define TSLM_CUDA_NOT_IMPLEMENTED(op, task) ::tslm::cuda::not_implemented(__FILE__, __LINE__, op, task)

}  // namespace cuda
}  // namespace tslm
