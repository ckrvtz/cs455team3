// cuda_internal.h: INTERNAL to the CUDA build (ROADMAP_CUDA.md §4.1, §4.3;
// CLAUDE.md §4.12). Not a contract header: included only by
// src/backend/backend_cuda.cu and src/ops/*_cuda.cu, never by host .cpp code.
//
// The only process-wide state ops may touch. Both resources are created once
// by backend_cuda (in dev_init, or at the first dev_* call when dev_init was
// not called) and are never created, resized or freed inside an op:
//   cublas_handle()   the cuBLAS handle: default stream, CUBLAS_DEFAULT_MATH
//                     (no TF32, D-B4), host pointer mode (alpha/beta are host
//                     scalars).
//   reduce_scratch()  REDUCE_SCRATCH_DOUBLES doubles of device memory for the
//                     two-pass deterministic reductions (sum_reduce,
//                     sumsq_reduce, §5.1): pass 1 writes one partial per
//                     block (at most REDUCE_MAX_BLOCKS), pass 2 writes the
//                     result at index REDUCE_MAX_BLOCKS.
// This scratch is not counted by dev_peak_bytes() (which counts dev_malloc
// requests only); it is 8 KiB.
#pragma once

#include <cublas_v2.h>

#include <cstddef>

namespace tslm {
namespace cuda {

constexpr int REDUCE_MAX_BLOCKS = 1024;
constexpr size_t REDUCE_SCRATCH_DOUBLES = size_t(REDUCE_MAX_BLOCKS) + 1;

cublasHandle_t cublas_handle();
double* reduce_scratch();

}  // namespace cuda
}  // namespace tslm
