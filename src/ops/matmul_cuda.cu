// matmul_cuda.cu: CUDA implementation of the matmul ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.4; D-B11: one .cu per op family, mirroring matmul_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B2.2) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void matmul_forward(real_t*, const real_t*, const real_t*, const real_t*, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("matmul_forward", "B2.2");
}

void matmul_backward(real_t*, real_t*, real_t*, const real_t*, const real_t*, const real_t*, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("matmul_backward", "B2.2");
}

}  // namespace ops
}  // namespace tslm
