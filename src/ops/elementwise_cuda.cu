// elementwise_cuda.cu: CUDA implementation of the GELU and residual ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.1; D-B11: one .cu per op family, mirroring elementwise_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B1.1) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void gelu_forward(real_t*, const real_t*, size_t) { TSLM_CUDA_NOT_IMPLEMENTED("gelu_forward", "B1.1"); }

void gelu_backward(real_t*, const real_t*, const real_t*, size_t) {
    TSLM_CUDA_NOT_IMPLEMENTED("gelu_backward", "B1.1");
}

void residual_forward(real_t*, const real_t*, const real_t*, size_t) {
    TSLM_CUDA_NOT_IMPLEMENTED("residual_forward", "B1.1");
}

void residual_backward(real_t*, real_t*, const real_t*, size_t) {
    TSLM_CUDA_NOT_IMPLEMENTED("residual_backward", "B1.1");
}

}  // namespace ops
}  // namespace tslm
