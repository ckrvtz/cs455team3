// softmax_xent_cuda.cu: CUDA implementation of the softmax + cross-entropy and sum_reduce ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.5, §5.1; D-B11: one .cu per op family, mirroring softmax_xent_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B3.2; sum_reduce B1.2) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void softmax_xent_forward(real_t*, real_t*, const real_t*, const int32_t*, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("softmax_xent_forward", "B3.2");
}

void softmax_xent_backward(real_t*, const real_t*, const int32_t*, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("softmax_xent_backward", "B3.2");
}

double sum_reduce(const real_t*, size_t) { TSLM_CUDA_NOT_IMPLEMENTED("sum_reduce", "B1.2"); }

}  // namespace ops
}  // namespace tslm
