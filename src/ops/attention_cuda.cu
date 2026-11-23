// attention_cuda.cu: CUDA implementation of the attention ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.6; D-B11: one .cu per op family, mirroring attention_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B4.1, B4.2) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void attention_forward(real_t*, real_t*, const real_t*, int, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("attention_forward", "B4.1");
}

void attention_backward(real_t*, real_t*, const real_t*, const real_t*, const real_t*, int, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("attention_backward", "B4.2");
}

}  // namespace ops
}  // namespace tslm
