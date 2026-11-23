// embedding_cuda.cu: CUDA implementation of the embedding ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.2; D-B11: one .cu per op family, mirroring embedding_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B1.3) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void embedding_forward(real_t*, const int32_t*, const real_t*, const real_t*, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("embedding_forward", "B1.3");
}

void embedding_backward(real_t*, real_t*, const real_t*, const int32_t*, int, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("embedding_backward", "B1.3");
}

}  // namespace ops
}  // namespace tslm
