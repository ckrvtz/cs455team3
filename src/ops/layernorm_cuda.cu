// layernorm_cuda.cu: CUDA implementation of the LayerNorm ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.3; D-B11: one .cu per op family, mirroring layernorm_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B3.1) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

void layernorm_forward(real_t*, real_t*, real_t*, const real_t*, const real_t*, const real_t*, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("layernorm_forward", "B3.1");
}

void layernorm_backward(real_t*, real_t*, real_t*, const real_t*, const real_t*, const real_t*, const real_t*,
                        const real_t*, int, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("layernorm_backward", "B3.1");
}

}  // namespace ops
}  // namespace tslm
