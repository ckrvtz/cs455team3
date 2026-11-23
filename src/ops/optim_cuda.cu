// optim_cuda.cu: CUDA implementation of the optimizer ops of include/tslm/ops.h
// (ROADMAP_CUDA.md §5.1; D-B11: one .cu per op family, mirroring optim_cpu.cpp).
//
// B0.2 STUB: every entry point has the exact ops.h signature and aborts with
// "not implemented: <op>" until its task (B1.1, B1.2, B1.4) replaces it with the kernels,
// their header comments (shapes, launch configuration, reduction order) and
// the matching tests/test_equiv_ops.cpp rows.
#include "tslm/ops.h"

#include "cuda_common.cuh"

namespace tslm {
namespace ops {

double sumsq_reduce(const real_t*, size_t) { TSLM_CUDA_NOT_IMPLEMENTED("sumsq_reduce", "B1.2"); }

void scale_inplace(real_t*, real_t, size_t) { TSLM_CUDA_NOT_IMPLEMENTED("scale_inplace", "B1.1"); }

void adamw_step(real_t*, const real_t*, real_t*, real_t*, size_t, real_t, real_t, real_t, real_t, real_t, int) {
    TSLM_CUDA_NOT_IMPLEMENTED("adamw_step", "B1.4");
}

}  // namespace ops
}  // namespace tslm
