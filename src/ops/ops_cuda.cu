// ops_cuda.cu: CUDA implementation of include/tslm/ops.h (ROADMAP Phase B).
// STUB (P7.5): compiled only with `make BACKEND=cuda` (by $(NVCC)), and it
// stops the build here until Phase B replaces it.
//
// Phase B implements, with the exact signatures of ops.h and the rules and
// per-op notes in docs/HANDOFF.md (overwrite vs accumulate, shapes,
// tolerances vs serial):
//   embedding_forward / embedding_backward
//   layernorm_forward / layernorm_backward
//   gelu_forward / gelu_backward          (stable x*sigmoid(2u) form)
//   residual_forward / residual_backward
//   matmul_forward / matmul_backward      (cuBLAS SGEMM, beta = 1 for grads, no TF32)
//   attention_forward / attention_backward (exact zeros above the diagonal; scratch only)
//   softmax_xent_forward / softmax_xent_backward
//   sum_reduce / sumsq_reduce             (double accumulation, result to host)
//   scale_inplace / adamw_step
// It also needs src/backend/backend_cuda.cu (dev_malloc = cudaMalloc, ...) and
// the BACKEND=cuda guard in include/tslm/build_config.h removed.
#error "tslm: src/ops/ops_cuda.cu is a stub; the CUDA kernels are Phase B (see docs/HANDOFF.md)"
