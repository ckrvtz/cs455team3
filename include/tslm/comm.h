// comm.h: collectives shim (ROADMAP §4.2, §4.5, CLAUDE.md §4.4).
//
// Every inter-process operation goes through these functions; <mpi.h> is
// included only in src/comm/comm_mpi.cpp (Phase C). The serial
// implementation (comm_serial.cpp) is one process: rank 0, size 1, and all
// collectives leave the data unchanged.
//
// Semantics (identical in every implementation):
//   comm_init(&argc, &argv)   once, before any other comm_* call. It takes
//                             pointers because MPI_Init may edit argv.
//   comm_rank(), comm_size()  this process's rank in [0, size), world size.
//   comm_broadcast(p, n, root)          p[0..n) on every rank := root's p[0..n)
//   comm_allreduce_avg(p, n)            p[i] := (sum over ranks of p[i]) / size
//   comm_allreduce_sum_double(p, n)     p[i] := sum over ranks of p[i]
//                                       (host doubles: loss sums, sq norms)
//   comm_barrier()
//   comm_finalize()           once, at exit; no comm_* call afterwards.
// real_t buffers may be device pointers in CUDA builds (Phase D).
// Step order (CLAUDE.md §4.7): zero_grads -> forward -> backward ->
// comm_allreduce_avg(grads) -> clip -> AdamW.
#pragma once

#include <cstddef>

#include "tslm/common.h"

namespace tslm {

void comm_init(int* argc, char*** argv);
int comm_rank();
int comm_size();
void comm_broadcast(real_t* data, size_t n, int root);
void comm_allreduce_avg(real_t* data, size_t n);
void comm_allreduce_sum_double(double* data, int n);
void comm_barrier();
void comm_finalize();

}  // namespace tslm
