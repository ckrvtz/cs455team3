// comm_mpi.cpp: MPI implementation of include/tslm/comm.h (ROADMAP Phase C).
// STUB (P7.5): compiled only with `make COMM=mpi` (by $(MPICXX)), and it
// stops the build here until Phase C replaces it. This is the only file
// that may include <mpi.h> (CLAUDE.md §4.4); the #error comes first so a
// missing MPI installation is not what gets reported.
//
// Phase C implements comm.h with the same semantics as comm_serial.cpp:
//   comm_init(&argc, &argv)        MPI_Init; record rank and size
//   comm_rank(), comm_size()       MPI_Comm_rank / MPI_Comm_size on MPI_COMM_WORLD
//   comm_broadcast(p, n, root)     MPI_Bcast of the flat params after init
//                                  (MPI_FLOAT, or MPI_DOUBLE when real_t is double)
//   comm_allreduce_avg(p, n)       ONE MPI_Allreduce(MPI_SUM) of the flat grads per
//                                  step, then scale by 1/size
//   comm_allreduce_sum_double(p,n) MPI_Allreduce(MPI_SUM) on host doubles
//   comm_barrier(), comm_finalize()
// Keep the init/active/finalized lifecycle checks of comm_serial.cpp. The
// step order is fixed: zero -> forward -> backward -> allreduce -> clip -> AdamW
// (the clip norm is computed after the allreduce so all ranks agree). In
// Phase D the buffers are device pointers: use CUDA-aware MPI if available,
// otherwise stage through host buffers. The BACKEND/COMM guards in
// include/tslm/build_config.h are removed when this lands.
#error "tslm: src/comm/comm_mpi.cpp is a stub; the MPI build is Phase C (see ROADMAP §4.5)"
