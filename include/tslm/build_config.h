// build_config.h: compile-time build selection (ROADMAP §4.1, CLAUDE.md §4.10).
//
// The Makefile passes exactly one macro from each group:
//   backend: TSLM_BACKEND_CPU   | TSLM_BACKEND_CUDA
//   comm:    TSLM_COMM_SERIAL   | TSLM_COMM_MPI
//   real:    TSLM_REAL_FLOAT    | TSLM_REAL_DOUBLE
// Missing, conflicting, or not-yet-supported selections fail at build time
// with #error (never at runtime). Selection is compile-time only; there is no
// virtual dispatch. This header must stay nvcc-compatible plain C++17.
#pragma once

// ---- exactly one per group ------------------------------------------------
#if defined(TSLM_BACKEND_CPU) + defined(TSLM_BACKEND_CUDA) != 1
#error "tslm: define exactly one of TSLM_BACKEND_CPU / TSLM_BACKEND_CUDA (use the Makefile: BACKEND=cpu|cuda)"
#endif
#if defined(TSLM_COMM_SERIAL) + defined(TSLM_COMM_MPI) != 1
#error "tslm: define exactly one of TSLM_COMM_SERIAL / TSLM_COMM_MPI (use the Makefile: COMM=serial|mpi)"
#endif
#if defined(TSLM_REAL_FLOAT) + defined(TSLM_REAL_DOUBLE) != 1
#error "tslm: define exactly one of TSLM_REAL_FLOAT / TSLM_REAL_DOUBLE (use the Makefile: REAL=float|double)"
#endif

// ---- combinations not supported in the current phase (Phase B) ------------
// Phase B added BACKEND=cuda (B0.2); Phase C adds COMM=mpi, Phase D adds both.
// Each phase removes the corresponding guard when its implementation lands.
#if defined(TSLM_BACKEND_CUDA) && defined(TSLM_COMM_MPI)
#error "tslm: BACKEND=cuda COMM=mpi is not supported yet (Phase D)"
#elif defined(TSLM_COMM_MPI)
#error "tslm: COMM=mpi is not supported yet (Phase C)"
#endif

namespace tslm {

#if defined(TSLM_BACKEND_CUDA)
constexpr const char* BUILD_BACKEND = "cuda";
#else
constexpr const char* BUILD_BACKEND = "cpu";
#endif

#if defined(TSLM_COMM_MPI)
constexpr const char* BUILD_COMM = "mpi";
#else
constexpr const char* BUILD_COMM = "serial";
#endif

#if defined(TSLM_REAL_DOUBLE)
constexpr bool REAL_IS_DOUBLE = true;
constexpr const char* BUILD_REAL = "double";
#else
constexpr bool REAL_IS_DOUBLE = false;
constexpr const char* BUILD_REAL = "float";
#endif

// Human-readable build tag, e.g. "backend=cpu comm=serial real=float mode=release".
// Intended for run manifests and benchmark CSVs.
const char* build_description();

}  // namespace tslm
