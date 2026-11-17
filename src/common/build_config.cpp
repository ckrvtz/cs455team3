// build_config.cpp: string form of the compile-time build selection.
// TSLM_BUILD_MODE is passed by the Makefile ("release" or "debug").
#include "tslm/build_config.h"

#ifndef TSLM_BUILD_MODE
#error "tslm: TSLM_BUILD_MODE must be defined by the Makefile"
#endif

namespace tslm {

#if defined(TSLM_BACKEND_CUDA)
#define TSLM_BACKEND_STR "cuda"
#else
#define TSLM_BACKEND_STR "cpu"
#endif
#if defined(TSLM_COMM_MPI)
#define TSLM_COMM_STR "mpi"
#else
#define TSLM_COMM_STR "serial"
#endif
#if defined(TSLM_REAL_DOUBLE)
#define TSLM_REAL_STR "double"
#else
#define TSLM_REAL_STR "float"
#endif

const char* build_description() {
    // Built with string-literal concatenation so it is a constant with no
    // allocation or static-initialization order concerns.
    return "backend=" TSLM_BACKEND_STR " comm=" TSLM_COMM_STR " real=" TSLM_REAL_STR
           " mode=" TSLM_BUILD_MODE;
}

}  // namespace tslm
