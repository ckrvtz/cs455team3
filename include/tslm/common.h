// common.h: project-wide basics. Include this first in every tslm file.
//
//   real_t   tensor element type: float, or double in the REAL=double build
//            used for gradient checks. Never hard-code float in ops/model.
//   CHECK    fatal check with file/line and a printf-style message (check.h).
//
// Must stay nvcc-compatible plain C++17 (no iostream, no <filesystem>).
#pragma once

#include "tslm/build_config.h"
#include "tslm/check.h"

namespace tslm {

#if defined(TSLM_REAL_DOUBLE)
using real_t = double;
#else
using real_t = float;
#endif

static_assert(sizeof(real_t) == (REAL_IS_DOUBLE ? sizeof(double) : sizeof(float)),
              "real_t disagrees with the REAL build selection");

}  // namespace tslm
