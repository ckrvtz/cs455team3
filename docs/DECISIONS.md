# Decision log (append-only)

Format: date, change, reason, impact on other builds (CUDA / MPI / CUDA+MPI).

## 2026-11-17: Build system layout (P0.1)

- **Change:** `Makefile` maps `BACKEND`/`COMM`/`REAL` to `-DTSLM_BACKEND_*`,
  `-DTSLM_COMM_*`, `-DTSLM_REAL_*` (plus `MODE=release|debug`). New header
  `include/tslm/build_config.h` (not in the ROADMAP §5 list) validates them
  and `#error`s on unsupported combinations. Make itself rejects unknown
  values. Each configuration builds into `build/<mode>-<backend>-<comm>-<real>/`;
  release float apps go to `bin/`, other mode/precision builds to
  `bin/<mode>-<real>/`. `apps/train.cpp` becomes `train_{serial,cuda,mpi,mpi_cuda}`.
  Library sources named `*_cpu.cpp`, `*_cuda.cpp`, `*_serial.cpp`, `*_mpi.cpp`
  are compiled only when that backend/comm is selected.
- **Reason:** compile-time selection with build-time failures (ROADMAP §4.1).
  Separate build directories keep `REAL=double` and debug objects from
  mixing with release objects.
- **Impact:** Phase B/C/D each remove their `#error` guard in
  `build_config.h` and add their toolchain to the Makefile (nvcc for `.cu`,
  mpicxx for `COMM=mpi`). Phase A currently compiles every configuration with
  `$(CXX)`, so the `#error` messages show up even where nvcc/mpicxx are not installed.
- Also: `.gitignore` (GitHub C++ template) ignored `Makefile`. That line was
  removed, and `data/ runs/ bin/ build/ dist/` were added.
