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

## 2026-11-17: Config format and error handling (P0.2)

- **Change:** `CHECK(cond, fmt, ...)` lives in `include/tslm/check.h` (the §5
  header list includes it) and is re-exported by `common.h`. `CUDA_CHECK` and
  `MPI_CHECK` are compile-time `static_assert` failures in builds without
  that backend until Phases B and C implement them. Config keys (one table in
  `src/common/config.cpp`): model `vocab_size max_seq_len n_layer n_embd
  n_head tie_embeddings`; train `run_name out_dir train_data val_data seed
  batch_size seq_len max_steps warmup_steps lr min_lr_frac weight_decay beta1
  beta2 adam_eps grad_clip log_every eval_every val_batches ckpt_every`.
  Precedence: defaults < `--config <file>` < `--key value`. Unknown and
  duplicate keys, and malformed or non-finite values, are errors. Full-line
  `#` comments are allowed; inline comments are not (paths may contain `#`).
- **Reason:** ROADMAP P0.2 / CLAUDE.md §9. The serialized form uses `%.17g`, so
  configs round-trip bit-exactly. P4.2 should write any extra `manifest.txt`
  metadata (git hash, compiler, CPU) as `#` comment lines, so that a manifest
  stays loadable with `--config`.
- **Placeholders (to tune, not spec):** `max_steps`, `eval_every`,
  `ckpt_every`, `val_batches` in all presets, and `smoke`'s `lr=1e-3` and
  `warmup_steps=20` (the 200-step default would cover its whole run).
- **Impact:** none on CUDA/MPI beyond removing the `CUDA_CHECK`/`MPI_CHECK`
  stubs in their phases. `ModelConfig` is all plain `int`, so it can be
  shared with nvcc code and the checkpoint header.

## 2026-11-17: `make dist` excludes PDFs

- **Change:** `make dist` excludes every `*.pdf`, such as the reference paper
  and the proposal, and its tarball check rejects any PDF that slips through.
- **Reason:** team decision. The reference documents are not part of the submission.
- **Impact:** none on any build.

## 2026-11-17: RNG seeding, runtime shims, test harness (P0.3)

- **Change:**
  - **RNG seeding:** `Rng(seed, stream)` sets
    `key = mix64(seed) ^ mix64(stream*0x9E3779B97F4A7C15 + 0xD1B54A32D192ED03)`,
    then fills the four xoshiro256** state words with four `splitmix64` steps
    from `key`.
  - **`normal()`:** Box–Muller with `u1 = 1 - uniform()` in (0,1], keeping
    the cosine branch only and caching nothing, so the RNG state is exactly
    the 4 words.
  - **`uniform_int`:** rejection sampling, so there is no modulo bias.
  - **Contract signatures:** `comm_init(int* argc, char*** argv)` takes
    pointers, as `MPI_Init` does. `backend.h` copies and memsets are untyped
    byte counts (like `cudaMemcpy`). `dev_malloc` returns 64-byte-aligned
    memory and aborts on failure.
  - **`PhaseTimer`:** phases `data, fwd, bwd, comm, opt` (the log.csv
    columns). Only one phase can be open at a time, and `begin`/`end` call
    `dev_sync()`.
  - **Test harness:** `tests/test.h` provides
    `TEST/TEST_MAIN/EXPECT_*/ASSERT_*` and a fork-based `dies_with_abort`
    (POSIX). The P0.1/P0.2 tests were moved onto it.
- **Reason:** ROADMAP §4.2, §4.6, P0.3. The known-answer values in
  `tests/test_rng.cpp` were recorded once. They were cross-checked against an
  independent from-the-definition implementation and the published
  splitmix64/xoshiro256** reference outputs.
- **Impact:** CUDA (Phase B) implements the same `backend.h` signatures with
  device calls; `dev_sync()` becomes `cudaDeviceSynchronize`. MPI (Phase C)
  implements `comm.h` in `comm_mpi.cpp`. Changing the seeding formula would
  change every initialization and data order, so treat it as frozen once
  golden files exist (P2.6).
- **Not yet assigned:** the `zero` step in §9 has no log.csv column. P4.2
  must decide which phase it is charged to (suggest `bwd`).
