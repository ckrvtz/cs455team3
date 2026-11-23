# CLAUDE.md: Rules for LLM-Assisted Development

Project: TinyStories small-scale transformer language model, trained in four modes (serial, CUDA, serial+MPI, CUDA+MPI). Current phase: **Phase B, CUDA on a single GPU** (`ROADMAP_CUDA.md`). Phase A (serial) is complete except the manual story evaluations (P6.3 pilot) and the P5.3 review (D-B9).

Read this file fully at the start of every session. Then read the relevant section of the roadmap for the task you were given:
- **Phase B tasks** (`B0.1` ... `B7.2`): `ROADMAP_CUDA.md`. Its §3 holds the team decisions (D-B1 ... D-B12).
- **Phase A tasks** (`P*.*`) and every contract (§2, §4, §6, §8, §9, §10, §12): `ROADMAP.md`. It remains the base specification.
- **Per-kernel requirements:** `docs/HANDOFF.md`.

The roadmaps are the technical specification; this file is the working agreement.

---

## 1. Hard constraints (never violate)

1. **Language:** all model, data, training, and tooling code that ships is **C/C++ (C++17)**, plus CUDA (`.cu`) and OpenMPI only in the builds that call for them. Build tooling may be shell scripts, Makefiles, `.json`, `.txt`, `.cfg`, spreadsheets.
2. **No Python in the deliverable.** Python/PyTorch is allowed only under `dev_only/` for offline planning, golden-file generation, validation, plots, and reports. Nothing in `make`, `make test`, `make dist`, or any run script may call Python or read a file that only Python can produce at build/test time. Golden data files generated offline are committed as plain data (`.bin`, `.txt`, `.json`) under `tests/golden/`.
3. **Bespoke model code.** The tokenizer algorithm, transformer ops, backward passes, optimizer, data loader, and training loop must be written for this project. Do **not** paste or closely paraphrase code from llm.c, nanoGPT, GGML, PyTorch internals, or similar. If a snippet or idea is adapted from any open source work, stop and tell the user, then record it in `THIRD_PARTY.md` (name, URL, version/commit, license, what was used, where) before finishing the task.
4. **Open source libraries** are allowed only with citation in `THIRD_PARTY.md`, and **only after the user approves the specific library**. Do not add dependencies on your own. Standard library, OpenMPI, the CUDA runtime and **cuBLAS** (approved, D-B2) are the expected set.
5. **Serial baseline purity:** no OpenMP, threads, SIMD intrinsics, or BLAS in the serial build. Compiler auto-vectorization at `-O3 -march=native` is fine. `-ffast-math` is forbidden.
6. **CUDA rules** (Phase B, `ROADMAP_CUDA.md` §3):
   - **cuBLAS:** only for the matrix products: `matmul_forward/backward` and the batched GEMMs inside `attention_forward/backward` (D-B2). Every other kernel is written for this project.
   - **No other GPU libraries** (cuDNN, CUTLASS, Thrust, CUB, FlashAttention, ...) without approval.
   - **Precision:** FP32 only (`double` in the gradcheck build). No `--use_fast_math`, no TF32 or reduced precision, and no fast-math intrinsics (`__expf`, `__fdividef`, ...) without approval (D-B4).
   - **Determinism:** no floating-point atomics in our kernels. Every reduction has a fixed order (D-B3).
   - **Target:** NVIDIA Quadro M2000M (sm_50, 4 GB), driver 580.178.04, **nvcc 12.9** (D-B1). Do not move to a CUDA 13.x toolkit: it cannot target sm_50.
7. **Never fabricate results.** Do not invent loss values, timings, speedups, or test outcomes. Report only what you actually ran, with the command. If you could not run something, say so.
8. **Never commit or generate large data.** `data/` and `runs/` are gitignored. Golden test files must stay small (target under a few hundred KB each).

---

## 2. Workflow rules

- **One task per session/prompt** (a ROADMAP task ID such as `P3.2`). Do not start adjacent tasks, refactor unrelated code, or "improve" files outside the task's scope. If you spot a problem elsewhere, report it at the end instead of fixing it.
- **Before coding:** state in 3-6 lines which ROADMAP task and sections you are following, which files you will create or touch, and any ambiguity. If the spec is ambiguous or two sections conflict, **ask** instead of guessing. If the user is unavailable, choose the option most consistent with ROADMAP, implement it, and flag it clearly.
- **Before declaring done:** run `make test` (and `make debug` tests when you touched memory handling) and show the real output summary. A task is done only when its ROADMAP DoD criteria pass.
- **GPU verification (Phase B).** CUDA code is done only after it has been compiled with `nvcc` and run on an NVIDIA GPU, with the real output in the report: `nvidia-smi` (GPU name, driver), `nvcc --version`, and the test summary.
  - The cloud development container has **no GPU**. There, write the code and run every CPU-side check: the CPU build, `make test`, and a compile-only `nvcc` check if a toolkit is installed.
  - Then state plainly **"NOT RUN ON A GPU"**, with the exact commands for the user to run on the GPU machine, and leave the task open until that output is back.
- **Measurement procedure.** These rules come from the stale-build correction in `docs/DECISIONS.md`.
  - After a mutation check (plant a bug, confirm the test fails), restore the file and `touch` it, so `make` recompiles it.
  - Rebuild the configuration from scratch (`make clean`, or `rm -rf build/<tag>`) before recording any number in DECISIONS, RUNS, a CSV or a README.
  - Every recorded number names its machine: CPU, plus GPU and driver for CUDA.
- **Tests first for numerics.** For any op or model change, write or update the test (FD check, golden compare, or hand-computed case) in the same change as the implementation.
- **Do not edit golden files** (`tests/golden/*`) or contract headers (`ops.h`, `backend.h`, `comm.h`, `layout.h`, checkpoint/log formats) to make a test pass. If a contract must change, stop, explain the CUDA/MPI impact, propose the change, and wait for approval. Approved changes get an entry in `docs/DECISIONS.md` (date, change, reason, impact on other builds).
- **Keep diffs small and reviewable.** Prefer new files over giant rewrites. Do not reformat files you did not need to touch.
- **Keep `README.md` current.** Any task that adds or changes something a user runs or configures (an app, a command-line option, a `make` target or variable, a script, a config key or preset, a required tool or dependency, or a user-visible output file) updates `README.md` in the same change. The README describes only what exists and has been run: every command in it must have been executed successfully in this task or an earlier one. Update the **Status** line when a phase milestone lands. Keep it concise; details belong in file header comments, `ROADMAP.md`, or `docs/`.
- **`docs/STATUS.md` is updated only on request.** Do not touch it during normal tasks. When asked to update it ("update STATUS.md with the latest additions"):
  1. Read its **Last updated** line (date, last ROADMAP task, commit) and review everything added since then (`git log <commit>..HEAD`, `git diff --stat <commit>..HEAD`, `docs/DECISIONS.md`).
  2. Keep the existing structure (build/docs, core infrastructure, data pipeline, model, tests, offline tools, next). Add new files and features to the right section, one line each: what the file or feature does, not how. Remove or reword anything that is no longer true.
  3. Only state results that were actually measured and are recorded elsewhere (test output, `DECISIONS.md`, committed CSVs); never estimate or round up.
  4. Update the summary paragraph, the test count, the **Next** section, and the **Last updated** line (date, last completed task, current commit).
  5. Commit it on its own (message: "Update STATUS.md through <task>"), separate from any task's code.
- **Final report format** (end of every task, short): what was implemented (files), test commands run and results, README changes (or "none needed"), deviations or decisions, open issues, suggested next task. No long recaps.
  - **CUDA tasks also report:**
    - the GPU and CUDA version used;
    - the worst rel-L2 per op touched;
    - sanitizer status;
    - for any timing, whether attention ran full-tile (cuBLAS) or causal-tile;
    - "NOT RUN ON A GPU" where that applies.

---

## 3. Build and test commands

```
make                      # release serial build -> bin/train_serial, bin/generate, ...
make test                 # all unit/integration tests (must pass before any task is "done")
make debug                # -O0 -g -fsanitize=address,undefined; then run tests under it
make test REAL=double     # double-precision build used for gradient checks
make dist                 # submission tarball; fails if any .py/.ipynb/requirements*.txt is present
make clean
make MODE=profile         # dev-only gprof build (scripts/profile.sh)
make COMM=mpi             # Phase C; both BACKEND=cuda COMM=mpi: Phase D (stubs until then)
```

Phase B (CUDA). The build plumbing landed in B0.2 (compile-checked; GPU verification pending). Every op still aborts with `not implemented: <op>` until its B1–B4 task, and the CUDA tests pass only after B0.3 (backend-agnostic tests) plus the kernels:

```
make BACKEND=cuda                    # CUDA build -> bin/cuda-float/train_cuda, bin/cuda-float/<app>
make BACKEND=cuda test               # the whole suite (+ test_equiv_ops from B0.4) on the GPU
make BACKEND=cuda REAL=double test   # double kernels -> bin/cuda-double/; runs the gradcheck
make BACKEND=cuda MODE=debug         # nvcc -O0 -g -G (no -lineinfo: -G overrides it), device sync after every launch; host code without ASan -> bin/cuda-debug-float/
make BACKEND=cuda sanitize           # every test binary under compute-sanitizer --tool memcheck (racecheck/synccheck on equivalence: B0.4/B5.3)
make equiv                           # serial vs CUDA, ROADMAP §10 cross-version protocol (B0.5; not implemented yet)
```

- CUDA apps never share `bin/` with the serial apps: `bin/cuda-<real>/` (release), `bin/cuda-<mode>-<real>/` otherwise.
- `MODE=profile` is serial-only (gprof); CUDA profiling uses nsys/ncu (`docs/PROFILE_CUDA.md`).
- The target stamp includes the nvcc version, `CUDA_ARCH` and the nvcc flags, so a toolkit or target change rebuilds everything.

Variables: `CUDA_ARCH` (default `sm_50`), `NVCC` (nvcc 12.9; a missing one stops make with one message), `COMPUTE_SANITIZER` (default `compute-sanitizer`), `MPICXX`.

Release flags: `-O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic`. CUDA compiles `src/*/*_cuda.cu` with `nvcc -std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo -Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra`; host `.cpp` files keep `$(CXX)` and the flags above; the link (by `$(CXX)`) adds `-lcublas -lcudart` with an rpath to the toolkit's `lib64`. Code must compile **warning-free**. If a command above is not yet implemented, say so; do not invent a substitute silently.

---

## 4. Architecture invariants

These exist so CUDA and MPI versions can be added without rewriting the model. See ROADMAP §4 and §8 for full detail.

1. **Layering:** `apps/` → `train/`, `model/` → `ops/`, `backend/`, `comm/`, `optim/`; `data/` is host-only. `model/` and `train/` never contain tensor math loops; they call `ops::` functions.
2. **`ops::` functions** are free functions taking raw `real_t*` pointers plus dimensions. No STL types in signatures, no allocation, no globals, no exceptions, no host-only assumptions (pointers may be device pointers later).
3. **All tensor memory goes through `backend.h`** (`dev_malloc`, `copy_h2d`, `dev_memset`, ...). Never call `malloc`, `new`, `std::vector`, or `memcpy` on parameter/gradient/activation tensors in `model/`, `train/`, or `optim/`.
4. **All collectives go through `comm.h`.** The serial implementation is no-ops (rank 0, size 1). Never `#include <mpi.h>` outside `src/comm/comm_mpi.cpp`.
5. **Flat buffers:** one contiguous `params`, `grads`, `m`, `v`, and one activation arena, laid out by `layout.h` (64-byte-aligned tensor offsets). **No allocation inside the training loop.**
6. **Gradient convention:** `*_forward` overwrites outputs; `*_backward` **accumulates (`+=`)**; the training loop zeroes `grads` once per step.
7. **Step order is fixed:** `zero_grads → forward → backward → comm_allreduce_avg(grads) → clip → AdamW`.
8. **Host-only always:** tokenization, data loading, RNG, weight initialization, logging, checkpoint I/O, sampling decisions.
9. **Shapes and layouts** (row-major; linear weight `[OC, C]`; fused `qkv [B,T,3C]`; `att [B,NH,T,T]`) are as defined in ROADMAP §6.2. Do not change them.
10. **Compile-time backend selection** via Makefile variables (`BACKEND`, `COMM`, `REAL`). No virtual dispatch or function-pointer tables in hot paths. Headers shared with `.cu` files must stay nvcc-compatible C++17 (no exotic features, no `std::filesystem` or iostream in `ops`/`model` headers).
11. **Determinism:** own RNG (`xoshiro256**`, Box-Muller). Never use `std::mt19937`, `std::normal_distribution`, `rand()`, or time-based seeds. Avoid unordered-container iteration where it affects numeric results. Same seed ⇒ bit-identical serial runs, and bit-identical CUDA runs on the same GPU (D-B3).
12. **No allocation and no host synchronisation inside ops**, in every backend.
    - The only device-to-host copy allowed inside ops is the scalar result of `sum_reduce`/`sumsq_reduce`.
    - The cuBLAS handle and the persistent reduction scratch are created once by `backend_cuda` in `dev_init`. Ops reach them only through the internal header `src/backend/cuda_internal.h` and never create handles or buffers.
13. **Device setup:** every app calls `dev_init(local_rank)` (0 until Phase D) before any other `dev_*` call. `dev_peak_bytes()` is the memory figure for `bench` (D-B8).
14. **Kernel files:** one `.cu` file per op family, mirroring the serial `*_cpu.cpp`: `src/ops/{embedding,layernorm,elementwise,matmul,attention,softmax_xent,optim}_cuda.cu` (D-B11).
    - Shared device helpers go in `src/ops/cuda_common.cuh`, and cuBLAS wrappers in `src/ops/blas_cuda.cuh`.
    - Kernel-internal helpers live in `.cuh` files, never in `ops.h`.
15. **One test suite for every build.** The whole suite passes in every build. No test is skipped or weakened in the CUDA build without a DECISIONS entry (D-B7).

---

## 5. Code style

- C++17, `snake_case` for functions/variables/files, `PascalCase` for types, `UPPER_CASE` for macros and constants. Namespace `tslm` for everything, `tslm::ops` for kernels.
- Use `real_t` for tensor elements (`float`, or `double` in the gradient-check build); never hard-code `float` in `ops`/`model`. Use `double` for loss/norm accumulations. Use `size_t` for element counts that can exceed 2³¹ or whose product with another dimension can; cast **before** multiplying (`size_t(N) * V`).
- Use `int32_t` for token ids, `uint16_t` for on-disk tokens, `uint64_t` for counts and seeds.
- `const` correctness on every input pointer; `__restrict__`-style qualifiers only where the aliasing guarantee is true.
- Error handling: `CHECK(cond, "message with values")` aborts with file/line for programmer and I/O errors. No silent fallbacks. Validate every file open/read/write and config key.
- No `using namespace std;` in headers. No macros where `constexpr` or inline functions work.
- Comments: each op file starts with a header comment stating shapes, layouts, accumulate/overwrite semantics, and the math. Explain *why* for non-obvious choices (blocking sizes, numerical tricks). Do not narrate trivial code.
- One responsibility per file; keep `apps/*.cpp` thin (parse args, call library).
- Logging goes through `logging.h`; never `printf` debug noise into committed code paths.
- **CUDA code:**
  - **Error checks:**
    - every CUDA runtime call goes through `CUDA_CHECK(...)`;
    - every cuBLAS call goes through `CUBLAS_CHECK(...)`;
    - every kernel launch is followed by `CUDA_CHECK(cudaGetLastError())`, plus a device sync in `MODE=debug`.
  - **Kernel structure:**
    - element-wise kernels use grid-stride loops;
    - block sizes are powers of two (prefer 256);
    - indices that multiply two dimensions are `size_t`, cast before multiplying.
  - **Header comments:** each kernel's comment states its launch configuration, tiling and reduction order. cuBLAS calls state their row-major → column-major mapping (`ROADMAP_CUDA.md` §5.4).
  - **Optimizations** are documented in `docs/PROFILE_CUDA.md` with their measured effect.

---

## 6. Numerical and testing requirements

- Softmax/cross-entropy subtract the row max. LayerNorm eps 1e-5, biased variance. GELU is the tanh approximation; backward is its exact derivative. Cross-entropy backward is `(p − onehot)/N`.
- Required test layers (ROADMAP §10): per-op unit tests; full-model **finite-difference gradient check in the `REAL=double` build**; **golden files** from PyTorch float64 (generated offline, compared with no Python at test time); initial-loss ≈ ln(10000); overfit-one-batch; determinism; checkpoint resume bit-exactness; causality; exact parameter counts (`base` tied = 5,785,088; untied = 8,345,088; `small` tied = 1,709,568; untied = 2,989,568).
- Tolerances: float32 vs float64 golden: loss rel err < 1e-5, per-tensor gradient rel-L2 < 1e-4. Double gradcheck: max rel err < 1e-6. Do not loosen a tolerance to get a pass; investigate the bug. If a loosening is genuinely justified, explain it and ask.
- Tests must be fast: all of `make test` should finish within a few minutes; the smoke-training test under 1 minute. Tests must not require the full dataset or network access (use tiny committed fixtures and generated synthetic data).
- When a test fails, report the failing assertion and values. Do not delete or weaken the test.
- **CUDA equivalence:**
  - Every new kernel comes with its rows in `tests/test_equiv_ops.cpp` in the same change: CUDA vs the serial reference kernels (`ops_ref`), over the shape sweep.
  - Tests of accumulating ops seed their outputs with non-zero values.
  - Per-op tolerances are those in `docs/HANDOFF.md` once B0.4 has confirmed them. Model-level equivalence follows the ROADMAP §10 protocol (`make equiv`).

---

## 7. Performance and honesty rules

- The serial baseline is the denominator of every speedup in the final report, so it must be **reasonable but honest**: cache-blocked matmul with auto-vectorizable inner loops; no naive strided triple loop; no heroic hand-tuning that CUDA comparisons would not match. Document every optimization applied in the source and in `docs/PROFILE.md`.
- Serial attention computes only the causal lower triangle. Note this in comments and benchmark docs.
- Benchmarks follow ROADMAP §12 exactly: fixed workload, 3 warmup steps discarded, median reported, ≥3 runs, machine spec and flags recorded, results written to a CSV by the `bench` app. Never quote a number that is not in a committed CSV.
- Do not run long jobs (more than ~5 minutes) without telling the user what you are about to run and why. Never start full-dataset serial training; it is infeasible (ROADMAP §13).
- **CUDA benchmarks:**
  - Bench CSVs record `version=cuda` and the GPU (`dev_name()`, driver).
  - Speedups are computed only from committed CSV rows, compare the same machine (serial re-benchmarked on the GPU machine's CPU, B6.4), and name both the CPU and the GPU.
  - They state whether attention ran full-tile (cuBLAS) or causal-tile; serial is triangular.
- **Profiling:** `nsys`/`ncu` are dev-time only, like gprof. Their results go in `docs/PROFILE_CUDA.md`.

---

## 8. Data and tokenizer rules

- Dataset: original `roneneldan/TinyStories` train and validation text files. Validation is always the official validation file; only the train set may be truncated, via `prepare_data --max_stories/--max_tokens`.
- Tokenizer: GPT-Neo byte-level BPE reimplemented in C++, reading `vocab.json` and `merges.txt` as data. Verified against offline-generated golden cases with exact id equality. Do not substitute a simplified or "approximately BPE" tokenizer.
- Vocabulary: size 10,000. id 0 = `<unk>`, id 1 = `<|endoftext|>`, ids 2..9999 = most frequent training tokens (ties by lower GPT id). Always report the non-`<unk>` coverage.
- Token files: header + `uint16`, formats exactly as ROADMAP §7.3. Loader semantics and sharding exactly as §7.4 and §4.5.

---

## 9. Config and file conventions

- Configs are `key=value` text files in `configs/` (`smoke.cfg`, `small.cfg`, `base.cfg`) with `--key value` CLI overrides. Unknown keys are errors. Every run writes the resolved config into `runs/<name>/manifest.txt`.
- `log.csv` columns and checkpoint format are **frozen** (ROADMAP §9). Add fields only through the contract-change process in §2.
- Checkpoints must remain loadable by all four builds: params/m/v are the flat host layout, little-endian.
- Do not hard-code absolute paths, usernames, hostnames, or cluster-specific settings in code. Cluster/Slurm specifics live only in `scripts/slurm/`.

---

## 10. Things you must not do

- Do not add Python, notebooks, or Python-dependent build/test steps outside `dev_only/`.
- Do not add dependencies, vendored code, or downloaded binaries without approval and a `THIRD_PARTY.md` entry.
- Do not change contracts (§4 invariants, `ops.h` signatures, layouts, file formats) without the approval process.
- Do not add dropout, KV cache, mixed precision, threading, or other features not in the current phase's tasks.
- Do not alter golden files, tolerances, or tests to force a pass.
- Do not claim code "should work" without compiling and running it; do not claim tests pass without running them.
- Do not claim CUDA code works, or quote kernel timings or speedups, without real output from a GPU run.
- Do not use floating-point atomics or non-deterministic reductions in kernels, or a GPU library other than cuBLAS, without approval.
- Do not leave `TODO`/stub code behind a passing test. Unimplemented paths must fail loudly (`CHECK(false, "not implemented: ...")` or `#error`).
- Do not use `std::regex` for tokenization, `std::rand`, or time-based seeding.
- Do not silently catch and ignore errors, NaNs, or out-of-range values.

---

## 11. Definition of Done checklist (every task)

- [ ] Implements exactly the ROADMAP task's scope and DoD
- [ ] Follows layering, backend/comm shims, accumulation convention, and layout contracts
- [ ] Tests added or updated; `make test` and relevant `make debug`/`REAL=double` runs pass (real output shown)
- [ ] CUDA tasks: `make BACKEND=cuda test` (and the relevant `REAL=double` / `sanitize` runs) pass ON THE GPU, with the output and the GPU named; otherwise reported as "NOT RUN ON A GPU" and left open
- [ ] Warning-free build; no sanitizer errors
- [ ] No Python, new dependency, or contract change introduced without approval
- [ ] File header comments describe shapes, semantics, and math for any new op
- [ ] `THIRD_PARTY.md` / `docs/DECISIONS.md` updated if applicable
- [ ] `README.md` updated for any user-facing change (or confirmed unaffected), with every new or changed command actually run
- [ ] Final report written in the §2 format

---

## 12. Glossary (shape notation)

`B` batch size, `T` sequence length (runtime ≤ max), `C` embedding dim, `NH` heads, `hs = C/NH` head dim, `V` vocab size (10,000), `L` layers, `N = B·T` rows. `OC` output channels of a linear layer. `gB = micro_batch × world_size` global batch. Presets: `smoke` (1L, 32d, T=32), `small` (2L, 128d, 4H, T=256), `base` (4L, 256d, 8H, T=256). "Tied" = output head shares `wte`.

## 13. Phases

- **Phase A (serial):** complete except the P6.3 manual story evaluations and the P5.3 review (D-B9). `ROADMAP.md`.
- **Phase B (CUDA, single GPU), current:** `ROADMAP_CUDA.md`.
  - The CUDA implementation of the identical `ops.h` signatures, one `.cu` per family, with cuBLAS for the matrix products.
  - The device backend shim (`dev_init`, `dev_peak_bytes`).
  - Batched-GEMM attention.
  - The same tests, plus the equivalence tests and the §10 protocol.
- **Phase C (serial + MPI):** `comm_mpi.cpp` with `MPI_Bcast` of the flat params after init and one `MPI_Allreduce` of the flat grads per step; rank-sharded loader; Slurm scripts.
- **Phase D (CUDA + MPI):** one rank per GPU. `dev_init(local_rank)`; device buffers passed to MPI (CUDA-aware MPI if available, otherwise staged through one pre-allocated host buffer); same step order.

Keep Phase B code free of anything that would block Phases C and D (`ROADMAP_CUDA.md` §9), but do not implement them early.
