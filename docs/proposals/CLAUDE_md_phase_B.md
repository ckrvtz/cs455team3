# PROPOSAL (pending team decision): CLAUDE.md changes for Phase B

**Status:** not applied. CLAUDE.md is unchanged until the team has met and
decided the open points in `ROADMAP_CUDA.md` §3 (D-B1 GPU target, D-B2
bespoke matmul vs cuBLAS, D-B8 `backend.h` additions, D-B9 gate waiver, D-B11
kernel file layout).

**After the meeting:**
1. Apply the accepted items to CLAUDE.md.
2. Record the decisions in `docs/DECISIONS.md`.
3. Delete this file.

Assistants must not treat anything here as a rule until it is in CLAUDE.md.

## Proposed changes

1. **Header and §13 (phase).**
   - Change: "Current phase: **Phase B, CUDA single GPU**
     (`ROADMAP_CUDA.md`)".
   - Task prompts use ROADMAP_CUDA.md task IDs (`B2.3`); ROADMAP.md stays the
     base contract.

2. **§1 hard constraints: CUDA rules.**
   - Kernels are written for this project.
   - No `--use_fast_math`, TF32 or reduced precision.
   - No floating-point atomics in kernels without approval (D-B3,
     determinism).
   - No third-party GPU libraries (cuDNN, CUTLASS, Thrust, CUB, ...)
     without approval.
   - cuBLAS only as decided in D-B2.

3. **§2 workflow: GPU verification.**
   - CUDA code is "done" only after it has been compiled with `nvcc` and run
     on an NVIDIA GPU, with `nvidia-smi` (GPU name, driver) and the test
     output shown.
   - Without a GPU, the assistant writes the code, runs every CPU-side check,
     states "NOT RUN ON A GPU" with the exact commands for the user to run,
     and leaves the task open.

4. **§2 workflow: measurement procedure.** Adds the procedure from
   `docs/DECISIONS.md` (stale-build correction and its update).
   - After a mutation check, restore the file and `touch` it.
   - Do a clean rebuild of the configuration before recording any number.
   - Every recorded number names its machine (CPU; GPU and driver for CUDA).

5. **§3 build commands.**
   - Add: `make BACKEND=cuda`, `make BACKEND=cuda test`,
     `make BACKEND=cuda REAL=double test`, `make BACKEND=cuda MODE=debug`,
     `make BACKEND=cuda sanitize` (compute-sanitizer), and `make equiv`
     (serial vs CUDA, ROADMAP §10 protocol).
   - Add the variables `CUDA_ARCH`, `NVCC` and `MPICXX`.

6. **§4 architecture invariants.**
   - No allocation and no host synchronisation inside ops. The only
     exception is the scalar results of `sum_reduce`/`sumsq_reduce`.
   - Kernel-internal helpers live in `src/ops/*.cuh`, never in `ops.h`.
   - The whole test suite passes in every build; no CUDA-only skips without
     a DECISIONS entry.

7. **§5 code style (CUDA).**
   - Every CUDA API call goes through `CUDA_CHECK`.
   - Every kernel launch is followed by `CUDA_CHECK(cudaGetLastError())`,
     plus a device sync in `MODE=debug`.
   - Element-wise kernels use grid-stride loops.
   - Each kernel's header comment states its launch configuration, tiling
     and reduction order.
   - Optimizations are documented in `docs/PROFILE_CUDA.md`.

8. **§6 testing.**
   - Every new kernel comes with its rows in `tests/test_equiv_ops.cpp`
     (CUDA vs the serial reference kernels `ops_ref`) in the same change.
   - Tests of accumulating ops seed their outputs with non-zero values.
   - Per-op tolerances are those in `docs/HANDOFF.md` once B0.4 has
     confirmed them; they are not loosened without approval.

9. **§7 performance.**
   - Bench CSVs record `version=cuda` and the GPU.
   - Every speedup is computed from committed CSV rows and states both
     machines, and whether attention ran full-tile or triangular.
   - `nsys`/`ncu` profiling is dev-time only, like gprof.

10. **§10 things you must not do.**
    - Do not claim CUDA code works, or quote kernel timings or speedups,
      without real GPU output.
    - Do not use float atomics or non-deterministic reductions without
      approval.
