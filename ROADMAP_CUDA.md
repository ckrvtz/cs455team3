# ROADMAP_CUDA: single-GPU CUDA build (Phase B)

This is the technical plan for Phase B: a CUDA version of the serial trainer
that runs on one GPU, extensible to MPI+CUDA on the cluster (Phase D). It
builds on the finished serial baseline and on `ROADMAP.md`, whose contracts
(§2, §4, §6, §8, §9, §10, §12) still apply unless this document says
otherwise. Working rules are in `CLAUDE.md`. Per-kernel requirements are in
`docs/HANDOFF.md`, which this roadmap refines.

Sources: `project-proposal.pdf` (CUDA plan and timeline), `TinyStories.pdf`
(model scale), `ROADMAP.md`, `docs/HANDOFF.md`, `docs/PROFILE.md`,
`docs/DECISIONS.md`, and the committed bench CSVs.

**Team decisions of 2026-11-18 are incorporated** (§3; `docs/DECISIONS.md`,
"Phase B team decisions"):
- target GPU Quadro M2000M with nvcc 12.9 (D-B1);
- cuBLAS for the matrix products (D-B2);
- `dev_init`/`dev_peak_bytes` added to `backend.h` (D-B8);
- Phase B may start (D-B9);
- one `.cu` file per op family (D-B11).

---

## 0. How to use this document (humans and LLMs)

- **Task IDs:** every task has an ID (`B2.3`). Prompt with: "Implement task
  B2.3 from ROADMAP_CUDA.md, following CLAUDE.md." One task per prompt.
- **Status labels:**
  - **(DECISION):** a default chosen here; it can be overridden with a
    `docs/DECISIONS.md` entry.
  - **(TEAM DECISION):** decided by the team; changing it needs the team
    and a DECISIONS entry. §3 lists them.
- **Contracts:** `include/tslm/ops.h`, `backend.h`, `comm.h`, `layout.h`,
  the checkpoint and `log.csv` formats, and the bench CSV columns.
  - They are frozen.
  - The only contract addition is in §4.4 (`dev_init`, `dev_peak_bytes`,
    approved and already made). Any other contract change needs approval
    and a DECISIONS entry first.
- **Definition of Done (DoD):** a task's acceptance criteria pass ON A GPU,
  with the real output shown. A CUDA task whose code was not compiled AND
  run on an NVIDIA GPU is not done, however plausible it looks (§11).
- **Numbers:** every number quoted in this file is either measured (with its
  source) or labelled an estimate.

---

## 1. Goals and non-goals

### Goals

1. **A CUDA build** (`make BACKEND=cuda` → `bin/cuda-float/train_cuda`, plus
   every other app) that implements every `ops.h` function in CUDA, with the
   model, training, data, checkpoint and app code unchanged:
   - the matrix products use cuBLAS (D-B2): `matmul_*` and the batched
     products inside `attention_*`;
   - every other kernel is written for this project.
2. **Equivalence with serial** under the ROADMAP §10 cross-version protocol,
   and the whole existing test suite (30 test programs, 155 test cases)
   passing in the CUDA build, including:
   - golden PyTorch comparisons;
   - the double-precision gradcheck;
   - determinism;
   - resume bit-exactness;
   - causality, overfit and smoke training.
3. **Honest performance results:**
   - speedup over serial on the identical workloads W-A, W-C and W-B
     (§12);
   - throughput scaling;
   - the proposal's "how much more of the dataset in the same time"
     comparison;
   - a CUDA training run on far more data than serial can handle, scored
     with the P6 prompt evaluation.
4. **A codebase ready for Phase D** (MPI+CUDA, one rank per GPU) with no
   rewrite: device buffers go straight to `comm.h`, rank-to-GPU mapping
   exists, and the step order is unchanged.

### Non-goals (Phase B)

- **Precision:** no mixed precision, FP16/BF16, or TF32; FP32 only
  (`real_t = float`, plus `double` for gradchecks).
- **Fusion and overlap:**
  - no flash-attention-style fused attention: `att` is materialized, as the
    `ops.h` contract requires (it is saved for backward);
  - no multi-stream overlap of compute and copies;
  - no CUDA graphs.
- **Generation:** no KV cache; generation keeps re-running the window.
- **Scope:** no multiple GPUs in one process (that is Phase D, via MPI).
- **Dependencies:** no third-party GPU libraries other than cuBLAS (D-B2,
  matrix products only): no cuDNN, CUTLASS, Thrust, CUB or FlashAttention.
  Adding any library needs team approval and a THIRD_PARTY.md entry.
- **No changes to the model, its hyperparameters, or the data pipeline.**

---

## 2. Starting point (what Phase A delivers)

All of the following is measured and recorded.

| Fact | Value | Source |
|---|---|---|
| Serial W-A step (small, B=32, T=256) | 3,610 ms median (3 runs: 3,567–3,640 ms), 2.25k–2.30k tok/s | `bench/serial_W-A.csv` |
| Serial W-C step (smoke) | ≈ 20.1 ms median | `bench/serial_W-C.csv` |
| Serial phase split (W-A) | backward 60.9%, forward 38.9%, optimizer 0.2%, data < 0.01% | `docs/PROFILE.md` |
| Serial op split (W-A, gprof) | matmul 81.4% (backward 56.9%, forward 24.5%), softmax-xent forward 6.1%, attention 6.6%, GELU 1.6%, LayerNorm 1.4% | `docs/PROFILE.md` |
| Device memory, `small` B=32 T=256 (tied) | 26.1 MB params+grads+m+v, 1,241.8 MB activation arena, 1,267.9 MB total | `docs/HANDOFF.md` |
| Device memory, `base` B=32 T=256 (tied) | 88.3 MB + 1,962.1 MB = 2,050.4 MB | `docs/HANDOFF.md` |
| Largest buffers | `logits`, `probs`, `d_logits`: 312.5 MB each (V = 10,000) | `docs/HANDOFF.md` |
| Reference machine for serial | Xeon @ 2.30GHz AVX-512, 1 core used | README |

Already in place for CUDA:

- **Shared code is backend-agnostic:** `model/`, `train/` and `optim/` touch
  tensors only through `ops::`, `dev_memset` and `copy_h2d/d2h/d2d` (audited
  for this roadmap). Host-side values (loss, grad norm, logits for sampling,
  checkpoint streaming) come through `copy_d2h` or a returned scalar.
- **Timing:** `PhaseTimer` calls `dev_sync()` at every phase boundary.
  `bench` times the same `train_step()` as training. `bench.h` already says
  CUDA reports device memory.
- **Build:** the Makefile has `BACKEND=cuda`, `NVCC ?= nvcc` and a `.cu`
  compile rule.
- **Stubs and guards:**
  - `src/ops/ops_cuda.cu` and `src/comm/comm_mpi.cpp` are `#error` stubs
    (P7.5);
  - `include/tslm/build_config.h` `#error`s on `BACKEND=cuda` until Phase B
    removes that guard;
  - `check.h` has a `CUDA_CHECK` placeholder that `#error`s in CUDA builds.
- **Portable checkpoints:** flat host layout; `real_bytes` in the header;
  streamed through `copy_d2h/h2d`.

Not yet compatible with CUDA (B0 fixes these):

- **Twelve test files touch tensor memory on the host.** They pass
  `std::vector` data straight to `ops::` or dereference `m->params`,
  `m->grads`, `m->m`, `m->v` or `model_logits(m)`:
  - `test_attention`, `test_checkpoint`, `test_eval`, `test_generate`;
  - `test_golden`, `test_model`, `test_model_backward`, `test_optim`;
  - `test_signoff`, `test_softmax_xent_backward`, `test_tokenizer`,
    `test_train`.

  The op unit tests (`test_ops_forward`, `test_matmul*`, ...) also call
  ops on host arrays.
- **The development container has no GPU and no CUDA toolkit**
  (`nvidia-smi`, `nvcc` absent). See §11.

Open Phase A items: the manual story evaluations (the P6.3 two-rater pilot
on `docs/runs/small_p52/samples_2400.txt`), then the P5.3 Serial Complete
review. P5.2 and P6.1 are done. Phase B may proceed (D-B9).

### 2.1 Target GPU and toolkit (D-B1)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA Quadro M2000M (Maxwell GM107), compute capability 5.0 | team; NVIDIA specifications |
| Driver | 580.178.04 (`nvidia-smi` reports CUDA 13.0, the newest runtime it supports) | team |
| Toolkit / compiler | CUDA Toolkit with **nvcc 12.9**; build `-arch=sm_50` | team |
| Cores / SMs | 640 CUDA cores, 5 SMs | NVIDIA specifications |
| FP32 peak | ≈ 1.4 TFLOP/s | NVIDIA specifications (estimate from cores × boost clock × 2) |
| FP64 | 1/32 of the FP32 rate (≈ 44 GFLOP/s) | NVIDIA specifications |
| Memory | 4 GB GDDR5, ≈ 80 GB/s | NVIDIA specifications |
| Shared memory | 48 KB per block | compute capability 5.0 limits |
| Tensor cores | none (TF32 and FP16 tensor paths do not exist) | Maxwell architecture |

B0.1 confirms every specification row on the machine
(`cudaGetDeviceProperties`, `nvidia-smi`, `nvcc --version`) and records it.
Consequences:

1. **Toolkit pinned at 12.9.** CUDA 13.0 removed offline compilation for
   compute capability < 7.5 (Maxwell, Pascal, Volta), per NVIDIA's CUDA 13.0
   release notes. Do not upgrade the toolkit to 13.x for this GPU. nvcc 12.9
   warns that sm_50 is deprecated; the Makefile passes
   `-Wno-deprecated-gpu-targets`.
2. **Memory:** `small` B=32 needs 1,268 MB and `base` B=32 needs 2,050 MB
   (HANDOFF), plus the CUDA context and cuBLAS workspace. That fits in
   4 GB.
   - If the GPU also drives a display, less memory is free, and a kernel
     run-time limit (display watchdog) may apply.
   - Check `nvidia-smi` for other processes, and for the "Kernel exec
     timeout" property, in B0.1.
3. **Double build:** the `REAL=double` build (gradcheck) runs at the slow
   FP64 rate. That is fine for the tiny gradcheck model; do not use the
   double build for anything large.
4. **Same-machine speedups:** the GPU sits in a laptop-class machine. If
   that machine is the teammate's i7-6820HQ that ran P5.2 (≈ 3.95 s/step for
   `small`, from that run's `log.csv`, not a benchmark), the serial
   benchmarks are re-run there, so every speedup compares the same machine
   (B6.4).

---

## 3. Decisions

### Settled here (DECISION)

| ID | Decision | Why |
|---|---|---|
| D-B3 | **Deterministic kernels:** no floating-point `atomicAdd` in any of our kernels; every reduction has a fixed order (fixed tree or fixed sequential order per output). cuBLAS is used only with routines and settings that NVIDIA documents as bit-wise reproducible run to run (same toolkit, same GPU, single stream, no atomics-based routines). Same binary + same GPU + same seed ⇒ bit-identical runs. | The existing determinism and resume-bit-exactness tests must pass unchanged in the CUDA build. Atomic float adds make the last bits run-dependent. |
| D-B4 | **FP32, no TF32, no fast math:** `nvcc` never gets `--use_fast_math`; no `__expf`/`__fdividef` style intrinsics unless approved; no tensor-core TF32 paths. | §10 tolerances assume FP32. CLAUDE.md forbids fast math for the serial build, and the same honesty applies here. |
| D-B5 | **Attention = the proposal's batched-GEMM design:** QKᵀ as a batched GEMM into `att`, a causal-masked row softmax kernel, then att·V as a batched GEMM, with the GEMMs on cuBLAS (D-B2). Backward mirrors it. cuBLAS computes full T×T tiles; a bespoke causal-tile GEMM is an optional optimization (B6.3). | Proposal ("batched matrix multiplication for QKᵀ and PV with a causal-masked softmax"), and `ops.h` saves `att`. |
| D-B6 | **One CUDA stream** (the default stream), synchronous semantics at phase boundaries via `dev_sync()`. | Simplicity and determinism; overlap is a non-goal. |
| D-B7 | **Tests run in every build:** the test suite is made backend-agnostic (B0.3), and the CUDA build adds equivalence tests against a reference copy of the serial kernels (B0.4). No test is skipped in the CUDA build without a DECISIONS entry. | ROADMAP §10: "later versions reuse the validation suite unchanged". |
| D-B10 | **Memory:** the existing flat buffers and single activation arena are allocated once with `cudaMalloc` through `dev_malloc`. No allocation in the training loop, including no `cudaMalloc` inside ops. Kernel launches use no dynamic allocation. cuBLAS gets its workspace once, at handle creation. | CLAUDE.md §4.5; the arena sizes (≤ 2.1 GB for `base`) fit the 4 GB M2000M (§2.1). |
| D-B12 | **Sanitizers:** `make BACKEND=cuda MODE=debug` builds with `-G -lineinfo`. A `make BACKEND=cuda sanitize` target runs the tests under `compute-sanitizer` (memcheck; racecheck and synccheck for kernels with shared memory). | The CUDA analogue of the serial ASan/UBSan build. |

### Decided by the team, 2026-11-18 (TEAM DECISION)

| ID | Decision | Consequences in this roadmap |
|---|---|---|
| D-B1 | **Target GPU:** NVIDIA Quadro M2000M (sm_50, 4 GB), driver 580.178.04, CUDA Toolkit **nvcc 12.9**. | §2.1; `CUDA_ARCH ?= sm_50`; toolkit pinned at 12.x; estimates in §7 sized for this GPU; B0.1 verifies the specifications. |
| D-B2 | **cuBLAS is used for the matrix products in the model,** with citation in `THIRD_PARTY.md` (added). | `matmul_forward/backward` call `cublasSgemm` (`cublasDgemm` in the double build); the attention ops call `cublasSgemmStridedBatched`. Everything else is bespoke. The cuBLAS handle is owned by `backend_cuda` (§4.3). The bespoke GEMM becomes an optional comparison exercise (B2.3). |
| D-B8 | **`backend.h` gains `dev_init(int local_rank)` and `dev_peak_bytes()`.** | Done 2026-11-18: header, CPU implementation, `test_runtime` test, and `dev_init(0)` in every app (§4.4). The CUDA implementation is part of B0.2. |
| D-B9 | **Phase B may begin.** Phase A waits only on the manual story evaluations (P6.3 pilot), then the P5.3 review. | Serial Complete sign-off is still required before B5.4. |
| D-B11 | **One `.cu` file per op family, mirroring the serial files:** `src/ops/{embedding,layernorm,elementwise,matmul,attention,softmax_xent,optim}_cuda.cu`, plus `src/ops/cuda_common.cuh` for shared device helpers. | The P7.5 stub `src/ops/ops_cuda.cu` is removed in B0.2. |

---

## 4. Architecture of the CUDA build

### 4.1 Files

```
include/tslm/ops.h               unchanged (contract)
include/tslm/backend.h           + dev_init, dev_peak_bytes (D-B8, added 2026-11-18)
include/tslm/check.h             CUDA_CHECK(call) implemented for BACKEND=cuda
src/backend/backend_cuda.cu      dev_malloc=cudaMalloc, dev_free, dev_memset=cudaMemset,
                                 copy_*=cudaMemcpy (synchronous), dev_sync=cudaDeviceSynchronize,
                                 dev_name=GPU name + compute capability, dev_init (cudaSetDevice,
                                 cuBLAS handle, reduction scratch), dev_peak_bytes
src/backend/cuda_internal.h      INTERNAL (not a contract): cublas_handle() and the persistent
                                 device scratch used by reductions; included only by *_cuda.cu
src/ops/cuda_common.cuh          device helpers: warp/block reductions (fixed order),
                                 grid-stride loop helpers, launch-check macro, ceil_div
src/ops/<family>_cuda.cu         one per op family (D-B11); host wrappers with the
                                 exact ops.h signatures launch the kernels
src/ops/blas_cuda.cuh            internal row-major wrappers over cublasSgemm /
                                 cublasSgemmStridedBatched (Dgemm when real_t is double)
tests/devbuf.h                   test helper: DevBuf<T> (device buffer from a host vector,
                                 to_host()), works in every build (B0.3)
tests/test_equiv_ops.cpp         CUDA ops vs reference serial kernels, shape sweep (B0.4)
src/ref/ (build rule only)       serial *_cpu.cpp kernels compiled again in namespace
                                 tslm::ops_ref for CUDA test binaries (B0.4)
docs/PROFILE_CUDA.md             every CUDA optimization, with why and its measured effect
bench/cuda_W-*.csv               CUDA benchmark results
```

### 4.2 Build system (Makefile)

- **Selecting the backend:** `BACKEND=cuda` selects `src/*/*_cuda.cu` and
  `src/backend/backend_cuda.cu`, compiled by `$(NVCC)`. Host `.cpp` files
  are compiled by `$(CXX)`. The link uses `$(NVCC)`, or `$(CXX)` plus
  `-lcudart -lcublas`.
- **`nvcc` flags:** `-std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo
  -Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra`, the same `-D`
  selection macros, and **no** `--use_fast_math`.
- **Toolkit:** nvcc 12.9 (D-B1; 13.x cannot target sm_50).
- **Configurations and stamps:**
  - `CUDA_ARCH ?= sm_50` (the reference GPU; `native` also works on the
    GPU machine). The target stamp (`build/<tag>/target.stamp`)
    must include `$(NVCC) --version` and `CUDA_ARCH`, so a different
    toolkit or GPU rebuilds everything.
  - Build directories: `build/release-cuda-serial-float/`, and so on.
  - Binaries: `bin/cuda-float/`; the trainer is `train_cuda` (the existing
    Makefile naming). Today the Makefile puts every release-float app in
    `bin/` whatever the backend, so a CUDA build would overwrite the serial
    `bench`, `generate`, and so on. B0.2 must give non-CPU backends their own
    `bin/<backend>-<real>/` directory; serial stays in `bin/`.
  - `make BACKEND=cuda REAL=double` must build: `real_t = double` kernels,
    used by the gradcheck.
  - `make BACKEND=cuda MODE=debug` uses `-G` (it overrides `-lineinfo`, so
    that flag is dropped: DECISIONS 2026-11-23); ASan is not mixed
    with CUDA.
  - `make BACKEND=cuda sanitize` runs each test binary under
    `compute-sanitizer --tool memcheck`.
- **Unchanged:**
  - `make` (CPU), `make test` and `make dist`. `make dist` already includes
    `.cu` files; its purity check is unchanged.
  - The P7.5 toolchain checks stay: a missing `nvcc` stops with one
    message.

### 4.3 Runtime conventions for every kernel and wrapper

1. **Host wrappers:** each `ops::` function is a host wrapper with the
   exact `ops.h` signature. It computes the launch configuration, launches
   one or more kernels on the default stream, and checks launch errors.
   Launch-check behaviour:
   - after every launch, `CUDA_CHECK(cudaGetLastError())`;
   - in `MODE=debug`, also `CUDA_CHECK(cudaDeviceSynchronize())`, so errors
     surface at the faulting op.
2. **No allocation and no host synchronisation inside ops**, and no
   `cudaMemcpy` except the scalar results of `sum_reduce`/`sumsq_reduce`,
   which must return a host `double`.
   - **Persistent resources:** the reduction scratch and the cuBLAS handle
     are created once by `backend_cuda` at `dev_init`/first use, never in
     the loop.
   - **How ops reach them:** through `src/backend/cuda_internal.h`. This is
     the only "global state" ops may touch, and ops never create handles or
     buffers.
   - **cuBLAS calls:**
     - every call is checked (`CUBLAS_CHECK`);
     - the handle uses the default stream and `CUBLAS_DEFAULT_MATH` (no
       TF32; the M2000M has no tensor cores anyway);
     - its pointer mode is host (scalar `alpha`/`beta` from host
       variables).
3. **Index types:** `size_t` for element offsets; `int` only where the
   value is provably < 2³¹. Every index that multiplies two dimensions
   casts first (`size_t(n) * C`), exactly as in serial.
4. **Overwrite vs accumulate:** `*_forward` overwrites and `*_backward`
   accumulates. A kernel that accumulates READS the existing value and adds
   its sum, never `=`. Tests seed outputs with non-zero values to catch
   this (B0.4).
5. **Block sizes:** powers of two between 128 and 1024. Prefer 256 unless a
   kernel documents a different choice (in a code comment and in
   `docs/PROFILE_CUDA.md`). Use grid-stride loops for element-wise kernels.
6. **Shared memory:** sized statically or with explicit dynamic sizes.
   Every kernel that uses it is run under `compute-sanitizer --tool
   racecheck` once (B5.3).
7. **Determinism (D-B3):** reductions use a fixed tree (warp shuffles, then
   a block tree, then a second fixed-order pass over block partials). No
   float atomics; integer atomics are allowed.
8. **Layouts are exactly `ops.h`/ROADMAP §6.2.** Do not transpose or
   repack tensors in the arena. Internal transposes happen only in
   registers or shared memory.

### 4.4 Contract additions (D-B8: approved and made 2026-11-18)

```cpp
// backend.h (additive)
// Select and initialise the device for this process. CUDA: cudaSetDevice(local_rank % device_count),
// create the cuBLAS handle and the reduction scratch. CPU: argument checks only.
// Optional with one device (the first dev_* call then uses device 0); if called, it must come
// before any other dev_* call, at most once, with local_rank >= 0.
void dev_init(int local_rank);
// High-water mark of the bytes requested through dev_malloc and not yet freed.
size_t dev_peak_bytes();
```

- **Implemented in the CPU backend** (the size is kept in a 64-byte block
  header, so alignment is unchanged) and tested by `test_runtime`
  `dev_init_and_peak_bytes`.
- **Callers:** `train`, `bench`, `eval_loss`, `generate`, `eval`,
  `gradcheck` and `dump_init` already call `dev_init(0)`. Phase D passes
  the node-local rank (§9).
- **`backend_cuda.cu` (B0.2)** implements the same semantics.
- **`bench`:** `peak_mem_MB` in CUDA builds = `dev_peak_bytes()/2²⁰`
  (device memory). The CSV columns are unchanged.
- **Manifests and bench:** `dev_name()` returns, e.g., "Quadro M2000M
  (sm_50), driver 580.178.04, CUDA runtime 12.9". The bench manifest records
  it, so every CUDA number names its GPU.

### 4.5 Host ↔ device data flow per step (unchanged from serial)

1. **data:** the loader fills host `x, y`, then `copy_h2d` (2 × B·T·4 B =
   64 KB for B=32, T=256). Pinned host memory is not needed at this size
   (§7).
2. **fwd, bwd, opt:** device only.
3. **Host scalars:** `loss` comes back through `sum_reduce`'s returned
   double, and the grad norm through `sumsq_reduce`.
4. **Validation:** device forwards; host scalar.
5. **Checkpoint:** streams `copy_d2h` in 64 KB chunks (already
   implemented). Generation copies one logits row (V floats) to the host per
   token (already implemented).

---

## 5. Kernel specifications

Each op must satisfy its paragraph in `docs/HANDOFF.md`: semantics,
overwrite/accumulate, special cases and the proposed tolerance. This section
adds the required design. "rel-L2" and the tolerances are as in HANDOFF; B0.4
confirms or revises them. N = B·T rows.

### 5.1 Element-wise family (`elementwise_cuda.cu`, `optim_cuda.cu`)

- **`residual_forward/backward`, `scale_inplace`:** grid-stride, one element
  per thread iteration, coalesced. Tolerance: bit-exact (the same IEEE
  operations as serial).
- **`gelu_forward/backward`:**
  - Use serial's stable form `x·σ(2u)`, with σ written exactly as in
    `src/ops/elementwise_cpu.cpp`, and the exact derivative.
  - Use `expf`/`tanhf` from CUDA's math library (not intrinsics, D-B4).
  - rel-L2 ≤ 1e-6.
- **`adamw_step`:**
  - One thread per element.
  - The host wrapper computes `inv_bc1 = 1/(1−β1^t)` and `inv_bc2` in
    double, casts them to `real_t`, and passes them as kernel arguments,
    exactly as serial does.
  - The decay term uses `p` before the update.
  - rel-L2 ≤ 1e-6 per tensor; the key-bias slice is checked by the §10
    bound.
- **`sum_reduce`, `sumsq_reduce`:**
  - Two-pass deterministic reduction. Pass 1: each block reduces a fixed,
    contiguous chunk into a partial in double (accumulate per thread in
    double, then a warp-shuffle tree in double). Pass 2: one block sums the
    partials in index order.
  - The result goes to the host through one 8-byte `cudaMemcpy` (the
    documented exception in §4.3).
  - Relative error ≤ 1e-12 vs serial.

### 5.2 Embedding (`embedding_cuda.cu`)

- **`embedding_forward`:** one thread per `(row n, channel c)`, with c
  fastest for coalescing: `out = wte[tok·C + c] + wpe[t·C + c]`. Bit-exact.
- **`embedding_backward`** (deterministic, no atomics):
  - **`dwte`, column-owner scheme:** launch C threads, thread c owning column
    c. It loops n = 0..N−1 in order:
    `dwte[tok[n]·C + c] += dout[n·C + c]`. Repeated tokens accumulate in
    the same order as serial, so the result is exact relative to serial.
    For C=256, N=8192 that is 2M adds over 256 threads; the profile shows it
    is negligible (verify in B6.1).
  - **`dwpe`:** one thread per `(t, c)`, summing over b in order.
  - **Tied model:** `dwte` already contains the head gradient. Accumulate,
    never overwrite.
  - **Tolerance:** rel-L2 ≤ 1e-6 (expected to be exact).

### 5.3 LayerNorm (`layernorm_cuda.cu`)

- **`layernorm_forward`:**
  - One warp (C ≤ 1024) or block per row.
  - Two-pass mean and biased variance, eps 1e-5, rstd. Write `mean[n]` and
    `rstd[n]`.
  - Accumulate the row sums in float with a fixed tree. Accumulating in
    double (as serial does) is optional; choose by the tolerance.
  - out, mean, rstd rel-L2 ≤ 1e-6.
- **`layernorm_backward`:**
  - **`dinp`:** per row, as HANDOFF gives it, using the saved mean and rstd.
  - **`dgamma`, `dbeta`:** column reductions over N, deterministic two-stage.
    Stage 1: grid (C-tiles × row-chunks), each block producing partials
    `[chunks, C]` into the op's own scratch (see the next bullet). Stage 2:
    sum the chunks in order per column, then add into `dgamma`/`dbeta`.
  - **Scratch:** the partials buffer must not be allocated in the op. Use a
    persistent scratch owned by `backend_cuda`, sized at `dev_init` for the
    largest need. Document the bound (`chunks × C` floats, e.g. 64 × 1024).
    This is the same mechanism as §5.1's reduction scratch; it is internal
    and not visible in `ops.h`.
  - rel-L2 ≤ 1e-5.

### 5.4 Matmul (`matmul_cuda.cu`, `blas_cuda.cuh`), the critical path, on cuBLAS (D-B2)

The serial profile puts 81% of time here.

- **Row-major on a column-major library.** A row-major matrix `X[r,c]` is,
  to cuBLAS, the column-major matrix `Xᵀ` with leading dimension `c`. So
  each product is computed as its transpose. With `cublasSgemm(handle,
  opA, opB, m, n, k, α, A, lda, B, ldb, β, C, ldc)`:

  | ops.h product | What cuBLAS computes (column-major) | Call |
  |---|---|---|
  | `out[N,OC] = inp[N,C]·W[OC,C]ᵀ` | `outᵀ[OC,N] = W·inpᵀ` | `opA=T, opB=N, m=OC, n=N, k=C, A=W lda=C, B=inp ldb=C, β=0, C=out ldc=OC` |
  | `dinp[N,C] += dout[N,OC]·W[OC,C]` | `dinpᵀ[C,N] += Wᵀ·doutᵀ` | `opA=N, opB=N, m=C, n=N, k=OC, A=W lda=C, B=dout ldb=OC, β=1, C=dinp ldc=C` |
  | `dW[OC,C] += dout[N,OC]ᵀ·inp[N,C]` | `dWᵀ[C,OC] += inpᵀ·dout` | `opA=N, opB=T, m=C, n=OC, k=N, A=inp lda=C, B=dout ldb=OC, β=1, C=dW ldc=C` |

  This mapping is the design intent; `test_equiv_ops` over the full shape
  sweep is what proves it, so an error here cannot survive B2.2.
- **Bias:**
  - Forward: GEMM with `β=0`, then a bespoke bias-broadcast kernel (skipped
    when `bias` is null).
  - `dbias[OC] += column sums of dout`: a bespoke deterministic column
    reduction as in §5.3, skipped when null.
- **Accumulate:** all three backward outputs accumulate (`β=1`). The tied
  head's `dW` is `wte`'s gradient, which later also receives the embedding
  gradient.
- **Precision:** `real_t=double` builds call `cublasDgemm` (the gradcheck
  build). The wrappers in `blas_cuda.cuh` pick S or D at compile time.
- **Edge shapes:** any `N, C, OC`, including N = 1 (generation), OC = V =
  10,000, and the tiny test shapes {1, 7, 16, 100}. cuBLAS handles these;
  the bias and dbias kernels bounds-check.
- **Tolerance:** rel-L2 ≤ 1e-5 per output.
- **Performance:** recorded in `docs/PROFILE_CUDA.md`.
  - Report cuBLAS GFLOP/s on the W-A head shape (N=8192, C=128,
    OC=10,000) against the M2000M's ≈ 1.4 TFLOP/s FP32 peak.
  - Optional (B2.3): a bespoke tiled SGEMM, measured against cuBLAS for the
    report. It is never used in the model unless the team re-decides D-B2.

### 5.5 Softmax + cross-entropy (`softmax_xent_cuda.cu`)

- **`softmax_xent_forward`:**
  - One block per row (V = 10,000).
  - Pass 1: row max (block reduce). Pass 2: Σ exp(l − max) (fixed-tree
    reduce, float or double).
  - Pass 3: `probs = exp(l − max)/sum`, and
    `losses[n] = log(sum) − (l[target] − max)`, the log-sum-exp form as in
    serial.
  - losses rel ≤ 1e-6 per element; probs rel-L2 ≤ 1e-6.
- **`softmax_xent_backward`:**
  - One thread per element: `dlogits += probs/N`, with the target entry
    computed as `(p_t − 1)/N`.
  - Accumulates; it feeds the tied head's dW.
  - rel-L2 ≤ 1e-6.

### 5.6 Attention (`attention_cuda.cu`), D-B5

`qkv [B,T,3C]`: Q, K, V for head h are strided views with leading dimension
3C and offsets `0, C, 2C` plus `h·hs`. `att [B,NH,T,T]`. `hs = C/NH`.

- **GEMMs on cuBLAS (D-B2):** for each batch row b, one
  `cublasSgemmStridedBatched` call over the NH heads.
  - Inside a row of `qkv`, head h starts `h·hs` elements after head 0, so
    the batch stride between heads is `hs`.
  - The leading dimension is 3C for Q, K, V and C for `out`/`dout`.
  - `att` and `scratch` use a batch stride of T·T.
  - The 1/√hs scale is the `α` of the QKᵀ (and dQ, dK) GEMMs.
  - The same row-major-as-transpose mapping as §5.4 applies; it is verified
    by `test_equiv_ops`.
  - B calls per op (B = 32) is acceptable; a pointer-array batched call over
    (b, h) is an optional optimization.
- **Forward:**
  1. Batched GEMM over (b, h): `S = Q·Kᵀ·(1/√hs)` into `att`.
  2. Row kernel: causal mask, max-subtracted softmax over t2 ≤ t, and
     **exact 0 written for t2 > t**.
  3. Batched GEMM: `out[b,t,h·hs:...] = att·V`, with the output's leading
     dimension C.
- **Backward**, accumulating into `dqkv`:
  1. `datt = dY·Vᵀ` into `scratch` (B·NH·T·T reals, given by the caller).
  2. `dV += attᵀ·dY`.
  3. Row kernel, in place in scratch:
     `dpre = att ⊙ (datt − rowsum(att ⊙ datt))`.
  4. `dQ += dpre·K/√hs`.
  5. `dK += dpreᵀ·Q/√hs`.
- **Causal efficiency:** cuBLAS computes the full T×T products, which the
  row kernels then mask: about 2× the serial attention arithmetic for these
  GEMMs. Benchmark notes must say "full-tile attention" (`docs/PROFILE.md`:
  serial is triangular). B6.3 optionally adds a bespoke causal-tile batched
  GEMM (tiles entirely above the diagonal skipped) and compares it with the
  cuBLAS version.
- **Exactness requirements:**
  - Positions < t are unaffected by token t: the causality test uses
    `memcmp`, so the masked entries must not leak through NaN, Inf or
    uninitialised scratch.
  - The upper triangle of `att` is exactly zero (tested).
  - The key-bias gradient is a structural zero, compared only by the
    HANDOFF bound.
- **Tolerance:** out and att rel-L2 ≤ 1e-5; dqkv rel-L2 ≤ 1e-5 on the Q and
  V slices.

---

## 6. Testing strategy

1. **Backend-agnostic suite (B0.3).** Every test that touches tensor memory
   uses `tests/devbuf.h`:

   ```cpp
   DevBuf<float> x(host_vector);   // dev_malloc + copy_h2d
   ops::gelu_forward(y.ptr(), x.ptr(), n);
   std::vector<float> out = y.to_host();   // copy_d2h
   ```

   - Model buffers are read with `copy_d2h` into host vectors, never
     dereferenced.
   - In the CPU build this is a no-op wrapper, so CPU results stay
     bit-identical (verified by the unchanged smoke, golden and checksum
     values).
   - After B0.3, `make test` must pass with identical recorded values in
     the CPU build.
2. **Reference kernels in the CUDA build (B0.4).** The serial
   `src/ops/*_cpu.cpp` files are compiled a second time with
   `-Dops=ops_ref` (a build rule only; no source or contract change) into
   `build/<tag>/ref/`, and linked only into CUDA test binaries. The
   resulting `tslm::ops_ref::*` are the serial kernels.
   - `tests/test_equiv_ops.cpp` runs every op, CUDA vs `ops_ref`, over a
     shape sweep:
     - smoke, small and base layer shapes;
     - N ∈ {1, 7, 64, 8192};
     - C ∈ {16, 32, 128, 256};
     - V ∈ {32, 10000};
     - T ∈ {1, 8, 32, 256}.
   - It uses random inputs from the project's `Rng` and seeds every
     accumulated output with non-zero values, checking `+=`.
   - It reports the worst rel-L2 per op.
   - In the CPU build the test compares ops with `ops_ref` (identical code),
     so it passes trivially and stays a no-op.
3. **The existing suite in the CUDA build** must pass with the same
   tolerances:
   - golden PyTorch float64 files: loss rel < 1e-5, grads rel-L2 < 1e-4,
     AdamW params rel-L2 < 1e-4 with the key-slice bound;
   - gradcheck in `make BACKEND=cuda REAL=double test` (double kernels,
     rel < 1e-6);
   - determinism (2 × 20 steps bit-identical, including the forked-child
     run), resume bit-exactness, causality (`memcmp`), overfit < 0.1, and
     smoke val < 6.0 in under 60 s.

   A failing test is a bug, not a reason to change the test (CLAUDE.md §6).
4. **Serial vs CUDA at preset scale (§10 protocol, B5.4).**
   `scripts/equiv.sh` (`make equiv`) builds both builds on the GPU machine,
   then:
   - runs a small new app, `bin/dump_step`, in each: same config, seed and
     batch; it writes the step-0 loss and every gradient tensor to a TSGD
     file under `runs/equiv/`;
   - compares them with `bin/compare_tensors`: loss rel < 1e-4, every
     tensor rel-L2 < 1e-3;
   - runs both builds for 100 steps of `smoke` and `small`, and checks that
     their final val losses agree within 1% (read from `log.csv`).

   No large reference files are committed. The script prints a table that
   is recorded in DECISIONS.
5. **Cross-build checkpoints (B5.4).** A serial checkpoint resumes in the
   CUDA build and vice versa: same file format; continue 10 steps; losses
   finite and within the §10 bounds of the uninterrupted run.
6. **Sanitizers (B5.3).** `make BACKEND=cuda sanitize` is clean for
   memcheck over the whole suite, and for racecheck and synccheck on
   `test_equiv_ops`.

---

## 7. Performance methodology and expectations

- **Benchmark rules:** exactly ROADMAP §12, with `version=cuda`:
  - `scripts/bench.sh` with `VERSION=cuda BIN=bin/cuda-float` (or the
    documented path);
  - 3 runs per workload, 3 warmup steps, prewarm, ±3% reproducibility;
  - the CSVs go in `bench/cuda_W-A.csv`, `bench/cuda_W-B.csv` and
    `bench/cuda_W-C.csv`, recording the GPU and driver;
  - W-B (`base`) is required for CUDA, since it is cheap on a GPU.
- **Speedup:** serial ÷ CUDA median step time on the identical workload.
  Also report throughput (tok/s).
  - Speedups are only meaningful on the same machine. The serial
    benchmarks are therefore re-run on the GPU machine's CPU (B6.4), and
    every speedup names both the CPU and the GPU.
  - Never quote a speedup that is not computed from committed CSV rows.
- **Fixed wall-clock budget** (the proposal's metric): train `small` for the
  same number of minutes with serial and CUDA (e.g. 30 min each; the exact
  budget is set in B6.5). Record tokens consumed and final val loss.
- **Estimates to plan with** (not results; sized for the M2000M, §2.1):
  - `small` is ≈ 9×10¹⁰ FLOP/step (ROADMAP §13). If cuBLAS and our kernels
    reach an effective 0.5–1.0 TFLOP/s (35–70% of the ≈ 1.4 TFLOP/s peak),
    that is ~90–180 ms/step: roughly 20–40× the serial step on the same
    laptop CPU (≈ 3.95 s/step in the P5.2 log).
  - One epoch of the full train set (463.8M tokens, 56,620 steps at 8,192
    tokens/step) is then ~1.5–3 h for `small`.
  - `base` (≈ 3×10¹¹ FLOP/step) is ~0.3–0.6 s/step, ~5–10 h per epoch.
  - Serial would need ~57 h per `small` epoch. The memory bandwidth
    (≈ 80 GB/s) also limits the element-wise and softmax kernels over the
    312 MB logits buffers.
  - Measure everything in B6; these numbers only size the plan.
- **Profiling:** `nsys profile` for the timeline and `ncu` for per-kernel
  metrics (dev-time only, like gprof). Record the per-kernel time share in
  `docs/PROFILE_CUDA.md` in the format of the serial profile. Expect matmul
  to still dominate. Softmax-xent over V = 10,000 and attention are the next
  candidates (serial: 6.1% and 6.6%).
- **Pinned memory and overlap** are not needed: per-step H2D is 64 KB.
  Revisit only if `data_ms` exceeds 2% of the CUDA step.

---

## 8. Task breakdown

Owners follow the proposal (Vincent: CUDA kernels; Conor: MPI, then MPI+CUDA
integration). The weeks follow its timeline:
- W9: CUDA matmul and element-wise kernels;
- W10: attention, reductions and loss, then "CUDA correctly validates";
- W11: CUDA done and MPI+CUDA started;
- W12: MPI+CUDA done and large-dataset runs.

Every task's DoD includes "real output from a GPU machine shown"
(CLAUDE.md, §11).

### B0: setup and test infrastructure (W9, before any kernel)

| ID | Task | Owner | DoD |
|---|---|---|---|
| B0.1 | Verify the D-B1 machine: `nvidia-smi` (driver 580.178.04, memory free, other processes), `nvcc --version` (12.9), `cudaGetDeviceProperties` (sm_50, 5 SMs, 4 GB, shared memory per block, kernel exec timeout), the cuBLAS library version, and a trivial kernel plus one `cublasSgemm` built with the project flags (`-arch=sm_50 -Wno-deprecated-gpu-targets`) | Vincent | Output recorded in DECISIONS; README machine table gains the GPU row; THIRD_PARTY cuBLAS version filled in |
| B0.2 | CUDA build plumbing: remove the `BACKEND=cuda` guard in `build_config.h`; implement `CUDA_CHECK` and `CUBLAS_CHECK`; write `backend_cuda.cu` (`dev_init`: cudaSetDevice, cuBLAS handle, reduction scratch; `dev_peak_bytes`) and `cuda_internal.h`; replace the stub `ops_cuda.cu` with the per-family files (D-B11) whose wrappers initially `CHECK(false, "not implemented: <op>")`; Makefile `.cu` flags (`-arch=sm_50`, `-Wno-deprecated-gpu-targets`), `-lcublas -lcudart`, `bin/cuda-<real>/`, stamp including the nvcc version, `MODE=debug` and `sanitize` targets; `bench` reports `dev_peak_bytes()` in CUDA builds | Vincent | `make BACKEND=cuda` builds every app; `bin/cuda-float/...` runs and aborts with "not implemented: …" at the first op; CPU build unchanged (`make test` green with identical recorded values) |
| B0.3 | Backend-agnostic tests: add `tests/devbuf.h`; migrate every test that passes host memory to `ops::` or reads model buffers directly (§2 list and all op tests) | Vincent | CPU `make test`, `make debug`, `make test REAL=double` green with unchanged smoke val (5.7445), golden and checksum values; grep shows no direct host access to device-capable buffers in `tests/` |
| B0.4 | Reference kernels and op equivalence: the `ops_ref` dual-compile rule; `tests/test_equiv_ops.cpp` (shape sweep, `+=` seeding, worst rel-L2 table); confirm or revise HANDOFF's proposed per-op tolerances | Vincent | Test builds and runs in the CUDA build (failing on unimplemented ops); passes trivially in the CPU build; tolerance table in DECISIONS |
| B0.5 | `bin/dump_step`, `bin/compare_tensors`, `scripts/equiv.sh` (`make equiv`) for the §10 protocol at preset scale | Conor or Vincent | Runs CPU vs CPU (identical → rel 0) on the GPU machine; README entry |

### B1: element-wise, embedding, reductions, optimizer (W9)

| ID | Task | DoD |
|---|---|---|
| B1.1 | `residual_forward/backward`, `scale_inplace`, `gelu_forward/backward` (§5.1) | Their `test_equiv_ops` rows pass; the op unit tests pass in the CUDA build |
| B1.2 | `sum_reduce`, `sumsq_reduce` (two-pass, double, deterministic) and the reduction scratch in `backend_cuda` | rel ≤ 1e-12 vs `ops_ref`; repeated calls bit-identical |
| B1.3 | `embedding_forward/backward` (column-owner `dwte`, §5.2) | Repeated-token and tied-accumulation tests pass; bit-identical across runs |
| B1.4 | `adamw_step` (host-side double bias corrections) | `test_optim` hand-computed cases pass in the CUDA build; equivalence row passes |

### B2: matmul (W9)

| ID | Task | DoD |
|---|---|---|
| B2.1 | `blas_cuda.cuh`: row-major wrappers over `cublasSgemm`/`cublasSgemmStridedBatched` (Dgemm for double) using the backend's handle (§5.4) | A direct test of each wrapper against a host reference on small non-square shapes (including transposes and β=1 accumulation) passes on the GPU |
| B2.2 | `matmul_forward/backward` on the wrappers, plus the bespoke bias-broadcast and deterministic `dbias` kernels | `matmul_forward/backward` equivalence over the full shape sweep incl. N=1 and OC=10,000; `test_matmul*` pass in the CUDA build; cuBLAS GFLOP/s on the W-A head shape recorded in `docs/PROFILE_CUDA.md` |
| B2.3 | Optional: bespoke tiled SGEMM (shared-memory tiles, register blocking), measured against cuBLAS for the report; not used in the model | Equivalence vs `ops_ref`; GFLOP/s vs cuBLAS in `docs/PROFILE_CUDA.md` |

### B3: LayerNorm and softmax + cross-entropy (W10)

| ID | Task | DoD |
|---|---|---|
| B3.1 | `layernorm_forward/backward` with the deterministic two-stage `dgamma`/`dbeta` (§5.3) | Equivalence rows pass; `test_ops_backward` FD checks pass in the CUDA build |
| B3.2 | `softmax_xent_forward/backward` (§5.5) | Equivalence rows pass; `test_softmax_xent*` pass in the CUDA build; uniform-logits loss = ln V |

### B4: attention (W10)

| ID | Task | DoD |
|---|---|---|
| B4.1 | `attention_forward` via `cublasSgemmStridedBatched` (QKᵀ, att·V) + the causal masked softmax row kernel (§5.6) | Equivalence passes; `test_attention` incl. exact-zero upper triangle and op-level causality |
| B4.2 | `attention_backward` via strided-batched GEMMs + the softmax-backward row kernel in `scratch` | Equivalence passes (Q/V slices; key slice by bound); `test_attention_backward` passes |

### B5: integration and sign-off (W10–W11, "CUDA correctly validates")

| ID | Task | DoD |
|---|---|---|
| B5.1 | Full suite in the CUDA build: `make BACKEND=cuda test` (all 30 programs + `test_equiv_ops`) | Green, output shown, GPU named |
| B5.2 | Double-precision CUDA build: `make BACKEND=cuda REAL=double test`, including the gradcheck | Green |
| B5.3 | Sanitizers: `make BACKEND=cuda sanitize` (memcheck over the suite; racecheck and synccheck on `test_equiv_ops`) | Zero errors reported |
| B5.4 | §10 cross-version protocol (`make equiv`): step-0 loss and grads, 100-step val loss for `smoke` and `small`; cross-build checkpoint resume both ways. Requires D-B9's gate sign-off first | Table in DECISIONS with all bounds met: "CUDA matches serial" |
| B5.5 | Apps in the CUDA build: `train_cuda`, `generate`, `eval`, `bench`, `gradcheck`; README CUDA section with every command run on the GPU machine | README updated and verified |

### B6: performance and results (W11–W12)

| ID | Task | DoD |
|---|---|---|
| B6.1 | `nsys`/`ncu` profile of W-A in the CUDA build; per-kernel share table | `docs/PROFILE_CUDA.md` + committed text profile |
| B6.2 | Optimize the top bespoke kernels by profile (softmax-xent row kernel, LayerNorm, element-wise, embedding backward; cuBLAS call shapes and batching), each change measured and documented, numerics re-verified | Each optimization: before/after numbers in PROFILE_CUDA.md; suite + equiv green |
| B6.3 | Optional: bespoke causal-tile batched GEMM for attention (§5.6), compared with the cuBLAS full-tile version | Equivalence + causality pass; attention time for both versions recorded |
| B6.4 | CUDA benchmarks W-A, W-B, W-C on the M2000M (3 runs each, ±3%); serial benchmarks re-run on the same machine's CPU (`bench/serial_<machine>_W-*.csv` or a `--machine_id` that names it); speedup table computed from the CSVs | CSVs committed; speedup table in `docs/RUNS.md` or a results doc, naming CPU and GPU and stating "full-tile attention" |
| B6.5 | Fixed-budget comparison (serial vs CUDA, same minutes) and a CUDA full-dataset `small` run (and `base` if time allows); eval samples scored with `eval/RUBRIC.md` as milestone `cuda-<preset>-step<k>` | Results recorded in `docs/RUNS.md` from `log.csv`; scores in `eval/scores.csv` |

### B7: hand-off to Phase D (W11)

| ID | Task | DoD |
|---|---|---|
| B7.1 | Device-pointer readiness of `comm.h`: document and test (with a fake 1-rank comm) that `comm_allreduce_avg(model->grads, ...)` and `comm_broadcast(model->params, ...)` are called with device pointers; decide CUDA-aware MPI vs host staging (§9) | DECISIONS entry; HANDOFF updated with the Phase D notes |
| B7.2 | Done 2026-11-18 (D-B8): every app calls `dev_init(0)`; Phase C/D replace 0 with the node-local rank from `comm_init` | n/a (recorded in DECISIONS) |

---

## 9. Phase D readiness (MPI+CUDA, one rank per GPU)

Phase B must leave these true:

1. **Step order unchanged:** `zero → forward → backward →
   comm_allreduce_avg(grads) → clip → AdamW`. The clip norm
   (`sumsq_reduce`) runs after the allreduce, so all ranks agree.
2. **One collective per step:** `grads` is one contiguous device buffer, so
   Phase D does ONE allreduce per step, either directly on the device
   pointer (CUDA-aware OpenMPI) or by staging through ONE pre-allocated
   pinned host buffer of `num_params_padded` reals.
   - Staging means `copy_d2h`, then `MPI_Allreduce`, then `copy_h2d`, all
     inside `comm_mpi.cpp` (the only file that includes `<mpi.h>`).
   - Allocate the staging buffer once at `comm_init`/first use, never per
     step.
3. **Rank-to-GPU mapping:** `comm_init` determines the node-local rank
   (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`) and apps call
   `dev_init(local_rank)` before any `dev_*` call (§4.4).
4. **Initial weights:** `comm_broadcast(params)` from rank 0 after init,
   on a device pointer (the same choice as 2.).
5. **Data sharding:** the loader already shards by `(rank, world_size)`
   with global-batch semantics (ROADMAP §4.5). Validation splits fixed
   batches by rank, and `evaluate` already all-reduces the loss.
6. **Files:** checkpoints, logs and bench rows are written by rank 0 only
   (already so in `train_run` and `bench`).
7. **Bit-reproducibility:** within a given world size, MPI's reduction order
   for a fixed communicator and algorithm is usually deterministic. Do not
   promise bit-identity across world sizes; the §10 protocol uses
   tolerances.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No GPU in the LLM's development container; code "looks right" but never ran | §11 verification protocol: no task is Done without GPU output; compile-only checks are labelled as such |
| Row-major/column-major mistakes in the cuBLAS calls (silently wrong products) | The §5.4 mapping table; B2.1 wrapper tests on non-square shapes; full `test_equiv_ops` sweep |
| Toolkit upgrade breaks the build (CUDA 13.x cannot target sm_50) | nvcc pinned at 12.9 (D-B1); the target stamp records the nvcc version |
| 4 GB GPU shared with a display (less free memory; kernel watchdog) | Check in B0.1; `dev_peak_bytes`; reduce B for `base` if needed; keep kernels short |
| Laptop thermal throttling makes timings drift | §12 protocol (3 runs, ±3%); note power and thermal state in the bench manifest; rerun when out of tolerance |
| FP64 gradcheck slow on Maxwell (1/32 rate) | Gradcheck uses the tiny model only |
| Non-determinism from atomics or uninitialised scratch breaks determinism/resume tests | D-B3 (no float atomics); `compute-sanitizer --tool initcheck`; tests seed accumulated outputs |
| Silent overwrite instead of accumulate (tied `wte` loses the head gradient) | Equivalence tests seed outputs non-zero; golden tied test; HANDOFF notes |
| Out-of-bounds on edge shapes (N=1, OC=10,000) | Shape sweep in `test_equiv_ops`; memcheck |
| Tolerance creep ("loosen to pass") | Tolerances fixed in B0.4 via DECISIONS; CLAUDE.md forbids loosening without approval |
| Speedup misreported (different machines, warm vs cold, full vs triangular attention) | §12 rules, machine recorded per row, attention mode stated, numbers only from committed CSVs |
| Stale objects across toolkits or GPUs (as happened across CPUs in Phase A) | Target stamp includes the nvcc version and `CUDA_ARCH`; clean rebuild before recording numbers (DECISIONS procedure) |
| Phase D blocked by host-pointer assumptions | §2 audit already clean; B7.1 test with device pointers |
| Arena too large for the free GPU memory | `base` needs 2.05 GB at B=32 of the M2000M's 4 GB; reduce B (arena ∝ B·T, `att` ∝ B·T²); `dev_peak_bytes` reported |

---

## 11. LLM working protocol for Phase B

These are now part of CLAUDE.md (applied 2026-11-18: §1.6 CUDA rules, §2
GPU verification and measurement procedure, §3 commands, §4.12–4.15, §5–§7,
§10, §11). If this list and CLAUDE.md ever disagree, CLAUDE.md wins.

1. **Where code is verified.** CUDA code is Done only after it has been
   compiled with `nvcc` AND run on an NVIDIA GPU, with the commands and
   their real output shown in the task report, including `nvidia-smi` GPU
   name and driver.
   - If the session has no GPU (this cloud container has none), the
     assistant writes the code and runs every CPU-side check: CPU build,
     CPU tests, and a compile-only `nvcc` check if a toolkit is installed.
   - It then states plainly "NOT RUN ON A GPU", with the exact commands
     for the user to run, and treats the task as open until that output
     comes back.
2. **Never fabricate GPU results:** no kernel timings, speedups or "passes
   on GPU" without output.
3. **One kernel family per task,** with its equivalence tests in the same
   change. Do not optimize before the naive version passes equivalence.
4. **Every optimization is measured:** before/after time from `ncu`/`nsys`
   or `bench`, re-verified numerics, and an entry in
   `docs/PROFILE_CUDA.md`.
5. **Contracts:** `ops.h` signatures, layouts, formats and the step order
   are frozen. Kernel-internal helpers live in `.cuh` files under
   `src/ops/`.
6. **Report format** (per CLAUDE.md §2), plus: GPU and CUDA version used;
   worst rel-L2 per op touched; sanitizer status; and whether attention ran
   full or triangular (for any timing).
7. **Mutation checks** for new tests: plant a bug, confirm the test fails,
   restore and `touch` the file, then do a clean rebuild before recording
   any number (DECISIONS procedure).

---

## 12. "CUDA Complete" gate (before Phase D integration)

- [ ] Every `ops.h` function implemented in CUDA; no "not implemented" path
  left; `ops_cuda.cu` stub removed
- [ ] `make BACKEND=cuda test` and `make BACKEND=cuda REAL=double test`
  green on the reference GPU, the Quadro M2000M (output recorded)
- [ ] `make BACKEND=cuda sanitize` clean
- [ ] `make equiv`: §10 protocol met for `smoke` and `small`; cross-build
  checkpoint resume works both ways
- [ ] CPU build unchanged: `make test` green with the recorded values
  (smoke 5.7445, checksum `0d80323551e3bb51` on the serial reference
  machine)
- [ ] `bench/cuda_W-A.csv`, `cuda_W-B.csv`, `cuda_W-C.csv` committed (±3%);
  serial re-benchmarked on the same machine's CPU; speedups computed from
  CSVs with CPU and GPU named
- [ ] `docs/PROFILE_CUDA.md` with the per-kernel profile and every
  optimization
- [ ] Fixed-budget comparison and a CUDA full-dataset run recorded in
  `docs/RUNS.md`; prompt samples scored
- [ ] README: CUDA build, test, run and bench commands verified on the GPU
  machine
- [ ] `docs/HANDOFF.md` updated with the final tolerances and the Phase D
  notes (B7)
