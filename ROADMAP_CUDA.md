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

---

## 0. How to use this document (humans and LLMs)

- **Task IDs:** every task has an ID (`B2.3`). Prompt with: "Implement task
  B2.3 from ROADMAP_CUDA.md, following CLAUDE.md." One task per prompt.
- **Status labels:**
  - **(DECISION):** a default chosen here; it can be overridden with a
    `docs/DECISIONS.md` entry.
  - **(NEEDS DECISION):** must be settled by the team before the task that
    depends on it starts. §3 lists them, each with a recommendation.
- **Contracts:** `include/tslm/ops.h`, `backend.h`, `comm.h`, `layout.h`,
  the checkpoint and `log.csv` formats, and the bench CSV columns.
  - They are frozen.
  - The only contract additions this roadmap anticipates are in §4.4. Each
    still needs explicit approval and a DECISIONS entry before it is made.
- **Definition of Done (DoD):** a task's acceptance criteria pass ON A GPU,
  with the real output shown. A CUDA task whose code was not compiled AND
  run on an NVIDIA GPU is not done, however plausible it looks (§11).
- **Numbers:** every number quoted in this file is either measured (with its
  source) or labelled an estimate.

---

## 1. Goals and non-goals

### Goals

1. **A CUDA build** (`make BACKEND=cuda` → `bin/train_cuda`, plus every
   other app) that implements every `ops.h` function as bespoke CUDA
   kernels, with the model, training, data, checkpoint and app code
   unchanged.
2. **Equivalence with serial** under the ROADMAP §10 cross-version protocol,
   and the whole existing test suite (30 test programs, 154 test cases)
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
- **Dependencies:** no third-party GPU libraries (cuDNN, CUTLASS, Thrust,
  CUB, FlashAttention, ...) in the model path. cuBLAS is governed by
  decision D-B2.
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

Open Phase A items: the P5.2 `small` run, the P6.1 grammatical check, the
P6.3 scoring pilot, and the P5.3 Serial Complete review. See decision D-B9.

---

## 3. Decisions

### Settled here (DECISION)

| ID | Decision | Why |
|---|---|---|
| D-B3 | **Deterministic kernels:** no floating-point `atomicAdd` in any kernel; every reduction has a fixed order (fixed tree or fixed sequential order per output). Same binary + same GPU + same seed ⇒ bit-identical runs. | The existing determinism and resume-bit-exactness tests must pass unchanged in the CUDA build. Atomic float adds make the last bits run-dependent. |
| D-B4 | **FP32, no TF32, no fast math:** `nvcc` never gets `--use_fast_math`; no `__expf`/`__fdividef` style intrinsics unless approved; no tensor-core TF32 paths. | §10 tolerances assume FP32. CLAUDE.md forbids fast math for the serial build, and the same honesty applies here. |
| D-B5 | **Attention = the proposal's batched-GEMM design:** QKᵀ as a batched GEMM into `att`, a causal-masked row softmax kernel, then att·V as a batched GEMM. Backward mirrors it. Full T×T tiles are allowed at first; upper-triangle tile skipping is an optimization (B6.3). | Proposal ("batched matrix multiplication for QKᵀ and PV with a causal-masked softmax"), and `ops.h` saves `att`. |
| D-B6 | **One CUDA stream** (the default stream), synchronous semantics at phase boundaries via `dev_sync()`. | Simplicity and determinism; overlap is a non-goal. |
| D-B7 | **Tests run in every build:** the test suite is made backend-agnostic (B0.3), and the CUDA build adds equivalence tests against a reference copy of the serial kernels (B0.4). No test is skipped in the CUDA build without a DECISIONS entry. | ROADMAP §10: "later versions reuse the validation suite unchanged". |
| D-B10 | **Memory:** the existing flat buffers and single activation arena are allocated once with `cudaMalloc` through `dev_malloc`. No allocation in the training loop, including no `cudaMalloc` inside ops. Kernel launches use no dynamic allocation. | CLAUDE.md §4.5; the arena sizes (≤ 2.1 GB for `base`) fit any GPU with ≥ 4 GB. |
| D-B12 | **Sanitizers:** `make BACKEND=cuda MODE=debug` builds with `-G -lineinfo`. A `make BACKEND=cuda sanitize` target runs the tests under `compute-sanitizer` (memcheck; racecheck and synccheck for kernels with shared memory). | The CUDA analogue of the serial ASan/UBSan build. |

### Needs a team decision before the dependent task (NEEDS DECISION)

| ID | Question | Recommendation | Blocks |
|---|---|---|---|
| D-B1 | **Which GPU machine(s) and CUDA version** are the targets (lab machine, cluster node type)? | Pick one reference GPU for all recorded numbers and write it in README's machine table. Require CUDA ≥ 11.8 (12.x preferred). Build with `CUDA_ARCH ?= native` (nvcc `-arch=native`, CUDA ≥ 11.6), overridable (`CUDA_ARCH=sm_80`). The cluster GPU type decides Phase D's arch. | B0.1 |
| D-B2 | **Matmul: bespoke kernels or cuBLAS?** `docs/HANDOFF.md` currently says "cuBLAS SGEMM ... is the expected implementation". The proposal says the team writes "the GPU kernels for the matrix multiplication", and CLAUDE.md requires bespoke transformer ops. | **Bespoke tiled SGEMM kernels are the default and the only path in the model.** cuBLAS may optionally be linked into the TEST and BENCH builds as an oracle and a performance yardstick, behind `MATMUL_REF=cublas`, if the team approves it (it is a CUDA-toolkit library, cited in THIRD_PARTY.md). HANDOFF's matmul paragraph is then amended. | B2.1 |
| D-B8 | **Contract additions to `backend.h`** (§4.4): `dev_init(int local_rank)` (device selection, needed for Phase D) and `dev_peak_bytes()` (device memory high-water mark for `bench`). | Approve both now. They are additive; the CPU implementation is a no-op and a byte counter. | B0.2 |
| D-B9 | **May Phase B start before the Serial Complete gate (ROADMAP §14) is signed off?** Open: the P5.2 run, the P6 grammatical check and scoring pilot, and the P5.3 review. None of them changes a contract. | Yes. Record a waiver in DECISIONS: Phase B may start, and the gate must be signed off before B5.4 (CUDA equivalence sign-off). | B0.1 |
| D-B11 | **Kernel source layout:** one `ops_cuda.cu`, or one `.cu` per op family? | One `.cu` per family, mirroring serial: `src/ops/{embedding,layernorm,elementwise,matmul,attention,softmax_xent,optim}_cuda.cu`, plus `src/ops/cuda_common.cuh` for shared device helpers (block reduce, launch macros). `ops_cuda.cu` (the P7.5 stub) is deleted in B0.2. | B0.2 |

---

## 4. Architecture of the CUDA build

### 4.1 Files

```
include/tslm/ops.h               unchanged (contract)
include/tslm/backend.h           + dev_init, dev_peak_bytes (D-B8, if approved)
include/tslm/check.h             CUDA_CHECK(call) implemented for BACKEND=cuda
src/backend/backend_cuda.cu      dev_malloc=cudaMalloc, dev_free, dev_memset=cudaMemset,
                                 copy_*=cudaMemcpy (synchronous), dev_sync=cudaDeviceSynchronize,
                                 dev_name=GPU name + compute capability, dev_init, dev_peak_bytes
src/ops/cuda_common.cuh          device helpers: warp/block reductions (fixed order),
                                 grid-stride loop helpers, launch-check macro, ceil_div
src/ops/<family>_cuda.cu         one per op family (D-B11); host wrappers with the
                                 exact ops.h signatures launch the kernels
src/ops/gemm_cuda.cuh            internal bespoke (batched, strided) SGEMM used by matmul
                                 and attention; NOT part of ops.h
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
  `-lcudart` (add `-lcublas` only if D-B2 approves the reference).
- **`nvcc` flags:** `-std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo
  -Xcompiler -Wall,-Wextra`, the same `-D` selection macros, and **no**
  `--use_fast_math`.
- **Configurations and stamps:**
  - `CUDA_ARCH ?= native`. The target stamp (`build/<tag>/target.stamp`)
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
  - `make BACKEND=cuda MODE=debug` uses `-G -lineinfo`; ASan is not mixed
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
   which must return a host `double`. Those two may use a small persistent
   device scratch allocated once by `backend_cuda` at `dev_init`/first use,
   never in the loop; document it.
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

### 4.4 Contract additions (pending approval D-B8)

```cpp
// backend.h (additive)
// Select and initialise the device for this process: CUDA cudaSetDevice(local_rank % device_count)
// plus a warm-up; CPU no-op. Called once by every app right after comm_init, before any dev_*.
void dev_init(int local_rank);
// High-water mark of bytes allocated through dev_malloc (CPU: same counter, host bytes).
size_t dev_peak_bytes();
```

- **Callers:**
  - `train`, `bench`, `generate`, `eval` and `gradcheck` call
    `dev_init(...)`;
  - serial and Phase B pass 0;
  - Phase D passes the node-local rank (§9).
- **`bench`:** `peak_mem_MB` in CUDA builds = `dev_peak_bytes()/2²⁰`
  (device memory). The CSV columns are unchanged.
- **Manifests and bench:** `dev_name()` returns, e.g., "NVIDIA A100-SXM4-40GB
  (sm_80), driver 535.x, CUDA runtime 12.2". The bench manifest records it,
  so every CUDA number names its GPU.

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

### 5.4 Matmul (`matmul_cuda.cu`, `gemm_cuda.cuh`), the critical path

The serial profile puts 81% of time here.

- **One internal bespoke GEMM:**
  - It computes `C[M,N] (=|+=) α·op(A)·op(B)` with row-major strides,
    optional batch strides, and transpose flags.
  - It is used by `matmul_forward`/`backward` and by the attention batched
    GEMMs.
  - Development order: a naive one-thread-per-output kernel first (the
    correctness reference for later steps), then shared-memory tiling, then
    register blocking.
  - The final kernel is expected to be a tiled kernel: block tile of the
    order of 64×64 or 128×64, K-tile 8–32, each thread computing a 4×4 or
    8×8 register micro-tile, double-buffered shared-memory loads optional.
    Document the choice in `docs/PROFILE_CUDA.md`.
- **The three uses:**
  - `matmul_forward`: `out[N,OC] = inp[N,C]·W[OC,C]ᵀ (+ bias)`.
  - `matmul_backward`: `dinp[N,C] += dout[N,OC]·W[OC,C]`;
    `dW[OC,C] += dout[N,OC]ᵀ·inp[N,C]`;
    `dbias[OC] += column sums of dout` (deterministic column reduction as in
    §5.3, skipped when null).
  - All three accumulate.
- **No split-K with atomics (D-B3).** If split-K is needed for `dW`
  (K = N = 8192), write per-split partials into scratch and reduce them in
  fixed order.
- **Edge shapes:** any `N, C, OC`, including N = 1 (generation), OC = V =
  10,000 (not a multiple of the tile), and the tiny test shapes
  {1, 7, 16, 100}. Bounds-check every load and store.
- **Tolerance:** rel-L2 ≤ 1e-5 per output.
- **Performance expectation:** to be measured.
  - The kernel's GFLOP/s on the W-A head shape (N=8192, C=128, OC=10,000)
    is reported in `docs/PROFILE_CUDA.md`.
  - If cuBLAS is approved as a yardstick (D-B2), the target is ≥ 50% of
    cuBLAS SGEMM on that shape. Otherwise the target is a documented
    fraction of the GPU's FP32 peak.

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
- **Causal efficiency:** the first implementation may compute full T×T
  tiles and mask them. B6.3 adds upper-triangle tile skipping in the GEMMs
  (tiles entirely above the diagonal are skipped, and diagonal tiles are
  masked). Benchmark notes must say which version ran (`docs/PROFILE.md`,
  serial is triangular).
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
  - Speedups are only meaningful on the same host CPU for data and host
    work. The serial number comes from its reference machine; state both
    machines next to every speedup.
  - Never quote a speedup that is not computed from committed CSV rows.
- **Fixed wall-clock budget** (the proposal's metric): train `small` for the
  same number of minutes with serial and CUDA (e.g. 30 min each; the exact
  budget is set in B6.5). Record tokens consumed and final val loss.
- **Estimates to plan with** (not results):
  - `small` is ≈ 9×10¹⁰ FLOP/step (ROADMAP §13). At an effective 5–10
    TFLOP/s that is ~10–20 ms/step, so one epoch of the full train set
    (463.8M tokens, 56,620 steps at 8,192 tokens/step) is ~10–20 min.
  - `base` (≈ 3×10¹¹ FLOP/step) is roughly 3× that.
  - Full-dataset training, which serial cannot do (serial W-A: 3.61 s/step,
    i.e. ~57 h per epoch), becomes practical. Measure it in B6.5.
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
| B0.1 | Settle D-B1 (GPU, CUDA version, arch) and D-B9 (gate waiver) in DECISIONS. Install and verify the toolkit on the GPU machine: `nvcc --version`, `nvidia-smi`, a trivial kernel compiled with the project flags | Vincent | DECISIONS entry; toolchain output recorded; README machine table gains a GPU row |
| B0.2 | CUDA build plumbing: remove the `BACKEND=cuda` guard in `build_config.h`; implement `CUDA_CHECK`; write `backend_cuda.cu` (incl. D-B8 functions if approved); split `ops_cuda.cu` into per-family files (D-B11) whose wrappers initially `CHECK(false, "not implemented: <op>")`; Makefile `.cu` flags, `CUDA_ARCH`, link, stamp including the nvcc version, `MODE=debug` and `sanitize` targets | Vincent | `make BACKEND=cuda` builds every app; `bin/cuda-float/...` runs and aborts with "not implemented: …" at the first op; CPU build unchanged (`make test` green with identical recorded values) |
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
| B2.1 | Settle D-B2. Write the internal GEMM, naive version (one thread per output, strided, batched, transpose flags, `=`/`+=`) | `matmul_forward/backward` equivalence over the full shape sweep incl. N=1 and OC=10,000 |
| B2.2 | Tiled shared-memory GEMM | Same tests; GFLOP/s on the W-A head shape recorded vs the naive version (and cuBLAS if approved) in `docs/PROFILE_CUDA.md` |
| B2.3 | Register-blocked GEMM, plus deterministic `dbias` and `dW` without atomic split-K | Same tests; performance table updated; `compute-sanitizer` racecheck clean |

### B3: LayerNorm and softmax + cross-entropy (W10)

| ID | Task | DoD |
|---|---|---|
| B3.1 | `layernorm_forward/backward` with the deterministic two-stage `dgamma`/`dbeta` (§5.3) | Equivalence rows pass; `test_ops_backward` FD checks pass in the CUDA build |
| B3.2 | `softmax_xent_forward/backward` (§5.5) | Equivalence rows pass; `test_softmax_xent*` pass in the CUDA build; uniform-logits loss = ln V |

### B4: attention (W10)

| ID | Task | DoD |
|---|---|---|
| B4.1 | `attention_forward` via batched GEMM + masked softmax kernel (full tiles allowed) | Equivalence passes; `test_attention` incl. exact-zero upper triangle and op-level causality |
| B4.2 | `attention_backward` via batched GEMMs + the softmax-backward row kernel in `scratch` | Equivalence passes (Q/V slices; key slice by bound); `test_attention_backward` passes |

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
| B6.2 | Optimize the top kernels by profile (GEMM tiling parameters, softmax-xent row kernel, LayerNorm), each change measured and documented, numerics re-verified | Each optimization: before/after numbers in PROFILE_CUDA.md; suite + equiv green |
| B6.3 | Causal tile skipping in attention GEMMs (§5.6) | Equivalence + causality pass; attention time before/after recorded |
| B6.4 | CUDA benchmarks W-A, W-B, W-C (3 runs each, ±3%); speedup table vs serial (computed from CSVs) | `bench/cuda_W-*.csv` committed; table in `docs/RUNS.md` or a results doc |
| B6.5 | Fixed-budget comparison (serial vs CUDA, same minutes) and a CUDA full-dataset `small` run (and `base` if time allows); eval samples scored with `eval/RUBRIC.md` as milestone `cuda-<preset>-step<k>` | Results recorded in `docs/RUNS.md` from `log.csv`; scores in `eval/scores.csv` |

### B7: hand-off to Phase D (W11)

| ID | Task | DoD |
|---|---|---|
| B7.1 | Device-pointer readiness of `comm.h`: document and test (with a fake 1-rank comm) that `comm_allreduce_avg(model->grads, ...)` and `comm_broadcast(model->params, ...)` are called with device pointers; decide CUDA-aware MPI vs host staging (§9) | DECISIONS entry; HANDOFF updated with the Phase D notes |
| B7.2 | `dev_init(local_rank)` wired in every app (if D-B8 approved) | Apps call it; serial and CUDA unchanged |

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
| GEMM kernel too slow, making the CUDA results look bad | Naive → tiled → register-blocked progression with measured GFLOP/s; optional cuBLAS yardstick (D-B2); profile-driven B6.2 |
| Non-determinism from atomics or uninitialised scratch breaks determinism/resume tests | D-B3 (no float atomics); `compute-sanitizer --tool initcheck`; tests seed accumulated outputs |
| Silent overwrite instead of accumulate (tied `wte` loses the head gradient) | Equivalence tests seed outputs non-zero; golden tied test; HANDOFF notes |
| Out-of-bounds on edge shapes (N=1, OC=10,000) | Shape sweep in `test_equiv_ops`; memcheck |
| Tolerance creep ("loosen to pass") | Tolerances fixed in B0.4 via DECISIONS; CLAUDE.md forbids loosening without approval |
| Speedup misreported (different machines, warm vs cold, full vs triangular attention) | §12 rules, machine recorded per row, attention mode stated, numbers only from committed CSVs |
| Stale objects across toolkits or GPUs (as happened across CPUs in Phase A) | Target stamp includes the nvcc version and `CUDA_ARCH`; clean rebuild before recording numbers (DECISIONS procedure) |
| Phase D blocked by host-pointer assumptions | §2 audit already clean; B7.1 test with device pointers |
| Arena does not fit a smaller GPU | `base` needs 2.05 GB at B=32; reduce B (arena ∝ B·T, `att` ∝ B·T²); `dev_peak_bytes` reported |

---

## 11. LLM working protocol for Phase B

These complement CLAUDE.md; see the proposed CLAUDE.md changes recorded
with this roadmap.

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
  green on the reference GPU (output recorded)
- [ ] `make BACKEND=cuda sanitize` clean
- [ ] `make equiv`: §10 protocol met for `smoke` and `small`; cross-build
  checkpoint resume works both ways
- [ ] CPU build unchanged: `make test` green with the recorded values
  (smoke 5.7445, checksum `0d80323551e3bb51` on the serial reference
  machine)
- [ ] `bench/cuda_W-A.csv`, `cuda_W-B.csv`, `cuda_W-C.csv` committed (±3%);
  speedups computed from CSVs with both machines named
- [ ] `docs/PROFILE_CUDA.md` with the per-kernel profile and every
  optimization
- [ ] Fixed-budget comparison and a CUDA full-dataset run recorded in
  `docs/RUNS.md`; prompt samples scored
- [ ] README: CUDA build, test, run and bench commands verified on the GPU
  machine
- [ ] `docs/HANDOFF.md` updated with the final tolerances and the Phase D
  notes (B7)
