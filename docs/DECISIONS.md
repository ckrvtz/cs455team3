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

## 2026-11-17: Parameter and activation layouts (P0.4)

- **Change:** `layout.h` defines the flat buffers.
  - **Parameter tensors:** one tensor per layer per weight, named
    `wte, wpe, h<l>.{ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_g, ln2_b,
    fc_w, fc_b, fcproj_w, fcproj_b}, lnf_g, lnf_b[, head_w]`, in that order.
  - **Padding:** each offset is rounded up to **16 elements**, not 16 floats'
    worth of bytes. In the double build that is 128 B, still 64-byte aligned.
    The padded length is then the same in every build.
  - **Weight decay:** applies to `wte` and every 2-D weight matrix except
    `wpe`. The untied `head_w` also decays (§2.3 lists the tied set; an untied
    head is the same kind of matrix).
  - **Activation arena:** holds the forward and saved-for-backward tensors
    from §6.2. The per-layer list and its names are in `layout.h`.
- **Reason:** ROADMAP §4.3, §6.2, P0.4. Per-layer named tensors map
  one-to-one to PyTorch parameter names for the golden files (P2.6/P3.7).
- **Measured** (`make test`): `base` B=32 T=256 arena = 386,293,760 elements
  ≈ 1473.6 MB float; `small` ≈ 841.3 MB; `smoke` ≈ 10.1 MB. This matches the
  §13 estimate of about 1.5–2 GB once backward scratch is added.
- **Impact:** layout is a contract. P3.5 appends backward scratch tensors
  (activation gradients, attention scratch) to `make_act_layout`; appending
  entries keeps existing offsets and this API. Checkpoints (P4.3) store
  `params/m/v` in this order, so reordering tensors would break checkpoint
  compatibility in all four builds.

## 2026-11-17: Pinned data sources (P1.1)

- **Change:** `scripts/download_data.sh` fetches the original TinyStories
  train/valid text (`roneneldan/TinyStories` @ `f54c09fd…`) and the GPT-Neo
  `vocab.json`/`merges.txt` (`EleutherAI/gpt-neo-125m` @ `21def018…`). Both
  are pinned by commit, and each file is checked against a SHA-256. Files go
  into `$DATA_DIR` (default `data/`). Existing files are kept if their
  checksum matches; a wrong checksum is an error. Interrupted downloads
  resume from `<file>.part`.
- **Reason:** ROADMAP §7.1, D3/D4. Pinning means every team member and all
  four builds tokenize the same bytes, so `prepare_data` output, coverage and
  validation loss are comparable.
- **Checksums:** dataset SHA-256s are the LFS hashes Hugging Face publishes.
  The tokenizer files are not in LFS. Their SHA-256s were computed after
  confirming `git hash-object` equals the blob ids listed at the pinned revision.
- **Impact:** none on the code. Cluster users can set `DATA_DIR` to scratch
  storage.

## 2026-11-17: Tokenizer implementation and golden fixtures (P1.2)

- **Change:**
  - **Character classes:** following §7.2, non-ASCII counts as a letter
    unless it is in an explicit table of Unicode whitespace, number and
    non-letter ranges (punctuation, symbols, emoji, marks, format
    characters), in `src/data/tokenizer.cpp`.
  - **Invalid UTF-8:** each invalid byte is one "other" character, so
    encoding stays lossless.
  - **Special tokens:** not recognized by `encode()`; `prepare_data`
    inserts EOS id 50256 itself.
  - **Golden file:** `tests/golden/tokenizer_cases.txt` has 304 cases (hand
    edge cases plus a seeded TinyStories sample) with expected ids **and**
    HF pre-token byte lengths.
  - **Subset fixture:** `make test` uses a committed subset
    `vocab`/`merges` fixture (`tokenizer_{vocab,merges}_subset.*`, about
    62 KB) so it needs no download. It also reruns the cases on the full
    files when `data/` has them.
- **Reason:** CLAUDE.md §6 forbids tests that need downloads, and the full
  vocab and merges files (1.35 MB) exceed the golden-file size target. The
  subset (only the merges applied to the golden cases, in original order) is
  exact for those inputs. The generator verifies this by re-tracing BPE
  against HF.
- **Why pre-token lengths are checked:** GPT-2 merges never cross pre-token
  boundaries, so many character-class errors leave ids unchanged.
  Deliberately breaking five table entries left every id check passing, but
  the pre-token check caught all five.
- **Offline validation** (scratchpad, not committed): C++ ids identical to HF
  on the whole validation file (21,990 stories, 4,702,514 tokens) and on
  every train story containing non-ASCII (130,115 stories, 27,164,705
  tokens). Pre-token splits were also identical on the latter.
- **Impact:** none on CUDA/MPI (host-only). `dev_only/requirements.txt` pins
  `tokenizers==0.21.1` for regenerating golden files.

## 2026-11-17: prepare_data details (P1.3)

- **Token file header:** §7.3 lists `magic(4) + u32 version + u32
  vocab_size + u64 num_tokens + 8 reserved` = 28 bytes but calls the header
  32 bytes. The extra 4 bytes are taken as natural-alignment padding before
  the u64, which is the layout a C struct with those fields has:
  `num_tokens` at offset 16, reserved at 24..31, bytes 12..15 zero. Every
  field is little-endian and written byte by byte. Spelled out in
  `include/tslm/token_file.h` (frozen).
- **Story splitting:** input is split on the literal `<|endoftext|>`. Each
  story is trimmed of leading/trailing ASCII whitespace (the files put `\n`
  on both sides of every separator) and empty stories are skipped. One EOS
  (new id 1) follows every story.
- **Truncation:** `--max_tokens N` keeps whole stories while the running
  total, EOS included, stays ≤ N. `--max_stories N` keeps the first N
  non-empty stories. Both apply to train only.
- **Vocab fill:** if fewer than V-2 distinct tokens occur in train,
  zero-count tokens fill the rest by the same (count desc, GPT id asc) order.
  `vocab_size` is an option (default 10,000) so tests can use a tiny V.
- **Hashes:** `meta.txt` hashes are 64-bit FNV-1a. The vocab hash covers the
  GPT ids in new-id order (u32 LE, `<unk>` = 0xFFFFFFFF). The output hashes
  cover the whole `.bin` files.
- **Impact:** the loader (P1.4) reads `num_tokens` at offset 16. CUDA/MPI
  builds read the same files unchanged.

## 2026-11-18: DataLoader details (P1.4)

- **Change:**
  - **Fisher–Yates direction:** `epoch_permutation` uses
    `for i = n-1 down to 1: j = rng.uniform_int(i+1); swap(perm[i], perm[j])`
    with `Rng(seed, 100 + epoch)`. That order is frozen by a known-answer
    test, and every build (CUDA/MPI) must reproduce it.
  - **Resume state:** `{epoch, step}` is the global position (same on every
    rank). An epoch rolls over lazily, so `{e, steps_per_epoch}` and
    `{e+1, 0}` serve the same next batch.
  - **Token validation:** every token is checked against the header's
    `vocab_size` as it is copied.
  - **POSIX `mmap`:** the token file is mapped with POSIX `mmap`, which is
    host-only and available on Linux/macOS clusters.
- **Reason:** ROADMAP §7.4, §4.5. With the state and permutation defined
  globally, a 4×8 sharded run serves the same rows as a serial batch of 32,
  which the sharding-equivalence test checks.
- **Impact:** Phase C ranks open the same file with `(rank, world_size)`.
  The checkpoint (P4.3) stores `LoaderState`.

## 2026-11-18: Gradient zeroing is timed under `bwd` (resolves the open `zero` phase question)

- **Change:** each training step zeroes `grads` once, as its **first** action
  (CLAUDE.md §4.7 order: `zero_grads → forward → backward →
  comm_allreduce_avg → clip → AdamW`). The time is charged to `Phase::Bwd`
  (`t.begin(Bwd); zero(grads); t.end(Bwd);`). `PhaseTimer` sums repeated
  begin/end pairs, so `bwd_ms` = zeroing + backward. Any activation-gradient
  buffers that backward accumulates into (P3.5) are zeroed under `bwd` as
  well. ROADMAP §9's separate `t.zero` phase, placed between forward and
  backward, is superseded: there is no `zero` phase and no `zero_ms` column.
- **Reason:** `log.csv`, the bench CSV and `PhaseTimer` (data, fwd, bwd,
  comm, opt) stay frozen and unchanged. The phase columns still sum to
  `step_ms`, so benchmarks stay honest. Zeroing exists only because backward
  accumulates (`+=`), so it belongs with backward. Running it before forward
  instead of after is numerically identical, because forward never touches
  `grads`.
- **Impact:** none on loss, gradients or parameters. All four builds must
  time zeroing the same way, so their `bwd_ms` values stay comparable. If
  P3.5/P7.2 profiling shows activation-gradient zeroing is a significant part
  of `bwd_ms`, record the split in `docs/PROFILE.md` instead of adding a
  column. Making backward overwrite on first write instead of zeroing would
  change the `+=` convention and needs approval.

## 2026-11-18: ops.h conventions (P2.1)

- **Change:** `include/tslm/ops.h` is created with the four P2.1 forward
  signatures, copied exactly from ROADMAP §8. Later tasks add the remaining
  §8 functions as they implement them, so no declared op lacks a definition.
  - **Aliasing:** output buffers must not overlap input buffers in any op.
    That leaves CUDA kernels free to reorder.
  - **File layout:** CPU kernels go in one file per op family,
    `src/ops/<family>_cpu.cpp`. The `_cpu` suffix means only `BACKEND=cpu`
    compiles them.
  - **LayerNorm numerics:** two passes (mean, then centered sum of squares),
    accumulated in double. `mean`/`rstd` are rounded to `real_t` once, and
    those rounded values are both used for the normalization and saved for
    backward.
- **Reason:** CLAUDE.md §4.2 and §5. The double accumulation and the
  consistent saved statistics help the float32-vs-float64 golden tolerances
  (§10).
- **Impact:** CUDA `ops_cuda.cu` (Phase B) implements the same signatures
  under the same no-overlap rule. Its LayerNorm reductions may reorder sums,
  which the §10 equivalence tolerances allow.

## 2026-11-18: Model API types (P2.5), APPROVED 2026-11-18 (option A)

- **Change** relative to the ROADMAP §8 Model API sketch:
  1. **Loss type:** `model_forward` returns **`double`** (the sketch says
     `float`). The value is the host-side mean of a double `sum_reduce`.
  2. **Logits type:** `model_logits` returns **`const real_t*`** (the sketch
     says `const float*`).
  3. **Struct visibility:** `struct Model` is defined in `model.h` instead of
     being opaque, so the training loop and optimizer can reach the
     `params`/`grads`/`m`/`v` pointers and layouts. The sketch already lists
     those members.
  4. **Device tokens:** `tokens`/`targets` passed to `model_forward` are
     device buffers (host memory in the CPU build). Per §9, the loop
     `copy_h2d`s them during the data phase.
  5. **`model_summary()`:** added, so apps can print parameter counts and the
     activation arena in MB at startup (P2.5 DoD).
- **Reason:**
  1. CLAUDE.md §5 requires double for loss accumulations, and the P3.6
     `REAL=double` finite-difference gradcheck needs a double-precision loss.
     A `float` return would cap it near 1e-7 relative.
  2. In the `REAL=double` build the logits are doubles, so a `float*` cannot
     point at them.
  - In the release float build both types hold exactly the same values as
    the sketch's.
- **Impact:** CUDA/MPI builds use the same signatures. `model_logits` is a
  device pointer there, so generation must `copy_d2h` the last-position
  slice. No file format changes.
- **Init (§2.3) as implemented:** tensors are classified by name suffix
  (`_b` → 0, `_g` → 1, `proj_w`/`fcproj_w` → N(0, 0.02/√(2L)), other 2-D
  weights → N(0, 0.02)). Tensor *i* uses `Rng(seed, i)`, as §4.6 requires.
- **Measured:** fresh `base` model on `data/val.bin` (seed 1337, 2×4×256
  tokens): initial loss **9.2641**, versus ln(10000) = 9.2103 (DoD ±0.15).

## 2026-11-18: Golden data format and forward golden cases (P2.6)

- **Change:**
  - **TSGD format:** golden tensors use a new named-tensor container
    (`include/tslm/tensor_file.h`): little-endian; `f32`/`f64`/`i32`/`i64`.
  - **`dump_init`:** the app writes the config, the seed, the
    `model_init_weights` params as f32 (optionally × `--scale`), and a batch
    drawn from `Rng(seed, 900000)`. That stream is kept apart from the init
    streams (tensor index) and the data streams (100 + epoch).
  - **`gen_golden.py`:** `dev_only/gen_golden.py` re-implements the model in
    float64 PyTorch 2.5.1 and writes the loss, per-position losses and
    logits.
  - **Committed cases:** `tiny` (§10 config, tied, standard init) and
    `tiny_untied_x10` (untied, weights × 10). Recipe in
    `tests/golden/README.md`.
- **Reason:** ROADMAP §10. At standard init the logits are nearly uniform, so
  loss is a weak signal. Deliberately dropping the attention 1/√hs scale
  changed the `tiny` loss by only 9.2e-6 relative (inside the §10 1e-5 loss
  tolerance) but moved its logits by 7.6e-4 relative L2, and the ×10 case
  by 1.2e-2 in loss. So `test_golden` also checks per-position losses and
  logits rel-L2 (< 1e-5), in addition to the §10 loss criterion.
- **Measured (`make test`):**

  | Build | `tiny` loss rel err | `tiny_untied_x10` loss rel err | Logits rel-L2 (worst) |
  |---|---|---|---|
  | float | 8.4e-9 | 3.1e-8 | 4.6e-7 |
  | double | 0 | 3.4e-16 | 9.2e-16 |

- **Tooling note:** `download.pytorch.org` (CPU wheel) is not allowed in the
  cloud environment, so the PyPI wheel was installed together with its CUDA
  runtime packages (~2.5 GB). PyTorch remains offline/dev-only: `make`,
  tests and `make dist` never use it.
- **Impact:** P3.7 extends `gen_golden.py` with gradients and AdamW steps on
  the same cases. CUDA/MPI builds compare against the same files.

## 2026-11-18: Model API types approved (resolves the P2.5 open question)

- **Change:** the team approved option A. `model_forward` returns `double`
  and `model_logits` returns `const real_t*`, as implemented in P2.5. These
  signatures supersede the `float` / `const float*` sketch in ROADMAP §8.
- **Reason:** the P3.6 gradient check (§10: central differences, ε = 1e-5,
  rel err < 1e-6, `REAL=double`) needs a double-precision loss. A `float`
  loss would give about 1e-2 error in the numeric gradients. In the
  double build the logits are doubles, so the pointer must be `real_t*`.
  Option C (raising ε) was rejected: it would loosen the §10 test.
- **Impact:** none on the float builds' values, training, `log.csv`,
  checkpoints or file formats. CUDA/MPI implement the same signatures:
  `model_forward` returns a host double, and `model_logits` is a device
  pointer (generation copies the last-position row to the host).

## 2026-11-18: GELU evaluated as x·σ(2u) instead of 0.5·x·(1+tanh u) (P3.1)

- **Change:** `gelu_forward`/`gelu_backward` evaluate the §6.3 tanh-GELU
  through the exact identity 1 + tanh(u) = 2σ(2u):
  `out = x·σ(2u)` and `d/dx = s + 2x·s·(1−s)·k0·(1+3k1x²)`, with `s = σ(2u)`
  and `1−s = σ(−2u)` computed directly.
- **Reason:** the literal `1 + tanh(u)` cancels badly when tanh(u) ≈ −1
  (x below about −4). In the `REAL=double` build, GELU and its derivative
  near x = −6 were only accurate to about 1e-7 relative. The P3.1
  finite-difference check (§10, rel err < 1e-6) failed there at 1.25e-4.
  With the σ form the worst error is 8.7e-10. The function is unchanged
  (same math, same §6.3 definition), so the backward is still its exact
  derivative.
- **Impact:** forward values change only at the rounding level. All
  forward tests still pass, including the PyTorch golden (loss rel err 0 /
  1.1e-16 in double). Note for CUDA (Phase B): use the same σ form; a
  literal-tanh kernel would agree only within the §10 equivalence
  tolerances, not bit for bit. Also, gradient checks of elementwise ops
  evaluate only the perturbed element's term of the scalar loss; that is
  mathematically identical and avoids rounding noise from unrelated terms.

## 2026-11-18: README.md written early and kept current every task

- **Change:** `README.md` exists now (after P3.1) rather than first appearing
  in P7.3. CLAUDE.md §2 and §11 now require every task that changes anything
  a user runs or configures to update the README in the same change. The
  README may only list commands that have actually been run, its Status line
  tracks phase milestones, and the §2 final report says what changed in it.
- **Reason:** teammates need working build, data and tool instructions
  during development, not only at hand-off. Updating the README task by task
  keeps it accurate, rather than reconstructing it from memory at the end.
- **Impact:** ROADMAP P7.3 changes from writing the README to completing
  it: adding exact reproduction commands for a smoke run, a `small` run and
  a benchmark, the machine spec table, and the clean-clone timing in its DoD.
  The ROADMAP edit itself is proposed separately. No effect on code, formats
  or the CUDA/MPI builds.

## 2026-11-18: Gradient check uses a 4th-order 5-point finite difference (for P3.6), approved

- **Change:** the full-model gradient check (P3.6, ROADMAP §10 "Gradcheck"
  row) estimates each numeric derivative with the 4th-order central stencil
  `f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h)` with **h = 1e-3**.
  This replaces the 2-point central difference with ε = 1e-5. The **pass
  criterion is unchanged:** every sampled entry (up to 20 random per tensor
  plus first and last) must have
  `|a − n| / max(|a|, |n|, 1e-8) < 1e-6` in the `REAL=double` build.
  - The test also prints, per tensor, the worst relative error, the gradient
    magnitude where it occurred, and the classic 2-point estimate, for
    diagnosis.
  - An entry that still fails is reported as a failure, never auto-passed,
    even if it is below the measured noise floor. Such cases are resolved
    case by case with the P3.7 PyTorch-autograd golden comparison.
    Noise-aware acceptance (an absolute noise-floor pass) is not adopted;
    it would need separate approval.
- **Reason:** the 2-point difference has ~1e-10 absolute error (truncation
  ε²|f'''|/6 plus rounding u·|L|/ε, with L ≈ ln 32 for the §10 model), so
  gradient entries below ~1e-4 cannot be checked to 1e-6 even when the
  backward is exact. That was observed in P3.3 and P3.4 (e.g. a 2.6e-7
  gradient differing from its finite difference by 6e-12, the noise level).
  The 5-point stencil (truncation O(h⁴), rounding ~1.5·u·|L|/h) is about 100×
  more accurate (~1e-12 absolute), making entries down to ~1e-6 checkable at
  the unchanged criterion. It costs twice the forward passes per entry,
  which is negligible for the tiny gradcheck model.
- **Rejected alternatives:** noise-aware acceptance (loosens the criterion for
  small entries); a smaller-magnitude objective (no longer checks the real
  loss end to end); threshold-based sampling (silently skips small-gradient
  tensors); a `REAL=long double` build (new build mode, not portable,
  no CUDA analogue).
- **Impact:** test methodology only; no code contracts, formats or tolerances
  change. P3.7 (golden autograd, per-tensor rel-L2 < 1e-4) remains the exact
  backstop for small gradients. CUDA/MPI builds use the §10 cross-version
  equivalence protocol, not this check. The per-op FD tests (P3.1–P3.4) keep
  their 2-point differences; they all pass, and switch to this stencil only
  if one of them needs it. The ROADMAP §10 "Gradcheck" row was updated to
  name the new method.
