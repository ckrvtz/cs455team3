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

## 2026-11-18: Backward buffers and model_backward wiring (P3.5)

- **Change:**
  - **Activation layout:** `make_act_layout` appends 8 backward buffers
    **after** all forward tensors, so forward offsets are unchanged. They are
    shared by all layers: `d_logits [N,V]`, `d_x`/`d_ln`/`d_atty [N,C]`,
    `d_qkv [N,3C]`, `d_fch_gelu`/`d_fch [N,4C]`, `att_scratch [B,NH,T,T]`.
    `ActLayout` gains matching index fields, additive as planned in the
    P0.4 entry.
  - **New arena sizes at B=32, T=256** (from `make test`): `base` 1962.1 MB
    (was 1473.6), `small` 1241.8 MB, `smoke` 15.2 MB at B=4, T=32. `base`
    stays within the §13 estimate of 1.5–2 GB. `d_logits` gets its own
    buffer rather than reusing `logits`, so `model_logits()` stays valid
    after backward. Reusing it would save ~330 MB at `base` if ever needed.
  - **`model_backward(Model*)`:** uses the last forward's saved activations
    and its tokens/targets pointers, which `Model` keeps. The caller must
    keep those buffers alive until backward has run.
  - **Residual handling:** one `d_x` buffer carries the residual-stream
    gradient top to bottom. Each residual add needs no extra op: `d_x` is
    the branch's `dout`, and the branch's LayerNorm backward then adds back
    into `d_x`.
  - **Zeroing:** activation-gradient buffers are zeroed inside
    `model_backward` right before each accumulating write. That time counts
    toward `bwd`, per the zero-phase decision. Parameter gradients
    accumulate and are zeroed by the training loop.
  - **Tied head:** `grads[wte]` receives the head contribution (matmul
    backward) and then the embedding contribution.
- **Testing notes for P3.6:**
  1. **Stencil evaluation order:** the 5-point stencil is evaluated
     differences-first, `[8(L(h)−L(−h)) − (L(2h)−L(−2h))]/(12h)`. Subtracting
     nearly equal losses is exact, so an entry the loss does not depend on
     (e.g. untied `wte` rows of absent tokens) yields exactly 0. The summed
     form `−L(2h)+8L(h)−8L(−h)+L(−2h)` left ~1e-13 of rounding, which failed
     the 1e-8 floor.
  2. **Key-bias gradients are structurally zero.** The key-bias slice of
     every `qkv_b` has exactly zero true gradient: shifting all keys equally
     changes every score of a query by the same amount, and softmax ignores
     that. Finite differences cannot confirm 0 to the §10 floor (loss
     rounding alone is ~1e-15). Measured: |d key-bias| / max|d q/v-bias| is
     ≤ 7e-17 in double and ≤ 2e-8 in float. These entries are verified by a
     direct check (`structurally_zero_gradients`) and excluded from FD
     sampling. P3.6 should treat them the same way, as a documented
     structural-zero case under the "case by case" clause of the gradcheck
     decision.
- **Impact:** CUDA/MPI use the same layout, so the arena grows the same way
  on device. `model_backward` calls only `ops::` functions.

## 2026-11-18: Gradcheck structural-zero exception for key biases (for P3.6), approved

- **Change:** the full-model gradcheck (P3.6) samples `qkv_b` only from its
  query `[0, C)` and value `[2C, 3C)` slices. The tensor's first index
  (query) and last index (value) are still checked, as §10 requires. It
  adds a **mandatory direct check** per layer, in the `REAL=double` build:
  `max |d key-bias| ≤ 1e-12 · max |d query/value-bias|`, reported per layer.
  If it fails, gradcheck fails. The ROADMAP §10 "Gradcheck" row was updated
  to match.
- **Reason:** the key-bias slice `[C, 2C)` has an exactly zero true gradient.
  Adding the same vector to every key adds the same amount to all of a
  query's scores, which softmax ignores. Finite differences cannot confirm
  0 to the §10 floor (1e-8 relative, i.e. 1e-14 absolute): the perturbation
  changes only forward rounding (~1e-15 in L), which gives ~1e-13 of
  numeric noise after dividing by 12h. That would fail the criterion even
  with a correct backward. The direct check is far stricter than FD could
  be: P3.5 measured ≤ 7e-17 in double (`structurally_zero_gradients`).
  This is the "case by case" resolution named in the gradcheck-stencil
  decision above.
- **Closed list:** the key-bias slice is the **only** exemption. Adding
  another requires (1) a mathematical proof that the gradient is
  structurally zero, (2) a direct check of it, and (3) team approval
  recorded here. Every other sampled entry must pass the §10 criterion;
  no generic noise-floor or tiny-gradient pass is allowed (rejected in the
  stencil decision).
- **Rejected alternatives:** a generic absolute tolerance for tiny
  gradients (hides real small-gradient bugs); removing key biases from the
  model (changes the §2.1/§2.2 contract, parameter counts and PyTorch
  equivalence); relying on P3.7 alone (per-tensor rel-L2 could hide a
  key-bias error); accepting recurring reported failures (trains people to
  ignore gradcheck).
- **Impact:** test methodology only; no code, contract or format change.
  P3.7's per-tensor comparison against PyTorch autograd still covers
  `qkv_b` as a whole.

## 2026-11-18: Gradcheck configuration: h = 3e-4 on init weights ×3 (P3.6), approved

- **Change:** the full-model gradcheck runs on the §10 tiny model with init
  weights **×3** and stencil step **h = 3e-4**. This amends the h = 1e-3 of
  the stencil decision above. The **pass criterion is unchanged**: rel err
  < 1e-6, floor 1e-8, every sampled entry, plus the key-bias check. The
  ROADMAP §10 row was updated. Implemented in `run_gradcheck`
  (`include/tslm/gradcheck.h`), the `gradcheck` app and
  `tests/test_gradcheck.cpp`.
- **Reason:** measured in the `REAL=double` build, 10 seeds per
  configuration:
  - **Standard init (×1) never passes, at any step size** (0/10 tied and
    untied, at h = 1e-3 and 3e-4). Some gradients there are ~1e-7 (e.g.
    layer-0 LayerNorm biases), below the 5-point stencil's rounding noise
    (~1.5·u·|L|/h ≈ 6e-13 at h = 1e-3), so they cannot be confirmed to 1e-6.
  - **h = 1e-3 also fails on strongly curved entries through truncation.**
    E.g. seed 99, `h1.proj_b`: rel err 1.75e-4 / 2.19e-6 / 1.8e-8 at
    h = 3e-3 / 1e-3 / 3e-4. That is the O(h⁴) scaling; the analytic value is
    correct, as smaller h and the 2-point estimate both converge to it.
  - **h = 3e-4 with ×3 passed 10/10 seeds tied and 10/10 untied**, worst rel
    err 1.5e-7 / 2.2e-7. Default seed 1337: worst 3.1e-8 tied, key-bias
    ratios ≤ 8e-17.
  - At ×3, attention is non-uniform and gradients sit well above the noise
    floor, while the same code is tested.
- **Not lost:** standard-init gradients are still covered exactly by the
  P3.7 PyTorch-autograd golden comparison (no FD noise).
- **Mutation evidence:** gradcheck failed on each of four planted backward
  bugs: a dropped LayerNorm term (52 tensors failing across the tied and
  untied runs), a missing attention dQ scale (36), a wrong GELU derivative
  factor (48), and the wrong LN-1 input in `model_backward` (30).
- **Impact:** test configuration only; no code contracts or formats change.

## 2026-11-18: Backward golden data (P3.7)

- **Change:**
  - **Generator:** `dev_only/gen_golden.py` now builds every parameter as a
    float64 leaf tensor, runs autograd on the mean loss, and adds
    `grad.<name>` for every parameter to `<case>_fwd.bin`.
  - **Regenerated files:** both committed cases (`tiny`, `tiny_untied_x10`)
    were regenerated with PyTorch 2.5.1 from the unchanged `_init.bin`
    inputs. Their `loss`/`losses`/`logits` tensors are **bit-identical** to
    the previous files (checked tensor by tensor). Only gradients were
    added, and the files grew from ~4 KB to 63–67 KB.
  - **Test:** `tests/test_golden.cpp` compares every parameter gradient,
    with per-tensor rel-L2 < 1e-4 (§10).
  - **AdamW:** the "after 5 AdamW steps" part of the §10 golden row belongs
    to P4.1 (which implements AdamW) and is added there.
- **Reason:** ROADMAP §10, P3.7. This regeneration was done with the
  generator, not by hand-editing the golden files (CLAUDE.md §2), because
  the generator gained an output, not because a test failed.
- **Measured** (worst per-tensor gradient rel-L2, limit 1e-4):

  | Build | `tiny` | `tiny_untied_x10` |
  |---|---|---|
  | float | 2.6e-7 | 5.7e-6 |
  | debug (float) | 4.0e-7 | 2.9e-5 |
  | double | 6.1e-16 | 7.6e-14 |

  The double-build `tiny` result confirms the standard-init gradients that
  the finite-difference gradcheck cannot resolve.
- **Impact:** CUDA/MPI builds compare against the same files (cross-version
  protocol, §10).

## 2026-11-18: AdamW golden trajectory; key-bias exception extended (P4.1), approved

- **Change:**
  - **Generator:** `dev_only/gen_golden.py` takes an optional third output,
    `<case>_adamw.bin`. It runs 5 steps of `torch.optim.AdamW` (float64)
    from the same init weights and batch: decay group = 2-D weights except
    `wpe`, global-norm clip 1.0, and the §2.3 schedule set by hand each step.
    It writes the hyperparameters, the per-step loss, the pre-clip grad norm,
    the lr, and every final parameter. Golden settings: lr 1e-3,
    min_lr_frac 0.1, warmup 2, max_steps 5, betas (0.9, 0.95), eps 1e-8,
    wd 0.1. Files: `tests/golden/tiny_adamw.bin` (58 KB) and
    `tiny_untied_x10_adamw.bin` (62 KB). The `_fwd.bin` files are unchanged.
  - **Test:** `tests/test_golden.cpp` runs the C++ step loop
    (`clip_grad_norm` → `lr_at` → `adamw_update`) and checks:
    - step losses rel < 1e-5;
    - grad norms rel < 1e-4;
    - lr rel < 1e-14;
    - final params per-tensor rel-L2 < 1e-4.
  - **Key-bias exception extended:** the structural-zero exception above
    (gradcheck, 2026-11-18) is extended to this one comparison, and the
    closed list stays closed otherwise. For each `qkv_b`, only the query and
    value slices go through the rel-L2 check. The key slice `[C, 2C)` must
    instead satisfy `max|value| ≤ steps · lr` (= 5e-3), the largest
    distance Adam can move a parameter from 0 in 5 steps.
  - **Golden lr is 1e-3, not 1e-2.**
- **Reasons:**
  - **Key slice:** its true gradient is exactly 0 (softmax shift
    invariance), so PyTorch's float64 values stay at ~1e-15. Our float
    gradient is rounding noise of ~1e-9, and Adam normalizes it by
    `sqrt(v)`, which turns noise into steps of size ~lr. A rel-L2 against
    ~0 cannot pass and says nothing about correctness.
  - **Golden lr:** at lr 1e-2 the ×10-weight untied case diverges
    chaotically in float: the step-5 loss rel error was 2.5e-4, and several
    biases sat at 1.1–1.5e-4. At 1e-3 the trajectory is well conditioned,
    so the comparison tests the optimizer, not float sensitivity.
- **Measured** (`make test`, `make debug`, `make test REAL=double`; worst
  values):

  | Build | case | step loss rel | grad-norm rel | params rel-L2 | key max \|value\| |
  |---|---|---|---|---|---|
  | float | tiny | 1.0e-8 | 5.7e-8 | 9.44e-7 | 6.8e-7 |
  | float | untied_x10 | 6.7e-7 | 3.7e-5 | 1.71e-5 | 3.8e-4 |
  | debug | tiny | 8.4e-9 | 4.8e-8 | 6.61e-7 | 4.4e-7 |
  | debug | untied_x10 | 4.2e-7 | 1.6e-5 | 1.41e-5 | 5.1e-4 |
  | double | tiny | 1.3e-16 | 1.5e-15 | 8.92e-16 | 5.7e-16 |
  | double | untied_x10 | 5.3e-15 | 2.0e-13 | 5.86e-14 | 8.5e-13 |

  The double build matches PyTorch to rounding on every tensor, including
  the key slice. This validates the update rule, the clip, the decay mask
  and the schedule exactly.
- **Mutation evidence:** the golden AdamW tests failed on each of four
  planted bugs:
  - dropped second-moment bias correction;
  - decay applied to every tensor;
  - cosine without the min-lr floor;
  - clip returning the post-clip norm.
- **Impact:** CUDA/MPI builds compare against the same files with the same
  key-slice rule. The `adamw_step` signature in `ops.h` is new (not a change
  to an existing contract).

## 2026-11-18: Training loop details (P4.2)

- **log.csv rows:** written at step 1, every `log_every` steps, and at the
  last step.
  - `train_loss`, `lr` and `grad_norm` are that step's values: deterministic,
    so builds can be compared row by row (`%.9g`).
  - The timing columns and `tok_per_s` are per-step means over the steps
    since the previous row; `step_ms` is the wall time of the whole step.
  - `tokens_seen` and `tok_per_s` count global tokens (B·T·world_size).
  - The frozen columns are unchanged.
- **Step body:** follows §9 with zeroing timed under `bwd` (decision above).
  - The loss is averaged over ranks under `comm`
    (`comm_allreduce_sum_double` / size).
  - `check_step_finite` aborts on a non-finite loss before the optimizer and
    on a non-finite grad norm after it (`clip_grad_norm` already refuses a
    non-finite norm before scaling).
  - After init, params are `comm_broadcast` from rank 0 (no-op in serial).
- **manifest.txt:** metadata lines start with `#`, so the config parser
  ignores them, followed by `config_to_text`. The manifest is therefore a
  valid `--config` file that reproduces the run (checked by a test and by
  hand).
  - Recorded: UTC date, git hash, build, compiler and flags, device, CPU
    model, cores, RAM, world size, parameter summary, train tokens and
    steps per epoch.
  - The Makefile passes the git hash (`-dirty` when tracked files differ
    from HEAD) and the flags to `src/train/manifest.cpp` only, through a
    stamp file, so a new commit recompiles one object.
- **Existing run directory:** files are overwritten, with a warning.
  Resuming is P4.3.
- **Not yet produced:** validation (P4.4) and checkpoints (P4.3) are not
  implemented. The loop leaves `val_loss`/`val_ppl` empty and writes no
  checkpoint files, and it says so in its startup log instead of failing,
  so the `smoke` preset (`eval_every=50`) can run. P4.3/P4.4 replace that
  notice.
- **Measured** (`bin/train_serial --config configs/smoke.cfg`, real
  `data/train.bin`, release float, this container's 4-core Xeon @ 2.10GHz):
  - first run: 54.5 s wall, of which 5.7 s user CPU; `data_ms` dominated
    (page-cache misses on the 927 MB mmapped file);
  - immediate rerun: 5.0 s, `data_ms` ≈ 0.03 ms/step;
  - both runs: loss 9.2202 → 6.2963, with identical loss/lr/grad_norm
    columns.
- **Impact:** CUDA/MPI builds use the same loop file; only the
  backend/comm shims differ.

## 2026-11-18: Checkpoint format and resume (P4.3)

- **Format** (`include/tslm/checkpoint.h`, version 1): a 128-byte
  little-endian header, then `params`, `m`, `v` as flat padded host buffers
  (§9).
  - **Header contents:** magic `TSCK`, version, ModelConfig, `step`,
    `tokens_seen`, loader `{epoch, step}`, `num_params`,
    `num_params_padded`.
  - **Additions beyond the §9 list:**
    - `real_bytes`: payload element size, 4 or 8;
    - `seed`, `global_batch` and `seq_len`: the loader position means
      nothing without them;
    - an FNV-1a 64 checksum of the payload;
    - 16 reserved zero bytes.
- **Why `real_bytes`:** the production builds are all float32, but the
  `REAL=double` build must be able to save and resume its own runs exactly.
  - A file with the other element size is converted on load: exact for
    float → double, rounded for double → float. A log line says so.
  - Both directions are tested: the float build loads a double file, and
    the double build loads a float file.
- **Why seed / T / global batch:** resuming with any of them changed would
  silently continue on a different data order, so a mismatch aborts.
  Model-config and parameter-count mismatches abort too.
- **Writing:**
  - The payload streams through a fixed 64 KB stack buffer
    (`copy_d2h`/`copy_h2d`), so no allocation happens in the loop and the
    code works on device pointers.
  - Each file is written to `<path>.tmp` and renamed.
  - `ckpt_<step>.bin` and `ckpt_latest.bin` are written as two full saves of
    identical bytes.
  - Saving happens after the step's timing (`step_ms` excludes I/O), rank 0
    only, followed by `comm_barrier`.
  - Cadence: every `ckpt_every` steps and always at `max_steps`
    (`ckpt_every=0`: only at the end).
- **Resume:** a new config key `resume` (checkpoint path; empty = fresh run).
  - It is a config key rather than an app-only flag, so a resumed run's
    manifest records it and reproduces the run.
  - Everything except params, m, v, step, tokens and loader position comes
    from the run's config. This lets `max_steps` be extended; a changed
    schedule is the user's choice.
  - In the same run directory, `log.csv` keeps rows up to the checkpoint
    step, and later rows (to be rerun) are dropped.
  - The AdamW step count is the checkpoint step. Training uses no randomness
    besides the loader permutation, which is a pure function of
    (seed, epoch).
- **Measured** (release float):
  - `tests/test_checkpoint.cpp`: resuming from step 10 of a 20-step run
    reproduces `ckpt_20.bin` byte for byte, both in a new directory and in
    place, crossing an epoch boundary at step 15. `log.csv` loss/lr/grad_norm
    are identical.
  - `configs/smoke.cfg` on real data: resuming from `ckpt_100.bin` gave a
    `ckpt_200.bin` byte-identical (`cmp`) to an uninterrupted run. The smoke
    checkpoint is 4,005,632 bytes; save time 13–18 ms for two files.
- **Mutation evidence:** the tests failed on each of five planted bugs:
  - loader state not restored;
  - AdamW step count off by one after resume;
  - `v` not saved;
  - checksum not verified;
  - old `log.csv` rows not dropped.
- **Impact:** all four builds read and write this format. Phase C ranks each
  load the same file (no broadcast needed); Phase B/D stream through the
  same `copy_*` calls.

## 2026-11-18: Validation pass (P4.4)

- **Fixed validation set:**
  - **What it is:** §9 says "first `val_batches` chunks", but the key is
    named for batches. It is read as the first `val_batches` *batches*:
    chunks `[0, val_batches·gB)` of `val_data` in file order, unshuffled.
  - **Batching:** batch i is chunks `[i·gB, (i+1)·gB)`, and rank r takes
    rows `[r·B, (r+1)·B)`, as in training (§4.5). The same tokens are used
    at every evaluation and in every build.
  - **Loader:** `DataLoader::fixed_batch` reads it without touching the
    shuffle state. A file too small for `val_batches` aborts at startup.
- **Loss:** `val_loss` = (sum of batch-mean losses over batches and ranks) /
  (val_batches · world_size). This equals the mean over all positions,
  because every rank-batch has B·T positions. Sums are in double, combined
  with `comm_allreduce_sum_double`. `val_ppl` = exp(`val_loss`).
- **Cadence:**
  - Validation runs every `eval_every` steps and at `max_steps`, after the
    step's timing, so `step_ms` excludes it (as with checkpoints).
  - Validation steps always get a `log.csv` row; `val_*` stay empty on
    other rows.
  - A fresh run also evaluates once before step 1 and prints the result to
    the console only. There is no step-0 row, because `train_loss` has no
    value there and the columns are fixed.
  - It is forward only and touches no params, grads or optimizer state.
    Tests show the training columns are identical whether validation runs
    every step or only at the end.
- **Measured:**
  - Fresh V = 10000 model on synthetic data (`tests/test_eval.cpp`, seeds 1,
    2, 1337): 9.21337, 9.21537, 9.20814 vs ln(10000) = 9.21034; the test
    bound is ±0.05.
  - `bin/train_serial --config configs/smoke.cfg` on real `data/val.bin`
    (8 batches, 1024 tokens): 9.2164 at init, then 7.8005 / 6.5740 /
    6.2122 / 6.1346 at steps 50 / 100 / 150 / 200. The train-loss column is
    unchanged from P4.2 (6.2963 at step 200).
- **Mutation evidence:** the tests failed on each of five planted bugs:
  - wrong mean divisor;
  - missing rank offset in `fixed_batch`;
  - `val_ppl` not exponentiated;
  - no validation at the last step;
  - every validation batch reading batch 0.
- **Impact:** the CUDA and MPI builds use the same loop and loader, so the
  val loss is comparable across builds (the §10 cross-version check: within
  1% after 100 steps).

## 2026-11-18: Overfit and smoke tests; smoke preset tuned; 100k fixture (P4.5), approved

- **Change:**
  - **`tests/test_smoke.cpp`** runs both §10 rows in `make test`, using
    `configs/smoke.cfg` loaded from disk. Only data paths and the output
    dir are overridden, plus steps and cadence for the overfit test.
  - **Smoke data:** the §10 "~1M-token subset" is replaced by a committed
    real-data fixture (approved), `tests/fixtures/smoke_train.bin` and
    `smoke_val.bin` (208 KB):
    - train: `prepare_data --max_tokens 100000`, 99,958 tokens, 522 stories;
    - val: the first 4,097 tokens of the matching `val.bin`, i.e. exactly
      the 32 fixed validation batches.

    It is regenerated byte-identically by the command in
    `tests/fixtures/README.md`. `dev_only/truncate_tokens.py` cuts the val
    file, offline only.
  - **ROADMAP §10** smoke row amended; the TinyStories THIRD_PARTY row
    records the redistributed excerpt (CDLA-Sharing-1.0).
  - **`smoke.cfg`:** `lr` 1e-3 → 1e-2 and `val_batches` 8 → 32. Its comment
    had marked these as placeholders to tune in P4.5.
- **Reason:**
  - CLAUDE.md §6 forbids tests that need the dataset, and a 1M subset
    (2 MB of tokens) is too large to commit.
  - 200 steps × 4 × 32 reads only 25,600 tokens, so 100k is never repeated
    within an epoch, and the subset is still real text.
  - At lr 1e-3 neither §10 criterion holds (measured below). 32 validation
    batches (4,096 tokens) make the val estimate less noisy at negligible
    cost.
- **Criteria as implemented:**
  - **Overfit:** a token file with exactly one global batch (the first 4
    chunks of the fixture, written at test time), 300 steps, train loss
    < 0.1.
  - **Smoke:** `val_loss` at steps 50/100/150/200 decreases "coarsely",
    meaning each may exceed the previous by at most 0.05; it must fall by
    more than 0.3 overall and end below 6.0. Wall time must be under 60 s,
    asserted in release builds only (the -O0 sanitizer build is not the
    timed configuration; the time is printed in all builds).
- **Measured** (release float, this container):

  | lr | smoke val at 50 / 100 / 150 / 200 (100k fixture) | overfit loss at 300 |
  |---|---|---|
  | 1e-3 | 7.85 / 6.60 / 6.21 / 6.13 (1M subset, 8 val batches) | 1.9766 |
  | 3e-3 | 6.19 / 6.12 / 6.07 / 6.04 (1M subset, 8 val batches) | 0.0061 |
  | 1e-2 | 6.2771 / 6.0214 / 5.7996 / 5.7444 | 0.0002 |

  - At lr 1e-2 on a real 1M subset, val ended at 5.76 (8 val batches), so
    the 100k fixture is representative.
  - `make test` times at lr 1e-2: smoke 6.4 s, overfit 7.6 s.
- **Mutation evidence:**
  - Restoring lr 1e-3 fails both tests.
  - Disabling gradient clipping does not fail them: the smoke run barely
    clips, and clipping is covered exactly by `test_optim`, the golden AdamW
    test and the reference-loop test.
- **Sanitizer build:**
  - A debug (-O0 + ASan/UBSan) step takes about 1.2 s; the 300-step overfit
    run alone took 377 s. So `make debug` runs a 10-step version of both
    tests (validation and checkpoint included) that checks only finite,
    decreasing losses, and the test prints that it did.
  - The §10 criteria are asserted in the release and `REAL=double` builds.
    Double measured: overfit 0.0002; smoke val 6.2771 / 6.0214 / 5.7999 /
    5.7434.
  - Times: `make test` 21 s, `make debug` 95 s, `make test REAL=double` 32 s.
- **`tests/test_config.cpp`** pins each preset's lr. The smoke value was
  updated 1e-3 → 1e-2 together with the preset; it is a deliberate retune,
  not a weakened test.
- **Full dataset:** `bin/train_serial --config configs/smoke.cfg` on the
  full `data/` now ends at val 5.7001 (step 200).
- **Impact:** the smoke preset is also benchmark workload W-C (§12). Its lr
  changes the numbers, not the cost per step.

## 2026-11-18: Sign-off tests for determinism, causality, resume, param counts (P5.1)

- **Coverage map** (the four P5.1 layers of ROADMAP §10, all in `make test`):

  | Layer | §10 criterion | Test |
  |---|---|---|
  | Param count | each preset matches §2.2 | `test_layout.cpp` `param_counts_match_roadmap` (base/small tied and untied exactly; smoke from the formula), since P0.4 |
  | Determinism | 2 runs × 20 steps: log losses bit-identical, final param checksum identical | `test_signoff.cpp` `determinism_two_runs` (new) |
  | Resume | 20 = 10 + checkpoint + 10, bit-identical params | `test_signoff.cpp` `resume_matches_uninterrupted` (new, smoke scale); `test_checkpoint.cpp` (tiny model, both directories) |
  | Causality | perturb token t; logits at positions < t unchanged | `test_signoff.cpp` `causality_full_model` (new, full model); `test_attention.cpp` (op level) |

- **Determinism and resume:** both run the `smoke` preset on the committed
  real-data fixture with `log_every=1`.
  - **Determinism:** every numeric `log.csv` column (step, tokens,
    train/val loss, ppl, lr, grad norm) must match. The final checkpoint
    must be byte-identical, and the FNV-1a of the final params is printed.
  - **Second run:** it runs in a forked child whose heap is dirtied first:
    2,000 blocks filled with 0xA5, every other one freed.
- **Why dirty the heap:** a plain `fork()` copies the parent's heap, so the
  child allocates at the same addresses as the parent. A planted bug that
  scaled lr by a buffer address passed without the dirtying and failed with
  it (measured). This catches address dependence and reads of reused small
  heap blocks. Uninitialized reads of fresh `mmap` pages (always zero) are
  not caught; that would need MemorySanitizer, which is not in the
  toolchain.
- **Causality:** a 2-layer model, tied and untied, perturbing batch row 0 at
  t ∈ {0, 1, 7, 15}.
  - Logits at positions < t must be bit-identical (`memcmp`).
  - Position t must change.
  - Batch row 1 must be bit-identical.
  - A planted one-token look-ahead in attention forward fails it.
- **Debug build:** determinism runs 2 × 6 steps and resume runs 6 = 3 + 3
  (a smoke step takes ~1.2 s under -O0 + sanitizers); causality is
  unchanged.
- **Measured** (this container): `test_signoff` takes 2.2 s in release.
  Final params FNV-1a, identical in both runs of each build:

  | Build | FNV-1a |
  |---|---|
  | release float | `0d80323551e3bb51` |
  | double | `a212a0f0162c37b2` |
  | debug, 6 steps | `3f1add97f2972ba6` |

  - The release value was reproduced from a from-scratch rebuild of
    `build/release-cpu-serial-float` and by three separate executions.
  - The very first test binary printed `3a787b8b4853b93d` (both runs
    agreeing) before a later rebuild. Its linked objects cannot be
    reconstructed, so that value is not used.
  - The checksum is a per-binary reference: different compilers or flags may
    legitimately round differently. The test requires only run-to-run
    equality.
- **Full suite:** `make test` 22 s, `make debug` 122 s,
  `make test REAL=double` 31 s; 27/27 in each.
- **Impact:** these tests are the §10 reference that Phases B–D must pass
  unchanged on their own builds.

## 2026-11-18: P5.2 run setup; run executed off-container (approved)

- **Change:**
  - **`configs/small.cfg` run values** (they had been placeholders "set in
    P5.2"):
    - `max_steps` 5000 → 2400: one epoch of the 20M-token subset is 2,441
      steps, so no sequence repeats.
    - `eval_every` 250 → 100.
    - `ckpt_every` 1000 → 250.

    Model and optimizer values are unchanged.
  - **Data:** `prepare_data --max_tokens 20000000` gives 19,999,905 tokens
    and 93,024 stories, with train coverage 99.8957%, val coverage 99.8648%
    and vocab hash `4de4b6c3947630ff`.
  - **`dev_only/plot_log.py`:** draws SVG curves from `log.csv` using the
    standard library only, so no plotting dependency (e.g. matplotlib) is
    added.
  - **`docs/RUNS.md`:** exact commands, a resume recipe, and the plateau
    criterion, fixed before the run.
- **Where it runs (approved):** the full run executes on a teammate machine
  or the cluster, not in this cloud container.
  - Measured here: 3.8 s/step warm, which puts the run at about 2.7 h
    including validation.
  - This container is reclaimed when idle and caps one background command
    at 2 h.
  - The returned `log.csv` is then plotted and assessed against the
    criterion. Results are recorded only from that file.
- **Pilot (here):** the production command shortened to 20 steps.
  - Loss 9.2461 → 7.4608; val 9.2382 at init, then 7.9299 / 7.4273.
  - Resuming in place from step 10 reproduced `ckpt_20.bin` byte for byte.
  - It validates the pipeline only.
- **Impact:** none on code or formats. The `small` preset is also benchmark
  workload W-A, which uses fixed step counts, so `max_steps` there is
  overridden.

## 2026-11-18: Generation app and sampling rules (P6.1)

- **Library:** `include/tslm/generate.h`, `src/train/generate.cpp`.
  - `sample_token` is host-side (§4.8). Its rules:
    - `<unk>` is excluded unless `--allow_unk 1`, since it decodes to no
      text.
    - Greedy takes the argmax, with ties going to the lowest id.
    - Otherwise: top-k (ties ranked by lower id), then
      softmax(logit / temperature) in double, accumulated in id order, and
      one `Rng::uniform()` draw.
  - `generate_ids` runs `model_forward` with B = 1 over a sliding window of
    the last `max_seq_len` tokens (no KV cache, a Phase A non-goal). It
    copies the last row of logits to the host and stops at EOS (not
    returned) or `max_new`.
  - Randomness uses only `Rng(seed, SAMPLING_STREAM = 1000000)`, a stream
    separate from init (0..#tensors) and data order (100 + epoch), so a run
    with the same options and seed is reproducible.
- **App:** `apps/generate.cpp`.
  - The model shape comes from the checkpoint header.
  - The prompt is encoded by our BPE, mapped to vocab ids (unknown → `<unk>`
    with a warning) and prefixed with EOS, which precedes every story in the
    training stream.
  - Output is the prompt plus the decoded continuation.
- **Known gap:** nothing ties a checkpoint to its `vocab_map.txt`; only the
  size is checked. A mismatched map would decode wrong words silently. A
  real fix (a vocab hash in the checkpoint or manifest) would be a format
  change and needs the contract process. For now it is documented in the
  app header and README.
- **DoD status:**
  - **Reproducible greedy output:** met and tested. Two runs give identical
    tokens, and each token equals the argmax of a direct forward over the
    window, including a context longer than `max_seq_len`.
  - **Grammatical continuations on `small`:** pending the P5.2 checkpoint.
    The smoke model (333k params, 200 steps) yields word salad, as
    expected.
- **Measured** (`tests/test_generate.cpp`): sampled frequencies over 200k
  draws match softmax(l/0.7) within 5 standard errors.
- **Mutation evidence:** the tests failed on each of four planted bugs:
  - a window over the first tokens instead of the last;
  - greedy ties going to the higher id;
  - temperature multiplied instead of divided;
  - EOS returned.

## 2026-11-18: Correction: release-float numbers recorded from a stale build (P4.5, P5.1)

- **What happened:**
  - Between P4.4 and P5.1, the incremental release-float build linked an
    object file that did not match its committed source.
  - The most likely cause: one of the mutation checks (plant a bug, build,
    restore the file with `cp`) left an object compiled from a mutated
    variant that `make` did not rebuild.
  - Its numerics differed from a fresh build only in the last bits. It
    passed every tolerance-based test (gradcheck, golden, smoke criteria),
    so nothing flagged it.
  - It disappeared in P5.1, when a mutation check forced
    `src/ops/attention_cpu.cpp` to recompile. That is consistent with the
    stale object being `attention_cpu.o`, but this was not proven.
- **Evidence (2026-11-18):**
  - The P4.5 commit (`a9e214c`) built from scratch in a separate worktree
    gives the same smoke result as today's build: train 6.1013, val 5.6889.
    The training code has not changed since P4.4, and the current binary
    reproduces itself byte for byte.
  - A from-scratch `build/release-cpu-serial-float` gives the values in the
    table below.
  - The lr 1e-3 smoke numbers recorded in P4.2–P4.4 (train 6.2963, val
    6.1346) reproduce exactly. The difference only shows at lr 1e-2, where
    small rounding differences are amplified over 200 steps.
- **Corrected values** (release float; every other recorded number was
  checked and still holds):

  | Record | Recorded | From a clean build |
  |---|---|---|
  | P4.5: `train_serial --config configs/smoke.cfg` on full `data/`, step 200 | train 6.0728, val 5.7001 | train 6.1013, val 5.6889 |
  | P4.5: `test_smoke` val at step 200 (fixture) | 5.7444 | 5.7445 (steps 50/100/150 unchanged: 6.2771 / 6.0214 / 5.7996) |
  | P5.1: first `test_signoff` final-params FNV-1a | `3a787b8b4853b93d` (already marked unusable) | `0d80323551e3bb51` (as recorded later in P5.1) |

  - All §10 criteria still pass with the same margins.
  - Unaffected and re-verified from scratch builds today:
    - double: smoke val 5.7434, overfit 0.0002, FNV `a212a0f0162c37b2`;
    - debug: FNV `3f1add97f2972ba6`;
    - release overfit: 0.0002.
  - `docs/STATUS.md` quotes "val loss 5.70" for the full-data smoke run. It
    will read 5.69 at its next requested update; STATUS is updated only on
    request.
- **Procedure from now on:**
  1. After a mutation check, restore the file and then `touch` it, so
     `make` always recompiles the restored source.
  2. Before recording any number in DECISIONS, RUNS or a CSV, rebuild the
     configuration that produced it from scratch (`rm -rf build/<tag>`, or
     `make clean`), and record the number from that build.
- **Impact:** no code change. The earlier entries stay as written
  (append-only log); this entry supersedes their affected numbers.

## 2026-11-18: Fixed prompt set, eval app and samples file layout (P6.2)

- **`eval/prompts.txt`:** 10 prompts written for this project in the style
  of the TinyStories paper examples (none copied):
  - 3 story openers;
  - 3 factual (fridge, night sky, rain);
  - 2 reasoning (counting, melting);
  - 2 consistency (who has which ball; dad helping with a stuck kite).

  The format is `<id> TAB <category> TAB <text>` with `#` comments, chosen
  so the text can contain quotes and commas. The file is frozen: its
  FNV-1a 64 is written into every samples file.
- **`bin/eval`:** for every prompt, one greedy and one temperature-0.8
  completion with seed 1337, at most 120 tokens, stopping at EOS.
  - These settings are compile-time constants, not options, so every run
    and build uses the same ones (the DoD's "stable across versions").
  - Output goes to `samples_<step>.txt` next to the checkpoint (§9 run
    outputs).
- **Layout:** fixed by `include/tslm/samples.h` and pinned by an exact
  expected string in `tests/test_samples.cpp`.
  - A 6-line `#` header: format version, checkpoint and step, model, build,
    prompts path/count/hash, settings.
  - Then per prompt a `=== prompt <id> [<category>] ===` block with greedy
    and sampled sections, each showing the prompt plus continuation.
  - There is no date or timing, so the file is byte-reproducible; the
    smoke checkpoint gave identical files on two runs.
  - Changes to the layout go through the contract process.
- **Shared code:** prompt encoding and decoding moved from
  `apps/generate.cpp` into `TextCodec` (`include/tslm/text_codec.h`) so
  `generate` and `eval` encode prompts identically. `generate` output is
  unchanged (checked on the smoke checkpoint).
- **Not yet meaningful:** the content of the completions. It needs the P5.2
  `small` checkpoint, since the smoke model produces newlines and word
  salad. Scoring is P6.3.
- **Mutation evidence:** the tests failed on each of two planted bugs: a
  dropped prompt-id ordering check, and a dropped "stopped at EOS" marker.

## 2026-11-18: Scoring template and rubric (P6.3); pilot pending

- **Change:**
  - **`eval/scores.csv`:** a header-only template with the §6 P6.3 columns,
    `milestone,prompt_id,grammar,consistency,sense,rater,notes`. The header
    is pinned by `tests/test_samples.cpp` (D6: format stays stable).
  - **`eval/RUBRIC.md`:**
    - anchored 0–5 scales for grammar, consistency and sense;
    - scoring rules and illustrative examples (written for the rubric, not
      model output);
    - a milestone naming convention (`<version>-<preset>-step<step>`);
    - a pilot procedure.
- **Which completion is scored:** the temperature-0.8 one. The CSV has no
  greedy/sampled column, and the specified format was kept rather than
  extended. Greedy completions of small models tend to loop, which would
  mostly measure the decoding rule; they remain in the samples files for
  build comparisons.
- **Pilot criterion:** the DoD says "agree within ±1". It is made concrete
  as: at least 27 of the 30 score pairs (10 prompts × 3 dimensions) differ
  by ≤ 1, and none by more than 2. On failure, the raters refine the
  anchors and re-pilot on a different samples file.
- **DoD status: pending.** The pilot needs two human raters and samples from
  a trained checkpoint (the P5.2 `small` run, still to be executed). No
  scores have been entered, and none were produced by the assistant.
