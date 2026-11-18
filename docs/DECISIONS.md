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

## 2026-11-18: Model API types (P2.5), needs team confirmation

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
