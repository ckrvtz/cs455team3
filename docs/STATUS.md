# Project status

**Last updated:** 2026-11-18, through ROADMAP task **P5.1**, with P5.2 set up (commit `1a2f6a5`).
This snapshot is updated on request, not after every task. For the live record
of decisions see [`DECISIONS.md`](DECISIONS.md); for build and run
instructions see [`../README.md`](../README.md).

Phases 0–4 and P5.1 of the serial build are done. The data pipeline, the
forward and backward passes, the AdamW optimizer, and the training loop with
validation and checkpoint/resume all work. They are checked by 27 test
binaries in three build modes (release, sanitizer debug, double precision),
including every ROADMAP §10 sign-off layer: gradient check, PyTorch golden
files, overfit, smoke training, determinism, resume, causality and parameter
counts. The first real `small` run (P5.2) is prepared and pilot-tested; it
runs off-container and its results are pending. Generation and benchmarking
are not built yet.

## Build and project docs
- **`Makefile`:** builds the four modes through `BACKEND`/`COMM`/`REAL`; CUDA and
  MPI currently stop with a clear error. Targets: `make`, `make test`,
  `make debug` (sanitizers), `make test REAL=double`, `make dist`, `make clean`.
- **`README.md`:** how to build, test, get the data, train and run the tools.
- **`docs/DECISIONS.md`:** every design decision, with its reason.
- **`docs/PROFILE.md`:** every serial optimization.
- **`docs/RUNS.md`:** commands, cost estimate and plateau criterion for the
  long P5.2 run; its results go here.
- **`THIRD_PARTY.md`:** citations (RNG algorithms, GPT-2 BPE scheme, dataset,
  tokenizer files, HF `tokenizers`).
- **`CLAUDE.md` / `ROADMAP.md`:** the working agreement and the technical plan,
  including the keep-the-README-current rule.

## Core infrastructure (`include/tslm/`, `src/common/`, `src/backend/`, `src/comm/`)
- **`build_config.h`, `common.h`, `check.h`:** the build mode, the `real_t` type
  (float, or double for gradient checks), and the `CHECK` error macro.
- **`config.h` + `configs/*.cfg`:** reads and validates `key=value` settings files
  and `--key value` options (including `resume`), plus the
  `smoke`/`small`/`base` presets.
- **`rng.h`:** our own random number generator.
- **`timer.h`:** times each training-step phase.
- **`logging.h/.cpp`:** console messages and the fixed-column `log.csv` writer.
- **`backend.h` / `backend_cpu.cpp`:** the memory layer. All model memory goes
  through it, so the CUDA build can swap in GPU memory later.
- **`comm.h` / `comm_serial.cpp`:** the communication layer, which does nothing
  in the serial build; MPI plugs in later.
- **`tensor_file.h/.cpp`:** a small file format for golden test data.

## Data pipeline (`src/data/`, `apps/`, `scripts/`)
- **`scripts/download_data.sh`:** downloads TinyStories and the GPT-Neo tokenizer
  files at pinned versions, with checksums.
- **`tokenizer.h/.cpp`:** our own GPT-2/GPT-Neo tokenizer. It matches Hugging
  Face exactly on 31M+ real tokens.
- **`prepare_data`** (app + `prepare.cpp`, `token_file.cpp`): turns the text into
  `train.bin`/`val.bin` with a 10,000-token vocabulary. Coverage is 99.88%, and
  reruns give byte-identical files.
- **`dataset.h/.cpp`:** the training data loader. It shuffles with a seed, can
  split batches across processes, and can resume exactly.
- **`inspect_data`** (app + `vocab_map.cpp`): prints decoded training samples.

## Model (`src/ops/`, `src/model/`)
- **`ops.h`:** the contract every numeric function follows, shared with the
  future CUDA version. The CPU implementations, all with forward and backward:
  - embedding, LayerNorm, GELU, residual;
  - matmul: cache-blocked and vectorized;
  - attention: causal lower triangle only, three stages;
  - softmax + cross-entropy and the loss sum;
  - optimizer ops: sum of squares, scale, AdamW step.
- **`layout.h/.cpp`:** places all parameters and activations in single flat
  buffers. Parameter counts match the spec exactly.
- **`model.h/.cpp`:** creates the model, initializes its weights, and runs the
  forward and backward passes (the tied embedding gets both gradient
  contributions). A fresh `base` model on real data has initial loss 9.26 vs
  ln(10000) = 9.21.
- **`dump_init`** (app): writes init weights and a batch for golden tests.
- **`gradcheck`** (app + `src/train/gradcheck.cpp`): finite-difference check of
  the whole backward pass, double build only. Worst relative error 3.1e-8
  (tied) at the default seed; 10/10 seeds pass tied and untied.

## Optimizer and training (`src/optim/`, `src/train/`, `apps/train.cpp`)
- **`optim.h/.cpp`:** learning-rate schedule (warmup, then cosine), global-norm
  gradient clipping, and AdamW with weight decay on 2-D weights only. Matches
  a 5-step PyTorch AdamW run (worst parameter error 1.7e-5 in float).
- **`train.h/.cpp` + `train_serial`** (app): the training loop with per-phase
  timing, NaN/Inf checks and `log.csv`. The smoke preset (lr tuned to 1e-2 in
  P4.5) runs 200 steps on the full data and ends at val loss 5.70.
- **`eval.h/.cpp`:** validation loss and perplexity on fixed validation
  batches, every `eval_every` steps; a fresh model scores ≈ ln(10000).
- **`manifest.cpp`:** writes each run's `manifest.txt` (git hash, build,
  machine, full config); it can be reloaded with `--config` to rerun.
- **`checkpoint.h/.cpp`:** checkpoint save/load (weights, optimizer state,
  data position) with a checksum; `--resume` continues a run bit-exactly.

## Tests (`tests/`)
- **`test.h`:** our own test harness.
- **27 test files** covering: build mode, config, RNG, memory/communication/
  timer, layout, tokenizer, `prepare_data`, data loader, vocab map, forward
  ops, matmul, attention, softmax, model, golden tests, backward ops (matmul,
  attention, softmax, model), gradient check, optimizer, training loop,
  checkpoints, validation, overfit + smoke training, and the sign-off tests
  (determinism across processes, resume, full-model causality).
- **`tests/fixtures/`:** a 208 KB excerpt of real tokenized TinyStories
  (100k train + 4,097 val tokens) so the smoke tests need no download.
- **`tests/golden/`:** small committed reference data: tokenizer cases, a subset
  tokenizer vocabulary, and the PyTorch float64 init weights, forward results,
  gradients and 5-step AdamW trajectories.

## Offline tools (`dev_only/`, never built or shipped)
- **`gen_tokenizer_golden.py`:** generates the tokenizer reference cases.
- **`gen_golden.py`:** the float64 PyTorch reference model (loss, gradients,
  AdamW trajectory).
- **`truncate_tokens.py`:** cuts the validation fixture from `val.bin`.
- **`plot_log.py`:** train/val loss curves from `log.csv` as SVG (standard
  library only).

## Next
Finish P5.2: run the 2,400-step `small` job per `docs/RUNS.md` (estimated
≈2.7 h from a measured 3.8 s/step pilot), then plot and assess its `log.csv`
against the pre-set plateau criterion. Phase 6 (generation, P6.1) can start
meanwhile; P5.3 (Serial Complete gate) follows both. Known item for P7.1:
benchmarks must pre-warm the token file, because cold-cache reads dominate
`data_ms` (smoke run 54.5 s cold vs 5.0 s warm).
