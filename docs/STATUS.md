# Project status

**Last updated:** 2026-11-18, through ROADMAP task **P3.1** (commit `ec009b0`).
This snapshot is updated on request, not after every task. For the live record
of decisions see [`DECISIONS.md`](DECISIONS.md); for build and run
instructions see [`../README.md`](../README.md).

Phases 0–2 of the serial build and the first Phase 3 task are done. The data
pipeline, the full forward pass and the first backward ops all work, and they
are checked by 16 test binaries in three build modes (release, sanitizer
debug, double precision). The remaining backward ops, the optimizer,
training, generation and benchmarking are not built yet.

## Build and project docs
- **`Makefile`:** builds the four modes through `BACKEND`/`COMM`/`REAL`; CUDA and
  MPI currently stop with a clear error. Targets: `make`, `make test`,
  `make debug` (sanitizers), `make test REAL=double`, `make dist`, `make clean`.
- **`README.md`:** how to build, test, get the data and run the tools.
- **`docs/DECISIONS.md`:** every design decision, with its reason.
- **`docs/PROFILE.md`:** every serial optimization.
- **`THIRD_PARTY.md`:** citations (RNG algorithms, GPT-2 BPE scheme, dataset,
  tokenizer files, HF `tokenizers`).
- **`CLAUDE.md` / `ROADMAP.md`:** the working agreement and the technical plan,
  including the keep-the-README-current rule.

## Core infrastructure (`include/tslm/`, `src/common/`, `src/backend/`, `src/comm/`)
- **`build_config.h`, `common.h`, `check.h`:** the build mode, the `real_t` type
  (float, or double for gradient checks), and the `CHECK` error macro.
- **`config.h` + `configs/*.cfg`:** reads and validates `key=value` settings files
  and `--key value` options, plus the `smoke`/`small`/`base` presets.
- **`rng.h`:** our own random number generator.
- **`timer.h`:** times each training-step phase.
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
  future CUDA version. The CPU implementations:
  - embedding, LayerNorm, GELU, residual: forward and backward;
  - matmul: forward only, cache-blocked and vectorized;
  - attention: forward only, causal, three stages;
  - softmax + cross-entropy and the loss sum: forward only.
- **`layout.h/.cpp`:** places all parameters and activations in single flat
  buffers. Parameter counts match the spec exactly.
- **`model.h/.cpp`:** creates the model, initializes its weights, and runs the
  forward pass. A fresh `base` model on real data has initial loss 9.26 vs
  ln(10000) = 9.21.
- **`dump_init`** (app): writes init weights and a batch for golden tests.

## Tests (`tests/`)
- **`test.h`:** our own test harness.
- **16 test files** covering: build mode, config, RNG, memory/communication/
  timer, layout, tokenizer, `prepare_data`, data loader, vocab map, forward
  ops, matmul, attention, softmax, model, golden tests, backward ops.
- **`tests/golden/`:** small committed reference data: tokenizer cases, a subset
  tokenizer vocabulary, and the PyTorch float64 init and forward results.

## Offline tools (`dev_only/`, never built or shipped)
- **`gen_tokenizer_golden.py`:** generates the tokenizer reference cases.
- **`gen_golden.py`:** the float64 PyTorch reference model.

## Next
P3.2 (matmul backward), then attention and softmax backward, model backward,
the full gradient check and the backward golden tests (P3.3–P3.7). After that
come the optimizer and the training loop (Phase 4).
