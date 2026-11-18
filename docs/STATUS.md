# Project status

**Last updated:** 2026-11-18, through ROADMAP task **P7.5**, plus the P5.2
results and verdict (commit `306f4cf`). This snapshot is updated on request,
not after every task.
- The live record of decisions: [`DECISIONS.md`](DECISIONS.md).
- Build and run instructions: [`../README.md`](../README.md).
- The next phase: [`../ROADMAP_CUDA.md`](../ROADMAP_CUDA.md).

**Summary:** the serial build (Phase A) is complete except for two review
items.
- **Working:** the data pipeline, forward and backward passes, AdamW,
  training with validation and checkpoint/resume, generation, prompt
  evaluation, benchmarking and profiling.
- **Tests:** 30 test programs (154 test cases) in three build modes
  (release, sanitizer debug, double precision), covering every ROADMAP §10
  sign-off layer.
- **First real `small` run (P5.2):** done. Val loss 2.6455 (ppl 14.1) after
  2,400 steps on 20M tokens; plateau accepted by the team.
- **README:** verified command by command from a clean clone.
- **Still open:** the two-rater scoring pilot (P6.3) and the Serial
  Complete review (P5.3).
- **Next:** the CUDA build (Phase B), planned in `ROADMAP_CUDA.md`.

## Build and project docs
- **`Makefile`:** builds the four modes through `BACKEND`/`COMM`/`REAL`.
  - Targets: `make`, `make test`, `make debug` (sanitizers),
    `make test REAL=double`, `make dist`, `make clean`, and the dev-only
    `make MODE=profile` (gprof).
  - Objects rebuild when the compiler, flags or CPU target change (target
    stamp).
  - CUDA and MPI stop with one clear message: the missing tool, or the
    stub.
- **`README.md`:** quick start (138 s of machine time from clone to an
  evaluated smoke model), reproduction recipes (smoke run, `small` run,
  benchmarks), the reference machine, and every command, all run from a
  clean clone.
- **`docs/DECISIONS.md`:** every design decision with its reason, including
  the corrections of numbers recorded from a stale build.
- **`docs/PROFILE.md`:** every serial optimization and the measured profile.
- **`docs/RUNS.md`:** the P5.2 run: commands, plateau criterion, results,
  the supplementary gap measurement and the generation check.
- **`docs/HANDOFF.md`:** what each `ops.h` kernel must preserve in CUDA, and
  device memory per preset.
- **`ROADMAP_CUDA.md`:** the plan for Phase B (single-GPU CUDA), extensible
  to MPI+CUDA.
- **`docs/proposals/CLAUDE_md_phase_B.md`:** proposed CLAUDE.md changes for
  Phase B, pending team decisions (not applied).
- **`THIRD_PARTY.md`:** citations, including the redistributed TinyStories
  test excerpt.
- **`CLAUDE.md` / `ROADMAP.md`:** the working agreement and the technical
  plan.

## Core infrastructure (`include/tslm/`, `src/common/`, `src/backend/`, `src/comm/`)
- **`build_config.h`, `common.h`, `check.h`:** the build mode, the `real_t`
  type (float, or double for gradient checks), and the `CHECK` error macro.
- **`config.h` + `configs/*.cfg`:** reads and validates `key=value` settings
  and `--key value` options (including `resume`), plus the
  `smoke`/`small`/`base` presets.
  - `smoke` was tuned to lr 1e-2 in P4.5.
  - `small` has the P5.2 run values (2,400 steps).
- **`rng.h`:** our own random number generator.
- **`timer.h`:** times each training-step phase.
- **`logging.h/.cpp`:** console messages and the fixed-column `log.csv`
  writer.
- **`backend.h` / `backend_cpu.cpp`:** the memory layer. All model memory
  goes through it, so the CUDA build can swap in GPU memory.
- **`comm.h` / `comm_serial.cpp`:** the communication layer, which does
  nothing in the serial build. `comm_mpi.cpp` is a stub for Phase C.
- **`tensor_file.h/.cpp`:** a small file format for golden test data.

## Data pipeline (`src/data/`, `apps/`, `scripts/`)
- **`scripts/download_data.sh`:** downloads TinyStories and the GPT-Neo
  tokenizer files at pinned versions, with checksums.
- **`tokenizer.h/.cpp`:** our own GPT-2/GPT-Neo tokenizer. It matches Hugging
  Face exactly on 31M+ real tokens.
- **`prepare_data`** (app + `prepare.cpp`, `token_file.cpp`): turns the text
  into `train.bin`/`val.bin` with a 10,000-token vocabulary. Coverage is
  99.88%, and reruns give byte-identical files.
- **`dataset.h/.cpp`:** the training data loader. It shuffles with a seed,
  can split batches across processes, resumes exactly, and reads fixed
  file-order batches for validation.
- **`text_codec.h/.cpp`:** prompt encoding and decoding, shared by
  generation and evaluation.
- **`inspect_data`** (app + `vocab_map.cpp`): prints decoded training
  samples.

## Model (`src/ops/`, `src/model/`)
- **`ops.h`:** the contract every numeric function follows, shared with the
  CUDA version. The CPU implementations, all with forward and backward:
  - embedding, LayerNorm, GELU, residual;
  - matmul: cache-blocked and vectorized;
  - attention: causal lower triangle only, three stages;
  - softmax + cross-entropy and the loss sum;
  - optimizer ops.

  `ops_cuda.cu` is a stub for Phase B.
- **`layout.h/.cpp`:** places all parameters and activations in single flat
  buffers. Parameter counts match the spec exactly.
- **`model.h/.cpp`:** creates the model, initializes its weights, and runs
  the forward and backward passes. A fresh `base` model on real data has
  initial loss 9.26 vs ln(10000) = 9.21.
- **`dump_init`** (app): writes init weights and a batch for golden tests.
- **`gradcheck`** (app + `src/train/gradcheck.cpp`): finite-difference check
  of the whole backward pass, double build only. Worst relative error
  3.1e-8 (tied) at the default seed; 10/10 seeds pass tied and untied.

## Optimizer and training (`src/optim/`, `src/train/`, `apps/`)
- **`optim.h/.cpp`:** learning-rate schedule (warmup, then cosine),
  global-norm gradient clipping, and AdamW with weight decay on 2-D weights
  only. Matches a 5-step PyTorch AdamW run (worst parameter error 1.7e-5 in
  float).
- **`train.h/.cpp` + `train_serial`:** the training loop.
  - One shared `train_step()` (also used by `bench`), per-phase timing,
    NaN/Inf checks and `log.csv`.
  - The smoke preset runs 200 steps on the full data and ends at val loss
    5.69 (corrected from 5.70; DECISIONS).
- **`eval.h/.cpp`:** validation loss and perplexity on fixed validation
  batches; a fresh model scores ≈ ln(10000).
- **`manifest.cpp`:** each run's `manifest.txt` (git hash, build, machine,
  full config); it can be reloaded with `--config` to rerun.
- **`checkpoint.h/.cpp`:** checkpoint save/load (weights, optimizer state,
  data position) with a checksum; `--resume` continues a run bit-exactly.
- **`eval_loss`** (app): the loss of a checkpoint on fixed batches of any
  token file. It reproduces a run's val loss exactly.

## Generation, evaluation, benchmarking
- **`generate.h/.cpp` + `generate`** (app): greedy or temperature/top-k
  sampling with a sliding context window and its own random stream. Greedy
  output is reproducible.
- **`samples.h/.cpp` + `eval`** (app): the 10 fixed prompts in
  `eval/prompts.txt`, one greedy and one temperature-0.8 completion each,
  written to `samples_<step>.txt` in a fixed, byte-reproducible layout.
- **`eval/RUBRIC.md` + `eval/scores.csv`:** the 0–5 scoring rubric
  (grammar, consistency, sense), the pilot procedure, and the empty scores
  template.
- **`bench.h/.cpp` + `bench`** (app) + **`scripts/bench.sh`:** one CSV row per
  fixed-workload run (ROADMAP §12).
  - It reads the token file once first, so the page cache is warm.
  - The first 3 steps are discarded.
  - Serial results (Xeon @ 2.30GHz with AVX-512, 1 core, 3 runs each,
    within ±1.2% of their mean):

    | Workload | Median step | Tokens/s | File |
    |---|---|---|---|
    | W-A (small, B=32) | 3,610 ms | 2.25k–2.30k | `bench/serial_W-A.csv` |
    | W-C (smoke) | 20.1 ms | 6.3k–6.4k | `bench/serial_W-C.csv` |
- **`scripts/profile.sh`:** gprof profile of the W-A step
  (`docs/profile/serial_W-A_gprof.txt`).
  - Matmul is 81.4% of sampled time; backward is 60.9% of step time.
  - Analysis in `docs/PROFILE.md`.

## Runs
- **P5.2 `small` run** (`docs/runs/small_p52/`: `log.csv`, `manifest.txt`,
  `curves.svg`, `samples_2400.txt`):
  - Run on a teammate's i7-6820HQ: 2,400 steps on 20M tokens, val loss
    5.9065 → 2.6455 (ppl 14.1). Its first 260 steps are bit-identical to a
    run on the reference Xeon.
  - Plateau criteria: 1 and 3 met; criterion 2 missed. On 100 batches each,
    train and val agree within 0.015. The plateau was accepted by the team.
  - Generation from its checkpoint is mostly grammatical TinyStories-style
    text.

## Tests (`tests/`)
- **`test.h`:** our own test harness.
- **30 test files (154 test cases)** covering:
  - build mode, config, RNG, memory/communication/timer, layout;
  - tokenizer, `prepare_data`, data loader, vocab map;
  - forward ops, matmul, attention, softmax, model, golden tests;
  - backward ops (matmul, attention, softmax, model), gradient check;
  - optimizer, training loop, checkpoints, validation;
  - overfit and smoke training;
  - the sign-off tests (determinism across processes, resume, full-model
    causality);
  - sampling and generation, the samples layout, and the bench library.
- **`tests/fixtures/`:** a 208 KB excerpt of real tokenized TinyStories, so
  the smoke tests need no download.
- **`tests/golden/`:** tokenizer cases and the PyTorch float64 init weights,
  forward results, gradients and 5-step AdamW trajectories.

## Offline tools (`dev_only/`, never built or shipped)
- **`gen_tokenizer_golden.py`:** generates the tokenizer reference cases.
- **`gen_golden.py`:** the float64 PyTorch reference model (loss, gradients,
  AdamW trajectory).
- **`truncate_tokens.py`:** cuts the validation fixture from `val.bin`.
- **`plot_log.py`:** train/val loss curves from `log.csv` as SVG (standard
  library only).

## Next
1. **P6.3 pilot:** two team members score the sampled completions in
   `docs/runs/small_p52/samples_2400.txt` following `eval/RUBRIC.md` (pass:
   ≥ 27 of 30 score pairs within ±1, none differing by more than 2).
2. **P5.3:** the Serial Complete review against ROADMAP §14.
3. **Phase B:** the team settles the open decisions in `ROADMAP_CUDA.md` §3
   (GPU target, bespoke matmul vs cuBLAS, `backend.h` additions, gate
   waiver, kernel file layout). Then apply the accepted CLAUDE.md proposals
   and start B0.1. Phase B needs a machine with an NVIDIA GPU; this
   development container has none.
