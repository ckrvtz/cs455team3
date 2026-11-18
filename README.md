# TinyStories small transformer (CS455 Team 3)

A small GPT-style language model trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), written
from scratch in C++17. It is planned in four builds: serial, CUDA, serial+MPI
and CUDA+MPI. The plan is in [`ROADMAP.md`](ROADMAP.md) and the working rules
are in [`CLAUDE.md`](CLAUDE.md).

**Status (Phase A, serial):** the serial build is complete and tested.
- **Done:**
  - the data pipeline: download, our own BPE tokenizer, dataset preparation,
    loader;
  - forward and backward passes, verified by a gradient check and PyTorch
    golden files;
  - AdamW;
  - training with validation and checkpoint/resume;
  - generation, prompt evaluation, benchmarks and profiling.
- **Pending:**
  - the first full `small` run and its curves (P5.2, recipe 2 below);
  - the scoring pilot (P6.3);
  - the CUDA hand-off notes (P7.4) and build stubs (P7.5);
  - the Serial Complete review (P5.3).

## Quick start (clean clone to a trained smoke model)

About 10 minutes on the reference machine below, most of it the 2 GB
download and tokenization.

```bash
git clone https://github.com/ckrvtz/cs455team3.git
cd cs455team3
make
make test
scripts/download_data.sh
bin/prepare_data
bin/train_serial --config configs/smoke.cfg
bin/generate --ckpt runs/smoke/ckpt_latest.bin --prompt "Once upon a time" --temperature 0.8 --top_k 50 --seed 7
bin/eval --ckpt runs/smoke/ckpt_latest.bin
```

- **`make test`** runs all 30 test programs in about 20 s, including the
  gradient check, the PyTorch golden comparisons, and the overfit and smoke
  training tests.
- **Training** writes `runs/smoke/` (`manifest.txt`, `log.csv`, checkpoints).
  At step 200 the validation loss is about 5.7.
- **The smoke model is tiny** (333k parameters), so its samples are word
  salad. Readable stories need recipe 2.

## Reproducing the results

### 1. Smoke run (seconds)

The quick start above. On the reference machine,
`bin/train_serial --config configs/smoke.cfg` finishes in about 5 s with a
warm page cache. It ends at train loss 6.1013 and val loss 5.6889
(`docs/DECISIONS.md`).

### 2. `small` run (P5.2: about 2.7 h serial)

```bash
mkdir -p data/p52
bin/prepare_data --max_tokens 20000000 --out_dir data/p52
bin/train_serial --config configs/small.cfg --train_data data/p52/train.bin --val_data data/p52/val.bin --run_name small_p52
bin/train_serial --config configs/small.cfg --train_data data/p52/train.bin --val_data data/p52/val.bin --run_name small_p52 --resume runs/small_p52/ckpt_latest.bin
python3 dev_only/plot_log.py runs/small_p52/curves.svg runs/small_p52/log.csv
bin/eval --ckpt runs/small_p52/ckpt_latest.bin --vocab_map data/p52/vocab_map.txt
```

- **What it is:** 20M training tokens and 2,400 steps (one epoch), with
  validation every 100 steps and checkpoints every 250.
- **Run it unattended** (`nohup`, `tmux`, or a batch job).
- **If it stops:** the second `train_serial` command (with `--resume`)
  continues from the last checkpoint, bit-identically.
- **Afterwards:** `plot_log.py` draws the loss curves and `eval` writes the
  prompt samples.
- **Details:** the expected cost and the plateau criterion are in
  [`docs/RUNS.md`](docs/RUNS.md).

### 3. Benchmarks (about 12 min)

```bash
make clean && make
scripts/bench.sh W-C
scripts/bench.sh W-A
```

- **Output:** each command appends 3 rows to `bench/serial_<workload>.csv`.
- **Reference results:** the committed CSVs, measured on the reference
  machine:

  | Workload | Median step time | Tokens/s |
  |---|---|---|
  | W-C (smoke) | ≈ 20.1 ms | 6.3k–6.4k |
  | W-A (small, B=32) | ≈ 3.61 s | 2.25k–2.30k |
- **Conditions:** run on an idle machine.

### Reference machine

All recorded timings come from this machine unless a record says otherwise.

| | |
|---|---|
| CPU | Intel Xeon Processor @ 2.30GHz with AVX-512 (cloud container), 4 vCPU; the serial build uses 1 |
| RAM | 16 GB |
| OS | Ubuntu 24.04.5 LTS |
| Compiler | g++ 13.3.0, flags `-std=c++17 -O3 -march=native` |
| Make | GNU Make 4.3 |

Earlier records (up to 2026-11-18 05:30 UTC) came from the same container on
an "Intel Xeon @ 2.10GHz" host (`docs/DECISIONS.md`). `-march=native` makes
objects CPU-specific. The Makefile rebuilds everything when the CPU or
compiler changes, and results are bit-reproducible per build, not across
CPUs.

## Requirements

- Linux or macOS, `g++` with C++17, GNU `make`
- `curl` and `sha256sum` (or `shasum`) for the data download
- About 3 GB of disk for the full dataset
- Optional, only to regenerate golden test files: Python 3 with
  `dev_only/requirements.txt`. Nothing in the build or tests uses Python.

## Build and test

```bash
make                     # release build: -O3 -march=native, apps in bin/
make test                # build and run every test, incl. overfit + smoke training (~20 s)
make debug               # -O0 -g with AddressSanitizer + UBSan, then run all tests
make test REAL=double    # double-precision build (used for gradient checks)
make dist                # submission tarball dist/tslm.tar.gz (fails if any .py remains)
make clean               # remove build/, bin/, dist/
```

- **Where apps go:** release apps are in `bin/`. Other builds put theirs in
  `bin/<mode>-<real>/`, for example `bin/debug-float/` or `bin/release-double/`.
- **Build directories:** each configuration builds in its own
  `build/<mode>-<backend>-<comm>-<real>/`, so switching builds never mixes
  object files.
- **Not available yet:** `make BACKEND=cuda` and `make COMM=mpi` are later
  phases; they currently stop with a clear `#error`.

### Gradient check

```bash
make REAL=double
bin/release-double/gradcheck                       # tied tiny model; prints a per-tensor table, exit 0 on PASS
bin/release-double/gradcheck --tie_embeddings 0    # untied head
```

This compares `model_backward` against finite differences on the ROADMAP
§10 tiny model, about 0.2 s. Options: `--seed N`, `--scale S` (init weight
multiplier, default 3), `--samples K` (default 20 per tensor), `--h H`
(stencil step, default 3e-4), plus model/batch config keys. It runs only in
the double-precision build; `bin/gradcheck` from a float build refuses with
a message. `make test REAL=double` also runs it.

## Data

```bash
scripts/download_data.sh     # ~1.96 GB into data/; pinned versions, SHA-256 verified
bin/prepare_data             # tokenize and build the vocab: about 1 minute for the full train set
bin/inspect_data --n 3       # print random decoded training sequences ([EOS] = story end)
```

- **`download_data.sh`** fetches the TinyStories train and validation text
  plus the GPT-Neo `vocab.json` and `merges.txt`. Rerunning it skips files
  already verified. Set `DATA_DIR=/path` to download somewhere else.
- **`prepare_data`** writes these files to `data/`:
  - `train.bin` and `val.bin`: token files;
  - `vocab_map.txt`: the 10,000-token vocabulary;
  - `meta.txt`: counts and coverage.

  Options:
  - `--max_stories N` / `--max_tokens N`: truncate the train set (whole stories);
  - `--vocab_size V`;
  - `--out_dir D`;
  - `--train_txt`, `--val_txt`, `--vocab_json`, `--merges_txt`: input paths.

  Reruns produce byte-identical output.
- **`inspect_data`** options: `--data`, `--vocab_map`, `--n`, `--T`, `--seed`.

`data/`, `runs/`, `bin/`, `build/` and `dist/` are git-ignored.

## Configuration

Model and training settings are `key=value` files in `configs/`: `smoke.cfg`
(tiny, for tests), `small.cfg` and `base.cfg`. Apps take `--config <file>`
plus `--key value` overrides. Unknown keys are errors.

| Preset | Layers | Width | Heads | Context | Parameters (tied head) |
|---|---|---|---|---|---|
| `smoke` | 1 | 32 | 2 | 32 | 333,792 |
| `small` | 2 | 128 | 4 | 256 | 1,709,568 |
| `base` | 4 | 256 | 8 | 256 | 5,785,088 |

## Training

```bash
bin/train_serial --config configs/smoke.cfg               # 200 steps of the tiny smoke model
bin/train_serial --config configs/smoke.cfg --max_steps 50 --run_name try1
bin/train_serial --config configs/smoke.cfg --ckpt_every 100                                  # also checkpoint at step 100
bin/train_serial --config configs/smoke.cfg --ckpt_every 100 --resume runs/smoke/ckpt_100.bin # continue from step 100
```

- **Outputs** go to `runs/<run_name>/` (`--out_dir` changes `runs`):
  - `manifest.txt`: date, git hash, build, compiler flags, CPU, then the full
    resolved config. `--config runs/<run_name>/manifest.txt` reruns the same config.
  - `log.csv`: one row at step 1, every `log_every` steps, at each
    validation and at the end: training loss, validation loss and perplexity
    (on validation rows), lr, grad norm, tokens/s, and per-step milliseconds
    for data, forward, backward, comm and optimizer.
  - `ckpt_<step>.bin` and `ckpt_latest.bin`: weights plus optimizer and data
    position, every `ckpt_every` steps (0 = none) and always at the last step.

  Rerunning a run name overwrites its files, with a warning.
- **Resume:** `--resume <checkpoint>` continues from the step after it, with
  the same seed, model, `batch_size` and `seq_len` (checked). In the same
  run directory, `log.csv` rows after the checkpoint are replaced. A resumed
  run gives the same results, bit for bit, as one that was never stopped.
- **Data:** training reads `data/train.bin` (`--train_data` to change it).
- **Validation:** every `eval_every` steps and at the end, on the same first
  `val_batches` batches of `data/val.bin` (`--val_data`), forward only. A
  fresh run also prints the validation loss before step 1 (about
  ln 10000 = 9.21).
- **Timing:** the first run after a reboot can be much slower. The 927 MB
  token file is read at random positions, and those reads are slow until the
  file is in the OS page cache; the extra time shows up in `data_ms`.

## Generation

```bash
bin/generate --ckpt runs/smoke/ckpt_latest.bin --prompt "Once upon a time" --greedy 1 --max_new 40
bin/generate --ckpt runs/smoke/ckpt_latest.bin --prompt "Once upon a time" --temperature 0.8 --top_k 50 --seed 7
```

- **Input:** a checkpoint from a training run; the model shape comes from
  it. `--vocab_map` (default `data/vocab_map.txt`) must come from the same
  `prepare_data` run as that training data, for example
  `data/p52/vocab_map.txt` for the P5.2 run. Only its size is checked.
- **Sampling:** `--greedy 1`, or `--temperature T` with optional
  `--top_k K`, and `--seed`. Output stops at end-of-story or after
  `--max_new` tokens. `<unk>` is never generated unless `--allow_unk 1`.
- **Output:** the prompt followed by the continuation. Same options and seed
  give the same text. The tiny smoke model produces word salad; readable
  stories need the `small` run.

## Prompt evaluation

```bash
bin/eval --ckpt runs/smoke/ckpt_latest.bin        # writes runs/smoke/samples_200.txt
```

- **Prompts:** the 10 fixed prompts in `eval/prompts.txt` (story openers,
  factual, reasoning, consistency).
- **Output:** for each prompt, one greedy completion and one at temperature
  0.8 with seed 1337 (at most 120 tokens). They go to
  `samples_<step>.txt` next to the checkpoint (`--out` to change it).
- **Comparability:** the settings are fixed and the file layout is fixed
  (`include/tslm/samples.h`), so files from different runs and builds
  compare line by line. The same binary and checkpoint reproduce the file
  byte for byte.
- **Vocab map:** `--vocab_map` follows the same rule as for `generate`.
- **Scoring:** completions are scored by hand, 0–5 on grammar, consistency
  and sense, following [`eval/RUBRIC.md`](eval/RUBRIC.md). Scores go in
  `eval/scores.csv`.

## Benchmarks

```bash
scripts/bench.sh W-C        # smoke, 200 steps, 3 runs -> bench/serial_W-C.csv
scripts/bench.sh W-A        # small, 50 steps, B=32, 3 runs (about 10 min serial) -> bench/serial_W-A.csv
```

- **What a run does:** `bin/bench` runs one fixed workload (ROADMAP §12)
  with the real training step and appends one CSV row.
  - **Row contents:** median and mean step time, tokens/s, the per-phase
    means, peak memory, machine id and git hash.
  - **Warmup:** the first 3 steps are discarded.
  - **Prewarm:** the training token file is read once first, so a cold
    page cache does not distort the data time.
- **Manifest:** `bench/<version>_<workload>.manifest.txt` records the
  build, flags, machine and config of the latest run.
- **Conditions:** run benchmarks on an idle machine, from a clean release
  build (`make clean && make`).

### Profiling (developers)

```bash
scripts/profile.sh          # gprof flat profile of the W-A step (13 steps, ~1 min) -> docs/profile/serial_W-A_gprof.txt
```

It builds `make MODE=profile` (the release flags plus `-pg`, in
`bin/profile-float/`; never used by `make test` or `make dist`). The
analysis is in [`docs/PROFILE.md`](docs/PROFILE.md).

## Long runs

Real training runs and their results are recorded in
[`docs/RUNS.md`](docs/RUNS.md). Loss curves are drawn from `log.csv` by an
offline script (standard-library Python, developers only):

```bash
python3 dev_only/plot_log.py curves.svg runs/<run_name>/log.csv
```

## Golden test data (developers only)

Tests compare against small reference files committed in `tests/golden/`;
see [`tests/golden/README.md`](tests/golden/README.md). They are regenerated
offline only when a model definition legitimately changes:

The exact commands (a Python virtual environment with
`dev_only/requirements.txt`, then `bin/dump_init` and
`dev_only/gen_golden.py`) are in that file. The PyPI `torch` wheel for Linux
also installs about 2.5 GB of CUDA libraries.

## Layout

```
include/tslm/   public headers (ops.h, model.h, layout.h, backend.h, comm.h, ...)
src/            library: common/ backend/ comm/ ops/ model/ optim/ data/ train/
apps/           command-line tools (thin main()s)
tests/          unit tests (test_*.cpp), golden data, and a 208 KB real-data fixture (tests/fixtures/)
configs/        presets       scripts/  data download, bench.sh, profile.sh
eval/           fixed prompts, scoring rubric and scores template       bench/  committed benchmark CSVs
docs/           DECISIONS.md (decision log), PROFILE.md (optimizations + profile), RUNS.md (long runs), STATUS.md
dev_only/       offline Python (golden files, fixture cutting, loss plots); never built or shipped
```
