# TinyStories small transformer (CS455 Team 3)

A small GPT-style language model trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), written
from scratch in C++17. It is planned in four builds: serial, CUDA, serial+MPI
and CUDA+MPI. The plan is in [`ROADMAP.md`](ROADMAP.md) and the working rules
are in [`CLAUDE.md`](CLAUDE.md).

**Status (Phase A, serial):** the data pipeline (download, tokenizer, dataset
preparation, loader), the full forward pass and the first backward ops are
done. The remaining backward ops, the optimizer, training, generation and
benchmarking are not built yet, so there is no `train` command.

## Requirements

- Linux or macOS, `g++` with C++17, GNU `make`
- `curl` and `sha256sum` (or `shasum`) for the data download
- About 3 GB of disk for the full dataset
- Optional, only to regenerate golden test files: Python 3 with
  `dev_only/requirements.txt`. Nothing in the build or tests uses Python.

## Build and test

```bash
make                     # release build: -O3 -march=native, apps in bin/
make test                # build and run every unit test (a few seconds)
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

## Golden test data (developers only)

Tests compare against small reference files committed in `tests/golden/`;
see [`tests/golden/README.md`](tests/golden/README.md). They are regenerated
offline only when a model definition legitimately changes:

```bash
python3 -m venv .venv && .venv/bin/pip install -r dev_only/requirements.txt
bin/dump_init --out tests/golden/tiny_init.bin ...            # full commands in tests/golden/README.md
.venv/bin/python dev_only/gen_golden.py tests/golden/tiny_init.bin tests/golden/tiny_fwd.bin
```

The PyPI `torch` wheel for Linux also installs about 2.5 GB of CUDA libraries.

## Layout

```
include/tslm/   public headers (ops.h, model.h, layout.h, backend.h, comm.h, ...)
src/            library: common/ backend/ comm/ ops/ model/ data/
apps/           command-line tools (thin main()s)
tests/          unit tests (test_*.cpp) and golden data
configs/        presets       scripts/  data download
docs/           DECISIONS.md (design decision log), PROFILE.md (serial optimizations)
dev_only/       offline Python for golden files; never built or shipped
```
