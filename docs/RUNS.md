# Real training runs

Instructions and records for the long serial runs (ROADMAP P5.2). A run is
recorded here only after its `log.csv` exists; nothing below is estimated
unless it says so.

## P5.2: first `small` run (20M train tokens)

**Status:** completed on a teammate machine (2026-11-18, results below).
Final val loss 2.6455 (ppl 14.1). Plateau criterion: 2 of 3 met; criterion
2 missed by 0.006. The verdict is pending team review. The P6.1/P6.3 checks
need the run's `ckpt_latest.bin`.

### What it is

- Preset `configs/small.cfg`: 2 layers, 128 wide, 4 heads, T=256, B=32
  (8,192 tokens/step), 1,709,568 parameters (tied), peak lr 1e-3 with 200
  warmup steps and cosine decay to 1e-4.
- Data: the first 19,999,905 train tokens (93,024 stories) via
  `prepare_data --max_tokens 20000000`; validation is the official file
  (4,667,152 tokens), of which the fixed first 20 batches (163,840 tokens)
  are evaluated.
- `max_steps=2400`: one epoch of the subset is 2,441 steps, so no training
  sequence is seen twice. Validation every 100 steps, checkpoints every 250.

### Commands

```bash
make
mkdir -p data/p52
bin/prepare_data --max_tokens 20000000 --out_dir data/p52
grep -E "train_tokens|vocab_hash" data/p52/meta.txt     # expect 19999905 and 4de4b6c3947630ff
bin/train_serial --config configs/small.cfg --train_data data/p52/train.bin --val_data data/p52/val.bin --run_name small_p52
```

Run the last command under `nohup`, `tmux`/`screen` or a batch job: it is
long. If it stops for any reason, rerun the same command with
`--resume runs/small_p52/ckpt_latest.bin` appended; the run continues from
the last checkpoint and is bit-identical to an uninterrupted one
(`log.csv` rows after the checkpoint are replaced).

Afterwards:

```bash
python3 dev_only/plot_log.py runs/small_p52/curves.svg runs/small_p52/log.csv
```

Send back `runs/small_p52/log.csv` and `runs/small_p52/manifest.txt` (small
text files) and keep `runs/small_p52/ckpt_latest.bin` locally (21 MB,
gitignored; Phase 6 generation starts from it).

### Expected cost (estimate)

Measured on this project's cloud container (Intel Xeon @ 2.30GHz with AVX-512, release
float build, warm page cache, pilot below): 3.8 s/step, and about 1.5 s per
validation batch. The estimate is therefore 2,400 × 3.8 s ≈ 2.5 h of steps
plus 24 evaluations × 20 batches × 1.5 s ≈ 12 min, about **2.7 h**. Other
machines differ; the first steps are slower while the token file enters the
page cache.

### Pilot (run here, 2026-11-18)

The production command with `--run_name p52_pilot --max_steps 20
--warmup_steps 5 --eval_every 10 --ckpt_every 10 --val_batches 4`:

| step | train loss | val loss |
|---|---|---|
| init | | 9.2382 |
| 1 | 9.2461 | |
| 10 | 8.0812 | 7.9299 |
| 20 | 7.4608 | 7.4273 |

3.8 s/step, checkpoints 84–108 ms. Resuming in place from `ckpt_10.bin`
reproduced `ckpt_20.bin` byte for byte, and `plot_log.py` rendered the log.
These are pipeline checks only, not results.

### Plateau criterion (fixed before the run)

The run's loss is called a defensible plateau if, on the logged validation
evaluations:

1. val loss falls by less than 0.05 nats over the last 500 steps
   (steps 1900 → 2400), and
2. the final train/val gap (mean train loss over the last 100 steps minus
   final val loss) is within ±0.1 nats (single epoch, so no overfitting is
   expected), and
3. no NaN/Inf and no val increase larger than 0.02 between consecutive
   evaluations after step 500.

If (1) fails, the loss was still improving and the next step is a longer
run on more data (up to the 50M tokens P5.2 allows), not a different
reading of the curve. ROADMAP §10 expects roughly 2–3 nats for truncated
serial runs; a loss stuck near 6–7 would be a failure.

### Results (run completed 2026-11-18)

- **Files:** `docs/runs/small_p52/log.csv` and `manifest.txt` (as returned
  by the teammate who ran it) and `curves.svg`
  (`python3 dev_only/plot_log.py docs/runs/small_p52/curves.svg docs/runs/small_p52/log.csv`).
- **Machine** (from the manifest): Intel Core i7-6820HQ @ 2.70GHz (AVX2, 8
  logical cores, 1 used), 15.9 GB RAM, g++ 16.2.1 (Red Hat), release float
  build. The git hash `1688a7fdf399` is a local commit that is not on the
  shared branch.
- **Data and config:** 19,999,905 train tokens (2,441 steps per epoch) and
  `configs/small.cfg` as committed, i.e. 2,400 steps.
- **Timing:** median step 3,948 ms and 2,075 tok/s (log rows from step 50
  on). Not a benchmark: `log.csv` timings are interval means without the
  §12 protocol.

| Step | 100 | 200 | 300 | 500 | 1000 | 1500 | 1900 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|---|
| val loss | 5.9065 | 4.2254 | 3.8845 | 3.4452 | 2.9691 | 2.7711 | 2.6934 | 2.6760 | 2.6582 | **2.6455** |
| val ppl | 367.4 | 68.4 | 48.6 | 31.3 | 19.5 | 16.0 | 14.8 | 14.5 | 14.3 | **14.1** |

**Plateau criterion** (as fixed before the run):

| Criterion | Measured | Result |
|---|---|---|
| 1. val falls < 0.05 nats over steps 1900 → 2400 | 2.6934 → 2.6455: 0.0479 | met (margin 0.002) |
| 2. mean train loss over the last 100 steps − final val within ±0.1 | 2.7516 − 2.6455 = +0.106 | **not met** (by 0.006) |
| 3. no NaN/Inf; no val increase > 0.02 after step 500 | none; val decreased at every evaluation | met |

Criterion 2 notes, recorded but not used to reinterpret the result:
- **Sampling:** `log.csv` keeps one training loss every 10 steps. The "mean
  over the last 100 steps" is therefore the mean of 10 single-step losses
  (steps 2310–2400, range 2.563–2.860), not of all 100 steps.
- **Direction:** the gap is val *below* train, the opposite of overfitting.

Whether the run is a defensible plateau (P5.2 DoD) is left to the team. A
more precise train-loss measure would be `evaluate()` over fixed training
batches using the final checkpoint.

**Cross-machine reproducibility:** steps 1–260 of this run match the
clean-clone verification run (Xeon @ 2.30GHz with AVX-512, g++ 13.3.0,
2026-11-18, P7.3), bit for bit in the printed digits. That is 108 logged
values (train loss, val loss, lr, grad norm), all identical to 9 significant
digits.

### Supplementary train/val gap measurement (team decision after criterion 2 missed)

- **Method,** fixed before computing:
  - the final checkpoint (step 2400);
  - the first 20 batches of `data/p52/train.bin` in file order (163,840
    tokens: same size and rule as the validation set);
  - the same `evaluate()` pass, via `bin/eval_loss`;
  - pass mark |train_eval − final val| ≤ 0.1.
- **Sanity check:** the same tool on `data/p52/val.bin` reproduced the run's
  final val_loss exactly (2.64546051), on a different machine.

| Measurement | Loss | Gap to final val 2.6455 | Result |
|---|---|---|---|
| Final model, first 20 train batches | 2.8296 | +0.184 | **not met** |

**Diagnostic** (decided after seeing the miss; it does not replace the
result above): the same checkpoint on 100 batches (819,200 tokens) of each
file.

| Batches | Train loss | Val loss | Gap |
|---|---|---|---|
| 20 (the fixed sets) | 2.8296 | 2.6455 | +0.184 |
| 100 | 2.7815 | 2.7668 | +0.015 |

- **Reading:** the fixed 20-batch validation set (the first 640 sequences of
  the official validation file) is about 0.12 nats easier than the first
  100 batches. So the gap in criterion 2 comes from that slice, not from a
  train/val difference in the model.
- **Consequence:** a longer run would lower train loss further and is
  unlikely to close a gap of this sign.
- **Other implication:** the logged `val_loss` of every run (always on the
  same 20 batches) underestimates the broader validation loss of this model
  by about 0.12. It remains comparable across runs and builds, which is
  its purpose.
- **Status:** the P5.2 plateau verdict remains with the team.

### Generation and prompt samples from the final checkpoint (P6.1, P6.2)

- **Greedy:** `bin/generate --ckpt runs/small_p52/ckpt_latest.bin --vocab_map
  data/p52/vocab_map.txt --prompt "Once upon a time" --greedy 1 --max_new 80`
  gives byte-identical output on repeated runs: "Once upon a time, there was
  a little girl named Lily. She loved to play outside in the park. One day,
  she saw a big, shiny rock. She wanted to play with it, but it was too big
  and it was too fast. ..."
- **Sampled** (temperature 0.8, seeds 1–3): fluent TinyStories-style text,
  mostly grammatical, with occasional errors ("there was a big, a little
  girl"; "Lily … He was three years old") and weak long-range consistency.
- **Fixed prompts:** `docs/runs/small_p52/samples_2400.txt`, from `bin/eval`
  (byte-identical on rerun). The factual and reasoning prompts are mostly
  answered wrongly ("opened the pot", "now she had a big spoon"), which the
  rubric's sense scale is meant to capture.
- **Scoring pilot (P6.3):** this file is the pilot set. It needs two human
  raters (`eval/RUBRIC.md`).
