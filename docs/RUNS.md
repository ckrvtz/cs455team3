# Real training runs

Instructions and records for the long serial runs (ROADMAP P5.2). A run is
recorded here only after its `log.csv` exists; nothing below is estimated
unless it says so.

## P5.2: first `small` run (20M train tokens)

**Status:** prepared and pilot-tested in this repo (see below); the full run
is to be executed on a teammate machine or the cluster, then its `log.csv`
is plotted and assessed here.

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

### Results

Pending: to be filled in from the returned `log.csv`.
