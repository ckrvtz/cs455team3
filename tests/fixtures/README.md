# Test fixtures

Small, committed data files read by `make test`. They are real TinyStories
text tokenized by our pipeline, so the smoke and overfit tests (ROADMAP §10,
P4.5) run on real data without the dataset being downloaded. Do not edit them
by hand.

| File | Contents | Read by |
|---|---|---|
| `smoke_train.bin` | the first 99,958 train tokens (522 stories) | `tests/test_smoke.cpp` |
| `smoke_val.bin` | the first 4,097 tokens of the matching `val.bin` (official validation file) | `tests/test_smoke.cpp` |

`smoke_val.bin` holds exactly the 32 fixed validation batches of 4 × 32
tokens that `configs/smoke.cfg` evaluates. Both files use the vocabulary
that `prepare_data` builds from the truncated train set (not the full-data
vocabulary), and the token-file format of ROADMAP §7.3.

To regenerate (needs `scripts/download_data.sh` data in `data/` and `make`;
the output is byte-identical; record any change in `docs/DECISIONS.md`):

```bash
mkdir -p /tmp/smokefx
bin/prepare_data --max_tokens 100000 --out_dir /tmp/smokefx
cp /tmp/smokefx/train.bin tests/fixtures/smoke_train.bin
python3 dev_only/truncate_tokens.py /tmp/smokefx/val.bin tests/fixtures/smoke_val.bin 4097
```

Source: TinyStories (Eldan & Li, 2023), CDLA-Sharing-1.0; see `THIRD_PARTY.md`.
