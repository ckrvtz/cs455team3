# Golden test data

Small, committed data files read by `make test`. Nothing here is generated
at build or test time. Do not edit them by hand (CLAUDE.md §2).

| Files | Producer | Read by |
|---|---|---|
| `tokenizer_cases.txt`, `tokenizer_vocab_subset.json`, `tokenizer_merges_subset.txt` | `dev_only/gen_tokenizer_golden.py` (HF `tokenizers`) | `tests/test_tokenizer.cpp`, `tests/test_prepare.cpp` |
| `<case>_init.bin` | `bin/dump_init` (C++) | `tests/test_golden.cpp`, `dev_only/gen_golden.py` |
| `<case>_fwd.bin` | `dev_only/gen_golden.py` (float64 PyTorch) | `tests/test_golden.cpp` |
| `<case>_adamw.bin` | `dev_only/gen_golden.py` (float64 `torch.optim.AdamW`) | `tests/test_golden.cpp` |

`.bin` files use the `TSGD` named-tensor format (`include/tslm/tensor_file.h`).
Despite the name, `<case>_fwd.bin` holds both the forward results (`loss`,
`losses`, `logits`) and, since P3.7, the autograd gradient of the mean loss
for every parameter (`grad.<name>`). `<case>_adamw.bin` (P4.1) holds a
5-step AdamW trajectory from the same init and batch: `hparams`,
`step_loss`, `step_grad_norm`, `step_lr` and the final `param.<name>`. Its
settings (lr 1e-3, warmup 2, cosine to 10%, clip 1.0, wd 0.1 on the decay
tensors) are the `ADAMW` dict in the generator; see `docs/DECISIONS.md`,
"AdamW golden trajectory" (P4.1).

## Forward, backward and AdamW golden cases (P2.6, P3.7, P4.1)

Both cases use the ROADMAP §10 gradcheck model: V=32, max_T=8, L=2, C=16,
NH=2, B=2, T=8, seed 1337.

| Case | Head | Weights |
|---|---|---|
| `tiny` | tied | standard init |
| `tiny_untied_x10` | untied | init × 10 (non-uniform logits, so attention/MLP errors show) |

To regenerate (only when the model definition legitimately changes; record
it in `docs/DECISIONS.md`). Always regenerate all three outputs together, so
the AdamW reference starts from the same init as the forward/backward one:

```bash
make
G="--vocab_size 32 --max_seq_len 8 --n_layer 2 --n_embd 16 --n_head 2 --batch_size 2 --seq_len 8 --seed 1337 --run_name golden"
bin/dump_init --out tests/golden/tiny_init.bin $G --tie_embeddings 1 --scale 1
bin/dump_init --out tests/golden/tiny_untied_x10_init.bin $G --tie_embeddings 0 --scale 10
python dev_only/gen_golden.py tests/golden/tiny_init.bin tests/golden/tiny_fwd.bin tests/golden/tiny_adamw.bin
python dev_only/gen_golden.py tests/golden/tiny_untied_x10_init.bin tests/golden/tiny_untied_x10_fwd.bin \
    tests/golden/tiny_untied_x10_adamw.bin
```
