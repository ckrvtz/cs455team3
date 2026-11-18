# ROADMAP: Serial Baseline (Phase A)

Project: TinyStories decoder-only transformer, CS455 Team 3 (Conor, Vincent)
Scope of this document: the **serial C++ version only**, designed as the skeleton that the CUDA, MPI, and MPI+CUDA versions extend.
Companion file: `CLAUDE.md` (rules for LLM-assisted development). If the two ever conflict, `CLAUDE.md` wins on process and this file wins on technical specification.

---

## 0. How to use this document (humans and LLMs)

- Every task has an ID (`P3.4` etc.). Prompt with: "Implement task P3.4 from ROADMAP.md, following CLAUDE.md." One task per prompt where possible.
- Sections 2, 4, 6, 8 are **contracts**. Do not change them silently. A change needs (a) an entry in `docs/DECISIONS.md`, (b) a note on the impact for CUDA/MPI, (c) updated tests.
- "Definition of Done" (DoD) for a task = all listed acceptance criteria pass **and** `make test` is green.
- Anything marked **(DECISION)** is a default chosen by this roadmap to unblock work. The team may override it, but record the change.
- `README.md` is maintained continuously: a task that adds or changes anything a user runs or configures also updates the README in the same change (CLAUDE.md §2), so it always matches the current state of the code. See `docs/DECISIONS.md`, 2026-11-18.

---

## 1. Goals and non-goals

### Goals
1. A correct, deterministic, reasonably efficient single-threaded C++17 trainer for a GPT-style model on TinyStories.
2. An architecture in which CUDA kernels, MPI gradient averaging, and both together can be **added without rewriting model, data, or training-loop code**.
3. A validation suite (finite-difference gradient check, golden-file comparison, determinism, overfit test) that later versions reuse unchanged to prove equivalence with serial.
4. Benchmark tooling that produces the speedup numbers the proposal promises.

### Non-goals (for Phase A)
- No OpenMP, threads, SIMD intrinsics, or BLAS in the baseline (speedups are measured against it; it must be honest and documented).
- No mixed precision, no dropout, no KV cache, no distributed code (only the no-op shims).
- No Python anywhere in the deliverable. Python/PyTorch lives only in `dev_only/` for offline golden-file generation and plotting.

---

## 2. Locked specification

### 2.1 Model

| Item | Value | Notes |
|---|---|---|
| Type | Decoder-only transformer, pre-LayerNorm (GPT-2 style) | |
| Presets | `smoke`, `small`, `base` (below) | proposal: start with 2L/128d, target 4L/256d/8H |
| Vocab | 10,000 | GPT-Neo BPE restricted to top-frequency tokens; see §7 for id layout |
| Context length T | 256 (smoke: 32) | |
| Positional encoding | Learned absolute (`wpe`) **(DECISION)** | matches GPT-Neo; proposal silent |
| Attention | Causal multi-head, scale 1/sqrt(head_dim), fused QKV projection | head_dim = C / NH must be an integer |
| MLP | Linear(C→4C), GELU (tanh approximation), Linear(4C→C) | |
| Norm | LayerNorm, eps = 1e-5, affine (gamma, beta) | final LN before head |
| Output head | **Tied to `wte` by default** **(DECISION)**; `tie_embeddings=0` adds a separate `[V,C]` head | |
| Dropout | None | keeps RNG off the device later |
| Precision | `float` (`real_t`); `double` build only for gradient check | |
| Loss | Mean cross-entropy over all B·T positions | |

Presets:

| Preset | L | C | NH | T | B | Purpose |
|---|---|---|---|---|---|---|
| `smoke` | 1 | 32 | 2 | 32 | 4 | CI-speed end-to-end test (seconds) |
| `small` | 2 | 128 | 4 | 256 | 32 | first real model (proposal's "smaller" model) |
| `base`  | 4 | 256 | 8 | 256 | 32 | proposal target (MLP hidden 1024 = 4C) |

Head count for `small` is not specified in the proposal; 4 (head_dim 32, same as `base`) is the default.

### 2.2 Parameter counts (use these as test oracles)

Per layer: LN1 (2C) + QKV (3C² + 3C) + proj (C² + C) + LN2 (2C) + fc (4C² + 4C) + fcproj (4C² + C) = 12C² + 13C.

| Preset | Tied | Untied |
|---|---|---|
| `base` (V=10000) | **5,785,088** | **8,345,088** |
| `small` (V=10000) | **1,709,568** | **2,989,568** |

Total = V·C (wte) + T·C (wpe) + L·(12C²+13C) + 2C (final LN) [+ V·C if untied].

> **Proposal discrepancy:** the proposal says "around 6M parameters (or half with tied embeddings)". The correct figures are ≈5.8M **tied** and ≈8.3M **untied**; tying removes the 2.56M-parameter head, not half the model. We default to tied, which gives the proposal's ≈6M. `make test` must assert the exact numbers above.

### 2.3 Training defaults (starting points, tunable via config)

| Setting | Default |
|---|---|
| Optimizer | AdamW, β1=0.9, β2=0.95, eps=1e-8 |
| Weight decay | 0.1, applied **only** to 2-D weight matrices (`qkv_w, proj_w, fc_w, fcproj_w, wte`); not to biases, LayerNorm params, or `wpe` **(DECISION)** |
| LR schedule | Linear warmup (200 steps) → cosine decay to 10% of peak |
| Peak LR | 1e-3 (`small`), 6e-4 (`base`); tune |
| Grad clip | Global L2 norm, max 1.0 |
| Init | All weights N(0, 0.02); biases 0; LN gamma 1, beta 0; residual-output projections (`proj_w`, `fcproj_w`) scaled by 1/sqrt(2L) |
| Batch | B=32 sequences × T=256 = 8,192 tokens/step |
| Seed | Single `seed` controls init, data order, sampling |

### 2.4 Required metrics (every run)

Training loss, validation loss, perplexity (= exp(val loss)), tokens/second, step time with per-phase breakdown (data, forward, backward, comm, optimizer). Plus manual generation scoring from the proposal (10 fixed prompts, 0–5 on grammar, consistency, sense, per milestone).

---

## 3. Decisions made and issues found in the source documents

| # | Item | Resolution |
|---|---|---|
| D1 | Param-count claim (see §2.2) | Use exact counts; tied default |
| D2 | Proposal typos: "Attenuation", "casual-masked" | Read as *attention*, *causal* |
| D3 | Dataset variant | Original `roneneldan/TinyStories` (the linked dataset). V2-GPT4 variant is an optional later experiment, never mixed silently |
| D4 | Validation split | Use the dataset's official validation file for `val.bin`; the training file is only truncated (never re-split) so validation numbers stay comparable across all four implementations |
| D5 | Tokenizer without Python | Byte-level BPE (GPT-2/GPT-Neo) must be **reimplemented in C++**, reading `vocab.json` + `merges.txt` as data files. Python/HF used offline only to generate golden test vectors |
| D6 | Paper evaluates with GPT-4; proposal uses manual 0–5 scoring | Follow proposal (manual); keep scoring CSV format stable |
| D7 | Paper uses window 256 / context 512 (GPT-Neo local attention) | We use full causal attention, T=256. No local-attention windowing |
| D8 | Serial compute cost | See §13: the `base` config is far too slow for full-dataset serial training. Serial runs use truncated data and/or the `small` preset; fixed-step benchmarks define speedup |
| D9 | Reference to llm.c-style projects | LLMs will tend to reproduce llm.c structure. Design here is bespoke; if any code is adapted from an open source project it must be cited in `THIRD_PARTY.md` (see CLAUDE.md) |

---

## 4. Architecture and extension contracts

The serial build must already contain the seams that later versions plug into. These seams are **mandatory even though they are trivial in serial.**

### 4.1 Four builds, one codebase

```
make                       # BACKEND=cpu  COMM=serial  -> bin/train_serial
make BACKEND=cuda          # BACKEND=cuda COMM=serial  -> bin/train_cuda        (Phase B)
make COMM=mpi              # BACKEND=cpu  COMM=mpi     -> bin/train_mpi         (Phase C)
make BACKEND=cuda COMM=mpi # BACKEND=cuda COMM=mpi     -> bin/train_mpi_cuda    (Phase D)
```
Unsupported combinations must fail at build time with a clear `#error`, not at runtime. Compile-time selection only (no virtual dispatch in hot paths).

### 4.2 Layers

```
apps/ (train, generate, eval, bench, prepare_data, ...)         <- thin main()s
   |
train/  (loop, schedule, logging, checkpoint)                   <- backend/comm-agnostic
model/  (config, param layout, activation layout, forward, backward)  <- backend-agnostic
   |                         |                        |
ops/ (ops.h + cpu impl)   backend/ (memory shim)   comm/ (collectives shim)
optim/ (adamw, clip)      data/ (tokenizer, loader; HOST only, always)
```

- **`ops.h`** declares every numeric kernel as a free function in `namespace ops` taking raw pointers + dimensions (§8). `ops_cpu.cpp` implements them now; `ops_cuda.cu` implements the *same signatures* later. `model/` calls only `ops::`, never loops over tensors itself.
- **`backend.h`** (memory shim): `dev_malloc(bytes)`, `dev_free`, `dev_memset`, `copy_h2d`, `copy_d2h`, `copy_d2d`, `dev_sync()`, `dev_name()`. CPU implementation = `malloc/memset/memcpy` and a no-op sync. `model/` and `train/` must never call `malloc`/`new`/`memcpy` on tensors directly. **All tensor memory goes through the shim**, so in CUDA builds those pointers are device pointers and model code is unchanged.
- **`comm.h`** (collectives shim): `comm_init(argc, argv)`, `comm_rank()`, `comm_size()`, `comm_broadcast(real_t*, size_t n, int root)`, `comm_allreduce_avg(real_t*, size_t n)`, `comm_allreduce_sum_double(double*, int n)`, `comm_barrier()`, `comm_finalize()`. Serial implementation = size 1, rank 0, all collectives no-ops.
- **`timer.h`**: `PhaseTimer` with named phases (`data, fwd, bwd, comm, opt`). In serial uses `steady_clock`. Later builds call `dev_sync()` before reading the clock. Call sites must already be in place.
- **Host/device rule:** tokenization, data loading, RNG, parameter *initialization*, logging, checkpoint file I/O, and sampling decisions are **host-side always**. Only `ops::`, optimizer, and gradient-norm kernels touch tensor data, and they go through pointers that may be device pointers. Host code that needs a value from a tensor (loss, grad norm) gets it via `copy_d2h` or a returned scalar.

### 4.3 Flat parameter, gradient, and optimizer buffers (critical for MPI)

- All parameters live in **one contiguous `real_t` buffer** `params[num_params_padded]`. Gradients in an identically laid out `grads[]`. AdamW state in `m[]`, `v[]`.
- Layout is computed by a pure function `ParamLayout make_param_layout(const ModelConfig&)` returning, for each named tensor: `offset`, `numel`, `shape`, `decay` (bool). Each tensor offset is padded to a multiple of **16 floats (64 bytes)**. `num_params` (logical, excludes padding) is reported separately from `num_params_padded`.
- Consequences (all intentional): one `comm_broadcast(params)` after init; one `comm_allreduce_avg(grads)` per step; one memset to zero grads; checkpoint = header + three blobs; checkpoints are portable across all four builds.
- **Activation arena:** all activations/saved-for-backward tensors live in one buffer with a layout computed by `make_act_layout(cfg, B, T)`. Allocated once at startup. **No allocation inside the training loop.**

### 4.4 Gradient accumulation convention

- Every `*_forward` **overwrites** its output.
- Every `*_backward` **accumulates (`+=`)** into `d*` outputs. (Required for residual branches and for the tied embedding, which receives gradient from both the input lookup and the output head.)
- `grads` is zeroed once per step by the training loop before backward.

### 4.5 Data-parallel semantics (define now, implement in Phase C)

- Loader API takes `(rank, world_size)`; serial passes `(0, 1)`.
- **Global batch** = `micro_batch × world_size` sequences. Rank r gets rows `[r·micro_batch, (r+1)·micro_batch)` of the global batch that serial would have produced for the same step/seed.
- Loss is the mean over the rank's local tokens; averaging local gradients with `comm_allreduce_avg` equals the serial gradient on the global batch **only if local batches have equal token counts** (always true here). Test in Phase C: serial `B=32` ≡ 4 ranks × `B=8`.
- Step order is fixed: `zero_grads → forward → backward → comm_allreduce_avg(grads) → clip → adamw`. Clipping happens **after** the allreduce so every rank computes the same norm.

### 4.6 Randomness

- Own RNG (`xoshiro256**` seeded via `splitmix64`), own `normal()` (Box–Muller in `double`). No `std::mt19937`, no `std::normal_distribution` (implementation-defined output).
- Init stream for tensor `i` is `Rng(seed, stream = i)`, so initialization is independent of init order and reproducible across builds.
- Data shuffle uses a separate stream; sampling in `generate` uses another.

---

## 5. Repository layout

```
.
├── CLAUDE.md
├── ROADMAP.md
├── README.md                 # build/run/reproduce; kept current every task, completed in P7.3
├── THIRD_PARTY.md            # every cited external source (see CLAUDE.md)
├── Makefile
├── configs/                  # smoke.cfg small.cfg base.cfg   (key=value text)
├── include/tslm/             # public headers (see §4, §8)
│   ├── common.h  config.h  rng.h  timer.h  check.h
│   ├── backend.h comm.h  ops.h
│   ├── layout.h  model.h  optim.h
│   ├── tokenizer.h  dataset.h
│   └── checkpoint.h  logging.h
├── src/
│   ├── common/   backend/   comm/   ops/   model/   optim/   data/   train/
├── apps/
│   ├── prepare_data.cpp   train.cpp   generate.cpp
│   ├── eval.cpp   bench.cpp   dump_init.cpp   inspect_data.cpp
├── tests/
│   ├── test.h  test_*.cpp
│   └── golden/               # small committed data files: .txt .json .bin
├── scripts/                  # download_data.sh, run_*.sh, slurm/*.sbatch
├── data/                     # GITIGNORED (raw text, train.bin, val.bin, vocab files)
├── runs/                     # GITIGNORED (logs, checkpoints)
├── docs/DECISIONS.md         # append-only decision log
└── dev_only/                 # NOT SUBMITTED: python reference, plotting; excluded by `make dist`
```

`make dist` builds the submission tarball and must exclude `dev_only/`, `data/`, `runs/`, and fail if any `*.py`, `*.ipynb`, or `requirements*.txt` remains in the tree.

---

## 6. Conventions: shapes, layouts, numerics

### 6.1 Notation
`B` batch, `T` sequence length (runtime `T ≤ max_T`), `C` embedding dim, `NH` heads, `hs = C/NH`, `V` vocab, `L` layers, `N = B·T` rows.

### 6.2 Memory layout
- Row-major, contiguous, `real_t`.
- Linear weights are stored **`[OC, C]`** (PyTorch `nn.Linear` layout): `out[n,o] = Σ_c inp[n,c]·W[o,c] + b[o]`. The tied head uses `wte [V, C]` directly as `W` (so `OC = V`).
- Fused QKV output `qkv [B, T, 3C]`: for token `(b,t)`, offsets `[0,C)` = Q, `[C,2C)` = K, `[2C,3C)` = V; head `h` occupies `[h·hs, (h+1)·hs)` inside each third.
- Attention probabilities `att [B, NH, T, T]` (post-softmax, causal; entries with key index > query index are 0 and are skipped in serial compute). Pre-softmax scores are **not** stored (softmax backward needs only `att`).
- Saved for LN backward: `mean [N]`, `rstd [N]`.
- Logits `[N, V]`; softmax probabilities are stored in a separate `probs [N, V]`; per-position losses `[N]`.

### 6.3 Numerics
- Softmax and cross-entropy: subtract row max; accumulate sums in `double` where cheap (loss sum always `double`).
- LayerNorm: biased variance, `rstd = 1/sqrt(var + 1e-5)`.
- GELU tanh form: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`; backward is the exact derivative of that expression.
- Cross-entropy backward: `dlogits[n,v] += (probs[n,v] − 1[v == target[n]]) / N`.
- `-ffast-math` is forbidden (breaks NaN checks and cross-build reproducibility). Record exact compiler flags in every run manifest.
- NaN/Inf guard: the training loop checks loss and grad norm each step and aborts with a diagnostic (step, phase) if non-finite.

---

## 7. Data pipeline specification

### 7.1 Inputs (obtained by `scripts/download_data.sh`; URLs verified at task time)
- TinyStories raw text: train and validation files from the Hugging Face dataset repo `roneneldan/TinyStories` (stories separated by the literal `<|endoftext|>`).
- GPT-Neo tokenizer files `vocab.json` and `merges.txt` (e.g. from `EleutherAI/gpt-neo-125M`). These are data files; the BPE algorithm itself is ours.
- Cite dataset (Eldan & Li 2023, arXiv:2305.07759) and GPT-Neo in `THIRD_PARTY.md`.

### 7.2 Tokenizer (`tokenizer.h/.cpp`)
- Byte-level BPE exactly as GPT-2/GPT-Neo: GPT-2 byte→unicode table, pre-tokenization pattern, merge-rank lookup, word-level result cache.
- Pre-tokenization pattern (GPT-2): contractions `'s 't 're 've 'm 'll 'd`, optional-space + letters, optional-space + digits, optional-space + other non-space symbols, whitespace runs (with the trailing-whitespace lookahead rule). Hand-code the scanner (no `std::regex`; it lacks `\p{L}`). Handle UTF-8: treat any non-ASCII codepoint as a letter unless it is in a small explicit set of Unicode punctuation/space you verify against golden tests (TinyStories contains a few curly quotes and similar).
- Minimal bespoke JSON reader sufficient for the flat `string → int` `vocab.json` (must handle `\uXXXX` escapes). A cited open-source JSON library is allowed instead; record it in `THIRD_PARTY.md`.
- API: `Tokenizer::load(vocab_json, merges_txt)`, `encode(std::string_view) -> std::vector<uint32_t>` (full 50,257 id space), `decode(ids) -> std::string`.
- **Golden test:** `tests/golden/tokenizer_cases.txt` ≥ 200 cases (text + expected ids) generated offline by the HF tokenizer: contractions, quotes, newlines, multiple spaces, digits, emoji/non-ASCII, empty string, very long word. Exact equality required.

### 7.3 `prepare_data` tool (two-pass)
1. Pass 1: read the story files, tokenize every story, append `<|endoftext|>` (GPT-Neo id 50256) after each, write raw GPT ids as `uint16` to a temp file; accumulate a `uint64 count[50257]` histogram. Flags: `--max_stories N` / `--max_tokens N` to truncate the **train** set.
2. Build vocabulary of size **V = 10,000**: id 0 = `<unk>`, id 1 = `<|endoftext|>`, ids 2..9999 = the 9,998 most frequent remaining GPT-Neo tokens (ties broken by lower GPT id, so the result is deterministic). Frequencies come from the **training** data only.
3. Pass 2: remap all ids (unmapped → 0) and write `data/train.bin`, `data/val.bin` (same map for val), plus `data/vocab_map.txt` (`new_id<TAB>gpt_id<TAB>escaped_token_string`) and `data/meta.txt` (story/token counts, **coverage = fraction of tokens that are not `<unk>`**, vocab hash).
4. `.bin` format: 32-byte header (`magic "TSTK"`, `u32 version`, `u32 vocab_size`, `u64 num_tokens`, 8 reserved) followed by `uint16` tokens, little-endian. Loader `mmap`s it.
- Report coverage in `meta.txt` and the README. If coverage < ~98% investigate before training (likely a tokenizer bug or wrong vocab selection).
- Expect the full train split to be on the order of 0.4–0.5 B tokens; the real count comes from `meta.txt`.

### 7.4 DataLoader (`dataset.h/.cpp`)
- Treat the token file as one packed stream (stories concatenated with EOS; no cross-document attention masking, **(DECISION)** for simplicity and parity with the paper's setup).
- Chunk `i` covers tokens `[i·T, i·T + T]` (T+1 tokens: `x = chunk[0..T)`, `y = chunk[1..T]`). Number of chunks `= (num_tokens − 1) / T`.
- Epoch order: seeded Fisher–Yates permutation of chunk indices (`Rng` stream 100 + epoch). Step `s` of an epoch uses permutation entries `[s·gB, (s+1)·gB)` where `gB = micro_batch × world_size`; rank r takes its contiguous sub-slice (§4.5). Remainder chunks that do not fill a batch are dropped per epoch.
- API: `next_batch(int32_t* x, int32_t* y)` fills host buffers (`[B,T]` each); `state()/set_state()` for checkpoint resume; `epoch()`, `steps_per_epoch()`.
- Tests: determinism for fixed seed; **sharding equivalence** (concatenating rows from 4 loaders with `(rank,4)` and `B=8` equals one loader with `B=32`, `(0,1)`); no duplicate chunks within an epoch; resume reproduces the same next batch.

---

## 8. Operator specifications (`ops.h`)

All functions: `namespace ops`, pointers may be device pointers in later builds, no STL types, no allocation, no global state, no exceptions. `real_t` is `float` (or `double` in the gradcheck build). Sizes are `int` unless they can exceed 2³¹ (use `size_t`; logits `N·V = 8.2e7` fits `int` but use `size_t` for products).

```cpp
// Embedding: out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:]
void embedding_forward (real_t* out, const int32_t* tokens, const real_t* wte, const real_t* wpe, int B, int T, int C);
void embedding_backward(real_t* dwte, real_t* dwpe, const real_t* dout, const int32_t* tokens, int B, int T, int C);

// LayerNorm over last dim, N rows
void layernorm_forward (real_t* out, real_t* mean, real_t* rstd, const real_t* inp,
                        const real_t* gamma, const real_t* beta, int N, int C);
void layernorm_backward(real_t* dinp, real_t* dgamma, real_t* dbeta, const real_t* dout,
                        const real_t* inp, const real_t* gamma, const real_t* mean, const real_t* rstd,
                        int N, int C);

// Linear: out[N,OC] = inp[N,C] * W[OC,C]^T (+ bias[OC], nullable)
void matmul_forward (real_t* out, const real_t* inp, const real_t* W, const real_t* bias, int N, int C, int OC);
// dinp += dout*W ; dW += dout^T*inp ; dbias += colsum(dout)  (dbias nullable)
void matmul_backward(real_t* dinp, real_t* dW, real_t* dbias, const real_t* dout,
                     const real_t* inp, const real_t* W, int N, int C, int OC);

// Causal multi-head attention on fused qkv[B,T,3C] -> out[B,T,C]; att[B,NH,T,T] saved
void attention_forward (real_t* out, real_t* att, const real_t* qkv, int B, int T, int C, int NH);
// scratch: B*NH*T*T reals (may be unused by serial; CUDA batched-GEMM version will use it)
void attention_backward(real_t* dqkv, real_t* scratch, const real_t* dout, const real_t* qkv,
                        const real_t* att, int B, int T, int C, int NH);

void gelu_forward (real_t* out, const real_t* inp, size_t n);
void gelu_backward(real_t* dinp, const real_t* inp, const real_t* dout, size_t n);

// out = a + b ; backward: da += dout, db += dout
void residual_forward (real_t* out, const real_t* a, const real_t* b, size_t n);
void residual_backward(real_t* da, real_t* db, const real_t* dout, size_t n);

// Softmax + cross entropy. losses[N] per-position; returns nothing (host reduces via helper below)
void softmax_xent_forward (real_t* probs, real_t* losses, const real_t* logits, const int32_t* targets, int N, int V);
void softmax_xent_backward(real_t* dlogits, const real_t* probs, const int32_t* targets, int N, int V); // += (p-onehot)/N
double sum_reduce(const real_t* x, size_t n);            // double-accumulated, used for mean loss

// Optimizer / clipping (also device-capable later)
double sumsq_reduce(const real_t* x, size_t n);          // for global grad norm
void   scale_inplace(real_t* x, real_t s, size_t n);
void   adamw_step(real_t* p, const real_t* g, real_t* m, real_t* v, size_t n,
                  real_t lr, real_t beta1, real_t beta2, real_t eps, real_t weight_decay, int step /*1-based*/);
void   zero(real_t* x, size_t n);
```

Serial implementation guidance:
- `matmul_*`: the baseline hot spot. Use a cache-blocked loop nest over row-major data with a vectorizable inner loop (let `-O3 -march=native` auto-vectorize; no intrinsics). Document the blocking parameters in the source. Never implement as a naive `i,j,k` triple loop with strided access to `W`.
- `attention_*`: structured as **three explicit stages** (scores = QKᵀ·scale with causal skip → row softmax → PV) mirroring the later batched-GEMM + masked-softmax CUDA design. Serial computes only the lower triangle; document that in the file header and benchmark notes, since CUDA may compute the full tile and mask.
- AdamW update: `m = β1·m + (1−β1)·g; v = β2·v + (1−β2)·g²; m̂ = m/(1−β1ᵗ); v̂ = v/(1−β2ᵗ); p −= lr·(m̂/(√v̂ + eps) + wd·p)` (decoupled weight decay as in the AdamW paper). The training loop calls `adamw_step` once per tensor in the layout (passing `wd = 0` for non-decay tensors).

### Model API (`model.h`)
```cpp
struct Model;   // holds cfg, layout, params/grads/m/v pointers, activation arena, (B,T) capacity
Model* model_create(const ModelConfig&, int B, int T);          // allocates via backend shim
void   model_init_weights(Model*, uint64_t seed);                // host-side init, then copy_h2d
float  model_forward(Model*, const int32_t* tokens, const int32_t* targets /*nullable*/, int B, int T); // returns mean loss (host float) or NAN if targets null
void   model_backward(Model*);                                   // uses saved activations; accumulates into grads
const float* model_logits(Model*);                               // for generation (last-position slice via helper)
void   model_free(Model*);
```
`model_forward` takes runtime `B,T ≤` capacity (generation uses `B=1`, arbitrary `T ≤ max_T`).

---

## 9. Training step specification

```
for step in 1..max_steps:
    lr = schedule(step)
    t.data:  loader.next_batch(x_host, y_host); copy_h2d(x_dev, y_dev)
    t.fwd:   loss = model_forward(model, x, y, B, T)
    t.zero:  zero(grads)
    t.bwd:   model_backward(model)
    t.comm:  comm_allreduce_avg(grads, num_params_padded)          // no-op in serial
             loss = comm-averaged loss                              // no-op in serial
    t.opt:   gnorm = sqrt(sumsq_reduce(grads)); if gnorm > clip: scale_inplace(grads, clip/gnorm)
             for tensor in layout: adamw_step(...)
    check_finite(loss, gnorm)
    log row every log_every steps; val loss every eval_every steps (fixed val batches: first `val_batches` chunks, same every time)
    checkpoint every ckpt_every steps and at the end
```

**Run outputs** under `runs/<run_name>/`: `manifest.txt` (git hash if available, full config, compiler + flags, CPU model, build type, date), `log.csv`, `ckpt_<step>.bin`, `ckpt_latest.bin`, `samples_<step>.txt`.

**`log.csv` columns (fixed for all four versions):**
`step, tokens_seen, train_loss, val_loss, val_ppl, lr, grad_norm, tok_per_s, step_ms, data_ms, fwd_ms, bwd_ms, comm_ms, opt_ms` (`val_*` empty when not evaluated).

**Checkpoint format** (little endian): header (`"TSCK"`, version, ModelConfig fields, `step`, `tokens_seen`, loader state, `num_params`, `num_params_padded`) then `params`, `m`, `v`. Resume must be bit-exact in the serial build (test in P4).

---

## 10. Validation strategy

| Layer | Test | Pass criterion |
|---|---|---|
| Unit: each op | Forward vs hand-computed tiny case; backward vs finite difference | see below |
| Gradcheck | Tiny model (`L=2,C=16,NH=2,V=32,T=8,B=2`) with init weights ×3, `real_t=double`, 4th-order 5-point central difference, h = 3e-4 (see `docs/DECISIONS.md`, 2026-11-18), up to 20 random indices per tensor **plus** first/last index; `qkv_b` is sampled from its query and value slices only (the key-bias slice has a structurally zero gradient, see `docs/DECISIONS.md`, 2026-11-18) | max relative error `|a−n|/max(|a|,|n|,1e-8)` < 1e-6 for all sampled entries; **and** per layer `max|d key-bias| ≤ 1e-12 · max|d q/v-bias|` |
| Golden (PyTorch, offline) | `dump_init` writes init weights + a fixed batch; `dev_only/gen_golden.py` builds an architecturally identical PyTorch model in float64, writes loss, a logits slice, and every tensor's gradient to `tests/golden/*.bin`; C++ test loads them with **no Python at test time** | float32 serial vs float64 golden: loss rel err < 1e-5; grads per-tensor rel-L2 err < 1e-4; after 5 AdamW steps (golden lr 1e-3) params rel-L2 < 1e-4, with each `qkv_b` key slice instead bounded by `max|value| ≤ steps·lr` (structural zero, see `docs/DECISIONS.md`, 2026-11-18, P4.1) |
| Init loss | Fresh `base` model on real data | initial loss within ±0.15 of ln(10000) ≈ 9.21 |
| Overfit | `smoke` config, one fixed batch, 300 steps | loss < 0.1 |
| Smoke training | `smoke` on a 100k-token real subset committed as `tests/fixtures/smoke_*.bin` (was ~1M; see `docs/DECISIONS.md`, 2026-11-18, P4.5), 200 steps | val loss decreases monotonically (coarsely) and ends below 6.0; run time under 1 min |
| Determinism | Same seed, same binary, 2 runs × 20 steps | log loss values bit-identical; final param checksum identical |
| Resume | Train 20 steps = train 10 + checkpoint + resume 10 | bit-identical params |
| Causality | Perturb token at position `t`; logits at positions `< t` | exactly unchanged |
| Param count | Layout for each preset | matches §2.2 exactly |
| Sanity vs paper | `small`/`base` on enough data | paper reports val loss ≈1.9 for 2L/128d, ≈1.6 for 2L/256d, ≈1.5 for 4L/256d after full training on the full dataset. Truncated serial runs will be worse (roughly 2–3); that is **not** a failure. A failure is loss stuck near 6–7 (unigram level) after thousands of steps |

**Cross-version equivalence protocol (used by Phases B–D, defined here):** same seed, same config, same global batch; compare serial vs X at step 0: loss rel err < 1e-4, per-tensor gradient rel-L2 < 1e-3 (CUDA reductions reorder sums; TF32 if ever enabled needs a looser documented bound); after 100 steps, val loss within 1% relative. `tests/equiv_*.cpp` and golden files from serial are the reference.

---

## 11. Phased task breakdown

Week mapping follows the proposal. **Owner** column is the proposal's suggested split; reassign as needed. P0 is done jointly at the start of week 7 because everything depends on it.

### Phase 0: Scaffold and seams (W7, both, ~2 days)

| ID | Task | DoD |
|---|---|---|
| P0.1 | Repo skeleton (§5), `Makefile` with `BACKEND`/`COMM`/`REAL` variables, targets `all test clean dist debug`. Release flags `-O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic`; debug `-O0 -g -fsanitize=address,undefined`. | `make`, `make test`, `make debug` run on a clean checkout; unsupported BACKEND/COMM combos `#error` |
| P0.2 | `common.h` (`real_t`, `CHECK(cond,msg)`, `CUDA_CHECK`/`MPI_CHECK` stubs), `config.h` (ModelConfig, TrainConfig, `key=value` file + `--key value` CLI parser, preset files) | unknown keys are errors; round-trips config to `manifest.txt` |
| P0.3 | `rng.h`, `timer.h`, `backend.h` + CPU impl, `comm.h` + serial impl, bespoke `tests/test.h` (ASSERT_EQ/NEAR, summary exit code) | RNG known-answer test for first 5 outputs of a fixed seed (values recorded once, committed); `normal()` mean/std within 1% over 1e6 samples |
| P0.4 | `layout.h/.cpp`: `make_param_layout`, `make_act_layout` per §4.3, §6.2 | param counts equal §2.2 for all presets tied/untied; offsets 64-byte aligned; no overlaps (test) |

### Phase 1: Data pipeline (W7, Conor)

| ID | Task | DoD |
|---|---|---|
| P1.1 | `scripts/download_data.sh` (dataset files + tokenizer files into `data/`, checksum print) | idempotent; documents source URLs and versions in `THIRD_PARTY.md` |
| P1.2 | Tokenizer (§7.2) | 100% pass on golden tokenizer cases; decode(encode(s)) == s for all ASCII/UTF-8 cases |
| P1.3 | `prepare_data` (§7.3) | produces `train.bin`, `val.bin`, `vocab_map.txt`, `meta.txt`; coverage reported; `--max_stories` works; deterministic output (hash identical across runs) |
| P1.4 | DataLoader (§7.4) | all four loader tests pass |
| P1.5 | `inspect_data` app: print N random decoded training sequences | visually sane English; EOS boundaries visible |

### Phase 2: Forward pass (W7–W8)

| ID | Task | Owner | DoD |
|---|---|---|---|
| P2.1 | embedding, layernorm, residual, gelu forward + unit tests | Conor | hand-computed cases match to 1e-6 |
| P2.2 | matmul forward (blocked) + test vs naive reference on random shapes incl. non-multiples of block size | Conor | exact-shape coverage: `N,C,OC` ∈ {1, 7, 64, 100, 256, 1024} subset |
| P2.3 | attention forward (3-stage, causal) + causality test | Vincent | matches naive reference; causality test passes; softmax rows sum to 1 within 1e-6 |
| P2.4 | softmax_xent forward + `sum_reduce` | Conor | loss on uniform logits = ln(V) |
| P2.5 | `model_create/init/forward` wiring (pre-LN block, tied head, final LN) | Conor | init loss within ±0.15 of ln(V) on real data; activation arena printed in MB at startup |
| P2.6 | `dump_init` app (writes init weights + fixed batch in the golden `.bin` format); `dev_only/gen_golden.py` (forward part) | Vincent | forward golden test passes (§10) |

### Phase 3: Backward pass and gradient check (W8)

| ID | Task | Owner | DoD |
|---|---|---|---|
| P3.1 | embedding, layernorm, residual, gelu backward | Conor | per-op finite-difference tests pass in double build |
| P3.2 | matmul backward (blocked; accumulate semantics) | Conor | FD test; verifies `+=` by calling twice and checking 2× |
| P3.3 | attention backward (incl. softmax backward: `dpre = att ⊙ (datt − Σ att·datt)`) | Vincent | FD test; scratch parameter accepted |
| P3.4 | softmax_xent backward | Conor | FD test |
| P3.5 | `model_backward` wiring, tied-embedding gradient accumulation from both paths | Conor | no gradient on padding/unused entries; all-zero check on tensors that must be zero |
| P3.6 | Full-model **gradcheck** app/test (§10) in `REAL=double` build | Vincent | all tensors pass tolerance; test prints worst relative error per tensor |
| P3.7 | Golden backward comparison (extend `gen_golden.py`) | Vincent | §10 golden criteria met |

### Phase 4: Optimizer, loop, checkpointing (W8)

| ID | Task | Owner | DoD |
|---|---|---|---|
| P4.1 | `adamw_step`, `sumsq_reduce`, `scale_inplace`, LR schedule, decay mask from layout | Vincent | AdamW matches golden 5-step trajectory; schedule unit test at warmup start/end, mid, final |
| P4.2 | Training loop (§9) with PhaseTimer, finite checks, CSV + manifest logging | Conor | `smoke` run completes and writes all outputs |
| P4.3 | Checkpoint save/load/resume (§9) | Conor | resume bit-exact test |
| P4.4 | Validation pass (fixed val batches, no backward) + perplexity | Vincent | val loss on a fresh model ≈ ln(V) |
| P4.5 | Overfit test and `smoke` training test wired into `make test` (smoke under 1 min) | Vincent | §10 criteria |

### Phase 5: Serial correctness sign-off (W9, both, "Serial Complete: Sync")

| ID | Task | DoD |
|---|---|---|
| P5.1 | Determinism, causality, resume, param-count tests in `make test` | green |
| P5.2 | First real run: `small` preset on a truncated train set (suggest 20–50 M tokens) to a loss plateau you can defend | train/val curves plotted from `log.csv` (plotting script lives in `dev_only/` or spreadsheet) |
| P5.3 | Review against the **Serial Complete gate**: every box in §14 checked | signed off in `docs/DECISIONS.md` |

### Phase 6: Generation and prompt evaluation (W9)

| ID | Task | Owner | DoD |
|---|---|---|---|
| P6.1 | `generate` app: load checkpoint, encode prompt (host tokenizer), loop forward with `B=1` and a sliding window of the last `max_T` tokens, sample (`--temperature`, `--top_k`, `--greedy`, `--seed`), stop at EOS or `--max_new`, decode | Conor | greedy output reproducible; reproduces the paper's style of "continues the story" behavior at least grammatically on `small` |
| P6.2 | `eval/prompts.txt` (10 fixed prompts modeled on the paper's examples: factual, reasoning, consistency, story openers) + `eval` app that writes completions to `runs/<run>/samples_<step>.txt` using fixed seed and both greedy and temp 0.8 | Vincent | file layout stable across versions |
| P6.3 | `eval/scores.csv` template (`milestone, prompt_id, grammar, consistency, sense` each 0–5, rater, notes) and a short scoring rubric in `eval/RUBRIC.md` | Vincent | rubric lets two raters agree within ±1 on a pilot set |

### Phase 7: Benchmarking, hardening, handoff (W9–W10 buffer)

| ID | Task | DoD |
|---|---|---|
| P7.1 | `bench` app (§12) emitting one CSV row per run. **Must handle the cold page cache:** random-chunk reads of the mmapped token file are slow until it is cached, for far more than 3 warmup steps (measured in P4.2: smoke run 54.5 s cold vs 5.0 s warm, nearly all in `data_ms`; see `docs/DECISIONS.md`, "Training loop details"). Proposed: `bench` reads the whole token file once before the first timed step and records that it did; the rule is the same for every build and gets a DECISIONS entry plus a §12 line when adopted | reproducible within ±3% across 3 runs |
| P7.2 | Profile serial (`perf` or `gprof`, dev-time only): confirm matmul dominates; record % by phase in `docs/PROFILE.md` | profile committed as text |
| P7.3 | Complete `README.md` (maintained since P3.1): add exact commands to reproduce a smoke run, a `small` run and a benchmark, plus the machine spec table; re-verify every command from a clean clone | a teammate can reproduce from a clean clone in under 30 min of attended time; every README command verified in that run |
| P7.4 | `docs/HANDOFF.md`: for each `ops::` function, a one-paragraph note of what its CUDA kernel must preserve (shapes, accumulation, tolerance), plus the memory arena sizes for each preset | complete for every function in `ops.h` |
| P7.5 | Stub `src/ops/ops_cuda.cu` and `src/comm/comm_mpi.cpp` that compile (or `#error` clearly) under the right Makefile variables | `make BACKEND=cuda` / `COMM=mpi` fail with a clear message or build stubs |

---

## 12. Benchmark methodology (defined once, reused by all versions)

- **Workload definition:** `(preset, global_batch, T, seed, steps)`. Reference workloads: **W-A** = `small`, 50 steps, B=32; **W-B** = `base`, 20 steps, B=32 (if tolerable); **W-C** = `smoke`, 200 steps.
- Discard the first 3 steps as warmup. Report **median** step time plus mean, tokens/s, and the per-phase breakdown. Run ≥3 times; report the spread.
- Fixed per run: input data file, seed, compiler and flags (recorded in `manifest.txt`), machine (CPU model, core count, RAM; GPU model and driver for CUDA), no other heavy processes. Note CPU governor/turbo state if known.
- **Speedup** = serial time ÷ version time **on the identical workload** (same global batch, same steps). Separately report **throughput scaling** (tokens/s) and **data processed in a fixed wall-clock budget** (the proposal's "how much more of the dataset" metric): run each version for a fixed number of minutes and record tokens consumed and final val loss.
- `bench` output CSV columns: `version, preset, global_batch, T, steps, ranks, median_step_ms, mean_step_ms, tok_per_s, data_ms, fwd_ms, bwd_ms, comm_ms, opt_ms, peak_mem_MB, machine_id, git_hash`.
- Never write a speedup number into a report that is not traceable to a row in a committed CSV.

---

## 13. Compute budget (estimates to plan with; measure to confirm)

FLOPs per step ≈ `6·N_matmul·B·T` (+ attention terms), with `B·T = 8,192`:

| Preset | Matmul params (blocks + head) | ≈ FLOP/step | At 10 GFLOP/s serial | At 2 GFLOP/s (poor) |
|---|---|---|---|---|
| `small` | 1.67 M | ≈ 9×10¹⁰ | ≈ 9 s | ≈ 45 s |
| `base` | 5.7 M | ≈ 3×10¹¹ | ≈ 30 s | ≈ 150 s |

Implications (this is why the proposal truncates data):
- 1,000 `base` steps (8.2 M tokens) ≈ 8 h serial at 10 GFLOP/s. The full dataset (~0.5 B tokens ≈ 60 k steps) is **out of reach** serially; do not attempt it.
- In `small`, the 128×10,000 output head dominates compute (~75% of FLOPs). Expect matmul to be >90% of runtime; the serial `matmul` quality determines baseline honesty.
- Serial full runs: use `small`, truncated train set, tens of M tokens. Use W-A/W-B fixed-step benchmarks for speedup tables. Later versions then show both identical-workload speedup and "more data in the same time".
- Memory: `base` at B=32,T=256 needs roughly 1.5–2 GB of activations (logits and probs alone ≈ 330 MB each). Print the exact arena size at startup. Check the cluster nodes/GPUs can hold it.

---

## 14. Serial Complete gate (all must be true before Phases B–D start)

- [ ] `make test` green, including gradcheck (double build), golden forward/backward/AdamW, determinism, resume, causality, param counts
- [ ] `smoke` and `small` runs complete; loss curves plotted; no NaNs
- [ ] `generate` produces grammatical story continuations from the `small` checkpoint
- [ ] Prompt evaluation run for the serial milestone scored and stored in `eval/scores.csv`
- [ ] Benchmark CSV for W-A, W-C committed with machine spec
- [ ] `ops.h`, `backend.h`, `comm.h` frozen; `HANDOFF.md` written
- [ ] Checkpoint and `log.csv` formats frozen
- [ ] `THIRD_PARTY.md` complete; no Python outside `dev_only/`; `make dist` passes its purity check
- [ ] `README.md` current: every command in it run successfully from a clean clone; Status line says Phase A is complete

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tokenizer mismatch silently degrades data | Golden tests; coverage metric; `inspect_data` eyeballing |
| Subtle gradient bug (attention, tied embedding) | Double-precision gradcheck on every tensor; golden grads |
| Serial too slow to iterate | `smoke`/`small` presets; fixed-step benchmarks; verify matmul blocking early (P2.2) |
| Interface churn breaks later CUDA/MPI work | Contracts frozen at §14; changes require DECISIONS.md entry |
| FP32 reduction-order differences blur "matches serial" claims | Tolerances predefined in §10; compare gradients not just loss |
| LLM drift (extra deps, Python creep, refactors) | `CLAUDE.md` rules; `make dist` purity check; one task per prompt |
| Proposal timeline pressure | P6/P7 may use W10 buffer; P0–P5 are the critical path |
