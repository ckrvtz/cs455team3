# Hand-off to Phases B–D: what each kernel must preserve

This is for whoever writes the CUDA kernels (Phase B: one file per op
family, `src/ops/<family>_cuda.cu`, `ROADMAP_CUDA.md` D-B11) and the MPI
builds (Phases C, D). The contract is `include/tslm/ops.h`: the signatures
are frozen, and the serial implementations in `src/ops/*_cpu.cpp` are the
reference. Each function below gets one paragraph covering:

- shapes and layout;
- whether it overwrites or accumulates;
- special cases the serial tests rely on;
- the tolerance a CUDA version must meet against serial.

## Rules for every kernel

These come from `ops.h` and CLAUDE.md §4.

- **Pointers:** all pointers are device pointers in CUDA builds.
- **No hidden resources:** kernels must not allocate, throw, sync the host
  or keep global state. The only workspace is the `scratch` argument of
  `attention_backward`.
- **Overwrite vs accumulate:** `*_forward` OVERWRITES its outputs.
  `*_backward` ACCUMULATES (`+=`). With cuBLAS this means `beta = 1` for
  every gradient GEMM.
- **No aliasing:** outputs never overlap inputs.
- **Layout:** row-major everywhere (§6.2). A linear weight is `[OC, C]`. The
  fused `qkv` is `[B, T, 3C]` with Q, K, V thirds, and head `h` at
  `[h·hs, (h+1)·hs)` inside each third. `att` is `[B, NH, T, T]`.
- **Sizes:** counts that can exceed 2³¹ are `size_t`. Compute every index as
  `size_t` before multiplying; `B·T·V` is 8.2·10⁷ already.
- **Synchronisation:** launches may be asynchronous. `PhaseTimer` calls
  `dev_sync()` at every phase boundary, so timings are charged correctly.
- **Precision:** `real_t` is `float` in production. Do not enable TF32 or
  fast-math. If TF32 is ever used, the tolerances below need a documented,
  approved change (§10).

### How CUDA is checked

- **Binding criteria,** from the ROADMAP §10 cross-version protocol, with the
  same seed, config and global batch:
  - step-0 loss relative error < 1e-4 vs serial;
  - per-tensor gradient relative L2 error < 1e-3;
  - after 100 steps, val loss within 1%.
- **The serial test suite** must pass unchanged on the CUDA build: golden
  files, gradcheck in the double build, determinism within one build,
  resume, causality, the smoke and overfit tests.
- **Per-op tolerances in the paragraphs below are PROPOSED:** they are the
  serial accuracy plus room for reordered float sums. Phase B confirms or
  revises them through `docs/DECISIONS.md`. "rel-L2" means
  ‖cuda − serial‖₂ / ‖serial‖₂ over the output tensor, with both run on the
  same inputs.

## Ops

**`embedding_forward(out, tokens, wte, wpe, B, T, C)`.**
- **Computes:** `out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:]`. `out` is
  `[B,T,C]` and is overwritten; `wpe` has at least T rows.
- **Token ids** are validated on the host by the DataLoader. The kernel does
  not check them.
- **Tolerance:** pure gathers and one add, so bit-exact.

**`embedding_backward(dwte, dwpe, dout, tokens, B, T, C)`.**
- **Computes:** `dwte[tokens[b,t],:] += dout[b,t,:]` and
  `dwpe[t,:] += dout[b,t,:]`, accumulating.
- **Repeated tokens must all contribute** (tested), and `dwpe` sums over the
  batch. A CUDA version needs atomics or a sort/segmented reduction.
- **Tied model:** `dwte` already holds the output head's gradient when this
  runs (`model_backward` calls the head matmul first). Never overwrite it.
- **Determinism:** float atomics make the summation order, and therefore the
  last bits, run-dependent. Prefer a deterministic reduction, or document
  that the CUDA build is not bit-reproducible run to run (serial is).
- **Tolerance:** rel-L2 ≤ 1e-6.

**`layernorm_forward(out, mean, rstd, inp, gamma, beta, N, C)`.**
- **Computes:** per row, the mean and the biased variance (divide by C), eps
  1e-5, `rstd = 1/sqrt(var + eps)`, `out = (x − mean)·rstd·gamma + beta`.
- **Saved values:** `mean` and `rstd` are `[N]`, and backward needs exactly
  these.
- **Precision:** serial accumulates the row sums in double. Float with
  warp-level two-pass (or Welford) sums is acceptable within tolerance.
- **Tolerance:** out, mean and rstd each rel-L2 ≤ 1e-6.

**`layernorm_backward(dinp, dgamma, dbeta, dout, inp, gamma, mean, rstd, N, C)`.**
- **Computes:** with `xhat = (inp − mean)·rstd` and `g = dout·gamma`:
  - `dinp += rstd·(g − mean_c(g) − xhat·mean_c(g·xhat))`;
  - `dgamma += Σ_n dout·xhat`;
  - `dbeta += Σ_n dout`.
- **Accumulates into all three.** `dgamma`/`dbeta` are column reductions
  over N, so atomics or a two-stage reduce.
- **Saved values:** use the forward's saved `mean`/`rstd`; do not recompute
  them.
- **Tolerance:** rel-L2 ≤ 1e-5.

**`gelu_forward(out, inp, n)`.**
- **Computes:** the tanh approximation
  `0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))`, element-wise, overwriting.
- **Evaluate it as `x·σ(2u)`** (u = the tanh argument), as serial does
  (DECISIONS, P3.1). It is mathematically identical and avoids cancellation
  for negative x, where `1 + tanh(u)` loses all precision.
- **Tolerance:** rel-L2 ≤ 1e-6.

**`gelu_backward(dinp, inp, dout, n)`.**
- **Computes:** `dinp += dout · d/dx gelu(x)`, the exact derivative of the
  tanh form (not the erf GELU).
- **Accumulates;** same stable `σ` evaluation as forward.
- **Tolerance:** rel-L2 ≤ 1e-6.

**`residual_forward(out, a, b, n)` / `residual_backward(da, db, dout, n)`.**
- **Forward:** `out = a + b`, overwriting.
- **Backward:** `da += dout; db += dout`, accumulating.
- **Aliasing:** `da` and `db` are different buffers; the residual stream
  gradient and the branch gradient are separate in `model_backward`.
- **Tolerance:** bit-exact.

**`matmul_forward(out, inp, W, bias, N, C, OC)`.**
- **Computes:** `out[n,o] = Σ_c inp[n,c]·W[o,c] + bias[o]`, overwriting.
  `W` is `[OC,C]` (row-major, i.e. `inp · Wᵀ`), and `bias` may be null (the
  output head has none).
- **Shape constraints:** none. Shapes include N = 1 (generation) and the
  head (OC = V = 10,000).
- **Profile:** this and `matmul_backward` are 81% of serial sampled time
  (`docs/PROFILE.md`), so they decide the speedup. cuBLAS SGEMM with the
  transposes mapped correctly is the implementation (team decision D-B2;
  mapping in `ROADMAP_CUDA.md` §5.4).
- **Tolerance:** rel-L2 ≤ 1e-5 in FP32 (no TF32).

**`matmul_backward(dinp, dW, dbias, dout, inp, W, N, C, OC)`.**
- **Computes:**
  - `dinp[n,c] += Σ_o dout[n,o]·W[o,c]`;
  - `dW[o,c] += Σ_n dout[n,o]·inp[n,c]`;
  - `dbias[o] += Σ_n dout[n,o]` (`dbias` may be null).
- **All three accumulate** (cuBLAS `beta = 1`).
- **Tied head:** `dW` is `wte`'s gradient, which later also receives the
  embedding gradient. Overwriting would silently lose half of it; the
  golden test catches this.
- **Tolerance:** rel-L2 ≤ 1e-5 for each output.

**`attention_forward(out, att, qkv, B, T, C, NH)`.**
- **Computes:** causal multi-head attention, `hs = C/NH`.
  - `att[b,h,t,t2] = softmax_{t2≤t}(Q_h(t)·K_h(t2)/√hs)` with the row max
    subtracted;
  - `out[b,t,h·hs+i] = Σ_{t2≤t} att·V_h(t2)[i]`.
- **The upper triangle (t2 > t) of `att` must be exactly 0.** It is saved
  for backward, and the tests check it.
- **Lower triangle only:** serial computes only the causal lower triangle.
  The CUDA version uses cuBLAS strided-batched GEMMs (D-B2,
  `ROADMAP_CUDA.md` §5.6). Those compute the full T×T tile, which is then
  masked, so it does about 2× the score/att·V arithmetic, and benchmark
  notes must say so (`docs/PROFILE.md`).
- **Causality:** must be exact. Logits at positions < t do not change when
  token t changes (`test_signoff`, `memcmp`).
- **Tolerance:** out and att rel-L2 ≤ 1e-5.

**`attention_backward(dqkv, scratch, dout, qkv, att, B, T, C, NH)`.**
- **Computes:**
  - `datt = dY·Vᵀ`;
  - `dV += attᵀ·dY`;
  - softmax backward `dpre = att ⊙ (datt − rowsum(att ⊙ datt))`;
  - `dQ += dpre·K/√hs`;
  - `dK += dpreᵀ·Q/√hs`.
- **Accumulates into `dqkv`** (`[B,T,3C]`, same layout as `qkv`), over
  causal entries only.
- **`scratch`:** `B·NH·T·T` reals of workspace, contents undefined on
  return. It is the only workspace any kernel gets.
- **Structural zero:** the key-bias gradient is exactly zero in exact
  arithmetic (softmax shift invariance; DECISIONS). The CUDA value will be
  rounding noise. Do not compare that slice by relative error; the golden
  and gradcheck tests already exclude it with a direct check.
- **Tolerance:** rel-L2 ≤ 1e-5 on the Q and V slices; key slice
  max |value| ≤ 1e-6 · max |Q/V slice|.

**`softmax_xent_forward(probs, losses, logits, targets, N, V)`.**
- **Computes:** per row, subtract the max, exponentiate, normalise. `probs`
  is `[N,V]`, saved and overwritten.
- **Loss:** `losses[n] = −log probs[n, targets[n]]`, computed in log-sum-exp
  form (not `log` of a rounded probability). Per-position losses only; the
  mean comes from `sum_reduce` on the host side.
- **Precision:** serial sums each row in double. At V = 10,000 per row a
  float tree reduction is fine.
- **Profile:** 6.1% of serial time plus `exp` in libm (`docs/PROFILE.md`).
- **Tolerance:** losses rel ≤ 1e-6 per element; probs rel-L2 ≤ 1e-6.

**`softmax_xent_backward(dlogits, probs, targets, N, V)`.**
- **Computes:** `dlogits[n,v] += (probs[n,v] − 1[v = target])/N`,
  accumulating; the gradient of the MEAN loss.
- **Target entry:** compute it as `(p_t − 1)/N`, not `p_t/N − 1/N`.
- **Tied model:** `dlogits` feeds the head matmul's backward, whose `dW` is
  the tied `wte` gradient.
- **Tolerance:** rel-L2 ≤ 1e-6.

**`sum_reduce(x, n)` → double.**
- **Computes:** the sum of n values, accumulated in double and returned to
  the host. It is used for the mean loss over N = B·T positions.
- **CUDA version:** a double-accumulating reduction, then one `copy_d2h`.
- **Tolerance:** rel ≤ 1e-12 vs serial on the same float inputs; only the
  summation order differs.

**`sumsq_reduce(x, n)` → double.**
- **Computes:** the sum of squares in double over the whole flat `grads`
  buffer (n = `num_params_padded`; padding is 0).
- **Clipping:** `sqrt` of it is the global norm. The clip test is
  `norm > max_norm` with no epsilon, so keep it a double reduction.
- **MPI builds:** call it AFTER `comm_allreduce_avg(grads)`, so every rank
  computes the same norm (§4.5 step order).
- **Tolerance:** rel ≤ 1e-12.

**`scale_inplace(x, s, n)`.**
- **Computes:** `x[i] *= s`, used for clipping with `s = clip/norm`.
- **Tolerance:** bit-exact.

**`adamw_step(p, g, m, v, n, lr, beta1, beta2, eps, weight_decay, step)`.**
- **Updates `p`, `m`, `v` in place,** per element:
  - `m = b1·m + (1−b1)·g`;
  - `v = b2·v + (1−b2)·g²`;
  - `p −= lr·( m·ibc1 / (sqrt(v·ibc2) + eps) + wd·p )`.
- **Bias corrections:** `ibc1 = 1/(1−b1^step)` and `ibc2 = 1/(1−b2^step)`
  are computed ONCE per call in double on the host side of the kernel
  launch, then cast to `real_t`. `step` is 1-based.
- **Decay term:** uses `p` before the update.
- **Masking:** the caller passes `weight_decay = 0` for non-decayed tensors
  (biases, LayerNorm, `wpe`). It is called once per tensor in layout order.
  A fused multi-tensor kernel is allowed if it keeps the per-tensor `wd`.
- **Tolerance:** rel-L2 ≤ 1e-6 per tensor after one step. Note that a
  structurally-zero gradient (the key-bias slice) turns rounding noise into
  steps of size ≈ lr, so compare that slice by the bound in §10.

## Memory per preset

Computed from `make_param_layout` / `make_act_layout` (float, 4 bytes). The
arena includes the backward buffers and `att_scratch`.

| Preset (B × T) | Head | Parameters | params + grads + m + v | Activation arena | Device total |
|---|---|---|---|---|---|
| `smoke` (4 × 32) | tied | 333,792 | 5.1 MB | 15.2 MB | 20.3 MB |
| `smoke` (4 × 32) | untied | 653,792 | 10.0 MB | 15.2 MB | 25.2 MB |
| `small` (32 × 256) | tied | 1,709,568 | 26.1 MB | 1,241.8 MB | 1,267.9 MB |
| `small` (32 × 256) | untied | 2,989,568 | 45.6 MB | 1,241.8 MB | 1,287.5 MB |
| `base` (32 × 256) | tied | 5,785,088 | 88.3 MB | 1,962.1 MB | 2,050.4 MB |
| `base` (32 × 256) | untied | 8,345,088 | 127.3 MB | 1,962.1 MB | 2,089.4 MB |

- **Largest arena entries** (`small` and `base` alike): `logits`, `probs`
  and `d_logits`, at 81,920,000 elements = 312.5 MB each, so 937.5 MB of
  the arena is the V = 10,000 output.
- **Attention:** each layer's `att` is `B·NH·T·T`: 32.0 MB per layer for
  `small` and 64.0 MB for `base`, plus one `att_scratch` of the same size.
- **Sizing a smaller GPU:** reduce B; the arena scales linearly in B·T,
  except `att`, which scales in B·T².
- **Host side:** the serial process's peak RSS on W-A was 2,023.8 MB
  (`bench/serial_W-A.csv`). That is more than the 1,267.9 MB above, because
  it also counts the touched pages of the memory-mapped 927 MB token file
  and the host batch buffers.
- **CUDA reporting:** `peak_mem_MB` in the CUDA build should report device
  memory (`bench.h`).
- **Generation** (`generate`/`eval`) uses B = 1, T = `max_seq_len`, so its
  arena is 1/32 of the table's value for `small` and `base`.
