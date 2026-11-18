# Serial baseline: optimizations and profile

The serial build is the denominator of every speedup we report, so it must be
reasonable and honest (CLAUDE.md §7). This file lists **every** optimization in
the serial kernels and why it is there. The profile itself (time share per
phase and op) is added in P7.2. Benchmark numbers appear only in committed
`bench` CSVs (ROADMAP §12), never here unless they are copied from one.

Build: `-O3 -march=native -std=c++17`. No `-ffast-math`, intrinsics, OpenMP,
threads or BLAS. Compiler auto-vectorization is the only SIMD used.

## matmul_forward (`src/ops/matmul_cpu.cpp`), P2.2

| Technique | Parameters | Why |
|---|---|---|
| Lane-partial dot products | `LANES = 32 B / sizeof(real_t)` (8 float, 4 double) | Without `-ffast-math`, GCC will not vectorize a single floating-point sum reduction. Keeping `LANES` independent partial sums per output gives the compiler parallel sums it may vectorize without reordering any one sum. The result is deterministic. GCC `-fopt-info-vec` reports the lane loops "vectorized using 32 byte vectors". |
| Register tile | 4 × 4 outputs | Each loaded input vector is reused for 4 weight rows and vice versa: 8 loads per 16 multiply-adds. |
| Cache blocking | NB = 128 input rows, KC = 256 reduction columns, OB = 64 weight rows | One weight block (64 × 256 × 4 B = 64 KiB) stays in L2 across 128 input rows. The 8 row slices one micro-tile touches (8 KiB) fit in L1. Round, untuned values for typical 32–48 KiB L1 / 1–2 MiB L2. |
| Edge kernels | 4×1, 1×4, 1×1 instantiations; scalar C tail | Correct for any N, C, OC (tested on {1,7,64,100,256,1024} and block-boundary shapes). N = 1 (generation) runs entirely on 1×4 / 1×1 kernels. |

Not done, on purpose (so the CUDA comparison stays fair): no hand-written
SIMD intrinsics, no repacking weights into a different layout (the op cannot
allocate), no autotuning, no threads.

Summation-order note: outputs equal bias + Σ over KC blocks of (Σ over lanes
of each lane's in-order partial sum + scalar tail). This differs from a naive
left-to-right sum by normal rounding error, which the matmul test checks
against a rigorous order-independent error bound.

## matmul_backward (`src/ops/matmul_cpu.cpp`), P3.2

| Technique | Parameters | Why |
|---|---|---|
| Output-row ("axpy") loop order | none | `dinp[n,:] += dout[n,o]·W[o,:]` and `dW[o,:] += dout[n,o]·inp[n,:]` make the innermost loop run along a contiguous output row with independent elements, so it vectorizes without `-ffast-math` and without the forward pass's lane trick. Each element is summed in plain ascending order (over `o` for `dinp`, over `n` for `dW`/`dbias`). GCC `-fopt-info-vec` reports all five inner loops vectorized with 32-byte vectors. |
| Register tile | 4 output rows | 4 `dinp` rows share each loaded `W` row; 4 `dW` rows share each loaded `inp` row. |
| Cache blocking | same NB = 128, KC = 256, OB = 64 | `dinp`: a 64 × 256 `W` block (64 KiB, L2) is reused across 128 rows. `dW`: a 128 × 256 `inp` block (128 KiB, L2) feeds 4 `dW` row slices (4 KiB, L1). |
| Leftovers | one row at a time | Correct for any N, C, OC; tested on block-boundary shapes. |

Not done: no fusing of the three gradients into one pass, no transposed
copies (the op cannot allocate), no threads.

## attention_forward (`src/ops/attention_cpu.cpp`), P2.3

| Technique | Why |
|---|---|
| **Causal lower triangle only** | Scores, softmax and att·V visit only key positions t2 ≤ t, about half the work of a full T × T tile. The masked upper triangle of `att` is written as exact zeros, not computed. **Benchmark note:** a CUDA kernel that computes the full tile and masks it does about 2× the attention arithmetic. Speedups must be read with that in mind, and the bench notes for every version must say whether attention is triangular or full. |
| Three explicit stages per (b, h) | Scores for all rows, then softmax for all rows, then att·V. This mirrors the Phase B batched-GEMM, masked-softmax, batched-GEMM design. Not fused, on purpose. |
| Lane-partial score dot products | Same `LANES` technique as matmul (vectorizes without `-ffast-math`, deterministic). |
| att·V as an in-order multiply-add per key | `o[i] += a * v[i]` over the head dimension is element-wise, so it vectorizes with no reduction reordering. |

Not done: no fusing of the stages, no tiling over T (T ≤ 256, so each head's
rows fit in cache), no threads.

## attention_backward (`src/ops/attention_cpu.cpp`), P3.3

Same structure and caveat as forward: **causal lower triangle only**, in
three explicit stages per (b, h):
1. `datt = dY·Vᵀ` and `dV += attᵀ·dY`;
2. softmax backward in place;
3. `dQ += dpre·K`, `dK += dpreᵀ·Q`.

The `scratch` argument (B·NH·T·T) holds `datt`/`dpre`; only its lower
triangle is touched. The `datt` dot products use the lane technique; `dV`,
`dQ` and `dK` are multiply-adds over the head dimension (vectorized, fixed
order). Not done: no recomputation-free fusing, no tiling over T, no threads.

## softmax_xent_forward / sum_reduce (`src/ops/softmax_xent_cpu.cpp`), P2.4

No special optimization. One pass for the row max, one for exp plus the
double-precision sum, one to normalize. At V = 10,000 the `exp` calls and
the double row sum run as scalar code: vectorizing them would need
`-ffast-math` or a vector math library, which the serial baseline forbids.
The op is memory/exp-bound and much cheaper than the head matmul that
produces the logits. `sum_reduce` is a sequential double-precision sum
(deterministic).

## softmax_xent_backward (`src/ops/softmax_xent_cpu.cpp`), P3.4

No special optimization. One pass per row: `dlogits += probs·(1/N)` over the
non-target entries (contiguous, vectorized) and `(p_t − 1)·(1/N)` for the
target. Memory-bound; it reads `probs` once.

## sumsq_reduce / scale_inplace / adamw_step (`src/ops/optim_cpu.cpp`), P4.1

No special optimization. `sumsq_reduce` is a sequential double-precision
sum (deterministic, scalar). `scale_inplace` and `adamw_step` are
element-wise loops over each parameter tensor; they auto-vectorize because
`sqrt` and division are plain IEEE operations. The bias corrections are
computed once per call in double. All three are memory-bound and run once
per step, which is negligible next to the matmuls.

## Profile of the serial step (P7.2)

Workload W-A (`small`: 2 layers, C=128, V=10,000, B=32, T=256; tied head).
Machine: this project's cloud container, Intel Xeon @ 2.30GHz with
AVX-512, 4 cores (1 used), 16 GB. Build: release flags, plus `-pg` for the
op table only.

### Share by phase

From `bench/serial_W-A.csv` (3 runs, mean of the per-run phase means; the
phases sum to the mean step of 3609.9 ms):

| Phase | ms/step | Share |
|---|---|---|
| backward (incl. grad zeroing) | 2197.6 | 60.9% |
| forward | 1404.5 | 38.9% |
| optimizer (clip + AdamW) | 7.6 | 0.2% |
| data (batch copy) | 0.13 | < 0.01% |
| comm (no-op in serial) | 0.000 | 0% |

### Share by op

The raw output is `docs/profile/serial_W-A_gprof.txt`, from
`scripts/profile.sh` at commit `90873e7` (13 steps, the first 3 being
warmup). gprof sampled 42.77 s. The same run's 13 steps took 13 × 3651.1 ms
(mean) ≈ 47.5 s, so about 90% of step time was sampled. The remainder is
time inside libm/libc (`exp`, `tanh`, `memset`), which is not compiled with
`-pg` and so not attributed, plus instrumentation overhead. The profiled
run's median step (3688.2 ms) is within 2.2% of the uninstrumented W-A
medians, so `-pg` barely perturbs the timing.

| Op | Self time | Share of sampled |
|---|---|---|
| **matmul (all)** | 34.81 s | **81.4%** |
| ├ `matmul_backward` (dinp and dweight) | 24.35 s | 56.9% |
| └ `matmul_forward` (4×4 micro-kernel + driver) | 10.46 s | 24.5% |
| `softmax_xent_forward` (V = 10,000, scalar `exp`) | 2.59 s | 6.1% |
| `attention_backward` | 1.82 s | 4.3% |
| `attention_forward` | 1.00 s | 2.3% |
| `softmax_xent_backward` | 0.67 s | 1.6% |
| `gelu_forward` + `gelu_backward` | 0.68 s | 1.6% |
| `layernorm_forward` + `layernorm_backward` | 0.58 s | 1.4% |
| `_init` (PLT stubs into libm/libc) | 0.45 s | 1.1% |
| AdamW, sum of squares, scale, residual, embedding | 0.11 s | 0.3% |

### Findings

- **Matmul dominates: 81% of sampled time, roughly 73% of step time.** This
  is below ROADMAP §13's ">90%" planning estimate. The rest is mostly ops
  that §13 did not count:
  - the cross-entropy forward over V = 10,000 (6.1%, plus its unattributed
    `exp` calls in libm);
  - causal attention at T = 256 (6.6%);
  - element-wise ops (about 3%).

  CUDA speedups will therefore depend on more than the matmul. These ops
  are the next candidates.
- **matmul backward costs 2.33× matmul forward** (24.35 s vs 10.46 s) for
  2× the arithmetic: it computes both dinp and dweight. Its kernels are
  about 14% less efficient per FLOP than the forward micro-kernel.
  - This is recorded, not tuned. The serial baseline stays "reasonable,
    not heroic" (CLAUDE.md §7), and the backward blocking is documented
    above.
- **The optimizer and the data path are negligible** (0.2% and < 0.01% of
  step time, with the token file pre-read by `bench`).
- **Per-call counts confirm the wiring:** 117 matmul calls = 13 steps × 9
  linears (4 per layer × 2 layers + the head), and 26 attention calls =
  13 × 2 layers.
