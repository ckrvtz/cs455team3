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

## attention_forward (`src/ops/attention_cpu.cpp`), P2.3

| Technique | Why |
|---|---|
| **Causal lower triangle only** | Scores, softmax and att·V visit only key positions t2 ≤ t, about half the work of a full T × T tile. The masked upper triangle of `att` is written as exact zeros, not computed. **Benchmark note:** a CUDA kernel that computes the full tile and masks it does about 2× the attention arithmetic. Speedups must be read with that in mind, and the bench notes for every version must say whether attention is triangular or full. |
| Three explicit stages per (b, h) | Scores for all rows, then softmax for all rows, then att·V. This mirrors the Phase B batched-GEMM, masked-softmax, batched-GEMM design. Not fused, on purpose. |
| Lane-partial score dot products | Same `LANES` technique as matmul (vectorizes without `-ffast-math`, deterministic). |
| att·V as an in-order multiply-add per key | `o[i] += a * v[i]` over the head dimension is element-wise, so it vectorizes with no reduction reordering. |

Not done: no fusing of the stages, no tiling over T (T ≤ 256, so each head's
rows fit in cache), no threads.
