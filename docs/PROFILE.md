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

## Serial attention (P2.3, to be added)

Will compute only the causal lower triangle (ROADMAP §8). That will be
documented here and in the benchmark notes when implemented.
