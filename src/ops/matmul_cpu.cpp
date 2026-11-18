// matmul_cpu.cpp: linear-layer forward and backward, cache-blocked, CPU (ROADMAP §8).
//
// Shapes (row-major): inp [N,C], W [OC,C] (PyTorch nn.Linear layout),
// bias [OC] or nullptr, out [N,OC].
// Math:      out[n,o] = sum_c inp[n,c] * W[o,c] + (bias ? bias[o] : 0)
// Semantics: OVERWRITES out (first set to bias or 0, then accumulated).
//
// Why this structure (serial baseline rules: -O3 -march=native, no
// -ffast-math, no intrinsics, no threads, no BLAS):
//  * Both operands of every dot product are contiguous rows (inp row n, W
//    row o), so no access is strided. But a plain `s += a[c]*b[c]` loop is
//    one floating-point reduction, and without -ffast-math GCC may not
//    reorder it, so it would not vectorize.
//  * Instead each output gets LANES independent partial sums,
//    acc[l] += a[c+l] * b[c+l]. Each acc[l] is its own sum, so vectorizing
//    the l loop changes no single sum's order: no reassociation is needed and
//    the result is deterministic. LANES = 32 bytes / sizeof(real_t), one AVX2
//    register (8 floats, 4 doubles). The lanes are added together once per
//    KC block.
//  * Register tile MR x NR = 4 x 4 outputs: every inp vector loaded is used
//    for 4 W rows and vice versa (8 loads per 16 multiply-adds), instead of
//    2 loads per multiply-add for one dot product at a time.
//  * Cache blocks: NB = 128 rows of inp, KC = 256 columns of C, OB = 64 rows
//    of W. One W block is 64 x 256 x 4 B = 64 KiB (L2-resident) and is
//    reused for all 128 inp rows. One inp block is 128 x 256 x 4 B = 128 KiB.
//    The 4 inp row slices and 4 W row slices that one micro-tile touches
//    (8 x 1 KiB) fit in L1. These sizes are fixed, round values chosen for
//    typical 32-48 KiB L1 / 1-2 MiB L2 caches, not tuned per machine
//    (see docs/PROFILE.md).
//  * Leftover rows and columns (N or OC not a multiple of 4) use 4x1, 1x4
//    and 1x1 instantiations of the same kernel; a C tail shorter than LANES
//    is summed in scalar code.
// Summation order: out = bias + sum over KC blocks (in order) of
// (sum of lanes (sum over that lane's c in order) + scalar tail). It is fixed
// for a given (C, LANES), so results are bit-reproducible run to run.
//
// BACKWARD (ACCUMULATES, +=):
//   dinp[n,c] += sum_o dout[n,o] * W[o,c]     (dinp [N,C])
//   dW[o,c]   += sum_n dout[n,o] * inp[n,c]   (dW   [OC,C])
//   dbias[o]  += sum_n dout[n,o]              (dbias [OC], skipped if null)
// Unlike forward, every product here can be arranged so the innermost
// loop runs along a contiguous row of the OUTPUT: dinp row n gathers
// dout[n,o] * (W row o), and dW row o gathers dout[n,o] * (inp row n). The
// inner `x[c] += d * w[c]` loop has independent elements, so it vectorizes
// without reordering any sum, and no lane trick is needed. Each output
// element is summed in plain ascending order (over o for dinp, over n for
// dW and dbias), so results are deterministic.
//  * Register tile: 4 output rows at once (4 dinp rows sharing each W row,
//    or 4 dW rows sharing each inp row), so every loaded vector feeds 4
//    multiply-adds.
//  * Cache blocks reuse NB/KC/OB: dinp loops (KC slice, NB rows, OB rows of
//    W), keeping a 64 x 256 W block (64 KiB) in L2 across 128 rows. dW
//    loops (KC slice, NB rows of inp), keeping a 128 x 256 inp block
//    (128 KiB) in L2 while 4 dW row slices (4 KiB) stay in L1.
//  * Leftover rows (N or OC not a multiple of 4) go one row at a time.
#include "tslm/ops.h"

namespace tslm {
namespace ops {

namespace {

constexpr size_t LANES = 32 / sizeof(real_t);
constexpr size_t NB = 128;  // inp rows per cache block
constexpr size_t KC = 256;  // reduction (C) columns per cache block
constexpr size_t OB = 64;   // W rows (outputs) per cache block

// out[i*ldo + j] += sum_{k<kc} a[i*lda + k] * w[j*ldw + k] for i < MR, j < NR.
template <int MR, int NR>
inline void micro_kernel(const real_t* a, size_t lda, const real_t* w, size_t ldw, size_t kc, real_t* out,
                         size_t ldo) {
    real_t acc[MR][NR][LANES] = {};
    const size_t kv = kc / LANES * LANES;
    for (size_t k = 0; k < kv; k += LANES) {
        for (int i = 0; i < MR; ++i) {
            const real_t* ai = a + static_cast<size_t>(i) * lda + k;
            for (int j = 0; j < NR; ++j) {
                const real_t* wj = w + static_cast<size_t>(j) * ldw + k;
                for (size_t l = 0; l < LANES; ++l) acc[i][j][l] += ai[l] * wj[l];
            }
        }
    }
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR; ++j) {
            real_t s = 0;
            for (size_t l = 0; l < LANES; ++l) s += acc[i][j][l];
            const real_t* ai = a + static_cast<size_t>(i) * lda;
            const real_t* wj = w + static_cast<size_t>(j) * ldw;
            for (size_t k = kv; k < kc; ++k) s += ai[k] * wj[k];
            out[static_cast<size_t>(i) * ldo + static_cast<size_t>(j)] += s;
        }
    }
}

}  // namespace

void matmul_forward(real_t* out, const real_t* inp, const real_t* W, const real_t* bias, int N, int C,
                    int OC) {
    const size_t n_rows = static_cast<size_t>(N);
    const size_t c_dim = static_cast<size_t>(C);
    const size_t oc = static_cast<size_t>(OC);

    // Overwrite: start every output at its bias (or zero).
    for (size_t n = 0; n < n_rows; ++n) {
        real_t* o = out + n * oc;
        if (bias) {
            for (size_t j = 0; j < oc; ++j) o[j] = bias[j];
        } else {
            for (size_t j = 0; j < oc; ++j) o[j] = 0;
        }
    }

    for (size_t n0 = 0; n0 < n_rows; n0 += NB) {
        const size_t n1 = n0 + NB < n_rows ? n0 + NB : n_rows;
        for (size_t c0 = 0; c0 < c_dim; c0 += KC) {
            const size_t kc = c0 + KC < c_dim ? KC : c_dim - c0;
            for (size_t o0 = 0; o0 < oc; o0 += OB) {
                const size_t o1 = o0 + OB < oc ? o0 + OB : oc;
                size_t n = n0;
                for (; n + 4 <= n1; n += 4) {
                    const real_t* a = inp + n * c_dim + c0;
                    size_t o = o0;
                    for (; o + 4 <= o1; o += 4)
                        micro_kernel<4, 4>(a, c_dim, W + o * c_dim + c0, c_dim, kc, out + n * oc + o, oc);
                    for (; o < o1; ++o)
                        micro_kernel<4, 1>(a, c_dim, W + o * c_dim + c0, c_dim, kc, out + n * oc + o, oc);
                }
                for (; n < n1; ++n) {
                    const real_t* a = inp + n * c_dim + c0;
                    size_t o = o0;
                    for (; o + 4 <= o1; o += 4)
                        micro_kernel<1, 4>(a, c_dim, W + o * c_dim + c0, c_dim, kc, out + n * oc + o, oc);
                    for (; o < o1; ++o)
                        micro_kernel<1, 1>(a, c_dim, W + o * c_dim + c0, c_dim, kc, out + n * oc + o, oc);
                }
            }
        }
    }
}

void matmul_backward(real_t* dinp, real_t* dW, real_t* dbias, const real_t* dout, const real_t* inp,
                     const real_t* W, int N, int C, int OC) {
    const size_t n_rows = static_cast<size_t>(N);
    const size_t c_dim = static_cast<size_t>(C);
    const size_t oc = static_cast<size_t>(OC);

    // dinp[n, c0:c1] += sum_o dout[n,o] * W[o, c0:c1], 4 rows of dinp at a time.
    for (size_t c0 = 0; c0 < c_dim; c0 += KC) {
        const size_t kc = c0 + KC < c_dim ? KC : c_dim - c0;
        for (size_t n0 = 0; n0 < n_rows; n0 += NB) {
            const size_t n1 = n0 + NB < n_rows ? n0 + NB : n_rows;
            for (size_t o0 = 0; o0 < oc; o0 += OB) {
                const size_t o1 = o0 + OB < oc ? o0 + OB : oc;
                size_t n = n0;
                for (; n + 4 <= n1; n += 4) {
                    real_t* x0 = dinp + (n + 0) * c_dim + c0;
                    real_t* x1 = dinp + (n + 1) * c_dim + c0;
                    real_t* x2 = dinp + (n + 2) * c_dim + c0;
                    real_t* x3 = dinp + (n + 3) * c_dim + c0;
                    for (size_t o = o0; o < o1; ++o) {
                        const real_t d0 = dout[(n + 0) * oc + o], d1 = dout[(n + 1) * oc + o];
                        const real_t d2 = dout[(n + 2) * oc + o], d3 = dout[(n + 3) * oc + o];
                        const real_t* w = W + o * c_dim + c0;
                        for (size_t c = 0; c < kc; ++c) {
                            const real_t wc = w[c];
                            x0[c] += d0 * wc;
                            x1[c] += d1 * wc;
                            x2[c] += d2 * wc;
                            x3[c] += d3 * wc;
                        }
                    }
                }
                for (; n < n1; ++n) {
                    real_t* x = dinp + n * c_dim + c0;
                    for (size_t o = o0; o < o1; ++o) {
                        const real_t d = dout[n * oc + o];
                        const real_t* w = W + o * c_dim + c0;
                        for (size_t c = 0; c < kc; ++c) x[c] += d * w[c];
                    }
                }
            }
        }
    }

    // dW[o, c0:c1] += sum_n dout[n,o] * inp[n, c0:c1], 4 rows of dW at a time.
    for (size_t c0 = 0; c0 < c_dim; c0 += KC) {
        const size_t kc = c0 + KC < c_dim ? KC : c_dim - c0;
        for (size_t n0 = 0; n0 < n_rows; n0 += NB) {
            const size_t n1 = n0 + NB < n_rows ? n0 + NB : n_rows;
            size_t o = 0;
            for (; o + 4 <= oc; o += 4) {
                real_t* g0 = dW + (o + 0) * c_dim + c0;
                real_t* g1 = dW + (o + 1) * c_dim + c0;
                real_t* g2 = dW + (o + 2) * c_dim + c0;
                real_t* g3 = dW + (o + 3) * c_dim + c0;
                for (size_t n = n0; n < n1; ++n) {
                    const real_t* dr = dout + n * oc + o;
                    const real_t d0 = dr[0], d1 = dr[1], d2 = dr[2], d3 = dr[3];
                    const real_t* x = inp + n * c_dim + c0;
                    for (size_t c = 0; c < kc; ++c) {
                        const real_t xc = x[c];
                        g0[c] += d0 * xc;
                        g1[c] += d1 * xc;
                        g2[c] += d2 * xc;
                        g3[c] += d3 * xc;
                    }
                }
            }
            for (; o < oc; ++o) {
                real_t* g = dW + o * c_dim + c0;
                for (size_t n = n0; n < n1; ++n) {
                    const real_t d = dout[n * oc + o];
                    const real_t* x = inp + n * c_dim + c0;
                    for (size_t c = 0; c < kc; ++c) g[c] += d * x[c];
                }
            }
        }
    }

    // dbias[o] += sum_n dout[n,o]: rows in order, contiguous (vectorizable) over o.
    if (dbias) {
        for (size_t n = 0; n < n_rows; ++n) {
            const real_t* dr = dout + n * oc;
            for (size_t o = 0; o < oc; ++o) dbias[o] += dr[o];
        }
    }
}

}  // namespace ops
}  // namespace tslm
