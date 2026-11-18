// matmul_cpu.cpp: linear-layer forward, cache-blocked, CPU (ROADMAP §8).
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

}  // namespace ops
}  // namespace tslm
