// embedding_cpu.cpp: token + position embedding, CPU (ROADMAP §8).
//
// Shapes (row-major): tokens [B,T] int32; wte [V,C]; wpe [max_T,C] (rows 0..T-1
// used); out [B,T,C].
// Math:     out[b,t,c] = wte[tokens[b,t], c] + wpe[t, c]
// Semantics: forward OVERWRITES out. Token ids are trusted to be in [0, V)
// (the DataLoader checks every token against vocab_size; kernels never CHECK).
// Cost is memory-bound (2 reads + 1 write per element); the inner loop over c
// is contiguous and auto-vectorizes.
#include "tslm/ops.h"

namespace tslm {
namespace ops {

void embedding_forward(real_t* out, const int32_t* tokens, const real_t* wte, const real_t* wpe,
                       int B, int T, int C) {
    const size_t Cs = static_cast<size_t>(C);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const size_t row = static_cast<size_t>(b) * static_cast<size_t>(T) + static_cast<size_t>(t);
            const real_t* te = wte + static_cast<size_t>(tokens[row]) * Cs;
            const real_t* pe = wpe + static_cast<size_t>(t) * Cs;
            real_t* o = out + row * Cs;
            for (size_t c = 0; c < Cs; ++c) o[c] = te[c] + pe[c];
        }
    }
}

}  // namespace ops
}  // namespace tslm
