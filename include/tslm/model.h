// model.h: GPT-style decoder-only transformer (ROADMAP §2.1, §8 Model API).
//
// Architecture (pre-LayerNorm, GPT-2 style; shapes in layout.h):
//   x = wte[tokens] + wpe[0..T)                                 (encoded)
//   for each layer l:
//     x = x + proj( attention( qkv( LN1(x) ) ) )                (res2)
//     x = x + fcproj( GELU( fc( LN2(x) ) ) )                    (res3)
//   logits = LNf(x) * head_w^T,  head_w = wte when tied (no head bias)
//   loss   = mean over B*T positions of cross-entropy(logits, targets)
// Every numeric step is an ops:: call; this layer only wires buffers.
//
// Memory (§4.3): ONE allocation each for params, grads, AdamW m and v (all
// num_params_padded long, layout.h order) and for the activation arena
// (make_act_layout at the capacity B x T). All go through backend.h, so in
// CUDA builds they are device pointers. All are zeroed at creation, so
// padding is always 0. Nothing is allocated after model_create, so the
// training loop never allocates.
//
// Init (§2.3, §4.6), host-side then copy_h2d: tensor i of the layout draws
// from Rng(seed, stream = i), so each tensor's values do not depend on the
// order tensors are initialized in.
//   wte, wpe, qkv_w, fc_w, head_w : N(0, 0.02)
//   proj_w, fcproj_w              : N(0, 0.02 / sqrt(2L))  (residual outputs)
//   all biases, LayerNorm beta    : 0;   LayerNorm gamma : 1
//
// tokens/targets passed to model_forward are DEVICE buffers (host memory in
// the CPU build) of B*T int32 ids in [0, V). The DataLoader validates the
// ids on the host; kernels do not re-check them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tslm/common.h"
#include "tslm/config.h"
#include "tslm/layout.h"

namespace tslm {

struct Model {
    ModelConfig cfg;
    ParamLayout params_layout;
    ActLayout acts_layout;
    int B_cap = 0, T_cap = 0;  // capacity the activation arena was laid out for
    real_t* params = nullptr;  // [num_params_padded]
    real_t* grads = nullptr;   // [num_params_padded]
    real_t* m = nullptr;       // AdamW first moment
    real_t* v = nullptr;       // AdamW second moment
    real_t* acts = nullptr;    // activation arena [acts_layout.num_elems]
    int B_last = 0, T_last = 0;  // shape of the most recent forward

    // Pointer to tensor i of a layout within its buffer.
    real_t* param(int i) const { return params + params_layout[i].offset; }
    real_t* act(int i) const { return acts + acts_layout[i].offset; }
};

// Validates cfg (CHECK) and 1 <= B, 1 <= T <= cfg.max_seq_len.
Model* model_create(const ModelConfig& cfg, int B, int T);
void model_free(Model* model);

// Host-side init per the rules above, then one copy_h2d into params.
void model_init_weights(Model* model, uint64_t seed);

// Forward pass for a runtime batch B x T (B <= B_cap, T <= T_cap). OVERWRITES
// the activations. Returns the mean cross-entropy loss over B*T positions as a
// host double, or NaN when targets == nullptr (then softmax/loss are
// skipped and only logits are produced).
double model_forward(Model* model, const int32_t* tokens, const int32_t* targets, int B, int T);

// Logits of the most recent forward, [B_last*T_last, V] row-major, in the
// activation arena (a device pointer in CUDA builds).
const real_t* model_logits(const Model* model);

// Bytes of the activation arena and of one parameter-sized buffer.
size_t model_act_bytes(const Model* model);
size_t model_param_bytes(const Model* model);

// One line for startup logs: parameter counts, buffer sizes, arena MB.
std::string model_summary(const Model* model);

}  // namespace tslm
