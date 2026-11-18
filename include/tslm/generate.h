// generate.h: autoregressive sampling (ROADMAP P6.1). Host-side driver: the
// numerics are model_forward; every sampling decision is made on the host
// (CLAUDE.md §4.8) from logits copied with copy_d2h.
//
// Loop (no KV cache, a Phase A non-goal): the sequence starts as `context`
// (vocab ids, non-empty). For each new token, the model runs a forward pass
// with B = 1 over the last min(len, max_seq_len) tokens (sliding window), and
// the logits of the final position are sampled. The token is appended.
// Generation stops after max_new tokens, or when eos_id is sampled (that EOS
// is not appended or returned). eos_id < 0 disables the EOS stop.
//
// sample_token(logits[V]):
//   - <unk> (UNK_ID) is excluded unless allow_unk (it decodes to no text).
//   - greedy: argmax; ties go to the lowest id.
//   - otherwise: keep the top_k largest logits (top_k <= 0 or >= V: keep all;
//     ties ranked by lower id), then p_i ∝ exp((l_i - max) / temperature),
//     computed in double over the kept ids in increasing id order; draw
//     u = rng.uniform() * sum and return the first id whose cumulative weight
//     exceeds u. temperature must be > 0.
// Randomness comes only from Rng(seed, SAMPLING_STREAM), a stream separate
// from init (streams 0..#tensors) and data order (100 + epoch) (§4.6). Same
// model, context, options and seed => the same output, so greedy output is
// reproducible by construction.
#pragma once

#include <cstdint>
#include <vector>

#include "tslm/common.h"
#include "tslm/model.h"
#include "tslm/rng.h"

namespace tslm {

constexpr uint64_t SAMPLING_STREAM = 1000000;

struct SampleOptions {
    bool greedy = false;
    double temperature = 1.0;
    int top_k = 0;           // <= 0: no top-k filter
    bool allow_unk = false;  // false: never emit <unk>
};

struct GenerateOptions {
    SampleOptions sample;
    int max_new = 200;
    int32_t eos_id = 1;      // EOS_ID; < 0 disables the stop
    uint64_t seed = 1337;
};

// Host logits of one position, [V]. CHECKs the options; rng is used only
// when not greedy.
int32_t sample_token(const real_t* logits, int V, const SampleOptions& opt, Rng* rng);

// Returns the generated ids (without the context and without a final EOS).
// The model must have B_cap >= 1 and T_cap = cfg.max_seq_len. Context ids
// must be in [0, V).
std::vector<int32_t> generate_ids(Model* model, const std::vector<int32_t>& context, const GenerateOptions& opt);

}  // namespace tslm
