// layout.h: flat parameter and activation layouts (ROADMAP §4.3, §6.2).
//
// PARAMETERS. All parameters live in ONE contiguous real_t buffer
// params[num_params_padded]. grads, AdamW m and v use the identical layout.
// make_param_layout() is a pure function of ModelConfig. It gives every
// named tensor an offset, numel, shape and decay flag. Order:
//   wte [V, C]           decay   token embedding (also the head when tied)
//   wpe [max_T, C]               learned positions
//   per layer l (prefix "h<l>."):
//     ln1_g [C], ln1_b [C]
//     qkv_w [3C, C] decay, qkv_b [3C]     fused Q|K|V projection
//     proj_w [C, C] decay, proj_b [C]     attention output projection
//     ln2_g [C], ln2_b [C]
//     fc_w [4C, C] decay, fc_b [4C]       MLP up-projection
//     fcproj_w [C, 4C] decay, fcproj_b [C] MLP down-projection
//   lnf_g [C], lnf_b [C]                  final LayerNorm
//   head_w [V, C]        decay   only when tie_embeddings = 0
// Linear weights are [OC, C] row-major (PyTorch nn.Linear layout, §6.2).
// decay = true exactly for 2-D weight matrices other than wpe (§2.3). The
// untied head_w counts as one.
//
// ACTIVATIONS. make_act_layout(cfg, B, T) lays out one real_t arena holding
// every forward output and everything saved for backward, for capacity
// B x T (N = B*T rows). Runtime batches may be smaller. Per layer:
//   ln1 [N,C], ln1_mean [N], ln1_rstd [N]   LN1 output and saved stats
//   qkv [N,3C]                              fused projection output
//   att [B,NH,T,T]                          post-softmax probs (causal)
//   atty [N,C]                              attention output (pre proj)
//   attproj [N,C]                           after proj
//   res2 [N,C]                              residual: x + attproj
//   ln2 [N,C], ln2_mean [N], ln2_rstd [N]
//   fch [N,4C]                              fc output (GELU input)
//   fch_gelu [N,4C]                         GELU output
//   fcproj [N,C]                            after fcproj
//   res3 [N,C]                              layer output: res2 + fcproj
// plus "encoded" [N,C] (embedding output = layer-0 input), lnf/lnf_mean/
// lnf_rstd, logits [N,V], probs [N,V], losses [N]. Integer token buffers
// are not in this real_t arena.
//
// BACKWARD BUFFERS (appended after every forward tensor, so forward offsets
// are unchanged). They are shared by all layers; model_backward zeroes each
// one before an accumulating op writes it:
//   d_logits [N,V]        dL/dlogits
//   d_x [N,C]             gradient of the residual stream (top to bottom)
//   d_ln [N,C]            gradient w.r.t. a LayerNorm output (lnf, ln2, ln1)
//   d_atty [N,C]          gradient w.r.t. the attention output (pre proj)
//   d_qkv [N,3C]          gradient w.r.t. qkv
//   d_fch_gelu [N,4C]     gradient w.r.t. the GELU output
//   d_fch [N,4C]          gradient w.r.t. the GELU input
//   att_scratch [B,NH,T,T] workspace for attention_backward
//
// PADDING. Every tensor offset is a multiple of LAYOUT_PAD_ELEMS = 16
// elements: 64 bytes for float, 128 for double. Allocations come from
// dev_malloc (64-byte aligned), so every tensor starts 64-byte aligned. The
// padding is counted in elements, not bytes, so the layout and the checkpoint
// blob sizes are identical in every build. num_params counts logical
// parameters only; num_params_padded is the buffer length.
//
// Host-side description only (std::vector/std::string); no tensor memory here.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tslm/config.h"

namespace tslm {

constexpr size_t LAYOUT_PAD_ELEMS = 16;

struct TensorDesc {
    std::string name;
    size_t offset = 0;  // in elements from the buffer start
    size_t numel = 0;
    int ndim = 0;
    int shape[4] = {0, 0, 0, 0};
    bool decay = false;  // AdamW weight decay applies (params only)
};

// Indices into ParamLayout::tensors for one transformer block.
struct LayerParamIdx {
    int ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b;
    int ln2_g, ln2_b, fc_w, fc_b, fcproj_w, fcproj_b;
};

struct ParamLayout {
    std::vector<TensorDesc> tensors;
    size_t num_params = 0;         // logical parameter count (no padding)
    size_t num_params_padded = 0;  // buffer length in elements
    int wte = -1, wpe = -1, lnf_g = -1, lnf_b = -1;
    int head_w = -1;  // tensor used as the output head: == wte when tied
    bool tied = true;
    std::vector<LayerParamIdx> layers;

    const TensorDesc& operator[](int i) const { return tensors.at(static_cast<size_t>(i)); }
    int find(const std::string& name) const;  // index, or -1 if absent
};

// Indices into ActLayout::tensors for one transformer block.
struct LayerActIdx {
    int ln1, ln1_mean, ln1_rstd, qkv, att, atty, attproj, res2;
    int ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, res3;
};

struct ActLayout {
    std::vector<TensorDesc> tensors;
    size_t num_elems = 0;  // arena length in elements (including padding)
    int B = 0, T = 0;      // capacity this layout was built for
    int encoded = -1, lnf = -1, lnf_mean = -1, lnf_rstd = -1;
    int logits = -1, probs = -1, losses = -1;
    std::vector<LayerActIdx> layers;
    // Backward buffers (shared across layers; see above).
    int d_logits = -1, d_x = -1, d_ln = -1, d_atty = -1, d_qkv = -1;
    int d_fch_gelu = -1, d_fch = -1, att_scratch = -1;

    const TensorDesc& operator[](int i) const { return tensors.at(static_cast<size_t>(i)); }
    int find(const std::string& name) const;
};

// Both CHECK-fail on an invalid config (see config_validate) or on
// B < 1, T < 1, or T > max_seq_len.
ParamLayout make_param_layout(const ModelConfig& cfg);
ActLayout make_act_layout(const ModelConfig& cfg, int B, int T);

}  // namespace tslm
