// model.cpp: model creation, weight init and forward wiring (see include/tslm/model.h).
// No tensor math here: every computation on params/activations is an ops::
// call. The only loops over values are the host-side init (§4.8).
#include "tslm/model.h"

#include "tslm/backend.h"
#include "tslm/ops.h"
#include "tslm/rng.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace tslm {

namespace {

real_t* alloc_zeroed(size_t elems) {
    real_t* p = static_cast<real_t*>(dev_malloc(elems * sizeof(real_t)));
    dev_memset(p, 0, elems * sizeof(real_t));
    return p;
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::string suf(suffix);
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

}  // namespace

Model* model_create(const ModelConfig& cfg, int B, int T) {
    Model* m = new Model();  // the small host-side descriptor, not a tensor
    m->cfg = cfg;
    m->params_layout = make_param_layout(cfg);  // CHECKs cfg
    m->acts_layout = make_act_layout(cfg, B, T);  // CHECKs B, T
    m->B_cap = B;
    m->T_cap = T;
    const size_t np = m->params_layout.num_params_padded;
    m->params = alloc_zeroed(np);
    m->grads = alloc_zeroed(np);
    m->m = alloc_zeroed(np);
    m->v = alloc_zeroed(np);
    m->acts = alloc_zeroed(m->acts_layout.num_elems);
    return m;
}

void model_free(Model* model) {
    if (!model) return;
    dev_free(model->params);
    dev_free(model->grads);
    dev_free(model->m);
    dev_free(model->v);
    dev_free(model->acts);
    delete model;
}

void model_init_weights(Model* model, uint64_t seed) {
    CHECK(model != nullptr, "model_init_weights: null model");
    const ParamLayout& L = model->params_layout;
    const double std_w = 0.02;
    const double std_resid = 0.02 / std::sqrt(2.0 * model->cfg.n_layer);
    // Host staging copy of the whole flat buffer, so the values reach the
    // device in one copy_h2d. Padding stays 0. Init runs once, outside
    // the training loop.
    std::vector<real_t> host(L.num_params_padded, real_t(0));
    for (size_t i = 0; i < L.tensors.size(); ++i) {
        const TensorDesc& t = L.tensors[i];
        real_t* dst = host.data() + t.offset;
        if (ends_with(t.name, "_b")) {  // biases and LayerNorm beta
            for (size_t k = 0; k < t.numel; ++k) dst[k] = real_t(0);
        } else if (ends_with(t.name, "_g")) {  // LayerNorm gamma
            for (size_t k = 0; k < t.numel; ++k) dst[k] = real_t(1);
        } else {
            const double sd = (ends_with(t.name, "proj_w")) ? std_resid : std_w;  // proj_w, fcproj_w
            CHECK(t.ndim == 2, "model_init_weights: unexpected tensor '%s'", t.name.c_str());
            Rng rng(seed, static_cast<uint64_t>(i));
            for (size_t k = 0; k < t.numel; ++k) dst[k] = static_cast<real_t>(rng.normal(0.0, sd));
        }
    }
    copy_h2d(model->params, host.data(), host.size() * sizeof(real_t));
}

double model_forward(Model* model, const int32_t* tokens, const int32_t* targets, int B, int T) {
    CHECK(model != nullptr, "model_forward: null model");
    CHECK(B >= 1 && B <= model->B_cap && T >= 1 && T <= model->T_cap,
          "model_forward: B=%d T=%d outside capacity B<=%d T<=%d", B, T, model->B_cap, model->T_cap);
    const ModelConfig& c = model->cfg;
    const ParamLayout& P = model->params_layout;
    const ActLayout& A = model->acts_layout;
    const int C = c.n_embd, NH = c.n_head, V = c.vocab_size;
    const int N = B * T;
    const size_t NC = static_cast<size_t>(N) * static_cast<size_t>(C);
    // Activations use the leading N rows of each capacity-sized tensor
    // (att uses the leading B*NH*T*T elements), so smaller runtime shapes fit.

    ops::embedding_forward(model->act(A.encoded), tokens, model->param(P.wte), model->param(P.wpe), B, T, C);
    const real_t* x = model->act(A.encoded);
    for (int l = 0; l < c.n_layer; ++l) {
        const LayerParamIdx& p = P.layers[static_cast<size_t>(l)];
        const LayerActIdx& a = A.layers[static_cast<size_t>(l)];
        ops::layernorm_forward(model->act(a.ln1), model->act(a.ln1_mean), model->act(a.ln1_rstd), x,
                               model->param(p.ln1_g), model->param(p.ln1_b), N, C);
        ops::matmul_forward(model->act(a.qkv), model->act(a.ln1), model->param(p.qkv_w), model->param(p.qkv_b), N,
                            C, 3 * C);
        ops::attention_forward(model->act(a.atty), model->act(a.att), model->act(a.qkv), B, T, C, NH);
        ops::matmul_forward(model->act(a.attproj), model->act(a.atty), model->param(p.proj_w),
                            model->param(p.proj_b), N, C, C);
        ops::residual_forward(model->act(a.res2), x, model->act(a.attproj), NC);
        ops::layernorm_forward(model->act(a.ln2), model->act(a.ln2_mean), model->act(a.ln2_rstd),
                               model->act(a.res2), model->param(p.ln2_g), model->param(p.ln2_b), N, C);
        ops::matmul_forward(model->act(a.fch), model->act(a.ln2), model->param(p.fc_w), model->param(p.fc_b), N, C,
                            4 * C);
        ops::gelu_forward(model->act(a.fch_gelu), model->act(a.fch), 4 * NC);
        ops::matmul_forward(model->act(a.fcproj), model->act(a.fch_gelu), model->param(p.fcproj_w),
                            model->param(p.fcproj_b), N, 4 * C, C);
        ops::residual_forward(model->act(a.res3), model->act(a.res2), model->act(a.fcproj), NC);
        x = model->act(a.res3);
    }
    ops::layernorm_forward(model->act(A.lnf), model->act(A.lnf_mean), model->act(A.lnf_rstd), x,
                           model->param(P.lnf_g), model->param(P.lnf_b), N, C);
    ops::matmul_forward(model->act(A.logits), model->act(A.lnf), model->param(P.head_w), nullptr, N, C, V);
    model->B_last = B;
    model->T_last = T;

    if (!targets) return std::numeric_limits<double>::quiet_NaN();
    ops::softmax_xent_forward(model->act(A.probs), model->act(A.losses), model->act(A.logits), targets, N, V);
    return ops::sum_reduce(model->act(A.losses), static_cast<size_t>(N)) / static_cast<double>(N);
}

const real_t* model_logits(const Model* model) { return model->act(model->acts_layout.logits); }

size_t model_act_bytes(const Model* model) { return model->acts_layout.num_elems * sizeof(real_t); }

size_t model_param_bytes(const Model* model) { return model->params_layout.num_params_padded * sizeof(real_t); }

std::string model_summary(const Model* model) {
    const ModelConfig& c = model->cfg;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "model: L=%d C=%d NH=%d V=%d max_T=%d tied=%d | params %zu (padded %zu) | "
                  "params/grads/m/v %.1f MB each | activation arena %.1f MB for B=%d T=%d | device %s",
                  c.n_layer, c.n_embd, c.n_head, c.vocab_size, c.max_seq_len, c.tie_embeddings,
                  model->params_layout.num_params, model->params_layout.num_params_padded,
                  double(model_param_bytes(model)) / (1024.0 * 1024.0),
                  double(model_act_bytes(model)) / (1024.0 * 1024.0), model->B_cap, model->T_cap, dev_name());
    return buf;
}

}  // namespace tslm
