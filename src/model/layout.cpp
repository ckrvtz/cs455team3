// layout.cpp: make_param_layout / make_act_layout (see include/tslm/layout.h).
// Tensors are appended in a fixed order. Each append rounds the running
// offset up to LAYOUT_PAD_ELEMS, so offsets are deterministic and depend
// only on (cfg, B, T).
#include "tslm/layout.h"

#include "tslm/common.h"

#include <initializer_list>

namespace tslm {

namespace {

size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Appends tensors to a list and tracks the padded end offset.
class Builder {
public:
    explicit Builder(std::vector<TensorDesc>* out) : out_(out) {}

    int add(const std::string& name, std::initializer_list<size_t> shape, bool decay = false) {
        CHECK(shape.size() >= 1 && shape.size() <= 4, "layout: tensor '%s' has %zu dims",
              name.c_str(), shape.size());
        TensorDesc d;
        d.name = name;
        d.offset = end_;
        d.numel = 1;
        d.ndim = static_cast<int>(shape.size());
        int i = 0;
        for (size_t s : shape) {
            CHECK(s >= 1 && s <= 0x7fffffff, "layout: tensor '%s' dim %d = %zu", name.c_str(), i, s);
            d.shape[i++] = static_cast<int>(s);
            d.numel *= s;
        }
        d.decay = decay;
        logical_ += d.numel;
        end_ = round_up(d.offset + d.numel, LAYOUT_PAD_ELEMS);
        out_->push_back(d);
        return static_cast<int>(out_->size()) - 1;
    }

    size_t end() const { return end_; }
    size_t logical() const { return logical_; }

private:
    std::vector<TensorDesc>* out_;
    size_t end_ = 0;
    size_t logical_ = 0;
};

void check_model(const ModelConfig& cfg) {
    Config full;
    full.model = cfg;
    full.train.seq_len = 1;  // train fields are irrelevant here; keep them valid
    std::string err;
    CHECK(config_validate(full, &err), "layout: %s", err.c_str());
}

int find_in(const std::vector<TensorDesc>& tensors, const std::string& name) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

int ParamLayout::find(const std::string& name) const { return find_in(tensors, name); }
int ActLayout::find(const std::string& name) const { return find_in(tensors, name); }

ParamLayout make_param_layout(const ModelConfig& cfg) {
    check_model(cfg);
    const size_t V = static_cast<size_t>(cfg.vocab_size);
    const size_t C = static_cast<size_t>(cfg.n_embd);
    const size_t maxT = static_cast<size_t>(cfg.max_seq_len);

    ParamLayout L;
    Builder b(&L.tensors);
    L.wte = b.add("wte", {V, C}, true);
    L.wpe = b.add("wpe", {maxT, C});
    for (int l = 0; l < cfg.n_layer; ++l) {
        const std::string p = "h" + std::to_string(l) + ".";
        LayerParamIdx x;
        x.ln1_g = b.add(p + "ln1_g", {C});
        x.ln1_b = b.add(p + "ln1_b", {C});
        x.qkv_w = b.add(p + "qkv_w", {3 * C, C}, true);
        x.qkv_b = b.add(p + "qkv_b", {3 * C});
        x.proj_w = b.add(p + "proj_w", {C, C}, true);
        x.proj_b = b.add(p + "proj_b", {C});
        x.ln2_g = b.add(p + "ln2_g", {C});
        x.ln2_b = b.add(p + "ln2_b", {C});
        x.fc_w = b.add(p + "fc_w", {4 * C, C}, true);
        x.fc_b = b.add(p + "fc_b", {4 * C});
        x.fcproj_w = b.add(p + "fcproj_w", {C, 4 * C}, true);
        x.fcproj_b = b.add(p + "fcproj_b", {C});
        L.layers.push_back(x);
    }
    L.lnf_g = b.add("lnf_g", {C});
    L.lnf_b = b.add("lnf_b", {C});
    L.tied = cfg.tie_embeddings != 0;
    L.head_w = L.tied ? L.wte : b.add("head_w", {V, C}, true);

    L.num_params = b.logical();
    L.num_params_padded = b.end();
    return L;
}

ActLayout make_act_layout(const ModelConfig& cfg, int B, int T) {
    check_model(cfg);
    CHECK(B >= 1, "make_act_layout: B=%d must be >= 1", B);
    CHECK(T >= 1 && T <= cfg.max_seq_len, "make_act_layout: T=%d must be in [1, max_seq_len=%d]", T,
          cfg.max_seq_len);
    const size_t Bs = static_cast<size_t>(B);
    const size_t Ts = static_cast<size_t>(T);
    const size_t N = Bs * Ts;
    const size_t C = static_cast<size_t>(cfg.n_embd);
    const size_t NH = static_cast<size_t>(cfg.n_head);
    const size_t V = static_cast<size_t>(cfg.vocab_size);

    ActLayout A;
    A.B = B;
    A.T = T;
    Builder b(&A.tensors);
    A.encoded = b.add("encoded", {N, C});
    for (int l = 0; l < cfg.n_layer; ++l) {
        const std::string p = "h" + std::to_string(l) + ".";
        LayerActIdx x;
        x.ln1 = b.add(p + "ln1", {N, C});
        x.ln1_mean = b.add(p + "ln1_mean", {N});
        x.ln1_rstd = b.add(p + "ln1_rstd", {N});
        x.qkv = b.add(p + "qkv", {N, 3 * C});
        x.att = b.add(p + "att", {Bs, NH, Ts, Ts});
        x.atty = b.add(p + "atty", {N, C});
        x.attproj = b.add(p + "attproj", {N, C});
        x.res2 = b.add(p + "res2", {N, C});
        x.ln2 = b.add(p + "ln2", {N, C});
        x.ln2_mean = b.add(p + "ln2_mean", {N});
        x.ln2_rstd = b.add(p + "ln2_rstd", {N});
        x.fch = b.add(p + "fch", {N, 4 * C});
        x.fch_gelu = b.add(p + "fch_gelu", {N, 4 * C});
        x.fcproj = b.add(p + "fcproj", {N, C});
        x.res3 = b.add(p + "res3", {N, C});
        A.layers.push_back(x);
    }
    A.lnf = b.add("lnf", {N, C});
    A.lnf_mean = b.add("lnf_mean", {N});
    A.lnf_rstd = b.add("lnf_rstd", {N});
    A.logits = b.add("logits", {N, V});
    A.probs = b.add("probs", {N, V});
    A.losses = b.add("losses", {N});
    // Backward buffers, after all forward tensors (forward offsets unchanged).
    A.d_logits = b.add("d_logits", {N, V});
    A.d_x = b.add("d_x", {N, C});
    A.d_ln = b.add("d_ln", {N, C});
    A.d_atty = b.add("d_atty", {N, C});
    A.d_qkv = b.add("d_qkv", {N, 3 * C});
    A.d_fch_gelu = b.add("d_fch_gelu", {N, 4 * C});
    A.d_fch = b.add("d_fch", {N, 4 * C});
    A.att_scratch = b.add("att_scratch", {Bs, NH, Ts, Ts});
    A.num_elems = b.end();
    return A;
}

}  // namespace tslm
