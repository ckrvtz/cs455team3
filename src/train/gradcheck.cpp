// gradcheck.cpp: full-model finite-difference gradient check (see include/tslm/gradcheck.h).
// Host-side test logic: samples entries, perturbs single parameters through
// the backend shim, and compares against model_backward. No tensor math.
#include "tslm/gradcheck.h"

#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/model.h"
#include "tslm/rng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

namespace tslm {

namespace {

bool ends_with(const std::string& s, const char* suffix) {
    const std::string suf(suffix);
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

real_t get_param(const Model* m, size_t i) {
    real_t v;
    copy_d2h(&v, m->params + i, sizeof(real_t));
    return v;
}

void set_param(Model* m, size_t i, real_t v) { copy_h2d(m->params + i, &v, sizeof(real_t)); }

}  // namespace

GradcheckOptions default_gradcheck_options() {
    GradcheckOptions o;
    o.cfg.vocab_size = 32;
    o.cfg.max_seq_len = 8;
    o.cfg.n_layer = 2;
    o.cfg.n_embd = 16;
    o.cfg.n_head = 2;
    o.cfg.tie_embeddings = 1;
    o.B = 2;
    o.T = 8;
    return o;
}

GradcheckReport run_gradcheck(const GradcheckOptions& opt) {
    CHECK(opt.samples >= 0 && opt.h > 0 && opt.eps_2pt > 0 && opt.tol > 0 && opt.floor > 0,
          "run_gradcheck: invalid options");
    Model* m = model_create(opt.cfg, opt.B, opt.T);  // CHECKs cfg, B, T
    model_init_weights(m, opt.seed);
    const ParamLayout& L = m->params_layout;
    const size_t np = L.num_params_padded;
    if (opt.weight_scale != 1.0) {
        std::vector<real_t> host(np);
        copy_d2h(host.data(), m->params, np * sizeof(real_t));
        for (real_t& v : host) v = static_cast<real_t>(double(v) * opt.weight_scale);
        copy_h2d(m->params, host.data(), np * sizeof(real_t));
    }

    // Fixed batch (host buffers double as device buffers in the CPU build;
    // copied to device-side buffers through the shim for portability).
    const size_t N = size_t(opt.B) * size_t(opt.T);
    std::vector<int32_t> hx(N), hy(N);
    Rng brng(opt.seed, GRADCHECK_BATCH_STREAM);
    for (auto& t : hx) t = static_cast<int32_t>(brng.uniform_int(uint64_t(opt.cfg.vocab_size)));
    for (auto& t : hy) t = static_cast<int32_t>(brng.uniform_int(uint64_t(opt.cfg.vocab_size)));
    int32_t* dx = static_cast<int32_t*>(dev_malloc(N * sizeof(int32_t)));
    int32_t* dy = static_cast<int32_t*>(dev_malloc(N * sizeof(int32_t)));
    copy_h2d(dx, hx.data(), N * sizeof(int32_t));
    copy_h2d(dy, hy.data(), N * sizeof(int32_t));
    auto loss = [&] { return model_forward(m, dx, dy, opt.B, opt.T); };

    // Analytic gradient.
    GradcheckReport r;
    dev_memset(m->grads, 0, np * sizeof(real_t));
    r.loss = loss();
    model_backward(m);
    std::vector<real_t> g(np);
    copy_d2h(g.data(), m->grads, np * sizeof(real_t));

    auto loss_at = [&](size_t i, real_t orig, double d) {
        set_param(m, i, static_cast<real_t>(double(orig) + d));
        const double l = loss();
        set_param(m, i, orig);
        return l;
    };

    const size_t C = size_t(opt.cfg.n_embd);
    Rng pick(opt.seed, GRADCHECK_BATCH_STREAM + 1);
    for (const TensorDesc& t : L.tensors) {
        const bool qkv_b = ends_with(t.name, "qkv_b");
        // Map a draw in [0, n_choices) to a tensor index (skipping the key slice for qkv_b).
        const size_t n_choices = qkv_b ? 2 * C : t.numel;
        auto index_of = [&](size_t k) { return qkv_b && k >= C ? k + C : k; };
        std::set<size_t> idx = {0, t.numel - 1};  // first (query) and last (value) for qkv_b
        const size_t want = std::min(n_choices, size_t(opt.samples) + 2);
        while (idx.size() < want) idx.insert(index_of(size_t(pick.uniform_int(n_choices))));

        TensorCheck tc;
        tc.name = t.name;
        for (size_t k : idx) {
            const size_t i = t.offset + k;
            const real_t orig = get_param(m, i);
            const double lp1 = loss_at(i, orig, opt.h), lm1 = loss_at(i, orig, -opt.h);
            const double lp2 = loss_at(i, orig, 2 * opt.h), lm2 = loss_at(i, orig, -2 * opt.h);
            const double num = (8.0 * (lp1 - lm1) - (lp2 - lm2)) / (12.0 * opt.h);
            const double num2 = (loss_at(i, orig, opt.eps_2pt) - loss_at(i, orig, -opt.eps_2pt)) / (2.0 * opt.eps_2pt);
            const double a = g[i];
            const double rel = std::fabs(a - num) / std::fmax(std::fmax(std::fabs(a), std::fabs(num)), opt.floor);
            const double rel2 = std::fabs(a - num2) / std::fmax(std::fmax(std::fabs(a), std::fabs(num2)), opt.floor);
            ++tc.checked;
            if (rel >= tc.worst_rel) {
                tc.worst_rel = rel;
                tc.grad_at_worst = a;
                tc.numeric_at_worst = num;
            }
            tc.worst_rel_2pt = rel2 > tc.worst_rel_2pt ? rel2 : tc.worst_rel_2pt;
            CHECK(get_param(m, i) == orig, "run_gradcheck: parameter %zu not restored", i);
        }
        tc.pass = tc.worst_rel < opt.tol;
        r.pass = r.pass && tc.pass;
        r.tensors.push_back(tc);
    }

    // Mandatory structural-zero check of the key-bias slice, per layer.
    for (const LayerParamIdx& p : L.layers) {
        const real_t* gb = g.data() + L[p.qkv_b].offset;
        double kmax = 0, qvmax = 0;
        for (size_t k = 0; k < 3 * C; ++k) {
            const double v = std::fabs(double(gb[k]));
            if (k >= C && k < 2 * C) kmax = v > kmax ? v : kmax;
            else qvmax = v > qvmax ? v : qvmax;
        }
        const double ratio = qvmax > 0 ? kmax / qvmax : (kmax > 0 ? INFINITY : 0.0);
        r.kbias_ratios.push_back(ratio);
        const bool ok = qvmax > 0 && ratio <= opt.kbias_ratio;
        r.kbias_pass = r.kbias_pass && ok;
    }
    r.pass = r.pass && r.kbias_pass;

    dev_free(dx);
    dev_free(dy);
    model_free(m);
    return r;
}

std::string format_gradcheck_report(const GradcheckOptions& opt, const GradcheckReport& r) {
    std::string out;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "gradcheck: L=%d C=%d NH=%d V=%d T=%d B=%d tied=%d seed=%llu scale=%g | loss %.12f\n"
                  "  5-point stencil h=%g (2-point eps=%g shown for diagnosis) | pass: rel < %g (floor %g)\n",
                  opt.cfg.n_layer, opt.cfg.n_embd, opt.cfg.n_head, opt.cfg.vocab_size, opt.T, opt.B,
                  opt.cfg.tie_embeddings, static_cast<unsigned long long>(opt.seed), opt.weight_scale, r.loss, opt.h,
                  opt.eps_2pt, opt.tol, opt.floor);
    out += buf;
    std::snprintf(buf, sizeof(buf), "  %-14s %5s %12s %14s %14s %12s  %s\n", "tensor", "n", "worst rel", "grad @ worst",
                  "numeric", "2pt worst", "");
    out += buf;
    for (const TensorCheck& t : r.tensors) {
        std::snprintf(buf, sizeof(buf), "  %-14s %5d %12.3e %14.6e %14.6e %12.3e  %s\n", t.name.c_str(), t.checked,
                      t.worst_rel, t.grad_at_worst, t.numeric_at_worst, t.worst_rel_2pt, t.pass ? "ok" : "FAIL");
        out += buf;
    }
    for (size_t l = 0; l < r.kbias_ratios.size(); ++l) {
        std::snprintf(buf, sizeof(buf), "  h%zu key-bias structural zero: max|d k-bias|/max|d q,v-bias| = %.2e (<= %g)  %s\n",
                      l, r.kbias_ratios[l], opt.kbias_ratio, r.kbias_ratios[l] <= opt.kbias_ratio ? "ok" : "FAIL");
        out += buf;
    }
    out += r.pass ? "gradcheck: PASS\n" : "gradcheck: FAIL\n";
    return out;
}

}  // namespace tslm
