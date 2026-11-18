// generate.cpp: sampling and the generation loop. See include/tslm/generate.h.
#include "tslm/generate.h"

#include <algorithm>
#include <cmath>

#include "tslm/backend.h"
#include "tslm/vocab_map.h"

namespace tslm {

int32_t sample_token(const real_t* logits, int V, const SampleOptions& opt, Rng* rng) {
    CHECK(V >= 2, "sample_token: V=%d", V);
    const int first = opt.allow_unk ? 0 : UNK_ID + 1;  // UNK_ID is 0
    if (opt.greedy) {
        int best = first;
        for (int i = first + 1; i < V; ++i)
            if (logits[i] > logits[best]) best = i;  // strict: ties keep the lower id
        return best;
    }
    CHECK(opt.temperature > 0.0 && std::isfinite(opt.temperature), "temperature %g must be > 0",
          opt.temperature);
    CHECK(rng != nullptr, "sample_token: rng is null");
    std::vector<int32_t> ids;
    ids.reserve(size_t(V - first));
    for (int i = first; i < V; ++i) ids.push_back(i);
    if (opt.top_k > 0 && size_t(opt.top_k) < ids.size()) {
        auto higher = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
        std::nth_element(ids.begin(), ids.begin() + (opt.top_k - 1), ids.end(), higher);
        ids.resize(size_t(opt.top_k));
        std::sort(ids.begin(), ids.end());  // accumulate in id order (deterministic)
    }
    double mx = -HUGE_VAL;
    for (int32_t i : ids) mx = std::max(mx, double(logits[i]));
    CHECK(std::isfinite(mx), "sample_token: non-finite logits");
    std::vector<double> w(ids.size());
    double sum = 0.0;
    for (size_t j = 0; j < ids.size(); ++j) {
        w[j] = std::exp((double(logits[ids[j]]) - mx) / opt.temperature);
        sum += w[j];
    }
    const double u = rng->uniform() * sum;
    double c = 0.0;
    for (size_t j = 0; j < ids.size(); ++j) {
        c += w[j];
        if (c > u) return ids[j];
    }
    return ids.back();  // u within rounding of sum
}

std::vector<int32_t> generate_ids(Model* model, const std::vector<int32_t>& context, const GenerateOptions& opt) {
    const int V = model->cfg.vocab_size, maxT = model->cfg.max_seq_len;
    CHECK(model->B_cap >= 1 && model->T_cap == maxT, "generate_ids: model needs B_cap >= 1 and T_cap = %d (has %d)",
          maxT, model->T_cap);
    CHECK(!context.empty(), "generate_ids: empty context");
    CHECK(opt.max_new >= 0, "generate_ids: max_new %d < 0", opt.max_new);
    for (int32_t t : context) CHECK(t >= 0 && t < V, "generate_ids: context id %d outside [0, %d)", t, V);

    std::vector<int32_t> seq = context, out;
    std::vector<real_t> row(static_cast<size_t>(V));
    int32_t* tok_dev = static_cast<int32_t*>(dev_malloc(size_t(maxT) * sizeof(int32_t)));
    Rng rng(opt.seed, SAMPLING_STREAM);
    for (int n = 0; n < opt.max_new; ++n) {
        const int T = static_cast<int>(std::min(seq.size(), size_t(maxT)));
        copy_h2d(tok_dev, seq.data() + (seq.size() - size_t(T)), size_t(T) * sizeof(int32_t));
        model_forward(model, tok_dev, nullptr, 1, T);
        copy_d2h(row.data(), model_logits(model) + size_t(T - 1) * size_t(V), size_t(V) * sizeof(real_t));
        const int32_t next = sample_token(row.data(), V, opt.sample, &rng);
        if (next == opt.eos_id) break;
        seq.push_back(next);
        out.push_back(next);
    }
    dev_free(tok_dev);
    return out;
}

}  // namespace tslm
