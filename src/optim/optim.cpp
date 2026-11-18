// optim.cpp: LR schedule, global-norm clipping and AdamW driver (see include/tslm/optim.h).
#include "tslm/optim.h"

#include "tslm/common.h"
#include "tslm/ops.h"

#include <cmath>

namespace tslm {

double lr_at(const TrainConfig& cfg, int step) {
    CHECK(step >= 1, "lr_at: step %d must be >= 1", step);
    const double peak = cfg.lr;
    const double floor_lr = cfg.min_lr_frac * peak;
    const int W = cfg.warmup_steps, S = cfg.max_steps;
    if (W > 0 && step <= W) return peak * static_cast<double>(step) / static_cast<double>(W);
    if (S <= W) return floor_lr;  // no decay phase left
    double q = static_cast<double>(step - W) / static_cast<double>(S - W);
    q = q > 1.0 ? 1.0 : q;
    const double pi = 3.14159265358979323846;
    return floor_lr + 0.5 * (peak - floor_lr) * (1.0 + std::cos(pi * q));
}

double clip_grad_norm(Model* model, double max_norm) {
    CHECK(max_norm > 0, "clip_grad_norm: max_norm %g must be > 0", max_norm);
    const size_t n = model->params_layout.num_params_padded;
    const double norm = std::sqrt(ops::sumsq_reduce(model->grads, n));
    CHECK(std::isfinite(norm), "clip_grad_norm: non-finite gradient norm");
    if (norm > max_norm) ops::scale_inplace(model->grads, static_cast<real_t>(max_norm / norm), n);
    return norm;
}

void adamw_update(Model* model, const TrainConfig& cfg, int step, double lr) {
    CHECK(step >= 1, "adamw_update: step %d must be >= 1", step);
    const real_t lr_r = static_cast<real_t>(lr);
    const real_t b1 = static_cast<real_t>(cfg.beta1), b2 = static_cast<real_t>(cfg.beta2);
    const real_t eps = static_cast<real_t>(cfg.adam_eps), wd = static_cast<real_t>(cfg.weight_decay);
    for (const TensorDesc& t : model->params_layout.tensors) {
        ops::adamw_step(model->params + t.offset, model->grads + t.offset, model->m + t.offset, model->v + t.offset,
                        t.numel, lr_r, b1, b2, eps, t.decay ? wd : real_t(0), step);
    }
}

}  // namespace tslm
