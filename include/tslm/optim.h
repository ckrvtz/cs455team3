// optim.h: learning-rate schedule, gradient clipping and AdamW over the
// model's flat buffers (ROADMAP §2.3, §4.3, §9). Host-side driver code;
// every tensor operation is an ops:: call.
//
// Learning rate at 1-based step s (§2.3: linear warmup, then cosine decay to
// min_lr_frac * peak), with W = warmup_steps, S = max_steps, P = lr,
// Pmin = min_lr_frac * P:
//   s <= W:  lr = P * s / W
//   s >  W:  q = min(1, (s - W) / (S - W)),  lr = Pmin + 0.5 (P - Pmin)(1 + cos(pi q))
// so lr(W) = P and lr(S) = Pmin. With W = 0 the cosine starts at step 1.
// Steps past S stay at Pmin.
//
// Step order (CLAUDE.md §4.7): ... backward -> comm_allreduce_avg(grads) ->
// clip_grad_norm -> adamw_update.
#pragma once

#include "tslm/config.h"
#include "tslm/model.h"

namespace tslm {

double lr_at(const TrainConfig& cfg, int step);

// Global L2 norm of model->grads (over all num_params_padded entries;
// padding is 0). If it exceeds max_norm, all grads are scaled by
// max_norm / norm. Returns the norm BEFORE clipping (what log.csv records).
double clip_grad_norm(Model* model, double max_norm);

// One AdamW step on every parameter tensor (ops::adamw_step per tensor in
// layout order). Weight decay is cfg.weight_decay for tensors whose layout
// entry has decay = true (2-D weight matrices except wpe; §2.3), else 0.
// Padding is never touched. step is 1-based.
void adamw_update(Model* model, const TrainConfig& cfg, int step, double lr);

}  // namespace tslm
