// train.h: the training loop (ROADMAP §9) and the run manifest. Host-side
// driver; every tensor operation goes through model_*, optim (ops::) and the
// backend/comm shims.
//
// One step (CLAUDE.md §4.7 order; phases as in log.csv, with grad zeroing
// timed under bwd per docs/DECISIONS.md 2026-11-18):
//   data : loader.next_batch(x, y) on the host; copy_h2d to the token buffers
//   bwd  : zero grads (dev_memset)
//   fwd  : loss = model_forward(x, y)
//   bwd  : model_backward
//   comm : comm_allreduce_avg(grads); loss averaged over ranks
//          check_step_finite(loss)
//   opt  : lr = lr_at(step); grad_norm = clip_grad_norm(clip); adamw_update
// Steps are 1-based. step_ms is the wall time of the whole step, so the five
// phase columns sum to it up to the untimed bookkeeping.
//
// log.csv gets a row at step 1, every log_every steps, and at the last step.
// train_loss, lr and grad_norm are that step's values (bit-reproducible, so
// rows can be compared across builds). The timing columns and tok_per_s are
// per-step means over the steps since the previous row. tokens_seen and
// tok_per_s count GLOBAL tokens (B * T * world_size per step).
//
// Outputs (rank 0 only) in <out_dir>/<run_name>/ (created if missing;
// existing files are overwritten with a warning):
//   manifest.txt  '#'-comment metadata (date, git hash, build, compiler and
//                 flags, device, CPU, cores, RAM, world size, parameter
//                 counts, data files), then config_to_text(cfg). The comment
//                 lines are ignored by the config parser, so
//                 `--config <run>/manifest.txt` reproduces the run's config.
//   log.csv       logging.h format.
// Not yet produced: validation columns (P4.4), checkpoints (P4.3) and
// samples (Phase 6). train_run states this in its startup log.
#pragma once

#include <cstdint>
#include <string>

#include "tslm/config.h"
#include "tslm/model.h"

namespace tslm {

struct TrainSummary {
    int steps = 0;
    uint64_t tokens_seen = 0;
    double first_loss = 0.0;  // train loss at step 1
    double final_loss = 0.0;  // train loss at the last step
    double wall_s = 0.0;      // total wall time of the step loop
};

// Run the full loop for cfg (validated with config_validate; CHECKs on any
// error). comm_init must have been called.
TrainSummary train_run(const Config& cfg);

// Abort (CHECK) with the step number if the loss or the gradient norm is
// NaN or infinite. CLAUDE.md §10: non-finite values are never ignored.
void check_step_finite(int step, double loss, double grad_norm);

// <out_dir>/<run_name>
std::string run_dir(const Config& cfg);

// Full manifest.txt text. `model` and the data sizes supply the parameter
// counts and dataset lines; world_size is recorded as given.
std::string manifest_text(const Config& cfg, const Model* model, uint64_t train_tokens,
                          uint64_t steps_per_epoch, int world_size);

// mkdir -p; CHECKs on failure.
void make_dirs(const std::string& path);

}  // namespace tslm
