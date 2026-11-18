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
// Validation (eval.h) runs every eval_every steps and at the last step, on
// the fixed first val_batches batches of val_data, after the step's timing
// (step_ms excludes it); a fresh run also evaluates once before step 1 and
// prints the result (expected ~ln V). val_loss / val_ppl are filled only on
// rows of steps where validation ran.
//
// log.csv gets a row at step 1, every log_every steps, at every validation
// step, and at the last step.
// train_loss, lr and grad_norm are that step's values (bit-reproducible, so
// rows can be compared across builds). The timing columns and tok_per_s are
// per-step means over the steps since the previous row. tokens_seen and
// tok_per_s count GLOBAL tokens (B * T * world_size per step).
//
// Outputs (rank 0 only) in <out_dir>/<run_name>/ (created if missing; a
// fresh run overwrites existing files with a warning):
//   manifest.txt  '#'-comment metadata (date, git hash, build, compiler and
//                 flags, device, CPU, cores, RAM, world size, parameter
//                 counts, data files), then config_to_text(cfg). The comment
//                 lines are ignored by the config parser, so
//                 `--config <run>/manifest.txt` reproduces the run's config.
//   log.csv       logging.h format.
//   ckpt_<step>.bin and ckpt_latest.bin (checkpoint.h) every ckpt_every
//                 steps (0 = none) and always at max_steps; written after the
//                 step's timing, so step_ms excludes checkpoint I/O.
// Not yet produced: samples (Phase 6).
//
// Resume (cfg.train.resume = checkpoint path): params, m, v, the step and the
// loader position come from the checkpoint; its model config, seed, T and
// global batch must equal the run's (checked). The run continues at step + 1
// up to max_steps; other settings (lr, max_steps, log_every, ...) come from
// cfg. log.csv in the run directory keeps its rows up to the checkpoint step
// and the later ones are dropped (they are rerun); manifest.txt is rewritten
// with the resolved config, which includes the resume key. A resumed serial
// run is bit-identical to an uninterrupted one (tests/test_checkpoint.cpp).
#pragma once

#include <cstdint>
#include <string>

#include "tslm/config.h"
#include "tslm/dataset.h"
#include "tslm/model.h"
#include "tslm/timer.h"

namespace tslm {

struct TrainSummary {
    int start_step = 1;  // first step run (> 1 when resumed)
    int steps = 0;       // last step run
    uint64_t tokens_seen = 0;
    double first_loss = 0.0;  // train loss at start_step
    double final_loss = 0.0;  // train loss at the last step
    double final_val_loss = 0.0;  // val loss at the last evaluation
    double wall_s = 0.0;      // total wall time of the step loop
};

// Host batch buffers and their device copies, each B*T ids.
struct StepBuffers {
    int32_t* x_host;
    int32_t* y_host;
    int32_t* x_dev;
    int32_t* y_dev;
};

struct StepResult {
    double loss = 0.0;       // mean over ranks
    double lr = 0.0;
    double grad_norm = 0.0;  // before clipping
    double step_ms = 0.0;    // wall time of the whole step
};

// One training step in the fixed order above, on the loader's next batch
// (B, T from the loader). Phase times ACCUMULATE into *timer. Shared by
// train_run and the bench app, so benchmarks time exactly the training step.
StepResult train_step(Model* model, DataLoader* loader, const TrainConfig& tc, int step, const StepBuffers& buf,
                      PhaseTimer* timer);

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

// The '#' metadata lines of a manifest (date, git, build, compiler, flags,
// device, CPU, cores, RAM, world size); shared by manifest_text and bench.
std::string build_info_text(int world_size);

// CPU model from /proc/cpuinfo ("unknown" elsewhere), and the git hash the
// Makefile compiled in ("<hash>", "<hash>-dirty" or "unknown").
std::string cpu_model_name();
const char* build_git_hash();

// mkdir -p; CHECKs on failure.
void make_dirs(const std::string& path);

}  // namespace tslm
