// train.cpp: the training loop (ROADMAP §9). See include/tslm/train.h for
// the step order, logging cadence and outputs.
#include "tslm/train.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "tslm/backend.h"
#include "tslm/checkpoint.h"
#include "tslm/comm.h"
#include "tslm/common.h"
#include "tslm/dataset.h"
#include "tslm/eval.h"
#include "tslm/logging.h"
#include "tslm/optim.h"
#include "tslm/timer.h"

namespace tslm {

void check_step_finite(int step, double loss, double grad_norm) {
    CHECK(std::isfinite(loss), "step %d: non-finite training loss %g", step, loss);
    CHECK(std::isfinite(grad_norm), "step %d: non-finite gradient norm %g", step, grad_norm);
}

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

void write_text_file(const std::string& path, const std::string& text) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    CHECK(f, "cannot create %s", path.c_str());
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    CHECK(std::fclose(f) == 0 && ok, "write to %s failed", path.c_str());
}

}  // namespace

StepResult train_step(Model* model, DataLoader* loader, const TrainConfig& tc, int step, const StepBuffers& buf,
                      PhaseTimer* timer) {
    const int B = loader->batch_size(), T = loader->seq_len(), world = comm_size();
    const size_t ntok = size_t(B) * size_t(T);
    const double t0 = now_seconds();

    timer->begin(Phase::Data);
    loader->next_batch(buf.x_host, buf.y_host);
    copy_h2d(buf.x_dev, buf.x_host, ntok * sizeof(int32_t));
    copy_h2d(buf.y_dev, buf.y_host, ntok * sizeof(int32_t));
    timer->end(Phase::Data);

    timer->begin(Phase::Bwd);  // grad zeroing is charged to bwd (DECISIONS 2026-11-18)
    dev_memset(model->grads, 0, model_param_bytes(model));
    timer->end(Phase::Bwd);

    timer->begin(Phase::Fwd);
    double loss = model_forward(model, buf.x_dev, buf.y_dev, B, T);
    timer->end(Phase::Fwd);

    timer->begin(Phase::Bwd);
    model_backward(model);
    timer->end(Phase::Bwd);

    timer->begin(Phase::Comm);
    comm_allreduce_avg(model->grads, model->params_layout.num_params_padded);
    comm_allreduce_sum_double(&loss, 1);
    loss /= world;
    timer->end(Phase::Comm);
    check_step_finite(step, loss, 0.0);

    StepResult r;
    timer->begin(Phase::Opt);
    r.lr = lr_at(tc, step);
    r.grad_norm = clip_grad_norm(model, tc.grad_clip);
    adamw_update(model, tc, step, r.lr);
    timer->end(Phase::Opt);
    check_step_finite(step, loss, r.grad_norm);

    r.loss = loss;
    r.step_ms = (now_seconds() - t0) * 1e3;
    return r;
}

TrainSummary train_run(const Config& cfg) {
    std::string err;
    CHECK(config_validate(cfg, &err), "%s", err.c_str());
    const TrainConfig& tc = cfg.train;
    const int rank = comm_rank(), world = comm_size();
    const bool lead = rank == 0;
    const int B = tc.batch_size, T = tc.seq_len;

    DataLoader loader;
    CHECK(loader.open(tc.train_data, B, T, tc.seed, rank, world, &err), "%s", err.c_str());
    CHECK(loader.vocab_size() == static_cast<uint32_t>(cfg.model.vocab_size),
          "%s has vocab_size %u but the model has vocab_size %d", tc.train_data.c_str(), loader.vocab_size(),
          cfg.model.vocab_size);

    const uint32_t global_batch = static_cast<uint32_t>(B * world);
    DataLoader val;
    CHECK(val.open(tc.val_data, B, T, tc.seed, rank, world, &err), "%s", err.c_str());
    CHECK(val.vocab_size() == loader.vocab_size(), "%s has vocab_size %u but %s has %u", tc.val_data.c_str(),
          val.vocab_size(), tc.train_data.c_str(), loader.vocab_size());
    CHECK(uint64_t(tc.val_batches) * global_batch <= val.num_chunks(),
          "val_batches=%d needs %llu chunks of T=%d at global batch %u; %s has %llu", tc.val_batches,
          static_cast<unsigned long long>(uint64_t(tc.val_batches) * global_batch), T, global_batch,
          tc.val_data.c_str(), static_cast<unsigned long long>(val.num_chunks()));

    Model* model = model_create(cfg.model, B, T);
    int start_step = 1;
    uint64_t tokens_at_start = 0;
    if (tc.resume.empty()) {
        model_init_weights(model, tc.seed);
    } else {
        // Every rank reads the same file, so params/m/v agree without a broadcast.
        CheckpointInfo ck;
        CHECK(checkpoint_load(tc.resume, model, &ck, &err), "%s", err.c_str());
        CHECK(ck.seed == tc.seed && ck.seq_len == static_cast<uint32_t>(T) && ck.global_batch == global_batch,
              "%s was written with seed %llu, T=%u, global batch %u; this run has seed %llu, T=%d, global batch "
              "%u (the data position would not mean the same thing)",
              tc.resume.c_str(), static_cast<unsigned long long>(ck.seed), ck.seq_len, ck.global_batch,
              static_cast<unsigned long long>(tc.seed), T, global_batch);
        CHECK(ck.step < static_cast<uint64_t>(tc.max_steps), "%s is at step %llu; max_steps=%d leaves nothing to run",
              tc.resume.c_str(), static_cast<unsigned long long>(ck.step), tc.max_steps);
        loader.set_state(ck.loader);
        start_step = static_cast<int>(ck.step) + 1;
        tokens_at_start = ck.tokens_seen;
    }
    // Every rank starts from rank 0's weights (no-op in serial; §4.3).
    comm_broadcast(model->params, model->params_layout.num_params_padded, 0);

    // Host batch buffers (token ids are host data) and their device copies.
    const size_t ntok = size_t(B) * size_t(T);
    std::vector<int32_t> x_host(ntok), y_host(ntok);
    int32_t* x_dev = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    int32_t* y_dev = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    const StepBuffers buf{x_host.data(), y_host.data(), x_dev, y_dev};

    const std::string dir = run_dir(cfg);
    CsvLog csv;
    if (lead) {
        make_dirs(dir);
        if (tc.resume.empty()) {
            if (file_exists(dir + "/log.csv")) log_warn("overwriting the existing run in %s", dir.c_str());
            CHECK(csv.open(dir + "/log.csv", &err), "%s", err.c_str());
        } else {
            CHECK(csv.open_resume(dir + "/log.csv", start_step - 1, &err), "%s", err.c_str());
            log_info("resumed from %s at step %d (log.csv rows after it dropped)", tc.resume.c_str(),
                     start_step - 1);
        }
        write_text_file(dir + "/manifest.txt",
                        manifest_text(cfg, model, loader.num_tokens(), loader.steps_per_epoch(), world));
        log_info("run %s: %s", dir.c_str(), model_summary(model).c_str());
        log_info("data %s: %llu tokens, %llu steps/epoch at global batch %d x T=%d; %d steps", tc.train_data.c_str(),
                 static_cast<unsigned long long>(loader.num_tokens()),
                 static_cast<unsigned long long>(loader.steps_per_epoch()), B * world, T, tc.max_steps);
        log_info("validation %s: first %d batches (%llu tokens) every %d steps and at the end",
                 tc.val_data.c_str(), tc.val_batches,
                 static_cast<unsigned long long>(uint64_t(tc.val_batches) * global_batch * uint64_t(T)),
                 tc.eval_every);
    }
    if (tc.resume.empty()) {
        // Fresh model: expect about ln(V) (P4.4); console only, log.csv starts at step 1.
        const double v0 = evaluate(model, val, tc.val_batches, x_host.data(), y_host.data(), x_dev, y_dev);
        if (lead)
            log_info("val at init: loss %.4f (ln V = %.4f), ppl %.1f", v0, std::log(double(cfg.model.vocab_size)),
                     std::exp(v0));
    }

    const uint64_t tokens_per_step = uint64_t(ntok) * uint64_t(world);
    PhaseTimer timer;
    TrainSummary sum;
    double interval_ms = 0.0;
    int interval_steps = 0;
    const double loop_t0 = now_seconds();

    sum.start_step = start_step;
    for (int step = start_step; step <= tc.max_steps; ++step) {
        const StepResult sr = train_step(model, &loader, tc, step, buf, &timer);
        const double loss = sr.loss, lr = sr.lr, gnorm = sr.grad_norm, step_ms = sr.step_ms;
        interval_ms += step_ms;
        ++interval_steps;
        sum.steps = step;
        sum.tokens_seen = tokens_at_start + uint64_t(step - start_step + 1) * tokens_per_step;
        if (step == start_step) sum.first_loss = loss;
        sum.final_loss = loss;

        // Validation runs after the step's timing, so step_ms excludes it.
        const bool eval_now = step % tc.eval_every == 0 || step == tc.max_steps;
        double val_loss = std::nan("");
        if (eval_now) {
            val_loss = evaluate(model, val, tc.val_batches, x_host.data(), y_host.data(), x_dev, y_dev);
            sum.final_val_loss = val_loss;
        }

        if (step == 1 || step % tc.log_every == 0 || eval_now) {
            const double n = interval_steps;
            LogRow r;
            r.val_loss = val_loss;
            r.val_ppl = std::exp(val_loss);
            r.step = step;
            r.tokens_seen = sum.tokens_seen;
            r.train_loss = loss;
            r.lr = lr;
            r.grad_norm = gnorm;
            r.tok_per_s = double(tokens_per_step) * n / (interval_ms / 1e3);
            r.step_ms = interval_ms / n;
            r.data_ms = timer.ms(Phase::Data) / n;
            r.fwd_ms = timer.ms(Phase::Fwd) / n;
            r.bwd_ms = timer.ms(Phase::Bwd) / n;
            r.comm_ms = timer.ms(Phase::Comm) / n;
            r.opt_ms = timer.ms(Phase::Opt) / n;
            if (lead) {
                csv.write(r);
                char vbuf[64] = "";
                if (eval_now) std::snprintf(vbuf, sizeof(vbuf), " | val %.4f ppl %.2f", val_loss, r.val_ppl);
                log_info("step %6d/%d | loss %.4f | lr %.3e | gnorm %.3f | %.1f ms/step | %.0f tok/s%s", step,
                         tc.max_steps, loss, lr, gnorm, r.step_ms, r.tok_per_s, vbuf);
            }
            timer.reset();
            interval_ms = 0.0;
            interval_steps = 0;
        }

        // Checkpoints are written after the step's timing, so step_ms excludes I/O.
        if ((tc.ckpt_every > 0 && step % tc.ckpt_every == 0) || step == tc.max_steps) {
            if (lead) {
                CheckpointInfo ck;
                ck.step = static_cast<uint64_t>(step);
                ck.tokens_seen = sum.tokens_seen;
                ck.loader = loader.state();
                ck.seed = tc.seed;
                ck.global_batch = global_batch;
                ck.seq_len = static_cast<uint32_t>(T);
                const double c0 = now_seconds();
                char name[64];
                std::snprintf(name, sizeof(name), "/ckpt_%d.bin", step);
                checkpoint_save(dir + name, model, ck);
                checkpoint_save(dir + "/ckpt_latest.bin", model, ck);
                log_info("checkpoint %s%s (+ ckpt_latest.bin), %.0f ms", dir.c_str(), name,
                         (now_seconds() - c0) * 1e3);
            }
            comm_barrier();  // no rank runs ahead of a checkpoint being written
        }
    }
    sum.wall_s = now_seconds() - loop_t0;
    if (lead) {
        csv.close();
        log_info("done: %d steps, %llu tokens, %.1f s; loss %.4f -> %.4f", sum.steps,
                 static_cast<unsigned long long>(sum.tokens_seen), sum.wall_s, sum.first_loss, sum.final_loss);
    }

    dev_free(x_dev);
    dev_free(y_dev);
    model_free(model);
    return sum;
}

}  // namespace tslm
