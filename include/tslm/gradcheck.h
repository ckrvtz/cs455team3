// gradcheck.h: full-model finite-difference gradient check (ROADMAP §10, P3.6).
//
// Method (ROADMAP §10 Gradcheck row; docs/DECISIONS.md 2026-11-18):
//  * Model and batch: GradcheckOptions::cfg (default: the §10 tiny model
//    L=2, C=16, NH=2, V=32, T=8, B=2), weights from model_init_weights(seed)
//    times weight_scale (default 3), and tokens/targets drawn uniformly from
//    Rng(seed, GRADCHECK_BATCH_STREAM). At standard init (scale 1) some
//    gradients are ~1e-7, below what finite differences can resolve to 1e-6;
//    x3 gives non-uniform attention and resolvable gradients while testing
//    the same code (docs/DECISIONS.md, gradcheck configuration).
//  * Analytic gradient: zero grads, model_forward, model_backward.
//  * Sampling per parameter tensor: its first and last index plus up to
//    `samples` distinct random indices. For qkv_b, samples are drawn only
//    from the query [0,C) and value [2C,3C) slices. The key-bias slice
//    [C,2C) has a structurally zero gradient (softmax shift invariance) and
//    is checked directly instead (below).
//  * Numeric derivative: 4th-order 5-point central difference, evaluated
//    differences-first so a loss independent of the entry gives exactly 0:
//      n = [8 (L(x+h) - L(x-h)) - (L(x+2h) - L(x-2h))] / (12 h)
//    The classic 2-point estimate (L(x+e) - L(x-e)) / 2e with e = 1e-5 is
//    also computed, for diagnosis only.
//  * Pass criterion: every sampled entry has |a - n| / max(|a|, |n|, floor)
//    < tol, AND for every layer max|d key-bias| <= kbias_ratio *
//    max|d query/value-bias|. Nothing passes on a noise floor.
// Defaults are the §10 values for the REAL=double build (h 3e-4, weights x3,
// tol 1e-6, floor 1e-8, kbias_ratio 1e-12). The float build cannot meet them.
//
// Parameters are read and perturbed only through copy_d2h / copy_h2d, one
// element at a time, so the check runs unchanged on device memory. Each
// perturbed parameter is restored to its exact original value.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tslm/config.h"

namespace tslm {

constexpr uint64_t GRADCHECK_BATCH_STREAM = 910000;  // distinct from init/data/golden streams

struct GradcheckOptions {
    ModelConfig cfg;  // set to the §10 tiny model by default_gradcheck_options()
    int B = 2, T = 8;
    uint64_t seed = 1337;
    double weight_scale = 3.0;  // init weights x3 (docs/DECISIONS.md: gradcheck configuration)
    int samples = 20;     // random indices per tensor, in addition to first/last
    double h = 3e-4;      // 5-point stencil step
    double eps_2pt = 1e-5;  // diagnostic 2-point step
    double tol = 1e-6;
    double floor = 1e-8;
    double kbias_ratio = 1e-12;
};

GradcheckOptions default_gradcheck_options();

struct TensorCheck {
    std::string name;
    int checked = 0;
    double worst_rel = 0;       // 5-point stencil, the pass criterion
    double grad_at_worst = 0;   // analytic value at the worst entry
    double numeric_at_worst = 0;
    double worst_rel_2pt = 0;   // diagnostic only
    bool pass = true;
};

struct GradcheckReport {
    double loss = 0;
    std::vector<TensorCheck> tensors;
    std::vector<double> kbias_ratios;  // per layer: max|d key-bias| / max|d q/v-bias|
    bool kbias_pass = true;
    bool pass = true;
};

GradcheckReport run_gradcheck(const GradcheckOptions& opt);

// Human-readable table: one line per tensor, the key-bias checks, verdict.
std::string format_gradcheck_report(const GradcheckOptions& opt, const GradcheckReport& r);

}  // namespace tslm
