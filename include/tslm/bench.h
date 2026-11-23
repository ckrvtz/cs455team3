// bench.h: fixed-workload benchmark (ROADMAP §12, P7.1). Host-side driver.
//
// One bench run = one workload (preset config, global batch, T, seed,
// steps = max_steps) timed with the exact training step (train_step): fresh
// model from the seed, no validation, no checkpoints, no log.csv. It
// produces ONE CSV row; a benchmark is >= 3 runs (scripts/bench.sh).
//
// Timing:
//   - Before the first step, the whole training token file is read once
//     (prewarm, on by default), so random-chunk reads do not hit a cold page
//     cache (measured in P4.2: 54.5 s cold vs 5.0 s warm for the smoke run).
//   - Steps 1..BENCH_WARMUP_STEPS are run but discarded; steps
//     BENCH_WARMUP_STEPS+1..steps are measured (steps must exceed the warmup).
//   - median_step_ms / mean_step_ms: median and mean of the measured step
//     wall times. tok_per_s = global tokens per step / (median_step_ms / 1000).
//   - data_ms .. opt_ms: per-step MEANS of each phase over the measured
//     steps (they sum to mean_step_ms up to untimed bookkeeping).
//   - peak_mem_MB: CPU builds: peak resident set size of the process
//     (getrusage ru_maxrss), host memory. CUDA builds: dev_peak_bytes() / 2^20,
//     the high-water mark of device memory requested through dev_malloc.
//
// CSV (FROZEN, §12): header exactly BENCH_CSV_HEADER; a row per run.
// Strings must not contain commas (machine_id is sanitized; version and git
// hash never do).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tslm/config.h"

namespace tslm {

constexpr int BENCH_WARMUP_STEPS = 3;
extern const char* const BENCH_CSV_HEADER;

struct BenchRow {
    std::string version;  // serial | cuda | mpi | mpi_cuda
    std::string preset;   // run_name of the config (smoke, small, base)
    int global_batch = 0, T = 0, steps = 0, ranks = 0;
    double median_step_ms = 0, mean_step_ms = 0, tok_per_s = 0;
    double data_ms = 0, fwd_ms = 0, bwd_ms = 0, comm_ms = 0, opt_ms = 0;
    double peak_mem_MB = 0;
    std::string machine_id, git_hash;
};

struct BenchOptions {
    bool prewarm = true;
    double prewarm_s = 0.0;   // out: seconds spent reading the token file
    uint64_t prewarm_bytes = 0;
    int measured_steps = 0;   // out: steps timed after the warmup
};

// Median of a non-empty list (mean of the two middle values when even).
double median_of(std::vector<double> v);

// Format one row (no newline) with %.3f ms, %.1f tok/s, %.1f MB.
std::string format_bench_row(const BenchRow& r);

// Append r to path; creates the file with the header if it does not exist,
// and refuses (false, *err) if an existing file's first line is not the header.
bool append_bench_row(const std::string& path, const BenchRow& r, std::string* err);

// "serial" / "cuda" / "mpi" / "mpi_cuda" for this build.
const char* build_version();

// CPU model, online cores and RAM, with commas replaced, e.g.
// "Intel(R) Xeon(R) Processor @ 2.10GHz | 4 cores | 16 GB".
std::string default_machine_id();

// Run the workload (cfg.train.max_steps steps) and fill everything except
// machine_id / git_hash / version. comm_init must have been called.
BenchRow run_bench(const Config& cfg, BenchOptions* opt);

}  // namespace tslm
