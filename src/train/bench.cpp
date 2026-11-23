// bench.cpp: see include/tslm/bench.h.
#include "tslm/bench.h"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/common.h"
#include "tslm/dataset.h"
#include "tslm/logging.h"
#include "tslm/model.h"
#include "tslm/timer.h"
#include "tslm/train.h"

namespace tslm {

const char* const BENCH_CSV_HEADER =
    "version,preset,global_batch,T,steps,ranks,median_step_ms,mean_step_ms,tok_per_s,data_ms,fwd_ms,bwd_ms,"
    "comm_ms,opt_ms,peak_mem_MB,machine_id,git_hash";

double median_of(std::vector<double> v) {
    CHECK(!v.empty(), "median_of: empty list");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string format_bench_row(const BenchRow& r) {
    for (const std::string* s : {&r.version, &r.preset, &r.machine_id, &r.git_hash})
        CHECK(s->find(',') == std::string::npos && s->find('\n') == std::string::npos,
              "bench CSV field '%s' contains a comma or newline", s->c_str());
    char b[1024];
    std::snprintf(b, sizeof(b), "%s,%s,%d,%d,%d,%d,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s",
                  r.version.c_str(), r.preset.c_str(), r.global_batch, r.T, r.steps, r.ranks, r.median_step_ms,
                  r.mean_step_ms, r.tok_per_s, r.data_ms, r.fwd_ms, r.bwd_ms, r.comm_ms, r.opt_ms, r.peak_mem_MB,
                  r.machine_id.c_str(), r.git_hash.c_str());
    return b;
}

bool append_bench_row(const std::string& path, const BenchRow& r, std::string* err) {
    const std::string row = format_bench_row(r);
    bool need_header = true;
    if (std::FILE* in = std::fopen(path.c_str(), "r")) {
        char line[512] = {0};
        const bool got = std::fgets(line, sizeof(line), in) != nullptr;
        std::fclose(in);
        if (got) {
            std::string first(line);
            if (!first.empty() && first.back() == '\n') first.pop_back();
            if (first != BENCH_CSV_HEADER) {
                if (err) *err = path + ": first line is not the bench CSV header; refusing to append";
                return false;
            }
            need_header = false;
        }
    }
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        if (err) *err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    bool ok = true;
    if (need_header) ok = std::fprintf(f, "%s\n", BENCH_CSV_HEADER) >= 0;
    ok = ok && std::fprintf(f, "%s\n", row.c_str()) >= 0;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok && err) *err = "write to " + path + " failed";
    return ok;
}

const char* build_version() {
#if defined(TSLM_BACKEND_CUDA) && defined(TSLM_COMM_MPI)
    return "mpi_cuda";
#elif defined(TSLM_BACKEND_CUDA)
    return "cuda";
#elif defined(TSLM_COMM_MPI)
    return "mpi";
#else
    return "serial";
#endif
}

std::string default_machine_id() {
    std::string cpu = cpu_model_name();
    std::replace(cpu.begin(), cpu.end(), ',', ';');
    const long cores = ::sysconf(_SC_NPROCESSORS_ONLN);
    const long pages = ::sysconf(_SC_PHYS_PAGES), page = ::sysconf(_SC_PAGE_SIZE);
    const double gb = (pages > 0 && page > 0) ? double(pages) * double(page) / (1024.0 * 1024.0 * 1024.0) : -1.0;
    char b[512];
    std::snprintf(b, sizeof(b), "%s | %ld cores | %.0f GB", cpu.c_str(), cores, gb);
    return b;
}

namespace {

// Read the whole file once so its pages are in the OS page cache.
void prewarm_file(const std::string& path, BenchOptions* opt) {
    const double t0 = now_seconds();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f, "prewarm: cannot open %s: %s", path.c_str(), std::strerror(errno));
    static char buf[1 << 20];
    uint64_t total = 0;
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) total += n;
    CHECK(!std::ferror(f), "prewarm: read error on %s", path.c_str());
    std::fclose(f);
    opt->prewarm_bytes = total;
    opt->prewarm_s = now_seconds() - t0;
}

}  // namespace

BenchRow run_bench(const Config& cfg, BenchOptions* opt) {
    std::string err;
    CHECK(config_validate(cfg, &err), "%s", err.c_str());
    const TrainConfig& tc = cfg.train;
    CHECK(tc.max_steps > BENCH_WARMUP_STEPS, "bench: max_steps=%d must exceed the %d warmup steps", tc.max_steps,
          BENCH_WARMUP_STEPS);
    CHECK(tc.resume.empty(), "bench: resume is not allowed (a benchmark starts from a fresh model)");
    const int rank = comm_rank(), world = comm_size(), B = tc.batch_size, T = tc.seq_len;

    if (opt->prewarm) prewarm_file(tc.train_data, opt);
    DataLoader loader;
    CHECK(loader.open(tc.train_data, B, T, tc.seed, rank, world, &err), "%s", err.c_str());
    CHECK(loader.vocab_size() == static_cast<uint32_t>(cfg.model.vocab_size),
          "%s has vocab_size %u but the model has vocab_size %d", tc.train_data.c_str(), loader.vocab_size(),
          cfg.model.vocab_size);
    Model* model = model_create(cfg.model, B, T);
    model_init_weights(model, tc.seed);
    comm_broadcast(model->params, model->params_layout.num_params_padded, 0);

    const size_t ntok = size_t(B) * size_t(T);
    std::vector<int32_t> x_host(ntok), y_host(ntok);
    int32_t* x_dev = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    int32_t* y_dev = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    const StepBuffers buf{x_host.data(), y_host.data(), x_dev, y_dev};

    std::vector<double> step_ms;
    PhaseTimer timer;
    double phase_sum[NUM_PHASES] = {0, 0, 0, 0, 0};
    for (int step = 1; step <= tc.max_steps; ++step) {
        timer.reset();
        const StepResult r = train_step(model, &loader, tc, step, buf, &timer);
        if (step <= BENCH_WARMUP_STEPS) continue;
        step_ms.push_back(r.step_ms);
        for (int p = 0; p < NUM_PHASES; ++p) phase_sum[p] += timer.ms(static_cast<Phase>(p));
    }
    dev_free(x_dev);
    dev_free(y_dev);
    model_free(model);

    BenchRow row;
    row.preset = tc.run_name;
    row.global_batch = B * world;
    row.T = T;
    row.steps = tc.max_steps;
    row.ranks = world;
    opt->measured_steps = static_cast<int>(step_ms.size());
    const double n = double(step_ms.size());
    row.median_step_ms = median_of(step_ms);
    double sum = 0;
    for (double v : step_ms) sum += v;
    row.mean_step_ms = sum / n;
    row.tok_per_s = double(ntok) * double(world) / (row.median_step_ms / 1e3);
    row.data_ms = phase_sum[int(Phase::Data)] / n;
    row.fwd_ms = phase_sum[int(Phase::Fwd)] / n;
    row.bwd_ms = phase_sum[int(Phase::Bwd)] / n;
    row.comm_ms = phase_sum[int(Phase::Comm)] / n;
    row.opt_ms = phase_sum[int(Phase::Opt)] / n;
#if defined(TSLM_BACKEND_CUDA)
    // CUDA: device memory requested through dev_malloc (ROADMAP_CUDA §4.4, D-B8).
    row.peak_mem_MB = double(dev_peak_bytes()) / (1024.0 * 1024.0);
#else
    struct rusage ru;
    CHECK(getrusage(RUSAGE_SELF, &ru) == 0, "getrusage failed");
    row.peak_mem_MB = double(ru.ru_maxrss) / 1024.0;  // Linux: kilobytes
#endif
    return row;
}

}  // namespace tslm
