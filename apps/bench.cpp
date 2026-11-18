// bench: one fixed-workload benchmark run -> one CSV row (ROADMAP §12, P7.1).
//
//   bin/bench --config configs/small.cfg --max_steps 50 --out bench/serial_W-A.csv
//             [--version serial] [--machine_id "..."] [--prewarm 1] [config overrides ...]
//
// The workload is the config (preset, batch_size x ranks, seq_len, seed,
// train_data) with steps = max_steps. Bench options (--out, --version,
// --machine_id, --prewarm) are taken out of the argument list; everything
// else is a config override. Appends one row to --out (creating it with the
// header) and rewrites <out without .csv>.manifest.txt with the build and
// machine metadata, the prewarm record and the resolved config. See
// include/tslm/bench.h for what is timed; scripts/bench.sh runs a workload 3x.
#include <cstdio>
#include <string>
#include <vector>

#include "tslm/bench.h"
#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/logging.h"
#include "tslm/train.h"

using namespace tslm;

int main(int argc, char** argv) {
    comm_init(&argc, &argv);
    dev_init(0);  // Phase D: the node-local rank
    std::string out, version = build_version(), machine_id = default_machine_id();
    BenchOptions opt;
    std::vector<const char*> rest = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool ours = a == "--out" || a == "--version" || a == "--machine_id" || a == "--prewarm";
        if (!ours) {
            rest.push_back(argv[i]);
            continue;
        }
        CHECK(i + 1 < argc, "missing value for '%s'", a.c_str());
        const std::string v = argv[++i];
        if (a == "--out") out = v;
        else if (a == "--version") version = v;
        else if (a == "--machine_id") machine_id = v;
        else opt.prewarm = v != "0";
    }
    CHECK(!out.empty(), "--out <bench csv> is required");
    CHECK(machine_id.find(',') == std::string::npos, "--machine_id must not contain commas");
    const Config cfg = config_from_args_or_die(static_cast<int>(rest.size()), rest.data());

    BenchRow row = run_bench(cfg, &opt);
    row.version = version;
    row.machine_id = machine_id;
    row.git_hash = build_git_hash();
    if (comm_rank() == 0) {
        std::string err;
        CHECK(append_bench_row(out, row, &err), "%s", err.c_str());
        const std::string base = out.size() > 4 && out.compare(out.size() - 4, 4, ".csv") == 0
                                     ? out.substr(0, out.size() - 4)
                                     : out;
        char b[256];
        std::snprintf(b, sizeof(b), "# prewarm: %s (%llu bytes of %s read in %.2f s)\n", opt.prewarm ? "yes" : "no",
                      static_cast<unsigned long long>(opt.prewarm_bytes), cfg.train.train_data.c_str(),
                      opt.prewarm_s);
        const std::string text = "# tslm bench manifest (ROADMAP §12): latest run appended to " + out + "\n" +
                                 build_info_text(comm_size()) + b + "# machine_id: " + machine_id + "\n" +
                                 config_to_text(cfg);
        std::FILE* f = std::fopen((base + ".manifest.txt").c_str(), "w");
        CHECK(f, "cannot write %s.manifest.txt", base.c_str());
        CHECK(std::fwrite(text.data(), 1, text.size(), f) == text.size() && std::fclose(f) == 0,
              "write to %s.manifest.txt failed", base.c_str());
        log_info("%s", BENCH_CSV_HEADER);
        log_info("%s", format_bench_row(row).c_str());
    }
    comm_finalize();
    return 0;
}
