// gradcheck: full-model finite-difference gradient check (ROADMAP §10, P3.6).
//
//   make REAL=double && bin/release-double/gradcheck [--tie_embeddings 0] [--seed N]
//        [--scale S] [--samples K] [--h H] [model/batch config keys...]
// Defaults: the §10 tiny model (V=32, max_seq_len=8, L=2, C=16, NH=2, B=2,
// T=8), seed 1337, scale 3, 20 samples per tensor, h = 3e-4. Config keys
// (config.h) override the model and batch (--batch_size, --seq_len, --seed).
// Prints a per-tensor table and exits 0 on PASS, 1 on FAIL. Requires the
// REAL=double build: the §10 criterion (rel err < 1e-6) is defined there.
#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/config.h"
#include "tslm/gradcheck.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tslm;

static double parse_double_or_die(const std::string& key, const std::string& v) {
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    CHECK(!v.empty() && errno == 0 && *end == '\0' && std::isfinite(x), "--%s: bad value '%s'", key.c_str(), v.c_str());
    return x;
}

int main(int argc, char** argv) {
    dev_init(0);
    CHECK(REAL_IS_DOUBLE, "gradcheck requires the double-precision build: run `make REAL=double` and use "
                          "bin/release-double/gradcheck");
    GradcheckOptions opt = default_gradcheck_options();
    // Start the config from the gradcheck defaults so config keys override them.
    Config cfg;
    cfg.model = opt.cfg;
    cfg.train.batch_size = opt.B;
    cfg.train.seq_len = opt.T;
    cfg.train.seed = opt.seed;
    std::string err;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s'", arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "scale") opt.weight_scale = parse_double_or_die(key, val);
        else if (key == "h") opt.h = parse_double_or_die(key, val);
        else if (key == "samples") opt.samples = static_cast<int>(parse_double_or_die(key, val));
        else CHECK(config_set(&cfg, key, val, &err), "%s", err.c_str());
    }
    CHECK(config_validate(cfg, &err), "%s", err.c_str());
    opt.cfg = cfg.model;
    opt.B = cfg.train.batch_size;
    opt.T = cfg.train.seq_len;
    opt.seed = cfg.train.seed;

    const GradcheckReport r = run_gradcheck(opt);
    std::fputs(format_gradcheck_report(opt, r).c_str(), stdout);
    return r.pass ? 0 : 1;
}
