// manifest.cpp: run directory and manifest.txt (ROADMAP §9, §12). See train.h.
//
// TSLM_GIT_HASH, TSLM_CXX and TSLM_CXXFLAGS are passed by the Makefile to
// this file only, so a new commit rebuilds just this object. Machine facts
// (CPU model, cores, RAM) come from POSIX sysconf and Linux /proc/cpuinfo;
// on other systems the CPU model is recorded as "unknown".
#include "tslm/train.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "tslm/backend.h"
#include "tslm/build_config.h"
#include "tslm/common.h"

#ifndef TSLM_GIT_HASH
#define TSLM_GIT_HASH "unknown"
#endif
#ifndef TSLM_CXX
#define TSLM_CXX "unknown"
#endif
#ifndef TSLM_CXXFLAGS
#define TSLM_CXXFLAGS "unknown"
#endif

namespace tslm {

std::string run_dir(const Config& cfg) { return cfg.train.out_dir + "/" + cfg.train.run_name; }

void make_dirs(const std::string& path) {
    CHECK(!path.empty(), "make_dirs: empty path");
    for (size_t pos = 1; pos <= path.size(); ++pos) {
        if (pos != path.size() && path[pos] != '/') continue;
        const std::string prefix = path.substr(0, pos);
        if (::mkdir(prefix.c_str(), 0755) != 0 && errno != EEXIST)
            CHECK(false, "cannot create directory %s: %s", prefix.c_str(), std::strerror(errno));
    }
    struct stat st;
    CHECK(::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode), "%s is not a directory", path.c_str());
}

std::string cpu_model_name() {
    std::FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return "unknown";
    char line[512];
    std::string model = "unknown";
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "model name", 10) != 0) continue;
        const char* colon = std::strchr(line, ':');
        if (!colon) continue;
        std::string s(colon + 1);
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        model = s;
        break;
    }
    std::fclose(f);
    return model;
}

const char* build_git_hash() { return TSLM_GIT_HASH; }

namespace {

std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return b;
}

}  // namespace

std::string build_info_text(int world_size) {
    const long cores = ::sysconf(_SC_NPROCESSORS_ONLN);
    const long pages = ::sysconf(_SC_PHYS_PAGES), page = ::sysconf(_SC_PAGE_SIZE);
    const double ram_mb = (pages > 0 && page > 0) ? double(pages) * double(page) / (1024.0 * 1024.0) : -1.0;
    char b[256];
    std::string s;
    s += "# date_utc: " + utc_now() + "\n";
    s += "# git: " TSLM_GIT_HASH "\n";
    s += std::string("# build: ") + build_description() + "\n";
    s += "# compiler: " TSLM_CXX " " __VERSION__ "\n";
    s += "# cxxflags: " TSLM_CXXFLAGS "\n";
#if defined(TSLM_NVCC)
    s += "# nvcc: " TSLM_NVCC "\n";
    s += "# nvccflags: " TSLM_NVCCFLAGS "\n";
#endif
    s += std::string("# device: ") + dev_name() + "\n";
    s += "# cpu: " + cpu_model_name() + "\n";
    std::snprintf(b, sizeof(b), "# cores_online: %ld\n# ram_mb: %.0f\n# world_size: %d\n", cores, ram_mb,
                  world_size);
    s += b;
    return s;
}

std::string manifest_text(const Config& cfg, const Model* model, uint64_t train_tokens, uint64_t steps_per_epoch,
                          int world_size) {
    char b[256];
    std::string s;
    s += "# tslm run manifest (ROADMAP §9). '#' lines are metadata; the rest is the resolved config,\n";
    s += "# reloadable with --config <this file>.\n";
    s += build_info_text(world_size);
    s += "# " + model_summary(model) + "\n";
    std::snprintf(b, sizeof(b), "# train_tokens: %llu\n# steps_per_epoch: %llu\n",
                  static_cast<unsigned long long>(train_tokens), static_cast<unsigned long long>(steps_per_epoch));
    s += b;
    s += config_to_text(cfg);
    return s;
}

}  // namespace tslm
