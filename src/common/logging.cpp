// logging.cpp: see include/tslm/logging.h.
#include "tslm/logging.h"

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

#include "tslm/common.h"

namespace tslm {

const char* const LOG_CSV_HEADER =
    "step,tokens_seen,train_loss,val_loss,val_ppl,lr,grad_norm,tok_per_s,step_ms,data_ms,fwd_ms,bwd_ms,"
    "comm_ms,opt_ms";

void log_info(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void log_warn(const char* fmt, ...) {
    std::fputs("warning: ", stderr);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

namespace {
std::string fmt_opt(double v) {
    if (std::isnan(v)) return std::string();
    char b[64];
    std::snprintf(b, sizeof(b), "%.9g", v);
    return b;
}
}  // namespace

std::string format_log_row(const LogRow& r) {
    char b[512];
    const std::string vl = fmt_opt(r.val_loss), vp = fmt_opt(r.val_ppl);
    std::snprintf(b, sizeof(b), "%d,%llu,%.9g,%s,%s,%.9g,%.9g,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", r.step,
                  static_cast<unsigned long long>(r.tokens_seen), r.train_loss, vl.c_str(), vp.c_str(), r.lr,
                  r.grad_norm, r.tok_per_s, r.step_ms, r.data_ms, r.fwd_ms, r.bwd_ms, r.comm_ms, r.opt_ms);
    return b;
}

CsvLog::~CsvLog() { close(); }

bool CsvLog::open(const std::string& path, std::string* err) {
    close();
    f_ = std::fopen(path.c_str(), "w");
    if (!f_) {
        if (err) *err = "cannot create " + path + ": " + std::strerror(errno);
        return false;
    }
    path_ = path;
    if (std::fprintf(f_, "%s\n", LOG_CSV_HEADER) < 0 || std::fflush(f_) != 0) {
        if (err) *err = "cannot write " + path;
        close();
        return false;
    }
    return true;
}

bool CsvLog::open_resume(const std::string& path, int keep_through_step, std::string* err) {
    close();
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in) return open(path, err);
    std::string text;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), in)) > 0) text.append(b, n);
    std::fclose(in);
    std::string kept;
    size_t pos = 0;
    bool first = true;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        if (first) {
            if (line != LOG_CSV_HEADER) {
                if (err) *err = path + ": unexpected header; not a tslm log.csv";
                return false;
            }
            first = false;
            continue;
        }
        if (line.empty()) continue;
        if (std::atoi(line.c_str()) <= keep_through_step) kept += line + "\n";
    }
    if (!open(path, err)) return false;
    if (!kept.empty() && (std::fputs(kept.c_str(), f_) < 0 || std::fflush(f_) != 0)) {
        if (err) *err = "cannot write " + path;
        close();
        return false;
    }
    return true;
}

void CsvLog::write(const LogRow& r) {
    CHECK(f_, "CsvLog::write: log is not open");
    const std::string line = format_log_row(r);
    CHECK(std::fprintf(f_, "%s\n", line.c_str()) >= 0 && std::fflush(f_) == 0, "write to %s failed",
          path_.c_str());
}

void CsvLog::close() {
    if (f_) {
        CHECK(std::fclose(f_) == 0, "closing %s failed", path_.c_str());
        f_ = nullptr;
    }
}

}  // namespace tslm
