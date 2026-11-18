// logging.h: console messages and the per-run log.csv writer (ROADMAP §9,
// CLAUDE.md §5 "Logging goes through logging.h"). Host-only.
//
// log_info / log_warn: printf-style. A newline is appended, and the stream
// is flushed so progress shows up promptly under batch schedulers. log_info
// writes to stdout; log_warn writes "warning: ..." to stderr. Callers decide
// which rank logs (the training loop logs from rank 0 only).
//
// log.csv (FROZEN, ROADMAP §9): the header line is exactly LOG_CSV_HEADER,
// and each row has the same 14 columns:
//   step, tokens_seen, train_loss, val_loss, val_ppl, lr, grad_norm,
//   tok_per_s, step_ms, data_ms, fwd_ms, bwd_ms, comm_ms, opt_ms
// A NaN val_loss / val_ppl is written as an empty field (not evaluated at
// that step). Losses, lr and grad_norm use %.9g, enough to distinguish any
// two floats, so builds can be compared row by row. Timings use %.3f ms and
// tok_per_s uses %.1f. Every row is flushed as soon as it is written, so a
// run that aborts still leaves every row logged before the abort.
#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace tslm {

#if defined(__GNUC__)
void log_info(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_warn(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#else
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
#endif

extern const char* const LOG_CSV_HEADER;

struct LogRow {
    int step = 0;
    uint64_t tokens_seen = 0;
    double train_loss = 0.0;
    double val_loss = std::numeric_limits<double>::quiet_NaN();  // NaN = not evaluated -> empty field
    double val_ppl = std::numeric_limits<double>::quiet_NaN();
    double lr = 0.0;
    double grad_norm = 0.0;
    double tok_per_s = 0.0;
    double step_ms = 0.0, data_ms = 0.0, fwd_ms = 0.0, bwd_ms = 0.0, comm_ms = 0.0, opt_ms = 0.0;
};

// Format one row (no trailing newline).
std::string format_log_row(const LogRow& r);

class CsvLog {
public:
    CsvLog() = default;
    ~CsvLog();
    CsvLog(const CsvLog&) = delete;
    CsvLog& operator=(const CsvLog&) = delete;

    // Create or truncate `path` and write the header. Returns false with *err
    // on failure.
    bool open(const std::string& path, std::string* err);
    // Resume an existing log: keep the header and every row with
    // step <= keep_through_step, drop later rows (they belong to steps that
    // will be rerun), then append. If `path` does not exist, behaves like
    // open(). Fails if the existing header is not LOG_CSV_HEADER.
    bool open_resume(const std::string& path, int keep_through_step, std::string* err);
    // Append one row and flush; CHECKs the write.
    void write(const LogRow& r);
    void close();

private:
    std::FILE* f_ = nullptr;
    std::string path_;
};

}  // namespace tslm
