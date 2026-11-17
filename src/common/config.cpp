// config.cpp: key=value / CLI config parsing, validation, serialization.
// See include/tslm/config.h for the format and precedence rules.
//
// The table returned by fields() is the only list of keys. Parsing,
// serialization, and the unknown-key check all use it, so they cannot drift
// apart. It is a plain vector built on demand (no unordered containers), so
// iteration order is fixed.
#include "tslm/config.h"

#include "tslm/common.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tslm {

namespace {

enum class Kind { Int, U64, Double, String };

struct Field {
    const char* key;
    Kind kind;
    void* ptr;
};

std::vector<Field> fields(Config* c) {
    ModelConfig& m = c->model;
    TrainConfig& t = c->train;
    return {
        // model
        {"vocab_size", Kind::Int, &m.vocab_size},
        {"max_seq_len", Kind::Int, &m.max_seq_len},
        {"n_layer", Kind::Int, &m.n_layer},
        {"n_embd", Kind::Int, &m.n_embd},
        {"n_head", Kind::Int, &m.n_head},
        {"tie_embeddings", Kind::Int, &m.tie_embeddings},
        // train
        {"run_name", Kind::String, &t.run_name},
        {"out_dir", Kind::String, &t.out_dir},
        {"train_data", Kind::String, &t.train_data},
        {"val_data", Kind::String, &t.val_data},
        {"seed", Kind::U64, &t.seed},
        {"batch_size", Kind::Int, &t.batch_size},
        {"seq_len", Kind::Int, &t.seq_len},
        {"max_steps", Kind::Int, &t.max_steps},
        {"warmup_steps", Kind::Int, &t.warmup_steps},
        {"lr", Kind::Double, &t.lr},
        {"min_lr_frac", Kind::Double, &t.min_lr_frac},
        {"weight_decay", Kind::Double, &t.weight_decay},
        {"beta1", Kind::Double, &t.beta1},
        {"beta2", Kind::Double, &t.beta2},
        {"adam_eps", Kind::Double, &t.adam_eps},
        {"grad_clip", Kind::Double, &t.grad_clip},
        {"log_every", Kind::Int, &t.log_every},
        {"eval_every", Kind::Int, &t.eval_every},
        {"val_batches", Kind::Int, &t.val_batches},
        {"ckpt_every", Kind::Int, &t.ckpt_every},
    };
}

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Strict numeric parsing: the whole string must be consumed, no overflow.
bool parse_int(const std::string& s, int* out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || *end != '\0' || v < INT32_MIN || v > INT32_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

bool parse_u64(const std::string& s, uint64_t* out) {
    // strtoull accepts a leading '-' and wraps it; reject anything but digits.
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || *end != '\0') return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

bool parse_double(const std::string& s, double* out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

std::string format_value(const Field& f) {
    char buf[64];
    switch (f.kind) {
        case Kind::Int:
            std::snprintf(buf, sizeof(buf), "%d", *static_cast<const int*>(f.ptr));
            return buf;
        case Kind::U64:
            std::snprintf(buf, sizeof(buf), "%llu",
                          static_cast<unsigned long long>(*static_cast<const uint64_t*>(f.ptr)));
            return buf;
        case Kind::Double:
            // 17 significant digits round-trip any IEEE double exactly.
            std::snprintf(buf, sizeof(buf), "%.17g", *static_cast<const double*>(f.ptr));
            return buf;
        case Kind::String:
            return *static_cast<const std::string*>(f.ptr);
    }
    return "";
}

}  // namespace

bool config_set(Config* cfg, const std::string& key, const std::string& value, std::string* err) {
    for (const Field& f : fields(cfg)) {
        if (key != f.key) continue;
        bool ok = true;
        switch (f.kind) {
            case Kind::Int: ok = parse_int(value, static_cast<int*>(f.ptr)); break;
            case Kind::U64: ok = parse_u64(value, static_cast<uint64_t*>(f.ptr)); break;
            case Kind::Double: ok = parse_double(value, static_cast<double*>(f.ptr)); break;
            case Kind::String:
                if (value.find('\n') != std::string::npos) ok = false;
                else *static_cast<std::string*>(f.ptr) = value;
                break;
        }
        if (!ok) {
            static const char* kind_names[] = {"integer", "unsigned 64-bit integer", "finite number",
                                               "single-line string"};
            return fail(err, "bad value '" + value + "' for key '" + key + "' (expected " +
                                 kind_names[static_cast<int>(f.kind)] + ")");
        }
        return true;
    }
    return fail(err, "unknown config key '" + key + "'");
}

bool config_parse_text(const std::string& text, const std::string& source, Config* cfg,
                       std::string* err) {
    std::vector<std::string> seen;
    size_t pos = 0;
    int line_no = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = trim(text.substr(pos, nl - pos));
        pos = nl + 1;
        ++line_no;
        if (line.empty() || line[0] == '#') continue;

        const std::string where = source + ":" + std::to_string(line_no) + ": ";
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return fail(err, where + "expected key=value, got '" + line + "'");
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty()) return fail(err, where + "empty key");
        for (const std::string& k : seen) {
            if (k == key) return fail(err, where + "duplicate key '" + key + "'");
        }
        seen.push_back(key);
        std::string e;
        if (!config_set(cfg, key, value, &e)) return fail(err, where + e);
    }
    return true;
}

bool config_load_file(const std::string& path, Config* cfg, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open config file '" + path + "'");
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    const bool read_error = std::ferror(f) != 0;
    std::fclose(f);
    if (read_error) return fail(err, "error reading config file '" + path + "'");
    return config_parse_text(text, path, cfg, err);
}

bool config_parse_args(int argc, const char* const* argv, Config* cfg, std::string* err) {
    // Collect pairs first so --config can be applied before overrides
    // regardless of its position on the command line.
    std::vector<std::string> keys, values;
    std::string config_path;
    bool have_config = false;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg.size() < 3 || arg.compare(0, 2, "--") != 0) {
            return fail(err, "unexpected argument '" + arg + "' (expected --key value)");
        }
        const std::string key = arg.substr(2);
        if (key.find('=') != std::string::npos) {
            return fail(err, "use '--key value', not '--key=value' (got '" + arg + "')");
        }
        if (i + 1 >= argc) return fail(err, "missing value for '" + arg + "'");
        const std::string value = argv[i + 1];
        if (key == "config") {
            if (have_config) return fail(err, "--config given more than once");
            have_config = true;
            config_path = value;
            continue;
        }
        for (const std::string& k : keys) {
            if (k == key) return fail(err, "command-line key '" + key + "' given more than once");
        }
        keys.push_back(key);
        values.push_back(value);
    }
    if (have_config && !config_load_file(config_path, cfg, err)) return false;
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string e;
        if (!config_set(cfg, keys[i], values[i], &e)) return fail(err, "command line: " + e);
    }
    return true;
}

bool config_validate(const Config& cfg, std::string* err) {
    const ModelConfig& m = cfg.model;
    const TrainConfig& t = cfg.train;
    auto bad = [&](const std::string& msg) { return fail(err, "invalid config: " + msg); };
    // 65536 = uint16 on-disk token ids (ROADMAP §7.3); ids 0/1 are reserved.
    if (m.vocab_size < 3 || m.vocab_size > 65536) return bad("vocab_size must be in [3, 65536]");
    if (m.max_seq_len < 1) return bad("max_seq_len must be >= 1");
    if (m.n_layer < 1) return bad("n_layer must be >= 1");
    if (m.n_embd < 1) return bad("n_embd must be >= 1");
    if (m.n_head < 1) return bad("n_head must be >= 1");
    if (m.n_embd % m.n_head != 0) {
        return bad("n_embd (" + std::to_string(m.n_embd) + ") must be divisible by n_head (" +
                   std::to_string(m.n_head) + ")");
    }
    if (m.tie_embeddings != 0 && m.tie_embeddings != 1) return bad("tie_embeddings must be 0 or 1");

    if (t.run_name.empty() || t.run_name.find('/') != std::string::npos ||
        t.run_name == "." || t.run_name == "..") {
        return bad("run_name must be a non-empty single path component");
    }
    if (t.out_dir.empty()) return bad("out_dir must be non-empty");
    if (t.train_data.empty() || t.val_data.empty()) return bad("train_data/val_data must be non-empty");
    if (t.batch_size < 1) return bad("batch_size must be >= 1");
    if (t.seq_len < 1 || t.seq_len > m.max_seq_len) {
        return bad("seq_len (" + std::to_string(t.seq_len) + ") must be in [1, max_seq_len=" +
                   std::to_string(m.max_seq_len) + "]");
    }
    if (t.max_steps < 1) return bad("max_steps must be >= 1");
    if (t.warmup_steps < 0) return bad("warmup_steps must be >= 0");
    if (!(t.lr > 0)) return bad("lr must be > 0");
    if (!(t.min_lr_frac >= 0 && t.min_lr_frac <= 1)) return bad("min_lr_frac must be in [0, 1]");
    if (!(t.weight_decay >= 0)) return bad("weight_decay must be >= 0");
    if (!(t.beta1 >= 0 && t.beta1 < 1)) return bad("beta1 must be in [0, 1)");
    if (!(t.beta2 >= 0 && t.beta2 < 1)) return bad("beta2 must be in [0, 1)");
    if (!(t.adam_eps > 0)) return bad("adam_eps must be > 0");
    if (!(t.grad_clip > 0)) return bad("grad_clip must be > 0");
    if (t.log_every < 1) return bad("log_every must be >= 1");
    if (t.eval_every < 1) return bad("eval_every must be >= 1");
    if (t.val_batches < 1) return bad("val_batches must be >= 1");
    if (t.ckpt_every < 0) return bad("ckpt_every must be >= 0 (0 = only at end)");
    return true;
}

std::string config_to_text(const Config& cfg) {
    // fields() needs a mutable Config to bind pointers; it is only read here.
    Config copy = cfg;
    std::string out;
    for (const Field& f : fields(&copy)) {
        out += f.key;
        out += '=';
        out += format_value(f);
        out += '\n';
    }
    return out;
}

bool config_save_file(const std::string& path, const Config& cfg, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return fail(err, "cannot open '" + path + "' for writing");
    const std::string text = config_to_text(cfg);
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) return fail(err, "error writing '" + path + "'");
    return true;
}

Config config_from_args_or_die(int argc, const char* const* argv) {
    Config cfg;
    std::string err;
    CHECK(config_parse_args(argc, argv, &cfg, &err), "%s", err.c_str());
    CHECK(config_validate(cfg, &err), "%s", err.c_str());
    return cfg;
}

}  // namespace tslm
