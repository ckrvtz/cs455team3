// config.h: model and training configuration (ROADMAP §2.1, §2.3, §9).
//
// Sources, applied in this order (later wins):
//   1. built-in defaults below (= `base` model, §2.3 training defaults)
//   2. a `key=value` file given with `--config <path>` (e.g. configs/small.cfg)
//   3. `--key value` command-line overrides
//
// File format: one `key=value` per line; whitespace around key and value is
// trimmed; blank lines and lines whose first non-space character is `#` are
// ignored. There are no inline comments, so a value may contain `#`.
// Unknown keys, duplicate keys in one source, and malformed values are errors.
//
// config_to_text() writes every key in a fixed order. Floating-point values
// use %.17g, so parsing the output reproduces the Config bit-exactly. Each
// run's manifest.txt embeds this text, so a run's config can be reloaded
// with `--config runs/<name>/manifest.txt`.
//
// Host-only. ModelConfig holds only plain integers so model.h and checkpoint
// headers can use it from nvcc-compiled code.
#pragma once

#include <cstdint>
#include <string>

namespace tslm {

struct ModelConfig {
    int vocab_size = 10000;   // V (token ids are stored as uint16 on disk)
    int max_seq_len = 256;    // max T; sets wpe rows [max_seq_len, C]
    int n_layer = 4;          // L
    int n_embd = 256;         // C
    int n_head = 8;           // NH; head_dim = C / NH must be an integer
    int tie_embeddings = 1;   // 1: output head shares wte; 0: separate [V,C] head
};

struct TrainConfig {
    std::string run_name = "base";          // outputs go to <out_dir>/<run_name>/
    std::string out_dir = "runs";
    std::string train_data = "data/train.bin";
    std::string val_data = "data/val.bin";
    uint64_t seed = 1337;                   // init, data order, sampling (§4.6)
    int batch_size = 32;                    // B: micro-batch per rank
    int seq_len = 256;                      // T used for training; <= max_seq_len
    int max_steps = 5000;
    int warmup_steps = 200;                 // linear warmup, then cosine decay
    double lr = 6e-4;                       // peak learning rate
    double min_lr_frac = 0.1;               // cosine floor as a fraction of lr
    double weight_decay = 0.1;              // 2-D weight matrices only (§2.3)
    double beta1 = 0.9;
    double beta2 = 0.95;
    double adam_eps = 1e-8;
    double grad_clip = 1.0;                 // global L2-norm clip
    int log_every = 10;                     // log.csv row cadence (steps)
    int eval_every = 250;                   // validation cadence (steps)
    int val_batches = 20;                   // fixed val batches per evaluation
    int ckpt_every = 1000;                  // checkpoint cadence; 0 = only at end
    std::string resume;                     // checkpoint to resume from; empty = fresh run
};

struct Config {
    ModelConfig model;
    TrainConfig train;
};

// All functions below return false and set *err (if non-null) to a message
// naming the source and key on failure. On failure *cfg may be partly updated.

// Set a single key from its string value.
bool config_set(Config* cfg, const std::string& key, const std::string& value, std::string* err);

// Apply `key=value` text. `source` names it in error messages (e.g. a path).
bool config_parse_text(const std::string& text, const std::string& source, Config* cfg,
                       std::string* err);

// Read a file and apply it with config_parse_text.
bool config_load_file(const std::string& path, Config* cfg, std::string* err);

// Apply argv[1..argc): an optional `--config <path>` (applied first, wherever it
// appears, at most once) and `--key value` overrides (each key at most once).
// Any other argument form is an error.
bool config_parse_args(int argc, const char* const* argv, Config* cfg, std::string* err);

// Range and consistency checks (C % NH == 0, seq_len <= max_seq_len, ...).
bool config_validate(const Config& cfg, std::string* err);

// All keys as `key=value` lines in a fixed order (re-parseable, bit-exact).
std::string config_to_text(const Config& cfg);

// Write config_to_text() to `path`, overwriting it.
bool config_save_file(const std::string& path, const Config& cfg, std::string* err);

// Typical app entry point: defaults + args + validate, aborting via CHECK with
// the error message on any failure.
Config config_from_args_or_die(int argc, const char* const* argv);

}  // namespace tslm
