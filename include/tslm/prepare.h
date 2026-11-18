// prepare.h: two-pass dataset preparation (ROADMAP §7.3, P1.3).
//
// Input text files hold stories separated by the literal "<|endoftext|>".
// Each story is trimmed of leading/trailing ASCII whitespace (the files put
// a newline on each side of every separator); empty stories are skipped.
//
// Pass 1: tokenize every story with the GPT-Neo BPE, append EOS (GPT id of
//   "<|endoftext|>", 50256), and stream the raw GPT ids as uint16 to a temp
//   file. A uint64 histogram over GPT ids is built from TRAIN only.
//   Truncation applies to train only, and only whole stories are kept:
//   --max_stories N keeps the first N stories; --max_tokens N keeps stories
//   while the running token count (including EOS) stays <= N.
// Vocabulary (size V, default 10,000):
//   id 0 = <unk>, id 1 = <|endoftext|>, ids 2..V-1 = the V-2 GPT tokens with
//   the highest train counts (EOS excluded), ties broken by lower GPT id.
//   Zero-count tokens rank last by the same rule, so V is always filled.
// Pass 2: remap train and val (same map; unmapped -> 0) into
//   <out_dir>/train.bin, val.bin (token_file.h format), plus
//   vocab_map.txt: "new_id<TAB>gpt_id<TAB>escaped token bytes" per line
//                  (<unk> has gpt_id -1; bytes outside printable ASCII,
//                  and '\', are written as \xHH)
//   meta.txt:      key=value counts, coverage (fraction of non-<unk>
//                  tokens) for train and val, vocab hash, output hashes.
// Every output is a pure function of the inputs and options, so reruns are
// byte-identical. Outputs are written to "<name>.part" and renamed when
// complete. Errors (I/O, bad options) abort via CHECK.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace tslm {

struct PrepareOptions {
    std::string train_txt = "data/TinyStories-train.txt";
    std::string val_txt = "data/TinyStories-valid.txt";
    std::string vocab_json = "data/vocab.json";
    std::string merges_txt = "data/merges.txt";
    std::string out_dir = "data";
    uint64_t max_stories = 0;  // 0 = no limit
    uint64_t max_tokens = 0;   // 0 = no limit
    int vocab_size = 10000;
    size_t read_chunk_bytes = size_t(16) << 20;  // input read size (tests use tiny chunks)
    std::function<void(const std::string&)> log;  // progress messages; may be empty
};

struct SplitStats {
    uint64_t stories = 0;
    uint64_t tokens = 0;      // including one EOS per story
    uint64_t unk_tokens = 0;  // tokens mapped to <unk>
    double coverage() const { return tokens ? 1.0 - double(unk_tokens) / double(tokens) : 0.0; }
};

struct PrepareStats {
    SplitStats train, val;
    uint64_t train_distinct_gpt_tokens = 0;
    uint64_t vocab_min_count = 0;  // train count of the least frequent kept token
    uint64_t vocab_hash = 0;       // FNV-1a over the gpt ids in new-id order
    uint64_t train_bin_hash = 0;   // FNV-1a of the whole train.bin file
    uint64_t val_bin_hash = 0;
};

PrepareStats prepare_data(const PrepareOptions& opt);

}  // namespace tslm
