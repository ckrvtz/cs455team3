// tokenizer.h: GPT-2 / GPT-Neo byte-level BPE, reimplemented (ROADMAP §7.2).
//
// Data files (read as data; the algorithm is ours):
//   vocab.json   flat JSON object { "<token>": id, ... } (GPT-Neo: 50,257 ids)
//   merges.txt   "#version" header line, then one merge "A B" per line in
//                priority order (line k = rank k; lower rank merges first)
// Tokens in both files are written in GPT-2's byte->unicode alphabet: every
// byte maps to one printable codepoint (printable Latin-1 bytes map to
// themselves, the other 68 bytes to U+0100..U+0143 in byte order).
//
// encode(text):
//  1. Pre-tokenize with GPT-2's pattern, implemented as a hand-written
//     scanner (no std::regex). At each position the first matching rule wins:
//        's 't 're 've 'm 'll 'd      (case-sensitive contractions)
//        ' '? letters+                (\p{L})
//        ' '? numbers+                (\p{N})
//        ' '? other+                  (not whitespace/letter/number)
//        whitespace+ not followed by non-whitespace  (i.e. a run of
//                                     whitespace before a word gives up its
//                                     last character to that word)
//        whitespace+
//     Only a literal U+0020 can prefix a word. Character classes: ASCII by
//     definition. A non-ASCII codepoint is a letter unless it is in an
//     explicit table of Unicode whitespace, number and non-letter ranges
//     (punctuation, symbols, emoji, combining marks, format characters).
//     The table is checked against HF-generated golden cases. Invalid UTF-8
//     bytes are treated as single "other" characters (bytes are still
//     encoded losslessly).
//  2. Each pre-token's raw bytes go through BPE: start from single-byte
//     tokens, repeatedly merge every non-overlapping left-to-right occurrence
//     of the adjacent pair with the lowest merge rank, until no adjacent pair
//     has a rank. Results are cached per distinct pre-token.
// Special tokens are NOT recognized: "<|endoftext|>" in text is encoded as
// ordinary text. prepare_data inserts the EOS id itself (§7.3).
//
// decode(ids) concatenates each token's raw bytes, so
// decode(encode(s)) == s for every byte string s.
//
// Host-only. encode() mutates the word cache, so one Tokenizer must not be
// used from several threads at once.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tslm {

class Tokenizer {
public:
    // Load from files. Returns false and sets *err on any I/O or format error.
    bool load(const std::string& vocab_json_path, const std::string& merges_txt_path,
              std::string* err);
    // Same, from in-memory file contents (used by tests).
    bool load_from_memory(const std::string& vocab_json, const std::string& merges_txt,
                          std::string* err);

    std::vector<uint32_t> encode(std::string_view text);
    std::string decode(const std::vector<uint32_t>& ids) const;

    // One past the largest id in vocab.json (50,257 for GPT-Neo).
    uint32_t vocab_size() const { return static_cast<uint32_t>(id_bytes_.size()); }
    size_t num_merges() const { return merge_count_; }
    // True if `id` is present in vocab.json.
    bool has_id(uint32_t id) const { return id < id_present_.size() && id_present_[id]; }
    // Raw bytes of a token (CHECK-fails on an unknown id).
    const std::string& token_bytes(uint32_t id) const;
    // Id of a token given by its raw bytes, or -1 if absent.
    int64_t find_token(std::string_view raw_bytes) const;

private:
    struct Merge {
        uint32_t rank;
        uint32_t result;
    };
    void bpe(std::string_view word, std::vector<uint32_t>* out);

    std::vector<std::string> id_bytes_;  // id -> raw bytes
    std::vector<bool> id_present_;
    std::unordered_map<std::string, uint32_t> bytes_to_id_;
    uint32_t byte_id_[256] = {};
    std::unordered_map<uint64_t, Merge> merges_;  // (left id << 32 | right id) -> merge
    size_t merge_count_ = 0;
    // Word cache: lookups only; never iterated, so its order cannot affect output.
    std::unordered_map<std::string, std::vector<uint32_t>> cache_;
};

// GPT-2 pre-tokenization (step 1 above): splits `text` into pieces whose
// concatenation is exactly `text`. Exposed for unit tests.
std::vector<std::string_view> gpt2_pretokenize(std::string_view text);

// Parse a flat JSON object of string keys to non-negative integer values
// (the vocab.json format). Supports all JSON string escapes, including
// \uXXXX with surrogate pairs. Keys are returned as UTF-8. Exposed for unit tests.
bool parse_json_string_int_object(const std::string& text,
                                  std::vector<std::pair<std::string, int64_t>>* out,
                                  std::string* err);

}  // namespace tslm
