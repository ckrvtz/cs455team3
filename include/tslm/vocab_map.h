// vocab_map.h: read data/vocab_map.txt and decode new token ids (ROADMAP §7.3).
//
// File format (written by prepare_data): one line per new id, in order,
//   new_id <TAB> gpt_id <TAB> escaped token bytes
// Bytes outside printable ASCII, and '\', are written as \xHH. Line 0 is
// "0<TAB>-1<TAB><unk>" (no GPT token). Line 1 is the EOS token.
//
// Decoding concatenates each id's raw bytes. Optional markers replace <unk>
// (id 0) and EOS (id 1), so tools can show story boundaries visibly. A
// sequence that starts mid-story may begin with a partial UTF-8 character;
// the bytes are passed through unchanged. Host-only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tslm {

constexpr int32_t UNK_ID = 0;
constexpr int32_t EOS_ID = 1;

struct VocabMap {
    std::vector<std::string> bytes;  // new id -> raw token bytes ("" for <unk>)
    std::vector<int64_t> gpt_ids;    // new id -> GPT-Neo id (-1 for <unk>)
    size_t size() const { return bytes.size(); }
};

// Strict parse: 3 TAB-separated fields per line, ids 0..V-1 in order,
// valid escapes, line 0 = <unk>. Returns false with *err naming the line.
bool parse_vocab_map(const std::string& text, VocabMap* out, std::string* err);
bool load_vocab_map(const std::string& path, VocabMap* out, std::string* err);

// Decode ids[0..n). eos_marker / unk_marker replace EOS / <unk> when non-null;
// with a null eos_marker, EOS decodes to its raw bytes "<|endoftext|>", and
// with a null unk_marker, <unk> decodes to "<unk>". CHECK-fails on an id
// outside [0, size()).
std::string decode_ids(const VocabMap& vm, const int32_t* ids, size_t n, const char* eos_marker,
                       const char* unk_marker);

}  // namespace tslm
