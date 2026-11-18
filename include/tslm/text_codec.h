// text_codec.h: text <-> vocabulary ids for generation and evaluation
// (ROADMAP P6.1, P6.2). Host-only.
//
// encode: GPT-2 byte-level BPE (tokenizer.h) -> GPT-Neo ids -> vocab ids via
// the vocab map (a GPT id outside the 10k vocabulary becomes <unk>, counted
// in *unk). The result is prefixed with EOS: in the training stream every
// story follows an EOS, so a prompt starts where a story starts.
// decode: vocab ids -> text (EOS prints as nothing, <unk> as "<unk>").
//
// The vocab map must come from the same prepare_data run as the model's
// training data; only its size can be checked (against the model's V).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tslm/tokenizer.h"
#include "tslm/vocab_map.h"

namespace tslm {

class TextCodec {
public:
    // Load the three files; returns false with *err on any problem.
    bool load(const std::string& vocab_map_path, const std::string& vocab_json_path,
              const std::string& merges_txt_path, std::string* err);
    size_t vocab_size() const { return vm_.size(); }
    std::vector<int32_t> encode_prompt(const std::string& text, int* unk);
    std::string decode(const std::vector<int32_t>& ids) const;

private:
    VocabMap vm_;
    Tokenizer tok_;
    std::vector<int32_t> to_vocab_;  // GPT-Neo id -> vocab id
};

}  // namespace tslm
