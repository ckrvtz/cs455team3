// text_codec.cpp: see include/tslm/text_codec.h.
#include "tslm/text_codec.h"

namespace tslm {

bool TextCodec::load(const std::string& vocab_map_path, const std::string& vocab_json_path,
                     const std::string& merges_txt_path, std::string* err) {
    if (!load_vocab_map(vocab_map_path, &vm_, err)) return false;
    if (!tok_.load(vocab_json_path, merges_txt_path, err)) return false;
    to_vocab_.assign(tok_.vocab_size(), UNK_ID);
    for (size_t i = 0; i < vm_.size(); ++i)
        if (vm_.gpt_ids[i] >= 0 && size_t(vm_.gpt_ids[i]) < to_vocab_.size())
            to_vocab_[size_t(vm_.gpt_ids[i])] = static_cast<int32_t>(i);
    return true;
}

std::vector<int32_t> TextCodec::encode_prompt(const std::string& text, int* unk) {
    std::vector<int32_t> ids = {EOS_ID};
    int n_unk = 0;
    for (uint32_t g : tok_.encode(text)) {
        const int32_t id = g < to_vocab_.size() ? to_vocab_[g] : UNK_ID;
        n_unk += id == UNK_ID;
        ids.push_back(id);
    }
    if (unk) *unk = n_unk;
    return ids;
}

std::string TextCodec::decode(const std::vector<int32_t>& ids) const {
    return decode_ids(vm_, ids.data(), ids.size(), "", "<unk>");
}

}  // namespace tslm
