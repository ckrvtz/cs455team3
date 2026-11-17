// prepare_data: build train.bin / val.bin / vocab_map.txt / meta.txt (ROADMAP §7.3).
//
//   bin/prepare_data [--train_txt P] [--val_txt P] [--vocab_json P] [--merges_txt P]
//                    [--out_dir D] [--max_stories N] [--max_tokens N] [--vocab_size V]
// Defaults read data/ (from scripts/download_data.sh) and write into data/.
// --max_stories / --max_tokens truncate the TRAIN set only (whole stories).
#include "tslm/common.h"
#include "tslm/prepare.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tslm;

static uint64_t parse_u64_or_die(const std::string& key, const std::string& v) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(v.c_str(), &end, 10);
    CHECK(!v.empty() && v.find_first_not_of("0123456789") == std::string::npos && errno == 0 && *end == '\0',
          "--%s: expected a non-negative integer, got '%s'", key.c_str(), v.c_str());
    return static_cast<uint64_t>(x);
}

int main(int argc, char** argv) {
    PrepareOptions opt;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s' (expected --key value)",
              arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "train_txt") opt.train_txt = val;
        else if (key == "val_txt") opt.val_txt = val;
        else if (key == "vocab_json") opt.vocab_json = val;
        else if (key == "merges_txt") opt.merges_txt = val;
        else if (key == "out_dir") opt.out_dir = val;
        else if (key == "max_stories") opt.max_stories = parse_u64_or_die(key, val);
        else if (key == "max_tokens") opt.max_tokens = parse_u64_or_die(key, val);
        else if (key == "vocab_size") {
            const uint64_t v = parse_u64_or_die(key, val);
            CHECK(v <= 65536, "--vocab_size %llu too large", static_cast<unsigned long long>(v));
            opt.vocab_size = static_cast<int>(v);
        } else {
            CHECK(false, "unknown option '--%s'", key.c_str());
        }
    }
    opt.log = [](const std::string& m) {
        std::fprintf(stderr, "%s\n", m.c_str());
        std::fflush(stderr);
    };

    const PrepareStats s = prepare_data(opt);
    std::printf("train: %llu stories, %llu tokens, coverage %.4f%% (%llu <unk>)\n",
                static_cast<unsigned long long>(s.train.stories), static_cast<unsigned long long>(s.train.tokens),
                100.0 * s.train.coverage(), static_cast<unsigned long long>(s.train.unk_tokens));
    std::printf("val:   %llu stories, %llu tokens, coverage %.4f%% (%llu <unk>)\n",
                static_cast<unsigned long long>(s.val.stories), static_cast<unsigned long long>(s.val.tokens),
                100.0 * s.val.coverage(), static_cast<unsigned long long>(s.val.unk_tokens));
    std::printf("distinct GPT tokens in train: %llu; least frequent kept token count: %llu\n",
                static_cast<unsigned long long>(s.train_distinct_gpt_tokens),
                static_cast<unsigned long long>(s.vocab_min_count));
    std::printf("wrote %s/{train.bin,val.bin,vocab_map.txt,meta.txt}\n", opt.out_dir.c_str());
    return 0;
}
