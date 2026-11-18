// generate: continue a story from a checkpoint (ROADMAP P6.1).
//
//   bin/generate --ckpt runs/<run>/ckpt_latest.bin [--prompt "Once upon a time"]
//                [--vocab_map data/vocab_map.txt] [--vocab_json data/vocab.json]
//                [--merges_txt data/merges.txt] [--max_new 200] [--greedy 1]
//                [--temperature 1.0] [--top_k 0] [--seed 1337] [--allow_unk 0]
//
// The model shape comes from the checkpoint header. --vocab_map MUST be the
// vocab_map.txt written by the same prepare_data run as the checkpoint's
// training data (a different one silently maps ids to the wrong tokens; its
// size is checked, its content cannot be). The prompt is encoded with our
// GPT-2 byte-level BPE, mapped to vocab ids (unknown tokens -> <unk>, with a
// warning), and prefixed with EOS, which in the training data precedes every
// story. Prints the prompt followed by the continuation; generation stops at
// EOS or after --max_new tokens. See include/tslm/generate.h for sampling.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tslm/checkpoint.h"
#include "tslm/common.h"
#include "tslm/generate.h"
#include "tslm/logging.h"
#include "tslm/model.h"
#include "tslm/text_codec.h"

using namespace tslm;

namespace {

long long parse_int_or_die(const std::string& key, const std::string& v) {
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(v.c_str(), &end, 10);
    CHECK(!v.empty() && errno == 0 && *end == '\0', "--%s: expected an integer, got '%s'", key.c_str(), v.c_str());
    return x;
}

double parse_double_or_die(const std::string& key, const std::string& v) {
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    CHECK(!v.empty() && errno == 0 && *end == '\0', "--%s: expected a number, got '%s'", key.c_str(), v.c_str());
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ckpt, prompt, vocab_map = "data/vocab_map.txt", vocab_json = "data/vocab.json",
                              merges_txt = "data/merges.txt";
    GenerateOptions opt;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s' (expected --key value)",
              arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "ckpt") ckpt = val;
        else if (key == "prompt") prompt = val;
        else if (key == "vocab_map") vocab_map = val;
        else if (key == "vocab_json") vocab_json = val;
        else if (key == "merges_txt") merges_txt = val;
        else if (key == "max_new") opt.max_new = static_cast<int>(parse_int_or_die(key, val));
        else if (key == "greedy") opt.sample.greedy = parse_int_or_die(key, val) != 0;
        else if (key == "temperature") opt.sample.temperature = parse_double_or_die(key, val);
        else if (key == "top_k") opt.sample.top_k = static_cast<int>(parse_int_or_die(key, val));
        else if (key == "seed") opt.seed = static_cast<uint64_t>(parse_int_or_die(key, val));
        else if (key == "allow_unk") opt.sample.allow_unk = parse_int_or_die(key, val) != 0;
        else CHECK(false, "unknown option '--%s'", key.c_str());
    }
    CHECK(!ckpt.empty(), "--ckpt <checkpoint file> is required");
    CHECK(opt.max_new >= 0 && opt.max_new <= 100000, "--max_new must be in [0, 100000]");
    CHECK(opt.sample.greedy || opt.sample.temperature > 0.0, "--temperature must be > 0 (or use --greedy 1)");

    std::string err;
    CheckpointInfo info;
    CHECK(checkpoint_read_info(ckpt, &info, &err), "%s", err.c_str());
    Model* model = model_create(info.model, 1, info.model.max_seq_len);
    CHECK(checkpoint_load(ckpt, model, &info, &err), "%s", err.c_str());

    TextCodec codec;
    CHECK(codec.load(vocab_map, vocab_json, merges_txt, &err), "%s", err.c_str());
    CHECK(codec.vocab_size() == size_t(info.model.vocab_size),
          "%s has %zu entries but the checkpoint has vocab_size %d", vocab_map.c_str(), codec.vocab_size(),
          info.model.vocab_size);
    int unk = 0;
    const std::vector<int32_t> context = codec.encode_prompt(prompt, &unk);
    if (unk > 0) log_warn("%d prompt token(s) are not in the vocabulary and became <unk>", unk);

    log_info("# %s | step %llu | %s | max_new %d, seed %llu", model_summary(model).c_str(),
             static_cast<unsigned long long>(info.step),
             opt.sample.greedy ? "greedy"
                               : ("temperature " + std::to_string(opt.sample.temperature) + ", top_k " +
                                  std::to_string(opt.sample.top_k))
                                     .c_str(),
             opt.max_new, static_cast<unsigned long long>(opt.seed));
    const std::vector<int32_t> out = generate_ids(model, context, opt);
    const std::string text = codec.decode(out);
    log_info("%s%s", prompt.c_str(), text.c_str());
    log_info("# %zu new tokens%s", out.size(), int(out.size()) < opt.max_new ? " (stopped at EOS)" : "");
    model_free(model);
    return 0;
}
