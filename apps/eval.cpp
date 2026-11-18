// eval: fixed-prompt completions for manual scoring (ROADMAP P6.2).
//
//   bin/eval --ckpt runs/<run>/ckpt_<step>.bin [--prompts eval/prompts.txt]
//            [--vocab_map data/vocab_map.txt] [--vocab_json data/vocab.json]
//            [--merges_txt data/merges.txt] [--out <file>]
//
// Writes <checkpoint dir>/samples_<step>.txt (or --out) in the stable layout
// of include/tslm/samples.h: for every prompt one greedy and one
// temperature-0.8 completion with a fixed seed. Sampling settings are fixed
// constants so files from different runs and builds compare directly.
// --vocab_map must come from the same prepare_data run as the checkpoint's
// training data (see apps/generate.cpp).
#include <cstdio>
#include <string>
#include <vector>

#include "tslm/backend.h"
#include "tslm/build_config.h"
#include "tslm/checkpoint.h"
#include "tslm/common.h"
#include "tslm/generate.h"
#include "tslm/logging.h"
#include "tslm/model.h"
#include "tslm/samples.h"
#include "tslm/text_codec.h"

using namespace tslm;

int main(int argc, char** argv) {
    dev_init(0);
    std::string ckpt, out, prompts_path = "eval/prompts.txt", vocab_map = "data/vocab_map.txt",
                           vocab_json = "data/vocab.json", merges_txt = "data/merges.txt";
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s' (expected --key value)",
              arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "ckpt") ckpt = val;
        else if (key == "out") out = val;
        else if (key == "prompts") prompts_path = val;
        else if (key == "vocab_map") vocab_map = val;
        else if (key == "vocab_json") vocab_json = val;
        else if (key == "merges_txt") merges_txt = val;
        else CHECK(false, "unknown option '--%s'", key.c_str());
    }
    CHECK(!ckpt.empty(), "--ckpt <checkpoint file> is required");

    std::string err;
    CheckpointInfo info;
    CHECK(checkpoint_read_info(ckpt, &info, &err), "%s", err.c_str());
    Model* model = model_create(info.model, 1, info.model.max_seq_len);
    CHECK(checkpoint_load(ckpt, model, &info, &err), "%s", err.c_str());
    TextCodec codec;
    CHECK(codec.load(vocab_map, vocab_json, merges_txt, &err), "%s", err.c_str());
    CHECK(codec.vocab_size() == size_t(info.model.vocab_size), "%s has %zu entries but the checkpoint has V=%d",
          vocab_map.c_str(), codec.vocab_size(), info.model.vocab_size);
    std::vector<EvalPrompt> prompts;
    SamplesHeader h;
    CHECK(load_prompts(prompts_path, &prompts, &h.prompts_hash, &err), "%s", err.c_str());

    if (out.empty()) {
        const size_t slash = ckpt.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "." : ckpt.substr(0, slash);
        out = dir + "/samples_" + std::to_string(info.step) + ".txt";
    }
    h.checkpoint = ckpt;
    h.step = info.step;
    h.model = model_summary(model);
    if (h.model.compare(0, 7, "model: ") == 0) h.model.erase(0, 7);  // the header line adds its own label
    h.build = build_description();
    h.prompts_path = prompts_path;

    std::vector<EvalCompletion> greedy, sampled;
    for (const EvalPrompt& p : prompts) {
        int unk = 0;
        const std::vector<int32_t> ctx = codec.encode_prompt(p.text, &unk);
        if (unk > 0) log_warn("prompt %d: %d token(s) not in the vocabulary became <unk>", p.id, unk);
        for (int mode = 0; mode < 2; ++mode) {
            GenerateOptions g;
            g.max_new = EVAL_MAX_NEW;
            g.seed = EVAL_SEED;
            g.sample.greedy = mode == 0;
            g.sample.temperature = EVAL_TEMPERATURE;
            const std::vector<int32_t> ids = generate_ids(model, ctx, g);
            EvalCompletion c;
            c.text = codec.decode(ids);
            c.tokens = static_cast<int>(ids.size());
            c.stopped_at_eos = c.tokens < EVAL_MAX_NEW;
            (mode == 0 ? greedy : sampled).push_back(c);
        }
        log_info("prompt %d [%s]: greedy %d tokens, sampled %d tokens", p.id, p.category.c_str(), greedy.back().tokens,
                 sampled.back().tokens);
    }
    const std::string text = format_samples(h, prompts, greedy, sampled);
    std::FILE* f = std::fopen(out.c_str(), "wb");
    CHECK(f, "cannot create %s", out.c_str());
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    CHECK(std::fclose(f) == 0 && ok, "write to %s failed", out.c_str());
    log_info("wrote %s (%zu prompts)", out.c_str(), prompts.size());
    model_free(model);
    return 0;
}
