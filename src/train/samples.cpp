// samples.cpp: see include/tslm/samples.h.
#include "tslm/samples.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tslm/common.h"
#include "tslm/token_file.h"  // fnv1a64

namespace tslm {

namespace {
bool fail(std::string* err, const std::string& m) {
    if (err) *err = m;
    return false;
}
}  // namespace

bool parse_prompts(const std::string& text, const std::string& source, std::vector<EvalPrompt>* out,
                   std::string* err) {
    out->clear();
    size_t pos = 0;
    int line_no = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::string where = source + ":" + std::to_string(line_no) + ": ";
        const size_t t1 = line.find('\t'), t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
        if (t2 == std::string::npos) return fail(err, where + "expected <id> TAB <category> TAB <prompt>");
        EvalPrompt p;
        const std::string id = line.substr(0, t1);
        char* end = nullptr;
        errno = 0;
        const long v = std::strtol(id.c_str(), &end, 10);
        if (id.empty() || *end != '\0' || errno != 0 || v < 1 || v > 1000000)
            return fail(err, where + "bad prompt id '" + id + "'");
        p.id = static_cast<int>(v);
        p.category = line.substr(t1 + 1, t2 - t1 - 1);
        p.text = line.substr(t2 + 1);
        if (p.category != "story" && p.category != "factual" && p.category != "reasoning" &&
            p.category != "consistency")
            return fail(err, where + "unknown category '" + p.category + "'");
        if (p.text.empty() || p.text.find('\t') != std::string::npos)
            return fail(err, where + "prompt text must be non-empty and contain no tabs");
        if (!out->empty() && p.id <= out->back().id)
            return fail(err, where + "prompt ids must be unique and increasing");
        out->push_back(p);
    }
    if (out->empty()) return fail(err, source + ": no prompts");
    return true;
}

bool load_prompts(const std::string& path, std::vector<EvalPrompt>* out, uint64_t* hash, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open " + path + ": " + std::strerror(errno));
    std::string text;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), f)) > 0) text.append(b, n);
    std::fclose(f);
    if (hash) *hash = fnv1a64(FNV1A64_INIT, text.data(), text.size());
    return parse_prompts(text, path, out, err);
}

std::string format_samples(const SamplesHeader& h, const std::vector<EvalPrompt>& prompts,
                           const std::vector<EvalCompletion>& greedy, const std::vector<EvalCompletion>& sampled) {
    CHECK(greedy.size() == prompts.size() && sampled.size() == prompts.size(),
          "format_samples: %zu prompts but %zu/%zu completions", prompts.size(), greedy.size(), sampled.size());
    char b[256];
    std::string s = "# tslm samples v1\n";
    s += "# checkpoint: " + h.checkpoint + " | step " + std::to_string(h.step) + "\n";
    s += "# model: " + h.model + "\n";
    s += "# build: " + h.build + "\n";
    std::snprintf(b, sizeof(b), " | %zu prompts | fnv1a64 %016llx\n", prompts.size(),
                  static_cast<unsigned long long>(h.prompts_hash));
    s += "# prompts: " + h.prompts_path + b;
    std::snprintf(b, sizeof(b), "# settings: max_new %d | greedy | temperature %g seed %llu\n", EVAL_MAX_NEW,
                  EVAL_TEMPERATURE, static_cast<unsigned long long>(EVAL_SEED));
    s += b;
    auto section = [&](const std::string& title, const EvalPrompt& p, const EvalCompletion& c) {
        std::snprintf(b, sizeof(b), " (%d tokens%s) ---\n", c.tokens, c.stopped_at_eos ? ", stopped at EOS" : "");
        s += "--- " + title + b;
        s += p.text + c.text + "\n";
    };
    for (size_t i = 0; i < prompts.size(); ++i) {
        s += "=== prompt " + std::to_string(prompts[i].id) + " [" + prompts[i].category + "] ===\n";
        s += prompts[i].text + "\n";
        section("greedy", prompts[i], greedy[i]);
        std::snprintf(b, sizeof(b), "temperature %g seed %llu", EVAL_TEMPERATURE,
                      static_cast<unsigned long long>(EVAL_SEED));
        section(b, prompts[i], sampled[i]);
    }
    return s;
}

}  // namespace tslm
