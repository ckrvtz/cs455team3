// samples.h: fixed prompt set and the samples_<step>.txt layout (ROADMAP
// P6.2). Host-only. The LAYOUT IS STABLE across builds and versions so that
// files from serial, CUDA and MPI runs compare line by line; change it only
// through the contract process (CLAUDE.md §2).
//
// Prompt file (eval/prompts.txt): UTF-8 lines; blank lines and lines starting
// with '#' are ignored; every other line is
//   <id> TAB <category> TAB <prompt text>
// with id a positive integer (unique, increasing), category one of story,
// factual, reasoning, consistency, and non-empty text without tabs.
//
// Fixed settings (constants, not options): each prompt gets one greedy
// completion and one sampled completion at temperature EVAL_TEMPERATURE
// (no top-k) with Rng seed EVAL_SEED, each at most EVAL_MAX_NEW tokens,
// stopping at EOS.
//
// samples_<step>.txt layout (every line ends in '\n'):
//   # tslm samples v1
//   # checkpoint: <path> | step <step>
//   # model: <model_summary>
//   # build: <build_description>
//   # prompts: <path> | <n> prompts | fnv1a64 <16 hex digits of the file bytes>
//   # settings: max_new <N> | greedy | temperature <T> seed <S>
//   (then for each prompt, in file order:)
//   === prompt <id> [<category>] ===
//   <prompt text>
//   --- greedy (<k> tokens<, stopped at EOS>) ---
//   <prompt text><continuation>        (may span several lines)
//   --- temperature <T> seed <S> (<k> tokens<, stopped at EOS>) ---
//   <prompt text><continuation>
// There is no date or timing in the file, so the same binary, checkpoint and
// prompts reproduce it byte for byte.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tslm {

constexpr int EVAL_MAX_NEW = 120;
constexpr double EVAL_TEMPERATURE = 0.8;
constexpr uint64_t EVAL_SEED = 1337;

struct EvalPrompt {
    int id = 0;
    std::string category;
    std::string text;
};

struct EvalCompletion {
    std::string text;      // decoded continuation (without the prompt)
    int tokens = 0;        // generated tokens
    bool stopped_at_eos = false;
};

struct SamplesHeader {
    std::string checkpoint;
    uint64_t step = 0;
    std::string model;
    std::string build;
    std::string prompts_path;
    uint64_t prompts_hash = 0;
};

bool parse_prompts(const std::string& text, const std::string& source, std::vector<EvalPrompt>* out,
                   std::string* err);
// Reads the file, parses it, and returns its FNV-1a 64 hash in *hash.
bool load_prompts(const std::string& path, std::vector<EvalPrompt>* out, uint64_t* hash, std::string* err);

// greedy[i] and sampled[i] belong to prompts[i].
std::string format_samples(const SamplesHeader& h, const std::vector<EvalPrompt>& prompts,
                           const std::vector<EvalCompletion>& greedy, const std::vector<EvalCompletion>& sampled);

}  // namespace tslm
