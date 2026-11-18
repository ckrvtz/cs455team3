// inspect_data: print N random decoded training sequences (ROADMAP P1.5).
//
//   bin/inspect_data [--data data/train.bin] [--vocab_map data/vocab_map.txt]
//                    [--n 5] [--T 256] [--seed 1337]
// Sequences are the first N chunks of epoch 0's order for --seed, drawn
// through the real DataLoader (B=1), so this also exercises the loader on
// real data. Each shows its T+1 tokens (x plus the final target). Story
// boundaries print as "[EOS]" followed by a newline; <unk> prints as "<unk>".
#include "tslm/common.h"
#include "tslm/dataset.h"
#include "tslm/vocab_map.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
    std::string data = "data/train.bin", vocab_path = "data/vocab_map.txt";
    uint64_t n = 5, T = 256, seed = 1337;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s' (expected --key value)",
              arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "data") data = val;
        else if (key == "vocab_map") vocab_path = val;
        else if (key == "n") n = parse_u64_or_die(key, val);
        else if (key == "T") T = parse_u64_or_die(key, val);
        else if (key == "seed") seed = parse_u64_or_die(key, val);
        else CHECK(false, "unknown option '--%s'", key.c_str());
    }
    CHECK(T >= 1 && T <= 1000000, "--T must be in [1, 1000000]");

    VocabMap vm;
    std::string err;
    CHECK(load_vocab_map(vocab_path, &vm, &err), "%s", err.c_str());
    DataLoader loader;
    CHECK(loader.open(data, 1, static_cast<int>(T), seed, 0, 1, &err), "%s", err.c_str());
    CHECK(vm.size() == loader.vocab_size(), "vocab_map has %zu entries but %s says vocab_size %u",
          vm.size(), data.c_str(), loader.vocab_size());
    CHECK(n <= loader.steps_per_epoch(), "--n %llu exceeds the %llu chunks available",
          static_cast<unsigned long long>(n), static_cast<unsigned long long>(loader.steps_per_epoch()));

    std::printf("%s: %llu tokens, %llu chunks of T=%llu, vocab %u, seed %llu\n", data.c_str(),
                static_cast<unsigned long long>(loader.num_tokens()),
                static_cast<unsigned long long>(loader.num_chunks()), static_cast<unsigned long long>(T),
                loader.vocab_size(), static_cast<unsigned long long>(seed));
    std::vector<int32_t> x(T), y(T), seq(T + 1);
    for (uint64_t k = 0; k < n; ++k) {
        loader.next_batch(x.data(), y.data());
        for (uint64_t t = 0; t < T; ++t) seq[t] = x[t];
        seq[T] = y[T - 1];
        int eos = 0, unk = 0;
        for (int32_t id : seq) {
            eos += id == EOS_ID;
            unk += id == UNK_ID;
        }
        std::printf("\n===== sequence %llu/%llu: %llu tokens, %d EOS, %d <unk> =====\n",
                    static_cast<unsigned long long>(k + 1), static_cast<unsigned long long>(n),
                    static_cast<unsigned long long>(T + 1), eos, unk);
        std::printf("%s\n", decode_ids(vm, seq.data(), seq.size(), "[EOS]\n", "<unk>").c_str());
    }
    return 0;
}
