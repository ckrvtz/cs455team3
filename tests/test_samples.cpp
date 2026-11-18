// test_samples.cpp: fixed prompt set and samples_<step>.txt layout (ROADMAP
// P6.2). DoD: the file layout is stable across versions, so the layout is
// pinned here by an exact expected string; any change fails this test and
// must go through the contract process.
//   - the committed eval/prompts.txt parses: 10 prompts, ids 1..10, 3 story,
//     3 factual, 2 reasoning, 2 consistency
//   - malformed prompt files are rejected with the line number
//   - load_prompts returns the FNV-1a 64 of the file bytes
//   - format_samples produces exactly the documented layout
#include "test.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "tslm/samples.h"
#include "tslm/token_file.h"

using namespace tslm;

TEST(committed_prompts_parse) {
    std::vector<EvalPrompt> p;
    uint64_t hash = 0;
    std::string err;
    ASSERT_TRUE(load_prompts("eval/prompts.txt", &p, &hash, &err));
    ASSERT_EQ(p.size(), size_t(10));
    std::map<std::string, int> cats;
    for (size_t i = 0; i < p.size(); ++i) {
        EXPECT_EQ(p[i].id, int(i + 1));
        EXPECT_TRUE(!p[i].text.empty());
        cats[p[i].category]++;
    }
    EXPECT_EQ(cats["story"], 3);
    EXPECT_EQ(cats["factual"], 3);
    EXPECT_EQ(cats["reasoning"], 2);
    EXPECT_EQ(cats["consistency"], 2);
    // The hash is of the raw file bytes.
    std::FILE* f = std::fopen("eval/prompts.txt", "rb");
    ASSERT_TRUE(f != nullptr);
    std::string bytes;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), f)) > 0) bytes.append(b, n);
    std::fclose(f);
    EXPECT_EQ(hash, fnv1a64(FNV1A64_INIT, bytes.data(), bytes.size()));
}

TEST(malformed_prompts_rejected) {
    std::vector<EvalPrompt> p;
    std::string err;
    EXPECT_TRUE(parse_prompts("# c\n\n1\tstory\tHello\r\n2\tfactual\tA b c\n", "ok", &p, &err));
    EXPECT_EQ(p.size(), size_t(2));
    EXPECT_EQ(p[0].text, std::string("Hello"));  // CR stripped
    const char* bad[] = {
        "1\tstory\n",                       // missing field
        "x\tstory\tHi\n",                   // bad id
        "1\tpoem\tHi\n",                    // unknown category
        "1\tstory\t\n",                     // empty text
        "2\tstory\tA\n1\tstory\tB\n",       // ids not increasing
        "1\tstory\tA\n1\tstory\tB\n",       // duplicate id
        "1\tstory\tA\tB\n",                 // tab in text
        "# only comments\n",                // no prompts
    };
    for (const char* t : bad) {
        EXPECT_TRUE(!parse_prompts(t, "bad", &p, &err));
        EXPECT_TRUE(err.find("bad") == 0);
    }
    EXPECT_TRUE(parse_prompts("1\tstory\tA\n2\tstory\tB\n", "f", &p, &err));
    EXPECT_TRUE(!parse_prompts("1\tstory\tA\n2\tstory\tB\n2\tstory\tC\n", "f", &p, &err));
    EXPECT_TRUE(err.find("f:3:") == 0);  // line number reported
}

TEST(samples_layout_is_exact) {
    SamplesHeader h;
    h.checkpoint = "runs/x/ckpt_7.bin";
    h.step = 7;
    h.model = "L=1 C=32";
    h.build = "backend=cpu comm=serial real=float mode=release";
    h.prompts_path = "eval/prompts.txt";
    h.prompts_hash = 0x0123456789abcdefull;
    const std::vector<EvalPrompt> p = {{1, "story", "Once"}, {3, "reasoning", "Two"}};
    const std::vector<EvalCompletion> g = {{" upon a time", 3, true}, {" and\ntwo", 120, false}};
    const std::vector<EvalCompletion> s = {{" there", 1, true}, {"", 0, true}};
    const std::string want =
        "# tslm samples v1\n"
        "# checkpoint: runs/x/ckpt_7.bin | step 7\n"
        "# model: L=1 C=32\n"
        "# build: backend=cpu comm=serial real=float mode=release\n"
        "# prompts: eval/prompts.txt | 2 prompts | fnv1a64 0123456789abcdef\n"
        "# settings: max_new 120 | greedy | temperature 0.8 seed 1337\n"
        "=== prompt 1 [story] ===\n"
        "Once\n"
        "--- greedy (3 tokens, stopped at EOS) ---\n"
        "Once upon a time\n"
        "--- temperature 0.8 seed 1337 (1 tokens, stopped at EOS) ---\n"
        "Once there\n"
        "=== prompt 3 [reasoning] ===\n"
        "Two\n"
        "--- greedy (120 tokens) ---\n"
        "Two and\ntwo\n"
        "--- temperature 0.8 seed 1337 (0 tokens, stopped at EOS) ---\n"
        "Two\n";
    EXPECT_EQ(format_samples(h, p, g, s), want);
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { format_samples(h, p, g, {}); }, &out));
}

TEST_MAIN()
