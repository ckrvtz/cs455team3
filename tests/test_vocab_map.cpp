// test_vocab_map.cpp: vocab_map.txt parsing and id decoding (ROADMAP P1.5).
#include "test.h"

#include "tslm/common.h"
#include "tslm/vocab_map.h"

#include <string>
#include <vector>

using namespace tslm;

namespace {
const std::string GOOD =
    "0\t-1\t<unk>\n"
    "1\t50256\t<|endoftext|>\n"
    "2\t13\t.\n"
    "3\t290\t and\n"
    "4\t198\t\\x0A\n"                // newline
    "5\t447\t\\xE2\\x80\n"           // partial UTF-8 (first 2 bytes of a curly quote)
    "6\t247\t\\x99\n"
    "7\t59\t\\x5C\n";               // backslash
}  // namespace

TEST(parse_and_decode) {
    VocabMap vm;
    std::string err;
    ASSERT_TRUE(parse_vocab_map(GOOD, &vm, &err));
    ASSERT_EQ(vm.size(), size_t{8});
    EXPECT_EQ(vm.gpt_ids[0], int64_t{-1});
    EXPECT_EQ(vm.gpt_ids[1], int64_t{50256});
    EXPECT_EQ(vm.bytes[3], std::string(" and"));
    EXPECT_EQ(vm.bytes[4], std::string("\n"));
    EXPECT_EQ(vm.bytes[7], std::string("\\"));

    const int32_t ids[] = {3, 2, 5, 6, 1, 0, 4, 7};
    // Tokens 5+6 rebuild U+2019 from split bytes.
    EXPECT_EQ(decode_ids(vm, ids, 8, "[EOS]", "?"), std::string(" and.\xE2\x80\x99[EOS]?\n\\"));
    EXPECT_EQ(decode_ids(vm, ids, 8, nullptr, nullptr),
              std::string(" and.\xE2\x80\x99<|endoftext|><unk>\n\\"));
    EXPECT_EQ(decode_ids(vm, ids, 0, nullptr, nullptr), std::string());
}

TEST(parse_errors) {
    VocabMap vm;
    std::string err;
    const std::vector<std::string> bad = {
        "",                                          // too small
        "0\t-1\t<unk>\n",                            // no EOS line
        "0\t-1\t<unk>\n2\t50256\tx\n",               // non-contiguous ids
        "0\t-1\tunk\n1\t50256\tx\n",                 // id 0 not <unk>
        "0\t5\t<unk>\n1\t50256\tx\n",                // id 0 gpt != -1
        "0\t-1\t<unk>\n1\t-3\tx\n",                  // negative gpt for real token
        "0\t-1\t<unk>\n1\t50256\n",                  // 2 fields
        "0\t-1\t<unk>\n1\t50256\tx\ty\n",            // 4 fields
        "0\t-1\t<unk>\n1\tabc\tx\n",                 // non-integer
        "0\t-1\t<unk>\n1\t50256\t\\x4\n",            // truncated escape
        "0\t-1\t<unk>\n1\t50256\t\\n\n",             // unsupported escape
        "0\t-1\t<unk>\n1\t50256\t\\xZZ\n",           // bad hex
    };
    for (const std::string& b : bad) {
        err.clear();
        if (parse_vocab_map(b, &vm, &err) || err.empty())
            test::report(__FILE__, __LINE__, "accepted bad vocab_map: " + b);
    }
    EXPECT_TRUE(!load_vocab_map("tests/golden/does_not_exist.txt", &vm, &err));
    EXPECT_TRUE(err.find("cannot open") != std::string::npos);
}

TEST(decode_out_of_range_aborts) {
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            VocabMap vm;
            std::string e;
            CHECK(parse_vocab_map(GOOD, &vm, &e), "%s", e.c_str());
            const int32_t ids[] = {8};
            decode_ids(vm, ids, 1, nullptr, nullptr);
        },
        &out));
    EXPECT_TRUE(out.find("outside vocab") != std::string::npos);
}

// Real data/vocab_map.txt from prepare_data, when present.
TEST(real_vocab_map_if_present) {
    VocabMap vm;
    std::string err;
    if (!load_vocab_map("data/vocab_map.txt", &vm, &err)) {
        std::printf("    data/vocab_map.txt not loadable (%s): real-data check not done\n", err.c_str());
        return;
    }
    EXPECT_EQ(vm.size(), size_t{10000});
    EXPECT_EQ(vm.bytes[1], std::string("<|endoftext|>"));
    EXPECT_EQ(vm.gpt_ids[1], int64_t{50256});
}

TEST_MAIN()
