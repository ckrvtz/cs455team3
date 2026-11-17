// test_tokenizer.cpp: GPT-2/GPT-Neo byte-level BPE (ROADMAP P1.2, §7.2).
//
// Golden cases (tests/golden/tokenizer_cases.txt, generated offline from the
// HF tokenizer by dev_only/gen_tokenizer_golden.py) must match exactly.
// They always run against the committed subset fixture (exact for these
// inputs; see the generator's docstring). When the full GPT-Neo files from
// scripts/download_data.sh are present in data/, the cases also run against
// them. Without network or data/, the subset run still covers every case.
#include "test.h"

#include "tslm/rng.h"
#include "tslm/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tslm;

namespace {

const char* CASES = "tests/golden/tokenizer_cases.txt";
const char* SUBSET_VOCAB = "tests/golden/tokenizer_vocab_subset.json";
const char* SUBSET_MERGES = "tests/golden/tokenizer_merges_subset.txt";
const char* FULL_VOCAB = "data/vocab.json";
const char* FULL_MERGES = "data/merges.txt";

struct Case {
    int line;
    std::string text;
    std::vector<uint32_t> ids;
    std::vector<size_t> piece_lens;  // HF pre-token byte lengths
};

std::vector<size_t> parse_numbers(const std::string& s) {
    std::vector<size_t> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(' ', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) v.push_back(static_cast<size_t>(std::stoull(s.substr(p, q - p))));
        p = q + 1;
    }
    return v;
}

bool file_exists(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}

std::string read_all(const char* path) {
    FILE* f = std::fopen(path, "rb");
    CHECK(f != nullptr, "cannot open %s", path);
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

std::string unescape(const std::string& s, int line) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\') {
            out.push_back(s[i]);
            continue;
        }
        CHECK(i + 1 < s.size(), "%s:%d: dangling backslash", CASES, line);
        const char e = s[++i];
        if (e == '\\') out.push_back('\\');
        else if (e == 'n') out.push_back('\n');
        else if (e == 't') out.push_back('\t');
        else if (e == 'r') out.push_back('\r');
        else if (e == 'x') {
            CHECK(i + 2 < s.size() && hexval(s[i + 1]) >= 0 && hexval(s[i + 2]) >= 0,
                  "%s:%d: bad \\x escape", CASES, line);
            out.push_back(static_cast<char>(hexval(s[i + 1]) * 16 + hexval(s[i + 2])));
            i += 2;
        } else {
            CHECK(false, "%s:%d: unknown escape \\%c", CASES, line, e);
        }
    }
    return out;
}

std::vector<Case> load_cases() {
    const std::string all = read_all(CASES);
    std::vector<Case> cases;
    size_t pos = 0;
    int line = 0;
    while (pos < all.size()) {
        size_t nl = all.find('\n', pos);
        if (nl == std::string::npos) nl = all.size();
        const std::string row = all.substr(pos, nl - pos);
        pos = nl + 1;
        ++line;
        if (!row.empty() && row[0] == '#') continue;
        const size_t tab1 = row.find('\t');
        const size_t tab2 = tab1 == std::string::npos ? tab1 : row.find('\t', tab1 + 1);
        CHECK(tab2 != std::string::npos, "%s:%d: expected 3 TAB-separated fields", CASES, line);
        Case c;
        c.line = line;
        c.text = unescape(row.substr(0, tab1), line);
        for (size_t id : parse_numbers(row.substr(tab1 + 1, tab2 - tab1 - 1)))
            c.ids.push_back(static_cast<uint32_t>(id));
        c.piece_lens = parse_numbers(row.substr(tab2 + 1));
        cases.push_back(std::move(c));
    }
    return cases;
}

std::string show_ids(const std::vector<uint32_t>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size() && i < 40; ++i) s += std::to_string(ids[i]) + " ";
    if (ids.size() > 40) s += "...";
    return s;
}

// Runs every golden case; returns the number of failures.
int run_golden(Tokenizer& tok, const std::vector<Case>& cases, const char* label) {
    int fails = 0;
    for (const Case& c : cases) {
        const std::vector<uint32_t> got = tok.encode(c.text);
        if (got != c.ids) {
            ++fails;
            test::report(CASES, c.line, std::string(label) + ": encode mismatch\n      want: " +
                                            show_ids(c.ids) + "\n      got:  " + show_ids(got));
        }
        if (tok.decode(c.ids) != c.text) {
            ++fails;
            test::report(CASES, c.line, std::string(label) + ": decode(expected ids) != text");
        }
    }
    return fails;
}

Tokenizer load_or_die(const char* vocab, const char* merges) {
    Tokenizer t;
    std::string err;
    CHECK(t.load(vocab, merges, &err), "%s", err.c_str());
    return t;
}

}  // namespace

TEST(golden_cases_subset_fixture) {
    const std::vector<Case> cases = load_cases();
    EXPECT_TRUE(cases.size() >= 200);  // ROADMAP §7.2
    Tokenizer tok = load_or_die(SUBSET_VOCAB, SUBSET_MERGES);
    const int fails = run_golden(tok, cases, "subset");
    std::printf("    %zu golden cases vs subset fixture (%zu merges): %d failures\n", cases.size(),
                tok.num_merges(), fails);
    // Second pass hits the word cache; results must not change.
    EXPECT_EQ(run_golden(tok, cases, "subset/cached"), 0);
}

// The scanner and its Unicode class table must split exactly as HF does.
// Ids alone cannot catch most class errors (GPT-2 merges never cross
// pre-token boundaries), so the boundaries are compared directly.
TEST(golden_pretokenization) {
    int fails = 0;
    const std::vector<Case> cases = load_cases();
    for (const Case& c : cases) {
        std::vector<size_t> got;
        for (std::string_view p : gpt2_pretokenize(c.text)) got.push_back(p.size());
        if (got != c.piece_lens) {
            ++fails;
            std::string g, w;
            for (size_t x : got) g += std::to_string(x) + " ";
            for (size_t x : c.piece_lens) w += std::to_string(x) + " ";
            test::report(CASES, c.line, "pre-token byte lengths\n      want: " + w + "\n      got:  " + g);
        }
    }
    std::printf("    %zu golden pre-tokenizations: %d failures\n", cases.size(), fails);
}

TEST(golden_cases_full_vocab_if_downloaded) {
    if (!file_exists(FULL_VOCAB) || !file_exists(FULL_MERGES)) {
        std::printf("    data/vocab.json or data/merges.txt not present: full-vocab run not done "
                    "(subset run above still covers every case)\n");
        return;
    }
    Tokenizer tok = load_or_die(FULL_VOCAB, FULL_MERGES);
    EXPECT_EQ(tok.vocab_size(), 50257u);
    EXPECT_EQ(tok.num_merges(), size_t{50000});
    EXPECT_EQ(tok.find_token("<|endoftext|>"), int64_t{50256});
    const std::vector<Case> cases = load_cases();
    const int fails = run_golden(tok, cases, "full");
    std::printf("    %zu golden cases vs full GPT-Neo vocab: %d failures\n", cases.size(), fails);
}

TEST(pretokenize_rules) {
    struct P { const char* text; std::vector<std::string> want; };
    const P cases[] = {
        {"Hello world!", {"Hello", " world", "!"}},
        {"I'm here", {"I", "'m", " here"}},
        {"I'M", {"I", "'", "M"}},
        {"a  b", {"a", " ", " b"}},
        {"a   b", {"a", "  ", " b"}},
        {"x  ", {"x", "  "}},
        {"  x", {" ", " x"}},
        {"a\tb", {"a", "\t", "b"}},
        {"a \tb", {"a", " ", "\t", "b"}},
        {"a\n\nb", {"a", "\n", "\n", "b"}},
        {"3.14", {"3", ".", "14"}},
        {"abc123", {"abc", "123"}},
        {" 42", {" 42"}},
        {"!!'s", {"!!'", "s"}},
        {" ", {" "}},
        {"", {}},
    };
    for (const P& p : cases) {
        const std::vector<std::string_view> got = gpt2_pretokenize(p.text);
        std::vector<std::string> g(got.begin(), got.end());
        if (g != p.want) {
            std::string s;
            for (const std::string& x : g) s += "[" + x + "]";
            test::report(__FILE__, __LINE__, std::string("pretokenize '") + p.text + "' -> " + s);
        }
    }
}

// Random byte strings (including invalid UTF-8): pieces tile the input,
// and decode(encode(s)) == s.
TEST(random_bytes_round_trip) {
    Tokenizer tok = load_or_die(SUBSET_VOCAB, SUBSET_MERGES);
    Rng rng(455, 0);
    const char alphabet[] = "ab 'sd\n\t.!1\xC3\xA9\xE2\x80\x9C\xF0\x9F\x98\x8A\x80\xFF";
    for (int iter = 0; iter < 2000; ++iter) {
        std::string s;
        const int len = static_cast<int>(rng.uniform_int(40));
        for (int k = 0; k < len; ++k) {
            if (rng.uniform() < 0.2) s.push_back(static_cast<char>(rng.uniform_int(256)));
            else s.push_back(alphabet[rng.uniform_int(sizeof(alphabet) - 1)]);
        }
        std::string joined;
        for (std::string_view piece : gpt2_pretokenize(s)) {
            ASSERT_TRUE(!piece.empty());
            joined.append(piece);
        }
        ASSERT_EQ(joined, s);
        ASSERT_EQ(tok.decode(tok.encode(s)), s);
    }
}

TEST(json_reader) {
    std::vector<std::pair<std::string, int64_t>> kv;
    std::string err;
    ASSERT_TRUE(parse_json_string_int_object(
        " { \"a\" : 1 , \"\\u0120b\\\"\\\\\\/\\n\" :2, \"\\ud83d\\ude00\": 30 } \n", &kv, &err));
    ASSERT_EQ(kv.size(), size_t{3});
    EXPECT_EQ(kv[0].first, std::string("a"));
    EXPECT_EQ(kv[1].first, std::string("\xC4\xA0" "b\"\\/\n"));
    EXPECT_EQ(kv[1].second, int64_t{2});
    EXPECT_EQ(kv[2].first, std::string("\xF0\x9F\x98\x80"));  // U+1F600 from a surrogate pair
    EXPECT_TRUE(parse_json_string_int_object("{}", &kv, &err) && kv.empty());

    const char* bad[] = {
        "", "[]", "{\"a\":1,}", "{\"a\":-1}", "{\"a\":1.5}", "{\"a\":01}", "{\"a\" 1}",
        "{\"a\":1} x", "{\"a\":1", "{\"\\ud83d\":1}", "{\"\\x\":1}", "{\"a\nb\":1}", "{a:1}",
    };
    for (const char* b : bad) {
        err.clear();
        if (parse_json_string_int_object(b, &kv, &err) || err.empty())
            test::report(__FILE__, __LINE__, std::string("JSON accepted: ") + b);
    }
}

TEST(load_errors) {
    // Minimal valid vocab: the 256 single-byte tokens (written in the byte
    // alphabet: printable bytes as themselves, others as U+0100 + k).
    std::string vocab = "{";
    uint32_t next = 0x100;
    for (int b = 0; b < 256; ++b) {
        const bool keeps = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        const uint32_t cp = keeps ? static_cast<uint32_t>(b) : next++;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "\"\\u%04X\":%d,", cp, b);
        vocab += buf;
    }
    std::string err;
    Tokenizer t;
    EXPECT_TRUE(t.load_from_memory(vocab + "\"ab\":256}", "#version: 0.2\na b\n", &err));
    EXPECT_EQ(t.encode("abab"), (std::vector<uint32_t>{256, 256}));
    EXPECT_EQ(t.encode("aab"), (std::vector<uint32_t>{'a', 256}));

    EXPECT_TRUE(!t.load_from_memory(vocab + "\"ab\":256}", "a c\n", &err));  // "ac" missing
    EXPECT_TRUE(err.find("missing from vocab") != std::string::npos);
    EXPECT_TRUE(!t.load_from_memory(vocab + "\"ab\":256}", "a b\na b\n", &err));
    EXPECT_TRUE(err.find("duplicate merge") != std::string::npos);
    EXPECT_TRUE(!t.load_from_memory(vocab + "\"ab\":5}", "", &err));  // id 5 reused
    EXPECT_TRUE(err.find("duplicate id") != std::string::npos);
    EXPECT_TRUE(!t.load_from_memory("{\"a\":0}", "", &err));
    EXPECT_TRUE(err.find("missing single-byte token") != std::string::npos);
    EXPECT_TRUE(!t.load("tests/golden/nope.json", SUBSET_MERGES, &err));
    EXPECT_TRUE(err.find("cannot open") != std::string::npos);

    std::string out;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            Tokenizer u = load_or_die(SUBSET_VOCAB, SUBSET_MERGES);
            u.decode({4000000000u});
        },
        &out));
    EXPECT_TRUE(out.find("unknown token id") != std::string::npos);
}

TEST_MAIN()
