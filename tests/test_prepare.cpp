// test_prepare.cpp: token file header and prepare_data (ROADMAP P1.3, §7.3).
// Runs on tiny synthetic stories with the committed subset tokenizer
// fixture (plus an "<|endoftext|>" entry), so it needs no download.
// The expected token stream comes from a separate, deliberately naive
// recomputation (direct tokenize + std::map counting) and is compared
// byte for byte.
#include "test.h"

#include "tslm/common.h"
#include "tslm/prepare.h"
#include "tslm/token_file.h"
#include "tslm/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace tslm;

namespace {

std::string tmpdir() {
    const char* t = std::getenv("TSLM_TEST_TMPDIR");
    return t ? t : ".";
}

std::string read_all(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f, "cannot open %s", path.c_str());
    std::string s;
    char buf[1 << 15];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

void write_all(const std::string& path, const std::string& s) {
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f && std::fwrite(s.data(), 1, s.size(), f) == s.size(), "cannot write %s", path.c_str());
    std::fclose(f);
}

const std::vector<std::string> TRAIN_STORIES = {
    "Once upon a time, there was a little girl named Lily.",
    "One day, Tim went to the park. He saw a big, red ball!",
    "The cat sat. The cat ran. The dog sat.",
    "Lily and Tim played all day.\nThey were happy.",
    "\"Hello,\" she said. \"I'm here!\"",
    "a a a a b b b c c d",
};
const std::vector<std::string> VAL_STORIES = {
    "The dog ran to the park.",
    "Zebra xylophone quokka!",  // mostly rare tokens -> <unk>
};

// Join stories as the TinyStories files do, with extra whitespace and an
// empty story to exercise trimming and skipping.
std::string make_file(const std::vector<std::string>& stories) {
    std::string s = "  ";
    for (size_t i = 0; i < stories.size(); ++i) {
        s += stories[i] + "\n<|endoftext|>\n";
        if (i == 1) s += "   \n<|endoftext|>\n";  // empty story
    }
    return s;  // ends with a separator
}

struct Fixture {
    std::string dir, train, val, vocab, merges;
};

Fixture make_fixture(const std::string& name) {
    Fixture fx;
    fx.dir = tmpdir() + "/" + name;
    const std::string mk = "mkdir -p '" + fx.dir + "'";
    CHECK(std::system(mk.c_str()) == 0, "mkdir failed");
    fx.train = fx.dir + "/train.txt";
    fx.val = fx.dir + "/val.txt";
    fx.vocab = fx.dir + "/vocab.json";
    fx.merges = "tests/golden/tokenizer_merges_subset.txt";
    write_all(fx.train, make_file(TRAIN_STORIES));
    write_all(fx.val, make_file(VAL_STORIES));
    std::string v = read_all("tests/golden/tokenizer_vocab_subset.json");
    const size_t close = v.rfind('}');
    v = v.substr(0, close) + ", \"<|endoftext|>\": 50256}";
    write_all(fx.vocab, v);
    return fx;
}

PrepareOptions options(const Fixture& fx, int V) {
    PrepareOptions o;
    o.train_txt = fx.train;
    o.val_txt = fx.val;
    o.vocab_json = fx.vocab;
    o.merges_txt = fx.merges;
    o.out_dir = fx.dir;
    o.vocab_size = V;
    return o;
}

struct TokenFile {
    TokenFileHeader h;
    std::vector<uint16_t> ids;
};

TokenFile read_token_file(const std::string& path) {
    const std::string s = read_all(path);
    TokenFile tf;
    CHECK(s.size() >= TOKEN_FILE_HEADER_BYTES, "%s too short", path.c_str());
    std::string err;
    CHECK(decode_token_header(reinterpret_cast<const unsigned char*>(s.data()), &tf.h, &err), "%s",
          err.c_str());
    CHECK(s.size() == TOKEN_FILE_HEADER_BYTES + 2 * tf.h.num_tokens, "%s: size/header mismatch",
          path.c_str());
    for (size_t i = TOKEN_FILE_HEADER_BYTES; i < s.size(); i += 2)
        tf.ids.push_back(static_cast<uint16_t>(static_cast<unsigned char>(s[i]) |
                                               (static_cast<unsigned char>(s[i + 1]) << 8)));
    return tf;
}

// Naive oracle: tokenize stories, count, rank with std::map, remap.
struct Oracle {
    std::vector<uint16_t> train, val;
    std::vector<int64_t> new_to_gpt;
};

Oracle oracle(const Fixture& fx, int V, size_t n_train_stories) {
    Tokenizer tok;
    std::string err;
    CHECK(tok.load(fx.vocab, fx.merges, &err), "%s", err.c_str());
    const uint32_t eos = 50256;
    std::vector<std::vector<uint32_t>> tr, va;
    std::map<uint32_t, uint64_t> count;
    for (size_t i = 0; i < n_train_stories; ++i) {
        auto ids = tok.encode(TRAIN_STORIES[i]);
        ids.push_back(eos);
        for (uint32_t id : ids) ++count[id];
        tr.push_back(ids);
    }
    for (const auto& s : VAL_STORIES) {
        auto ids = tok.encode(s);
        ids.push_back(eos);
        va.push_back(ids);
    }
    // Rank all known ids except EOS: count desc, then id asc.
    std::vector<std::pair<uint32_t, uint64_t>> ranked;
    for (uint32_t id = 0; id < tok.vocab_size(); ++id) {
        if (id == eos || !tok.has_id(id)) continue;
        const auto it = count.find(id);
        ranked.push_back({id, it == count.end() ? 0 : it->second});
    }
    for (size_t i = 0; i < ranked.size(); ++i)  // simple selection sort: obviously correct
        for (size_t j = i + 1; j < ranked.size(); ++j) {
            const bool better = ranked[j].second > ranked[i].second ||
                                (ranked[j].second == ranked[i].second && ranked[j].first < ranked[i].first);
            if (better) std::swap(ranked[i], ranked[j]);
        }
    Oracle o;
    o.new_to_gpt = {-1, eos};
    std::map<uint32_t, uint16_t> to_new{{eos, 1}};
    for (int k = 0; k < V - 2; ++k) {
        o.new_to_gpt.push_back(ranked[size_t(k)].first);
        to_new[ranked[size_t(k)].first] = static_cast<uint16_t>(k + 2);
    }
    auto remap = [&](const std::vector<std::vector<uint32_t>>& stories, std::vector<uint16_t>* out) {
        for (const auto& s : stories)
            for (uint32_t id : s) {
                const auto it = to_new.find(id);
                out->push_back(it == to_new.end() ? 0 : it->second);
            }
    };
    remap(tr, &o.train);
    remap(va, &o.val);
    return o;
}

}  // namespace

TEST(token_header_round_trip_and_validation) {
    TokenFileHeader h;
    h.vocab_size = 10000;
    h.num_tokens = 0x0123456789ABCDEFull;
    unsigned char b[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, b);
    EXPECT_TRUE(b[0] == 'T' && b[1] == 'S' && b[2] == 'T' && b[3] == 'K');
    EXPECT_EQ(int(b[4]), 1);              // version, little-endian
    EXPECT_EQ(int(b[8]), 10000 & 0xFF);   // vocab_size low byte
    EXPECT_EQ(int(b[9]), 10000 >> 8);
    EXPECT_EQ(int(b[16]), 0xEF);          // num_tokens at offset 16, LE
    EXPECT_EQ(int(b[23]), 0x01);
    TokenFileHeader d;
    std::string err;
    ASSERT_TRUE(decode_token_header(b, &d, &err));
    EXPECT_EQ(d.vocab_size, 10000u);
    EXPECT_EQ(d.num_tokens, h.num_tokens);

    unsigned char bad[TOKEN_FILE_HEADER_BYTES];
    for (int which = 0; which < 4; ++which) {
        for (size_t i = 0; i < sizeof(b); ++i) bad[i] = b[i];
        if (which == 0) bad[0] = 'X';   // magic
        if (which == 1) bad[4] = 2;     // version
        if (which == 2) bad[13] = 1;    // padding
        if (which == 3) bad[30] = 1;    // reserved
        EXPECT_TRUE(!decode_token_header(bad, &d, &err));
    }
    EXPECT_EQ(fnv1a64(FNV1A64_INIT, "a", 1), 0xaf63dc4c8601ec8cull);  // published FNV-1a test vector
}

TEST(prepare_matches_naive_oracle) {
    const Fixture fx = make_fixture("prep_oracle");
    const int V = 40;
    const PrepareStats s = prepare_data(options(fx, V));
    const Oracle o = oracle(fx, V, TRAIN_STORIES.size());

    EXPECT_EQ(s.train.stories, uint64_t(TRAIN_STORIES.size()));  // empty story skipped
    EXPECT_EQ(s.val.stories, uint64_t(VAL_STORIES.size()));
    const TokenFile tr = read_token_file(fx.dir + "/train.bin");
    const TokenFile va = read_token_file(fx.dir + "/val.bin");
    EXPECT_EQ(tr.h.vocab_size, uint32_t(V));
    EXPECT_TRUE(tr.ids == o.train);
    EXPECT_TRUE(va.ids == o.val);
    EXPECT_EQ(s.train.tokens, uint64_t(o.train.size()));

    // One EOS (id 1) per story, each story's last token.
    int eos = 0;
    for (uint16_t id : tr.ids) eos += id == 1;
    EXPECT_EQ(eos, int(TRAIN_STORIES.size()));
    EXPECT_EQ(tr.ids.back(), uint16_t{1});

    // Coverage = non-<unk> fraction.
    uint64_t unk = 0;
    for (uint16_t id : va.ids) unk += id == 0;
    EXPECT_EQ(s.val.unk_tokens, unk);
    EXPECT_TRUE(unk > 0);
    EXPECT_NEAR(s.val.coverage(), 1.0 - double(unk) / double(va.ids.size()), 1e-12);
    EXPECT_EQ(s.train.coverage(), 1.0 - double(s.train.unk_tokens) / double(s.train.tokens));

    // vocab_map.txt: V lines, fixed first two, gpt ids follow the oracle.
    const std::string vm = read_all(fx.dir + "/vocab_map.txt");
    const std::string head = "0\t-1\t<unk>\n1\t50256\t<|endoftext|>\n";
    EXPECT_EQ(vm.substr(0, head.size()), head);
    int lines = 0;
    size_t pos = 0;
    while (pos < vm.size()) {
        const size_t nl = vm.find('\n', pos);
        const std::string row = vm.substr(pos, nl - pos);
        const size_t t1 = row.find('\t'), t2 = row.find('\t', t1 + 1);
        EXPECT_EQ(std::stoll(row.substr(0, t1)), int64_t(lines));
        EXPECT_EQ(std::stoll(row.substr(t1 + 1, t2 - t1 - 1)), o.new_to_gpt[size_t(lines)]);
        ++lines;
        pos = nl + 1;
    }
    EXPECT_EQ(lines, V);
    // Tokens are written as raw (escaped) bytes, never in GPT-2's byte
    // alphabet: 'Ġ' (U+0120, the alphabet's space) must not appear.
    EXPECT_TRUE(vm.find("\xC4\xA0") == std::string::npos);

    const std::string meta = read_all(fx.dir + "/meta.txt");
    EXPECT_TRUE(meta.find("vocab_size=40\n") != std::string::npos);
    EXPECT_TRUE(meta.find("train_stories=6\n") != std::string::npos);
    EXPECT_TRUE(meta.find("val_coverage=") != std::string::npos);
    EXPECT_TRUE(meta.find("vocab_hash_fnv1a64=") != std::string::npos);
}

TEST(vocab_map_escapes_token_bytes) {
    const Fixture fx = make_fixture("prep_escape");
    prepare_data(options(fx, 40));
    const std::string vm = read_all(fx.dir + "/vocab_map.txt");
    // " and"-style tokens keep their leading space; '\n' must appear escaped.
    EXPECT_TRUE(vm.find("\t the\n") != std::string::npos || vm.find("\t a\n") != std::string::npos);
    size_t tabs = 0, nls = 0;
    for (char c : vm) {
        tabs += c == '\t';
        nls += c == '\n';
    }
    EXPECT_EQ(tabs, size_t(2 * 40));  // exactly two TABs per line: no raw TAB in tokens
    EXPECT_EQ(nls, size_t(40));       // no raw newline in tokens
}

TEST(max_stories_and_max_tokens_truncate_train_only) {
    const Fixture fx = make_fixture("prep_trunc");
    PrepareOptions o = options(fx, 40);
    o.max_stories = 2;
    const PrepareStats a = prepare_data(o);
    EXPECT_EQ(a.train.stories, uint64_t{2});
    EXPECT_EQ(a.val.stories, uint64_t(VAL_STORIES.size()));
    const Oracle oa = oracle(fx, 40, 2);
    EXPECT_TRUE(read_token_file(fx.dir + "/train.bin").ids == oa.train);
    EXPECT_TRUE(read_token_file(fx.dir + "/val.bin").ids == oa.val);  // val unaffected

    // max_tokens: keep whole stories while the total stays <= limit.
    Tokenizer tok;
    std::string err;
    ASSERT_TRUE(tok.load(fx.vocab, fx.merges, &err));
    const uint64_t n0 = tok.encode(TRAIN_STORIES[0]).size() + 1;
    const uint64_t n1 = tok.encode(TRAIN_STORIES[1]).size() + 1;
    o.max_stories = 0;
    o.max_tokens = n0 + n1 + 1;  // room for two stories, not three
    const PrepareStats b = prepare_data(o);
    EXPECT_EQ(b.train.stories, uint64_t{2});
    EXPECT_EQ(b.train.tokens, n0 + n1);
    o.max_tokens = n0 + n1;  // exactly two
    EXPECT_EQ(prepare_data(o).train.stories, uint64_t{2});
    o.max_tokens = n0 - 1;  // not even one story fits
    std::string out;
    EXPECT_TRUE(test::dies_with_abort([&] { prepare_data(o); }, &out));
    EXPECT_TRUE(out.find("no training stories") != std::string::npos);
}

// meta.txt records the input paths, which differ between the two fixture
// directories; every other line must match.
static std::string drop_sources(const std::string& s) {
    std::string out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) nl = s.size() - 1;
        const std::string line = s.substr(pos, nl - pos + 1);
        if (line.rfind("train_source=", 0) != 0 && line.rfind("val_source=", 0) != 0) out += line;
        pos = nl + 1;
    }
    return out;
}

TEST(deterministic_and_chunk_size_independent) {
    const Fixture fa = make_fixture("prep_det_a");
    const Fixture fb = make_fixture("prep_det_b");
    PrepareOptions a = options(fa, 40), b = options(fb, 40);
    b.read_chunk_bytes = 5;  // separators split across reads
    const PrepareStats sa = prepare_data(a), sb = prepare_data(b);
    for (const char* f : {"/train.bin", "/val.bin", "/vocab_map.txt", "/meta.txt"}) {
        const bool same = drop_sources(read_all(fa.dir + f)) == drop_sources(read_all(fb.dir + f));
        if (!same) test::report(__FILE__, __LINE__, std::string("outputs differ: ") + f);
    }
    EXPECT_EQ(sa.train_bin_hash, sb.train_bin_hash);
    EXPECT_EQ(sa.vocab_hash, sb.vocab_hash);
    // The reported hash is the hash of the file on disk.
    const std::string tb = read_all(fa.dir + "/train.bin");
    EXPECT_EQ(sa.train_bin_hash, fnv1a64(FNV1A64_INIT, tb.data(), tb.size()));
    // No temp/part files left behind.
    FILE* t = std::fopen((fa.dir + "/train.gpt16.tmp").c_str(), "rb");
    EXPECT_TRUE(t == nullptr);
    if (t) std::fclose(t);
}

TEST(bad_inputs_abort) {
    const Fixture fx = make_fixture("prep_bad");
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(
        [&] {
            PrepareOptions o = options(fx, 40);
            o.train_txt = fx.dir + "/missing.txt";
            prepare_data(o);
        },
        &out));
    EXPECT_TRUE(out.find("cannot open") != std::string::npos);
    out.clear();
    EXPECT_TRUE(test::dies_with_abort([&] { prepare_data(options(fx, 100000)); }, &out));
    out.clear();
    EXPECT_TRUE(test::dies_with_abort([&] { prepare_data(options(fx, 5000)); }, &out));  // > fixture vocab
    EXPECT_TRUE(out.find("candidate tokens") != std::string::npos);
}

TEST_MAIN()
