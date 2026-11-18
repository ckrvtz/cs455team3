// prepare.cpp: two-pass dataset preparation (see include/tslm/prepare.h).
#include "tslm/prepare.h"

#include "tslm/common.h"
#include "tslm/token_file.h"
#include "tslm/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace tslm {

namespace {

const std::string STORY_SEP = "<|endoftext|>";

bool is_ascii_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

std::string trim_ascii(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && is_ascii_space(s[b])) ++b;
    while (e > b && is_ascii_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

FILE* open_or_die(const std::string& path, const char* mode) {
    FILE* f = std::fopen(path.c_str(), mode);
    CHECK(f != nullptr, "prepare_data: cannot open '%s' (mode %s)", path.c_str(), mode);
    return f;
}

void write_or_die(FILE* f, const void* p, size_t n, const std::string& path) {
    CHECK(std::fwrite(p, 1, n, f) == n, "prepare_data: write error on '%s'", path.c_str());
}

void close_or_die(FILE* f, const std::string& path) {
    CHECK(std::fclose(f) == 0, "prepare_data: error closing '%s'", path.c_str());
}

void rename_or_die(const std::string& from, const std::string& to) {
    CHECK(std::rename(from.c_str(), to.c_str()) == 0, "prepare_data: cannot rename '%s' -> '%s'",
          from.c_str(), to.c_str());
}

void write_text_atomic(const std::string& path, const std::string& text) {
    const std::string part = path + ".part";
    FILE* f = open_or_die(part, "wb");
    write_or_die(f, text.data(), text.size(), part);
    close_or_die(f, part);
    rename_or_die(part, path);
}

// Streams non-empty, trimmed stories from a text file, reading fixed-size
// chunks. A separator split across two chunks is found once the next chunk
// arrives, because the unconsumed tail is kept in the buffer.
class StoryReader {
public:
    StoryReader(const std::string& path, size_t chunk)
        : path_(path), f_(open_or_die(path, "rb")), chunk_(chunk) {
        CHECK(chunk_ >= 1, "prepare_data: read_chunk_bytes must be >= 1");
    }
    ~StoryReader() { std::fclose(f_); }
    StoryReader(const StoryReader&) = delete;
    StoryReader& operator=(const StoryReader&) = delete;

    bool next(std::string* story) {
        for (;;) {
            const size_t s = buf_.find(STORY_SEP, pos_);
            if (s != std::string::npos) {
                *story = trim_ascii(buf_.substr(pos_, s - pos_));
                pos_ = s + STORY_SEP.size();
                if (!story->empty()) return true;
                continue;
            }
            if (eof_) {
                if (pos_ >= buf_.size()) return false;
                *story = trim_ascii(buf_.substr(pos_));
                pos_ = buf_.size();
                if (!story->empty()) return true;
                return false;
            }
            buf_.erase(0, pos_);
            pos_ = 0;
            const size_t old = buf_.size();
            buf_.resize(old + chunk_);
            const size_t n = std::fread(&buf_[old], 1, chunk_, f_);
            buf_.resize(old + n);
            CHECK(!std::ferror(f_), "prepare_data: read error on '%s'", path_.c_str());
            if (n == 0) eof_ = true;
        }
    }

private:
    std::string path_;
    FILE* f_;
    size_t chunk_;
    std::string buf_;
    size_t pos_ = 0;
    bool eof_ = false;
};

// Buffered little-endian uint16 writer that also hashes the bytes it writes.
class U16Writer {
public:
    explicit U16Writer(const std::string& path) : path_(path), f_(open_or_die(path, "wb")) {}
    void raw(const void* p, size_t n) {
        flush();
        write_or_die(f_, p, n, path_);
        hash_ = fnv1a64(hash_, p, n);
    }
    void put(uint32_t v) {
        buf_.push_back(static_cast<unsigned char>(v & 0xFF));
        buf_.push_back(static_cast<unsigned char>(v >> 8));
        if (buf_.size() >= (1u << 20)) flush();
    }
    void close() {
        flush();
        close_or_die(f_, path_);
    }
    uint64_t hash() const { return hash_; }

private:
    void flush() {
        if (buf_.empty()) return;
        write_or_die(f_, buf_.data(), buf_.size(), path_);
        hash_ = fnv1a64(hash_, buf_.data(), buf_.size());
        buf_.clear();
    }
    std::string path_;
    FILE* f_;
    std::vector<unsigned char> buf_;
    uint64_t hash_ = FNV1A64_INIT;
};

// Pass 1 for one split: GPT ids (with EOS per story) -> raw uint16 temp file.
SplitStats tokenize_split(const std::string& txt, const std::string& tmp, Tokenizer& tok,
                          uint32_t eos, const PrepareOptions& opt, bool is_train,
                          std::vector<uint64_t>* hist) {
    SplitStats st;
    StoryReader reader(txt, opt.read_chunk_bytes);
    U16Writer out(tmp);
    std::string story;
    while (true) {
        if (is_train && opt.max_stories && st.stories == opt.max_stories) break;
        if (!reader.next(&story)) break;
        std::vector<uint32_t> ids = tok.encode(story);
        ids.push_back(eos);
        if (is_train && opt.max_tokens && st.tokens + ids.size() > opt.max_tokens) break;
        for (uint32_t id : ids) {
            CHECK(id <= 0xFFFF, "prepare_data: GPT id %u does not fit uint16", id);
            out.put(id);
            if (hist) ++(*hist)[id];
        }
        st.stories += 1;
        st.tokens += ids.size();
        if (opt.log && st.stories % 200000 == 0) {
            opt.log("  " + txt + ": " + std::to_string(st.stories) + " stories, " +
                    std::to_string(st.tokens) + " tokens");
        }
    }
    out.close();
    return st;
}

// Pass 2 for one split: remap the temp file into a token file.
uint64_t remap_split(const std::string& tmp, const std::string& bin, uint32_t vocab_size,
                     const std::vector<uint16_t>& gpt_to_new, SplitStats* st) {
    const std::string part = bin + ".part";
    U16Writer out(part);
    TokenFileHeader h;
    h.vocab_size = vocab_size;
    h.num_tokens = st->tokens;
    unsigned char hdr[TOKEN_FILE_HEADER_BYTES];
    encode_token_header(h, hdr);
    out.raw(hdr, sizeof(hdr));

    FILE* in = open_or_die(tmp, "rb");
    std::vector<unsigned char> buf(size_t(1) << 20);
    uint64_t seen = 0;
    st->unk_tokens = 0;
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), in)) > 0) {
        CHECK(n % 2 == 0, "prepare_data: odd-sized read from '%s'", tmp.c_str());
        for (size_t i = 0; i < n; i += 2) {
            const uint32_t gpt = static_cast<uint32_t>(buf[i]) | (static_cast<uint32_t>(buf[i + 1]) << 8);
            CHECK(gpt < gpt_to_new.size(), "prepare_data: GPT id %u out of range", gpt);
            const uint16_t id = gpt_to_new[gpt];
            st->unk_tokens += (id == 0);
            out.put(id);
        }
        seen += n / 2;
    }
    CHECK(!std::ferror(in), "prepare_data: read error on '%s'", tmp.c_str());
    std::fclose(in);
    CHECK(seen == st->tokens, "prepare_data: '%s' has %llu tokens, expected %llu", tmp.c_str(),
          static_cast<unsigned long long>(seen), static_cast<unsigned long long>(st->tokens));
    out.close();
    rename_or_die(part, bin);
    std::remove(tmp.c_str());
    return out.hash();
}

std::string escape_bytes(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '\\') {
            out.push_back(static_cast<char>(c));
        } else {
            char b[8];
            std::snprintf(b, sizeof(b), "\\x%02X", c);
            out += b;
        }
    }
    return out;
}

std::string hex64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}

std::string fmt_double(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.6f", v);
    return b;
}

}  // namespace

PrepareStats prepare_data(const PrepareOptions& opt) {
    CHECK(opt.vocab_size >= 3 && opt.vocab_size <= 65536,
          "prepare_data: vocab_size %d must be in [3, 65536]", opt.vocab_size);
    auto log = [&](const std::string& m) {
        if (opt.log) opt.log(m);
    };

    Tokenizer tok;
    std::string err;
    CHECK(tok.load(opt.vocab_json, opt.merges_txt, &err), "prepare_data: %s", err.c_str());
    const int64_t eos_signed = tok.find_token("<|endoftext|>");
    CHECK(eos_signed >= 0, "prepare_data: '<|endoftext|>' not in %s", opt.vocab_json.c_str());
    const uint32_t eos = static_cast<uint32_t>(eos_signed);
    CHECK(tok.vocab_size() <= 65536, "prepare_data: GPT vocab (%u) does not fit uint16", tok.vocab_size());

    PrepareStats ps;
    const std::string train_tmp = opt.out_dir + "/train.gpt16.tmp";
    const std::string val_tmp = opt.out_dir + "/val.gpt16.tmp";

    // ---- pass 1
    std::vector<uint64_t> hist(tok.vocab_size(), 0);
    log("pass 1: tokenizing " + opt.train_txt);
    ps.train = tokenize_split(opt.train_txt, train_tmp, tok, eos, opt, true, &hist);
    log("pass 1: tokenizing " + opt.val_txt);
    ps.val = tokenize_split(opt.val_txt, val_tmp, tok, eos, opt, false, nullptr);
    CHECK(ps.train.stories > 0, "prepare_data: no training stories selected");

    // ---- vocabulary
    std::vector<uint32_t> cand;
    for (uint32_t id = 0; id < tok.vocab_size(); ++id) {
        if (id != eos && tok.has_id(id)) cand.push_back(id);
        if (hist[id] > 0 && id != eos) ++ps.train_distinct_gpt_tokens;
    }
    const size_t keep = static_cast<size_t>(opt.vocab_size) - 2;
    CHECK(cand.size() >= keep, "prepare_data: only %zu candidate tokens for %zu vocab slots",
          cand.size(), keep);
    // Total order (count desc, GPT id asc) -> std::sort result is unique.
    std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) {
        return hist[a] != hist[b] ? hist[a] > hist[b] : a < b;
    });
    std::vector<int64_t> new_to_gpt(static_cast<size_t>(opt.vocab_size));
    std::vector<uint16_t> gpt_to_new(tok.vocab_size(), 0);
    new_to_gpt[0] = -1;
    new_to_gpt[1] = eos;
    gpt_to_new[eos] = 1;
    for (size_t k = 0; k < keep; ++k) {
        new_to_gpt[k + 2] = cand[k];
        gpt_to_new[cand[k]] = static_cast<uint16_t>(k + 2);
    }
    ps.vocab_min_count = hist[cand[keep - 1]];
    ps.vocab_hash = FNV1A64_INIT;
    for (int64_t g : new_to_gpt) {
        const uint32_t v = g < 0 ? 0xFFFFFFFFu : static_cast<uint32_t>(g);
        const unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                                    static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
        ps.vocab_hash = fnv1a64(ps.vocab_hash, b, 4);
    }

    // ---- pass 2
    const uint32_t V = static_cast<uint32_t>(opt.vocab_size);
    log("pass 2: writing train.bin / val.bin");
    ps.train_bin_hash = remap_split(train_tmp, opt.out_dir + "/train.bin", V, gpt_to_new, &ps.train);
    ps.val_bin_hash = remap_split(val_tmp, opt.out_dir + "/val.bin", V, gpt_to_new, &ps.val);

    std::string vm;
    for (size_t k = 0; k < new_to_gpt.size(); ++k) {
        const int64_t g = new_to_gpt[k];
        const std::string text = k == 0 ? "<unk>" : escape_bytes(tok.token_bytes(static_cast<uint32_t>(g)));
        vm += std::to_string(k) + "\t" + std::to_string(g) + "\t" + text + "\n";
    }
    write_text_atomic(opt.out_dir + "/vocab_map.txt", vm);

    std::string meta;
    auto kv = [&](const std::string& k, const std::string& v) { meta += k + "=" + v + "\n"; };
    kv("token_file_version", std::to_string(TOKEN_FILE_VERSION));
    kv("vocab_size", std::to_string(opt.vocab_size));
    kv("gpt_vocab_size", std::to_string(tok.vocab_size()));
    kv("eos_gpt_id", std::to_string(eos));
    kv("train_source", opt.train_txt);
    kv("val_source", opt.val_txt);
    kv("max_stories", std::to_string(opt.max_stories));
    kv("max_tokens", std::to_string(opt.max_tokens));
    kv("train_stories", std::to_string(ps.train.stories));
    kv("train_tokens", std::to_string(ps.train.tokens));
    kv("train_unk_tokens", std::to_string(ps.train.unk_tokens));
    kv("train_coverage", fmt_double(ps.train.coverage()));
    kv("val_stories", std::to_string(ps.val.stories));
    kv("val_tokens", std::to_string(ps.val.tokens));
    kv("val_unk_tokens", std::to_string(ps.val.unk_tokens));
    kv("val_coverage", fmt_double(ps.val.coverage()));
    kv("train_distinct_gpt_tokens", std::to_string(ps.train_distinct_gpt_tokens));
    kv("vocab_min_count", std::to_string(ps.vocab_min_count));
    kv("vocab_hash_fnv1a64", hex64(ps.vocab_hash));
    kv("train_bin_fnv1a64", hex64(ps.train_bin_hash));
    kv("val_bin_fnv1a64", hex64(ps.val_bin_hash));
    write_text_atomic(opt.out_dir + "/meta.txt", meta);
    return ps;
}

}  // namespace tslm
