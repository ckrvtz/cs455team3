// vocab_map.cpp: vocab_map.txt reader and id decoder (see include/tslm/vocab_map.h).
#include "tslm/vocab_map.h"

#include "tslm/common.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace tslm {

namespace {

int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

bool parse_i64(const std::string& s, int64_t* v) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || *end != '\0') return false;
    *v = x;
    return true;
}

}  // namespace

bool parse_vocab_map(const std::string& text, VocabMap* out, std::string* err) {
    out->bytes.clear();
    out->gpt_ids.clear();
    size_t pos = 0;
    int line_no = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        auto fail = [&](const std::string& m) {
            if (err) *err = "vocab_map line " + std::to_string(line_no) + ": " + m;
            return false;
        };
        const size_t t1 = line.find('\t');
        const size_t t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
        if (t2 == std::string::npos || line.find('\t', t2 + 1) != std::string::npos)
            return fail("expected exactly 3 TAB-separated fields");
        int64_t id, gpt;
        if (!parse_i64(line.substr(0, t1), &id) || !parse_i64(line.substr(t1 + 1, t2 - t1 - 1), &gpt))
            return fail("bad integer field");
        if (id != static_cast<int64_t>(out->bytes.size()))
            return fail("expected id " + std::to_string(out->bytes.size()) + ", got " + std::to_string(id));
        const std::string esc = line.substr(t2 + 1);
        std::string raw;
        for (size_t i = 0; i < esc.size(); ++i) {
            if (esc[i] != '\\') {
                raw.push_back(esc[i]);
                continue;
            }
            // Need esc[i+1..i+3] = "xHH".
            if (i + 3 >= esc.size() || esc[i + 1] != 'x' || hexval(esc[i + 2]) < 0 || hexval(esc[i + 3]) < 0)
                return fail("bad escape (expected \\xHH)");
            raw.push_back(static_cast<char>(hexval(esc[i + 2]) * 16 + hexval(esc[i + 3])));
            i += 3;
        }
        if (id == UNK_ID) {
            if (gpt != -1 || esc != "<unk>") return fail("id 0 must be '<unk>' with gpt_id -1");
            raw.clear();
        } else if (gpt < 0) {
            return fail("gpt_id must be >= 0 for ids other than 0");
        }
        out->bytes.push_back(raw);
        out->gpt_ids.push_back(gpt);
    }
    if (out->bytes.size() < 2) {
        if (err) *err = "vocab_map: needs at least <unk> and EOS";
        return false;
    }
    return true;
}

bool load_vocab_map(const std::string& path, VocabMap* out, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open '" + path + "'";
        return false;
    }
    std::string text;
    char buf[1 << 15];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) {
        if (err) *err = "error reading '" + path + "'";
        return false;
    }
    std::string e;
    if (!parse_vocab_map(text, out, &e)) {
        if (err) *err = path + ": " + e;
        return false;
    }
    return true;
}

std::string decode_ids(const VocabMap& vm, const int32_t* ids, size_t n, const char* eos_marker,
                       const char* unk_marker) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        const int32_t id = ids[i];
        CHECK(id >= 0 && static_cast<size_t>(id) < vm.size(), "decode_ids: id %d outside vocab of %zu",
              id, vm.size());
        if (id == UNK_ID) out += unk_marker ? unk_marker : "<unk>";
        else if (id == EOS_ID && eos_marker) out += eos_marker;
        else out += vm.bytes[static_cast<size_t>(id)];
    }
    return out;
}

}  // namespace tslm
