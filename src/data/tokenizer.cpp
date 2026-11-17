// tokenizer.cpp: GPT-2 byte-level BPE (see include/tslm/tokenizer.h).
//
// Sections: (1) UTF-8 helpers, (2) Unicode character classes for the
// pre-tokenizer, (3) the pre-tokenization scanner, (4) the minimal JSON
// reader, (5) the GPT-2 byte<->unicode alphabet, (6) Tokenizer load/encode/decode.
#include "tslm/tokenizer.h"

#include "tslm/common.h"

#include <cstdio>

namespace tslm {

namespace {

// ---------------------------------------------------------------------------
// (1) UTF-8
// ---------------------------------------------------------------------------

constexpr uint32_t INVALID_CP = 0xFFFFFFFFu;

// Decode one codepoint at s[i]. Sets *len to the bytes consumed (>= 1).
// Malformed, overlong, surrogate or out-of-range sequences decode as
// INVALID_CP with *len = 1, so scanning always advances byte by byte.
uint32_t utf8_decode(std::string_view s, size_t i, size_t* len) {
    const unsigned char c0 = static_cast<unsigned char>(s[i]);
    *len = 1;
    if (c0 < 0x80) return c0;
    int n;
    uint32_t cp, min;
    if ((c0 & 0xE0) == 0xC0) { n = 1; cp = c0 & 0x1F; min = 0x80; }
    else if ((c0 & 0xF0) == 0xE0) { n = 2; cp = c0 & 0x0F; min = 0x800; }
    else if ((c0 & 0xF8) == 0xF0) { n = 3; cp = c0 & 0x07; min = 0x10000; }
    else return INVALID_CP;
    if (i + static_cast<size_t>(n) >= s.size()) return INVALID_CP;  // truncated sequence
    for (int k = 1; k <= n; ++k) {
        const unsigned char c = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
        if ((c & 0xC0) != 0x80) return INVALID_CP;
        cp = (cp << 6) | (c & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return INVALID_CP;
    *len = static_cast<size_t>(n) + 1;
    return cp;
}

void utf8_append(uint32_t cp, std::string* out) {
    if (cp < 0x80) {
        out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---------------------------------------------------------------------------
// (2) Character classes
// ---------------------------------------------------------------------------

enum class Cls : uint8_t { Letter, Number, Space, Other };

struct Range {
    uint32_t lo, hi;
    Cls cls;
};

// Non-ASCII codepoints that are NOT letters, sorted and non-overlapping.
// Anything non-ASCII not listed here is classed as a letter (ROADMAP §7.2).
// Space = Unicode White_Space (what \s matches). Number = \p{N} (Nd/Nl/No).
// Other = punctuation, symbols, emoji, marks, controls, format characters.
// Coverage: Latin-1, general/supplemental punctuation, letterlike and number
// forms, arrows/math/technical/box/shapes/dingbats, CJK punctuation,
// fullwidth forms, emoji and symbol planes, and the combining marks of
// Latin, Greek, Cyrillic, Hebrew, Arabic, Devanagari and Thai. Other scripts'
// marks/digits default to letters, a known approximation; TinyStories is
// English (coverage is reported by prepare_data).
constexpr Range NON_LETTER[] = {
    {0x0080, 0x0084, Cls::Other}, {0x0085, 0x0085, Cls::Space}, {0x0086, 0x009F, Cls::Other},
    {0x00A0, 0x00A0, Cls::Space}, {0x00A1, 0x00A9, Cls::Other}, {0x00AB, 0x00B1, Cls::Other},
    {0x00B2, 0x00B3, Cls::Number}, {0x00B4, 0x00B4, Cls::Other}, {0x00B6, 0x00B8, Cls::Other},
    {0x00B9, 0x00B9, Cls::Number}, {0x00BB, 0x00BB, Cls::Other}, {0x00BC, 0x00BE, Cls::Number},
    {0x00BF, 0x00BF, Cls::Other}, {0x00D7, 0x00D7, Cls::Other}, {0x00F7, 0x00F7, Cls::Other},
    {0x02C2, 0x02C5, Cls::Other}, {0x02D2, 0x02DF, Cls::Other}, {0x02E5, 0x02EB, Cls::Other},
    {0x02ED, 0x02ED, Cls::Other}, {0x02EF, 0x036F, Cls::Other},  // modifier symbols, combining marks
    {0x0375, 0x0375, Cls::Other}, {0x037E, 0x037E, Cls::Other}, {0x0384, 0x0385, Cls::Other},
    {0x0387, 0x0387, Cls::Other}, {0x03F6, 0x03F6, Cls::Other}, {0x0482, 0x0489, Cls::Other},
    {0x055A, 0x055F, Cls::Other}, {0x0589, 0x058A, Cls::Other}, {0x058D, 0x058F, Cls::Other},
    {0x0591, 0x05C7, Cls::Other},  // Hebrew points and punctuation (all non-letters)
    {0x05F3, 0x05F4, Cls::Other}, {0x0600, 0x061F, Cls::Other}, {0x064B, 0x065F, Cls::Other},
    {0x0660, 0x0669, Cls::Number}, {0x066A, 0x066D, Cls::Other}, {0x0670, 0x0670, Cls::Other},
    {0x06D4, 0x06D4, Cls::Other}, {0x06D6, 0x06E4, Cls::Other}, {0x06E7, 0x06ED, Cls::Other},
    {0x06F0, 0x06F9, Cls::Number}, {0x06FD, 0x06FE, Cls::Other},
    {0x0900, 0x0903, Cls::Other}, {0x093A, 0x093C, Cls::Other}, {0x093E, 0x094F, Cls::Other},
    {0x0951, 0x0957, Cls::Other}, {0x0962, 0x0965, Cls::Other}, {0x0966, 0x096F, Cls::Number},
    {0x0970, 0x0970, Cls::Other}, {0x0E31, 0x0E31, Cls::Other}, {0x0E34, 0x0E3A, Cls::Other},
    {0x0E3F, 0x0E3F, Cls::Other}, {0x0E47, 0x0E4F, Cls::Other}, {0x0E50, 0x0E59, Cls::Number},
    {0x0E5A, 0x0E5B, Cls::Other}, {0x1680, 0x1680, Cls::Space},
    {0x1AB0, 0x1AFF, Cls::Other}, {0x1DC0, 0x1DFF, Cls::Other},
    {0x2000, 0x200A, Cls::Space}, {0x200B, 0x2027, Cls::Other}, {0x2028, 0x2029, Cls::Space},
    {0x202A, 0x202E, Cls::Other}, {0x202F, 0x202F, Cls::Space}, {0x2030, 0x205E, Cls::Other},
    {0x205F, 0x205F, Cls::Space}, {0x2060, 0x206F, Cls::Other}, {0x2070, 0x2070, Cls::Number},
    {0x2074, 0x2079, Cls::Number}, {0x207A, 0x207E, Cls::Other}, {0x2080, 0x2089, Cls::Number},
    {0x208A, 0x208E, Cls::Other}, {0x20A0, 0x20C0, Cls::Other}, {0x20D0, 0x20F0, Cls::Other},
    // Letterlike symbols 2100-214F: the non-letters between the letter code points.
    {0x2100, 0x2101, Cls::Other}, {0x2103, 0x2106, Cls::Other}, {0x2108, 0x2109, Cls::Other},
    {0x2114, 0x2114, Cls::Other}, {0x2116, 0x2118, Cls::Other}, {0x211E, 0x2123, Cls::Other},
    {0x2125, 0x2125, Cls::Other}, {0x2127, 0x2127, Cls::Other}, {0x2129, 0x2129, Cls::Other},
    {0x212E, 0x212E, Cls::Other}, {0x213A, 0x213B, Cls::Other}, {0x2140, 0x2144, Cls::Other},
    {0x214A, 0x214D, Cls::Other}, {0x214F, 0x214F, Cls::Other},
    {0x2150, 0x2182, Cls::Number}, {0x2185, 0x2189, Cls::Number}, {0x218A, 0x245F, Cls::Other},
    {0x2460, 0x249B, Cls::Number}, {0x249C, 0x24E9, Cls::Other}, {0x24EA, 0x24FF, Cls::Number},
    {0x2500, 0x2775, Cls::Other}, {0x2776, 0x2793, Cls::Number}, {0x2794, 0x2BFF, Cls::Other},
    {0x2CF9, 0x2CFF, Cls::Other}, {0x2E00, 0x2E7F, Cls::Other},
    {0x3000, 0x3000, Cls::Space}, {0x3001, 0x3004, Cls::Other}, {0x3007, 0x3007, Cls::Number},
    {0x3008, 0x3020, Cls::Other}, {0x3021, 0x3029, Cls::Number}, {0x302A, 0x3030, Cls::Other},
    {0x3036, 0x3037, Cls::Other}, {0x3038, 0x303A, Cls::Number}, {0x303D, 0x303F, Cls::Other},
    {0x3099, 0x309C, Cls::Other}, {0x30A0, 0x30A0, Cls::Other}, {0x30FB, 0x30FB, Cls::Other},
    {0x3200, 0x321F, Cls::Other}, {0x3220, 0x3229, Cls::Number}, {0x322A, 0x3247, Cls::Other},
    {0x3248, 0x324F, Cls::Number}, {0x3250, 0x3250, Cls::Other}, {0x3251, 0x325F, Cls::Number},
    {0x3260, 0x327F, Cls::Other}, {0x3280, 0x3289, Cls::Number}, {0x328A, 0x32B0, Cls::Other},
    {0x32B1, 0x32BF, Cls::Number}, {0x32C0, 0x33FF, Cls::Other},
    {0xE000, 0xF8FF, Cls::Other},  // private use
    {0xFE00, 0xFE6F, Cls::Other},  // variation selectors, vertical/small punctuation forms
    {0xFEFF, 0xFEFF, Cls::Other}, {0xFF01, 0xFF0F, Cls::Other}, {0xFF10, 0xFF19, Cls::Number},
    {0xFF1A, 0xFF20, Cls::Other}, {0xFF3B, 0xFF40, Cls::Other}, {0xFF5B, 0xFF65, Cls::Other},
    {0xFFE0, 0xFFFF, Cls::Other},
    {0x1D7CE, 0x1D7FF, Cls::Number},  // mathematical digits
    {0x1F000, 0x1F0FF, Cls::Other}, {0x1F100, 0x1F10C, Cls::Number}, {0x1F10D, 0x1FAFF, Cls::Other},
    {0xE0000, 0xE01EF, Cls::Other},  // tags, variation selectors supplement
    {0xF0000, 0x10FFFF, Cls::Other},  // supplementary private use
};

Cls classify(uint32_t cp) {
    if (cp == INVALID_CP) return Cls::Other;
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return Cls::Letter;
        if (cp >= '0' && cp <= '9') return Cls::Number;
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return Cls::Space;
        return Cls::Other;
    }
    size_t lo = 0, hi = sizeof(NON_LETTER) / sizeof(NON_LETTER[0]);
    while (lo < hi) {  // first range with hi >= cp
        const size_t mid = (lo + hi) / 2;
        if (NON_LETTER[mid].hi < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < sizeof(NON_LETTER) / sizeof(NON_LETTER[0]) && NON_LETTER[lo].lo <= cp) {
        return NON_LETTER[lo].cls;
    }
    return Cls::Letter;
}

// ---------------------------------------------------------------------------
// (5) GPT-2 byte <-> unicode alphabet
// ---------------------------------------------------------------------------

// Bytes '!'..'~', 0xA1..0xAC and 0xAE..0xFF keep their own codepoint; the
// remaining 68 bytes, in increasing order, get U+0100, U+0101, ...
struct ByteAlphabet {
    uint32_t byte_to_cp[256];
    int16_t cp_to_byte[0x144];  // inverse; -1 where unused
    ByteAlphabet() {
        for (int16_t& v : cp_to_byte) v = -1;
        uint32_t next = 0x100;
        for (int b = 0; b < 256; ++b) {
            const bool keeps = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
            byte_to_cp[b] = keeps ? static_cast<uint32_t>(b) : next++;
            cp_to_byte[byte_to_cp[b]] = static_cast<int16_t>(b);
        }
    }
};

const ByteAlphabet& alphabet() {
    static const ByteAlphabet a;
    return a;
}

// Convert a token written in the byte alphabet to its raw bytes.
bool alphabet_to_bytes(const std::string& token, std::string* out) {
    out->clear();
    const ByteAlphabet& a = alphabet();
    size_t i = 0;
    while (i < token.size()) {
        size_t len;
        const uint32_t cp = utf8_decode(token, i, &len);
        if (cp == INVALID_CP || cp >= 0x144 || a.cp_to_byte[cp] < 0) return false;
        out->push_back(static_cast<char>(a.cp_to_byte[cp]));
        i += len;
    }
    return true;
}

std::string printable(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F) {
            out.push_back(static_cast<char>(c));
        } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02X", c);
            out += buf;
        }
    }
    return out;
}

bool read_file(const std::string& path, std::string* out, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open '" + path + "'";
        return false;
    }
    out->clear();
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad && err) *err = "error reading '" + path + "'";
    return !bad;
}

}  // namespace

// ---------------------------------------------------------------------------
// (3) Pre-tokenization scanner (GPT-2 pattern, see tokenizer.h)
// ---------------------------------------------------------------------------

std::vector<std::string_view> gpt2_pretokenize(std::string_view text) {
    std::vector<std::string_view> out;
    const size_t n = text.size();
    size_t i = 0;
    auto cls_at = [&](size_t p, size_t* len) { return classify(utf8_decode(text, p, len)); };

    while (i < n) {
        // Contractions: 's 't 're 've 'm 'll 'd (ASCII, case-sensitive).
        if (text[i] == '\'') {
            static const char* const SUFFIXES[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            size_t matched = 0;
            for (const char* suf : SUFFIXES) {
                const std::string_view sv(suf);
                if (text.substr(i + 1, sv.size()) == sv) {
                    matched = 1 + sv.size();
                    break;
                }
            }
            if (matched) {
                out.push_back(text.substr(i, matched));
                i += matched;
                continue;
            }
        }

        // ' '? (letters+ | numbers+ | other+). A U+0020 followed by a
        // non-space character joins that character's run. Otherwise
        // (space followed by space, or space at the end) fall through to
        // the whitespace rules.
        const size_t j = (text[i] == ' ' && i + 1 < n) ? i + 1 : i;
        size_t len;
        const Cls c = cls_at(j, &len);
        if (c != Cls::Space) {
            size_t k = j + len;
            while (k < n) {
                size_t l2;
                if (cls_at(k, &l2) != c) break;
                k += l2;
            }
            out.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }

        // Whitespace run starting at i: \s+(?!\S) | \s+
        size_t k = i, last_start = i;
        int count = 0;
        while (k < n) {
            size_t l2;
            if (cls_at(k, &l2) != Cls::Space) break;
            last_start = k;
            k += l2;
            ++count;
        }
        CHECK(count > 0, "gpt2_pretokenize: no rule matched at byte %zu", i);
        if (k < n && count >= 2) k = last_start;  // leave the last space for the next word
        out.push_back(text.substr(i, k - i));
        i = k;
    }
    return out;
}

// ---------------------------------------------------------------------------
// (4) Minimal JSON reader: { "string": non-negative integer, ... }
// ---------------------------------------------------------------------------

namespace {

struct JsonReader {
    const std::string& s;
    size_t p = 0;
    std::string err;

    bool fail(const std::string& msg) {
        err = "JSON error at byte " + std::to_string(p) + ": " + msg;
        return false;
    }
    void skip_ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
    }
    bool expect(char c) {
        skip_ws();
        if (p >= s.size() || s[p] != c) return fail(std::string("expected '") + c + "'");
        ++p;
        return true;
    }
    bool hex4(uint32_t* v) {
        if (p + 4 > s.size()) return fail("truncated \\u escape");
        *v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[p++];
            uint32_t d;
            if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') d = static_cast<uint32_t>(c - 'A' + 10);
            else return fail("bad hex digit in \\u escape");
            *v = (*v << 4) | d;
        }
        return true;
    }
    bool string(std::string* out) {
        skip_ws();
        if (p >= s.size() || s[p] != '"') return fail("expected string");
        ++p;
        out->clear();
        while (true) {
            if (p >= s.size()) return fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(s[p++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                out->push_back(static_cast<char>(c));
                continue;
            }
            if (p >= s.size()) return fail("unterminated escape");
            const char e = s[p++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(&cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate: need a low one
                        uint32_t lo = 0;
                        if (p + 2 > s.size() || s[p] != '\\' || s[p + 1] != 'u')
                            return fail("unpaired high surrogate");
                        p += 2;
                        if (!hex4(&lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return fail("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("unpaired low surrogate");
                    }
                    utf8_append(cp, out);
                    break;
                }
                default: return fail(std::string("invalid escape '\\") + e + "'");
            }
        }
    }
    bool integer(int64_t* v) {
        skip_ws();
        const size_t start = p;
        int64_t x = 0;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            if (x > (INT64_MAX - 9) / 10) return fail("integer too large");
            x = x * 10 + (s[p] - '0');
            ++p;
        }
        if (p == start) return fail("expected non-negative integer");
        if (p - start > 1 && s[start] == '0') return fail("leading zero in integer");
        if (p < s.size() && (s[p] == '.' || s[p] == 'e' || s[p] == 'E'))
            return fail("expected integer, got a number with fraction/exponent");
        *v = x;
        return true;
    }
};

}  // namespace

bool parse_json_string_int_object(const std::string& text,
                                  std::vector<std::pair<std::string, int64_t>>* out,
                                  std::string* err) {
    JsonReader r{text, 0, std::string()};
    out->clear();
    auto done = [&](bool ok) {
        if (!ok && err) *err = r.err;
        return ok;
    };
    if (!r.expect('{')) return done(false);
    r.skip_ws();
    if (r.p < text.size() && text[r.p] == '}') {
        ++r.p;
    } else {
        while (true) {
            std::string key;
            int64_t val;
            if (!r.string(&key) || !r.expect(':') || !r.integer(&val)) return done(false);
            out->emplace_back(std::move(key), val);
            r.skip_ws();
            if (r.p < text.size() && text[r.p] == ',') {
                ++r.p;
                continue;
            }
            if (!r.expect('}')) return done(false);
            break;
        }
    }
    r.skip_ws();
    if (r.p != text.size()) return done(r.fail("trailing content after object"));
    return true;
}

// ---------------------------------------------------------------------------
// (6) Tokenizer
// ---------------------------------------------------------------------------

bool Tokenizer::load(const std::string& vocab_json_path, const std::string& merges_txt_path,
                     std::string* err) {
    std::string vocab, merges;
    if (!read_file(vocab_json_path, &vocab, err) || !read_file(merges_txt_path, &merges, err))
        return false;
    std::string e;
    if (!load_from_memory(vocab, merges, &e)) {
        if (err) *err = vocab_json_path + " / " + merges_txt_path + ": " + e;
        return false;
    }
    return true;
}

bool Tokenizer::load_from_memory(const std::string& vocab_json, const std::string& merges_txt,
                                 std::string* err) {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    *this = Tokenizer();

    std::vector<std::pair<std::string, int64_t>> entries;
    std::string e;
    if (!parse_json_string_int_object(vocab_json, &entries, &e)) return fail("vocab.json: " + e);
    if (entries.empty()) return fail("vocab.json: empty vocabulary");
    int64_t max_id = 0;
    for (const auto& kv : entries) max_id = kv.second > max_id ? kv.second : max_id;
    if (max_id >= (int64_t(1) << 31)) return fail("vocab.json: id too large");
    id_bytes_.assign(static_cast<size_t>(max_id) + 1, std::string());
    id_present_.assign(static_cast<size_t>(max_id) + 1, false);
    for (const auto& kv : entries) {
        const uint32_t id = static_cast<uint32_t>(kv.second);
        std::string raw;
        if (!alphabet_to_bytes(kv.first, &raw))
            return fail("vocab.json: token for id " + std::to_string(id) +
                        " is not in the GPT-2 byte alphabet");
        if (id_present_[id]) return fail("vocab.json: duplicate id " + std::to_string(id));
        if (!bytes_to_id_.emplace(raw, id).second)
            return fail("vocab.json: duplicate token '" + printable(raw) + "'");
        id_present_[id] = true;
        id_bytes_[id] = std::move(raw);
    }
    for (int b = 0; b < 256; ++b) {
        const auto it = bytes_to_id_.find(std::string(1, static_cast<char>(b)));
        if (it == bytes_to_id_.end())
            return fail("vocab.json: missing single-byte token for byte " + std::to_string(b));
        byte_id_[b] = it->second;
    }

    // merges.txt: optional "#version..." header, then "A B" per line.
    size_t pos = 0;
    int line_no = 0;
    uint32_t rank = 0;
    while (pos < merges_txt.size()) {
        size_t nl = merges_txt.find('\n', pos);
        if (nl == std::string::npos) nl = merges_txt.size();
        std::string line = merges_txt.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line_no == 1 && line.rfind("#version", 0) == 0) continue;
        const std::string where = "merges.txt:" + std::to_string(line_no) + ": ";
        const size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size() ||
            line.find(' ', sp + 1) != std::string::npos)
            return fail(where + "expected 'A B'");
        std::string a, b;
        if (!alphabet_to_bytes(line.substr(0, sp), &a) || !alphabet_to_bytes(line.substr(sp + 1), &b))
            return fail(where + "token not in the GPT-2 byte alphabet");
        const int64_t ia = find_token(a), ib = find_token(b), iab = find_token(a + b);
        if (ia < 0 || ib < 0 || iab < 0)
            return fail(where + "merge '" + line + "' refers to a token missing from vocab.json");
        const uint64_t key = (static_cast<uint64_t>(ia) << 32) | static_cast<uint64_t>(ib);
        if (!merges_.emplace(key, Merge{rank, static_cast<uint32_t>(iab)}).second)
            return fail(where + "duplicate merge '" + line + "'");
        ++rank;
    }
    merge_count_ = rank;
    return true;
}

const std::string& Tokenizer::token_bytes(uint32_t id) const {
    CHECK(has_id(id), "Tokenizer: unknown token id %u (vocab_size %u)", id, vocab_size());
    return id_bytes_[id];
}

int64_t Tokenizer::find_token(std::string_view raw_bytes) const {
    const auto it = bytes_to_id_.find(std::string(raw_bytes));
    return it == bytes_to_id_.end() ? -1 : static_cast<int64_t>(it->second);
}

void Tokenizer::bpe(std::string_view word, std::vector<uint32_t>* out) {
    const std::string key(word);
    const auto hit = cache_.find(key);
    if (hit != cache_.end()) {
        out->insert(out->end(), hit->second.begin(), hit->second.end());
        return;
    }
    std::vector<uint32_t> syms, next;
    syms.reserve(word.size());
    for (char ch : word) syms.push_back(byte_id_[static_cast<unsigned char>(ch)]);

    while (syms.size() >= 2) {
        // Lowest-rank adjacent pair present in merges.
        uint32_t best_rank = UINT32_MAX, left = 0, right = 0, merged = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            const uint64_t k = (static_cast<uint64_t>(syms[i]) << 32) | syms[i + 1];
            const auto it = merges_.find(k);
            if (it != merges_.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                left = syms[i];
                right = syms[i + 1];
                merged = it->second.result;
            }
        }
        if (best_rank == UINT32_MAX) break;
        // Merge every non-overlapping occurrence, left to right.
        next.clear();
        for (size_t i = 0; i < syms.size();) {
            if (i + 1 < syms.size() && syms[i] == left && syms[i + 1] == right) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(syms[i]);
                i += 1;
            }
        }
        syms.swap(next);
    }
    out->insert(out->end(), syms.begin(), syms.end());
    cache_.emplace(key, std::move(syms));
}

std::vector<uint32_t> Tokenizer::encode(std::string_view text) {
    CHECK(!id_bytes_.empty(), "Tokenizer::encode called before load");
    std::vector<uint32_t> ids;
    for (std::string_view piece : gpt2_pretokenize(text)) bpe(piece, &ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out;
    for (uint32_t id : ids) out += token_bytes(id);
    return out;
}

}  // namespace tslm
