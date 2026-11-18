// token_file.cpp: token file header codec and FNV-1a (see include/tslm/token_file.h).
#include "tslm/token_file.h"

namespace tslm {

namespace {

void put_u32(unsigned char* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
void put_u64(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
uint32_t get_u32(const unsigned char* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
    return v;
}
uint64_t get_u64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

}  // namespace

void encode_token_header(const TokenFileHeader& h, unsigned char out[TOKEN_FILE_HEADER_BYTES]) {
    for (size_t i = 0; i < TOKEN_FILE_HEADER_BYTES; ++i) out[i] = 0;
    out[0] = 'T';
    out[1] = 'S';
    out[2] = 'T';
    out[3] = 'K';
    put_u32(out + 4, h.version);
    put_u32(out + 8, h.vocab_size);
    put_u64(out + 16, h.num_tokens);
}

bool decode_token_header(const unsigned char in[TOKEN_FILE_HEADER_BYTES], TokenFileHeader* h,
                         std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = "token file header: " + m;
        return false;
    };
    if (in[0] != 'T' || in[1] != 'S' || in[2] != 'T' || in[3] != 'K') return fail("bad magic (expected TSTK)");
    h->version = get_u32(in + 4);
    h->vocab_size = get_u32(in + 8);
    h->num_tokens = get_u64(in + 16);
    if (h->version != TOKEN_FILE_VERSION)
        return fail("unsupported version " + std::to_string(h->version));
    if (h->vocab_size < 1 || h->vocab_size > 65536) return fail("vocab_size out of uint16 range");
    for (size_t i = 12; i < 16; ++i)
        if (in[i] != 0) return fail("nonzero padding");
    for (size_t i = 24; i < 32; ++i)
        if (in[i] != 0) return fail("nonzero reserved bytes");
    return true;
}

uint64_t fnv1a64(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

}  // namespace tslm
