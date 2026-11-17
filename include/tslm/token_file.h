// token_file.h: on-disk token file format (ROADMAP §7.3). FROZEN CONTRACT.
//
// train.bin / val.bin = 32-byte header + num_tokens little-endian uint16 ids.
//   offset  size  field
//        0     4  magic "TSTK"
//        4     4  u32 version (= TOKEN_FILE_VERSION)
//        8     4  u32 vocab_size (every id is < vocab_size)
//       12     4  zero padding (aligns num_tokens; natural C layout)
//       16     8  u64 num_tokens
//       24     8  reserved, zero
// All integers are little-endian. The header is written byte by byte, so the
// file is identical on any host. The loader (P1.4) mmaps the file and reads
// tokens from offset 32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tslm {

constexpr size_t TOKEN_FILE_HEADER_BYTES = 32;
constexpr uint32_t TOKEN_FILE_VERSION = 1;

struct TokenFileHeader {
    uint32_t version = TOKEN_FILE_VERSION;
    uint32_t vocab_size = 0;
    uint64_t num_tokens = 0;
};

// Serialize into exactly TOKEN_FILE_HEADER_BYTES bytes.
void encode_token_header(const TokenFileHeader& h, unsigned char out[TOKEN_FILE_HEADER_BYTES]);

// Parse and validate (magic, version, zero padding/reserved, vocab_size >= 1).
bool decode_token_header(const unsigned char in[TOKEN_FILE_HEADER_BYTES], TokenFileHeader* h,
                         std::string* err);

// 64-bit FNV-1a, for content hashes in meta.txt (offset basis
// 0xcbf29ce484222325, prime 0x100000001b3). Feed bytes incrementally:
//   uint64_t h = FNV1A64_INIT; h = fnv1a64(h, data, n); ...
constexpr uint64_t FNV1A64_INIT = 0xcbf29ce484222325ull;
uint64_t fnv1a64(uint64_t h, const void* data, size_t n);

}  // namespace tslm
