// dataset.cpp: mmap-backed token DataLoader (see include/tslm/dataset.h).
#include "tslm/dataset.h"

#include "tslm/common.h"
#include "tslm/rng.h"
#include "tslm/token_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tslm {

void epoch_permutation(uint64_t seed, uint64_t epoch, uint32_t n, uint32_t* out) {
    for (uint32_t i = 0; i < n; ++i) out[i] = i;
    Rng rng(seed, 100 + epoch);  // ROADMAP §7.4: stream 100 + epoch
    for (uint32_t i = n; i-- > 1;) {
        const uint32_t j = static_cast<uint32_t>(rng.uniform_int(uint64_t(i) + 1));
        const uint32_t t = out[i];
        out[i] = out[j];
        out[j] = t;
    }
}

DataLoader::~DataLoader() { close(); }

void DataLoader::close() {
    if (map_) munmap(map_, map_bytes_);
    map_ = nullptr;
    map_bytes_ = 0;
    tokens_ = nullptr;
}

bool DataLoader::open(const std::string& path, int B, int T, uint64_t seed, int rank,
                      int world_size, std::string* err) {
    close();
    auto fail = [&](const std::string& m) {
        if (err) *err = "DataLoader(" + path + "): " + m;
        close();
        return false;
    };
    if (B < 1 || T < 1) return fail("B and T must be >= 1");
    if (world_size < 1 || rank < 0 || rank >= world_size) return fail("rank must be in [0, world_size)");

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return fail("cannot open");
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return fail("fstat failed");
    }
    const size_t bytes = static_cast<size_t>(st.st_size);
    if (bytes < TOKEN_FILE_HEADER_BYTES) {
        ::close(fd);
        return fail("file shorter than the 32-byte header");
    }
    void* m = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping stays valid after close
    if (m == MAP_FAILED) return fail("mmap failed");
    map_ = m;
    map_bytes_ = bytes;

    TokenFileHeader h;
    std::string e;
    if (!decode_token_header(static_cast<const unsigned char*>(map_), &h, &e)) return fail(e);
    if (bytes != TOKEN_FILE_HEADER_BYTES + 2 * h.num_tokens)
        return fail("file size " + std::to_string(bytes) + " does not match header num_tokens " +
                    std::to_string(h.num_tokens));
    tokens_ = static_cast<const unsigned char*>(map_) + TOKEN_FILE_HEADER_BYTES;
    num_tokens_ = h.num_tokens;
    vocab_size_ = h.vocab_size;
    B_ = B;
    T_ = T;
    rank_ = rank;
    world_ = world_size;
    seed_ = seed;
    num_chunks_ = num_tokens_ >= 1 ? (num_tokens_ - 1) / static_cast<uint64_t>(T) : 0;
    if (num_chunks_ > UINT32_MAX) return fail("too many chunks for a 32-bit permutation");
    const uint64_t gB = static_cast<uint64_t>(B) * static_cast<uint64_t>(world_size);
    steps_per_epoch_ = num_chunks_ / gB;
    if (steps_per_epoch_ == 0)
        return fail(std::to_string(num_chunks_) + " chunks of T=" + std::to_string(T) +
                    " cannot fill one global batch of " + std::to_string(gB));
    state_ = LoaderState{};
    perm_.clear();
    perm_epoch_ = UINT64_MAX;
    return true;
}

void DataLoader::ensure_permutation() {
    if (perm_epoch_ == state_.epoch) return;
    perm_.resize(num_chunks_);
    epoch_permutation(seed_, state_.epoch, static_cast<uint32_t>(num_chunks_), perm_.data());
    perm_epoch_ = state_.epoch;
}

void DataLoader::set_state(const LoaderState& s) {
    CHECK(tokens_ != nullptr, "DataLoader::set_state before open");
    CHECK(s.step <= steps_per_epoch_, "DataLoader::set_state: step %llu > steps_per_epoch %llu",
          static_cast<unsigned long long>(s.step), static_cast<unsigned long long>(steps_per_epoch_));
    state_ = s;
}

void DataLoader::next_batch(int32_t* x, int32_t* y) {
    CHECK(tokens_ != nullptr, "DataLoader::next_batch before open");
    if (state_.step >= steps_per_epoch_) {
        state_.epoch += 1;
        state_.step = 0;
    }
    ensure_permutation();
    const size_t T = static_cast<size_t>(T_);
    const uint64_t gB = static_cast<uint64_t>(B_) * static_cast<uint64_t>(world_);
    const uint64_t base = state_.step * gB + static_cast<uint64_t>(rank_) * static_cast<uint64_t>(B_);
    for (int b = 0; b < B_; ++b)
        copy_chunk(perm_[base + static_cast<uint64_t>(b)], x + static_cast<size_t>(b) * T,
                   y + static_cast<size_t>(b) * T);
    state_.step += 1;
}

void DataLoader::fixed_batch(uint64_t index, int32_t* x, int32_t* y) const {
    CHECK(tokens_ != nullptr, "DataLoader::fixed_batch before open");
    const size_t T = static_cast<size_t>(T_);
    const uint64_t gB = static_cast<uint64_t>(B_) * static_cast<uint64_t>(world_);
    CHECK((index + 1) * gB <= num_chunks_, "DataLoader::fixed_batch: batch %llu needs %llu chunks, file has %llu",
          static_cast<unsigned long long>(index), static_cast<unsigned long long>((index + 1) * gB),
          static_cast<unsigned long long>(num_chunks_));
    const uint64_t base = index * gB + static_cast<uint64_t>(rank_) * static_cast<uint64_t>(B_);
    for (int b = 0; b < B_; ++b)
        copy_chunk(base + static_cast<uint64_t>(b), x + static_cast<size_t>(b) * T, y + static_cast<size_t>(b) * T);
}

void DataLoader::copy_chunk(uint64_t chunk, int32_t* xr, int32_t* yr) const {
    const size_t T = static_cast<size_t>(T_);
    const unsigned char* src = tokens_ + 2 * chunk * T;  // T+1 tokens from here
    for (size_t t = 0; t <= T; ++t) {
        const int32_t tok = static_cast<int32_t>(src[2 * t] | (src[2 * t + 1] << 8));
        CHECK(static_cast<uint32_t>(tok) < vocab_size_,
              "DataLoader: token %d at position %llu >= vocab_size %u (corrupt file?)", tok,
              static_cast<unsigned long long>(chunk * T + t), vocab_size_);
        if (t < T) xr[t] = tok;
        if (t > 0) yr[t - 1] = tok;
    }
}

}  // namespace tslm
