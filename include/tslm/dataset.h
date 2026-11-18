// dataset.h: token-file DataLoader (ROADMAP §7.4, §4.5). Host-only.
//
// The token file (token_file.h) is one packed stream of stories, each ending
// in EOS; there is no cross-document masking. With sequence length T:
//   chunk i = tokens [i*T, i*T + T]   (T+1 tokens; consecutive chunks share
//             their boundary token)
//   x = chunk[0..T), y = chunk[1..T]  (next-token targets)
//   num_chunks = (num_tokens - 1) / T
// Epoch e visits the chunks in the order epoch_permutation(seed, e, n):
//   Fisher-Yates with Rng(seed, stream = 100 + e), for i = n-1 down to 1,
//   j = uniform_int(i + 1), swap(perm[i], perm[j]).
// Global batch gB = B * world_size. Step s of an epoch uses permutation
// entries [s*gB, (s+1)*gB); rank r takes rows [r*B, (r+1)*B) of that slice,
// so world_size ranks with micro-batch B together see exactly the batch a
// single process with batch gB would see (§4.5). steps_per_epoch =
// num_chunks / gB; the remaining chunks of each epoch are dropped.
//
// next_batch() fills HOST buffers x[B*T], y[B*T] (row-major [B,T], int32)
// and advances one step, rolling over to the next epoch when needed. The
// resume state is {epoch, step}, where step is the next step to be served.
// It describes the GLOBAL position, so it is the same on every rank and can
// be restored with any world size that keeps the same gB.
//
// The file is mmapped read-only; every token is checked against the header's
// vocab_size as it is copied out.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tslm {

struct LoaderState {
    uint64_t epoch = 0;
    uint64_t step = 0;  // next step within the epoch, in [0, steps_per_epoch]
};

// The epoch's chunk order (see above). out has n entries.
void epoch_permutation(uint64_t seed, uint64_t epoch, uint32_t n, uint32_t* out);

class DataLoader {
public:
    DataLoader() = default;
    ~DataLoader();
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // Map `path` and configure batching. Returns false with *err on any
    // problem: I/O error, bad header, size mismatch, B/T/rank/world out of
    // range, or fewer than gB chunks (steps_per_epoch would be 0).
    bool open(const std::string& path, int B, int T, uint64_t seed, int rank, int world_size,
              std::string* err);

    void next_batch(int32_t* x, int32_t* y);

    LoaderState state() const { return state_; }
    void set_state(const LoaderState& s);  // CHECKs step <= steps_per_epoch

    uint64_t epoch() const { return state_.epoch; }
    uint64_t steps_per_epoch() const { return steps_per_epoch_; }
    uint64_t num_tokens() const { return num_tokens_; }
    uint64_t num_chunks() const { return num_chunks_; }
    uint32_t vocab_size() const { return vocab_size_; }

private:
    void close();
    void ensure_permutation();

    void* map_ = nullptr;
    size_t map_bytes_ = 0;
    const unsigned char* tokens_ = nullptr;  // little-endian uint16 stream
    uint64_t num_tokens_ = 0, num_chunks_ = 0, steps_per_epoch_ = 0;
    uint32_t vocab_size_ = 0;
    int B_ = 0, T_ = 0, rank_ = 0, world_ = 1;
    uint64_t seed_ = 0;
    LoaderState state_;
    std::vector<uint32_t> perm_;
    uint64_t perm_epoch_ = UINT64_MAX;  // epoch perm_ was built for
};

}  // namespace tslm
