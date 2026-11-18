// checkpoint.h: training checkpoints (ROADMAP §9, §4.3). FROZEN FORMAT once
// committed: changes need the contract-change process (CLAUDE.md §2).
//
// File = 128-byte header + payload. All integers are little-endian.
//   offset size field
//        0    4 magic "TSCK"
//        4    4 u32 version (= CKPT_VERSION)
//        8    4 u32 real_bytes: element size of the payload (4 = float32, 8 = float64)
//       12    4 u32 header_bytes (= CKPT_HEADER_BYTES)
//       16   24 i32 x 6 ModelConfig: vocab_size, max_seq_len, n_layer, n_embd,
//                n_head, tie_embeddings
//       40    8 u64 step         last completed step (1-based)
//       48    8 u64 tokens_seen  global tokens consumed through `step`
//       56    8 u64 loader epoch  \ DataLoader state AFTER `step`, i.e. the
//       64    8 u64 loader step   / position of the next batch
//       72    8 u64 seed          data order depends on it
//       80    4 u32 global_batch  B * world_size  \ the loader position only
//       84    4 u32 seq_len       T                / means something for these
//       88    8 u64 num_params
//       96    8 u64 num_params_padded
//      104    8 u64 payload FNV-1a 64 (token_file.h) over all payload bytes
//      112   16 reserved, zero
//      128      payload: params, m, v; each num_params_padded elements of
//               real_bytes, little-endian IEEE, in the flat layout.h order
//               (padding included; it is always 0)
// The AdamW step count is `step` (adamw_update is called with step + 1 next).
// Nothing else is needed for an exact resume: the loader order is a pure
// function of (seed, epoch), and training uses no other randomness.
//
// Portability: the payload is the flat HOST layout, so every build (serial,
// CUDA, MPI) reads and writes the same files. A file whose real_bytes differs
// from the build's real_t is converted element by element on load (exact for
// float -> double; rounded for double -> float) and the loader logs that.
//
// I/O streams through a fixed 64 KB buffer with copy_d2h / copy_h2d, so
// saving inside the training loop allocates nothing and works on device
// pointers. Saves write <path>.tmp and rename() it, so an interrupted save
// never leaves a truncated file under the final name. Host-only.
#pragma once

#include <cstdint>
#include <string>

#include "tslm/config.h"
#include "tslm/dataset.h"
#include "tslm/model.h"

namespace tslm {

constexpr uint32_t CKPT_VERSION = 1;
constexpr uint32_t CKPT_HEADER_BYTES = 128;

struct CheckpointInfo {
    uint32_t real_bytes = 0;  // set by checkpoint_save / read from file
    ModelConfig model;
    uint64_t step = 0;
    uint64_t tokens_seen = 0;
    LoaderState loader;
    uint64_t seed = 0;
    uint32_t global_batch = 0;
    uint32_t seq_len = 0;
    uint64_t num_params = 0;
    uint64_t num_params_padded = 0;
    uint64_t payload_hash = 0;
};

// Write model->params, m and v with `info` (model / num_params* / real_bytes /
// payload_hash are filled in from the model). CHECKs every I/O step.
void checkpoint_save(const std::string& path, const Model* model, const CheckpointInfo& info);

// Read and validate the header only (magic, version, sizes, file length).
// Returns false with *err on any problem.
bool checkpoint_read_info(const std::string& path, CheckpointInfo* info, std::string* err);

// Load params, m and v into `model`. The checkpoint's ModelConfig and
// parameter counts must match the model exactly, and the payload hash must
// verify; otherwise returns false with *err (the model may then hold partly
// loaded values and must not be used).
bool checkpoint_load(const std::string& path, Model* model, CheckpointInfo* info, std::string* err);

}  // namespace tslm
