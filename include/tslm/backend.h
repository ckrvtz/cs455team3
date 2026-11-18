// backend.h: device-memory shim (ROADMAP §4.2, CLAUDE.md §4.3).
//
// ALL tensor memory (params, grads, AdamW m/v, activation arena, device-side
// token buffers) is allocated and moved through these functions. model/,
// train/ and optim/ never call malloc/new/memcpy/memset on tensors. In the
// CUDA build (Phase B) the same calls become cudaMalloc/cudaMemcpy/...; the
// pointers are then device pointers and calling code is unchanged.
//
// Semantics (identical in every backend):
//   dev_malloc(bytes)   aborts (CHECK) on failure; never returns nullptr for
//                       bytes > 0. The result is aligned to DEV_ALIGN bytes.
//                       Contents are uninitialized. dev_malloc(0) returns nullptr.
//   dev_free(p)         p may be nullptr.
//   dev_memset(p,v,n)   sets n bytes to (unsigned char)v.
//   copy_h2d/d2h/d2d    copy n bytes; regions must not overlap.
//   dev_sync()          waits for all outstanding device work (no-op on CPU).
//                       PhaseTimer calls it before reading the clock.
//   dev_name()          human-readable device name for manifests/benchmarks.
//   dev_init(local_rank) selects and initialises the device for this process
//                       (CUDA: device local_rank % device_count; CPU: no device
//                       work). Optional with one device: the first dev_* call
//                       then uses device 0. If called, it must come before any
//                       other dev_* call, at most once, with local_rank >= 0.
//                       Apps call dev_init(0) (Phase D: the node-local rank).
//   dev_peak_bytes()    high-water mark of the bytes requested through
//                       dev_malloc and not yet freed, since process start.
//                       bench reports it as peak_mem_MB in CUDA builds.
// (dev_init and dev_peak_bytes were added for Phase B/D: docs/DECISIONS.md,
// 2026-11-18, D-B8.)
// The *_d2h/h2d naming is used even on CPU so that call sites already state
// the direction that matters in the CUDA build.
//
// Must stay nvcc-compatible plain C++17.
#pragma once

#include <cstddef>

namespace tslm {

constexpr size_t DEV_ALIGN = 64;  // matches the 64-byte tensor offsets in layout.h

void* dev_malloc(size_t bytes);
void dev_free(void* p);
void dev_memset(void* p, int value, size_t bytes);
void copy_h2d(void* dst_dev, const void* src_host, size_t bytes);
void copy_d2h(void* dst_host, const void* src_dev, size_t bytes);
void copy_d2d(void* dst_dev, const void* src_dev, size_t bytes);
void dev_sync();
const char* dev_name();
void dev_init(int local_rank);
size_t dev_peak_bytes();

}  // namespace tslm
