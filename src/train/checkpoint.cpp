// checkpoint.cpp: see include/tslm/checkpoint.h for the format.
#include "tslm/checkpoint.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/logging.h"
#include "tslm/token_file.h"  // fnv1a64

namespace tslm {

namespace {

constexpr size_t CHUNK_BYTES = 64 * 1024;  // streaming buffer (stack)

bool host_is_little_endian() {
    const uint16_t one = 1;
    unsigned char b[2];
    std::memcpy(b, &one, 2);
    return b[0] == 1;
}

void put_u32(unsigned char* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
void put_u64(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
uint32_t get_u32(const unsigned char* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i);
    return v;
}
uint64_t get_u64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}

void encode_header(const CheckpointInfo& c, unsigned char h[CKPT_HEADER_BYTES]) {
    std::memset(h, 0, CKPT_HEADER_BYTES);
    std::memcpy(h, "TSCK", 4);
    put_u32(h + 4, CKPT_VERSION);
    put_u32(h + 8, c.real_bytes);
    put_u32(h + 12, CKPT_HEADER_BYTES);
    const int32_t mc[6] = {c.model.vocab_size, c.model.max_seq_len, c.model.n_layer,
                           c.model.n_embd, c.model.n_head, c.model.tie_embeddings};
    for (int i = 0; i < 6; ++i) put_u32(h + 16 + 4 * i, static_cast<uint32_t>(mc[i]));
    put_u64(h + 40, c.step);
    put_u64(h + 48, c.tokens_seen);
    put_u64(h + 56, c.loader.epoch);
    put_u64(h + 64, c.loader.step);
    put_u64(h + 72, c.seed);
    put_u32(h + 80, c.global_batch);
    put_u32(h + 84, c.seq_len);
    put_u64(h + 88, c.num_params);
    put_u64(h + 96, c.num_params_padded);
    put_u64(h + 104, c.payload_hash);
}

bool decode_header(const unsigned char h[CKPT_HEADER_BYTES], CheckpointInfo* c, std::string* why) {
    if (std::memcmp(h, "TSCK", 4) != 0) return *why = "bad magic (not a checkpoint)", false;
    const uint32_t version = get_u32(h + 4);
    if (version != CKPT_VERSION)
        return *why = "unsupported version " + std::to_string(version), false;
    c->real_bytes = get_u32(h + 8);
    if (c->real_bytes != 4 && c->real_bytes != 8)
        return *why = "bad real_bytes " + std::to_string(c->real_bytes), false;
    if (get_u32(h + 12) != CKPT_HEADER_BYTES) return *why = "bad header_bytes", false;
    int32_t mc[6];
    for (int i = 0; i < 6; ++i) mc[i] = static_cast<int32_t>(get_u32(h + 16 + 4 * i));
    c->model.vocab_size = mc[0];
    c->model.max_seq_len = mc[1];
    c->model.n_layer = mc[2];
    c->model.n_embd = mc[3];
    c->model.n_head = mc[4];
    c->model.tie_embeddings = mc[5];
    c->step = get_u64(h + 40);
    c->tokens_seen = get_u64(h + 48);
    c->loader.epoch = get_u64(h + 56);
    c->loader.step = get_u64(h + 64);
    c->seed = get_u64(h + 72);
    c->global_batch = get_u32(h + 80);
    c->seq_len = get_u32(h + 84);
    c->num_params = get_u64(h + 88);
    c->num_params_padded = get_u64(h + 96);
    c->payload_hash = get_u64(h + 104);
    for (size_t i = 112; i < CKPT_HEADER_BYTES; ++i)
        if (h[i] != 0) return *why = "nonzero reserved header bytes", false;
    if (c->num_params > c->num_params_padded) return *why = "num_params > num_params_padded", false;
    return true;
}

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

}  // namespace

void checkpoint_save(const std::string& path, const Model* model, const CheckpointInfo& info_in) {
    CHECK(host_is_little_endian(), "checkpoints assume a little-endian host");
    CheckpointInfo info = info_in;
    info.real_bytes = sizeof(real_t);
    info.model = model->cfg;
    info.num_params = model->params_layout.num_params;
    info.num_params_padded = model->params_layout.num_params_padded;
    info.payload_hash = FNV1A64_INIT;

    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    CHECK(f, "cannot create %s: %s", tmp.c_str(), std::strerror(errno));
    unsigned char hdr[CKPT_HEADER_BYTES];
    encode_header(info, hdr);  // hash patched in below
    CHECK(std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr), "write to %s failed", tmp.c_str());

    unsigned char buf[CHUNK_BYTES];
    const size_t per_chunk = CHUNK_BYTES / sizeof(real_t);
    const real_t* blobs[3] = {model->params, model->m, model->v};
    for (const real_t* blob : blobs) {
        for (size_t off = 0; off < info.num_params_padded; off += per_chunk) {
            const size_t n = std::min(per_chunk, size_t(info.num_params_padded) - off);
            const size_t bytes = n * sizeof(real_t);
            copy_d2h(buf, blob + off, bytes);
            info.payload_hash = fnv1a64(info.payload_hash, buf, bytes);
            CHECK(std::fwrite(buf, 1, bytes, f) == bytes, "write to %s failed", tmp.c_str());
        }
    }
    put_u64(hdr + 104, info.payload_hash);
    CHECK(std::fseek(f, 104, SEEK_SET) == 0 && std::fwrite(hdr + 104, 1, 8, f) == 8, "write to %s failed",
          tmp.c_str());
    CHECK(std::fflush(f) == 0, "flush of %s failed", tmp.c_str());
    CHECK(std::fclose(f) == 0, "close of %s failed", tmp.c_str());
    CHECK(std::rename(tmp.c_str(), path.c_str()) == 0, "cannot rename %s to %s: %s", tmp.c_str(), path.c_str(),
          std::strerror(errno));
}

bool checkpoint_read_info(const std::string& path, CheckpointInfo* info, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open " + path + ": " + std::strerror(errno));
    unsigned char hdr[CKPT_HEADER_BYTES];
    const bool got = std::fread(hdr, 1, sizeof(hdr), f) == sizeof(hdr);
    long size = -1;
    if (got && std::fseek(f, 0, SEEK_END) == 0) size = std::ftell(f);
    std::fclose(f);
    if (!got) return fail(err, path + ": file shorter than the checkpoint header");
    std::string why;
    if (!decode_header(hdr, info, &why)) return fail(err, path + ": " + why);
    const uint64_t want = CKPT_HEADER_BYTES + 3 * info->num_params_padded * uint64_t(info->real_bytes);
    if (size < 0 || uint64_t(size) != want)
        return fail(err, path + ": file is " + std::to_string(size) + " bytes, header implies " +
                             std::to_string(want) + " (truncated or corrupt)");
    return true;
}

bool checkpoint_load(const std::string& path, Model* model, CheckpointInfo* info, std::string* err) {
    CHECK(host_is_little_endian(), "checkpoints assume a little-endian host");
    if (!checkpoint_read_info(path, info, err)) return false;
    const ModelConfig& a = info->model;
    const ModelConfig& b = model->cfg;
    if (a.vocab_size != b.vocab_size || a.max_seq_len != b.max_seq_len || a.n_layer != b.n_layer ||
        a.n_embd != b.n_embd || a.n_head != b.n_head || a.tie_embeddings != b.tie_embeddings) {
        char m[256];
        std::snprintf(m, sizeof(m),
                      "%s: model config differs from the run's (V=%d T=%d L=%d C=%d NH=%d tied=%d vs V=%d T=%d L=%d "
                      "C=%d NH=%d tied=%d)",
                      path.c_str(), a.vocab_size, a.max_seq_len, a.n_layer, a.n_embd, a.n_head, a.tie_embeddings,
                      b.vocab_size, b.max_seq_len, b.n_layer, b.n_embd, b.n_head, b.tie_embeddings);
        return fail(err, m);
    }
    if (info->num_params != model->params_layout.num_params ||
        info->num_params_padded != model->params_layout.num_params_padded)
        return fail(err, path + ": parameter counts differ from the model's layout");
    if (info->real_bytes != sizeof(real_t))
        log_info("checkpoint %s holds %u-byte reals; converting to this build's %zu-byte real_t", path.c_str(),
                 info->real_bytes, sizeof(real_t));

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open " + path + ": " + std::strerror(errno));
    if (std::fseek(f, CKPT_HEADER_BYTES, SEEK_SET) != 0) {
        std::fclose(f);
        return fail(err, path + ": seek failed");
    }
    // Elements per chunk: bounded by both the file element size and real_t.
    const size_t rb = info->real_bytes;
    const size_t per_chunk = CHUNK_BYTES / 8;
    unsigned char buf[CHUNK_BYTES];
    real_t conv[CHUNK_BYTES / 8];
    uint64_t hash = FNV1A64_INIT;
    real_t* blobs[3] = {model->params, model->m, model->v};
    for (real_t* blob : blobs) {
        for (size_t off = 0; off < info->num_params_padded; off += per_chunk) {
            const size_t n = std::min(per_chunk, size_t(info->num_params_padded) - off);
            if (std::fread(buf, 1, n * rb, f) != n * rb) {
                std::fclose(f);
                return fail(err, path + ": read failed");
            }
            hash = fnv1a64(hash, buf, n * rb);
            if (rb == sizeof(real_t)) {
                copy_h2d(blob + off, buf, n * rb);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    if (rb == 4) {
                        float x;
                        std::memcpy(&x, buf + 4 * i, 4);
                        conv[i] = static_cast<real_t>(x);
                    } else {
                        double x;
                        std::memcpy(&x, buf + 8 * i, 8);
                        conv[i] = static_cast<real_t>(x);
                    }
                }
                copy_h2d(blob + off, conv, n * sizeof(real_t));
            }
        }
    }
    std::fclose(f);
    if (hash != info->payload_hash) return fail(err, path + ": payload checksum mismatch (corrupt file)");
    return true;
}

}  // namespace tslm
