// dump_init: write a model's init weights plus a fixed batch for golden tests
// (ROADMAP §10, P2.6).
//
//   bin/dump_init --out tests/golden/tiny_init.bin [--scale S] [config keys...]
// Config keys (config.h) set the model and the batch: e.g. --vocab_size 32
// --max_seq_len 8 --n_layer 2 --n_embd 16 --n_head 2 --tie_embeddings 1
// --batch_size 2 --seq_len 8 --seed 1337. Unknown keys are errors.
//
// The output (tensor_file.h format) holds:
//   config  i32 [8]  vocab_size, max_seq_len, n_layer, n_embd, n_head, tie, B, T
//   seed    i64 [1]; scale f64 [1]
//   tokens, targets i32 [B,T]   uniform ids from Rng(seed, GOLDEN_BATCH_STREAM)
//   param.<name> f32 [shape]    model_init_weights(seed), each value times
//                               scale (in real_t), stored as f32
// dev_only/gen_golden.py reads this file and writes the float64 PyTorch
// results. --scale > 1 makes logits non-uniform so attention/MLP errors
// show in the golden comparison.
#include "tslm/backend.h"
#include "tslm/common.h"
#include "tslm/config.h"
#include "tslm/model.h"
#include "tslm/rng.h"
#include "tslm/tensor_file.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tslm;

// Distinct from init streams (tensor index) and data streams (100 + epoch).
static constexpr uint64_t GOLDEN_BATCH_STREAM = 900000;

int main(int argc, char** argv) {
    Config cfg;
    std::string out, err;
    double scale = 1.0;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s'", arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "out") {
            out = val;
        } else if (key == "scale") {
            char* end = nullptr;
            errno = 0;
            scale = std::strtod(val.c_str(), &end);
            CHECK(errno == 0 && *end == '\0' && std::isfinite(scale) && scale > 0, "--scale: bad value '%s'",
                  val.c_str());
        } else {
            CHECK(config_set(&cfg, key, val, &err), "%s", err.c_str());
        }
    }
    CHECK(!out.empty(), "--out is required");
    CHECK(config_validate(cfg, &err), "%s", err.c_str());
    const ModelConfig& mc = cfg.model;
    const int B = cfg.train.batch_size, T = cfg.train.seq_len;

    Model* m = model_create(mc, B, T);
    std::printf("%s\n", model_summary(m).c_str());
    model_init_weights(m, cfg.train.seed);
    const ParamLayout& L = m->params_layout;
    std::vector<real_t> host(L.num_params_padded);
    copy_d2h(host.data(), m->params, host.size() * sizeof(real_t));

    TensorFile tf;
    const int32_t conf[8] = {mc.vocab_size, mc.max_seq_len, mc.n_layer, mc.n_embd, mc.n_head, mc.tie_embeddings, B, T};
    tf.add_i32("config", {8}, conf);
    const int64_t seed = static_cast<int64_t>(cfg.train.seed);
    tf.add_i64("seed", {1}, &seed);
    tf.add_f64("scale", {1}, &scale);

    Rng rng(cfg.train.seed, GOLDEN_BATCH_STREAM);
    std::vector<int32_t> tokens(size_t(B) * T), targets(size_t(B) * T);
    for (auto& t : tokens) t = static_cast<int32_t>(rng.uniform_int(uint64_t(mc.vocab_size)));
    for (auto& t : targets) t = static_cast<int32_t>(rng.uniform_int(uint64_t(mc.vocab_size)));
    tf.add_i32("tokens", {uint64_t(B), uint64_t(T)}, tokens.data());
    tf.add_i32("targets", {uint64_t(B), uint64_t(T)}, targets.data());

    const real_t s = static_cast<real_t>(scale);
    for (const TensorDesc& t : L.tensors) {
        std::vector<float> v(t.numel);
        for (size_t k = 0; k < t.numel; ++k) v[k] = static_cast<float>(host[t.offset + k] * s);
        std::vector<uint64_t> dims;
        for (int d = 0; d < t.ndim; ++d) dims.push_back(uint64_t(t.shape[d]));
        tf.add_f32("param." + t.name, dims, v.data());
    }
    CHECK(save_tensor_file(out, tf, &err), "%s", err.c_str());
    std::printf("wrote %s: %zu tensors (%zu params, B=%d T=%d, seed %llu, scale %g)\n", out.c_str(),
                tf.tensors.size(), L.num_params, B, T, static_cast<unsigned long long>(cfg.train.seed), scale);
    model_free(m);
    return 0;
}
