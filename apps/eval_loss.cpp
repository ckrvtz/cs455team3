// eval_loss: mean cross-entropy of a checkpoint on fixed batches of a token
// file (forward only).
//
//   bin/eval_loss --ckpt <checkpoint> --data <token file> [--batches 20]
//                 [--batch_size 32] [--seq_len <model max_seq_len>]
//
// Uses exactly the validation pass of training (include/tslm/eval.h): the
// first --batches batches of the file in FILE ORDER (no shuffling), each
// batch_size x seq_len, mean loss over all positions. With the run's own
// val.bin and the training defaults, it reproduces the run's val_loss. With
// train.bin it gives the final model's loss on fixed training data (used for
// the P5.2 train/val gap; docs/RUNS.md). Prints loss, perplexity and tokens.
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "tslm/backend.h"
#include "tslm/checkpoint.h"
#include "tslm/comm.h"
#include "tslm/common.h"
#include "tslm/dataset.h"
#include "tslm/eval.h"
#include "tslm/logging.h"
#include "tslm/model.h"

using namespace tslm;

namespace {
int parse_int_or_die(const std::string& key, const std::string& v) {
    errno = 0;
    char* end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    CHECK(!v.empty() && errno == 0 && *end == '\0' && x >= 1 && x <= 1000000, "--%s: expected a positive integer, got '%s'",
          key.c_str(), v.c_str());
    return static_cast<int>(x);
}
}  // namespace

int main(int argc, char** argv) {
    comm_init(&argc, &argv);
    dev_init(0);  // Phase D: the node-local rank
    std::string ckpt, data;
    int batches = 20, B = 32, T = 0;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        CHECK(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '%s' (expected --key value)",
              arg.c_str());
        CHECK(i + 1 < argc, "missing value for '%s'", arg.c_str());
        const std::string key = arg.substr(2), val = argv[i + 1];
        if (key == "ckpt") ckpt = val;
        else if (key == "data") data = val;
        else if (key == "batches") batches = parse_int_or_die(key, val);
        else if (key == "batch_size") B = parse_int_or_die(key, val);
        else if (key == "seq_len") T = parse_int_or_die(key, val);
        else CHECK(false, "unknown option '--%s'", key.c_str());
    }
    CHECK(!ckpt.empty() && !data.empty(), "--ckpt <checkpoint> and --data <token file> are required");

    std::string err;
    CheckpointInfo info;
    CHECK(checkpoint_read_info(ckpt, &info, &err), "%s", err.c_str());
    if (T == 0) T = info.model.max_seq_len;
    CHECK(T <= info.model.max_seq_len, "--seq_len %d exceeds the model's max_seq_len %d", T, info.model.max_seq_len);
    Model* model = model_create(info.model, B, T);
    CHECK(checkpoint_load(ckpt, model, &info, &err), "%s", err.c_str());

    DataLoader loader;
    CHECK(loader.open(data, B, T, info.seed, comm_rank(), comm_size(), &err), "%s", err.c_str());
    CHECK(loader.vocab_size() == static_cast<uint32_t>(info.model.vocab_size),
          "%s has vocab_size %u but the checkpoint has %d", data.c_str(), loader.vocab_size(), info.model.vocab_size);
    const uint64_t gB = uint64_t(B) * uint64_t(comm_size());
    CHECK(uint64_t(batches) * gB <= loader.num_chunks(), "--batches %d needs %llu chunks; %s has %llu", batches,
          static_cast<unsigned long long>(uint64_t(batches) * gB), data.c_str(),
          static_cast<unsigned long long>(loader.num_chunks()));

    const size_t ntok = size_t(B) * size_t(T);
    std::vector<int32_t> xh(ntok), yh(ntok);
    int32_t* xd = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    int32_t* yd = static_cast<int32_t*>(dev_malloc(ntok * sizeof(int32_t)));
    const double loss = evaluate(model, loader, batches, xh.data(), yh.data(), xd, yd);
    if (comm_rank() == 0)
        log_info("%s on %s: first %d batches of %d x %d in file order (%llu tokens): loss %.8f, ppl %.4f (step %llu)",
                 ckpt.c_str(), data.c_str(), batches, B, T,
                 static_cast<unsigned long long>(uint64_t(batches) * gB * uint64_t(T)), loss, std::exp(loss),
                 static_cast<unsigned long long>(info.step));
    dev_free(xd);
    dev_free(yd);
    model_free(model);
    comm_finalize();
    return 0;
}
