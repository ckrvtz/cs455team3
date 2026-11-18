// eval.h: validation loss and perplexity (ROADMAP §9, §2.4, P4.4). Host-side
// driver; the numerics are model_forward (no backward, no optimizer).
//
// The validation set is FIXED: batches 0 .. n_batches-1 of the validation
// file in file order (DataLoader::fixed_batch), i.e. the first
// n_batches * gB chunks, the same at every evaluation and in every build.
// Rank r evaluates rows [r*B, (r+1)*B) of each batch.
//
//   val_loss = mean over all n_batches * gB * T positions of the
//              cross-entropy = (sum over ranks and batches of the batch-mean
//              loss) / (n_batches * world_size)
// That equality holds because every rank-batch has exactly B*T positions.
// Batch means are summed in double, in batch order; ranks are combined with
// comm_allreduce_sum_double, so every rank returns the same value.
//   val_ppl = exp(val_loss)
//
// Side effects: overwrites the model's activation arena (like any forward)
// and the caller's token buffers; params, grads, m and v are untouched.
#pragma once

#include <cstdint>

#include "tslm/dataset.h"
#include "tslm/model.h"

namespace tslm {

// x_host/y_host: host buffers, x_dev/y_dev: device buffers, each
// val.batch_size() * val.seq_len() ids. CHECKs n_batches >= 1, that the file
// holds n_batches * gB chunks, and that the result is finite.
double evaluate(Model* model, const DataLoader& val, int n_batches, int32_t* x_host, int32_t* y_host,
                int32_t* x_dev, int32_t* y_dev);

}  // namespace tslm
