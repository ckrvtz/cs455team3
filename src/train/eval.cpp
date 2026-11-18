// eval.cpp: validation pass. See include/tslm/eval.h.
#include "tslm/eval.h"

#include <cmath>

#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/common.h"

namespace tslm {

double evaluate(Model* model, const DataLoader& val, int n_batches, int32_t* x_host, int32_t* y_host,
                int32_t* x_dev, int32_t* y_dev) {
    CHECK(n_batches >= 1, "evaluate: n_batches %d must be >= 1", n_batches);
    const int B = val.batch_size(), T = val.seq_len();
    const size_t bytes = size_t(B) * size_t(T) * sizeof(int32_t);
    double sum = 0.0;
    for (int i = 0; i < n_batches; ++i) {
        val.fixed_batch(static_cast<uint64_t>(i), x_host, y_host);
        copy_h2d(x_dev, x_host, bytes);
        copy_h2d(y_dev, y_host, bytes);
        sum += model_forward(model, x_dev, y_dev, B, T);
    }
    comm_allreduce_sum_double(&sum, 1);
    const double loss = sum / (double(n_batches) * double(comm_size()));
    CHECK(std::isfinite(loss), "evaluate: non-finite validation loss %g", loss);
    return loss;
}

}  // namespace tslm
