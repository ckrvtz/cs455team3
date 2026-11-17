// comm_serial.cpp: single-process implementation of comm.h (COMM=serial).
// Rank 0 of 1; every collective leaves the data unchanged (the average or
// sum over one rank is the value itself). The init/finalize lifecycle is
// still enforced, so call-order bugs show up in the serial build before
// they would hang an MPI job.
#include "tslm/comm.h"

namespace tslm {

namespace {
enum class State { Uninit, Active, Finalized };
State g_state = State::Uninit;

void require_active(const char* fn) {
    CHECK(g_state == State::Active, "%s called %s", fn,
          g_state == State::Uninit ? "before comm_init" : "after comm_finalize");
}
}  // namespace

void comm_init(int* argc, char*** argv) {
    (void)argc;
    (void)argv;
    CHECK(g_state == State::Uninit, "comm_init called twice");
    g_state = State::Active;
}

int comm_rank() {
    require_active("comm_rank");
    return 0;
}

int comm_size() {
    require_active("comm_size");
    return 1;
}

void comm_broadcast(real_t* data, size_t n, int root) {
    require_active("comm_broadcast");
    CHECK(root == 0, "comm_broadcast: root %d out of range for size 1", root);
    CHECK(n == 0 || data != nullptr, "comm_broadcast: null data with n=%zu", n);
}

void comm_allreduce_avg(real_t* data, size_t n) {
    require_active("comm_allreduce_avg");
    CHECK(n == 0 || data != nullptr, "comm_allreduce_avg: null data with n=%zu", n);
}

void comm_allreduce_sum_double(double* data, int n) {
    require_active("comm_allreduce_sum_double");
    CHECK(n >= 0, "comm_allreduce_sum_double: negative n=%d", n);
    CHECK(n == 0 || data != nullptr, "comm_allreduce_sum_double: null data with n=%d", n);
}

void comm_barrier() { require_active("comm_barrier"); }

void comm_finalize() {
    require_active("comm_finalize");
    g_state = State::Finalized;
}

}  // namespace tslm
