// train.cpp: training app (ROADMAP §9). Built as bin/train_serial (and
// train_cuda / train_mpi / train_mpi_cuda in later phases).
//
//   bin/train_serial --config configs/smoke.cfg [--key value ...]
//
// Writes runs/<run_name>/{manifest.txt,log.csv}; see include/tslm/train.h.
#include "tslm/backend.h"
#include "tslm/comm.h"
#include "tslm/config.h"
#include "tslm/train.h"

int main(int argc, char** argv) {
    tslm::comm_init(&argc, &argv);
    tslm::dev_init(0);  // Phase D: the node-local rank
    const tslm::Config cfg = tslm::config_from_args_or_die(argc, argv);
    tslm::train_run(cfg);
    tslm::comm_finalize();
    return 0;
}
