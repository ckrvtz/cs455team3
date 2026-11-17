// timer.h: wall-clock phase timing (ROADMAP §2.4, §4.2, §9).
//
// PhaseTimer accumulates milliseconds per training-step phase. The phases
// match the log.csv / bench columns: data, fwd, bwd, comm, opt.
//   t.begin(Phase::Fwd); ...; t.end(Phase::Fwd);
// Both begin() and end() call dev_sync() before reading the clock (a no-op on
// CPU). Asynchronous device work (Phase B) is then charged to the phase that
// launched it, and the call sites need no change. Only one phase may be open
// at a time: phases partition the step, and nesting would double-count.
//
// Clock: std::chrono::steady_clock (monotonic). Host-only.
#pragma once

#include <chrono>

#include "tslm/backend.h"
#include "tslm/common.h"

namespace tslm {

enum class Phase : int { Data = 0, Fwd, Bwd, Comm, Opt, Count };

constexpr int NUM_PHASES = static_cast<int>(Phase::Count);

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Data: return "data";
        case Phase::Fwd: return "fwd";
        case Phase::Bwd: return "bwd";
        case Phase::Comm: return "comm";
        case Phase::Opt: return "opt";
        case Phase::Count: break;
    }
    return "?";
}

// Seconds on the monotonic clock since an arbitrary fixed origin.
inline double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

class PhaseTimer {
public:
    PhaseTimer() { reset(); }

    // Zero all accumulators (e.g. at the start of each logged interval).
    void reset() {
        CHECK(open_ < 0, "PhaseTimer::reset while phase '%s' is open",
              phase_name(static_cast<Phase>(open_)));
        for (double& v : ms_) v = 0.0;
    }

    void begin(Phase p) {
        check_phase(p);
        CHECK(open_ < 0, "PhaseTimer::begin('%s') while '%s' is still open", phase_name(p),
              phase_name(static_cast<Phase>(open_)));
        dev_sync();
        open_ = static_cast<int>(p);
        t0_ = now_seconds();
    }

    void end(Phase p) {
        check_phase(p);
        CHECK(open_ == static_cast<int>(p), "PhaseTimer::end('%s') but open phase is '%s'",
              phase_name(p), open_ < 0 ? "<none>" : phase_name(static_cast<Phase>(open_)));
        dev_sync();
        ms_[open_] += (now_seconds() - t0_) * 1e3;
        open_ = -1;
    }

    // Accumulated milliseconds for one phase, and the sum over all phases.
    double ms(Phase p) const {
        check_phase(p);
        return ms_[static_cast<int>(p)];
    }
    double total_ms() const {
        double s = 0.0;
        for (double v : ms_) s += v;
        return s;
    }

private:
    static void check_phase(Phase p) {
        const int i = static_cast<int>(p);
        CHECK(i >= 0 && i < NUM_PHASES, "PhaseTimer: invalid phase %d", i);
    }

    double ms_[NUM_PHASES];
    double t0_ = 0.0;
    int open_ = -1;  // index of the open phase, or -1
};

}  // namespace tslm
