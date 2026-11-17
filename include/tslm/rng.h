// rng.h: deterministic host-side random numbers (ROADMAP §4.6, CLAUDE.md §4.11).
//
// Generator: xoshiro256** (Blackman & Vigna), 256-bit state, 64-bit output.
// Seeding: splitmix64 (Steele, Lea & Flood; constants as published by Vigna).
// Both are implemented here from their published definitions; see
// THIRD_PARTY.md. std::mt19937 / std::normal_distribution / rand() are never
// used, because their output is implementation-defined or global.
//
// Rng(seed, stream) gives an independent sequence for every (seed, stream)
// pair. ROADMAP stream assignments: parameter tensor i -> stream i (init order
// independent); data shuffle for epoch e -> stream 100 + e; generation
// sampling uses its own stream.
//
// Derivation of the 256-bit state from (seed, stream):
//   key  = mix64(seed) ^ mix64(stream * GOLDEN + STREAM_SALT)
//   s[i] = splitmix64_next(&key), i = 0..3
// mix64 is the splitmix64 output finalizer, a bijection on 64-bit values. So
// for a fixed seed, distinct streams always give distinct keys, and for a
// fixed stream, distinct seeds always do.
//
// normal() is Box-Muller computed in double. It uses two uniforms per sample
// and keeps no cached second value, so the generator's whole state is the
// four xoshiro words.
//
// Host-only. Not thread-safe: use one Rng per consumer.
#pragma once

#include <cmath>
#include <cstdint>

#include "tslm/common.h"

namespace tslm {

constexpr uint64_t SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15ull;
// Arbitrary odd constant; it keeps stream 0 from reducing to mix64(0).
constexpr uint64_t RNG_STREAM_SALT = 0xD1B54A32D192ED03ull;

inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// splitmix64 output finalizer (bijective 64-bit mix).
inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One splitmix64 step: advance *state by the golden gamma, return its mix.
inline uint64_t splitmix64_next(uint64_t* state) {
    *state += SPLITMIX_GOLDEN;
    return mix64(*state);
}

class Rng {
public:
    explicit Rng(uint64_t seed, uint64_t stream = 0) {
        uint64_t key = mix64(seed) ^ mix64(stream * SPLITMIX_GOLDEN + RNG_STREAM_SALT);
        for (uint64_t& w : s_) w = splitmix64_next(&key);
        // An all-zero state is the one fixed point of xoshiro. Four
        // consecutive splitmix outputs cannot all be zero (mix64 is a
        // bijection and the inputs are distinct), so this check never fires.
        CHECK((s_[0] | s_[1] | s_[2] | s_[3]) != 0, "Rng: all-zero state");
    }

    // Raw state, for known-answer tests against the bare xoshiro256** definition.
    static Rng from_state(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
        CHECK((s0 | s1 | s2 | s3) != 0, "Rng::from_state: all-zero state is invalid");
        Rng r;
        r.s_[0] = s0;
        r.s_[1] = s1;
        r.s_[2] = s2;
        r.s_[3] = s3;
        return r;
    }

    // xoshiro256**: output = rotl(s1 * 5, 7) * 9, then the linear state update.
    uint64_t next_u64() {
        const uint64_t out = rotl64(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl64(s_[3], 45);
        return out;
    }

    // Uniform double in [0, 1) with 53 random bits (exact multiples of 2^-53).
    double uniform() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

    // Uniform integer in [0, n), n >= 1, with no modulo bias: draws below
    // 2^64 mod n are rejected, so every residue is equally likely.
    uint64_t uniform_int(uint64_t n) {
        CHECK(n >= 1, "Rng::uniform_int: n must be >= 1 (got %llu)",
              static_cast<unsigned long long>(n));
        const uint64_t threshold = (0 - n) % n;  // == 2^64 mod n
        for (;;) {
            const uint64_t x = next_u64();
            if (x >= threshold) return x % n;
        }
    }

    // Standard normal via Box-Muller: sqrt(-2 ln u1) * cos(2*pi*u2), with
    // u1 in (0, 1] so the log is finite.
    double normal() {
        const double u1 = 1.0 - uniform();
        const double u2 = uniform();
        constexpr double TWO_PI = 6.283185307179586476925286766559;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
    }

    double normal(double mean, double stddev) { return mean + stddev * normal(); }

private:
    Rng() : s_{0, 0, 0, 0} {}
    uint64_t s_[4];
};

}  // namespace tslm
