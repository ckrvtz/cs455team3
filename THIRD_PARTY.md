# Third-party sources

Every external library, code snippet, or closely adapted idea used in the
deliverable must be listed here before the task that introduced it is
considered done (CLAUDE.md §1.3–1.4). Libraries require team approval first.

| Name | URL | Version / commit | License | What was used | Where |
|---|---|---|---|---|---|
| xoshiro256** PRNG (Blackman & Vigna, "Scrambled linear pseudorandom number generators", ACM TOMS 2021) | https://prng.di.unimi.it/ | algorithm as published (reference C `xoshiro256starstar.c`) | Public domain (CC0) | Algorithm definition and constants (output `rotl(s1*5,7)*9`, shift 17, rotation 45); code written from the definition, not copied | `include/tslm/rng.h` |
| splitmix64 (Steele, Lea & Flood, OOPSLA 2014; constants from Vigna's `splitmix64.c`) | https://prng.di.unimi.it/splitmix64.c | as published | Public domain (CC0) | Seeding of xoshiro256** and the `mix64` finalizer constants; code written from the definition | `include/tslm/rng.h` |

No vendored code. Expected toolchain dependencies (not vendored): C++17 standard
library; OpenMPI (Phase C/D); CUDA toolkit (Phase B/D).
