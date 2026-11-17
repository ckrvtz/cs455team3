# Third-party sources

Every external library, code snippet, or closely adapted idea used in the
deliverable must be listed here before the task that introduced it is
considered done (CLAUDE.md §1.3–1.4). Libraries require team approval first.

| Name | URL | Version / commit | License | What was used | Where |
|---|---|---|---|---|---|
| xoshiro256** PRNG (Blackman & Vigna, "Scrambled linear pseudorandom number generators", ACM TOMS 2021) | https://prng.di.unimi.it/ | algorithm as published (reference C `xoshiro256starstar.c`) | Public domain (CC0) | Algorithm definition and constants (output `rotl(s1*5,7)*9`, shift 17, rotation 45); code written from the definition, not copied | `include/tslm/rng.h` |
| splitmix64 (Steele, Lea & Flood, OOPSLA 2014; constants from Vigna's `splitmix64.c`) | https://prng.di.unimi.it/splitmix64.c | as published | Public domain (CC0) | Seeding of xoshiro256** and the `mix64` finalizer constants; code written from the definition | `include/tslm/rng.h` |
| TinyStories dataset (Eldan & Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", arXiv:2305.07759, 2023) | https://huggingface.co/datasets/roneneldan/TinyStories | revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; files `TinyStories-train.txt` (sha256 `c5cf5e22…401f`), `TinyStories-valid.txt` (sha256 `94e43181…38b4`) | CDLA-Sharing-1.0 | Training and validation text (data only; not redistributed; fetched into gitignored `data/`) | `scripts/download_data.sh` |
| GPT-Neo tokenizer data files (EleutherAI GPT-Neo 125M; the vocabulary and merges are the GPT-2 byte-level BPE of Radford et al., 2019) | https://huggingface.co/EleutherAI/gpt-neo-125m | revision `21def0189f5705e2521767faed922f1f15e7d7db`; `vocab.json` (sha256 `03087853…2f27`), `merges.txt` (sha256 `1ce16647…adc5`) | MIT | `vocab.json` and `merges.txt` read as data by our own BPE implementation (§7.2); no tokenizer code used | `scripts/download_data.sh` |

No vendored code. Expected toolchain dependencies (not vendored): C++17 standard
library; OpenMPI (Phase C/D); CUDA toolkit (Phase B/D).
