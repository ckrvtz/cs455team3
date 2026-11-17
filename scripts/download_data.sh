#!/usr/bin/env bash
# download_data.sh: fetch the raw inputs for prepare_data (ROADMAP §7.1, P1.1).
#
#   scripts/download_data.sh            # into <repo>/data/
#   DATA_DIR=/scratch/me/tslm scripts/download_data.sh
#
# Downloads, at pinned Hugging Face revisions (see THIRD_PARTY.md):
#   TinyStories-train.txt, TinyStories-valid.txt   roneneldan/TinyStories (original, not V2)
#   vocab.json, merges.txt                         EleutherAI/gpt-neo-125m tokenizer data
# About 1.96 GB in total, almost all of it the train file.
#
# Idempotent: a file that already exists with the expected SHA-256 is left
# alone. Downloads go to <file>.part (resumed if interrupted) and are renamed
# into place only after the checksum matches, so a final file is never partial.
# A present file with the WRONG checksum is an error (it is never silently
# overwritten); delete it and rerun. Prints the SHA-256 of every file at the end.
#
# Requires: bash, curl, and sha256sum (Linux) or shasum (macOS).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"

HF="https://huggingface.co"
TS_REV="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"   # roneneldan/TinyStories, 2024-08-12
NEO_REV="21def0189f5705e2521767faed922f1f15e7d7db"  # EleutherAI/gpt-neo-125m, 2024-01-31

# name | url | sha256 | size in bytes
# Dataset checksums are the LFS sha256 published by Hugging Face. The
# tokenizer files are not in LFS: their sha256 was computed from files whose
# git blob ids matched the ones Hugging Face lists at NEO_REV.
FILES=(
  "TinyStories-train.txt|$HF/datasets/roneneldan/TinyStories/resolve/$TS_REV/TinyStories-train.txt|c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f|1924281556"
  "TinyStories-valid.txt|$HF/datasets/roneneldan/TinyStories/resolve/$TS_REV/TinyStories-valid.txt|94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4|19447282"
  "vocab.json|$HF/EleutherAI/gpt-neo-125m/resolve/$NEO_REV/vocab.json|03087853bc70c618b66e7c7a43e787d2db4c469416beac9a483e53dad1f72f27|898669"
  "merges.txt|$HF/EleutherAI/gpt-neo-125m/resolve/$NEO_REV/merges.txt|1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5|456318"
)

die() { echo "download_data.sh: error: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl not found"
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "neither sha256sum nor shasum found"
fi
size_of() { wc -c < "$1" | tr -d ' '; }

mkdir -p "$DATA_DIR"
SUMMARY=()
echo "data directory: $DATA_DIR"

for entry in "${FILES[@]}"; do
  IFS='|' read -r name url want_sha want_size <<< "$entry"
  dest="$DATA_DIR/$name"
  part="$dest.part"

  if [[ -f "$dest" ]]; then
    have_sha="$(sha256_of "$dest")"
    [[ "$have_sha" == "$want_sha" ]] || die "$dest exists but its sha256 is $have_sha, expected $want_sha. Delete it and rerun."
    SUMMARY+=("  $have_sha  $name")
    echo "  [cached]     $name"
    continue
  fi

  echo "  [download]   $name ($want_size bytes)"
  # -C - resumes an interrupted .part; --fail turns HTTP errors into failures.
  curl --fail --location --show-error --progress-bar --retry 5 --retry-delay 5 \
       --continue-at - --output "$part" "$url" \
    || die "download failed for $name ($url). A partial file is kept at $part; rerun to resume."

  have_size="$(size_of "$part")"
  have_sha="$(sha256_of "$part")"
  if [[ "$have_sha" != "$want_sha" ]]; then
    rm -f "$part"
    die "$name: checksum mismatch (got $have_sha, size $have_size; expected $want_sha, size $want_size). Removed the partial file; rerun."
  fi
  mv "$part" "$dest"
  SUMMARY+=("  $have_sha  $name")
  echo "  [verified]   $name"
done

echo
echo "SHA-256 (all verified against pinned values):"
printf "%s\n" "${SUMMARY[@]}"
