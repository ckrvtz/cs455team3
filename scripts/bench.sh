#!/usr/bin/env bash
# bench.sh: run a ROADMAP §12 reference workload N times (default 3) with
# bin/bench, appending one row per run to bench/<version>_<workload>.csv.
#
#   scripts/bench.sh W-A [runs]     small, 50 steps, B=32, T=256
#   scripts/bench.sh W-B [runs]     base,  20 steps, B=32, T=256
#   scripts/bench.sh W-C [runs]     smoke, 200 steps, B=4,  T=32
#
# Environment: BIN (default bin), VERSION (default serial), OUT_DIR (default
# bench), DATA (default data/train.bin). Run on an otherwise idle machine
# from a clean release build (make clean && make), as ROADMAP §12 requires.
set -euo pipefail
W=${1:?usage: scripts/bench.sh W-A|W-B|W-C [runs]}
RUNS=${2:-3}
BIN=${BIN:-bin}
VERSION=${VERSION:-serial}
OUT_DIR=${OUT_DIR:-bench}
DATA=${DATA:-data/train.bin}
case "$W" in
  W-A) ARGS=(--config configs/small.cfg --max_steps 50 --batch_size 32) ;;
  W-B) ARGS=(--config configs/base.cfg --max_steps 20 --batch_size 32) ;;
  W-C) ARGS=(--config configs/smoke.cfg --max_steps 200 --batch_size 4) ;;
  *) echo "unknown workload '$W' (W-A, W-B or W-C)" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"
for i in $(seq 1 "$RUNS"); do
  echo "== $W run $i/$RUNS"
  "$BIN/bench" "${ARGS[@]}" --train_data "$DATA" --version "$VERSION" --out "$OUT_DIR/${VERSION}_${W}.csv"
done
