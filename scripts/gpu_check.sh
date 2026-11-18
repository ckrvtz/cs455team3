#!/usr/bin/env bash
# gpu_check.sh: B0.1 machine verification (ROADMAP_CUDA.md §2.1, §8 B0.1).
# Run on the Phase B GPU machine (D-B1: Quadro M2000M, sm_50, nvcc 12.9):
#
#   scripts/gpu_check.sh
#
# Builds scripts/gpu_check.cu with the project's planned CUDA flags
# (CLAUDE.md §3), runs it, and writes everything to
# build/gpu_check/report.txt: date, kernel, CPU, nvidia-smi, the nvidia-smi
# query CSV, nvcc --version, compiler output, and the check program's output.
# Exit status 0 only if the build is warning-free and the check prints
# "RESULT: PASS". Environment: NVCC (default nvcc), CUDA_ARCH (default sm_50).
set -euo pipefail
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_50}
OUT=build/gpu_check
REPORT=$OUT/report.txt
mkdir -p "$OUT"
: > "$REPORT"
say() { echo "$@" | tee -a "$REPORT"; }
section() { say ""; say "== $1"; }
run() { "$@" 2>&1 | tee -a "$REPORT"; }

command -v "$NVCC" >/dev/null || { echo "gpu_check: '$NVCC' not found (set NVCC=/path/to/nvcc)" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "gpu_check: nvidia-smi not found (is the NVIDIA driver installed?)" >&2; exit 1; }

section "machine"
say "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "uname: $(uname -srm)"
say "cpu: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
say "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

section "nvidia-smi"
run nvidia-smi
section "nvidia-smi query"
run nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap,pstate --format=csv

section "nvcc --version"
run "$NVCC" --version

section "build (CUDA_ARCH=$CUDA_ARCH)"
CMD=("$NVCC" -std=c++17 -O3 -arch="$CUDA_ARCH" -lineinfo -Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra
     scripts/gpu_check.cu -o "$OUT/gpu_check" -lcublas)
say "${CMD[*]}"
set +e
BUILD_LOG=$("${CMD[@]}" 2>&1)
BUILD_RC=$?
set -e
[ -n "$BUILD_LOG" ] && say "$BUILD_LOG"
if [ "$BUILD_RC" -ne 0 ]; then say "build: FAILED (exit $BUILD_RC)"; exit 1; fi
if [ -n "$BUILD_LOG" ]; then say "build: compiler output above (the project requires a warning-free build)"; else say "build: ok, no compiler output"; fi

section "check program"
set +e
"$OUT/gpu_check" 2>&1 | tee -a "$REPORT"
CHECK_RC=${PIPESTATUS[0]}
set -e

say ""
say "report: $REPORT"
[ "$CHECK_RC" -eq 0 ] && [ -z "$BUILD_LOG" ]
