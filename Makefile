# TinyStories transformer LM: build for all four modes (ROADMAP §4.1, §5).
#
#   make                       release serial build  -> bin/train_serial, bin/<app>, ...
#   make test                  build + run every tests/test_*.cpp
#   make debug                 -O0 -g ASan/UBSan build, then run the tests under it
#   make test REAL=double      double-precision build (gradient checks)
#   make BACKEND=cuda          CUDA build (Phase B) -> bin/cuda-float/train_cuda, bin/cuda-float/<app>, ...
#   make BACKEND=cuda MODE=debug   nvcc -G and a device sync after every kernel launch
#   make BACKEND=cuda sanitize     run the tests under compute-sanitizer --tool memcheck
#   make COMM=mpi              Phase C | BACKEND=cuda COMM=mpi: Phase D (stubs until then)
#                              NVCC=, CUDA_ARCH= (default sm_50), MPICXX= select the tools
#   make dist                  submission tarball (fails if Python files remain)
#   make MODE=profile          dev-only gprof build (release flags + -pg) -> bin/profile-float/
#   make clean
#
# Selection is compile-time only: BACKEND/COMM/REAL become -D macros checked by
# include/tslm/build_config.h, which #errors on unsupported combinations.
# Each configuration builds into its own build/<tag>/ directory, so switching
# REAL or MODE never mixes object files.

BACKEND ?= cpu
COMM    ?= serial
REAL    ?= float
MODE    ?= release

# ---- validate values (combinations are checked by #error in build_config.h)
ifeq ($(filter $(BACKEND),cpu cuda),)
$(error BACKEND must be cpu or cuda (got '$(BACKEND)'))
endif
ifeq ($(filter $(COMM),serial mpi),)
$(error COMM must be serial or mpi (got '$(COMM)'))
endif
ifeq ($(filter $(REAL),float double),)
$(error REAL must be float or double (got '$(REAL)'))
endif
ifeq ($(filter $(MODE),release debug profile),)
$(error MODE must be release, debug or profile (got '$(MODE)'))
endif

# ---- toolchain and flags
CXX      ?= g++
NVCC     ?= nvcc
MPICXX   ?= mpicxx

# Phase B/C/D toolchains (P7.5). BACKEND=cuda compiles src/*/*_cuda.cu with
# $(NVCC); COMM=mpi compiles everything with the MPI wrapper $(MPICXX). A
# missing tool stops make here with one message, before anything compiles.
# With the tools present, COMM=mpi still stops at the stub and the guards in
# include/tslm/build_config.h until Phases C and D are implemented.
ifneq ($(if $(MAKECMDGOALS),$(filter-out clean dist,$(MAKECMDGOALS)),all),)
ifeq ($(BACKEND),cuda)
ifeq ($(shell command -v $(NVCC) 2>/dev/null),)
$(error BACKEND=cuda needs the CUDA compiler '$(NVCC)' (CUDA toolkit 12.x; 13.x cannot target sm_50), which was not found; set NVCC=/path/to/nvcc)
endif
endif
ifeq ($(COMM),mpi)
ifeq ($(shell command -v $(MPICXX) 2>/dev/null),)
$(error COMM=mpi needs the MPI compiler wrapper '$(MPICXX)' (OpenMPI), which was not found; set MPICXX=/path/to/mpicxx. The MPI build itself is Phase C and not implemented yet: see src/comm/comm_mpi.cpp)
endif
CXX := $(MPICXX)
endif
endif

CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wpedantic
CPPFLAGS := -Iinclude -MMD -MP \
            -DTSLM_BACKEND_$(shell echo $(BACKEND) | tr a-z A-Z) \
            -DTSLM_COMM_$(shell echo $(COMM) | tr a-z A-Z) \
            -DTSLM_REAL_$(shell echo $(REAL) | tr a-z A-Z) \
            -DTSLM_BUILD_MODE='"$(MODE)"'

ifeq ($(MODE),release)
# -O3 -march=native only; -ffast-math is forbidden (CLAUDE.md §1.5).
OPTFLAGS := -O3 -march=native
LDSAN    :=
else ifeq ($(MODE),profile)
# Dev-time profiling only (P7.2, scripts/profile.sh): the release flags plus
# gprof instrumentation. Never used by make test or make dist.
OPTFLAGS := -O3 -march=native -pg
LDSAN    := -pg
else ifeq ($(BACKEND),cuda)
# CUDA debug: no ASan/UBSan (they are not mixed with the CUDA runtime; use
# `make BACKEND=cuda sanitize` for device memory errors).
OPTFLAGS := -O0 -g -fno-omit-frame-pointer
LDSAN    :=
else
OPTFLAGS := -O0 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
LDSAN    := -fsanitize=address,undefined
endif

CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARN)
LDFLAGS  := $(LDSAN)
LDLIBS   :=

# ---- CUDA (BACKEND=cuda; ROADMAP_CUDA.md §4.2, D-B1, D-B4)
# nvcc compiles src/*/*_cuda.cu; host .cpp files keep $(CXX) and the flags
# above. No --use_fast_math (D-B4). Release keeps -lineinfo for profilers;
# MODE=debug uses -G (device debug info, which overrides -lineinfo: nvcc warns
# if both are given) and TSLM_CUDA_DEBUG_SYNC, which makes every kernel
# launch check synchronise (src/ops/cuda_common.cuh). Apps and tests are
# linked by $(CXX) against the toolkit's shared cuBLAS and runtime, with an
# rpath to the toolkit's lib64 so the binaries run without LD_LIBRARY_PATH.
CUDA_ARCH ?= sm_50
comma     := ,
ifeq ($(BACKEND),cuda)
ifeq ($(MODE),profile)
$(error MODE=profile is the serial gprof build; profile CUDA builds with nsys/ncu (docs/PROFILE_CUDA.md))
endif
NVCCFLAGS := -std=c++17 -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra
ifeq ($(MODE),debug)
NVCCFLAGS += -O0 -g -G -DTSLM_CUDA_DEBUG_SYNC
else
NVCCFLAGS += -O3 -lineinfo
endif
CUDA_HOME   ?= $(patsubst %/bin/,%,$(dir $(realpath $(shell command -v $(NVCC) 2>/dev/null))))
CUDA_LIBDIR := $(wildcard $(CUDA_HOME)/lib64)
LDLIBS      += $(if $(CUDA_LIBDIR),-L$(CUDA_LIBDIR) -Wl$(comma)-rpath$(comma)$(CUDA_LIBDIR)) -lcublas -lcudart
NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null | grep -o 'release [0-9.]*, V[0-9.]*')
endif
COMPUTE_SANITIZER ?= compute-sanitizer

# ---- output locations
CFG_TAG   := $(MODE)-$(BACKEND)-$(COMM)-$(REAL)
BUILD_DIR := build/$(CFG_TAG)

# Code-generation target stamp: with -march=native the objects are specific
# to the CPU that compiled them, and make cannot see a CPU change by itself
# (this container moved between CPUs once; docs/DECISIONS.md). The stamp
# holds a checksum of the compiler's predefined macros under the actual flags
# (instruction-set and tuning macros such as __AVX512F__ and __tune_<cpu>__
# appear there). EVERY object depends on it (the %.o rule below), so a
# different CPU, compiler or flag set rebuilds everything. Defined here, before
# that rule, because make expands rule prerequisites when it reads them.
TARGET_ID    := $(shell $(CXX) $(CXXFLAGS) -dM -E -x c++ - </dev/null 2>/dev/null | cksum | cut -d' ' -f1)
# CUDA builds add the nvcc version, CUDA_ARCH and the nvcc flags, so a
# different toolkit or GPU target also rebuilds everything (ROADMAP_CUDA §4.2).
ifeq ($(BACKEND),cuda)
TARGET_ID    := $(TARGET_ID) $(shell $(NVCC) --version 2>/dev/null | cksum | cut -d' ' -f1) $(NVCCFLAGS)
endif
TARGET_STAMP := $(BUILD_DIR)/target.stamp
# The default release float CPU build puts apps in bin/ (as ROADMAP §4.1
# names them); every other mode/precision gets its own bin/<mode>-<real>/.
# CUDA builds never share a directory with the CPU apps (ROADMAP_CUDA §4.2):
# bin/cuda-<real>/ for release, bin/cuda-<mode>-<real>/ otherwise.
ifeq ($(BACKEND),cuda)
BIN_DIR := bin/cuda-$(if $(filter release,$(MODE)),,$(MODE)-)$(REAL)
else ifeq ($(MODE)-$(REAL),release-float)
BIN_DIR := bin
else
BIN_DIR := bin/$(MODE)-$(REAL)
endif

ifeq ($(BACKEND)-$(COMM),cpu-serial)
TRAIN_SUFFIX := serial
else ifeq ($(BACKEND)-$(COMM),cuda-serial)
TRAIN_SUFFIX := cuda
else ifeq ($(BACKEND)-$(COMM),cpu-mpi)
TRAIN_SUFFIX := mpi
else
TRAIN_SUFFIX := mpi_cuda
endif

# ---- sources
# Library = every src/*/*.cpp. Files named *_cpu/*_cuda/*_serial/*_mpi.cpp are
# implementation-specific and only compiled when their BACKEND/COMM is selected.
VARIANT_PATTERNS := %_cpu.cpp %_cuda.cpp %_serial.cpp %_mpi.cpp
ALL_LIB_SRCS     := $(wildcard src/*/*.cpp)
LIB_SRCS := $(filter-out $(VARIANT_PATTERNS),$(ALL_LIB_SRCS)) \
            $(filter %_$(BACKEND).cpp %_$(COMM).cpp,$(ALL_LIB_SRCS))
LIB_OBJS := $(LIB_SRCS:%.cpp=$(BUILD_DIR)/%.o)
# CUDA sources (*_cuda.cu) only in BACKEND=cuda builds, compiled by $(NVCC).
CU_SRCS  := $(if $(filter cuda,$(BACKEND)),$(wildcard src/*/*_cuda.cu))
LIB_OBJS += $(CU_SRCS:%.cu=$(BUILD_DIR)/%.o)
LIB      := $(BUILD_DIR)/libtslm.a

APP_SRCS := $(wildcard apps/*.cpp)
APP_BINS := $(foreach s,$(APP_SRCS),$(BIN_DIR)/$(if $(filter apps/train.cpp,$(s)),train_$(TRAIN_SUFFIX),$(basename $(notdir $(s)))))

TEST_SRCS := $(wildcard tests/test_*.cpp)
TEST_BINS := $(TEST_SRCS:tests/%.cpp=$(BUILD_DIR)/tests/%)

.PHONY: all test debug sanitize dist clean
# Keep intermediate objects (e.g. test .o files) so rebuilds stay incremental.
.SECONDARY:
.DEFAULT_GOAL := all

all: $(LIB) $(APP_BINS)

# Tests run from the repo root (golden files are referenced as tests/golden/...).
test: $(TEST_BINS)
	@mkdir -p $(BUILD_DIR)/tests/tmp; export TSLM_TEST_TMPDIR=$(BUILD_DIR)/tests/tmp; \
	pass=0; fail=0; failed=""; \
	for t in $(TEST_BINS); do \
	  echo "== $$t"; \
	  if ./$$t; then pass=$$((pass+1)); else fail=$$((fail+1)); failed="$$failed $$t"; fi; \
	done; \
	echo "== [$(CFG_TAG)] $$pass passed, $$fail failed"; \
	if [ $$fail -ne 0 ]; then echo "FAILED:$$failed"; exit 1; fi

debug:
	$(MAKE) MODE=debug all test

# CUDA only: every test binary under compute-sanitizer memcheck (ROADMAP_CUDA
# §4.2; racecheck/synccheck on the equivalence test come with B0.4/B5.3).
sanitize: $(TEST_BINS)
	@if [ "$(BACKEND)" != cuda ]; then echo "make sanitize needs BACKEND=cuda (CPU builds: make debug)"; exit 1; fi
	@command -v $(COMPUTE_SANITIZER) >/dev/null || { echo "make sanitize: '$(COMPUTE_SANITIZER)' not found (set COMPUTE_SANITIZER=)"; exit 1; }
	@mkdir -p $(BUILD_DIR)/tests/tmp; export TSLM_TEST_TMPDIR=$(BUILD_DIR)/tests/tmp; \
	pass=0; fail=0; failed=""; \
	for t in $(TEST_BINS); do \
	  echo "== $$t (memcheck)"; \
	  if $(COMPUTE_SANITIZER) --tool memcheck --error-exitcode 99 ./$$t; then pass=$$((pass+1)); else fail=$$((fail+1)); failed="$$failed $$t"; fi; \
	done; \
	echo "== [$(CFG_TAG) memcheck] $$pass passed, $$fail failed"; \
	if [ $$fail -ne 0 ]; then echo "FAILED:$$failed"; exit 1; fi

# ---- build rules
$(BUILD_DIR)/%.o: %.cpp $(TARGET_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# nvcc gets the same -I/-D selection flags and its own NVCCFLAGS (above);
# -MT names the object in the .d file (nvcc would write only its basename).
$(BUILD_DIR)/%.o: %.cu $(TARGET_STAMP)
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) -MT $@ $(NVCCFLAGS) -c $< -o $@

# ---- build facts for runs/<name>/manifest.txt (src/train/manifest.cpp only).
# The git hash ("-dirty" if tracked files differ from HEAD, "unknown" outside a
# git checkout) and the compiler flags go to that one object, so a new commit
# rebuilds only it. build_info.txt is rewritten only when the facts change and
# manifest.o depends on it.
GIT_HASH    := $(shell h=$$(git rev-parse --short=12 HEAD 2>/dev/null) || h=unknown; \
                 if [ "$$h" != unknown ] && ! git diff --quiet HEAD -- 2>/dev/null; then h=$$h-dirty; fi; echo $$h)
BUILD_INFO  := $(GIT_HASH) | $(CXX) $(CXXFLAGS)$(if $(NVCCFLAGS), | $(NVCC) $(NVCC_VERSION) $(NVCCFLAGS))
BUILD_STAMP := $(BUILD_DIR)/build_info.txt

# Skipped when the only goals are clean/dist (no build happens then).
ifneq ($(if $(MAKECMDGOALS),$(filter-out clean dist,$(MAKECMDGOALS)),all),)
$(shell mkdir -p $(BUILD_DIR); [ "$$(cat $(BUILD_STAMP) 2>/dev/null)" = '$(BUILD_INFO)' ] || printf '%s' '$(BUILD_INFO)' > $(BUILD_STAMP))
$(shell [ "$$(cat $(TARGET_STAMP) 2>/dev/null)" = '$(TARGET_ID)' ] || printf '%s' '$(TARGET_ID)' > $(TARGET_STAMP))
endif
MANIFEST_OBJ := $(BUILD_DIR)/src/train/manifest.o
$(MANIFEST_OBJ): $(BUILD_STAMP)
$(MANIFEST_OBJ): CPPFLAGS += -DTSLM_GIT_HASH='"$(GIT_HASH)"' -DTSLM_CXX='"$(CXX)"' -DTSLM_CXXFLAGS='"$(CXXFLAGS)"'
ifeq ($(BACKEND),cuda)
$(MANIFEST_OBJ): CPPFLAGS += -DTSLM_NVCC='"$(NVCC) $(NVCC_VERSION)"' -DTSLM_NVCCFLAGS='"$(NVCCFLAGS)"'
endif

$(LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	ar rcs $@ $^

$(BIN_DIR)/train_$(TRAIN_SUFFIX): $(BUILD_DIR)/apps/train.o $(LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/%: $(BUILD_DIR)/apps/%.o $(LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/%: $(BUILD_DIR)/tests/%.o $(LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# ---- submission tarball (ROADMAP §5): excludes dev_only/, data/, runs/ and
# build products, plus every *.pdf (the reference papers/proposal are not part
# of the submission); refuses to package if Python artifacts remain in the tree.
DIST_DIR     := dist
DIST_NAME    := tslm
DIST_EXCLUDE := ./.git ./dev_only ./data ./runs ./build ./bin ./$(DIST_DIR)
DIST_EXCLUDE_PATTERNS := '*.pdf'

dist:
	@bad=$$(find . \( $(foreach d,$(DIST_EXCLUDE),-path $(d) -o) -false \) -prune -o \
	         -type f \( -name '*.py' -o -name '*.ipynb' -o -name 'requirements*.txt' \) -print); \
	if [ -n "$$bad" ]; then echo "make dist: Python files outside dev_only/:"; echo "$$bad"; exit 1; fi
	@mkdir -p $(DIST_DIR)
	tar -czf $(DIST_DIR)/$(DIST_NAME).tar.gz $(foreach d,$(DIST_EXCLUDE),--exclude=$(d)) \
	    $(foreach p,$(DIST_EXCLUDE_PATTERNS),--exclude=$(p)) --transform 's,^\.,$(DIST_NAME),' .
	@if tar -tzf $(DIST_DIR)/$(DIST_NAME).tar.gz | grep -E '(\.py|\.ipynb|\.pdf|/requirements[^/]*\.txt)$$|/dev_only/'; then \
	  echo "make dist: forbidden files in tarball"; rm -f $(DIST_DIR)/$(DIST_NAME).tar.gz; exit 1; fi
	@echo "wrote $(DIST_DIR)/$(DIST_NAME).tar.gz"

clean:
	rm -rf build bin $(DIST_DIR)

-include $(LIB_OBJS:.o=.d) $(TEST_BINS:=.d) $(APP_SRCS:%.cpp=$(BUILD_DIR)/%.d)
