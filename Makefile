# TinyStories transformer LM: build for all four modes (ROADMAP §4.1, §5).
#
#   make                       release serial build  -> bin/train_serial, bin/<app>, ...
#   make test                  build + run every tests/test_*.cpp
#   make debug                 -O0 -g ASan/UBSan build, then run the tests under it
#   make test REAL=double      double-precision build (gradient checks)
#   make BACKEND=cuda          Phase B   | make COMM=mpi  Phase C | both: Phase D
#   make dist                  submission tarball (fails if Python files remain)
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
ifeq ($(filter $(MODE),release debug),)
$(error MODE must be release or debug (got '$(MODE)'))
endif

# ---- toolchain and flags
CXX      ?= g++
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
else
OPTFLAGS := -O0 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
LDSAN    := -fsanitize=address,undefined
endif

CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARN)
LDFLAGS  := $(LDSAN)
LDLIBS   :=

# ---- output locations
CFG_TAG   := $(MODE)-$(BACKEND)-$(COMM)-$(REAL)
BUILD_DIR := build/$(CFG_TAG)
# The default release float build puts apps in bin/ (as ROADMAP §4.1 names
# them); every other mode/precision gets its own bin/<mode>-<real>/.
ifeq ($(MODE)-$(REAL),release-float)
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
LIB      := $(BUILD_DIR)/libtslm.a

APP_SRCS := $(wildcard apps/*.cpp)
APP_BINS := $(foreach s,$(APP_SRCS),$(BIN_DIR)/$(if $(filter apps/train.cpp,$(s)),train_$(TRAIN_SUFFIX),$(basename $(notdir $(s)))))

TEST_SRCS := $(wildcard tests/test_*.cpp)
TEST_BINS := $(TEST_SRCS:tests/%.cpp=$(BUILD_DIR)/tests/%)

.PHONY: all test debug dist clean
# Keep intermediate objects (e.g. test .o files) so rebuilds stay incremental.
.SECONDARY:
.DEFAULT_GOAL := all

all: $(LIB) $(APP_BINS)

# Tests run from the repo root (golden files are referenced as tests/golden/...).
test: $(TEST_BINS)
	@pass=0; fail=0; failed=""; \
	for t in $(TEST_BINS); do \
	  echo "== $$t"; \
	  if ./$$t; then pass=$$((pass+1)); else fail=$$((fail+1)); failed="$$failed $$t"; fi; \
	done; \
	echo "== [$(CFG_TAG)] $$pass passed, $$fail failed"; \
	if [ $$fail -ne 0 ]; then echo "FAILED:$$failed"; exit 1; fi

debug:
	$(MAKE) MODE=debug all test

# ---- build rules
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

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
# build products; refuses to package if Python artifacts remain in the tree.
DIST_DIR     := dist
DIST_NAME    := tslm
DIST_EXCLUDE := ./.git ./dev_only ./data ./runs ./build ./bin ./$(DIST_DIR)

dist:
	@bad=$$(find . \( $(foreach d,$(DIST_EXCLUDE),-path $(d) -o) -false \) -prune -o \
	         -type f \( -name '*.py' -o -name '*.ipynb' -o -name 'requirements*.txt' \) -print); \
	if [ -n "$$bad" ]; then echo "make dist: Python files outside dev_only/:"; echo "$$bad"; exit 1; fi
	@mkdir -p $(DIST_DIR)
	tar -czf $(DIST_DIR)/$(DIST_NAME).tar.gz $(foreach d,$(DIST_EXCLUDE),--exclude=$(d)) \
	    --transform 's,^\.,$(DIST_NAME),' .
	@if tar -tzf $(DIST_DIR)/$(DIST_NAME).tar.gz | grep -E '(\.py|\.ipynb|/requirements[^/]*\.txt)$$|/dev_only/'; then \
	  echo "make dist: forbidden files in tarball"; rm -f $(DIST_DIR)/$(DIST_NAME).tar.gz; exit 1; fi
	@echo "wrote $(DIST_DIR)/$(DIST_NAME).tar.gz"

clean:
	rm -rf build bin $(DIST_DIR)

-include $(LIB_OBJS:.o=.d) $(TEST_BINS:=.d) $(APP_SRCS:%.cpp=$(BUILD_DIR)/%.d)
