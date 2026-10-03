CUDA_HOME ?= /usr/local/cuda
.DEFAULT_GOAL := all
NVCC ?= $(CUDA_HOME)/bin/nvcc
CC ?= gcc
CUDA_ARCH ?= native

BUILD_DIR ?= ../../target/native/cuda
TARGET ?= $(BUILD_DIR)/libcubbs_bbs_cuda.so
SHA256_SMOKE ?= $(BUILD_DIR)/sha256_smoke
H2S_SMOKE ?= $(BUILD_DIR)/hash_to_scalar_smoke
PREPARE_B_SMOKE ?= $(BUILD_DIR)/prepare_b_smoke
H2S_BENCH ?= $(BUILD_DIR)/hash_to_scalar_bench
MSM_ROUTE_BENCH ?= $(BUILD_DIR)/msm_route_bench

NVCCFLAGS ?= -O3 --std=c++17 -Xcompiler -fPIC -arch=$(CUDA_ARCH)
CXXFLAGS ?= -O2 -std=c++17
# Audited default: SPPARK v0.1.15, commit
# 17278d74295392f9813f009300b257a688422b7a. Record any override in releases.
SPPARK_ROOT ?= ../../ref/implementations/supranational-sppark
# Historical flag name retained for build compatibility. It now enables only
# SPPARK field/curve headers used by cuBBS-Cred kernels; no SPPARK MSM object is
# compiled or linked.
CUBBS_BBS_ENABLE_SPPARK_MSM ?= 1
# Optional third-party Pippenger reference routes 7/8. Disabled in normal
# production builds; enable only for explicit comparative benchmarks.
CUBBS_BBS_ENABLE_SPPARK_REFERENCE ?= 0
# Audited default: BLST 0.3.16, commit
# e7f90de551e8df682f3cc99067d204d8b90d27ad. Override only with an exact,
# recorded source revision. Prefer the explicit audited checkout; the Cargo
# cache fallback is for already-materialized sources only.
BLST_VERSION ?= 0.3.16
BLST_CHECKOUT_SRC ?= ../../ref/implementations/supranational-blst/src
BLST_CARGO_SRC := $(firstword $(sort $(wildcard $(HOME)/.cargo/registry/src/*/blst-$(BLST_VERSION)/blst/src)))
BLST_SRC ?= $(if $(wildcard $(BLST_CHECKOUT_SRC)/server.c),$(BLST_CHECKOUT_SRC),$(BLST_CARGO_SRC))
ifeq ($(strip $(BLST_SRC)),)
$(error BLST $(BLST_VERSION) source not found; see docs/build-run-http-benchmark.zh-CN.md for pinned checkouts or set BLST_SRC to an audited source tree)
endif
BLST_ROOT ?= $(patsubst %/src,%,$(BLST_SRC))
BLST_BINDINGS ?= $(BLST_ROOT)/bindings
BLST_OBJ_DIR ?= $(BUILD_DIR)/blst
BLST_SERVER_OBJ ?= $(BLST_OBJ_DIR)/server.o
BLST_ASSEMBLY_OBJ ?= $(BLST_OBJ_DIR)/assembly.o
G1_MSM_BLST_OBJ ?= $(BUILD_DIR)/g1_msm_blst.o
CUBBS_BBS_CUDA_OBJ ?= $(BUILD_DIR)/cubbs_bbs_cuda.o
PREPARE_B_MSM_OBJ ?= $(BUILD_DIR)/prepare_b_msm.o
PAIRING_OBJ ?= $(BUILD_DIR)/pairing.o
PREPARE_B_MSM_MODULES := $(filter-out src/kernels/pairing.cuh,$(wildcard src/kernels/*.cuh))
BLST_OBJS := $(BLST_SERVER_OBJ) $(BLST_ASSEMBLY_OBJ) $(G1_MSM_BLST_OBJ)
CUDA_OBJS := $(CUBBS_BBS_CUDA_OBJ) $(PREPARE_B_MSM_OBJ) $(PAIRING_OBJ)
SPPARK_REFERENCE_OBJS :=
INCLUDES := -Iinclude -I$(BLST_BINDINGS)
ifeq ($(CUBBS_BBS_ENABLE_SPPARK_MSM),1)
INCLUDES += -I$(SPPARK_ROOT) -I$(BLST_SRC)
NVCCFLAGS += -DCUBBS_BBS_ENABLE_SPPARK_MSM
endif
ifeq ($(CUBBS_BBS_ENABLE_SPPARK_REFERENCE),1)
ifneq ($(CUBBS_BBS_ENABLE_SPPARK_MSM),1)
$(error CUBBS_BBS_ENABLE_SPPARK_REFERENCE=1 requires CUBBS_BBS_ENABLE_SPPARK_MSM=1)
endif
NVCCFLAGS += -DCUBBS_BBS_ENABLE_SPPARK_REFERENCE
SPPARK_REFERENCE_OBJS += $(BUILD_DIR)/sppark_all_gpus.o
endif
.PHONY: all clean smoke bench route-bench FORCE

# Changed architectures/flags/toolchains must not silently reuse stale objects.
BUILD_CONFIG := $(BUILD_DIR)/build-config.txt
$(BUILD_CONFIG): FORCE
	@mkdir -p $(BUILD_DIR)
	@{ printf '%s\n' 'NVCC=$(NVCC)' 'CC=$(CC)' 'CXX=$(CXX)' 'NVCCFLAGS=$(NVCCFLAGS)' 'CXXFLAGS=$(CXXFLAGS)' 'INCLUDES=$(INCLUDES)' 'BLST_ROOT=$(BLST_ROOT)'; $(NVCC) --version; } > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

# Dependency files also cover transitive upstream field/curve headers.
-include $(CUDA_OBJS:.o=.d) $(BLST_OBJS:.o=.d) $(SPPARK_REFERENCE_OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(CUDA_OBJS) $(BLST_OBJS) $(SPPARK_REFERENCE_OBJS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --shared $(CUDA_OBJS) $(BLST_OBJS) $(SPPARK_REFERENCE_OBJS) -o $@
$(CUBBS_BBS_CUDA_OBJ): src/cubbs_bbs_cuda.cu src/sha256.cuh src/prepare_b_msm.cuh include/cubbs_bbs_gpu_abi.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(PREPARE_B_MSM_OBJ): src/prepare_b_msm.cu src/prepare_b_msm.cuh $(PREPARE_B_MSM_MODULES) include/cubbs_bbs_gpu_abi.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(PAIRING_OBJ): src/pairing.cu src/prepare_b_msm.cuh src/kernels/pairing.cuh include/cubbs_bbs_gpu_abi.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(BLST_SERVER_OBJ): $(BLST_SRC)/server.c
	mkdir -p $(BLST_OBJ_DIR)
	$(CC) -O2 -fPIC -fno-builtin -Wno-unused-function -Wno-unused-command-line-argument -I$(BLST_SRC) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(BLST_ASSEMBLY_OBJ): $(BLST_ROOT)/build/assembly.S
	mkdir -p $(BLST_OBJ_DIR)
	$(CC) -O2 -fPIC -I$(BLST_SRC) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(G1_MSM_BLST_OBJ): src/g1_msm_blst.cpp src/prepare_b_msm.cuh include/cubbs_bbs_gpu_abi.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD_DIR)/sppark_all_gpus.o: $(SPPARK_ROOT)/util/all_gpus.cpp
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@
$(SHA256_SMOKE): tests/sha256_smoke.cpp include/cubbs_bbs_gpu_abi.h $(TARGET)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcubbs_bbs_cuda -Wl,-rpath,$(abspath $(BUILD_DIR)) -o $@

$(H2S_SMOKE): tests/hash_to_scalar_smoke.cpp include/cubbs_bbs_gpu_abi.h $(TARGET)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcubbs_bbs_cuda -Wl,-rpath,$(abspath $(BUILD_DIR)) -o $@

$(PREPARE_B_SMOKE): tests/prepare_b_smoke.cpp include/cubbs_bbs_gpu_abi.h $(TARGET)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcubbs_bbs_cuda -Wl,-rpath,$(abspath $(BUILD_DIR)) -o $@

$(H2S_BENCH): tests/hash_to_scalar_bench.cpp include/cubbs_bbs_gpu_abi.h $(TARGET)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcubbs_bbs_cuda -Wl,-rpath,$(abspath $(BUILD_DIR)) -o $@

$(MSM_ROUTE_BENCH): tests/msm_route_bench.cpp include/cubbs_bbs_gpu_abi.h $(TARGET)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcubbs_bbs_cuda -Wl,-rpath,$(abspath $(BUILD_DIR)) -o $@

smoke: $(SHA256_SMOKE) $(H2S_SMOKE) $(PREPARE_B_SMOKE)
	$(SHA256_SMOKE)
	$(H2S_SMOKE)
	$(PREPARE_B_SMOKE)

bench: $(H2S_BENCH)
	$(H2S_BENCH)

route-bench: $(MSM_ROUTE_BENCH)

$(CUDA_OBJS) $(BLST_OBJS) $(SPPARK_REFERENCE_OBJS): $(BUILD_CONFIG)

clean:
	rm -rf $(BUILD_DIR)
