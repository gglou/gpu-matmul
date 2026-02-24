# Makefile for CUDA compilation
# - Docker targets (check, build): for macOS or systems without native CUDA
# - Native targets (build-native, run): for systems with CUDA installed (e.g., Colab)
#
# Usage:
#   make run KERNEL=naive              (default)
#   make run KERNEL=coalesced
#   make run KERNEL=shared_mem
#   make run KERNEL=1d_blocktiling
#   make run KERNEL=2d_blocktiling
#   make run KERNEL=2d_blocktiling_vectorized
#   make run KERNEL=autotune
#   make run KERNEL=reduce_shared_memory_banks
#
# Profile a kernel with ncu (single launch, no benchmark loop):
#   make profile KERNEL=naive
#   make profile KERNEL=2d_blocktiling_vectorized
#
# Inspect SASS / PTX (T4 = sm_75):
#   make sass KERNEL=2d_blocktiling_vectorized
#   make ptx  KERNEL=2d_blocktiling_vectorized

CUDA_IMAGE = nvidia/cuda:12.6.3-devel-ubuntu22.04

# Target GPU architecture (T4 = sm_75)
SM ?= 75

# Which kernel to build/run (override on command line)
KERNEL ?= naive

TARGET = run_$(KERNEL)

# Source directories
SRC_DIR = src
KERNEL_DIR = $(SRC_DIR)/kernels

# The runner file for the selected kernel
RUNNER = $(SRC_DIR)/run_$(KERNEL).cu

# Library sources (shared across all runners)
SOURCES = $(SRC_DIR)/benchmark.cu \
          $(SRC_DIR)/utils.cu \
          $(KERNEL_DIR)/cublas_matmul.cu \
          $(KERNEL_DIR)/naive_kernel.cu \
          $(KERNEL_DIR)/coalesced_kernel.cu

# Libraries
LIBS = -lcublas

# All sources for the selected kernel
ALL_SOURCES = $(RUNNER) $(SOURCES)

# Include directories
INCLUDES = -I./$(SRC_DIR)

# ============================================================================
# Docker targets (for macOS or systems without native CUDA)
# ============================================================================

.PHONY: check
check:
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc $(INCLUDES) $(addprefix /workspace/,$(ALL_SOURCES)) $(LIBS) -o /dev/null 2>&1 \
		| grep -E "error|warning" || echo "✓ No compilation errors found"

.PHONY: build
build:
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc $(INCLUDES) $(addprefix /workspace/,$(ALL_SOURCES)) $(LIBS) \
		-o /workspace/$(TARGET) 2>&1 \
		| grep -E "error|warning" || echo "✓ Build successful"

# Output directory for SASS/PTX inspection files
INSPECT_DIR = inspect

.PHONY: sass
sass:
	@mkdir -p $(INSPECT_DIR)
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc $(INCLUDES) /workspace/$(RUNNER) \
		-arch=sm_$(SM) --cubin \
		-o /workspace/$(INSPECT_DIR)/$(TARGET).sm$(SM).cubin 2>&1
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		cuobjdump --dump-sass /workspace/$(INSPECT_DIR)/$(TARGET).sm$(SM).cubin \
		> $(INSPECT_DIR)/$(TARGET).sm$(SM).sass
	@echo "✓ SASS written to $(INSPECT_DIR)/$(TARGET).sm$(SM).sass"

.PHONY: ptx
ptx:
	@mkdir -p $(INSPECT_DIR)
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc $(INCLUDES) /workspace/$(RUNNER) \
		-arch=sm_$(SM) --ptx \
		-o /workspace/$(INSPECT_DIR)/$(TARGET).sm$(SM).ptx 2>&1
	@echo "✓ PTX written to $(INSPECT_DIR)/$(TARGET).sm$(SM).ptx"

.PHONY: clean
clean:
	rm -f run_naive run_coalesced run_shared_mem run_1d_blocktiling run_2d_blocktiling run_2d_blocktiling_vectorized run_autotune run_reduce_shared_memory_banks
	rm -rf $(INSPECT_DIR)

.PHONY: list-sources
list-sources:
	@echo "Runner:  $(RUNNER)"
	@echo "Sources: $(SOURCES)"

# ============================================================================
# Native CUDA targets (for systems with CUDA installed, e.g., Google Colab)
# ============================================================================

.PHONY: build-native
build-native:
	nvcc $(INCLUDES) $(ALL_SOURCES) $(LIBS) -o $(TARGET)
	@echo "✓ Build successful: $(TARGET)"

.PHONY: run
run: build-native
	./$(TARGET)

.PHONY: profile
profile: build-native
	ncu --set full ./$(TARGET) --profile

.DEFAULT_GOAL := check
