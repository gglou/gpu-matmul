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
#   make run KERNEL=autotune_vectorized

CUDA_IMAGE = nvidia/cuda:12.0.0-devel-ubuntu22.04

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

.PHONY: clean
clean:
	rm -f run_naive run_coalesced run_shared_mem run_1d_blocktiling run_2d_blocktiling run_2d_blocktiling_vectorized run_autotune_vectorized

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

.DEFAULT_GOAL := check
