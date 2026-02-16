# Makefile for CUDA compilation
# - Docker targets (check, build): for macOS or systems without native CUDA
# - Native targets (build-native, run): for systems with CUDA installed (e.g., Colab)

CUDA_IMAGE = nvidia/cuda:12.0.0-devel-ubuntu22.04
TARGET = matrix_mul

# Source directories
SRC_DIR = src
KERNEL_DIR = $(SRC_DIR)/kernels

# Source files
MAIN_SRC = $(SRC_DIR)/main.cu
SOURCES = $(SRC_DIR)/benchmark.cu \
          $(SRC_DIR)/utils.cu \
          $(KERNEL_DIR)/cublas_matmul.cu \
          $(KERNEL_DIR)/naive_kernel.cu

# Libraries
LIBS = -lcublas

# All sources combined
ALL_SOURCES = $(MAIN_SRC) $(SOURCES)

# Include directories
INCLUDES = -I./$(SRC_DIR)

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
	rm -f $(TARGET)

.PHONY: list-sources
list-sources:
	@echo "Main: $(MAIN_SRC)"
	@echo "Sources: $(SOURCES)"

# ============================================================================
# Native CUDA targets (for systems with CUDA installed, e.g., Google Colab)
# ============================================================================

.PHONY: build-native
build-native:
	nvcc $(INCLUDES) $(ALL_SOURCES) $(LIBS) -o $(TARGET)
	@echo "✓ Build successful"

.PHONY: run
run: build-native
	./$(TARGET)

.DEFAULT_GOAL := check
