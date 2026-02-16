# Makefile for CUDA compilation checking on macOS via Docker

CUDA_IMAGE = nvidia/cuda:12.0.0-devel-ubuntu22.04
SOURCE = matrix_mul.cu
TARGET = matrix_mul

.PHONY: check
check:
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc -c /workspace/$(SOURCE) -o /dev/null 2>&1 \
		| grep -E "error|warning" || echo "✓ No compilation errors found"

.PHONY: build
build:
	@docker run --rm -v $$(pwd):/workspace $(CUDA_IMAGE) \
		nvcc /workspace/$(SOURCE) -o /workspace/$(TARGET) 2>&1 \
		| grep -E "error|warning" || echo "✓ Build successful"

.PHONY: clean
clean:
	rm -f $(TARGET)

.DEFAULT_GOAL := check
