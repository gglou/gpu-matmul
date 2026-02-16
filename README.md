# gpu-matmul
Implementing matmul on GPUs from scratch. 

Kernel is code is written explicitly by me.
Rest of the code is AI-assisted.

Mainly going through these two very well written articles:
https://siboehm.com/articles/22/CUDA-MMM
https://www.aleksagordic.com/blog/matmul

## Building and Testing

This project uses Docker to compile CUDA code on macOS (or any system without native CUDA support).

### Commands

- `make check` - Checks the CUDA code for compilation errors and warnings without producing an executable
- `make build` - Compiles the CUDA code and produces the `matrix_mul` executable
- `make clean` - Removes the compiled executable

Both `check` and `build` use the `nvidia/cuda:12.0.0-devel-ubuntu22.04` Docker image to run the NVIDIA CUDA compiler (`nvcc`).
