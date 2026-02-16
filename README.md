# gpu-matmul
Implementing matmul on GPUs from scratch. 

Kernel is code is written explicitly by me.
Rest of the code is AI-assisted.

Mainly going through these two very well written articles:
https://siboehm.com/articles/22/CUDA-MMM
https://www.aleksagordic.com/blog/matmul

## Project Structure

```
gpu-matmul/
├── Makefile
├── README.md
└── src/
    ├── main.cu           # Main program
    ├── common.h          # Shared types (MatrixDims, BenchmarkResult, kernel typedef)
    ├── benchmark.h/cu    # GPU benchmarking framework
    ├── utils.h/cu        # Matrix initialization and verification
    └── kernels/
        ├── cpu_matmul.h/cu    # CPU reference implementation
        └── naive_kernel.h/cu  # Naive GPU kernel
```

## Adding New Kernels

1. Create a new kernel file in `src/kernels/` (e.g., `tiled_kernel.h` and `tiled_kernel.cu`)
2. Add the source file to `SOURCES` in the `Makefile`
3. Include the header in `src/main.cu`
4. Benchmark it with:

```cpp
BenchmarkResult result = benchmark_gpu_kernel(
    your_kernel_matmul,     // kernel function
    "Your Kernel Name",     // name for display
    d_a, d_b, d_c,          // device pointers
    dims,                   // matrix dimensions
    dim3(16, 16)            // optional: custom block size
);
gpu_results.push_back(result);
```

## Building and Testing

### Native CUDA (Google Colab, Linux with CUDA)

```bash
make build-native    # Compile with nvcc
make run             # Build and run
```

### Docker (macOS or systems without native CUDA)

```bash
make check           # Check for compilation errors
make build           # Compile via Docker
```

### Other Commands

- `make clean` - Removes the compiled executable
- `make list-sources` - Shows all source files being compiled

The Docker targets use the `nvidia/cuda:12.0.0-devel-ubuntu22.04` image.
