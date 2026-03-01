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
├── Justfile
├── README.md
└── src/
    ├── common.h          # Shared types (MatrixDims, BenchmarkResult, kernel typedef)
    ├── benchmark.h/cu    # GPU benchmarking framework
    ├── utils.h/cu        # Matrix initialization and verification
    └── kernels/
        ├── cpu_matmul.h/cu    # CPU reference implementation
        └── naive_kernel.h/cu  # Naive GPU kernel
```

## Adding New Kernels

1. Create a new kernel file in `src/kernels/` (e.g., `tiled_kernel.h`)
2. Create a corresponding `src/run_tiled.cu` entry point
3. Include the header in the entry point and wire up the kernel
4. Build and run it with:

```bash
just run tiled
```

## Building and Testing

Requires a local CUDA install (native only).

```bash
just run naive                  # Build + run the naive kernel
just build 2d_blocktiling       # Compile only
just check naive                # Compile-check (no output binary)
just profile reduce_shared_memory_banks  # Build + ncu profile
just sass 2d_blocktiling_vectorized      # Dump SASS assembly
just ptx  2d_blocktiling_vectorized      # Dump PTX IR
just autotune reduce_shared_memory_banks # Build + autotune
just run-all                    # Build + run every kernel sequentially
```

### Other Commands

- `just clean` - Removes all compiled binaries and `inspect/` output
- `just` - Lists all available recipes
