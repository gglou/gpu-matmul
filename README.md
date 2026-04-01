# gpu-matmul

Implementing matrix multiplication on GPUs from scratch, progressively optimizing from a naive kernel up to pipelined, warp-tiled kernels with async memory loads.

Kernel code is written explicitly by me. The rest of the infrastructure is AI-assisted.

References:
- https://siboehm.com/articles/22/CUDA-MMM
- https://www.aleksagordic.com/blog/matmul

## Kernels

| Kernel | Description |
|--------|-------------|
| `naive` | Baseline — each thread computes one output element |
| `coalesced` | Global memory accesses coalesced across warps |
| `shared_mem` | Tiled shared-memory blocking |
| `1d_blocktiling` | 1D register tiling — each thread computes a column of outputs |
| `2d_blocktiling` | 2D register tiling — each thread computes a TM×TN output tile |
| `2d_blocktiling_vectorized` | 2D tiling with float4 loads and on-the-fly A transpose |
| `2d_blocktiling_transpose` | 2D tiling with pre-transposed A layout |
| `warptiling` | Warp-level tiling on top of 2D block tiling |
| `pipelining` | 2-stage async pipeline (cp.async) with pre-transposed A |
| `double_buffering_pipeline` | Double-buffered pipeline with on-the-fly A transpose |
| `ping_pong_pipeline` | Ping-pong buffered pipeline |

## Project Structure

```
gpu-matmul/
├── Justfile                    # Build, run, profile, and inspect recipes
├── src/
│   ├── common.h                # Shared types (MatrixDims, BenchmarkResult, kernel typedef)
│   ├── benchmark.h/cu          # GPU benchmarking framework
│   ├── utils.h/cu              # Matrix initialization and verification
│   ├── test_harness.h          # Test context, run modes, autotune infrastructure
│   ├── run_<kernel>.cu         # Entry point for each kernel (single instantiation)
│   ├── kernels/                # Kernel implementations
│   │   ├── naive_kernel.h
│   │   ├── coalesced_kernel.h
│   │   ├── shared_mem_kernel.h
│   │   ├── 1d_blocktiling_kernel.h
│   │   ├── 2d_blocktiling_kernel.h
│   │   ├── 2d_blocktiling_vectorized_kernel.h
│   │   ├── 2d_blocktiling_transpose_kernel.h
│   │   ├── warptiling_kernel.h
│   │   ├── pipelining_kernel.h
│   │   ├── double_buffering_pipeline_kernel.h
│   │   ├── ping_pong_pipeline.h
│   │   └── cublas_matmul.h/cu
│   └── autotune/               # Autotune entry points (sweep many configs)
│       ├── run_2d_blocktiling_autotune.cu
│       ├── run_2d_blocktiling_vectorized_autotune.cu
│       ├── run_2d_blocktiling_transpose_autotune.cu
│       ├── run_warptiling_autotune.cu
│       ├── run_pipelining_autotune.cu
│       ├── run_double_buffering_pipeline_autotune.cu
│       └── run_ping_pong_pipeline_autotune.cu
├── profiler/
│   ├── ncu_dashboard.py        # NCU metrics → 6-panel PNG dashboard
│   ├── sass_report.py          # SASS analysis → 6-panel PNG dashboard
│   └── output/                 # Generated PNGs and CSVs
└── inspect/                    # Raw SASS / PTX dumps
```

## Usage

Requires a local CUDA install. All commands run from the repo root.

```bash
# Build and run
just run warptiling              # Build + benchmark a kernel
just run-all                     # Build + run every kernel sequentially

# Inspect assembly
just sass warptiling             # Dump SASS → inspect/run_warptiling.sm120.sass
just ptx  warptiling             # Dump PTX  → inspect/run_warptiling.sm120.ptx

# Profile
just profile warptiling          # ncu full profile (interactive)
just plot    warptiling          # NCU metrics → profiler/output/warptiling_<ts>.png
just sass-report warptiling      # SASS analysis → profiler/output/warptiling_sass.png

# Autotune (sweeps all configs, separate binary)
just autotune warptiling

# Utilities
just build warptiling            # Compile only
just check warptiling            # Compile-check (no binary produced)
just clean                       # Remove all binaries and inspect/output
just                             # List all recipes
```
