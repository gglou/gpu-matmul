#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"
#include <vector>

// ============================================================================
// GPU Benchmarking
// ============================================================================

// Benchmark any GPU kernel that follows the MatmulKernel signature
// C = alpha * A * B + beta * C  (alpha=1, beta=0 → pure matmul)
BenchmarkResult benchmark_gpu_kernel(
    MatmulKernel kernel,
    const char* kernel_name,
    float *d_a, float *d_b, float *d_c, 
    const MatrixDims &dims,
    dim3 threadsPerBlock = dim3(16, 16),
    int num_runs = 100,
    dim3 blocksPerGrid = dim3(0, 0),
    float alpha = 1.0f,
    float beta  = 0.0f
);

// ============================================================================
// Result Printing
// ============================================================================

void print_benchmark_result(const BenchmarkResult &result);
void compare_kernels(const std::vector<BenchmarkResult> &results, const BenchmarkResult &baseline_result);

#endif // BENCHMARK_H
