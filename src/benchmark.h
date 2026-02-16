#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"
#include <vector>

// ============================================================================
// GPU Benchmarking
// ============================================================================

// Benchmark any GPU kernel that follows the MatmulKernel signature
BenchmarkResult benchmark_gpu_kernel(
    MatmulKernel kernel,
    const char* kernel_name,
    float *d_a, float *d_b, float *d_c, 
    const MatrixDims &dims,
    dim3 threadsPerBlock = dim3(16, 16),
    int num_runs = 100
);

// ============================================================================
// Result Printing
// ============================================================================

void print_benchmark_result(const BenchmarkResult &result);
void compare_kernels(const std::vector<BenchmarkResult> &results, const BenchmarkResult &cpu_result);

#endif // BENCHMARK_H
