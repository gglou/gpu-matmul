#ifndef CUBLAS_MATMUL_H
#define CUBLAS_MATMUL_H

#include <cublas_v2.h>
#include "../common.h"

// ============================================================================
// cuBLAS Matrix Multiplication (Optimized Baseline)
// ============================================================================

// Initialize cuBLAS handle (call once at program start)
void cublas_init();

// Destroy cuBLAS handle (call once at program end)
void cublas_destroy();

// Benchmark cuBLAS SGEMM implementation
// Uses cublasSgemm under the hood - highly optimized by NVIDIA
BenchmarkResult benchmark_cublas(float *d_a, float *d_b, float *d_c, 
                                  const MatrixDims &dims, int num_runs = 100);

#endif // CUBLAS_MATMUL_H
