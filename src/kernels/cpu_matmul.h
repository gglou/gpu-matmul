#ifndef CPU_MATMUL_H
#define CPU_MATMUL_H

#include "../common.h"

// ============================================================================
// CPU Matrix Multiplication (Reference Implementation)
// ============================================================================

// CPU matrix multiplication for verification
// C = A * B where A is M x K, B is K x N, C is M x N
void cpu_matmul(float *a, float *b, float *c, int M, int N, int K);

// Benchmark CPU implementation
BenchmarkResult benchmark_cpu(float *h_a, float *h_b, float *h_c, 
                               const MatrixDims &dims, int num_runs = 10);

#endif // CPU_MATMUL_H
