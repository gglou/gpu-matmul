#include "cpu_matmul.h"
#include <iostream>
#include <chrono>
#include <algorithm>

// ============================================================================
// CPU Matrix Multiplication (Reference Implementation)
// ============================================================================

void cpu_matmul(float *a, float *b, float *c, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += a[row * K + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

// ============================================================================
// CPU Benchmarking
// ============================================================================

BenchmarkResult benchmark_cpu(float *h_a, float *h_b, float *h_c, 
                               const MatrixDims &dims, int num_runs) {
    // Warm-up run
    cpu_matmul(h_a, h_b, h_c, dims.M, dims.N, dims.K);
    
    // Benchmark runs
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    
    std::cout << "Benchmarking 'CPU' over " << num_runs << " runs...\n";
    
    for (int run = 0; run < num_runs; ++run) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(h_a, h_b, h_c, dims.M, dims.N, dims.K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double milliseconds = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }
    
    return {"CPU", total_time / num_runs, min_time, max_time, num_runs};
}
