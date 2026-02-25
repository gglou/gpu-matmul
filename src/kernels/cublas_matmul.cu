#include "cublas_matmul.h"
#include <iostream>
#include <algorithm>

// Global cuBLAS handle
static cublasHandle_t cublas_handle;

// ============================================================================
// cuBLAS Initialization
// ============================================================================

void cublas_init() {
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!\n";
        exit(1);
    }
}

void cublas_destroy() {
    cublasDestroy(cublas_handle);
}

// ============================================================================
// cuBLAS Matrix Multiplication
// ============================================================================

// cuBLAS uses column-major ordering. For row-major matrices A, B, C where C = A * B:
// We compute: C^T = B^T * A^T (which gives us C in row-major format)
// This is equivalent to calling cublasSgemm with swapped A and B, and transposed dims.

static void cublas_matmul(float *d_a, float *d_b, float *d_c, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // For row-major: C(M,N) = A(M,K) * B(K,N)
    // cuBLAS (column-major): C^T(N,M) = B^T(N,K) * A^T(K,M)
    // So we call: cublasSgemm(B, A, C) with dims (N, M, K)
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose (we're doing the swap trick)
                N, M, K,                    // Dimensions (swapped for row-major)
                &alpha,
                d_b, N,                     // B with leading dimension N
                d_a, K,                     // A with leading dimension K
                &beta,
                d_c, N);                    // C with leading dimension N
}

// ============================================================================
// cuBLAS Benchmarking
// ============================================================================

BenchmarkResult benchmark_cublas(float *d_a, float *d_b, float *d_c, 
                                  const MatrixDims &dims, int num_runs) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    cublas_matmul(d_a, d_b, d_c, dims.M, dims.N, dims.K);
    cudaDeviceSynchronize();

    // Benchmark runs
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    std::cout << "Benchmarking 'cuBLAS' over " << num_runs << " runs...\n";

    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);
        cublas_matmul(d_a, d_b, d_c, dims.M, dims.N, dims.K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_ms = total_time / num_runs;
    // 2*M*N*K for the multiply-accumulate + M*N each for: alpha scale, beta scale, final add
    double flops = 2.0 * dims.M * dims.N * dims.K + 3.0 * dims.M * dims.N;
    double gflops = flops / (avg_ms * 1e6);

    return {"cuBLAS", avg_ms, min_time, max_time, num_runs, gflops};
}
