#include <iostream>
#include <vector>

#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"
#include "kernels/naive_kernel.h"
#include "kernels/coalesced_kernel.h"
// Add more kernels here as you implement them:
// #include "kernels/tiled_kernel.h"
// #include "kernels/shared_mem_kernel.h"

int main() {
    // Matrix dimensions: M x K * K x N = M x N
    // A: 2048 × 1024, B: 1024 × 1536, C: 2048 × 1536
    // On purpose divisible by 32.
    MatrixDims dims = {2048, 1536, 1024};
    
    // Initialize cuBLAS
    cublas_init();
    
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   GPU MATMUL BENCHMARK                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Matrix multiplication: (" << dims.M << " x " << dims.K << ") * (" 
              << dims.K << " x " << dims.N << ") = (" << dims.M << " x " << dims.N << ")\n\n";

    // ========================================================================
    // Memory Allocation
    // ========================================================================
    
    // Host memory
    float *h_a = (float*)malloc(sizeof(float) * dims.M * dims.K);
    float *h_b = (float*)malloc(sizeof(float) * dims.K * dims.N);
    float *h_c = (float*)malloc(sizeof(float) * dims.M * dims.N);
    float *h_c_expected = (float*)malloc(sizeof(float) * dims.M * dims.N);

    // Initialize matrices with random values
    initialize_matrices(h_a, h_b, dims);

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * dims.M * dims.K);
    cudaMalloc((void**)&d_b, sizeof(float) * dims.K * dims.N);
    cudaMalloc((void**)&d_c, sizeof(float) * dims.M * dims.N);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, sizeof(float) * dims.M * dims.K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * dims.K * dims.N, cudaMemcpyHostToDevice);

    // ========================================================================
    // Benchmark GPU Kernels
    // ========================================================================
    
    std::vector<BenchmarkResult> gpu_results;
    
    // Naive kernel (uncoalesced memory access)
    BenchmarkResult naive_result = benchmark_gpu_kernel(
        naive_kernel_matmul, 
        "Naive Kernel", 
        d_a, d_b, d_c, 
        dims
    );
    gpu_results.push_back(naive_result);
    print_benchmark_result(naive_result);

    // Copy result for verification (using last kernel run)
    cudaMemcpy(h_c, d_c, sizeof(float) * dims.M * dims.N, cudaMemcpyDeviceToHost);

    // Memory coalesced kernel
    BenchmarkResult coalesced_result = benchmark_gpu_kernel(
        coalesced_kernel_matmul, 
        "Coalesced Kernel", 
        d_a, d_b, d_c, 
        dims
    );
    gpu_results.push_back(coalesced_result);
    print_benchmark_result(coalesced_result);

    // ========================================================================
    // Add more kernels here as you implement them:
    // ========================================================================
    
    // Example: Tiled kernel
    // BenchmarkResult tiled_result = benchmark_gpu_kernel(
    //     tiled_kernel_matmul, 
    //     "Tiled Kernel", 
    //     d_a, d_b, d_c, 
    //     dims,
    //     dim3(32, 32)  // Can customize block size per kernel
    // );
    // gpu_results.push_back(tiled_result);
    // print_benchmark_result(tiled_result);

    // ========================================================================
    // Benchmark cuBLAS (optimized baseline)
    // ========================================================================
    
    // Store cuBLAS result in h_c_expected for verification
    BenchmarkResult cublas_result = benchmark_cublas(d_a, d_b, d_c, dims);
    cudaMemcpy(h_c_expected, d_c, sizeof(float) * dims.M * dims.N, cudaMemcpyDeviceToHost);
    print_benchmark_result(cublas_result);

    // ========================================================================
    // Compare Results
    // ========================================================================
    
    compare_kernels(gpu_results, cublas_result);

    // ========================================================================
    // Verify Correctness
    // ========================================================================
    
    verify_results(h_c, h_c_expected, dims.M * dims.N, "cuBLAS");

    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cublas_destroy();
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_expected);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
