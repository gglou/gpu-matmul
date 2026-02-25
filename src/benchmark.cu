#include "benchmark.h"
#include <iostream>
#include <algorithm>

// ============================================================================
// GPU Benchmarking
// ============================================================================

BenchmarkResult benchmark_gpu_kernel(
    MatmulKernel kernel,
    const char* kernel_name,
    float *d_a, float *d_b, float *d_c, 
    const MatrixDims &dims,
    dim3 threadsPerBlock,
    int num_runs,
    dim3 blocksPerGrid,
    float alpha,
    float beta
) {
    // If no custom grid was provided, compute it assuming 1 thread = 1 output element.
    if (blocksPerGrid.x == 0 && blocksPerGrid.y == 0) {
        blocksPerGrid = dim3((dims.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (dims.M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, dims.M, dims.N, dims.K, alpha, beta);
    cudaDeviceSynchronize();

    // Benchmark runs
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    std::cout << "Benchmarking '" << kernel_name << "' over " << num_runs << " runs...\n";

    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, dims.M, dims.N, dims.K, alpha, beta);
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

    return {kernel_name, avg_ms, min_time, max_time, num_runs, gflops};
}

// ============================================================================
// Result Printing
// ============================================================================

void print_benchmark_result(const BenchmarkResult &result) {
    std::cout << "\n=== " << result.kernel_name << " ===\n";
    std::cout << "Average time: " << result.avg_time << " ms\n";
    std::cout << "Min time:     " << result.min_time << " ms\n";
    std::cout << "Max time:     " << result.max_time << " ms\n";
    std::cout << "GFLOPS:       " << result.gflops << "\n";
}

void compare_kernels(const std::vector<BenchmarkResult> &results, const BenchmarkResult &baseline_result) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║               PERFORMANCE COMPARISON                         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ " << baseline_result.kernel_name << ": " << baseline_result.avg_time << " ms | " << baseline_result.gflops << " GFLOPS (baseline)\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &result = results[i];
        float ratio_vs_baseline = result.avg_time / baseline_result.avg_time;
        
        std::cout << "║ " << result.kernel_name << ":\n";
        std::cout << "║   Time:    " << result.avg_time << " ms\n";
        std::cout << "║   GFLOPS:  " << result.gflops << "\n";
        std::cout << "║   Ratio:   " << ratio_vs_baseline << "x vs " << baseline_result.kernel_name << "\n";
        
        // Compare against first kernel (naive) if this isn't the first kernel
        if (i > 0) {
            float speedup_vs_naive = results[0].avg_time / result.avg_time;
            std::cout << "║   Speedup: " << speedup_vs_naive << "x vs " << results[0].kernel_name << "\n";
        }
        
        if (i < results.size() - 1) {
            std::cout << "║\n";
        }
    }
    
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
}
