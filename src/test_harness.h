#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"

// ============================================================================
// Run mode
// ============================================================================

enum class RunMode { Benchmark, Profile };

inline RunMode parse_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--profile") == 0) return RunMode::Profile;
    return RunMode::Benchmark;
}

// ============================================================================
// Test context — owns all host/device memory for one matmul test
// ============================================================================

struct MatmulTestContext {
    RunMode mode;
    MatrixDims dims;
    float *h_a, *h_b, *h_c, *h_ref;
    float *d_a, *d_b, *d_c, *d_c_ref;
    BenchmarkResult cublas_result;
};

inline MatmulTestContext setup_test(
    const char* name, RunMode mode,
    int M = 4096, int N = 4096, int K = 4096)
{
    MatmulTestContext ctx{};
    ctx.mode = mode;
    ctx.dims = {M, N, K};

    std::cout << name;
    if (mode == RunMode::Profile) std::cout << "  [PROFILE mode]";
    std::cout << "\nMatrix: (" << M << " x " << K << ") * ("
              << K << " x " << N << ") = (" << M << " x " << N << ")\n\n";

    // Host allocations
    ctx.h_a   = (float*)malloc(sizeof(float) * M * K);
    ctx.h_b   = (float*)malloc(sizeof(float) * K * N);
    ctx.h_c   = (float*)malloc(sizeof(float) * M * N);
    ctx.h_ref = (float*)malloc(sizeof(float) * M * N);
    initialize_matrices(ctx.h_a, ctx.h_b, ctx.dims);

    // Device allocations
    cudaMalloc((void**)&ctx.d_a,     sizeof(float) * M * K);
    cudaMalloc((void**)&ctx.d_b,     sizeof(float) * K * N);
    cudaMalloc((void**)&ctx.d_c,     sizeof(float) * M * N);
    cudaMalloc((void**)&ctx.d_c_ref, sizeof(float) * M * N);
    cudaMemcpy(ctx.d_a, ctx.h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_b, ctx.h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // cuBLAS reference — skipped entirely in profile mode (no verification needed).
    if (mode == RunMode::Benchmark) {
        cublas_init();
        ctx.cublas_result = benchmark_cublas(ctx.d_a, ctx.d_b, ctx.d_c_ref, ctx.dims);
        cudaMemcpy(ctx.h_ref, ctx.d_c_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        print_benchmark_result(ctx.cublas_result);
    }

    return ctx;
}

// ============================================================================
// Kernel execution — mode-aware (standard MatmulKernel signature)
// ============================================================================

inline BenchmarkResult run_kernel(
    const MatmulTestContext& ctx,
    MatmulKernel kernel,
    const char* name,
    dim3 threads,
    dim3 blocks = dim3(0, 0),
    int num_runs = 100)
{
    if (blocks.x == 0 && blocks.y == 0) {
        blocks = dim3(
            (ctx.dims.N + threads.x - 1) / threads.x,
            (ctx.dims.M + threads.y - 1) / threads.y);
    }

    if (ctx.mode == RunMode::Profile) {
        std::cout << "Profiling '" << name << "' (single launch)...\n";
        cudaProfilerStart();
        kernel<<<blocks, threads>>>(
            ctx.d_a, ctx.d_b, ctx.d_c,
            ctx.dims.M, ctx.dims.N, ctx.dims.K);
        cudaDeviceSynchronize();
        cudaProfilerStop();
        return {name, 0.0, 0.0, 0.0, 1, 0.0};
    }

    // Benchmark mode — delegate to existing benchmark function
    BenchmarkResult result = benchmark_gpu_kernel(
        kernel, name,
        ctx.d_a, ctx.d_b, ctx.d_c,
        ctx.dims, threads, num_runs, blocks);
    print_benchmark_result(result);
    return result;
}

// ============================================================================
// Kernel execution — lambda overload for non-standard launches
// (e.g. kernels needing transposed A, custom pointer args, etc.)
// ============================================================================

template <typename LaunchFn>
BenchmarkResult run_kernel_custom(
    const MatmulTestContext& ctx,
    const char* name,
    LaunchFn&& launch,
    int num_runs = 100)
{
    if (ctx.mode == RunMode::Profile) {
        std::cout << "Profiling '" << name << "' (single launch)...\n";
        cudaProfilerStart();
        launch();
        cudaDeviceSynchronize();
        cudaProfilerStop();
        return {name, 0.0, 0.0, 0.0, 1, 0.0};
    }

    // Benchmark mode — warmup + timed runs
    launch();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0, mn = 1e9f, mx = 0;

    std::cout << "Benchmarking '" << name << "' over " << num_runs << " runs...\n";
    for (int i = 0; i < num_runs; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
        mn = std::min(mn, ms);
        mx = std::max(mx, ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg = total / num_runs;
    double gflops = 2.0 * ctx.dims.M * ctx.dims.N * ctx.dims.K / (avg * 1e6);
    BenchmarkResult result = {name, avg, (double)mn, (double)mx, num_runs, gflops};
    print_benchmark_result(result);
    return result;
}

// ============================================================================
// Verify + report
// ============================================================================

inline void verify_and_report(MatmulTestContext& ctx, const BenchmarkResult& result) {
    if (ctx.mode == RunMode::Profile) return;

    int M = ctx.dims.M, N = ctx.dims.N;
    cudaMemcpy(ctx.h_c, ctx.d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    verify_results(ctx.h_c, ctx.h_ref, M * N, "cuBLAS");

    std::cout << "Speed: " << ctx.cublas_result.avg_time / result.avg_time
              << "x cuBLAS  (" << result.gflops << " vs "
              << ctx.cublas_result.gflops << " GFLOPS)\n";
}

// ============================================================================
// Cleanup
// ============================================================================

inline void cleanup_test(MatmulTestContext& ctx) {
    if (ctx.mode == RunMode::Benchmark) cublas_destroy();
    free(ctx.h_a); free(ctx.h_b); free(ctx.h_c); free(ctx.h_ref);
    cudaFree(ctx.d_a); cudaFree(ctx.d_b); cudaFree(ctx.d_c); cudaFree(ctx.d_c_ref);
}

#endif // TEST_HARNESS_H
