#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"

// ============================================================================
// Run mode
// ============================================================================

enum class RunMode { Benchmark, Profile, Autotune };

inline RunMode parse_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--profile")  == 0) return RunMode::Profile;
        if (strcmp(argv[i], "--autotune") == 0) return RunMode::Autotune;
    }
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
    if (mode == RunMode::Profile)  std::cout << "  [PROFILE mode]";
    if (mode == RunMode::Autotune) std::cout << "  [AUTOTUNE mode]";
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

    // cuBLAS reference — skipped in profile mode; needed for benchmark and autotune.
    if (mode == RunMode::Benchmark || mode == RunMode::Autotune) {
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
            ctx.dims.M, ctx.dims.N, ctx.dims.K, 1.0f, 0.0f);
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
    // 2*M*N*K for the multiply-accumulate + M*N each for: alpha scale, beta scale, final add
    double gflops = (2.0 * ctx.dims.M * ctx.dims.N * ctx.dims.K + 3.0 * ctx.dims.M * ctx.dims.N) / (avg * 1e6);
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
    if (ctx.mode == RunMode::Benchmark || ctx.mode == RunMode::Autotune) cublas_destroy();
    free(ctx.h_a); free(ctx.h_b); free(ctx.h_c); free(ctx.h_ref);
    cudaFree(ctx.d_a); cudaFree(ctx.d_b); cudaFree(ctx.d_c); cudaFree(ctx.d_c_ref);
}

// ============================================================================
// Autotune infrastructure
// ============================================================================

// Compile-time tile config descriptor — used to build AllConfigs tuples
template <int BM_, int BN_, int BK_, int TM_, int TN_>
struct TileConfig {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
};

struct AutotuneResult {
    int BM, BN, BK, TM, TN, numThreads, shmem_bytes;
    double gflops, avg_ms;
};

// Benchmark a single (BM, BN, BK, TM, TN) config using the provided Launcher.
// Launcher must expose:  template<int BM, int BN, int BK, int TM, int TN> void launch() const;
template<int BM, int BN, int BK, int TM, int TN, typename Launcher>
AutotuneResult bench_one(const MatmulTestContext& ctx, const Launcher& launcher,
                         int num_runs = 50) {
    // Warm-up
    launcher.template launch<BM, BN, BK, TM, TN>();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;
    for (int r = 0; r < num_runs; r++) {
        cudaEventRecord(start);
        launcher.template launch<BM, BN, BK, TM, TN>();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg    = total / num_runs;
    // 2*M*N*K for the multiply-accumulate + M*N each for: alpha scale, beta scale, final add
    double gflops = (2.0 * ctx.dims.M * ctx.dims.N * ctx.dims.K + 3.0 * ctx.dims.M * ctx.dims.N) / (avg * 1e6);
    int nt        = (BN / TN) * (BM / TM);
    int shmem     = (BM * (BK + 1) + BK * BN) * (int)sizeof(float);
    return {BM, BN, BK, TM, TN, nt, shmem, gflops, avg};
}

// Iterate over a tuple of TileConfigs, bench each, print a ranked table.
template<typename... Cfgs, typename Launcher>
void run_autotune_tiled(std::tuple<Cfgs...>, const MatmulTestContext& ctx,
                        const Launcher& launcher, int num_runs = 50) {
    const double cublas_gflops = ctx.cublas_result.gflops;
    constexpr int N = sizeof...(Cfgs);
    std::cout << "Testing " << N << " configs (" << num_runs << " runs each)...\n\n";

    std::vector<AutotuneResult> results;
    results.reserve(N);
    (results.push_back(
        bench_one<Cfgs::BM, Cfgs::BN, Cfgs::BK, Cfgs::TM, Cfgs::TN>(ctx, launcher, num_runs)
    ), ...);

    // Find best
    int best = 0;
    for (int i = 1; i < N; i++)
        if (results[i].gflops > results[best].gflops) best = i;

    // Print table
    std::cout << std::fixed;
    std::cout << "  #  | BM  | BN  | BK | TM | TN | Thrds | Shmem  | GFLOPS  | vs cuBLAS\n";
    std::cout << " ----+-----+-----+----+----+----+-------+--------+---------+----------\n";
    for (int i = 0; i < N; i++) {
        const auto& r = results[i];
        std::cout << (i == best ? " >> " : "    ")
                  << std::setw(2) << (i + 1) << " |"
                  << std::setw(4) << r.BM  << " |"
                  << std::setw(4) << r.BN  << " |"
                  << std::setw(3) << r.BK  << " |"
                  << std::setw(3) << r.TM  << " |"
                  << std::setw(3) << r.TN  << " |"
                  << std::setw(6) << r.numThreads << " |"
                  << std::setw(5) << (r.shmem_bytes / 1024) << " KB |"
                  << std::setprecision(1) << std::setw(8) << r.gflops << " |  "
                  << std::setprecision(2) << std::setw(6) << (r.gflops / cublas_gflops) << "x\n";
    }

    const auto& b = results[best];
    std::cout << "\nBest:  BM=" << b.BM << "  BN=" << b.BN
              << "  BK=" << b.BK << "  TM=" << b.TM << "  TN=" << b.TN << "\n"
              << "       " << std::setprecision(1) << b.gflops << " GFLOPS  ("
              << std::setprecision(2) << (b.gflops / cublas_gflops) << "x cuBLAS)\n";
}

#endif // TEST_HARNESS_H
