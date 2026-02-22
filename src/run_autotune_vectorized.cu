#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <tuple>

#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"
#include "kernels/2d_blocktiling_vectorized_kernel.h"

// Transpose A (MxK row-major) -> A_T (KxM row-major)
__global__ void transpose_to_col_major(float *out, const float *in, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
        out[col * rows + row] = in[row * cols + col];
}

struct AutotuneResult {
    int BM, BN, BK, TM, TN;
    int numThreads;
    int shmem_bytes;
    double gflops;
    double avg_ms;
};

// Compile-time config descriptor
template <int BM_, int BN_, int BK_, int TM_, int TN_>
struct TileConfig {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
};

// Add / remove configs here
using AllConfigs = std::tuple<
    TileConfig< 64,  64,  8, 8, 8>,
    TileConfig< 64,  64, 16, 8, 8>,
    TileConfig< 64, 128,  8, 8, 8>,
    TileConfig< 64, 128, 16, 8, 8>,
    TileConfig<128,  64,  8, 8, 8>,
    TileConfig<128,  64, 16, 8, 8>,
    TileConfig<128, 128,  8, 8, 8>,
    TileConfig<128, 128, 16, 8, 8>,
    TileConfig<128, 128, 16, 8, 4>,
    TileConfig<128, 128, 16, 4, 8>,
    TileConfig<256, 128, 16, 8, 8>,
    TileConfig<128, 256, 16, 8, 8>
>;

// Benchmark a single configuration (warm-up + timed runs)
template <int BM, int BN, int BK, int TM, int TN>
AutotuneResult bench_config(float *d_a_t, float *d_b, float *d_c,
                            int M, int N, int K, int num_runs) {
    const int numThreads = (BN / TN) * (BM / TM);
    const int shmem = (BM * BK + BK * BN) * sizeof(float);
    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warm-up
    blocktiling_2d_vectorized_kernel<BM, BN, BK, TM, TN>
        <<<blocks, threads>>>(d_a_t, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int r = 0; r < num_runs; r++) {
        cudaEventRecord(start);
        blocktiling_2d_vectorized_kernel<BM, BN, BK, TM, TN>
            <<<blocks, threads>>>(d_a_t, d_b, d_c, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg = total_ms / num_runs;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg * 1e6);

    return {BM, BN, BK, TM, TN, numThreads, shmem, gflops, avg};
}

// Iterate over AllConfigs at compile time (C++17 fold expression)
template <typename... Cfgs>
void bench_all(std::tuple<Cfgs...>,
               float *d_a_t, float *d_b, float *d_c,
               int M, int N, int K, int runs,
               std::vector<AutotuneResult> &results) {
    (results.push_back(
        bench_config<Cfgs::BM, Cfgs::BN, Cfgs::BK, Cfgs::TM, Cfgs::TN>(
            d_a_t, d_b, d_c, M, N, K, runs)), ...);
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    MatrixDims dims = {M, N, K};
    const int RUNS = 50;

    std::cout << "Autotuner: 2D Block Tiling Vectorized (A^T + float4)\n";
    std::cout << "Matrix: " << M << " x " << K << "  *  "
              << K << " x " << N << "\n\n";

    float *h_a   = (float*)malloc(sizeof(float) * M * K);
    float *h_b   = (float*)malloc(sizeof(float) * K * N);
    float *h_c   = (float*)malloc(sizeof(float) * M * N);
    float *h_ref = (float*)malloc(sizeof(float) * M * N);

    initialize_matrices(h_a, h_b, dims);

    float *d_a, *d_b, *d_c, *d_a_t;
    cudaMalloc((void**)&d_a,   sizeof(float) * M * K);
    cudaMalloc((void**)&d_b,   sizeof(float) * K * N);
    cudaMalloc((void**)&d_c,   sizeof(float) * M * N);
    cudaMalloc((void**)&d_a_t, sizeof(float) * K * M);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // Transpose A
    {
        dim3 t(32, 32);
        dim3 b((K + 31) / 32, (M + 31) / 32);
        transpose_to_col_major<<<b, t>>>(d_a_t, d_a, M, K);
        cudaDeviceSynchronize();
    }

    // cuBLAS reference
    cublas_init();
    BenchmarkResult cublas_ref = benchmark_cublas(d_a, d_b, d_c, dims, 100);
    cudaMemcpy(h_ref, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "cuBLAS baseline: " << std::fixed << std::setprecision(1)
              << cublas_ref.gflops << " GFLOPS  ("
              << std::setprecision(2) << cublas_ref.avg_time << " ms)\n\n";

    // Benchmark all configurations
    std::cout << "Testing " << std::tuple_size<AllConfigs>::value
              << " configurations (" << RUNS << " runs each)...\n\n";
    std::vector<AutotuneResult> results;
    results.reserve(std::tuple_size<AllConfigs>::value);

    bench_all(AllConfigs{}, d_a_t, d_b, d_c, M, N, K, RUNS, results);

    // Find best
    int bestIdx = 0;
    for (int i = 1; i < (int)results.size(); i++) {
        if (results[i].gflops > results[bestIdx].gflops)
            bestIdx = i;
    }

    // Print table
    std::cout << std::fixed;
    std::cout << "  #  | BM  | BN  | BK | TM | TN | Thrds | Shmem  | GFLOPS  | vs cuBLAS\n";
    std::cout << " ----+-----+-----+----+----+----+-------+--------+---------+----------\n";

    for (int i = 0; i < (int)results.size(); i++) {
        const auto &r = results[i];
        std::cout << (i == bestIdx ? " >> " : "    ")
                  << std::setw(2) << (i + 1) << " |"
                  << std::setw(4) << r.BM  << " |"
                  << std::setw(4) << r.BN  << " |"
                  << std::setw(3) << r.BK  << " |"
                  << std::setw(3) << r.TM  << " |"
                  << std::setw(3) << r.TN  << " |"
                  << std::setw(6) << r.numThreads << " |"
                  << std::setw(5) << (r.shmem_bytes / 1024) << " KB |"
                  << std::setprecision(1) << std::setw(8) << r.gflops << " |  "
                  << std::setprecision(2) << (r.gflops / cublas_ref.gflops) << "x\n";
    }

    // Winner
    const auto &best = results[bestIdx];
    std::cout << "\nBest:  BM=" << best.BM << "  BN=" << best.BN
              << "  BK=" << best.BK << "  TM=" << best.TM << "  TN=" << best.TN << "\n";
    std::cout << "       " << std::setprecision(1) << best.gflops << " GFLOPS  ("
              << std::setprecision(2) << (best.gflops / cublas_ref.gflops)
              << "x cuBLAS)\n";

    // Verify last config (all configs compute the same C = A*B)
    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "\nVerification (last config): ";
    verify_results(h_c, h_ref, M * N, "cuBLAS");

    // Cleanup
    cublas_destroy();
    free(h_a); free(h_b); free(h_c); free(h_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_a_t);

    return 0;
}
