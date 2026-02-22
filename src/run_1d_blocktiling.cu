#include <iostream>

#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"
#include "kernels/1d_blocktiling_kernel.h"

int main() {

    // ── Dimensions ──────────────────────────────────────────────────────────
    // A: M × K   B: K × N   C: M × N
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    MatrixDims dims = {M, N, K};

    // ── Tile parameters ──────────────────────────────────────────────────────
    constexpr int BM = 64, BN = 64, BK = 8;
    constexpr int TM = 8;

    std::cout << "1D Block Tiling Kernel\n";
    std::cout << "Matrix: (" << M << " x " << K << ") * ("
              << K << " x " << N << ") = (" << M << " x " << N << ")\n";
    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << "\n\n";

    // ── Allocate ────────────────────────────────────────────────────────────
    float *h_a = (float*)malloc(sizeof(float) * M * K);
    float *h_b = (float*)malloc(sizeof(float) * K * N);
    float *h_c = (float*)malloc(sizeof(float) * M * N);
    float *h_ref = (float*)malloc(sizeof(float) * M * N);

    initialize_matrices(h_a, h_b, dims);

    float *d_a, *d_b, *d_c, *d_c_ref;
    cudaMalloc((void**)&d_a,     sizeof(float) * M * K);
    cudaMalloc((void**)&d_b,     sizeof(float) * K * N);
    cudaMalloc((void**)&d_c,     sizeof(float) * M * N);
    cudaMalloc((void**)&d_c_ref, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // ── cuBLAS reference ────────────────────────────────────────────────────
    cublas_init();
    BenchmarkResult cublas_result = benchmark_cublas(d_a, d_b, d_c_ref, dims);
    cudaMemcpy(h_ref, d_c_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    print_benchmark_result(cublas_result);

    // ── Kernel ──────────────────────────────────────────────────────────────
    dim3 threads(BN, BM / TM);               // (64, 8)
    dim3 blocks((N + BN - 1) / BN,
                (M + BM - 1) / BM);

    BenchmarkResult result = benchmark_gpu_kernel(
        blocktiling_1d_kernel<BM, BN, BK, TM>,
        "1D Block Tiling Kernel",
        d_a, d_b, d_c,
        dims,
        threads,
        100,
        blocks
    );
    print_benchmark_result(result);

    // ── Verify ──────────────────────────────────────────────────────────────
    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    verify_results(h_c, h_ref, M * N, "cuBLAS");

    std::cout << "Speed: " << cublas_result.avg_time / result.avg_time
              << "x cuBLAS  (" << result.gflops << " vs "
              << cublas_result.gflops << " GFLOPS)\n";

    // ── Cleanup ─────────────────────────────────────────────────────────────
    cublas_destroy();
    free(h_a); free(h_b); free(h_c); free(h_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_c_ref);

    return 0;
}
