#include <iostream>

#include "common.h"
#include "benchmark.h"
#include "utils.h"
#include "kernels/cublas_matmul.h"
#include "kernels/2d_blocktiling_vectorized_kernel.h"

// Transpose A (MxK row-major) -> A_T (KxM row-major)
// After: a_t[k * M + m] = a[m * K + k]
__global__ void transpose_to_col_major(float *out, const float *in, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
        out[col * rows + row] = in[row * cols + col];
}

int main() {

    // A: M x K   B: K x N   C: M x N
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    MatrixDims dims = {M, N, K};

    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 8, TN = 8;

    std::cout << "2D Block Tiling Vectorized Kernel (A^T + float4)\n";
    std::cout << "Matrix: (" << M << " x " << K << ") * ("
              << K << " x " << N << ") = (" << M << " x " << N << ")\n";
    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN << "\n\n";

    float *h_a = (float*)malloc(sizeof(float) * M * K);
    float *h_b = (float*)malloc(sizeof(float) * K * N);
    float *h_c = (float*)malloc(sizeof(float) * M * N);
    float *h_ref = (float*)malloc(sizeof(float) * M * N);

    initialize_matrices(h_a, h_b, dims);

    float *d_a, *d_b, *d_c, *d_a_t;
    cudaMalloc((void**)&d_a,   sizeof(float) * M * K);
    cudaMalloc((void**)&d_b,   sizeof(float) * K * N);
    cudaMalloc((void**)&d_c,   sizeof(float) * M * N);
    cudaMalloc((void**)&d_a_t, sizeof(float) * K * M);   // transposed

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // Transpose A -> A_T
    {
        dim3 tThreads(32, 32);
        dim3 tBlocks((K + 31) / 32, (M + 31) / 32);
        transpose_to_col_major<<<tBlocks, tThreads>>>(d_a_t, d_a, M, K);
        cudaDeviceSynchronize();
        std::cout << "Transposed A  (MxK row-major -> KxM row-major)\n\n";
    }

    // cuBLAS reference
    cublas_init();
    BenchmarkResult cublas_result = benchmark_cublas(d_a, d_b, d_c, dims);
    cudaMemcpy(h_ref, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    print_benchmark_result(cublas_result);

    // Kernel
    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((N + BN - 1) / BN,
                (M + BM - 1) / BM);

    BenchmarkResult result = benchmark_gpu_kernel(
        blocktiling_2d_vectorized_kernel<BM, BN, BK, TM, TN>,
        "2D Block Tiling Vectorized (A^T + float4)",
        d_a_t, d_b, d_c,       // note: pass transposed A
        dims,
        threads,
        100,
        blocks
    );

    // Check for launch / async errors the benchmark doesn't catch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";

    print_benchmark_result(result);

    // Verify
    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    verify_results(h_c, h_ref, M * N, "cuBLAS");

    std::cout << "Speed: " << cublas_result.avg_time / result.avg_time
              << "x cuBLAS  (" << result.gflops << " vs "
              << cublas_result.gflops << " GFLOPS)\n";

    // Cleanup
    cublas_destroy();
    free(h_a); free(h_b); free(h_c); free(h_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_a_t);

    return 0;
}
