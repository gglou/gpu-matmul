#include "test_harness.h"

int main(int argc, char** argv) {
    RunMode mode = parse_mode(argc, argv);
    int M = 4096, N = 4096, K = 4096;

    std::cout << "cuBLAS SGEMM";
    if (mode == RunMode::Profile) std::cout << "  [PROFILE mode]";
    std::cout << "\nMatrix: (" << M << " x " << K << ") * ("
              << K << " x " << N << ") = (" << M << " x " << N << ")\n\n";

    // Host allocations
    float *h_a = (float*)malloc(sizeof(float) * M * K);
    float *h_b = (float*)malloc(sizeof(float) * K * N);
    float *h_c = (float*)malloc(sizeof(float) * M * N);
    MatrixDims dims = {M, N, K};
    initialize_matrices(h_a, h_b, dims);

    // Device allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);
    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    cublas_init();

    if (mode == RunMode::Profile) {
        const float alpha = 1.0f, beta = 0.0f;
        std::cout << "Profiling 'cuBLAS SGEMM' (single launch)...\n";
        cudaProfilerStart();
        cublasSgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        cudaDeviceSynchronize();
        cudaProfilerStop();
    } else {
        BenchmarkResult result = benchmark_cublas(d_a, d_b, d_c, dims);
        print_benchmark_result(result);
    }

    cublas_destroy();
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
