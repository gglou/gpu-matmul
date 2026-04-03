#include "test_harness.h"
#include "kernels/col_major_warptiling_kernel.h"

// Transpose B from col-major K×N to row-major K×N (= col-major N×K, ld=N).
static float* transpose_b(float* d_b, int K, int N) {
    float *d_b_t;
    cudaMalloc(&d_b_t, sizeof(float) * K * N);
    float one = 1.0f, zero = 0.0f;
    cublasSgeam(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                N, K, &one, d_b, K, &zero, d_b_t, N, d_b_t, N);
    cudaDeviceSynchronize();
    return d_b_t;
}

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 4, TN = 4;
    constexpr int WM = 64, WN = 64, WSUBN = 8;

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;

    auto ctx = setup_test("Col Major Warp Tiling Kernel", parse_mode(argc, argv));
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    // Recompute reference as col-major C: C = A_row * B_col.
    if (ctx.mode != RunMode::Profile) {
        float one = 1.0f, zero = 0.0f;
        cublasSgemm(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                    &one, ctx.d_a, K, ctx.d_b, K, &zero, ctx.d_c_ref, M);
        cudaDeviceSynchronize();
        cudaMemcpy(ctx.h_ref, ctx.d_c_ref, sizeof(float) * M * N,
                   cudaMemcpyDeviceToHost);
    }

    float *d_b_t = transpose_b(ctx.d_b, K, N);

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto kernel_fn = col_major_warptiling_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;

    BenchmarkResult result = run_kernel_custom(ctx, "Col Major Warp Tiling", [&]() {
        kernel_fn<<<blocks, threads>>>(
            ctx.d_a, d_b_t, ctx.d_c, M, N, K, 1.0f, 0.0f);
    });

    verify_and_report(ctx, result);
    cudaFree(d_b_t);
    cleanup_test(ctx);
    return 0;
}
