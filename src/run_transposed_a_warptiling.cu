#include "test_harness.h"
#include "kernels/transposed_a_warptiling_kernel.h"

static float* transpose_a(float* d_a, int M, int K) {
    float *d_a_t;
    cudaMalloc(&d_a_t, sizeof(float) * M * K);
    float one = 1.0f, zero = 0.0f;
    cublasSgeam(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                M, K, &one, d_a, K, &zero, d_a_t, M, d_a_t, M);
    cudaDeviceSynchronize();
    return d_a_t;
}

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 4, TN = 4;
    constexpr int WM = 64, WN = 64, WSUBN = 8;

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;

    auto ctx = setup_test("Transposed A Warp Tiling Kernel", parse_mode(argc, argv));
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    float *d_a_t = transpose_a(ctx.d_a, M, K);

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto kernel_fn = transposed_a_warptiling_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;

    BenchmarkResult result = run_kernel_custom(ctx, "Transposed A Warp Tiling", [&]() {
        kernel_fn<<<blocks, threads>>>(
            d_a_t, ctx.d_b, ctx.d_c, M, N, K, 1.0f, 0.0f);
    });

    verify_and_report(ctx, result);
    cudaFree(d_a_t);
    cleanup_test(ctx);
    return 0;
}
