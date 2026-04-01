#include "test_harness.h"
#include "kernels/warptiling_kernel.h"

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 4, TN = 4;
    constexpr int WM = 64, WN = 64, WSUBN = 8;

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;
    constexpr int shmem_size = (BK * (BM + 4) + BK * BN) * (int)sizeof(float);

    auto ctx = setup_test("Warp Tiling Kernel", parse_mode(argc, argv));
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto kernel_fn = warptiling_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

    BenchmarkResult result = run_kernel_custom(ctx, "Warp Tiling", [&]() {
        kernel_fn<<<blocks, threads, shmem_size>>>(
            ctx.d_a, ctx.d_b, ctx.d_c, M, N, K, 1.0f, 0.0f);
    });

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
