#include "test_harness.h"
#include "kernels/2d_blocktiling_kernel.h"

int main(int argc, char** argv) {
    // BM=128, BN=128 gives (BN/TN) * (BM/TM) = 16 * 16 = 256 threads = 8 warps.
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    auto ctx = setup_test("2D Block Tiling Kernel", parse_mode(argc, argv));
    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN << "\n\n";

    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((ctx.dims.N + BN - 1) / BN,
                (ctx.dims.M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, blocktiling_2d_kernel<BM, BN, BK, TM, TN>,
        "2D Block Tiling Kernel", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
