#include "test_harness.h"
#include "kernels/2d_blocktiling_transpose_kernel.h"

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    auto ctx = setup_test("2D Block Tiling Transpose Kernel (on-the-fly A^T)",
                          parse_mode(argc, argv));

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN << "\n\n";

    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((ctx.dims.N + BN - 1) / BN,
                (ctx.dims.M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN>,
        "2D Block Tiling Transpose (on-the-fly A^T)", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
