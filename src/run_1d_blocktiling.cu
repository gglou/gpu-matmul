#include "test_harness.h"
#include "kernels/1d_blocktiling_kernel.h"

int main(int argc, char** argv) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

    auto ctx = setup_test("1D Block Tiling Kernel", parse_mode(argc, argv));
    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << "\n\n";

    dim3 threads(BN, BM / TM);
    dim3 blocks((ctx.dims.N + BN - 1) / BN,
                (ctx.dims.M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, blocktiling_1d_kernel<BM, BN, BK, TM>,
        "1D Block Tiling Kernel", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
