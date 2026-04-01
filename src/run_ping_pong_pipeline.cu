#include "test_harness.h"
#include "kernels/ping_pong_pipeline.h"

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 4, TN = 4;
    constexpr int WM = 128, WN = 32, WSUBN = 4;

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;

    auto ctx = setup_test("Ping Pong Pipeline Kernel", parse_mode(argc, argv));
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, ping_pong_pipeline_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>,
        "Ping Pong Pipeline", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
