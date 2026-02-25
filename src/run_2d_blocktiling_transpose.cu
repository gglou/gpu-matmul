#include "test_harness.h"
#include "kernels/2d_blocktiling_transpose_kernel.h"

// ── Autotune configs ──────────────────────────────────────────────────────────
// Constraints (float4 loads along K for A, along N for B):
//   numThreads = (BM/TM)*(BN/TN) <= 1024
//   BK % 4 == 0                          (float4 A loads along K)
//   BN % 4 == 0                          (float4 B loads along N)
//   rowStrideA = numThreads*4/BK          must divide BM
//   (BK*BN) % (numThreads*4) == 0
//   shared memory (BK*BM + BK*BN)*4 <= 65536 bytes
using AllConfigs = std::tuple<
    TileConfig<128, 128, 16, 8, 8>,
    TileConfig<128, 128, 16, 8, 4>,
    TileConfig<128, 128, 16, 4, 8>,
    TileConfig<128, 128, 32, 8, 8>,
    TileConfig<128, 128, 32, 8, 4>,
    TileConfig<128, 128, 32, 4, 8>,
    TileConfig< 64, 128, 16, 4, 8>,
    TileConfig< 64, 128, 32, 4, 8>,
    TileConfig<128,  64, 16, 8, 4>,
    TileConfig<128,  64, 32, 8, 4>,
    TileConfig< 64,  64, 16, 4, 4>,
    TileConfig< 64,  64, 32, 4, 4>
>;

struct Launcher {
    float *d_a, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN>
    void launch() const {
        dim3 threads(BN / TN, BM / TM);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN>
            <<<blocks, threads>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    auto ctx = setup_test("2D Block Tiling Transpose Kernel (on-the-fly A^T)", parse_mode(argc, argv));

    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    if (ctx.mode == RunMode::Autotune) {
        run_autotune_tiled(AllConfigs{}, ctx,
                           Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
        cleanup_test(ctx);
        return 0;
    }

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN << "\n\n";

    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN>,
        "2D Block Tiling Transpose (on-the-fly A^T)", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
