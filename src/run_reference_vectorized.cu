#include "test_harness.h"
#include "kernels/reference_vectorized_kernel.h"

// ── Autotune configs ──────────────────────────────────────────────────────────
// This kernel transposes A on-the-fly (no pre-transposed A required).
// Uses a 1-D thread block of numThreads = (BM*BN)/(TM*TN).
//
// Constraints:
//   numThreads = (BM/TM)*(BN/TN) <= 1024
//   BK % 4 == 0                          (float4 A loads: BK/4 cols per thread)
//   rowStrideA = numThreads*4/BK          must divide BM
//   rowStrideB = numThreads/(BN/4)        must divide BK
//   shared memory (BK*BM + BK*BN)*4 <= 65536 bytes
using AllConfigs = std::tuple<
    TileConfig<128, 128,  8, 8, 8>,  // original article config
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
        constexpr int numThreads = (BM * BN) / (TM * TN);
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        reference_vectorized_kernel<BM, BN, BK, TM, TN>
            <<<blocks, threads>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Original article config
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int numThreads = (BM * BN) / (TM * TN);

    auto ctx = setup_test("Reference Vectorized Kernel (on-the-fly A^T + float4)",
                          parse_mode(argc, argv));

    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    if (ctx.mode == RunMode::Autotune) {
        run_autotune_tiled(AllConfigs{}, ctx,
                           Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
        cleanup_test(ctx);
        return 0;
    }

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << "  (" << numThreads << " threads/block, 1-D)\n\n";

    // 1-D thread block; explicit grid so run_kernel gets the right tile dims
    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, reference_vectorized_kernel<BM, BN, BK, TM, TN>,
        "Reference Vectorized (on-the-fly A^T + float4)", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
