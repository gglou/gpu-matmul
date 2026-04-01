#include "test_harness.h"
#include "kernels/2d_blocktiling_kernel.h"

// This kernel uses scalar shared-memory loads (no float4) and operates on A
// directly (no transpose). Constraints:
//   numThreads = (BM/TM)*(BN/TN) <= 1024
//   (BM*BK) % numThreads == 0   and   (BK*BN) % numThreads == 0
//   shared memory (BM*BK + BK*BN)*4 <= 65536 bytes
using AllConfigs = std::tuple<
    TileConfig<128, 128,  8, 8, 8>,
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
    TileConfig< 64,  64,  8, 8, 8>,
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
        blocktiling_2d_kernel<BM, BN, BK, TM, TN>
            <<<blocks, threads>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

int main() {
    auto ctx = setup_test("2D Block Tiling Autotune", RunMode::Autotune);
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;
    run_autotune_tiled(AllConfigs{}, ctx, Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
    cleanup_test(ctx);
    return 0;
}
