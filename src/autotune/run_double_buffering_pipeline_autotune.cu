#include "test_harness.h"
#include "kernels/double_buffering_pipeline_kernel.h"

// A is NOT pre-transposed: loaded M×K row-major with float4 register loads,
// transposed on-the-fly into As[k][m].  Only B uses cp.async.
// Hardcoded 2-stage pipeline with dynamic shared memory.
//
// shmem = 2 * (BK*(BM+4) + BK*BN) * 4
//   BK=8  → ~16 KB     BK=16 → ~33 KB     BK=32 → ~65 KB
//
// Constraints:
//   numWarps  = (BM/WM)*(BN/WN);  numThreads = numWarps*32  <= 1024
//   WSUBM     = 32 / WSUBN                  (must be integer)
//   WNITER    = WN / (WSUBN * TN)            (must be integer)
//   WMITER    = (WM*WN)/(TM*TN*32) / WNITER  (must be integer)
//   (BK*BM) % (numThreads*4) == 0           (A loads — float4 along K)
//   (BK*BN) % (numThreads*4) == 0           (B loads)
//   TN % 4 == 0                              (float4 epilogue stores)
using AllConfigs = std::tuple<
    // ═════════════════════════════════════════════════════════════════════════
    // BK=8  (~16 KB)
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    threads
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  32,   4>,  //  128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   4>,  //  128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   8>,  //  128
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  32,   2>,  //  128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   4>,  //  128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   8>,  //  128
    WarpTileConfig<         128, 128,  8,  8,  8,  32,  64,   8>,  //  256
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  32,   4>,  //  256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   8>,  //  256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   4>,  //  256
    // ═════════════════════════════════════════════════════════════════════════
    // BK=16  (~33 KB)
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    threads
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  32,   4>,  //  256
    WarpTileConfig<         128, 128, 16,  8,  8,  32,  64,   8>,  //  256
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   4>,  //  128
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   8>,  //  128
    WarpTileConfig<         128, 128, 16,  8,  8, 128,  32,   4>,  //  128
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  32,   4>,  //  512
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  32,   8>,  //  512
    WarpTileConfig<         128, 128, 16,  4,  8,  32,  32,   4>,  //  512
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  16,   4>,  //  512
    WarpTileConfig<         128, 128, 16,  4,  4,  16,  64,   8>,  //  512
    // ═════════════════════════════════════════════════════════════════════════
    // BK=32  (~65 KB)
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    threads
    WarpTileConfig<         128, 128, 32,  8,  8,  64,  32,   4>,  //  256
    WarpTileConfig<         128, 128, 32,  8,  8,  32,  64,   8>,  //  256
    WarpTileConfig<         128, 128, 32,  8,  8,  64,  64,   4>,  //  128
    WarpTileConfig<         128, 128, 32,  8,  8,  64,  64,   8>,  //  128
    WarpTileConfig<         128, 128, 32,  8,  8, 128,  32,   4>,  //  128
    WarpTileConfig<         128, 128, 32,  4,  4,  32,  32,   4>,  //  512
    WarpTileConfig<         128, 128, 32,  4,  4,  32,  32,   8>,  //  512
    WarpTileConfig<         128, 128, 32,  4,  4,  64,  16,   4>,  //  512
    WarpTileConfig<         128, 128, 32,  4,  4,  16,  64,   8>   //  512
>;

struct Launcher {
    float *d_a, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
    void launch() const {
        constexpr int numWarps   = (BM / WM) * (BN / WN);
        constexpr int numThreads = numWarps * 32;
        constexpr int shmem_size = 2 * (BK * (BM + 4) + BK * BN) * (int)sizeof(float);
        auto kernel = blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        kernel<<<blocks, threads, shmem_size>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

int main() {
    auto ctx = setup_test("Double Buffering Pipeline Autotune", RunMode::Autotune);
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;
    run_autotune_warp_tiled(AllConfigs{}, ctx,
                            Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
    cleanup_test(ctx);
    return 0;
}
