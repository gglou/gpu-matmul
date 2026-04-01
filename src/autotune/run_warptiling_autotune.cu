#include "test_harness.h"
#include "kernels/warptiling_kernel.h"

// All configs use BM=128, BN=128.  BK ∈ {8,16}  (8 KB / 16 KB shared memory).
//
// Constraints:
//   numWarps  = (BM/WM)*(BN/WN);  numThreads = numWarps*32  <= 1024
//   WSUBM     = 32 / WSUBN                  (must be integer)
//   WNITER    = WN / (WSUBN * TN)            (must be integer)
//   WMITER    = (WM*WN)/(TM*TN*32) / WNITER  (must be integer)
//   (BM*BK) % (numThreads*4) == 0           (A loads)
//   (BK*BN) % (numThreads*4) == 0           (B loads)
//
// The configs below sweep WMITER × WNITER across {1,2,4,8} × {1,2,4,8}
// while varying WSUBN to change the thread layout inside each warp.
using AllConfigs = std::tuple<
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── threadTiles = 2  (WM*WN = 1024) ─────────────────────────────────────────────
    WarpTileConfig<         128, 128, 16,  4,  4,  16,  64,   8>,  //  1×2     512
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  32,   4>,  //  1×2     512
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  32,   8>,  //  2×1     512
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  16,   4>,  //  2×1     512
    // ── threadTiles = 4  (WM*WN = 2048) ─────────────────────────────────────────────
    WarpTileConfig<         128, 128, 16,  4,  4,  16, 128,   8>,  //  1×4     256
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  64,   4>,  //  1×4     256
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  64,   8>,  //  2×2     256
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  32,   4>,  //  2×2     256
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  32,   8>,  //  4×1     256
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  16,   4>,  //  4×1     256
    // ── threadTiles = 8  (WM*WN = 4096) ─────────────────────────────────────────────
    WarpTileConfig<         128, 128, 16,  4,  4,  32, 128,   4>,  //  1×8     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   2>,  //  1×8     128
    WarpTileConfig<         128, 128, 16,  4,  4,  32, 128,   8>,  //  2×4     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  32,   4>,  //  4×2     128
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  32,   8>,  //  8×1     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,  16>,  //  8×1     128
    // ── TM=TN=8 configs  (keep WMITER*WNITER <= 2 to limit register pressure) ──
    //   threadSum = 64*WMITER*WNITER, regM = 8*WMITER, regN = 8*WNITER  (floats)
    //   threadTiles=1 → 80 reg-floats;  threadTiles=2 → 152 reg-floats
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── threadTiles = 1  (WM*WN = 2048) ─────────────────────────────────────────────
    WarpTileConfig<         128, 128, 16,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  32,   4>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8,  16, 128,  16>,  //  1×1     256
    // ── threadTiles = 2  (WM*WN = 4096) ─────────────────────────────────────────────
    WarpTileConfig<         128, 128, 16,  8,  8,  32, 128,   8>,  //  1×2     128
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   8>,  //  2×1     128
    WarpTileConfig<         128, 128, 16,  8,  8, 128,  32,   4>,  //  2×1     128
    // ═══════════════════════════════════════════════════════════════════════════
    // BK=8 variants  (8 KB shared → higher occupancy, more K-loop iterations)
    // Note: BK=8 requires numThreads <= 256  ((128*8)/(numThreads*4) >= 1)
    // ═══════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── TM=TN=4, BK=8 ──────────────────────────────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  4,  4,  16, 128,   8>,  //  1×4     256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   8>,  //  2×2     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   8>,  //  4×1     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  32,   8>,  //  8×1     128
    // ── TM=TN=8, BK=8 ──────────────────────────────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   8>   //  2×1     128
>;

struct Launcher {
    float *d_a, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
    void launch() const {
        constexpr int numWarps   = (BM / WM) * (BN / WN);
        constexpr int numThreads = numWarps * 32;
        constexpr int shmem_size = (BK * (BM + 4) + BK * BN) * (int)sizeof(float);
        auto kernel = warptiling_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        kernel<<<blocks, threads, shmem_size>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

int main() {
    auto ctx = setup_test("Warp Tiling Autotune", RunMode::Autotune);
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;
    run_autotune_warp_tiled(AllConfigs{}, ctx,
                            Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
    cleanup_test(ctx);
    return 0;
}
