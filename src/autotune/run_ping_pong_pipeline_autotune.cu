#include "test_harness.h"
#include "kernels/ping_pong_pipeline.h"

// 2-stage pipelining with cp.async doubles shared memory:
//   shmem = 2 * (BM*(BK+4) + BK*BN) * 4   (As has +4 padding for 16B alignment)
//   BK=4  → ~12 KB     BK=8 → ~20 KB     BK=16 → ~36 KB
//
// BK=8 is the sweet spot at ~20 KB (cp.async needs +4 padding for 16B alignment).
// BK=4 gives max occupancy but doubles K-loop iterations.
// BK=16 is included for reference but ~33 KB will hurt occupancy hard.
//
// Constraints:
//   numWarps  = (BM/WM)*(BN/WN);  numThreads = numWarps*32  <= 1024
//   WSUBM     = 32 / WSUBN                  (must be integer)
//   WNITER    = WN / (WSUBN * TN)            (must be integer)
//   WMITER    = (WM*WN)/(TM*TN*32) / WNITER  (must be integer)
//   (BM*BK) % (numThreads*4) == 0           (A loads)
//   (BK*BN) % (numThreads*4) == 0           (B loads)
using AllConfigs = std::tuple<
    // ═════════════════════════════════════════════════════════════════════════
    // BK=8  (~16 KB with 2 stages) — PRIMARY configs
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── TM=TN=4, threadTiles=4 (WM*WN=2048), 256 threads ───────────────────────────
    WarpTileConfig<         128, 128,  8,  4,  4,  16, 128,   8>,  //  1×4     256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   4>,  //  1×4     256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   8>,  //  2×2     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   4>,  //  2×2     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   8>,  //  4×1     256
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  16,   4>,  //  4×1     256
    // ── TM=TN=4, threadTiles=8 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128,  8,  4,  4,  32, 128,   4>,  //  1×8     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   2>,  //  1×8     128
    WarpTileConfig<         128, 128,  8,  4,  4,  32, 128,   8>,  //  2×4     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  32,   4>,  //  4×2     128
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  32,   8>,  //  8×1     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,  16>,  //  8×1     128
    // ── TM=TN=8, threadTiles=1 (WM*WN=2048), 256 threads ───────────────────────────
    WarpTileConfig<         128, 128,  8,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  32,   4>,  //  1×1     256
    // ── TM=TN=8, threadTiles=2 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128,  8,  8,  8,  32, 128,   8>,  //  1×2     128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   8>,  //  2×1     128
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  32,   4>,  //  2×1     128
    // ═════════════════════════════════════════════════════════════════════════
    // BK=4  (~8 KB with 2 stages) — max occupancy, 2× K-loop iterations
    // Note: BK=4 only works with 128 threads  ((128*4)/(numThreads*4) >= 1)
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── TM=TN=4, threadTiles=8 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128,  4,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128,  4,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128,  4,  4,  4, 128,  32,   8>,  //  8×1     128
    // ── TM=TN=8, threadTiles=2 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128,  4,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128,  4,  8,  8,  64,  64,   8>,  //  2×1     128
    // ═════════════════════════════════════════════════════════════════════════
    // BK=16  (~36 KB with 2 stages) — expanded sweep despite lower occupancy
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMITER×WNITER  threads
    // ── TM=TN=4, threadTiles=4 (WM*WN=2048), 256 threads ───────────────────────────
    WarpTileConfig<         128, 128, 16,  4,  4,  16, 128,   8>,  //  1×4     256
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  64,   4>,  //  1×4     256
    WarpTileConfig<         128, 128, 16,  4,  4,  32,  64,   8>,  //  2×2     256
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  32,   4>,  //  2×2     256
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  32,   8>,  //  4×1     256
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  16,   4>,  //  4×1     256
    // ── TM=TN=4, threadTiles=8 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128, 16,  4,  4,  32, 128,   4>,  //  1×8     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   2>,  //  1×8     128
    WarpTileConfig<         128, 128, 16,  4,  4,  32, 128,   8>,  //  2×4     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  32,   4>,  //  4×2     128
    WarpTileConfig<         128, 128, 16,  4,  4, 128,  32,   8>,  //  8×1     128
    WarpTileConfig<         128, 128, 16,  4,  4,  64,  64,  16>,  //  8×1     128
    // ── TM=TN=8, threadTiles=1 (WM*WN=2048), 256 threads ───────────────────────────
    WarpTileConfig<         128, 128, 16,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  32,   4>,  //  1×1     256
    // ── TM=TN=8, threadTiles=2 (WM*WN=4096), 128 threads ───────────────────────────
    WarpTileConfig<         128, 128, 16,  8,  8,  32, 128,   8>,  //  1×2     128
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  64,   8>,  //  2×1     128
    WarpTileConfig<         128, 128, 16,  8,  8, 128,  32,   4>   //  2×1     128
>;

struct Launcher {
    float *d_a, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
    void launch() const {
        constexpr int numWarps   = (BM / WM) * (BN / WN);
        constexpr int numThreads = numWarps * 32;
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        ping_pong_pipeline_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>
            <<<blocks, threads>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

int main() {
    auto ctx = setup_test("Ping Pong Pipeline Autotune", RunMode::Autotune);
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;
    run_autotune_warp_tiled(AllConfigs{}, ctx,
                            Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
    cleanup_test(ctx);
    return 0;
}
