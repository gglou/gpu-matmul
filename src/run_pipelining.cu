#include "test_harness.h"
#include "kernels/pipelining_kernel.h"

// ── Warp-tile config descriptor ───────────────────────────────────────────────
// Extends TileConfig with WM, WN, WSUBN for warp-level tiling.
template <int BM_, int BN_, int BK_, int TM_, int TN_,
          int WM_, int WN_, int WSUBN_>
struct WarpTileConfig {
    static constexpr int BM = BM_, BN = BN_, BK = BK_;
    static constexpr int TM = TM_, TN = TN_;
    static constexpr int WM = WM_, WN = WN_, WSUBN = WSUBN_;
};

// ── Autotune configs ──────────────────────────────────────────────────────────
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

// ── Launcher ──────────────────────────────────────────────────────────────────
struct Launcher {
    float *d_a, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
    void launch() const {
        constexpr int numWarps   = (BM / WM) * (BN / WN);
        constexpr int numThreads = numWarps * 32;
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>
            <<<blocks, threads>>>(d_a, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

// ── Autotune helpers (warp-tiling variant) ────────────────────────────────────
struct WarpAutotuneResult {
    int BM, BN, BK, TM, TN, WM, WN, WSUBN;
    int WMITER, WNITER, numThreads, shmem_bytes;
    double gflops, avg_ms;
};

template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN,
         typename L>
WarpAutotuneResult bench_one_warp(const MatmulTestContext& ctx,
                                  const L& launcher, int num_runs = 50) {
    // Warm-up
    launcher.template launch<BM, BN, BK, TM, TN, WM, WN, WSUBN>();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;
    for (int r = 0; r < num_runs; r++) {
        cudaEventRecord(start);
        launcher.template launch<BM, BN, BK, TM, TN, WM, WN, WSUBN>();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg    = total / num_runs;
    double gflops = (2.0 * ctx.dims.M * ctx.dims.N * ctx.dims.K +
                     3.0 * ctx.dims.M * ctx.dims.N) / (avg * 1e6);
    constexpr int nw       = (BM / WM) * (BN / WN);
    constexpr int WSUBM    = 32 / WSUBN;
    constexpr int wniter   = WN / (WSUBN * TN);
    constexpr int wmiter   = WM / (WSUBM * TM);
    int nt    = nw * 32;
    // 2 stages: As[2][BM][BK+4] + Bs[2][BK][BN]
    int shmem = 2 * (BM * (BK + 4) + BK * BN) * (int)sizeof(float);
    return {BM, BN, BK, TM, TN, WM, WN, WSUBN,
            wmiter, wniter, nt, shmem, gflops, avg};
}

template<typename... Cfgs, typename L>
void run_autotune_warp_tiled(std::tuple<Cfgs...>,
                             const MatmulTestContext& ctx,
                             const L& launcher, int num_runs = 50) {
    const double cublas_gflops = ctx.cublas_result.gflops;
    constexpr int NC = sizeof...(Cfgs);
    std::cout << "Testing " << NC << " configs (" << num_runs
              << " runs each)...\n\n";

    std::vector<WarpAutotuneResult> results;
    results.reserve(NC);
    (results.push_back(
        bench_one_warp<Cfgs::BM, Cfgs::BN, Cfgs::BK,
                       Cfgs::TM, Cfgs::TN,
                       Cfgs::WM, Cfgs::WN, Cfgs::WSUBN>(ctx, launcher, num_runs)
    ), ...);

    // Find best
    int best = 0;
    for (int i = 1; i < NC; i++)
        if (results[i].gflops > results[best].gflops) best = i;

    // Print table
    std::cout << std::fixed;
    std::cout << "  #  | BK | TM | WM  | WN  | WSUBN | WMI | WNI | Thrds | Shmem  | GFLOPS  | vs cuBLAS\n";
    std::cout << " ----+----+----+-----+-----+-------+-----+-----+-------+--------+---------+----------\n";
    for (int i = 0; i < NC; i++) {
        const auto& r = results[i];
        std::cout << (i == best ? " >> " : "    ")
                  << std::setw(2) << (i + 1) << " |"
                  << std::setw(3) << r.BK  << " |"
                  << std::setw(3) << r.TM  << " |"
                  << std::setw(4) << r.WM  << " |"
                  << std::setw(4) << r.WN  << " |"
                  << std::setw(6) << r.WSUBN << " |"
                  << std::setw(4) << r.WMITER << " |"
                  << std::setw(4) << r.WNITER << " |"
                  << std::setw(6) << r.numThreads << " |"
                  << std::setw(5) << (r.shmem_bytes / 1024) << " KB |"
                  << std::setprecision(1) << std::setw(8) << r.gflops << " |  "
                  << std::setprecision(2) << std::setw(6)
                  << (r.gflops / cublas_gflops) << "x\n";
    }

    const auto& b = results[best];
    std::cout << "\nBest:  BK=" << b.BK << "  TM=" << b.TM
              << "  WM=" << b.WM << "  WN=" << b.WN
              << "  WSUBN=" << b.WSUBN
              << "  (WMITER=" << b.WMITER << " WNITER=" << b.WNITER << ")\n"
              << "       " << std::setprecision(1) << b.gflops << " GFLOPS  ("
              << std::setprecision(2) << (b.gflops / cublas_gflops)
              << "x cuBLAS)\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 4, TN = 4;
    constexpr int WM = 64, WN = 64, WSUBN = 8;

    auto ctx = setup_test("Pipelining Kernel", parse_mode(argc, argv));

    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    if (ctx.mode == RunMode::Autotune) {
        run_autotune_warp_tiled(AllConfigs{}, ctx,
                                Launcher{ctx.d_a, ctx.d_b, ctx.d_c, M, N, K});
        cleanup_test(ctx);
        return 0;
    }

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;
    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    BenchmarkResult result = run_kernel(
        ctx, blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>,
        "Pipelining", threads, blocks);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
