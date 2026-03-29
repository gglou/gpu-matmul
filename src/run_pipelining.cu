#include "test_harness.h"
#include "kernels/pipelining_kernel.h"
#include <thread>
#include <chrono>

// ── Warp-tile config descriptor ───────────────────────────────────────────────
template <int BM_, int BN_, int BK_, int TM_, int TN_,
          int WM_, int WN_, int WSUBN_>
struct WarpTileConfig {
    static constexpr int BM = BM_, BN = BN_, BK = BK_;
    static constexpr int TM = TM_, TN = TN_;
    static constexpr int WM = WM_, WN = WN_, WSUBN = WSUBN_;
};

// ── Autotune configs ──────────────────────────────────────────────────────────
// A is pre-transposed to K×M row-major so both A and B use cp.async.
// As[BK][BM+4]: +4 pads stride for bank-conflict reduction.
// Dynamic shared memory: shmem = 2 * (BK*(BM+4) + BK*BN) * 4  (no 48 KB cap)
//   BK=4 → ~8 KB   BK=8 → ~16 KB   BK=16 → ~33 KB
//
// Constraints:
//   numWarps  = (BM/WM)*(BN/WN);  numThreads = numWarps*32  <= 1024
//   WSUBM     = 32 / WSUBN                  (must be integer)
//   WNITER    = WN / (WSUBN * TN)            (must be integer)
//   WMITER    = (WM*WN)/(TM*TN*32) / WNITER  (must be integer)
//   (BK*BM) % (numThreads*4) == 0           (A loads)
//   (BK*BN) % (numThreads*4) == 0           (B loads)
//   TN % 4 == 0                              (float4 epilogue stores)
using AllConfigs = std::tuple<
    // ═════════════════════════════════════════════════════════════════════════
    // BK=8  (~16 KB) — PRIMARY search space
    // ═════════════════════════════════════════════════════════════════════════
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMI×WNI  threads
    // ── TM=TN=4, 256 threads ─────────────────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  4,  4,  16, 128,   8>,  //  1×4     256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   4>,  //  1×4     256
    WarpTileConfig<         128, 128,  8,  4,  4,  32,  64,   8>,  //  2×2     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   4>,  //  2×2     256
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  32,   8>,  //  4×1     256
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  16,   4>,  //  4×1     256
    // ── TM=TN=4, 128 threads (skip WMITER=8 / WNITER=8 — they stall) ────────
    WarpTileConfig<         128, 128,  8,  4,  4,  32, 128,   8>,  //  2×4     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128,  8,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128,  8,  4,  4, 128,  32,   4>,  //  4×2     128
    // ── TM=TN=8, 256 threads ─────────────────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  32,   4>,  //  1×1     256
    WarpTileConfig<         128, 128,  8,  8,  8,  16, 128,  16>,  //  1×1     256
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  16,   2>,  //  1×1     256
    // ── TM=TN=8, 128 threads (expanded around best config) ───────────────────
    WarpTileConfig<         128, 128,  8,  8,  8,  32, 128,   8>,  //  1×2     128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  32,   2>,  //  1×2     128
    WarpTileConfig<         128, 128,  8,  8,  8,  64,  64,   8>,  //  2×1     128
    WarpTileConfig<         128, 128,  8,  8,  8, 128,  32,   4>,  //  2×1     128  ★
    WarpTileConfig<         128, 128,  8,  8,  8,  32, 128,  16>,  //  2×1     128
    // ── Asymmetric TM=8 TN=4, 128 threads ────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  8,  4, 128,  32,   4>,  //  2×2     128
    WarpTileConfig<         128, 128,  8,  8,  4,  64,  64,   8>,  //  2×2     128
    // ── Asymmetric TM=4 TN=8, 128 threads ────────────────────────────────────
    WarpTileConfig<         128, 128,  8,  4,  8, 128,  32,   2>,  //  2×2     128
    WarpTileConfig<         128, 128,  8,  4,  8,  64,  64,   4>,  //  2×2     128
    // ═════════════════════════════════════════════════════════════════════════
    // BK=4  (~8 KB) — max occupancy, 2× K-loop iterations vs BK=8
    // ═════════════════════════════════════════════════════════════════════════
    WarpTileConfig<         128, 128,  4,  4,  4,  64,  64,   4>,  //  2×4     128
    WarpTileConfig<         128, 128,  4,  4,  4,  64,  64,   8>,  //  4×2     128
    WarpTileConfig<         128, 128,  4,  4,  4, 128,  32,   8>,  //  8×1     128
    WarpTileConfig<         128, 128,  4,  8,  8,  64,  64,   4>,  //  1×2     128
    WarpTileConfig<         128, 128,  4,  8,  8,  64,  64,   8>,  //  2×1     128
    // ═════════════════════════════════════════════════════════════════════════
    // BK=16  (~33 KB) — only TM=8 256-thread (the rest consistently stall)
    // ═════════════════════════════════════════════════════════════════════════
    WarpTileConfig<         128, 128, 16,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  32,   4>   //  1×1     256
>;

// ── Transpose A (M×K → K×M) using cublasSgeam ────────────────────────────────
static float* transpose_a(float* d_a, int M, int K) {
    float *d_a_t;
    cudaMalloc(&d_a_t, sizeof(float) * M * K);
    float one = 1.0f, zero = 0.0f;
    // d_a is M×K row-major = K×M col-major (lda=K).
    // We want d_a_t as K×M row-major = M×K col-major (ldc=M).
    // cublasSgeam: C(m,n col-major) = alpha * op(A) + beta * op(B)
    //   m=M, n=K, transa=T → op(A)=A^T, A is K×M col-major → A^T is M×K. ✓
    cublasSgeam(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                M, K, &one, d_a, K, &zero, d_a_t, M, d_a_t, M);
    cudaDeviceSynchronize();
    return d_a_t;
}

// ── Launcher ──────────────────────────────────────────────────────────────────
struct Launcher {
    float *d_a_t, *d_b, *d_c;
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
        kernel<<<blocks, threads, shmem_size>>>(d_a_t, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

// ── Autotune helpers (warp-tiling variant) ────────────────────────────────────
struct WarpAutotuneResult {
    int BM, BN, BK, TM, TN, WM, WN, WSUBN;
    int WMITER, WNITER, numThreads, shmem_bytes;
    double gflops, min_ms;
};

template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN,
         typename L>
WarpAutotuneResult bench_one_warp(const MatmulTestContext& ctx,
                                  const L& launcher, int num_runs = 50) {
    launcher.template launch<BM, BN, BK, TM, TN, WM, WN, WSUBN>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Launch failed (BK=" << BK << " TM=" << TM << " TN=" << TN
                  << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN
                  << "): " << cudaGetErrorString(err) << "\n";
        constexpr int nw     = (BM / WM) * (BN / WN);
        constexpr int WSUBM  = 32 / WSUBN;
        constexpr int wniter = WN / (WSUBN * TN);
        constexpr int wmiter = WM / (WSUBM * TM);
        int nt    = nw * 32;
        int shmem = 2 * (BK * (BM + 4) + BK * BN) * (int)sizeof(float);
        return {BM, BN, BK, TM, TN, WM, WN, WSUBN,
                wmiter, wniter, nt, shmem, 0.0, 0.0};
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mn = 1e9f;
    for (int r = 0; r < num_runs; r++) {
        cudaEventRecord(start);
        launcher.template launch<BM, BN, BK, TM, TN, WM, WN, WSUBN>();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        mn = std::min(mn, ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double flops  = 2.0 * ctx.dims.M * ctx.dims.N * ctx.dims.K +
                    3.0 * ctx.dims.M * ctx.dims.N;
    double gflops = flops / (mn * 1e6);
    constexpr int nw       = (BM / WM) * (BN / WN);
    constexpr int WSUBM    = 32 / WSUBN;
    constexpr int wniter   = WN / (WSUBN * TN);
    constexpr int wmiter   = WM / (WSUBM * TM);
    int nt    = nw * 32;
    int shmem = 2 * (BK * (BM + 4) + BK * BN) * (int)sizeof(float);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    return {BM, BN, BK, TM, TN, WM, WN, WSUBN,
            wmiter, wniter, nt, shmem, gflops, (double)mn};
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

    // Filter out failed launches (0 GFLOPS)
    std::vector<int> valid;
    for (int i = 0; i < NC; i++)
        if (results[i].gflops > 0) valid.push_back(i);

    int best = valid.empty() ? 0 : valid[0];
    for (int i : valid)
        if (results[i].gflops > results[best].gflops) best = i;

    std::cout << std::fixed;
    std::cout << "  #  | BK | TM | TN | WM  | WN  | WSUBN | WMI | WNI | Thrds | Shmem  | GFLOPS  | vs cuBLAS\n";
    std::cout << " ----+----+----+----+-----+-----+-------+-----+-----+-------+--------+---------+----------\n";
    for (int i = 0; i < NC; i++) {
        const auto& r = results[i];
        if (r.gflops <= 0) continue;
        std::cout << (i == best ? " >> " : "    ")
                  << std::setw(2) << (i + 1) << " |"
                  << std::setw(3) << r.BK  << " |"
                  << std::setw(3) << r.TM  << " |"
                  << std::setw(3) << r.TN  << " |"
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
    std::cout << "\nBest:  BK=" << b.BK << "  TM=" << b.TM << "  TN=" << b.TN
              << "  WM=" << b.WM << "  WN=" << b.WN
              << "  WSUBN=" << b.WSUBN
              << "  (WMITER=" << b.WMITER << " WNITER=" << b.WNITER << ")\n"
              << "       " << std::setprecision(1) << b.gflops << " GFLOPS  ("
              << std::setprecision(2) << (b.gflops / cublas_gflops)
              << "x cuBLAS)\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 8, TN = 8;
    constexpr int WM = 64, WN = 32, WSUBN = 4;

    auto ctx = setup_test("Pipelining Kernel", parse_mode(argc, argv));

    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    // Pre-transpose A: M×K row-major → K×M row-major (one-time cost, not benchmarked).
    float *d_a_t = transpose_a(ctx.d_a, M, K);

    if (ctx.mode == RunMode::Autotune) {
        run_autotune_warp_tiled(AllConfigs{}, ctx,
                                Launcher{d_a_t, ctx.d_b, ctx.d_c, M, N, K});
        cudaFree(d_a_t);
        cleanup_test(ctx);
        return 0;
    }

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " WM=" << WM << " WN=" << WN << " WSUBN=" << WSUBN << "\n\n";

    constexpr int numWarps   = (BM / WM) * (BN / WN);
    constexpr int numThreads = numWarps * 32;
    constexpr int shmem_size = 2 * (BK * (BM + 4) + BK * BN) * (int)sizeof(float);
    dim3 threads(numThreads);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto kernel_fn = blocktiling_2d_transpose_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    BenchmarkResult result = run_kernel_custom(ctx, "Pipelining", [&]() {
        kernel_fn<<<blocks, threads, shmem_size>>>(
            d_a_t, ctx.d_b, ctx.d_c, M, N, K, 1.0f, 0.0f);
    });

    verify_and_report(ctx, result);
    cudaFree(d_a_t);
    cleanup_test(ctx);
    return 0;
}
