#include "test_harness.h"
#include "kernels/pipelining_kernel.h"

// A is pre-transposed to K×M row-major so both A and B use cp.async.
// As[BK][BM+4]: +4 pads stride for bank-conflict reduction.
// Dynamic shared memory: shmem = 2 * (BK*(BM+4) + BK*BN) * 4  (no 48 KB cap)
//   128×128: BK=8 → ~16 KB   BK=16 → ~33 KB
//   128×256 / 256×128: BK=8 → ~24 KB
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
    // Only 256-thread configs — 128-thread BK=16 stalls, 64-thread too few.
    // Asymmetric BK=16 (128×256 / 256×128) at exactly 48 KB static smem.
    //
    //                        BM   BN   BK  TM  TN   WM   WN  WSUBN    WMI×WNI  threads
    // ═══ 128×128 BK=16 (~32 KB) — the proven baseline ══════════════════════
    WarpTileConfig<         128, 128, 16,  8,  8,  64,  32,   4>,  //  2×1     256  ★ runner best
    WarpTileConfig<         128, 128, 16,  8,  8,  32,  64,   8>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8,  16, 128,  16>,  //  1×1     256
    WarpTileConfig<         128, 128, 16,  8,  8, 128,  16,   2>,  //  1×1     256
    // ═══ 128×256 BK=16 (~48 KB) — wide, higher arith intensity ═════════════
    WarpTileConfig<         128, 256, 16,  8,  8,  64,  64,   4>,  //  1×2     256
    WarpTileConfig<         128, 256, 16,  8,  8,  64,  64,   8>,  //  2×1     256
    WarpTileConfig<         128, 256, 16,  8,  8, 128,  32,   4>,  //  2×1     256
    WarpTileConfig<         128, 256, 16,  8,  8,  32, 128,   8>,  //  1×2     256
    WarpTileConfig<         128, 256, 16,  8,  8,  32, 128,  16>,  //  2×1     256
    // ═══ 256×128 BK=16 (~48 KB) — tall, higher arith intensity ═════════════
    WarpTileConfig<         256, 128, 16,  8,  8,  64,  64,   4>,  //  1×2     256
    WarpTileConfig<         256, 128, 16,  8,  8,  64,  64,   8>,  //  2×1     256
    WarpTileConfig<         256, 128, 16,  8,  8, 128,  32,   4>,  //  2×1     256
    WarpTileConfig<         256, 128, 16,  8,  8,  32, 128,   8>,  //  1×2     256
    WarpTileConfig<         256, 128, 16,  8,  8,  32, 128,  16>,  //  2×1     256
    // ═══ 64×128 / 128×64 BK=16 (~25 KB) — small tile, maybe more blocks ═══
    WarpTileConfig<          64, 128, 16,  8,  8,  32,  64,   8>,  //  1×1     128
    WarpTileConfig<          64, 128, 16,  8,  8,  64,  32,   4>,  //  1×1     128
    WarpTileConfig<         128,  64, 16,  8,  8,  64,  32,   4>,  //  1×1     128
    WarpTileConfig<         128,  64, 16,  8,  8,  32,  64,   8>,  //  1×1     128
    // ═══ 128×256 / 256×128 BK=8 (~24 KB) ═══════════════════════════════════
    WarpTileConfig<         128, 256,  8,  8,  8,  64,  64,   4>,  //  1×2     256
    WarpTileConfig<         128, 256,  8,  8,  8, 128,  32,   4>,  //  2×1     256
    WarpTileConfig<         256, 128,  8,  8,  8,  64,  64,   4>,  //  1×2     256
    WarpTileConfig<         256, 128,  8,  8,  8, 128,  32,   4>   //  2×1     256
>;

static float* transpose_a(float* d_a, int M, int K) {
    float *d_a_t;
    cudaMalloc(&d_a_t, sizeof(float) * M * K);
    float one = 1.0f, zero = 0.0f;
    cublasSgeam(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                M, K, &one, d_a, K, &zero, d_a_t, M, d_a_t, M);
    cudaDeviceSynchronize();
    return d_a_t;
}

struct Launcher {
    float *d_a_t, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
    void launch() const {
        constexpr int numWarps   = (BM / WM) * (BN / WN);
        constexpr int numThreads = numWarps * 32;
        auto kernel = pipelining_kernel<BM, BN, BK, TM, TN, WM, WN, WSUBN>;
        dim3 threads(numThreads);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        kernel<<<blocks, threads>>>(d_a_t, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};

int main() {
    auto ctx = setup_test("Pipelining Autotune", RunMode::Autotune);
    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;
    float *d_a_t = transpose_a(ctx.d_a, M, K);
    run_autotune_warp_tiled(AllConfigs{}, ctx,
                            Launcher{d_a_t, ctx.d_b, ctx.d_c, M, N, K});
    cudaFree(d_a_t);
    cleanup_test(ctx);
    return 0;
}
