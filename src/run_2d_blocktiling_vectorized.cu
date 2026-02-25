#include "test_harness.h"
#include "kernels/2d_blocktiling_vectorized_kernel.h"

// Transpose A (MxK row-major) -> A_T (KxM row-major)
// After: a_t[k * M + m] = a[m * K + k]
__global__ void transpose_to_col_major(float *out, const float *in, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
        out[col * rows + row] = in[row * cols + col];
}

// ── Autotune configs ──────────────────────────────────────────────────────────
// Constraints (float4 loads):
//   numThreads = (BM/TM)*(BN/TN) <= 1024
//   (BM*BK) % (numThreads*4) == 0   and   (BK*BN) % (numThreads*4) == 0
//   shared memory (BM*(BK+1) + BK*BN)*4 <= 65536 bytes
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
    float *d_a_t, *d_b, *d_c;
    int M, N, K;

    template<int BM, int BN, int BK, int TM, int TN>
    void launch() const {
        dim3 threads(BN / TN, BM / TM);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        blocktiling_2d_vectorized_kernel<BM, BN, BK, TM, TN>
            <<<blocks, threads>>>(d_a_t, d_b, d_c, M, N, K, 1.0f, 0.0f);
    }
};
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    auto ctx = setup_test("2D Block Tiling Vectorized Kernel (A^T + float4)", parse_mode(argc, argv));

    int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

    // Transpose A -> A_T (done before profiler / autotune region)
    float *d_a_t;
    cudaMalloc((void**)&d_a_t, sizeof(float) * K * M);
    {
        dim3 t(32, 32);
        dim3 b((K + 31) / 32, (M + 31) / 32);
        transpose_to_col_major<<<b, t>>>(d_a_t, ctx.d_a, M, K);
        cudaDeviceSynchronize();
        std::cout << "Transposed A  (MxK row-major -> KxM row-major)\n\n";
    }

    if (ctx.mode == RunMode::Autotune) {
        run_autotune_tiled(AllConfigs{}, ctx, Launcher{d_a_t, ctx.d_b, ctx.d_c, M, N, K});
        cudaFree(d_a_t);
        cleanup_test(ctx);
        return 0;
    }

    std::cout << "Tile:   BM=" << BM << " BN=" << BN
              << " BK=" << BK << " TM=" << TM << " TN=" << TN << "\n\n";

    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    BenchmarkResult result = run_kernel_custom(ctx,
        "2D Block Tiling Vectorized (A^T + float4)",
        [&]{ blocktiling_2d_vectorized_kernel<BM, BN, BK, TM, TN>
                 <<<blocks, threads>>>(d_a_t, ctx.d_b, ctx.d_c, M, N, K, 1.0f, 0.0f); });

    verify_and_report(ctx, result);
    cudaFree(d_a_t);
    cleanup_test(ctx);
    return 0;
}
