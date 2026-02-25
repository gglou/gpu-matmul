#ifndef REFERENCE_VECTORIZED_KERNEL_H
#define REFERENCE_VECTORIZED_KERNEL_H

// Reference vectorized GEMM kernel, closely following:
//   https://siboehm.com/articles/22/CUDA-MMM  (kernel 9 / sgemmVectorize)
//
// Key differences from our blocktiling_2d_vectorized_kernel:
//   • No pre-transposed A — A is transposed on-the-fly into shared memory
//     during the load phase, eliminating the separate transpose kernel.
//   • 1-D thread block  — dim3((BM*BN)/(TM*TN)) instead of dim3(BN/TN, BM/TM).
//   • Explicit register caches  — regM[TM] / regN[TN] are filled before the
//     inner-most double loop, so the compiler can keep them in registers.
//   • Shared memory is 1-D flat arrays for both As and Bs.
//
// Constraints:
//   numThreads = (BM * BN) / (TM * TN)   must be <= 1024
//   BK  must be divisible by 4            (float4 A loads)
//   BN  must be divisible by 4            (float4 B loads)
//   rowStrideA = numThreads * 4 / BK      must divide BM
//   rowStrideB = numThreads / (BN / 4)    must divide BK

template <int BM, int BN, int BK, int TM, int TN>
__global__ void reference_vectorized_kernel(
    float *A, float *B, float *C,
    int M, int N, int K,
    float alpha, float beta
) {
    constexpr int numThreads = (BM * BN) / (TM * TN);

    // 1-D thread indexing within the BM×BN output tile
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Flat shared memory:
    //   As stored transposed  →  As[k * BM + m]  (conflict-free column reads)
    //   Bs stored row-major   →  Bs[k * BN + n]
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    // Advance base pointers to this block's output tile
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // ── A load indices ────────────────────────────────────────────────────────
    // Each thread loads a float4 from one row of the BM×BK A-tile and writes
    // it transposed into As.  innerRowA selects the row, innerColA the float4.
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    // How many rows all threads cover in one pass
    constexpr int rowStrideA = (numThreads * 4) / BK;

    // ── B load indices ────────────────────────────────────────────────────────
    // Each thread loads a float4 from one row of the BK×BN B-tile (coalesced).
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    constexpr int rowStrideB = numThreads / (BN / 4);

    // Per-thread accumulators and register caches
    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ── Load A tile, transposing into As ──────────────────────────────────
        // Consecutive threads → consecutive innerRowA → stride-1 read from A
        // (coalesced). The transposition maps [row][col] → As[col * BM + row]
        // so the later column reads are also stride-1 (conflict-free).
        for (int offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = *reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4]);
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // ── Load B tile (coalesced, no transpose) ─────────────────────────────
        for (int offset = 0; offset < BK; offset += rowStrideB) {
            *reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4]) =
            *reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4]);
        }

        __syncthreads();

        // Advance A and B to the next BK-wide strip
        A += BK;
        B += BK * N;

        // ── Outer-product accumulate ──────────────────────────────────────────
        // Load each As column and Bs row into registers first; this lets the
        // compiler keep regM/regN in the register file and avoids repeated
        // shared-memory reads inside the inner loop.
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }

        __syncthreads();
    }

    // ── Write back: C = alpha * (A*B) + beta * C  (float4 stores) ────────────
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
            float4 c_reg = *reinterpret_cast<const float4 *>(
                &C[(threadRow * TM + m) * N + threadCol * TN + n]);
            float4 result = make_float4(
                alpha * threadResults[m * TN + n + 0] + beta * c_reg.x,
                alpha * threadResults[m * TN + n + 1] + beta * c_reg.y,
                alpha * threadResults[m * TN + n + 2] + beta * c_reg.z,
                alpha * threadResults[m * TN + n + 3] + beta * c_reg.w
            );
            *reinterpret_cast<float4 *>(
                &C[(threadRow * TM + m) * N + threadCol * TN + n]) = result;
        }
    }
}

#endif // REFERENCE_VECTORIZED_KERNEL_H
