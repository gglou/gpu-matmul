#ifndef REDUCE_SHARED_MEMORY_BANKS_KERNEL_H
#define REDUCE_SHARED_MEMORY_BANKS_KERNEL_H

// XOR-swizzle: spread accesses across all 32 banks.
template <int T>
__device__ __forceinline__ int swizzle(int idx) {
    return idx ^ ((idx >> 5) & (T - 1));
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void reduce_shared_memory_banks_kernel(
    float *a,   // A: M×K row-major
    float *b,
    float *c,
    int M, int N, int K,
    float alpha, float beta
) {
    // As[k][m] transposed + swizzled on m index.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // 2D block tiling on register file.
    float threadSum[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // thread "coordinates"
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // block "coordinates"
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int cCol = BN * bx + tx * TN;
    const int cRow = BM * by + ty * TM;

    const int numThreads = blockDim.x * blockDim.y;
    const int linearThreadId = ty * blockDim.x + tx;

    // A load: float4 along K, scatter-transposed into As[k][swizzle(m)].
    const int innerColA = linearThreadId % (BK / 4);
    const int innerRowA = linearThreadId / (BK / 4);
    constexpr int rowStrideA = ((BM * BN) / (TM * TN) * 4) / BK;

    const int loadPerThreadB = (BK * BN) / (numThreads * 4);

    for (int i = 0; i < K; i += BK) {

        // Load A tile, transposing on-the-fly into As[k][swizzle(m)].
        for (int offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = *reinterpret_cast<const float4 *>(
                &a[(BM * by + innerRowA + offset) * K + i + innerColA * 4]);
            As[innerColA * 4 + 0][swizzle<TM>(innerRowA + offset)] = tmp.x;
            As[innerColA * 4 + 1][swizzle<TM>(innerRowA + offset)] = tmp.y;
            As[innerColA * 4 + 2][swizzle<TM>(innerRowA + offset)] = tmp.z;
            As[innerColA * 4 + 3][swizzle<TM>(innerRowA + offset)] = tmp.w;
        }

        // Load Bs from B (row-major, coalesced) with swizzle.
        for (int lb = 0; lb < loadPerThreadB; lb++) {
            int idx = (linearThreadId + lb * numThreads) * 4;
            int bRow = idx / BN;
            int bCol = idx % BN;

            float4 val = *reinterpret_cast<const float4 *>(
                &b[(i + bRow) * N + BN * bx + bCol]);

            Bs[bRow][swizzle<TN>(bCol + 0)] = val.x;
            Bs[bRow][swizzle<TN>(bCol + 1)] = val.y;
            Bs[bRow][swizzle<TN>(bCol + 2)] = val.z;
            Bs[bRow][swizzle<TN>(bCol + 3)] = val.w;
        }

        __syncthreads();

        for (int j = 0; j < BK; j++) {
            for (int tid_m = 0; tid_m < TM; tid_m++)
                regM[tid_m] = As[j][swizzle<TM>(ty * TM + tid_m)];
            for (int tid_n = 0; tid_n < TN; tid_n++)
                regN[tid_n] = Bs[j][swizzle<TN>(tx * TN + tid_n)];
            for (int tid_m = 0; tid_m < TM; tid_m++)
                for (int tid_n = 0; tid_n < TN; tid_n++)
                    threadSum[tid_m * TN + tid_n] += regM[tid_m] * regN[tid_n];
        }

        __syncthreads();
    }

    // C = alpha * (A*B) + beta * C  — float4 stores (TN must be a multiple of 4).
    for (int tid_m = 0; tid_m < TM; tid_m++) {
        for (int tid_n = 0; tid_n < TN; tid_n += 4) {
            const int base = (cRow + tid_m) * N + cCol + tid_n;
            // Load c values first into a register
            float4 c_reg = *reinterpret_cast<const float4 *>(&c[base]);

            // Do math in registers
            float4 result = make_float4(
                alpha * threadSum[tid_m * TN + tid_n + 0] + beta * c_reg.x,
                alpha * threadSum[tid_m * TN + tid_n + 1] + beta * c_reg.y,
                alpha * threadSum[tid_m * TN + tid_n + 2] + beta * c_reg.z,
                alpha * threadSum[tid_m * TN + tid_n + 3] + beta * c_reg.w
            );

            // Assign back the result
            *reinterpret_cast<float4 *>(&c[base]) = result;
        }
    }
}

#endif // REDUCE_SHARED_MEMORY_BANKS_KERNEL_H
