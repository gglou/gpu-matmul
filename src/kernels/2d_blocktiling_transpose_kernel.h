#ifndef BLOCKTILING_2D_TRANSPOSE_KERNEL_H
#define BLOCKTILING_2D_TRANSPOSE_KERNEL_H

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_transpose_kernel(
    float *a_t,   // transposed A: KxM row-major  (a_t[k*M+m] = A[m][k])
    float *b,   
    float *c, 
    int M, int N, int K,
    float alpha, float beta
) {
    // shared memory cache.
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN];

    // 2D block tiling on register file.
    float threadSum[TM * TN] = {0.0f};

    // thread "coordinates"
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // block "coordinates"
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int cCol = BN * bx + tx * TN;
    const int cRow = BM * by + ty * TM;

    // Calculate how many float4 loads per thread we need for As and Bs.
    const int numThreads = blockDim.x * blockDim.y;
    const int linearThreadId = ty * blockDim.x + tx;
    const int loadPerThreadA = (BM * BK) / (numThreads * 4);
    const int loadPerThreadB = (BK * BN) / (numThreads * 4);

    for (int i = 0; i < K; i += BK) {

        // Load As from transposed A.
        // Linearize in column-major order (aCol outer, aRow inner)
        // so consecutive threads -> consecutive aRow -> stride-1 in A_T -> coalesced.
        for (int la = 0; la < loadPerThreadA; la++) {
            const int idx = (linearThreadId + la * numThreads) * 4;
            const int aCol = idx / BM;
            const int aRow = idx % BM;

            // Here aCol and aRow are switched because A is transposed.
            float4 val = *reinterpret_cast<const float4 *>(
                &a_t[(i + aCol) * M + BM * by + aRow]);

            As[aRow + 0][aCol] = val.x;
            As[aRow + 1][aCol] = val.y;
            As[aRow + 2][aCol] = val.z;
            As[aRow + 3][aCol] = val.w;
        }

        // Load Bs from B (row-major linearization).
        // Consecutive threads -> consecutive bCol -> stride-1 in B -> coalesced.
        for (int lb = 0; lb < loadPerThreadB; lb++) {
            int idx = (linearThreadId + lb * numThreads) * 4;
            int bRow = idx / BN;
            int bCol = idx % BN;

            float4 val = *reinterpret_cast<const float4 *>(
                &b[(i + bRow) * N + BN * bx + bCol]);

            Bs[bRow][bCol + 0] = val.x;
            Bs[bRow][bCol + 1] = val.y;
            Bs[bRow][bCol + 2] = val.z;
            Bs[bRow][bCol + 3] = val.w;
        }

        __syncthreads();

        // j -> tid_m -> tid_n: load each As value once, reuse it across all TN columns.
        for (int j = 0; j < BK; j++) {
            for (int tid_m = 0; tid_m < TM; tid_m++) {
                float aVal = As[ty * TM + tid_m][j];  // loaded once, reused TN times
                for (int tid_n = 0; tid_n < TN; tid_n++) {
                    threadSum[tid_m * TN + tid_n] += aVal * Bs[j][tx * TN + tid_n];
                }
            }
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

#endif // BLOCKTILING_2D_TRANSPOSE_KERNEL_H
