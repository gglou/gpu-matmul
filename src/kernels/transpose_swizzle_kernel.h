#ifndef TRANSPOSE_SWIZZLE_KERNEL_H
#define TRANSPOSE_SWIZZLE_KERNEL_H

// XOR-swizzle to reduce shared-memory bank conflicts.
//
// Problem:  when threads in a warp access shared memory with a stride
//           that is a power-of-two, the low bits of the address repeat,
//           causing multi-way bank conflicts.
//
// Fix:      XOR bits [5 + log2(T)-1 : 5] (which DO vary across threads)
//           into the bottom log2(T) bits, spreading accesses across all
//           32 banks.
//
// Use the same call on both STORE and LOAD paths.
template <int T>
__device__ __forceinline__ int swizzle(int idx) {
    return idx ^ ((idx >> 5) & (T - 1));
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void transpose_swizzle_kernel(
    float *a_t,   // transposed A: KxM row-major  (a_t[k*M+m] = A[m][k])
    float *b,   
    float *c, 
    int M, int N, int K,
    float alpha, float beta
) {
    // shared memory cache.
    // As is stored column-major (k outer, m inner) with XOR-swizzle on the
    // m index to eliminate bank conflicts.
    __shared__ float As[BK][BM];
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

    // Calculate how many scalar loads per thread we need for As and Bs.
    const int numThreads = blockDim.x * blockDim.y;
    const int linearThreadId = ty * blockDim.x + tx;
    const int loadPerThreadA = (BM * BK) / numThreads;
    const int loadPerThreadB = (BK * BN) / numThreads;

    for (int i = 0; i < K; i += BK) {

        // Load As from transposed A.
        // Linearize in column-major order (aCol outer, aRow inner)
        // so consecutive threads -> consecutive aRow -> stride-1 in A_T -> coalesced.
        for (int la = 0; la < loadPerThreadA; la++) {
            const int idx = linearThreadId + la * numThreads;
            const int aCol = idx / BM;
            const int aRow = idx % BM;

            As[aCol][swizzle<TM>(aRow)] = a_t[(i + aCol) * M + BM * by + aRow];
        }

        // Load Bs from B (row-major linearization).
        // Consecutive threads -> consecutive bCol -> stride-1 in B -> coalesced.
        for (int lb = 0; lb < loadPerThreadB; lb++) {
            int idx = linearThreadId + lb * numThreads;
            int bRow = idx / BN;
            int bCol = idx % BN;

            Bs[bRow][swizzle<TN>(bCol)] = b[(i + bRow) * N + BN * bx + bCol];
        }

        __syncthreads();

        // j -> tid_m -> tid_n: load each As value once, reuse it across all TN columns.
        for (int j = 0; j < BK; j++) {
            for (int tid_m = 0; tid_m < TM; tid_m++) {
                float aVal = As[j][swizzle<TM>(ty * TM + tid_m)];  // loaded once, reused TN times
                for (int tid_n = 0; tid_n < TN; tid_n++) {
                    threadSum[tid_m * TN + tid_n] += aVal * Bs[j][swizzle<TN>(tx * TN + tid_n)];
                }
            }
        }

        __syncthreads();
    }

    // C = alpha * (A*B) + beta * C  — scalar stores.
    for (int tid_m = 0; tid_m < TM; tid_m++) {
        for (int tid_n = 0; tid_n < TN; tid_n++) {
            const int idx = (cRow + tid_m) * N + cCol + tid_n;
            c[idx] = alpha * threadSum[tid_m * TN + tid_n] + beta * c[idx];
        }
    }
}

#endif // TRANSPOSE_SWIZZLE_KERNEL_H
