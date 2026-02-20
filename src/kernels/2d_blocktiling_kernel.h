#ifndef BLOCKTILING_2D_KERNEL_H
#define BLOCKTILING_2D_KERNEL_H

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_kernel(float *a, float *b, float *c, int M, int N, int K) {

    // shared memory cache.
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // 2D - block tiling on register file.
    float threadSum[TM * TN] = {0.0f};

    // thread "coordinates"
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // block "coordinates"
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Col (x-coordinate) of C matrix.
    // We advance by TM x TN blocks. (Therefore the tx * TN and ty * TM).
    const int cCol = BN * bx + tx * TN;
    const int cRow = BM * by + ty * TM;

    // Each thread is responsible for calculating TM x TN
    // values in the C matrix.
    for (int i = 0; i < K; i+= BK) {
      // Calculate how many mem loads per thread we need for As and Bs.
      const int numThreads = blockDim.x * blockDim.y;
      const int loadPerThreadA = (BM * BK) / numThreads;
      const int loadPerThreadB = (BK * BN) / numThreads;

      // Linearize thread ID so consecutive warp threads get consecutive aCol
      // values → full warp reads a contiguous row of A (coalesced).
      const int linearThreadId = ty * blockDim.x + tx;
      for (int la = 0; la < loadPerThreadA; la++) {
         const int idx = linearThreadId + la * numThreads;
         const int aRow = idx / BK;
         const int aCol = idx % BK;

         if ((BM * by + aRow) < M && (i + aCol) < K) {
             As[aRow][aCol] = a[(BM * by + aRow) * K + i + aCol];
         } else {
             As[aRow][aCol] = 0.0f;
         }
      }

      // Load Bs[BK][BN]: reuse linearThreadId for correct 2D mapping.
      for (int lb = 0; lb < loadPerThreadB; lb++) {
        int linearIdx = linearThreadId + lb * numThreads;
        int bRow = linearIdx / BN;
        int bCol = linearIdx % BN;
        if ((i + bRow) < K && (BN * bx + bCol) < N) {
          Bs[bRow][bCol] = b[(i + bRow) * N + BN * bx + bCol];
        } else {
          Bs[bRow][bCol] = 0.0f;
        }
      }
      
      __syncthreads();

        // j → tid_m → tid_n: load each As value once, reuse it across all TN columns.
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

    // Assign back to Cij.
    for (int tid_m = 0; tid_m < TM; tid_m++) {
        for (int tid_n = 0; tid_n < TN; tid_n++) {
            if ((cRow + tid_m) < M && (cCol + tid_n) < N) {
                c[(cRow + tid_m) * N + cCol + tid_n] = threadSum[tid_m * TN + tid_n];
            }
        }
    }
}

#endif // BLOCKTILING_2D_KERNEL_H
