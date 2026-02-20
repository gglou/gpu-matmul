#ifndef BLOCKTILING_2D_KERNEL_H
#define BLOCKTILING_2D_KERNEL_H

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_kernel(float *a, float *b, float *c, int M, int N, int K) {

    // shared memory cache.
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

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

      // To force memory coalesces, we want la to be contributing to y-axis.
      // so that the next thread in the warp reads from the +1 on the x-axis.
      // We also want tx to advance the column and ty the rows.
      for (int la = 0; la < loadPerThreadA; la++) {
         // We "switch" ty & tx because they have the same value.
         // Otherwise we'd linerize the thread id and "swap" their increments.
         const int aRow = ty * TM + la;
         const int aCol = tx;
         
         if ((BM * by + aRow) < M && (i + aCol) < K) {
             As[aRow][aCol] = a[(BM * by + aRow) * K + i + aCol];
         } else {
             As[aRow][aCol] = 0.0f;
         }
      }

      // Load Bs[BK][BN]: linearize thread id for correct 2D mapping.
      const int tid = ty * blockDim.x + tx;
      for (int lb = 0; lb < loadPerThreadB; lb++) {
        int linearIdx = tid + lb * numThreads;
        int bRow = linearIdx / BN;
        int bCol = linearIdx % BN;
        if ((i + bRow) < K && (BN * bx + bCol) < N) {
          Bs[bRow][bCol] = b[(i + bRow) * N + BN * bx + bCol];
        } else {
          Bs[bRow][bCol] = 0.0f;
        }
      }
      
      __syncthreads();

        // Calculate the correct Tids.
        for (int tid_n = 0; tid_n < TN; tid_n++) {
            for (int j = 0; j < BK; j++) {
                float bVal = Bs[j][(tx * TN + tid_n)];
                for (int tid_m = 0; tid_m < TM; tid_m++) {
                    threadSum[tid_m * TN + tid_n] += As[ty * TM + tid_m][j] * bVal;
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
