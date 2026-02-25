#ifndef BLOCKTILING_1D_KERNEL_H
#define BLOCKTILING_1D_KERNEL_H

template <int BM, int BN, int BK, int TM>
__global__ void blocktiling_1d_kernel(float *a, float *b, float *c, int M, int N, int K,
                                       float alpha, float beta) {

    // shared memory cache.
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 1D - block tiling on register file.
    float threadSum[TM] = {0.0f};

    // thread "coordinates"
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // block "coordinates"
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Col (x-coordinate) of C matrix.
    const int cCol = BN * bx + tx;
    // Row of the C matrix.
    const int cStartRow = BM * by + ty * TM;

    // Each thread is responsible for calculating TM x 1
    // values in the C matrix.

    for (int i = 0; i < K; i+= BK) {

      // force memory coalescing.
      // Now aCol will be (0..BK, 0..BK, ) BM times. (Using the faster index x).
      const int linearThreadId = tx + ty * blockDim.x;
      
      const int aCol = linearThreadId % BK;
      // Now aRow will be (0, 0, ..., 1, 1, ...., BM-1, ...).
      const int aRow = linearThreadId / BK;

      if ((BM * by + aRow) < M && (i + aCol) < K) {
        As[aRow][aCol] = a[(BM * by + aRow) * K + i + aCol];
      }
      else {
        As[aRow][aCol] = 0.0f;
      }

      // (i + ty) * N to go to the correct row.
      // cCol to go to the correct column.
      if ((i + ty) < K && cCol < N) {
        Bs[ty][tx] = b[(i + ty) * N + cCol];
      } else {
        Bs[ty][tx] = 0.0f;
      }
      
      __syncthreads();
      
      for (int j = 0; j < BK; j++) {
        float b_val = Bs[j][tx];  // load once from shmem
        for (int tid = 0; tid < TM; tid++) {
            threadSum[tid] += As[ty * TM + tid][j] * b_val;
        }
    }

      __syncthreads();

    }

    // C = alpha * (A*B) + beta * C
    for (int tid = 0; tid < TM; tid++) {
      if ((cStartRow + tid) < M && cCol < N) {
        const int idx = (cStartRow + tid) * N + cCol;
        c[idx] = alpha * threadSum[tid] + beta * c[idx];
      }
    }
}

#endif // BLOCKTILING_1D_KERNEL_H
