#include "1d_blocktiling_kernel.h"

__global__ void blocktiling_1d_kernel(float *a, float *b, float *c, int M, int N, int K) {

    // BM * BK block of C will be calculated in each block.
    const int BM = 64;
    const int BN = 64;
    
    // BK is the "depth" of the tile.
    // Mainly represents the slice that we store in shared memory.
    // In the previous example we simply stored the entire block's
    // dimensions in the shared memory.
    const int BK = 8;

    // Each thread is calculating 8 cells of c.
    // This is not related to BK.
    // Because TM = 8, then we should use (64, 8) blocks.
    const int TM = 8;
    // This the "column" dimension for the Cij that each thread
    // is calculating. 2d_blocktiling will have 1 > value.
    const int TN = 1;

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

    // Each thread is responsible for calculating TM x TN
    // values in the C matrix.

    for (int i = 0; i < K; i+= BK) {

      // force memory coalescing.
      // Now aCol will be (0..8, 0..8, ) 64 times. (Using the faster index x).
      const int linearThreadId = tx + ty * blockDim.x;
      
      const int aCol = linearThreadId % BK;
      // Now aRow will be (0, 0, ..., 1, 1, ...., 63, ...).
      const int aRow = (BM * by + linearThreadId / BK);

      if ((BM * by + tx) < M && (i + ty) < K) {
        As[aRow][aCol] = a[aRow * K + i + aCol];
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

    for (int tid = 0; tid < TM; tid++) {
      if ((cStartRow + tid) < M && cCol < N) {
        c[(cStartRow + tid) * N + cCol] = threadSum[tid];
      }
    }
}
