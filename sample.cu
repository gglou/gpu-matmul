#include "utils.cuh"
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

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
      const int linearThreadId = tx + ty * blockDim.x;
      const int aCol = linearThreadId % BK;
      const int aRow = linearThreadId / BK;

      if ((BM * by + aRow) < M && (i + aCol) < K) {
        As[aRow][aCol] = A[(BM * by + aRow) * K + i + aCol];
      }
      else {
        As[aRow][aCol] = 0.0f;
      }

      // (i + ty) * N to go to the correct row.
      // cCol to go to the correct column.
      if ((i + ty) < K && cCol < N) {
        Bs[ty][tx] = B[(i + ty) * N + cCol];
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
        C[(cStartRow + tid) * N + cCol] = alpha * threadSum[tid] + beta * C[(cStartRow + tid) * N + cCol];
      }
    }
}

void launch_kernel(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    // divide the entire problem into BN x BM blocks
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    // BN * (BM/TM) = 64 * 8 = 512 threads per block
    dim3 blockDim(BN, BM / TM);
    sgemm_smem<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}
