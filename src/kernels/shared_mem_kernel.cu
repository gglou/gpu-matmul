#include "shared_mem_kernel"

#define BLOCKSIZE 16

__global__ void shared_mem_kernel(float *a, float *b, float *c, int M, int N, int K) {
    // x = column.
    int cX = blockIdx.x * blockDim.x + threadIdx.x;
    // y = row.
    int cY = blockIdx.y * blockDim.y + threadIdx.y;

    // shared memory cache.
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    float sum = 0.0f;

    // Shared memory has the same dimensions as the block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int i = 0; i < K; i+= BLOCKSIZE) {

      // cY * K to "go" the correct row.
      // i to go the correct block. (ith block * BLOCKSIZE)
      // tx --> load the value that corresponds to the tx.
      if (cY < M && (i + tx) < K) {
        As[ty][tx] = a[cY * K + i + tx];
      } else {
        As[ty][tx] = 0.0f;
      }

      // (i + ty) * N to go to the correct row.
      // cX to go to the correct column.
      if ((i + ty) < K && cX < N) {
        Bs[ty][tx] = b[(i + ty) * N + cX];
      } else {
        Bs[ty][tx] = 0.0f;
      }
      
      __syncthreads();
      
      for (int j = 0; j < BLOCKSIZE; j++) {
          sum+= As[ty][j] * Bs[j][tx];
      }

      __syncthreads();

    }

    if (cY < M && cX < N) {
      c[cY * N + cX] = sum;
    }
}
