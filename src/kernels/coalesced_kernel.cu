#include "coalesced_kernel.h"

__global__ void coalesced_kernel_matmul(float *a, float *b, float *c, int M, int N, int K) {
    // x = column.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // y = row.
    int y = blockIdx.y * blockDim.y + threadIdx.y;
  
    // threadId = threadIdx.x + blockDim.x * threadIdx.y

    // Boundary check. 
    // If the block dimensions do not perfectly divide the [C] matrix
    // we will cause undefined out of bounds behaviour. (Possibly crash).
    if (y < M && x < N) {
        // Sum is stored in a register.
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
          sum += a[y * K + i] * b[i * N + x];
        }

        // Set the result.
        // Store (ST) assignments are fire & forget.
        // LD (memory accesses) are more expensive.
        c[y * N + x] = sum;
    }
}
