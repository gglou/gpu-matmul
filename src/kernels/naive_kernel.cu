#include "naive_kernel.h"

__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K) {
    // x = row. threadIdx.x varies the row → uncoalesced access.
    int x = blockIdx.y * blockDim.y + threadIdx.x;
    // y = column.
    int y = blockIdx.x * blockDim.x + threadIdx.y;
  
    // threadId = threadIdx.x + blockDim.x * threadIdx.y

    // Boundary check. 
    // If the block dimensions do not perfectly divide the [C] matrix
    // we will cause undefined out of bounds behaviour. (Possibly crash).
    if (x < M && y < N) {
        // Sum is stored in a register.
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
          sum += a[x * K + i] * b[i * N + y];
        }

        // Set the result.
        // Store (ST) assignments are fire & forget.
        // LD (memory accesses) are more expensive.
        c[x * N + y] = sum;
    }
}
