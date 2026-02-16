#include "naive_kernel.h"

// ============================================================================
// Naive GPU Kernel
// ============================================================================

__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K) {
    // x = column.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // y = row.
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check. 
    // If the block dimensions do not perfectly divide the [C] matrix
    // we will cause undefined out of bounds behaviour. (Possibly crash).
    if (y < M && x < N) {
        // Sum is stored in a register.
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
          // 2 global memory accesses.
          // b[i * N + x] memory accesses across the threads in the warp are coalesced.
          sum += a[y * K + i] * b[i * N + x];
        }

        // Set the result.
        // Store (ST) assignments are fire & forget.
        // LD (memory accesses) are more expensive.
        c[y * N + x] = sum;
    }
}
