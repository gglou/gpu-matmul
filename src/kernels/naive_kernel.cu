#include "naive_kernel.h"

// ============================================================================
// Naive GPU Kernel
// ============================================================================

__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K) {
    // Row of the resulting matrix.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Column of the resulting matrix.
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check. 
    // If the block dimensions do not perfectly divide the [C] matrix
    // we will cause undefined out of bounds behaviour. (Possibly crash).
    if (row < M && col < N) {
        // Sum is stored in a register.
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
          // 2 global memory accesses.
          // a[row * K + i] might be coalesced since they reside in contiguous
          // places in the memory.
          sum += a[row * K + i] * b[i * N + col];
        }

        // Set the result.
        // Store (ST) assignments are fire & forget.
        // LD (memory accesses) are more expensive.
        c[row * N + col] = sum;
    }
}
