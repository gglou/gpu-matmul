#ifndef COALESCED_KERNEL_H
#define COALESCED_KERNEL_H

// Memory coalesced kernel for CUDA matrix multiplication.
// Multiplying a matrix of [A] M x K and [B] K x N dimensions.
// The resulting matrix [C] will be of M x N dimensions.

__global__ void coalesced_kernel_matmul(float *a, float *b, float *c, int M, int N, int K,
                                         float alpha, float beta) {
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

        // C = alpha * (A*B) + beta * C
        c[y * N + x] = alpha * sum + beta * c[y * N + x];
    }
}

#endif // COALESCED_KERNEL_H
