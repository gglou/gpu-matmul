#ifndef NAIVE_KERNEL_H
#define NAIVE_KERNEL_H

// Naive kernel for CUDA matrix multiplication.
// Multiplying a matrix of [A] M x K and [B] K x N dimensions.
// The resulting matrix [C] will be of M x N dimensions.

__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K,
                                     float alpha, float beta) {
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

        // C = alpha * (A*B) + beta * C
        c[x * N + y] = alpha * sum + beta * c[x * N + y];
    }
}

#endif // NAIVE_KERNEL_H
