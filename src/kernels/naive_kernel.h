#ifndef NAIVE_KERNEL_H
#define NAIVE_KERNEL_H

// Naive kernel for CUDA matrix multiplication.
// Multiplying a matrix of [A] M x K and [B] K x N dimensions.
// The resulting matrix [C] will be of M x N dimensions.

__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K);

#endif // NAIVE_KERNEL_H
