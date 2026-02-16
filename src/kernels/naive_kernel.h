#ifndef NAIVE_KERNEL_H
#define NAIVE_KERNEL_H

// ============================================================================
// Naive GPU Kernel
// ============================================================================

// Naive kernel for CUDA matrix multiplication.
// Multiplying a matrix of [A] M x K and [B] K x N dimensions.
// The resulting matrix [C] will be of M x N dimensions.
//
// Each thread computes one element of C.
// Performance characteristics:
// - 2 global memory accesses per iteration
// - a[row * K + i] accesses may be coalesced (contiguous memory)
// - b[i * N + col] accesses are strided (not coalesced)
__global__ void naive_kernel_matmul(float *a, float *b, float *c, int M, int N, int K);

#endif // NAIVE_KERNEL_H
