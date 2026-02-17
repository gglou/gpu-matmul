#ifndef SHARED_MEM_KERNEL_H
#define SHARED_MEM_KERNEL_H

__global__ void shared_mem_kernel(float *a, float *b, float *c, int M, int N, int K);

#endif // SHARED_MEM_KERNEL_H
