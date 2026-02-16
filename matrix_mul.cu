#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>

// CPU matrix multiplication for verification
// C = A * B where A is M x K, B is K x N, C is M x N
void cpu_matmul(float *a, float *b, float *c, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += a[row * K + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

// Naive kernel for CUDA kernel for matrix multiplication.
// Multiplying a matrix of [A] M x K and [B] K x N dimensions.
// The resulting matrix [C] will be of M x N dimensions.
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

int main() {
    // Matrix dimensions, are M x K and K x N.
    const int N = 900;
    const int M = 1000;
    const int K = 800;

    // Allocate host memory
    float *h_a = (float*)malloc(sizeof(float) * M * K);
    float *h_b = (float*)malloc(sizeof(float) * N * K);
    float *h_c = (float*)malloc(sizeof(float) * M * N);
    float *h_c_expected = (float*)malloc(sizeof(float) * M * N);

    // Initialize host input data
    // Random number generator with uniform distribution [0, 1]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Initialize matrix A (M x K)
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = dis(gen);
    }
    
    // Initialize matrix B (K x N)
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = dis(gen);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * M * K);  // Matrix A: M x K
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);  // Matrix B: K x N
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);  // Matrix C: M x N

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run (first run is often slower due to GPU initialization)
    naive_kernel_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark the kernel with multiple runs
    const int num_runs = 100;
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    std::cout << "Benchmarking kernel over " << num_runs << " runs...\n";

    for (int run = 0; run < num_runs; ++run) {
        // Start timing
        cudaEventRecord(start);
        
        // Launch the kernel
        naive_kernel_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }

    float avg_time = total_time / num_runs;

    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << "Average time: " << avg_time << " ms\n";
    std::cout << "Min time:     " << min_time << " ms\n";
    std::cout << "Max time:     " << max_time << " ms\n";
    std::cout << "=========================\n\n";

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results from device to host
    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Compute CPU reference result
    cpu_matmul(h_a, h_b, h_c_expected, M, N, K);

    // Verify results
    float max_error = 0.0f;
    const float tolerance = 1e-3f;  // Allow small floating point differences
    
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_c[i] - h_c_expected[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cout << "Mismatch at index " << i << ": GPU=" << h_c[i] 
                      << " CPU=" << h_c_expected[i] << " error=" << error << "\n";
            break;
        }
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_expected);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}