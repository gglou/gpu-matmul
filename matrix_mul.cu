#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>

// ============================================================================
// Data Structures
// ============================================================================

struct BenchmarkResult {
    double avg_time;
    double min_time;
    double max_time;
    int num_runs;
};

struct MatrixDims {
    int M, N, K;  // M x K * K x N = M x N
};

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

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

// ============================================================================
// Initialization and Memory Management
// ============================================================================

void initialize_matrices(float *h_a, float *h_b, const MatrixDims &dims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Initialize matrix A (M x K)
    for (int i = 0; i < dims.M * dims.K; ++i) {
        h_a[i] = dis(gen);
    }
    
    // Initialize matrix B (K x N)
    for (int i = 0; i < dims.K * dims.N; ++i) {
        h_b[i] = dis(gen);
    }
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

BenchmarkResult benchmark_gpu(float *d_a, float *d_b, float *d_c, 
                               const MatrixDims &dims, int num_runs = 100) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dims.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dims.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    naive_kernel_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, dims.M, dims.N, dims.K);
    cudaDeviceSynchronize();

    // Benchmark runs
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    std::cout << "Benchmarking GPU kernel over " << num_runs << " runs...\n";

    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);
        naive_kernel_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, dims.M, dims.N, dims.K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {total_time / num_runs, min_time, max_time, num_runs};
}

BenchmarkResult benchmark_cpu(float *h_a, float *h_b, float *h_c, 
                               const MatrixDims &dims, int num_runs = 10) {
    // Warm-up run
    cpu_matmul(h_a, h_b, h_c, dims.M, dims.N, dims.K);
    
    // Benchmark runs
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    
    std::cout << "Benchmarking CPU implementation over " << num_runs << " runs...\n";
    
    for (int run = 0; run < num_runs; ++run) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(h_a, h_b, h_c, dims.M, dims.N, dims.K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double milliseconds = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }
    
    return {total_time / num_runs, min_time, max_time, num_runs};
}

void print_benchmark_results(const BenchmarkResult &gpu_result, const BenchmarkResult &cpu_result) {
    std::cout << "\n=== GPU Benchmark Results ===\n";
    std::cout << "Average time: " << gpu_result.avg_time << " ms\n";
    std::cout << "Min time:     " << gpu_result.min_time << " ms\n";
    std::cout << "Max time:     " << gpu_result.max_time << " ms\n";
    std::cout << "=============================\n\n";
    
    std::cout << "=== CPU Benchmark Results ===\n";
    std::cout << "Average time: " << cpu_result.avg_time << " ms\n";
    std::cout << "Min time:     " << cpu_result.min_time << " ms\n";
    std::cout << "Max time:     " << cpu_result.max_time << " ms\n";
    std::cout << "=============================\n\n";
    
    float speedup = cpu_result.avg_time / gpu_result.avg_time;
    std::cout << "=== Performance Comparison ===\n";
    std::cout << "GPU Average: " << gpu_result.avg_time << " ms\n";
    std::cout << "CPU Average: " << cpu_result.avg_time << " ms\n";
    std::cout << "Speedup:     " << speedup << "x\n";
    std::cout << "==============================\n\n";
}

// ============================================================================
// Verification
// ============================================================================

bool verify_results(float *gpu_result, float *cpu_result, int size, float tolerance = 1e-3f) {
    float max_error = 0.0f;
    bool passed = true;
    
    for (int i = 0; i < size; ++i) {
        float error = std::abs(gpu_result[i] - cpu_result[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cout << "❌ Verification FAILED!\n";
            std::cout << "Mismatch at index " << i << ": GPU=" << gpu_result[i] 
                      << " CPU=" << cpu_result[i] << " error=" << error << "\n";
            passed = false;
            break;
        }
    }
    
    if (passed) {
        std::cout << "✅ Verification PASSED! Max error: " << max_error << "\n\n";
    }
    
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Matrix dimensions: M x K * K x N = M x N
    MatrixDims dims = {1000, 900, 800};
    
    std::cout << "Matrix multiplication: (" << dims.M << " x " << dims.K << ") * (" 
              << dims.K << " x " << dims.N << ") = (" << dims.M << " x " << dims.N << ")\n\n";

    // Allocate host memory
    float *h_a = (float*)malloc(sizeof(float) * dims.M * dims.K);
    float *h_b = (float*)malloc(sizeof(float) * dims.K * dims.N);
    float *h_c = (float*)malloc(sizeof(float) * dims.M * dims.N);
    float *h_c_expected = (float*)malloc(sizeof(float) * dims.M * dims.N);

    // Initialize matrices with random values
    initialize_matrices(h_a, h_b, dims);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * dims.M * dims.K);
    cudaMalloc((void**)&d_b, sizeof(float) * dims.K * dims.N);
    cudaMalloc((void**)&d_c, sizeof(float) * dims.M * dims.N);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, sizeof(float) * dims.M * dims.K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * dims.K * dims.N, cudaMemcpyHostToDevice);

    // Benchmark GPU
    BenchmarkResult gpu_result = benchmark_gpu(d_a, d_b, d_c, dims);
    cudaMemcpy(h_c, d_c, sizeof(float) * dims.M * dims.N, cudaMemcpyDeviceToHost);

    // Benchmark CPU
    BenchmarkResult cpu_result = benchmark_cpu(h_a, h_b, h_c_expected, dims);

    // Print results
    print_benchmark_results(gpu_result, cpu_result);

    // Verify correctness
    verify_results(h_c, h_c_expected, dims.M * dims.N);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_expected);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
