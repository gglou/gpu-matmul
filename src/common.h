#ifndef COMMON_H
#define COMMON_H

// ============================================================================
// Shared Data Structures
// ============================================================================

struct BenchmarkResult {
    const char* kernel_name;
    double avg_time;
    double min_time;
    double max_time;
    int num_runs;
};

struct MatrixDims {
    int M, N, K;  // M x K * K x N = M x N
};

// Kernel function pointer type
typedef void (*MatmulKernel)(float*, float*, float*, int, int, int);

#endif // COMMON_H
