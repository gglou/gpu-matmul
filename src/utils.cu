#include "utils.h"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

// ============================================================================
// Initialization
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
// Verification
// ============================================================================

bool verify_results(float *gpu_result, float *cpu_result, int size, float tolerance) {
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
