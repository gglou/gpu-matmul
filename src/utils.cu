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

bool verify_results(float *gpu_result, float *reference_result, int size, 
                    const char* reference_name, float tolerance) {
    float max_error = 0.0f;
    bool passed = true;
    
    for (int i = 0; i < size; ++i) {
        float error = std::abs(gpu_result[i] - reference_result[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cout << "❌ Verification FAILED!\n";
            std::cout.precision(10);
            std::cout << std::fixed;
            std::cout << "Mismatch at index " << i << ": kernel=" << gpu_result[i] 
                      << " " << reference_name << "=" << reference_result[i] 
                      << " error=" << error << "\n";
            passed = false;
            break;
        }
    }
    
    if (passed) {
        std::cout << "✅ Verification PASSED (vs " << reference_name << ")! Max error: " << max_error << "\n\n";
    }
    
    return passed;
}
