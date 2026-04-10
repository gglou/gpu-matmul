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
    float max_error = 0.0f; // tracks max relative error
    bool passed = true;
    
    for (int i = 0; i < size; ++i) {
        float abs_error = std::abs(gpu_result[i] - reference_result[i]);
        float denom = std::max({std::abs(gpu_result[i]), std::abs(reference_result[i]), 1e-8f});
        float rel_error = abs_error / denom;
        max_error = std::max(max_error, rel_error);
        if (rel_error > tolerance) {
            std::cout << "❌ Verification FAILED!\n";
            std::cout.precision(10);
            std::cout << std::fixed;
            std::cout << "Mismatch at index " << i << ": kernel=" << gpu_result[i] 
                      << " " << reference_name << "=" << reference_result[i] 
                      << " abs_error=" << abs_error << " rel_error=" << rel_error << "\n";
            passed = false;
            break;
        }
    }
    
    if (passed) {
        std::cout << "✅ Verification PASSED (vs " << reference_name << ")! Max relative error: " << max_error << "\n\n";
    }
    
    return passed;
}
