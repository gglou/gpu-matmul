#ifndef UTILS_H
#define UTILS_H

#include "common.h"

// ============================================================================
// Initialization
// ============================================================================

void initialize_matrices(float *h_a, float *h_b, const MatrixDims &dims);

// ============================================================================
// Verification
// ============================================================================

bool verify_results(float *gpu_result, float *reference_result, int size, 
                    const char* reference_name = "reference", float tolerance = 1e-3f);

#endif // UTILS_H
