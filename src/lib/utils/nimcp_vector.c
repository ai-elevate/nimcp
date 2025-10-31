/**
 * @file nimcp_vector.c
 * @brief Vector mathematics utilities implementation
 *
 * WHAT: Fast vector operations for neural network computations
 * WHY: Centralize common vector math to eliminate code duplication
 * HOW: Optimized implementations with numerical stability
 */

#include "utils/nimcp_vector.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Basic Vector Operations
//=============================================================================

float nimcp_vector_dot_product(const float* a, const float* b, uint32_t size) {
    if (!a || !b || size == 0) {
        return 0.0f;
    }

    float dot = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        dot += a[i] * b[i];
    }

    return dot;
}

float nimcp_vector_norm_l2(const float* vec, uint32_t size) {
    if (!vec || size == 0) {
        return 0.0f;
    }

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        sum_sq += vec[i] * vec[i];
    }

    return sqrtf(sum_sq);
}

float nimcp_vector_norm_l1(const float* vec, uint32_t size) {
    if (!vec || size == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        sum += fabsf(vec[i]);
    }

    return sum;
}

void nimcp_vector_copy(const float* src, float* dst, uint32_t size) {
    if (!src || !dst || size == 0) {
        return;
    }

    memcpy(dst, src, size * sizeof(float));
}

//=============================================================================
// Similarity and Distance Metrics
//=============================================================================

float nimcp_vector_cosine_similarity(const float* a, const float* b, uint32_t size) {
    if (!a || !b || size == 0) {
        return 0.0f;
    }

    /**
     * WHAT: Single-pass computation of dot product and norms
     * WHY: More efficient than separate passes
     * HOW: Accumulate all values in one loop
     */
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < size; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    /**
     * WHAT: Compute denominator with numerical stability
     * WHY: Avoid division by zero or very small numbers
     * HOW: Check for zero vectors and use epsilon guard
     */
    float denom = sqrtf(norm_a) * sqrtf(norm_b);

    if (denom < NIMCP_VECTOR_EPSILON) {
        /**
         * WHAT: Handle special cases
         * WHY: Define behavior for zero vectors
         * HOW:
         *  - Both zero: Perfect match (1.0)
         *  - One zero, one non-zero: No similarity (0.0)
         */
        if (norm_a < NIMCP_VECTOR_EPSILON && norm_b < NIMCP_VECTOR_EPSILON) {
            return 1.0f;  /* Both zero = perfect match */
        }
        return 0.0f;  /* One zero, one non-zero = no similarity */
    }

    return dot_product / denom;
}

float nimcp_vector_cosine_distance(const float* a, const float* b, uint32_t size) {
    /**
     * WHAT: Cosine distance = 1 - cosine similarity
     * WHY: Convert similarity metric to distance metric
     * HOW: Simple transformation
     *
     * RANGE: [0, 2]
     *   0.0 = identical direction
     *   1.0 = orthogonal
     *   2.0 = opposite direction
     */
    return 1.0f - nimcp_vector_cosine_similarity(a, b, size);
}

float nimcp_vector_euclidean_distance(const float* a, const float* b, uint32_t size) {
    if (!a || !b || size == 0) {
        return 0.0f;
    }

    float sum_sq_diff = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }

    return sqrtf(sum_sq_diff);
}

//=============================================================================
// Normalization Functions
//=============================================================================

float nimcp_vector_normalize_l2(float* vec, uint32_t size, float target_norm) {
    if (!vec || size == 0) {
        return 0.0f;
    }

    /**
     * WHAT: Compute current L2 norm
     * WHY: Need to know scale factor
     */
    float current_norm = nimcp_vector_norm_l2(vec, size);

    /**
     * WHAT: Check if vector is effectively zero
     * WHY: Can't normalize zero vector
     * HOW: Return 0.0 and leave vector unchanged
     */
    if (current_norm < NIMCP_VECTOR_EPSILON) {
        return 0.0f;
    }

    /**
     * WHAT: Scale vector to target norm
     * WHY: Preserve direction, change magnitude
     * HOW: vec[i] = vec[i] * (target_norm / current_norm)
     */
    float scale = target_norm / current_norm;
    for (uint32_t i = 0; i < size; i++) {
        vec[i] *= scale;
    }

    return current_norm;
}

float nimcp_vector_normalize_l1(float* vec, uint32_t size, float target_norm) {
    if (!vec || size == 0) {
        return 0.0f;
    }

    /**
     * WHAT: Compute current L1 norm
     * WHY: Need to know scale factor
     */
    float current_norm = nimcp_vector_norm_l1(vec, size);

    /**
     * WHAT: Check if vector is effectively zero
     * WHY: Can't normalize zero vector
     * HOW: Return 0.0 and leave vector unchanged
     */
    if (current_norm < NIMCP_VECTOR_EPSILON) {
        return 0.0f;
    }

    /**
     * WHAT: Scale vector to target norm
     * WHY: Preserve signs, change absolute sum
     * HOW: vec[i] = vec[i] * (target_norm / current_norm)
     */
    float scale = target_norm / current_norm;
    for (uint32_t i = 0; i < size; i++) {
        vec[i] *= scale;
    }

    return current_norm;
}
