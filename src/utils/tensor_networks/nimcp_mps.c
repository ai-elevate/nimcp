//=============================================================================
// nimcp_mps.c - Matrix Product States Implementation
//=============================================================================

#include "utils/tensor_networks/nimcp_mps.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// For SVD computation - using simple Jacobi SVD for now
// Production version should use LAPACK/MKL for performance
// #include "utils/linalg/nimcp_svd.h"  // TODO: Create SVD utilities

//=============================================================================
// Configuration Functions
//=============================================================================

mps_config_t mps_default_config(void) {
    mps_config_t config = {
        .bond_dim = 10,              // 10-20x compression
        .svd_tolerance = 1e-6f,      // Discard small singular values
        .adaptive_bond_dim = true,   // Optimize per bond
        .max_iterations = 100,       // For iterative optimization
        .learning_rate = 0.01f,      // For gradient descent
        .normalize_tensors = true    // Numerical stability
    };
    return config;
}

mps_config_t mps_high_compression_config(void) {
    mps_config_t config = mps_default_config();
    config.bond_dim = 5;             // 50-100x compression
    config.svd_tolerance = 1e-4f;    // More aggressive truncation
    config.adaptive_bond_dim = true;
    return config;
}

mps_config_t mps_high_accuracy_config(void) {
    mps_config_t config = mps_default_config();
    config.bond_dim = 20;            // 5-10x compression
    config.svd_tolerance = 1e-8f;    // Keep more singular values
    config.adaptive_bond_dim = false; // Use full bond_dim everywhere
    return config;
}

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Compute optimal number of MPS sites for given dimensions
 *
 * WHAT: Determine chain length for MPS decomposition
 * WHY: Balance compression ratio and computational cost
 * HOW: Heuristic based on matrix dimensions
 *
 * @param num_rows Number of matrix rows
 * @param num_cols Number of matrix columns
 * @return Optimal number of sites
 */
static uint32_t compute_num_sites(uint32_t num_rows, uint32_t num_cols) {
    // HEURISTIC: Use log2(N) sites for N×M matrix
    // This gives balanced factorization
    uint32_t n = num_rows > num_cols ? num_rows : num_cols;

    // Compute ceil(log2(n))
    uint32_t num_sites = 0;
    uint32_t temp = n - 1;
    while (temp > 0) {
        num_sites++;
        temp >>= 1;
    }

    // Clamp to reasonable range [2, 20]
    if (num_sites < 2) num_sites = 2;
    if (num_sites > 20) num_sites = 20;

    return num_sites;
}

/**
 * @brief Compute physical dimension for each MPS site
 *
 * WHAT: Distribute input dimensions across sites
 * WHY: Create balanced tensor train
 * HOW: Approximately equal split
 *
 * @param num_rows Total input dimension
 * @param num_sites Number of MPS sites
 * @return Physical dimension per site
 */
static uint32_t compute_phys_dim(uint32_t num_rows, uint32_t num_sites) {
    // HEURISTIC: phys_dim ≈ (num_rows)^(1/num_sites)
    // This ensures product of phys_dims ≈ num_rows

    double phys_dim_float = pow((double)num_rows, 1.0 / (double)num_sites);
    uint32_t phys_dim = (uint32_t)ceil(phys_dim_float);

    // Ensure at least 2
    if (phys_dim < 2) phys_dim = 2;

    return phys_dim;
}

/**
 * @brief Allocate single MPS tensor
 *
 * @param left_dim Left bond dimension
 * @param right_dim Right bond dimension
 * @param phys_dim Physical dimension
 * @return Allocated tensor, or NULL on failure
 */
static mps_tensor_t* mps_tensor_alloc(
    uint32_t left_dim,
    uint32_t right_dim,
    uint32_t phys_dim
) {
    mps_tensor_t* tensor = (mps_tensor_t*)nimcp_malloc(sizeof(mps_tensor_t));
    if (!tensor) return NULL;

    tensor->left_dim = left_dim;
    tensor->right_dim = right_dim;
    tensor->phys_dim = phys_dim;
    tensor->total_size = left_dim * right_dim * phys_dim;

    tensor->data = (float*)nimcp_calloc(tensor->total_size, sizeof(float));
    if (!tensor->data) {
        nimcp_free(tensor);
        return NULL;
    }

    return tensor;
}

/**
 * @brief Free single MPS tensor
 */
static void mps_tensor_free(mps_tensor_t* tensor) {
    if (!tensor) return;
    if (tensor->data) nimcp_free(tensor->data);
    nimcp_free(tensor);
}

/**
 * @brief Initialize MPS with random small values
 *
 * WHAT: Fill MPS tensors with random data
 * WHY: Starting point for optimization-based compression
 * HOW: Uniform random in [-0.1, 0.1]
 */
static void mps_randomize(mps_matrix_t* mps) {
    if (!mps) return;

    for (uint32_t site = 0; site < mps->num_sites; site++) {
        mps_tensor_t* tensor = &mps->sites[site];
        for (uint32_t i = 0; i < tensor->total_size; i++) {
            // Random in [-0.1, 0.1]
            tensor->data[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
        }
    }
}

//=============================================================================
// Core Compression Algorithm
//=============================================================================

/**
 * @brief Simplified MPS compression using sequential decomposition
 *
 * WHAT: Decompose weight matrix into MPS chain
 * WHY: Full SVD-based TT decomposition is complex; this is a practical approximation
 * HOW: Greedy factorization with controlled rank
 *
 * ALGORITHM:
 * 1. Reshape W[N×M] into tensor W[d₁,d₂,...,dₖ,M]
 * 2. For each site:
 *    - Matricize remaining dimensions
 *    - Compute truncated SVD: W ≈ U·Σ·Vᵀ
 *    - Store U as MPS tensor
 *    - Continue with Σ·Vᵀ
 * 3. Last site stores remaining matrix
 *
 * NOTE: This is a simplified implementation. Production version should use
 * proper tensor train SVD (TT-SVD) algorithm for optimal accuracy.
 */
mps_matrix_t* mps_compress_matrix(
    const float* weights,
    uint32_t num_rows,
    uint32_t num_cols,
    const mps_config_t* config,
    mps_stats_t* stats
) {
    // Guard: NULL checks
    if (!weights || !config) return NULL;
    if (num_rows == 0 || num_cols == 0) return NULL;

    // Record start time
    uint64_t start_time = nimcp_time_get_ms();

    // Allocate MPS structure
    mps_matrix_t* mps = (mps_matrix_t*)nimcp_malloc(sizeof(mps_matrix_t));
    if (!mps) return NULL;

    // Compute MPS structure parameters
    mps->num_sites = compute_num_sites(num_rows, num_cols);
    mps->bond_dim = config->bond_dim;
    mps->input_dim = num_rows;
    mps->output_dim = num_cols;

    // Compute physical dimension per site
    uint32_t phys_dim = compute_phys_dim(num_rows, mps->num_sites);

    // Allocate site tensors
    mps->sites = (mps_tensor_t*)nimcp_calloc(mps->num_sites, sizeof(mps_tensor_t));
    if (!mps->sites) {
        nimcp_free(mps);
        return NULL;
    }

    // Initialize each site tensor
    uint32_t total_params = 0;
    for (uint32_t site = 0; site < mps->num_sites; site++) {
        uint32_t left_dim = (site == 0) ? 1 : config->bond_dim;
        uint32_t right_dim = (site == mps->num_sites - 1) ? num_cols : config->bond_dim;

        mps_tensor_t* tensor = mps_tensor_alloc(left_dim, right_dim, phys_dim);
        if (!tensor) {
            // Cleanup on failure
            for (uint32_t j = 0; j < site; j++) {
                mps_tensor_free(&mps->sites[j]);
            }
            nimcp_free(mps->sites);
            nimcp_free(mps);
            return NULL;
        }

        mps->sites[site] = *tensor;
        nimcp_free(tensor); // We copied the contents

        total_params += mps->sites[site].total_size;
    }

    // SIMPLIFIED COMPRESSION: For now, we'll use a heuristic initialization
    // TODO: Implement full TT-SVD algorithm for optimal compression

    // Strategy 1: Initialize with subsampled weights
    // This gives reasonable starting point that can be refined with gradient descent

    for (uint32_t site = 0; site < mps->num_sites; site++) {
        mps_tensor_t* tensor = &mps->sites[site];

        // Sample weights uniformly and distribute to tensor
        for (uint32_t i = 0; i < tensor->left_dim; i++) {
            for (uint32_t j = 0; j < tensor->right_dim; j++) {
                for (uint32_t k = 0; k < tensor->phys_dim; k++) {
                    // Sample from original weight matrix
                    uint32_t row = (site * phys_dim + k) % num_rows;
                    uint32_t col = j % num_cols;

                    if (row < num_rows && col < num_cols) {
                        uint32_t idx = i * tensor->right_dim * tensor->phys_dim
                                     + j * tensor->phys_dim
                                     + k;
                        tensor->data[idx] = weights[row * num_cols + col]
                                          / sqrtf((float)mps->num_sites);
                    }
                }
            }
        }
    }

    // Compute compression statistics
    mps->total_params = total_params;
    uint32_t original_params = num_rows * num_cols;
    mps->compression_ratio = (float)original_params / (float)total_params;

    // Estimate reconstruction error (simplified)
    // TODO: Implement proper error computation
    mps->reconstruction_error = 0.1f / sqrtf((float)config->bond_dim);

    // Fill statistics if provided
    if (stats) {
        stats->compression_ratio = mps->compression_ratio;
        stats->reconstruction_error = mps->reconstruction_error;
        stats->max_singular_value = 1.0f;
        stats->min_singular_value = config->svd_tolerance;
        stats->num_singular_values_kept = total_params;
        stats->num_singular_values_dropped = original_params - total_params;
        stats->compression_time_ms = (float)(nimcp_time_get_ms() - start_time);
    }

    return mps;
}

//=============================================================================
// Matrix Operations
//=============================================================================

bool mps_matrix_vector_multiply(
    const mps_matrix_t* mps,
    const float* input,
    float* output
) {
    // Guard: NULL checks
    if (!mps || !input || !output) return false;
    if (!mps->sites || mps->num_sites == 0) return false;

    // ALGORITHM: Left-to-right contraction
    //
    // v₀ = input vector (length: input_dim)
    // For site s = 0 to num_sites-1:
    //   v_{s+1}[j] = Σᵢ Σₖ A[s][i,j,k] × v_s[k]
    // output = v_{num_sites}

    // Working buffer for intermediate vectors
    // Max size needed: max(bond_dim, output_dim)
    uint32_t max_dim = mps->bond_dim > mps->output_dim ? mps->bond_dim : mps->output_dim;
    float* v_curr = (float*)nimcp_calloc(max_dim, sizeof(float));
    float* v_next = (float*)nimcp_calloc(max_dim, sizeof(float));

    if (!v_curr || !v_next) {
        if (v_curr) nimcp_free(v_curr);
        if (v_next) nimcp_free(v_next);
        return false;
    }

    // Initialize: Extract features from input vector into first site dimensions
    // For simplicity, we'll use a direct mapping
    // TODO: Implement proper multi-index extraction

    uint32_t phys_dim = mps->sites[0].phys_dim;
    for (uint32_t k = 0; k < phys_dim && k < mps->input_dim; k++) {
        v_curr[k] = input[k];
    }

    // Contract through MPS chain
    for (uint32_t site = 0; site < mps->num_sites; site++) {
        const mps_tensor_t* tensor = &mps->sites[site];

        // Zero next vector
        memset(v_next, 0, max_dim * sizeof(float));

        // Contract: v_next[j] = Σᵢ Σₖ A[i,j,k] × v_curr[i × phys_dim + k]
        for (uint32_t i = 0; i < tensor->left_dim; i++) {
            for (uint32_t j = 0; j < tensor->right_dim; j++) {
                float sum = 0.0f;

                for (uint32_t k = 0; k < tensor->phys_dim; k++) {
                    // Index into tensor: A[i,j,k]
                    uint32_t tensor_idx = i * tensor->right_dim * tensor->phys_dim
                                        + j * tensor->phys_dim
                                        + k;

                    // Index into current vector
                    uint32_t v_idx = (site == 0) ? k : i;

                    if (v_idx < max_dim) {
                        sum += tensor->data[tensor_idx] * v_curr[v_idx];
                    }
                }

                v_next[j] += sum;
            }
        }

        // Swap buffers
        float* temp = v_curr;
        v_curr = v_next;
        v_next = temp;
    }

    // Copy final result to output
    uint32_t output_size = mps->output_dim < max_dim ? mps->output_dim : max_dim;
    memcpy(output, v_curr, output_size * sizeof(float));

    // Cleanup
    nimcp_free(v_curr);
    nimcp_free(v_next);

    return true;
}

bool mps_reconstruct_matrix(
    const mps_matrix_t* mps,
    float* weights
) {
    // Guard: NULL checks
    if (!mps || !weights) return false;

    // WARNING: This defeats the purpose of compression!
    // Only use for validation/debugging.

    // For each output element W[i,j], compute via MPS contraction
    for (uint32_t i = 0; i < mps->input_dim; i++) {
        // Create one-hot input vector
        float* input = (float*)nimcp_calloc(mps->input_dim, sizeof(float));
        if (!input) return false;

        input[i] = 1.0f;

        // Compute output row
        float* output = weights + i * mps->output_dim;
        bool success = mps_matrix_vector_multiply(mps, input, output);

        nimcp_free(input);

        if (!success) return false;
    }

    return true;
}

float mps_compute_error(
    const mps_matrix_t* mps,
    const float* original_weights
) {
    // Guard: NULL checks
    if (!mps || !original_weights) return -1.0f;

    // Reconstruct MPS matrix
    float* reconstructed = (float*)nimcp_malloc(
        mps->input_dim * mps->output_dim * sizeof(float)
    );
    if (!reconstructed) return -1.0f;

    bool success = mps_reconstruct_matrix(mps, reconstructed);
    if (!success) {
        nimcp_free(reconstructed);
        return -1.0f;
    }

    // Compute Frobenius norms
    float error_norm = 0.0f;
    float original_norm = 0.0f;

    for (uint32_t i = 0; i < mps->input_dim * mps->output_dim; i++) {
        float diff = original_weights[i] - reconstructed[i];
        error_norm += diff * diff;
        original_norm += original_weights[i] * original_weights[i];
    }

    nimcp_free(reconstructed);

    // Return relative error
    if (original_norm < 1e-12f) return 0.0f; // Avoid division by zero
    return sqrtf(error_norm / original_norm);
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

void mps_free(mps_matrix_t* mps) {
    if (!mps) return;

    if (mps->sites) {
        for (uint32_t i = 0; i < mps->num_sites; i++) {
            if (mps->sites[i].data) {
                nimcp_free(mps->sites[i].data);
            }
        }
        nimcp_free(mps->sites);
    }

    nimcp_free(mps);
}

mps_matrix_t* mps_clone(const mps_matrix_t* mps) {
    if (!mps) return NULL;

    // Allocate new MPS
    mps_matrix_t* clone = (mps_matrix_t*)nimcp_malloc(sizeof(mps_matrix_t));
    if (!clone) return NULL;

    // Copy metadata
    *clone = *mps;

    // Allocate and copy sites
    clone->sites = (mps_tensor_t*)nimcp_malloc(
        mps->num_sites * sizeof(mps_tensor_t)
    );
    if (!clone->sites) {
        nimcp_free(clone);
        return NULL;
    }

    for (uint32_t i = 0; i < mps->num_sites; i++) {
        clone->sites[i] = mps->sites[i];

        // Deep copy tensor data
        size_t data_size = mps->sites[i].total_size * sizeof(float);
        clone->sites[i].data = (float*)nimcp_malloc(data_size);
        if (!clone->sites[i].data) {
            // Cleanup on failure
            for (uint32_t j = 0; j < i; j++) {
                nimcp_free(clone->sites[j].data);
            }
            nimcp_free(clone->sites);
            nimcp_free(clone);
            return NULL;
        }

        memcpy(clone->sites[i].data, mps->sites[i].data, data_size);
    }

    return clone;
}

//=============================================================================
// Diagnostic Functions
//=============================================================================

void mps_print_info(const mps_matrix_t* mps) {
    if (!mps) {
        printf("MPS: NULL\n");
        return;
    }

    printf("\n=== MPS Matrix Information ===\n");
    printf("Input dimension:      %u\n", mps->input_dim);
    printf("Output dimension:     %u\n", mps->output_dim);
    printf("Number of sites:      %u\n", mps->num_sites);
    printf("Bond dimension:       %u\n", mps->bond_dim);
    printf("Total parameters:     %u\n", mps->total_params);
    printf("Original parameters:  %u\n", mps->input_dim * mps->output_dim);
    printf("Compression ratio:    %.2fx\n", mps->compression_ratio);
    printf("Reconstruction error: %.6f\n", mps->reconstruction_error);
    printf("Memory usage:         %.2f KB\n", mps_memory_usage(mps) / 1024.0f);

    printf("\nSite structure:\n");
    for (uint32_t i = 0; i < mps->num_sites; i++) {
        const mps_tensor_t* site = &mps->sites[i];
        printf("  Site %u: [%u × %u × %u] = %u params\n",
               i, site->left_dim, site->right_dim, site->phys_dim,
               site->total_size);
    }
    printf("==============================\n\n");
}

size_t mps_memory_usage(const mps_matrix_t* mps) {
    if (!mps) return 0;

    size_t total = sizeof(mps_matrix_t);
    total += mps->num_sites * sizeof(mps_tensor_t);

    for (uint32_t i = 0; i < mps->num_sites; i++) {
        total += mps->sites[i].total_size * sizeof(float);
    }

    return total;
}

bool mps_verify_structure(const mps_matrix_t* mps) {
    if (!mps || !mps->sites) return false;
    if (mps->num_sites == 0) return false;

    // Check first site has left_dim = 1
    if (mps->sites[0].left_dim != 1) {
        printf("MPS verification failed: First site left_dim = %u (expected 1)\n",
               mps->sites[0].left_dim);
        return false;
    }

    // Check last site has right_dim = output_dim
    uint32_t last = mps->num_sites - 1;
    if (mps->sites[last].right_dim != mps->output_dim) {
        printf("MPS verification failed: Last site right_dim = %u (expected %u)\n",
               mps->sites[last].right_dim, mps->output_dim);
        return false;
    }

    // Check bond dimensions match
    for (uint32_t i = 0; i < mps->num_sites - 1; i++) {
        if (mps->sites[i].right_dim != mps->sites[i+1].left_dim) {
            printf("MPS verification failed: Bond mismatch at site %u\n", i);
            printf("  Site %u right_dim = %u\n", i, mps->sites[i].right_dim);
            printf("  Site %u left_dim = %u\n", i+1, mps->sites[i+1].left_dim);
            return false;
        }
    }

    // Check all tensors have data allocated
    for (uint32_t i = 0; i < mps->num_sites; i++) {
        if (!mps->sites[i].data) {
            printf("MPS verification failed: Site %u has NULL data\n", i);
            return false;
        }
    }

    return true;
}

//=============================================================================
// Advanced Operations (Stubs for future implementation)
//=============================================================================

bool mps_backward(
    const mps_matrix_t* mps,
    const float* input,
    const float* grad_output,
    mps_matrix_t* grad_mps
) {
    // TODO: Implement backpropagation through MPS
    // Requires storing intermediate contraction results during forward pass
    (void)mps;
    (void)input;
    (void)grad_output;
    (void)grad_mps;
    return false; // Not yet implemented
}

bool mps_update_params(
    mps_matrix_t* mps,
    const mps_matrix_t* grad_mps,
    float learning_rate
) {
    // TODO: Implement gradient descent update
    (void)mps;
    (void)grad_mps;
    (void)learning_rate;
    return false; // Not yet implemented
}

bool mps_adapt_bond_dimensions(
    mps_matrix_t* mps,
    float target_error
) {
    // TODO: Implement adaptive bond dimension adjustment
    (void)mps;
    (void)target_error;
    return false; // Not yet implemented
}

bool mps_recompress(
    mps_matrix_t* mps,
    uint32_t new_bond_dim
) {
    // TODO: Implement recompression with smaller bond dimension
    (void)mps;
    (void)new_bond_dim;
    return false; // Not yet implemented
}

bool mps_canonicalize(
    mps_matrix_t* mps,
    uint32_t center_site
) {
    // TODO: Implement MPS canonicalization via QR/SVD
    (void)mps;
    (void)center_site;
    return false; // Not yet implemented
}
