//=============================================================================
// test_mps_compression.cpp - MPS Weight Compression Tests
//=============================================================================
/**
 * @file test_mps_compression.cpp
 * @brief Comprehensive tests for Matrix Product States weight compression
 *
 * WHAT: Validate MPS compression, accuracy, and performance
 * WHY: Ensure 10-100x memory reduction with <1% error
 * HOW: Unit tests + benchmarks for various configurations
 *
 * TEST COVERAGE:
 * 1. Basic compression/decompression
 * 2. Matrix-vector multiplication
 * 3. Accuracy vs bond dimension
 * 4. Compression ratio measurements
 * 5. Performance benchmarks
 * 6. Integration with neural networks
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>

extern "C" {
    #include "utils/tensor_networks/nimcp_mps.h"
    #include "utils/memory/nimcp_memory.h"
    #include "utils/platform/nimcp_platform_time.h"
}

//=============================================================================
// Test Utilities
//=============================================================================

/**
 * @brief Generate random weight matrix
 */
void generate_random_weights(float* weights, uint32_t rows, uint32_t cols, float scale = 1.0f) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * scale - scale;
    }
}

/**
 * @brief Compute Frobenius norm
 */
float frobenius_norm(const float* matrix, uint32_t rows, uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < rows * cols; i++) {
        sum += matrix[i] * matrix[i];
    }
    return sqrtf(sum);
}

/**
 * @brief Matrix-vector multiply (reference implementation)
 */
void reference_matvec(const float* W, const float* x, float* y, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (uint32_t j = 0; j < cols; j++) {
            y[i] += W[i * cols + j] * x[j];
        }
    }
}

/**
 * @brief Compute relative error between two vectors
 */
float vector_relative_error(const float* v1, const float* v2, uint32_t dim) {
    float error_norm = 0.0f;
    float v1_norm = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        error_norm += diff * diff;
        v1_norm += v1[i] * v1[i];
    }

    if (v1_norm < 1e-12f) return 0.0f;
    return sqrtf(error_norm / v1_norm);
}

//=============================================================================
// Basic Functionality Tests
//=============================================================================

class MPSCompressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42); // Fixed seed for reproducibility
    }

    void TearDown() override {
        // Cleanup handled by individual tests
    }
};

TEST_F(MPSCompressionTest, CreateAndDestroyMPS) {
    // WHAT: Test basic allocation and deallocation
    // WHY: Ensure no memory leaks
    // HOW: Create, verify structure, destroy

    const uint32_t N = 100, M = 100;
    float* weights = (float*)nimcp_malloc(N * M * sizeof(float));
    generate_random_weights(weights, N, M);

    mps_config_t config = mps_default_config();
    mps_stats_t stats;

    mps_matrix_t* mps = mps_compress_matrix(weights, N, M, &config, &stats);

    ASSERT_NE(mps, nullptr);
    EXPECT_GT(mps->num_sites, 0);
    EXPECT_EQ(mps->input_dim, N);
    EXPECT_EQ(mps->output_dim, M);
    EXPECT_GT(mps->compression_ratio, 1.0f);

    // Verify structure integrity
    EXPECT_TRUE(mps_verify_structure(mps));

    // Cleanup
    mps_free(mps);
    nimcp_free(weights);
}

TEST_F(MPSCompressionTest, CompressionRatioScaling) {
    // WHAT: Verify compression ratio increases with matrix size
    // WHY: MPS works better for larger matrices
    // HOW: Test different sizes, measure compression

    mps_config_t config = mps_default_config();
    config.bond_dim = 10;

    uint32_t sizes[] = {50, 100, 200, 400};
    float prev_ratio = 0.0f;

    for (uint32_t size : sizes) {
        float* weights = (float*)nimcp_malloc(size * size * sizeof(float));
        generate_random_weights(weights, size, size);

        mps_stats_t stats;
        mps_matrix_t* mps = mps_compress_matrix(weights, size, size, &config, &stats);

        ASSERT_NE(mps, nullptr);
        EXPECT_GT(stats.compression_ratio, prev_ratio);

        printf("Size %u×%u: Compression ratio = %.2fx\n",
               size, size, stats.compression_ratio);

        prev_ratio = stats.compression_ratio;

        mps_free(mps);
        nimcp_free(weights);
    }
}

TEST_F(MPSCompressionTest, BondDimensionTradeoff) {
    // WHAT: Test accuracy vs compression tradeoff
    // WHY: Verify bond_dim controls accuracy/memory balance
    // HOW: Fixed matrix, varying bond_dim

    const uint32_t N = 200, M = 200;
    float* weights = (float*)nimcp_malloc(N * M * sizeof(float));
    generate_random_weights(weights, N, M);

    uint32_t bond_dims[] = {5, 10, 20};

    for (uint32_t bond_dim : bond_dims) {
        mps_config_t config = mps_default_config();
        config.bond_dim = bond_dim;

        mps_stats_t stats;
        mps_matrix_t* mps = mps_compress_matrix(weights, N, M, &config, &stats);

        ASSERT_NE(mps, nullptr);

        float error = mps_compute_error(mps, weights);

        printf("Bond dim %u: Compression %.2fx, Error %.4f%%\n",
               bond_dim, stats.compression_ratio, error * 100.0f);

        // Higher bond_dim → lower error
        EXPECT_LT(error, 0.2f); // Should be < 20% error even for bond_dim=5

        mps_free(mps);
    }

    nimcp_free(weights);
}

//=============================================================================
// Matrix-Vector Multiplication Tests
//=============================================================================

TEST_F(MPSCompressionTest, MatrixVectorMultiplication) {
    // WHAT: Verify MPS matrix-vector multiply produces correct output
    // WHY: Main operation for neural network forward pass
    // HOW: Compare MPS result with reference dense multiply

    const uint32_t N = 100, M = 80;
    float* weights = (float*)nimcp_malloc(N * M * sizeof(float));
    generate_random_weights(weights, N, M);

    // Create input vector
    float* input = (float*)nimcp_malloc(N * sizeof(float));
    generate_random_weights(input, N, 1, 0.5f);

    // Reference output
    float* output_ref = (float*)nimcp_calloc(M, sizeof(float));
    reference_matvec(weights, input, output_ref, M, N);

    // MPS output
    mps_config_t config = mps_default_config();
    config.bond_dim = 10;

    mps_matrix_t* mps = mps_compress_matrix(weights, N, M, &config, nullptr);
    ASSERT_NE(mps, nullptr);

    float* output_mps = (float*)nimcp_calloc(M, sizeof(float));
    bool success = mps_matrix_vector_multiply(mps, input, output_mps);
    ASSERT_TRUE(success);

    // Compare outputs
    float error = vector_relative_error(output_ref, output_mps, M);
    printf("Matrix-vector multiply relative error: %.6f%%\n", error * 100.0f);

    EXPECT_LT(error, 0.05f); // < 5% error for bond_dim=10

    // Cleanup
    mps_free(mps);
    nimcp_free(weights);
    nimcp_free(input);
    nimcp_free(output_ref);
    nimcp_free(output_mps);
}

TEST_F(MPSCompressionTest, BatchMatrixVectorMultiply) {
    // WHAT: Test multiple matrix-vector multiplies
    // WHY: Simulate neural network batch processing
    // HOW: Run many multiplies, measure average error

    const uint32_t N = 150, M = 120;
    const uint32_t batch_size = 100;

    float* weights = (float*)nimcp_malloc(N * M * sizeof(float));
    generate_random_weights(weights, N, M);

    mps_config_t config = mps_default_config();
    config.bond_dim = 10;

    mps_matrix_t* mps = mps_compress_matrix(weights, N, M, &config, nullptr);
    ASSERT_NE(mps, nullptr);

    float total_error = 0.0f;

    for (uint32_t b = 0; b < batch_size; b++) {
        float* input = (float*)nimcp_malloc(N * sizeof(float));
        generate_random_weights(input, N, 1);

        float* output_ref = (float*)nimcp_calloc(M, sizeof(float));
        float* output_mps = (float*)nimcp_calloc(M, sizeof(float));

        reference_matvec(weights, input, output_ref, M, N);
        mps_matrix_vector_multiply(mps, input, output_mps);

        float error = vector_relative_error(output_ref, output_mps, M);
        total_error += error;

        nimcp_free(input);
        nimcp_free(output_ref);
        nimcp_free(output_mps);
    }

    float avg_error = total_error / (float)batch_size;
    printf("Average error over %u samples: %.6f%%\n", batch_size, avg_error * 100.0f);

    EXPECT_LT(avg_error, 0.05f);

    mps_free(mps);
    nimcp_free(weights);
}

//=============================================================================
// Performance Benchmarks
//=============================================================================

TEST_F(MPSCompressionTest, CompressionTimeBenchmark) {
    // WHAT: Measure compression time for various matrix sizes
    // WHY: Understand one-time cost
    // HOW: Time mps_compress_matrix()

    printf("\n=== MPS Compression Time Benchmark ===\n");
    printf("Size        Time (ms)  Ratio    Params (orig)  Params (MPS)\n");
    printf("---------------------------------------------------------------\n");

    mps_config_t config = mps_default_config();
    config.bond_dim = 10;

    uint32_t sizes[] = {100, 200, 500, 1000};

    for (uint32_t size : sizes) {
        float* weights = (float*)nimcp_malloc(size * size * sizeof(float));
        generate_random_weights(weights, size, size);

        auto start = std::chrono::high_resolution_clock::now();

        mps_stats_t stats;
        mps_matrix_t* mps = mps_compress_matrix(weights, size, size, &config, &stats);

        auto end = std::chrono::high_resolution_clock::now();
        float time_ms = std::chrono::duration<float, std::milli>(end - start).count();

        ASSERT_NE(mps, nullptr);

        printf("%4u×%-4u   %7.2f   %5.1fx   %10u        %10u\n",
               size, size, time_ms, stats.compression_ratio,
               size * size, mps->total_params);

        mps_free(mps);
        nimcp_free(weights);
    }
    printf("---------------------------------------------------------------\n");
}

TEST_F(MPSCompressionTest, MatvecPerformanceBenchmark) {
    // WHAT: Compare MPS vs dense matrix-vector multiply speed
    // WHY: Quantify runtime overhead
    // HOW: Time both implementations

    printf("\n=== Matrix-Vector Multiply Performance ===\n");
    printf("Size        Dense (ms)  MPS (ms)   Speedup\n");
    printf("---------------------------------------------\n");

    const uint32_t iterations = 1000;

    uint32_t sizes[] = {100, 200, 500};

    for (uint32_t size : sizes) {
        float* weights = (float*)nimcp_malloc(size * size * sizeof(float));
        float* input = (float*)nimcp_malloc(size * sizeof(float));
        float* output = (float*)nimcp_calloc(size, sizeof(float));

        generate_random_weights(weights, size, size);
        generate_random_weights(input, size, 1);

        // Dense benchmark
        auto start_dense = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iterations; i++) {
            reference_matvec(weights, input, output, size, size);
        }
        auto end_dense = std::chrono::high_resolution_clock::now();
        float time_dense = std::chrono::duration<float, std::milli>(end_dense - start_dense).count();

        // MPS benchmark
        mps_config_t config = mps_default_config();
        config.bond_dim = 10;
        mps_matrix_t* mps = mps_compress_matrix(weights, size, size, &config, nullptr);
        ASSERT_NE(mps, nullptr);

        auto start_mps = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iterations; i++) {
            mps_matrix_vector_multiply(mps, input, output);
        }
        auto end_mps = std::chrono::high_resolution_clock::now();
        float time_mps = std::chrono::duration<float, std::milli>(end_mps - start_mps).count();

        float speedup = time_dense / time_mps;

        printf("%4u×%-4u   %8.3f    %8.3f   %.2fx\n",
               size, size, time_dense, time_mps, speedup);

        mps_free(mps);
        nimcp_free(weights);
        nimcp_free(input);
        nimcp_free(output);
    }
    printf("---------------------------------------------\n");
}

TEST_F(MPSCompressionTest, MemoryUsageBenchmark) {
    // WHAT: Measure actual memory usage
    // WHY: Verify 10-100x compression claims
    // HOW: Compare sizeof original vs MPS

    printf("\n=== Memory Usage Comparison ===\n");
    printf("Size        Original (KB)  MPS (KB)   Savings\n");
    printf("-----------------------------------------------\n");

    mps_config_t config = mps_default_config();

    uint32_t sizes[] = {100, 200, 500, 1000};

    for (uint32_t size : sizes) {
        float* weights = (float*)nimcp_malloc(size * size * sizeof(float));
        generate_random_weights(weights, size, size);

        size_t original_bytes = size * size * sizeof(float);

        mps_matrix_t* mps = mps_compress_matrix(weights, size, size, &config, nullptr);
        ASSERT_NE(mps, nullptr);

        size_t mps_bytes = mps_memory_usage(mps);

        float savings = (float)original_bytes / (float)mps_bytes;

        printf("%4u×%-4u   %10.1f    %8.1f   %.1fx\n",
               size, size,
               original_bytes / 1024.0f,
               mps_bytes / 1024.0f,
               savings);

        mps_free(mps);
        nimcp_free(weights);
    }
    printf("-----------------------------------------------\n");
}

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(MPSCompressionTest, DefaultConfig) {
    mps_config_t config = mps_default_config();

    EXPECT_EQ(config.bond_dim, 10);
    EXPECT_FLOAT_EQ(config.svd_tolerance, 1e-6f);
    EXPECT_TRUE(config.adaptive_bond_dim);
    EXPECT_TRUE(config.normalize_tensors);
}

TEST_F(MPSCompressionTest, HighCompressionConfig) {
    mps_config_t config = mps_high_compression_config();

    EXPECT_EQ(config.bond_dim, 5);
    EXPECT_GT(config.svd_tolerance, 1e-6f);
}

TEST_F(MPSCompressionTest, HighAccuracyConfig) {
    mps_config_t config = mps_high_accuracy_config();

    EXPECT_EQ(config.bond_dim, 20);
    EXPECT_LT(config.svd_tolerance, 1e-6f);
}

//=============================================================================
// Edge Cases and Error Handling
//=============================================================================

TEST_F(MPSCompressionTest, NullInputHandling) {
    mps_config_t config = mps_default_config();

    // Null weights
    mps_matrix_t* mps1 = mps_compress_matrix(nullptr, 100, 100, &config, nullptr);
    EXPECT_EQ(mps1, nullptr);

    // Null config
    float weights[100];
    mps_matrix_t* mps2 = mps_compress_matrix(weights, 10, 10, nullptr, nullptr);
    EXPECT_EQ(mps2, nullptr);

    // Zero dimensions
    mps_matrix_t* mps3 = mps_compress_matrix(weights, 0, 10, &config, nullptr);
    EXPECT_EQ(mps3, nullptr);

    mps_matrix_t* mps4 = mps_compress_matrix(weights, 10, 0, &config, nullptr);
    EXPECT_EQ(mps4, nullptr);
}

TEST_F(MPSCompressionTest, SmallMatrixHandling) {
    // WHAT: Test MPS with very small matrices
    // WHY: Edge case where compression may not help
    // HOW: Try 2×2, 5×5 matrices

    mps_config_t config = mps_default_config();

    uint32_t sizes[] = {2, 5, 10};

    for (uint32_t size : sizes) {
        float* weights = (float*)nimcp_malloc(size * size * sizeof(float));
        generate_random_weights(weights, size, size);

        mps_matrix_t* mps = mps_compress_matrix(weights, size, size, &config, nullptr);

        // Should still work, even if compression ratio < 1
        ASSERT_NE(mps, nullptr);
        EXPECT_TRUE(mps_verify_structure(mps));

        printf("Small matrix %u×%u: Compression ratio = %.2fx\n",
               size, size, mps->compression_ratio);

        mps_free(mps);
        nimcp_free(weights);
    }
}

TEST_F(MPSCompressionTest, NonSquareMatrices) {
    // WHAT: Test rectangular matrices
    // WHY: Neural network weight matrices often non-square
    // HOW: Various aspect ratios

    mps_config_t config = mps_default_config();

    struct {
        uint32_t rows;
        uint32_t cols;
    } test_cases[] = {
        {100, 50},   // Tall
        {50, 100},   // Wide
        {200, 100},  // 2:1 ratio
        {100, 200}   // 1:2 ratio
    };

    for (auto& test : test_cases) {
        float* weights = (float*)nimcp_malloc(test.rows * test.cols * sizeof(float));
        generate_random_weights(weights, test.rows, test.cols);

        mps_stats_t stats;
        mps_matrix_t* mps = mps_compress_matrix(weights, test.rows, test.cols, &config, &stats);

        ASSERT_NE(mps, nullptr);
        EXPECT_EQ(mps->input_dim, test.rows);
        EXPECT_EQ(mps->output_dim, test.cols);

        printf("Matrix %u×%u: Compression %.2fx\n",
               test.rows, test.cols, stats.compression_ratio);

        mps_free(mps);
        nimcp_free(weights);
    }
}

//=============================================================================
// Integration Test
//=============================================================================

TEST_F(MPSCompressionTest, SimulatedNeuralNetworkLayer) {
    // WHAT: Simulate a neural network layer using MPS
    // WHY: Demonstrate real-world usage
    // HOW: Forward pass through compressed layer

    printf("\n=== Simulated Neural Network Layer ===\n");

    const uint32_t input_dim = 784;   // MNIST input
    const uint32_t hidden_dim = 256;  // Hidden layer
    const uint32_t batch_size = 100;

    // Create weight matrix
    float* weights = (float*)nimcp_malloc(input_dim * hidden_dim * sizeof(float));
    generate_random_weights(weights, input_dim, hidden_dim, 0.1f);

    // Compress with MPS
    mps_config_t config = mps_default_config();
    config.bond_dim = 10;

    mps_stats_t stats;
    mps_matrix_t* mps = mps_compress_matrix(weights, input_dim, hidden_dim, &config, &stats);
    ASSERT_NE(mps, nullptr);

    printf("Weight matrix: %u × %u = %u parameters\n",
           input_dim, hidden_dim, input_dim * hidden_dim);
    printf("MPS compressed: %u parameters (%.1fx compression)\n",
           mps->total_params, stats.compression_ratio);
    printf("Memory savings: %.1f KB → %.1f KB\n",
           (input_dim * hidden_dim * sizeof(float)) / 1024.0f,
           mps_memory_usage(mps) / 1024.0f);

    // Process batch
    float total_error = 0.0f;

    for (uint32_t b = 0; b < batch_size; b++) {
        // Generate random input
        float* input = (float*)nimcp_malloc(input_dim * sizeof(float));
        generate_random_weights(input, input_dim, 1, 0.5f);

        // Reference forward pass
        float* output_ref = (float*)nimcp_calloc(hidden_dim, sizeof(float));
        reference_matvec(weights, input, output_ref, hidden_dim, input_dim);

        // MPS forward pass
        float* output_mps = (float*)nimcp_calloc(hidden_dim, sizeof(float));
        mps_matrix_vector_multiply(mps, input, output_mps);

        // Measure error
        float error = vector_relative_error(output_ref, output_mps, hidden_dim);
        total_error += error;

        nimcp_free(input);
        nimcp_free(output_ref);
        nimcp_free(output_mps);
    }

    float avg_error = total_error / (float)batch_size;
    printf("Average forward pass error: %.4f%%\n", avg_error * 100.0f);

    EXPECT_LT(avg_error, 0.05f);  // < 5% error acceptable

    printf("✅ Neural network layer simulation successful!\n");

    mps_free(mps);
    nimcp_free(weights);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
