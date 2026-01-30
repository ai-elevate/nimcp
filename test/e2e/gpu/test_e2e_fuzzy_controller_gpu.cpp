/* ============================================================================
 * E2E Test: Fuzzy Controller GPU Pipeline
 * ============================================================================
 * WHAT: End-to-end test of GPU-accelerated fuzzy controller workflow
 * WHY:  Validate complete fuzzy control system on GPU
 * HOW:  Design FIS -> Train ANFIS -> Deploy controller -> Evaluate performance
 * ============================================================================
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#ifdef NIMCP_ENABLE_CUDA
extern "C" {
#include "gpu/fuzzy/nimcp_fuzzy_gpu.h"
#include "gpu/fuzzy/nimcp_fuzzy_anfis_gpu.h"
#include "gpu/nimcp_gpu_context.h"
#include "cognitive/fuzzy/nimcp_fuzzy.h"
}
#endif

namespace {

constexpr float TOLERANCE = 1e-4f;
constexpr float CONTROL_TOLERANCE = 0.1f;

class FuzzyControllerE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef NIMCP_ENABLE_CUDA
        ctx_ = nimcp_gpu_context_create(nullptr);
#endif
    }

    void TearDown() override {
#ifdef NIMCP_ENABLE_CUDA
        if (ctx_) {
            nimcp_gpu_context_destroy(ctx_);
            ctx_ = nullptr;
        }
#endif
    }

#ifdef NIMCP_ENABLE_CUDA
    nimcp_gpu_context_t* ctx_ = nullptr;
#endif
};

/* ============================================================================
 * E2E Test: Fuzzy PD Controller for Inverted Pendulum
 * ============================================================================ */
TEST_F(FuzzyControllerE2ETest, InvertedPendulumControl) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required";

    // ==================== Phase 1: Design Fuzzy Inference System ====================
    fuzzy_inference_engine_t* fis = fuzzy_inference_create(FUZZY_INFERENCE_MAMDANI, 2, 1);
    ASSERT_NE(fis, nullptr);

    // Input 1: Angle error (radians)
    fuzzy_variable_t* angle_error = fuzzy_variable_create("angle_error", -0.5f, 0.5f);
    fuzzy_variable_add_mf(angle_error, "NB", FUZZY_MF_GAUSSIAN, (float[]){-0.5f, 0.15f}, 2);
    fuzzy_variable_add_mf(angle_error, "NS", FUZZY_MF_GAUSSIAN, (float[]){-0.25f, 0.10f}, 2);
    fuzzy_variable_add_mf(angle_error, "ZE", FUZZY_MF_GAUSSIAN, (float[]){0.0f, 0.10f}, 2);
    fuzzy_variable_add_mf(angle_error, "PS", FUZZY_MF_GAUSSIAN, (float[]){0.25f, 0.10f}, 2);
    fuzzy_variable_add_mf(angle_error, "PB", FUZZY_MF_GAUSSIAN, (float[]){0.5f, 0.15f}, 2);
    fuzzy_inference_add_input(fis, angle_error);

    // Input 2: Angular velocity (rad/s)
    fuzzy_variable_t* angular_vel = fuzzy_variable_create("angular_velocity", -2.0f, 2.0f);
    fuzzy_variable_add_mf(angular_vel, "NB", FUZZY_MF_GAUSSIAN, (float[]){-2.0f, 0.6f}, 2);
    fuzzy_variable_add_mf(angular_vel, "NS", FUZZY_MF_GAUSSIAN, (float[]){-1.0f, 0.4f}, 2);
    fuzzy_variable_add_mf(angular_vel, "ZE", FUZZY_MF_GAUSSIAN, (float[]){0.0f, 0.4f}, 2);
    fuzzy_variable_add_mf(angular_vel, "PS", FUZZY_MF_GAUSSIAN, (float[]){1.0f, 0.4f}, 2);
    fuzzy_variable_add_mf(angular_vel, "PB", FUZZY_MF_GAUSSIAN, (float[]){2.0f, 0.6f}, 2);
    fuzzy_inference_add_input(fis, angular_vel);

    // Output: Force (N)
    fuzzy_variable_t* force = fuzzy_variable_create("force", -20.0f, 20.0f);
    fuzzy_variable_add_mf(force, "NB", FUZZY_MF_TRIANGULAR, (float[]){-20.0f, -20.0f, -10.0f}, 3);
    fuzzy_variable_add_mf(force, "NS", FUZZY_MF_TRIANGULAR, (float[]){-15.0f, -5.0f, 0.0f}, 3);
    fuzzy_variable_add_mf(force, "ZE", FUZZY_MF_TRIANGULAR, (float[]){-5.0f, 0.0f, 5.0f}, 3);
    fuzzy_variable_add_mf(force, "PS", FUZZY_MF_TRIANGULAR, (float[]){0.0f, 5.0f, 15.0f}, 3);
    fuzzy_variable_add_mf(force, "PB", FUZZY_MF_TRIANGULAR, (float[]){10.0f, 20.0f, 20.0f}, 3);
    fuzzy_inference_add_output(fis, force);

    // Rules (classic inverted pendulum rules)
    fuzzy_inference_add_rule(fis, "IF angle_error IS NB AND angular_velocity IS NB THEN force IS NB");
    fuzzy_inference_add_rule(fis, "IF angle_error IS NB AND angular_velocity IS ZE THEN force IS NB");
    fuzzy_inference_add_rule(fis, "IF angle_error IS NS AND angular_velocity IS NS THEN force IS NS");
    fuzzy_inference_add_rule(fis, "IF angle_error IS ZE AND angular_velocity IS ZE THEN force IS ZE");
    fuzzy_inference_add_rule(fis, "IF angle_error IS PS AND angular_velocity IS PS THEN force IS PS");
    fuzzy_inference_add_rule(fis, "IF angle_error IS PB AND angular_velocity IS ZE THEN force IS PB");
    fuzzy_inference_add_rule(fis, "IF angle_error IS PB AND angular_velocity IS PB THEN force IS PB");
    fuzzy_inference_add_rule(fis, "IF angle_error IS NB AND angular_velocity IS PB THEN force IS ZE");
    fuzzy_inference_add_rule(fis, "IF angle_error IS PB AND angular_velocity IS NB THEN force IS ZE");

    // ==================== Phase 2: Create GPU State ====================
    nimcp_gpu_fuzzy_inference_state_t* gpu_state = nimcp_gpu_fuzzy_state_create(ctx_, fis);
    ASSERT_NE(gpu_state, nullptr);

    // ==================== Phase 3: Simulate Control Loop ====================
    const int sim_steps = 1000;
    const float dt = 0.01f;  // 10ms time step

    // Pendulum parameters
    const float m = 1.0f;   // mass (kg)
    const float l = 1.0f;   // length (m)
    const float g = 9.81f;  // gravity (m/s^2)
    const float b = 0.1f;   // damping

    // Initial conditions: 15 degrees off vertical
    float theta = 0.26f;  // ~15 degrees
    float omega = 0.0f;

    // Track control performance
    float max_error = 0.0f;
    float total_energy = 0.0f;
    int steps_to_stable = sim_steps;
    bool stabilized = false;

    for (int step = 0; step < sim_steps; step++) {
        // ---- Get control input from fuzzy controller ----
        float inputs[2] = {theta, omega};
        float control_force = 0.0f;

        nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_create(
            ctx_, (size_t[]){1, 2}, 2, NIMCP_GPU_DTYPE_FLOAT32);
        nimcp_gpu_tensor_t* output_tensor = nimcp_gpu_tensor_create(
            ctx_, (size_t[]){1, 1}, 2, NIMCP_GPU_DTYPE_FLOAT32);

        if (input_tensor && output_tensor) {
            nimcp_gpu_tensor_copy_from_host(input_tensor, inputs, sizeof(inputs));

            nimcp_gpu_inference_params_t params = {
                .t_norm = FUZZY_TNORM_MIN,
                .t_conorm = FUZZY_TCONORM_MAX,
                .defuzz_method = FUZZY_DEFUZZ_CENTROID,
                .resolution = 100
            };

            if (nimcp_gpu_fuzzy_inference_batch(ctx_, gpu_state, input_tensor, output_tensor, &params)) {
                nimcp_gpu_tensor_copy_to_host(output_tensor, &control_force, sizeof(control_force));
            }

            nimcp_gpu_tensor_destroy(input_tensor);
            nimcp_gpu_tensor_destroy(output_tensor);
        }

        // ---- Update pendulum dynamics ----
        // theta'' = (g*sin(theta) - b*omega + F*cos(theta)/m) / l
        float accel = (g * std::sin(theta) - b * omega + control_force * std::cos(theta) / m) / l;
        omega += accel * dt;
        theta += omega * dt;

        // Track metrics
        max_error = std::max(max_error, std::abs(theta));
        total_energy += std::abs(control_force) * dt;

        // Check if stabilized (small angle and velocity)
        if (!stabilized && std::abs(theta) < 0.05f && std::abs(omega) < 0.1f) {
            stabilized = true;
            steps_to_stable = step;
        }
    }

    // ==================== Phase 4: Verify Control Performance ====================
    EXPECT_TRUE(stabilized) << "Controller should stabilize pendulum";
    EXPECT_LT(steps_to_stable, 500) << "Should stabilize within 5 seconds";
    EXPECT_LT(std::abs(theta), 0.1f) << "Final angle should be near vertical";
    EXPECT_LT(std::abs(omega), 0.2f) << "Final velocity should be small";

    std::cout << "Fuzzy PD Controller Results:" << std::endl;
    std::cout << "  Steps to stabilize: " << steps_to_stable << std::endl;
    std::cout << "  Max angle error: " << max_error << " rad" << std::endl;
    std::cout << "  Total control effort: " << total_energy << " N*s" << std::endl;

    nimcp_gpu_fuzzy_state_destroy(gpu_state);
    fuzzy_inference_destroy(fis);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * E2E Test: ANFIS Function Approximation
 * ============================================================================ */
TEST_F(FuzzyControllerE2ETest, ANFISFunctionApproximation) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required";

    // ==================== Phase 1: Generate Training Data ====================
    // Target function: Mackey-Glass (chaotic time series)
    const int n_samples = 2000;
    const int train_size = 1500;
    const int test_size = n_samples - train_size;
    const int embedding_dim = 4;

    std::vector<float> series(n_samples + 100);
    series[0] = 0.9f;

    // Generate Mackey-Glass series
    for (int i = 1; i < static_cast<int>(series.size()); i++) {
        int delay_idx = std::max(0, i - 17);
        float x_tau = series[delay_idx];
        series[i] = series[i-1] + 0.2f * x_tau / (1.0f + std::pow(x_tau, 10.0f)) - 0.1f * series[i-1];
    }

    // Create embedded input/target pairs
    std::vector<float> train_inputs(train_size * embedding_dim);
    std::vector<float> train_targets(train_size);
    std::vector<float> test_inputs(test_size * embedding_dim);
    std::vector<float> test_targets(test_size);

    for (int i = 0; i < n_samples; i++) {
        float* inputs_ptr = (i < train_size) ?
            &train_inputs[i * embedding_dim] : &test_inputs[(i - train_size) * embedding_dim];
        float* target_ptr = (i < train_size) ?
            &train_targets[i] : &test_targets[i - train_size];

        for (int j = 0; j < embedding_dim; j++) {
            inputs_ptr[j] = series[100 + i - (embedding_dim - 1 - j) * 6];
        }
        *target_ptr = series[100 + i + 6];  // Predict 6 steps ahead
    }

    // ==================== Phase 2: Create and Train ANFIS ====================
    nimcp_gpu_anfis_params_t anfis_params = {
        .num_inputs = static_cast<uint32_t>(embedding_dim),
        .num_mfs_per_input = 3,
        .mf_type = FUZZY_MF_GAUSSIAN,
        .learning_rate = 0.01f,
        .momentum = 0.9f
    };

    nimcp_gpu_anfis_state_t* anfis = nimcp_gpu_anfis_create(ctx_, &anfis_params);
    ASSERT_NE(anfis, nullptr);

    nimcp_gpu_tensor_t* train_input_tensor = nimcp_gpu_tensor_create(
        ctx_, (size_t[]){static_cast<size_t>(train_size), static_cast<size_t>(embedding_dim)},
        2, NIMCP_GPU_DTYPE_FLOAT32);
    nimcp_gpu_tensor_t* train_target_tensor = nimcp_gpu_tensor_create(
        ctx_, (size_t[]){static_cast<size_t>(train_size), 1},
        2, NIMCP_GPU_DTYPE_FLOAT32);

    ASSERT_NE(train_input_tensor, nullptr);
    ASSERT_NE(train_target_tensor, nullptr);

    nimcp_gpu_tensor_copy_from_host(train_input_tensor, train_inputs.data(),
        train_size * embedding_dim * sizeof(float));
    nimcp_gpu_tensor_copy_from_host(train_target_tensor, train_targets.data(),
        train_size * sizeof(float));

    float initial_error = 0.0f, final_error = 0.0f;

    auto train_start = std::chrono::high_resolution_clock::now();

    nimcp_gpu_anfis_train_params_t train_params = {
        .num_epochs = 100,
        .batch_size = 100,
        .early_stop_threshold = 0.001f,
        .shuffle = true,
        .hybrid_learning = true,
        .lse_lambda = 0.001f
    };

    bool train_success = nimcp_gpu_anfis_train(ctx_, anfis,
        train_input_tensor, train_target_tensor,
        &train_params, &initial_error, &final_error);

    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

    EXPECT_TRUE(train_success) << "ANFIS training should succeed";
    EXPECT_LT(final_error, initial_error) << "Error should decrease";

    // ==================== Phase 3: Evaluate on Test Set ====================
    nimcp_gpu_tensor_t* test_input_tensor = nimcp_gpu_tensor_create(
        ctx_, (size_t[]){static_cast<size_t>(test_size), static_cast<size_t>(embedding_dim)},
        2, NIMCP_GPU_DTYPE_FLOAT32);
    nimcp_gpu_tensor_t* test_output_tensor = nimcp_gpu_tensor_create(
        ctx_, (size_t[]){static_cast<size_t>(test_size), 1},
        2, NIMCP_GPU_DTYPE_FLOAT32);

    ASSERT_NE(test_input_tensor, nullptr);
    ASSERT_NE(test_output_tensor, nullptr);

    nimcp_gpu_tensor_copy_from_host(test_input_tensor, test_inputs.data(),
        test_size * embedding_dim * sizeof(float));

    bool pred_success = nimcp_gpu_anfis_forward(ctx_, anfis, test_input_tensor, test_output_tensor);
    EXPECT_TRUE(pred_success);

    std::vector<float> predictions(test_size);
    nimcp_gpu_tensor_copy_to_host(test_output_tensor, predictions.data(),
        test_size * sizeof(float));

    // Calculate RMSE
    float mse = 0.0f;
    for (int i = 0; i < test_size; i++) {
        float diff = predictions[i] - test_targets[i];
        mse += diff * diff;
    }
    float rmse = std::sqrt(mse / test_size);

    // Calculate NMSE (Normalized MSE)
    float target_var = 0.0f;
    float target_mean = 0.0f;
    for (const auto& t : test_targets) target_mean += t;
    target_mean /= test_size;
    for (const auto& t : test_targets) {
        float diff = t - target_mean;
        target_var += diff * diff;
    }
    target_var /= test_size;
    float nmse = (mse / test_size) / target_var;

    std::cout << "ANFIS Mackey-Glass Prediction Results:" << std::endl;
    std::cout << "  Training time: " << train_time.count() << " ms" << std::endl;
    std::cout << "  Initial MSE: " << initial_error << std::endl;
    std::cout << "  Final MSE: " << final_error << std::endl;
    std::cout << "  Test RMSE: " << rmse << std::endl;
    std::cout << "  Test NMSE: " << nmse << std::endl;

    // ANFIS should achieve reasonable prediction (NMSE < 0.1)
    EXPECT_LT(nmse, 0.3f) << "NMSE should be reasonable for chaotic series";

    nimcp_gpu_tensor_destroy(train_input_tensor);
    nimcp_gpu_tensor_destroy(train_target_tensor);
    nimcp_gpu_tensor_destroy(test_input_tensor);
    nimcp_gpu_tensor_destroy(test_output_tensor);
    nimcp_gpu_anfis_destroy(anfis);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * E2E Test: Real-time Batch Inference Performance
 * ============================================================================ */
TEST_F(FuzzyControllerE2ETest, RealTimeBatchPerformance) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required";

    // Create a reasonably complex FIS
    fuzzy_inference_engine_t* fis = fuzzy_inference_create(FUZZY_INFERENCE_MAMDANI, 4, 2);
    ASSERT_NE(fis, nullptr);

    for (int i = 0; i < 4; i++) {
        char name[32];
        snprintf(name, sizeof(name), "input%d", i);
        fuzzy_variable_t* var = fuzzy_variable_create(name, 0.0f, 10.0f);
        for (int j = 0; j < 5; j++) {
            char mf_name[32];
            snprintf(mf_name, sizeof(mf_name), "mf%d", j);
            float center = j * 2.5f;
            fuzzy_variable_add_mf(var, mf_name, FUZZY_MF_GAUSSIAN, (float[]){center, 1.0f}, 2);
        }
        fuzzy_inference_add_input(fis, var);
    }

    for (int i = 0; i < 2; i++) {
        char name[32];
        snprintf(name, sizeof(name), "output%d", i);
        fuzzy_variable_t* var = fuzzy_variable_create(name, 0.0f, 10.0f);
        for (int j = 0; j < 5; j++) {
            char mf_name[32];
            snprintf(mf_name, sizeof(mf_name), "mf%d", j);
            fuzzy_variable_add_mf(var, mf_name, FUZZY_MF_TRIANGULAR,
                (float[]){(j-0.5f)*2.5f, j*2.5f, (j+0.5f)*2.5f}, 3);
        }
        fuzzy_inference_add_output(fis, var);
    }

    // Add rules (simplified)
    fuzzy_inference_add_rule(fis, "IF input0 IS mf0 AND input1 IS mf0 THEN output0 IS mf0");
    fuzzy_inference_add_rule(fis, "IF input0 IS mf4 AND input1 IS mf4 THEN output0 IS mf4");
    fuzzy_inference_add_rule(fis, "IF input2 IS mf2 THEN output1 IS mf2");

    nimcp_gpu_fuzzy_inference_state_t* gpu_state = nimcp_gpu_fuzzy_state_create(ctx_, fis);
    ASSERT_NE(gpu_state, nullptr);

    // Benchmark different batch sizes
    std::vector<size_t> batch_sizes = {1, 10, 100, 1000, 10000};

    for (size_t batch_size : batch_sizes) {
        std::vector<float> inputs(batch_size * 4);
        std::vector<float> outputs(batch_size * 2);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        for (auto& x : inputs) x = dist(gen);

        nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_create(
            ctx_, (size_t[]){batch_size, 4}, 2, NIMCP_GPU_DTYPE_FLOAT32);
        nimcp_gpu_tensor_t* output_tensor = nimcp_gpu_tensor_create(
            ctx_, (size_t[]){batch_size, 2}, 2, NIMCP_GPU_DTYPE_FLOAT32);

        ASSERT_NE(input_tensor, nullptr);
        ASSERT_NE(output_tensor, nullptr);

        nimcp_gpu_tensor_copy_from_host(input_tensor, inputs.data(),
            batch_size * 4 * sizeof(float));

        nimcp_gpu_inference_params_t params = {
            .t_norm = FUZZY_TNORM_MIN,
            .t_conorm = FUZZY_TCONORM_MAX,
            .defuzz_method = FUZZY_DEFUZZ_CENTROID,
            .resolution = 50
        };

        // Warm-up
        nimcp_gpu_fuzzy_inference_batch(ctx_, gpu_state, input_tensor, output_tensor, &params);

        // Timed run
        const int num_iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; i++) {
            nimcp_gpu_fuzzy_inference_batch(ctx_, gpu_state, input_tensor, output_tensor, &params);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float avg_time_us = static_cast<float>(elapsed.count()) / num_iterations;
        float throughput = batch_size / (avg_time_us / 1e6f);

        std::cout << "Batch " << batch_size << ": "
                  << avg_time_us << " us avg, "
                  << throughput << " samples/sec" << std::endl;

        // For real-time control at 1kHz, need < 1ms per inference
        if (batch_size <= 100) {
            EXPECT_LT(avg_time_us, 1000.0f)
                << "Small batch should be < 1ms for real-time control";
        }

        nimcp_gpu_tensor_destroy(input_tensor);
        nimcp_gpu_tensor_destroy(output_tensor);
    }

    nimcp_gpu_fuzzy_state_destroy(gpu_state);
    fuzzy_inference_destroy(fis);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

}  // namespace
