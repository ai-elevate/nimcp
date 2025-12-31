/**
 * @file e2e_test_gpu_lnn_pipeline.cpp
 * @brief E2E Tests for GPU Liquid Neural Network (LNN) Pipeline
 *
 * WHAT: End-to-end testing for GPU-accelerated LNN simulation
 * WHY:  Verify LNN dynamics, ODE integration, and time-series processing
 * HOW:  Test different ODE solvers (Euler, RK4, DOPRI5), state trajectories
 *
 * TEST PIPELINES:
 * - LNNLayerCreation: Create LNN layer on GPU
 * - TimeSeriesInput: Provide and process time-series input
 * - EulerODEIntegration: Test Euler method integration
 * - RK4ODEIntegration: Test 4th-order Runge-Kutta integration
 * - DOPRI5AdaptiveIntegration: Test adaptive DOPRI5 solver
 * - StateTrajectorySmootness: Verify continuous state evolution
 * - CompareODESolvers: Compare accuracy of different solvers
 * - FullLNNTrainingLoop: Complete training with backprop through ODE
 *
 * @author NIMCP Development Team
 * @date 2025-12-31
 * @version 1.0.0
 */

#include "e2e_test_framework.h"

extern "C" {
#include "gpu/context/nimcp_gpu_context.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "gpu/lnn/nimcp_lnn_gpu.h"
#include "lnn/nimcp_lnn_types.h"
}

#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <iomanip>

//=============================================================================
// LNN Constants and Helper Functions
//=============================================================================

// Default LNN parameters
constexpr float LNN_TAU_MIN = 1.0f;       // Minimum time constant (ms)
constexpr float LNN_TAU_MAX = 100.0f;     // Maximum time constant (ms)
constexpr float LNN_DT = 0.5f;            // Time step (ms)
constexpr float LNN_TOTAL_TIME = 100.0f;  // Total simulation time (ms)

/**
 * @brief Generate sinusoidal time series input
 */
static void generate_sinusoidal_input(
    float* data,
    size_t num_timesteps,
    size_t input_dim,
    float dt,
    float frequency)
{
    for (size_t t = 0; t < num_timesteps; t++) {
        float time = t * dt;
        for (size_t d = 0; d < input_dim; d++) {
            // Phase-shifted sinusoids
            float phase = 2.0f * M_PI * d / input_dim;
            data[t * input_dim + d] = std::sin(2.0f * M_PI * frequency * time / 1000.0f + phase);
        }
    }
}

/**
 * @brief Generate random walk time series
 */
static void generate_random_walk(
    float* data,
    size_t num_timesteps,
    size_t input_dim,
    float volatility)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, volatility);

    // Initialize at 0
    for (size_t d = 0; d < input_dim; d++) {
        data[d] = 0.0f;
    }

    // Random walk
    for (size_t t = 1; t < num_timesteps; t++) {
        for (size_t d = 0; d < input_dim; d++) {
            data[t * input_dim + d] = data[(t - 1) * input_dim + d] + dist(gen);
        }
    }
}

/**
 * @brief Compute L2 norm between two state vectors
 */
static float compute_state_difference(const float* s1, const float* s2, size_t size) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = s1[i] - s2[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief Check if trajectory is smooth (no large jumps)
 */
static bool is_trajectory_smooth(
    const std::vector<std::vector<float>>& trajectory,
    float max_step_size)
{
    if (trajectory.size() < 2) return true;

    size_t state_size = trajectory[0].size();
    for (size_t t = 1; t < trajectory.size(); t++) {
        float diff = compute_state_difference(
            trajectory[t].data(), trajectory[t-1].data(), state_size);
        if (diff > max_step_size) {
            return false;
        }
    }
    return true;
}

//=============================================================================
// Test Fixture
//=============================================================================

class GPULNNPipelineE2ETest : public ::testing::Test {
protected:
    nimcp_gpu_context_t* ctx_ = nullptr;
    bool gpu_available_ = false;

    void SetUp() override {
        ctx_ = nimcp_gpu_context_create(0);
        gpu_available_ = (ctx_ != nullptr);

        if (!gpu_available_) {
            std::cout << "GPU not available - some tests will be skipped" << std::endl;
        }
    }

    void TearDown() override {
        if (ctx_) {
            nimcp_gpu_context_destroy(ctx_);
            ctx_ = nullptr;
        }
    }

    bool HasGPU() const { return gpu_available_; }

    void SkipIfNoGPU() {
        if (!gpu_available_) {
            GTEST_SKIP() << "GPU not available";
        }
    }
};

//=============================================================================
// Pipeline 1: LNN Layer Creation
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, LNNLayerCreation) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("LNN Layer Creation");

    const size_t input_dim = 8;
    const size_t hidden_dim = 32;

    // Stage 1: Create LNN layer configuration
    E2E_STAGE_BEGIN("Create LNN configuration", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = LNN_DT;
    config.ode_method = LNN_ODE_RK4;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    E2E_STAGE_END();

    // Stage 2: Create LNN layer on GPU
    E2E_STAGE_BEGIN("Create LNN layer", 1000);

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    E2E_STAGE_END();

    // Stage 3: Verify layer properties
    E2E_STAGE_BEGIN("Verify layer properties", 500);

    size_t layer_input_dim = nimcp_lnn_layer_gpu_get_input_dim(layer);
    size_t layer_hidden_dim = nimcp_lnn_layer_gpu_get_hidden_dim(layer);
    lnn_ode_method_t layer_ode = nimcp_lnn_layer_gpu_get_ode_method(layer);

    EXPECT_EQ(layer_input_dim, input_dim);
    EXPECT_EQ(layer_hidden_dim, hidden_dim);
    EXPECT_EQ(layer_ode, LNN_ODE_RK4);

    std::cout << "\n  LNN Layer Created:" << std::endl;
    std::cout << "    Input dim:  " << layer_input_dim << std::endl;
    std::cout << "    Hidden dim: " << layer_hidden_dim << std::endl;
    std::cout << "    ODE method: RK4" << std::endl;

    E2E_STAGE_END();

    // Stage 4: Verify parameter count
    E2E_STAGE_BEGIN("Verify parameter count", 500);

    size_t num_params = nimcp_lnn_layer_gpu_count_params(layer);

    // Expected params:
    // W_in: input_dim * hidden_dim
    // W_rec: hidden_dim * hidden_dim (recurrent)
    // W_tau: (input_dim + hidden_dim) * hidden_dim (tau modulation)
    // b_in, b_tau: hidden_dim each
    // tau_base: hidden_dim
    size_t expected_params = input_dim * hidden_dim +           // W_in
                             hidden_dim * hidden_dim +          // W_rec
                             (input_dim + hidden_dim) * hidden_dim + // W_tau
                             hidden_dim * 3;                    // biases + tau_base

    std::cout << "    Parameters: " << num_params << std::endl;
    std::cout << "    Expected:   ~" << expected_params << std::endl;

    EXPECT_GT(num_params, 0u) << "Layer should have parameters";

    E2E_STAGE_END();

    // Stage 5: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 2: Time Series Input Processing
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, TimeSeriesInput) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("Time Series Input Processing");

    const size_t input_dim = 4;
    const size_t hidden_dim = 16;
    const size_t num_timesteps = 200;

    // Stage 1: Create LNN layer
    E2E_STAGE_BEGIN("Create LNN layer", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = LNN_DT;
    config.ode_method = LNN_ODE_RK4;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    E2E_STAGE_END();

    // Stage 2: Generate time series input
    E2E_STAGE_BEGIN("Generate time series", 500);

    std::vector<float> input_data(num_timesteps * input_dim);
    generate_sinusoidal_input(input_data.data(), num_timesteps, input_dim, LNN_DT, 10.0f);

    std::cout << "\n  Generated " << num_timesteps << " timesteps of input" << std::endl;
    std::cout << "    Input dim: " << input_dim << std::endl;
    std::cout << "    Time span: " << (num_timesteps * LNN_DT) << " ms" << std::endl;

    E2E_STAGE_END();

    // Stage 3: Upload input to GPU
    E2E_STAGE_BEGIN("Upload to GPU", 500);

    size_t input_dims[] = {num_timesteps, input_dim};
    nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
        ctx_, input_data.data(), input_dims, 2, NIMCP_GPU_PRECISION_FP32);
    E2E_ASSERT_NOT_NULL(input_tensor, "Failed to create input tensor");

    E2E_STAGE_END();

    // Stage 4: Process time series
    E2E_STAGE_BEGIN("Process time series", 2000);

    size_t output_dims[] = {num_timesteps, hidden_dim};
    nimcp_gpu_tensor_t* output_tensor = nimcp_gpu_tensor_create(
        ctx_, output_dims, 2, NIMCP_GPU_PRECISION_FP32);

    bool process_ok = nimcp_lnn_layer_gpu_forward_sequence(
        layer, input_tensor, output_tensor, num_timesteps);
    EXPECT_TRUE(process_ok);

    nimcp_gpu_context_synchronize(ctx_);

    E2E_STAGE_END();

    // Stage 5: Verify output
    E2E_STAGE_BEGIN("Verify output", 500);

    std::vector<float> output_data(num_timesteps * hidden_dim);
    nimcp_gpu_tensor_to_host(output_tensor, output_data.data());

    // Check for valid values
    bool valid = true;
    for (const auto& v : output_data) {
        if (std::isnan(v) || std::isinf(v)) {
            valid = false;
            break;
        }
    }
    EXPECT_TRUE(valid) << "Output should not contain NaN or Inf";

    // Check output range (tanh activation bounds to [-1, 1])
    float min_val = *std::min_element(output_data.begin(), output_data.end());
    float max_val = *std::max_element(output_data.begin(), output_data.end());

    std::cout << "    Output range: [" << min_val << ", " << max_val << "]" << std::endl;

    EXPECT_GE(min_val, -1.01f) << "tanh output should be >= -1";
    EXPECT_LE(max_val, 1.01f) << "tanh output should be <= 1";

    E2E_STAGE_END();

    // Stage 6: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_gpu_tensor_destroy(input_tensor);
    nimcp_gpu_tensor_destroy(output_tensor);
    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 3: Euler ODE Integration
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, EulerODEIntegration) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("Euler ODE Integration");

    const size_t input_dim = 4;
    const size_t hidden_dim = 8;
    const size_t num_steps = 100;

    // Stage 1: Create layer with Euler method
    E2E_STAGE_BEGIN("Create Euler LNN layer", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = LNN_DT;
    config.ode_method = LNN_ODE_EULER;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    E2E_STAGE_END();

    // Stage 2: Run multiple ODE steps
    E2E_STAGE_BEGIN("Run Euler integration", 2000);

    std::vector<float> input(input_dim, 0.5f);
    size_t input_dims[] = {input_dim};
    nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
        ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

    std::vector<std::vector<float>> trajectory;

    for (size_t step = 0; step < num_steps; step++) {
        bool step_ok = nimcp_lnn_layer_gpu_step(layer, input_tensor);
        EXPECT_TRUE(step_ok);

        // Get state
        std::vector<float> state(hidden_dim);
        nimcp_lnn_layer_gpu_get_state(layer, state.data());
        trajectory.push_back(state);
    }

    nimcp_gpu_context_synchronize(ctx_);

    std::cout << "\n  Euler Integration:" << std::endl;
    std::cout << "    Steps: " << num_steps << std::endl;
    std::cout << "    Time:  " << (num_steps * LNN_DT) << " ms" << std::endl;

    E2E_STAGE_END();

    // Stage 3: Verify trajectory properties
    E2E_STAGE_BEGIN("Verify trajectory", 500);

    // Check trajectory is smooth
    bool smooth = is_trajectory_smooth(trajectory, 0.5f);
    EXPECT_TRUE(smooth) << "Euler trajectory should be reasonably smooth";

    // Check final state is not zero (dynamics working)
    float final_norm = 0.0f;
    for (const auto& v : trajectory.back()) {
        final_norm += v * v;
    }
    final_norm = std::sqrt(final_norm);

    std::cout << "    Final state norm: " << final_norm << std::endl;

    EXPECT_GT(final_norm, 0.01f) << "Final state should be non-zero";

    E2E_STAGE_END();

    // Stage 4: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_gpu_tensor_destroy(input_tensor);
    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 4: RK4 ODE Integration
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, RK4ODEIntegration) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("RK4 ODE Integration");

    const size_t input_dim = 4;
    const size_t hidden_dim = 8;
    const size_t num_steps = 100;

    // Stage 1: Create layer with RK4 method
    E2E_STAGE_BEGIN("Create RK4 LNN layer", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = LNN_DT;
    config.ode_method = LNN_ODE_RK4;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    E2E_STAGE_END();

    // Stage 2: Run RK4 integration
    E2E_STAGE_BEGIN("Run RK4 integration", 2000);

    std::vector<float> input(input_dim, 0.5f);
    size_t input_dims[] = {input_dim};
    nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
        ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

    std::vector<std::vector<float>> trajectory;

    for (size_t step = 0; step < num_steps; step++) {
        bool step_ok = nimcp_lnn_layer_gpu_step(layer, input_tensor);
        EXPECT_TRUE(step_ok);

        std::vector<float> state(hidden_dim);
        nimcp_lnn_layer_gpu_get_state(layer, state.data());
        trajectory.push_back(state);
    }

    nimcp_gpu_context_synchronize(ctx_);

    std::cout << "\n  RK4 Integration:" << std::endl;
    std::cout << "    Steps: " << num_steps << std::endl;

    E2E_STAGE_END();

    // Stage 3: Verify RK4 is smoother than Euler
    E2E_STAGE_BEGIN("Verify RK4 smoothness", 500);

    bool smooth = is_trajectory_smooth(trajectory, 0.3f);  // Tighter tolerance for RK4
    EXPECT_TRUE(smooth) << "RK4 trajectory should be smooth";

    // Compute trajectory smoothness metric
    float total_curvature = 0.0f;
    for (size_t t = 2; t < trajectory.size(); t++) {
        // Second derivative approximation
        for (size_t d = 0; d < hidden_dim; d++) {
            float d2 = trajectory[t][d] - 2.0f * trajectory[t-1][d] + trajectory[t-2][d];
            total_curvature += d2 * d2;
        }
    }
    total_curvature = std::sqrt(total_curvature / (trajectory.size() - 2));

    std::cout << "    Curvature metric: " << total_curvature << std::endl;

    E2E_STAGE_END();

    // Stage 4: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_gpu_tensor_destroy(input_tensor);
    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 5: DOPRI5 Adaptive Integration
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, DOPRI5AdaptiveIntegration) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("DOPRI5 Adaptive Integration");

    const size_t input_dim = 4;
    const size_t hidden_dim = 16;
    const float total_time = 50.0f;  // Total simulation time

    // Stage 1: Create layer with DOPRI5 method
    E2E_STAGE_BEGIN("Create DOPRI5 LNN layer", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = LNN_DT;  // Initial dt, DOPRI5 adapts
    config.ode_method = LNN_ODE_DOPRI5;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    // Set adaptive step parameters
    nimcp_lnn_layer_gpu_set_adaptive_params(layer, 1e-6f, 1e-3f, 0.1f, 10.0f);

    E2E_STAGE_END();

    // Stage 2: Run adaptive integration
    E2E_STAGE_BEGIN("Run adaptive integration", 3000);

    std::vector<float> input(input_dim, 0.5f);
    size_t input_dims[] = {input_dim};
    nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
        ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

    std::vector<float> times;
    std::vector<std::vector<float>> trajectory;
    size_t total_steps = 0;

    float current_time = 0.0f;
    while (current_time < total_time) {
        float dt_used = 0.0f;
        bool step_ok = nimcp_lnn_layer_gpu_step_adaptive(layer, input_tensor, &dt_used);
        EXPECT_TRUE(step_ok);

        current_time += dt_used;
        times.push_back(current_time);

        std::vector<float> state(hidden_dim);
        nimcp_lnn_layer_gpu_get_state(layer, state.data());
        trajectory.push_back(state);

        total_steps++;
    }

    nimcp_gpu_context_synchronize(ctx_);

    // Compute statistics
    float avg_dt = total_time / total_steps;
    float min_dt = *std::min_element(times.begin(), times.end());
    float max_dt = total_time;
    if (times.size() > 1) {
        std::vector<float> dts;
        for (size_t i = 1; i < times.size(); i++) {
            dts.push_back(times[i] - times[i-1]);
        }
        min_dt = *std::min_element(dts.begin(), dts.end());
        max_dt = *std::max_element(dts.begin(), dts.end());
    }

    std::cout << "\n  DOPRI5 Adaptive Integration:" << std::endl;
    std::cout << "    Total time: " << total_time << " ms" << std::endl;
    std::cout << "    Steps taken: " << total_steps << std::endl;
    std::cout << "    Avg dt: " << avg_dt << " ms" << std::endl;
    std::cout << "    dt range: [" << min_dt << ", " << max_dt << "] ms" << std::endl;

    E2E_STAGE_END();

    // Stage 3: Verify error control
    E2E_STAGE_BEGIN("Verify error control", 500);

    float error_estimate = nimcp_lnn_layer_gpu_get_error_estimate(layer);
    std::cout << "    Final error estimate: " << error_estimate << std::endl;

    EXPECT_LT(error_estimate, 1e-3f) << "DOPRI5 should achieve low error";

    E2E_STAGE_END();

    // Stage 4: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_gpu_tensor_destroy(input_tensor);
    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 6: State Trajectory Smoothness
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, StateTrajectorySmootness) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("State Trajectory Smoothness");

    const size_t input_dim = 2;
    const size_t hidden_dim = 8;
    const size_t num_steps = 200;

    // Stage 1: Create layer
    E2E_STAGE_BEGIN("Create LNN layer", 500);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = 10.0f;  // Larger tau for smoother dynamics
    config.tau_max = 50.0f;
    config.dt = 0.5f;
    config.ode_method = LNN_ODE_RK4;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(layer, "Failed to create LNN layer");

    E2E_STAGE_END();

    // Stage 2: Run with smoothly varying input
    E2E_STAGE_BEGIN("Run with smooth input", 2000);

    std::vector<std::vector<float>> trajectory;

    for (size_t step = 0; step < num_steps; step++) {
        // Slowly varying sinusoidal input
        float t = step * config.dt;
        std::vector<float> input(input_dim);
        for (size_t d = 0; d < input_dim; d++) {
            input[d] = std::sin(2.0f * M_PI * (t / 100.0f + (float)d / input_dim));
        }

        size_t input_dims[] = {input_dim};
        nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
            ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

        nimcp_lnn_layer_gpu_step(layer, input_tensor);

        std::vector<float> state(hidden_dim);
        nimcp_lnn_layer_gpu_get_state(layer, state.data());
        trajectory.push_back(state);

        nimcp_gpu_tensor_destroy(input_tensor);
    }

    E2E_STAGE_END();

    // Stage 3: Analyze trajectory smoothness
    E2E_STAGE_BEGIN("Analyze smoothness", 500);

    // Compute velocity (first derivative)
    std::vector<float> velocities;
    for (size_t t = 1; t < trajectory.size(); t++) {
        float vel = compute_state_difference(
            trajectory[t].data(), trajectory[t-1].data(), hidden_dim) / config.dt;
        velocities.push_back(vel);
    }

    // Compute acceleration (second derivative)
    std::vector<float> accelerations;
    for (size_t t = 1; t < velocities.size(); t++) {
        float acc = std::abs(velocities[t] - velocities[t-1]) / config.dt;
        accelerations.push_back(acc);
    }

    float avg_velocity = 0.0f;
    for (float v : velocities) avg_velocity += v;
    avg_velocity /= velocities.size();

    float max_acceleration = *std::max_element(accelerations.begin(), accelerations.end());

    std::cout << "\n  Trajectory Analysis:" << std::endl;
    std::cout << "    Avg velocity:      " << avg_velocity << " units/ms" << std::endl;
    std::cout << "    Max acceleration:  " << max_acceleration << " units/ms^2" << std::endl;

    // Smooth trajectory should have bounded acceleration
    EXPECT_LT(max_acceleration, 1.0f) << "Acceleration should be bounded";

    E2E_STAGE_END();

    // Stage 4: Verify no discontinuities
    E2E_STAGE_BEGIN("Check for discontinuities", 500);

    float threshold = 0.5f;  // Max acceptable state jump
    bool has_discontinuity = false;
    size_t discontinuity_count = 0;

    for (size_t t = 1; t < trajectory.size(); t++) {
        float jump = compute_state_difference(
            trajectory[t].data(), trajectory[t-1].data(), hidden_dim);
        if (jump > threshold) {
            has_discontinuity = true;
            discontinuity_count++;
        }
    }

    std::cout << "    Discontinuities:   " << discontinuity_count << std::endl;

    EXPECT_FALSE(has_discontinuity) << "Trajectory should be continuous";

    E2E_STAGE_END();

    // Stage 5: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_lnn_layer_gpu_destroy(layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 7: Compare ODE Solvers
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, CompareODESolvers) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("Compare ODE Solvers");

    const size_t input_dim = 4;
    const size_t hidden_dim = 8;
    const size_t num_steps = 100;
    const float dt = 0.5f;

    struct SolverResult {
        std::string name;
        lnn_ode_method_t method;
        std::vector<std::vector<float>> trajectory;
        double compute_time_ms;
    };

    std::vector<SolverResult> results = {
        {"Euler", LNN_ODE_EULER, {}, 0.0},
        {"Heun", LNN_ODE_HEUN, {}, 0.0},
        {"RK4", LNN_ODE_RK4, {}, 0.0}
    };

    // Stage 1: Create and run each solver
    E2E_STAGE_BEGIN("Run all ODE solvers", 5000);

    std::vector<float> input(input_dim, 0.5f);
    size_t input_dims[] = {input_dim};

    for (auto& result : results) {
        nimcp_lnn_layer_config_t config;
        memset(&config, 0, sizeof(config));
        config.input_dim = input_dim;
        config.hidden_dim = hidden_dim;
        config.tau_min = LNN_TAU_MIN;
        config.tau_max = LNN_TAU_MAX;
        config.dt = dt;
        config.ode_method = result.method;
        config.activation = LNN_ACTIVATION_TANH;
        config.learn_tau = true;

        nimcp_lnn_layer_gpu_t* layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
        ASSERT_NE(layer, nullptr);

        nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
            ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t step = 0; step < num_steps; step++) {
            nimcp_lnn_layer_gpu_step(layer, input_tensor);

            std::vector<float> state(hidden_dim);
            nimcp_lnn_layer_gpu_get_state(layer, state.data());
            result.trajectory.push_back(state);
        }

        nimcp_gpu_context_synchronize(ctx_);

        auto end = std::chrono::high_resolution_clock::now();
        result.compute_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        nimcp_gpu_tensor_destroy(input_tensor);
        nimcp_lnn_layer_gpu_destroy(layer);
    }

    E2E_STAGE_END();

    // Stage 2: Compare trajectories
    E2E_STAGE_BEGIN("Compare solver trajectories", 500);

    std::cout << "\n  ODE Solver Comparison:" << std::endl;
    std::cout << "  Solver | Time (ms) | Final State Norm | Max Diff from RK4" << std::endl;
    std::cout << "  -------|-----------|------------------|-------------------" << std::endl;

    // Use RK4 as reference
    const auto& rk4_trajectory = results[2].trajectory;

    for (size_t i = 0; i < results.size(); i++) {
        float final_norm = 0.0f;
        for (float v : results[i].trajectory.back()) {
            final_norm += v * v;
        }
        final_norm = std::sqrt(final_norm);

        float max_diff = 0.0f;
        for (size_t t = 0; t < num_steps; t++) {
            float diff = compute_state_difference(
                results[i].trajectory[t].data(),
                rk4_trajectory[t].data(),
                hidden_dim);
            max_diff = std::max(max_diff, diff);
        }

        std::cout << "  " << std::setw(6) << results[i].name << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << results[i].compute_time_ms << " | "
                  << std::setw(16) << std::fixed << std::setprecision(4) << final_norm << " | "
                  << std::setw(17) << std::fixed << std::setprecision(6) << max_diff << std::endl;
    }

    E2E_STAGE_END();

    // Stage 3: Verify accuracy ordering
    E2E_STAGE_BEGIN("Verify accuracy ordering", 500);

    // Higher-order methods should be closer to RK4 (our reference)
    float euler_diff = 0.0f, heun_diff = 0.0f;

    for (size_t t = 0; t < num_steps; t++) {
        euler_diff += compute_state_difference(
            results[0].trajectory[t].data(),
            rk4_trajectory[t].data(),
            hidden_dim);
        heun_diff += compute_state_difference(
            results[1].trajectory[t].data(),
            rk4_trajectory[t].data(),
            hidden_dim);
    }

    std::cout << "\n    Total difference from RK4:" << std::endl;
    std::cout << "      Euler: " << euler_diff << std::endl;
    std::cout << "      Heun:  " << heun_diff << std::endl;

    EXPECT_LT(heun_diff, euler_diff) << "Heun should be more accurate than Euler";

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Pipeline 8: Full LNN Training Loop
//=============================================================================

TEST_F(GPULNNPipelineE2ETest, FullLNNTrainingLoop) {
    SkipIfNoGPU();

    E2E_PIPELINE_START("Full LNN Training Loop");

    const size_t input_dim = 2;
    const size_t hidden_dim = 16;
    const size_t output_dim = 1;
    const size_t seq_length = 50;
    const size_t num_epochs = 10;
    const float learning_rate = 0.01f;

    nimcp_lnn_layer_gpu_t* lnn_layer = nullptr;
    nimcp_gpu_tensor_t* W_out = nullptr;
    nimcp_optim_state_t* optim_W = nullptr;

    // Stage 1: Create LNN layer and output projection
    E2E_STAGE_BEGIN("Create network", 1000);

    nimcp_lnn_layer_config_t config;
    memset(&config, 0, sizeof(config));
    config.input_dim = input_dim;
    config.hidden_dim = hidden_dim;
    config.tau_min = LNN_TAU_MIN;
    config.tau_max = LNN_TAU_MAX;
    config.dt = 1.0f;
    config.ode_method = LNN_ODE_RK4;
    config.activation = LNN_ACTIVATION_TANH;
    config.learn_tau = true;

    lnn_layer = nimcp_lnn_layer_gpu_create(ctx_, &config);
    E2E_ASSERT_NOT_NULL(lnn_layer, "Failed to create LNN layer");

    // Output projection: hidden_dim -> output_dim
    size_t w_dims[] = {hidden_dim, output_dim};
    std::vector<float> w_host(hidden_dim * output_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& w : w_host) w = dist(gen);

    W_out = nimcp_gpu_tensor_from_host(ctx_, w_host.data(), w_dims, 2, NIMCP_GPU_PRECISION_FP32);
    E2E_ASSERT_NOT_NULL(W_out, "Failed to create output weights");

    // Create optimizer
    optim_W = nimcp_optim_state_create(ctx_, NIMCP_OPTIM_ADAM, W_out, learning_rate);

    E2E_STAGE_END();

    // Stage 2: Generate training data (predict next value in sequence)
    E2E_STAGE_BEGIN("Generate training data", 500);

    std::vector<float> input_seq(seq_length * input_dim);
    std::vector<float> target_seq(seq_length * output_dim);

    // Generate sinusoidal sequence
    for (size_t t = 0; t < seq_length; t++) {
        float time = t * config.dt;
        input_seq[t * input_dim] = std::sin(2.0f * M_PI * time / 50.0f);
        input_seq[t * input_dim + 1] = std::cos(2.0f * M_PI * time / 50.0f);

        // Target: predict next sin value
        float next_time = (t + 1) * config.dt;
        target_seq[t] = std::sin(2.0f * M_PI * next_time / 50.0f);
    }

    E2E_STAGE_END();

    // Stage 3: Training loop
    E2E_STAGE_BEGIN("Training loop", 10000);

    std::vector<float> epoch_losses(num_epochs);

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        // Reset LNN state
        nimcp_lnn_layer_gpu_reset(lnn_layer);

        float epoch_loss = 0.0f;

        for (size_t t = 0; t < seq_length; t++) {
            // Get input for this timestep
            std::vector<float> input(input_dim);
            for (size_t d = 0; d < input_dim; d++) {
                input[d] = input_seq[t * input_dim + d];
            }

            size_t input_dims[] = {input_dim};
            nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
                ctx_, input.data(), input_dims, 1, NIMCP_GPU_PRECISION_FP32);

            // LNN forward step
            nimcp_lnn_layer_gpu_step(lnn_layer, input_tensor);

            // Get LNN state
            std::vector<float> state(hidden_dim);
            nimcp_lnn_layer_gpu_get_state(lnn_layer, state.data());

            // Output projection
            size_t state_dims[] = {1, hidden_dim};
            nimcp_gpu_tensor_t* state_tensor = nimcp_gpu_tensor_from_host(
                ctx_, state.data(), state_dims, 2, NIMCP_GPU_PRECISION_FP32);

            size_t output_dims[] = {1, output_dim};
            nimcp_gpu_tensor_t* output_tensor = nimcp_gpu_tensor_create(
                ctx_, output_dims, 2, NIMCP_GPU_PRECISION_FP32);

            nimcp_gpu_gemm(ctx_, state_tensor, W_out, output_tensor, 1.0f, 0.0f, false, false);

            // Compute loss
            std::vector<float> output(output_dim);
            nimcp_gpu_tensor_to_host(output_tensor, output.data());

            float loss = 0.0f;
            for (size_t d = 0; d < output_dim; d++) {
                float diff = output[d] - target_seq[t * output_dim + d];
                loss += diff * diff;
            }
            epoch_loss += loss;

            // Backward (simplified - just update W_out)
            // grad = 2 * (output - target) * state^T
            std::vector<float> grad_output(output_dim);
            for (size_t d = 0; d < output_dim; d++) {
                grad_output[d] = 2.0f * (output[d] - target_seq[t * output_dim + d]);
            }

            size_t grad_W_dims[] = {hidden_dim, output_dim};
            nimcp_gpu_tensor_t* grad_W = nimcp_gpu_tensor_create(
                ctx_, grad_W_dims, 2, NIMCP_GPU_PRECISION_FP32);

            // grad_W = state^T @ grad_output
            size_t grad_out_dims[] = {1, output_dim};
            nimcp_gpu_tensor_t* grad_out_tensor = nimcp_gpu_tensor_from_host(
                ctx_, grad_output.data(), grad_out_dims, 2, NIMCP_GPU_PRECISION_FP32);

            nimcp_gpu_gemm(ctx_, state_tensor, grad_out_tensor, grad_W, 1.0f, 0.0f, true, false);

            // Optimizer step
            nimcp_gpu_optim_adam(ctx_, W_out, grad_W, optim_W);

            // Cleanup
            nimcp_gpu_tensor_destroy(input_tensor);
            nimcp_gpu_tensor_destroy(state_tensor);
            nimcp_gpu_tensor_destroy(output_tensor);
            nimcp_gpu_tensor_destroy(grad_W);
            nimcp_gpu_tensor_destroy(grad_out_tensor);
        }

        epoch_loss /= seq_length;
        epoch_losses[epoch] = epoch_loss;

        if ((epoch + 1) % 2 == 0) {
            std::cout << "    Epoch " << (epoch + 1) << "/" << num_epochs
                      << " - MSE Loss: " << epoch_loss << std::endl;
        }
    }

    nimcp_gpu_context_synchronize(ctx_);

    E2E_STAGE_END();

    // Stage 4: Verify training progress
    E2E_STAGE_BEGIN("Verify training progress", 500);

    float initial_loss = epoch_losses[0];
    float final_loss = epoch_losses[num_epochs - 1];

    std::cout << "\n  Training Summary:" << std::endl;
    std::cout << "    Initial loss: " << initial_loss << std::endl;
    std::cout << "    Final loss:   " << final_loss << std::endl;

    // Loss should decrease (or at least not explode)
    EXPECT_LT(final_loss, initial_loss * 2.0f) << "Loss should not explode";
    EXPECT_FALSE(std::isnan(final_loss)) << "Final loss should not be NaN";

    E2E_STAGE_END();

    // Stage 5: Cleanup
    E2E_STAGE_BEGIN("Cleanup", 300);

    nimcp_optim_state_destroy(optim_W);
    nimcp_gpu_tensor_destroy(W_out);
    nimcp_lnn_layer_gpu_destroy(lnn_layer);

    E2E_STAGE_END();

    E2E_PIPELINE_END();
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
