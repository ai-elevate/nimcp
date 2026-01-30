/* ============================================================================
 * Financial GPU Regression Tests
 * ============================================================================
 * WHAT: Regression tests for GPU financial computing operations
 * WHY:  Prevent reintroduction of fixed GPU financial-related bugs
 * HOW:  Test specific bug scenarios, numerical stability, edge cases
 * ============================================================================
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>
#include <random>

#ifdef NIMCP_ENABLE_CUDA
extern "C" {
#include "gpu/financial/nimcp_financial_monte_carlo_gpu.h"
#include "gpu/financial/nimcp_financial_optimization_gpu.h"
#include "gpu/financial/nimcp_financial_risk_gpu.h"
#include "gpu/financial/nimcp_financial_derivatives_gpu.h"
#include "gpu/nimcp_gpu_context.h"
}
#endif

namespace {

constexpr float TOLERANCE = 1e-5f;
constexpr float STATISTICAL_TOLERANCE = 0.05f;

class FinancialGPURegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef NIMCP_ENABLE_CUDA
        ctx_ = nimcp_gpu_context_create(nullptr);
        if (ctx_) {
            rng_ = fin_gpu_rng_create(ctx_, 10000, 12345);
        }
#endif
    }

    void TearDown() override {
#ifdef NIMCP_ENABLE_CUDA
        if (rng_) {
            fin_gpu_rng_destroy(rng_);
            rng_ = nullptr;
        }
        if (ctx_) {
            nimcp_gpu_context_destroy(ctx_);
            ctx_ = nullptr;
        }
#endif
    }

#ifdef NIMCP_ENABLE_CUDA
    nimcp_gpu_context_t* ctx_ = nullptr;
    fin_gpu_rng_t* rng_ = nullptr;
#endif
};

/* ============================================================================
 * Bug #1: Monte Carlo drift-volatility scaling error
 * ============================================================================
 * Symptom: Mean terminal value doesn't match expected with geometric Brownian
 * Fix: Correct drift term: (mu - 0.5*sigma^2)*dt, not mu*dt
 */
TEST_F(FinancialGPURegressionTest, MonteCarloGBMDriftCorrection) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";
    ASSERT_NE(rng_, nullptr) << "RNG required for this test";

    // With correct drift-volatility adjustment:
    // E[S_T] = S_0 * exp(mu * T)  (NOT mu - 0.5*sigma^2 for expectation!)
    const float S0 = 100.0f;
    const float mu = 0.10f;
    const float sigma = 0.30f;
    const float T = 1.0f;
    const int num_paths = 100000;

    fin_monte_carlo_gpu_params_t params = {
        .initial_value = S0,
        .drift = mu,
        .volatility = sigma,
        .time_horizon = T,
        .num_steps = 252,
        .num_paths = static_cast<uint32_t>(num_paths)
    };

    fin_monte_carlo_gpu_result_t result;
    result.terminal_values = new float[num_paths];
    result.path_returns = new float[num_paths];

    bool success = fin_monte_carlo_gpu_simulate(ctx_, rng_, &params, &result);

    if (success) {
        // Expected mean: S0 * exp(mu * T)
        float expected_mean = S0 * std::exp(mu * T);

        // Mean should be close to expected (within statistical tolerance)
        float rel_error = std::abs(result.mean_terminal - expected_mean) / expected_mean;
        EXPECT_LT(rel_error, STATISTICAL_TOLERANCE)
            << "GBM mean terminal value: got " << result.mean_terminal
            << ", expected " << expected_mean
            << " (relative error: " << rel_error * 100 << "%)";
    }

    delete[] result.terminal_values;
    delete[] result.path_returns;
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #2: VaR sorting instability with duplicate values
 * ============================================================================
 * Symptom: Incorrect VaR when many returns have same value
 * Fix: Use stable sort or handle ties properly in percentile calculation
 */
TEST_F(FinancialGPURegressionTest, VaRSortingStability) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Create returns with many duplicate values
    const int n = 10000;
    std::vector<float> returns(n);

    // 50% of values are -0.01, 50% are 0.01
    for (int i = 0; i < n; i++) {
        returns[i] = (i < n/2) ? -0.01f : 0.01f;
    }

    fin_risk_gpu_params_t params = {
        .confidence_level = 0.95f,
        .time_horizon_days = 1
    };

    fin_risk_gpu_result_t result;

    bool success = fin_risk_gpu_compute(ctx_, returns.data(), n, &params, &result);

    if (success) {
        // 95% VaR should be -0.01 (5th percentile is in the lower half)
        EXPECT_NEAR(result.var_95, -0.01f, 0.001f)
            << "VaR should handle duplicate values correctly";
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #3: Portfolio optimization singular covariance matrix
 * ============================================================================
 * Symptom: Crash or garbage results with perfectly correlated assets
 * Fix: Add regularization or detect singularity
 */
TEST_F(FinancialGPURegressionTest, OptimizationSingularCovariance) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Perfectly correlated assets -> singular covariance matrix
    const int n = 3;
    float returns[] = {0.05f, 0.05f, 0.05f};  // Same return
    float covariance[] = {
        0.04f, 0.04f, 0.04f,  // Row 1: perfect correlation
        0.04f, 0.04f, 0.04f,  // Row 2: same as row 1
        0.04f, 0.04f, 0.04f   // Row 3: same as rows 1 & 2
    };

    fin_optimization_gpu_params_t params = {
        .target_return = 0.05f,
        .max_iterations = 1000,
        .convergence_threshold = 1e-6f,
        .constraint_type = FIN_OPT_CONSTRAINT_LONG_ONLY,
        .risk_aversion = 1.0f
    };

    fin_optimization_gpu_result_t result;
    result.weights = new float[n];

    bool success = fin_optimization_gpu_mean_variance(
        ctx_, returns, covariance, n, &params, &result);

    if (success) {
        // Should return valid weights that sum to 1
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            EXPECT_TRUE(std::isfinite(result.weights[i]))
                << "Weight " << i << " should be finite";
            sum += result.weights[i];
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Weights should sum to 1";
    }
    // It's also acceptable to fail gracefully on singular matrix

    delete[] result.weights;
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #4: Black-Scholes near-zero time to maturity
 * ============================================================================
 * Symptom: NaN or incorrect price as T -> 0
 * Fix: Handle limit case explicitly
 */
TEST_F(FinancialGPURegressionTest, BlackScholesNearZeroTime) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Very small times to maturity
    std::vector<float> small_times = {
        1e-10f, 1e-6f, 1e-4f, 0.001f, 0.0f
    };

    for (float T : small_times) {
        float S = 100.0f, K = 95.0f, r = 0.05f, sigma = 0.2f;

        float spots[] = {S};
        float strikes[] = {K};
        float rates[] = {r};
        float vols[] = {sigma};
        float times[] = {T};
        bool is_call[] = {true};
        float prices[1] = {0.0f};

        bool success = fin_derivatives_gpu_black_scholes_batch(
            ctx_, spots, strikes, rates, vols, times, is_call, prices, 1);

        if (success) {
            EXPECT_TRUE(std::isfinite(prices[0]))
                << "BS price should be finite for T=" << T;

            // At T=0, call should be worth max(S-K, 0) = 5
            if (T < 1e-6f) {
                float intrinsic = std::max(S - K, 0.0f);
                EXPECT_NEAR(prices[0], intrinsic, 0.01f)
                    << "Near-zero time call should equal intrinsic";
            }
        }
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #5: Implied volatility Newton-Raphson divergence
 * ============================================================================
 * Symptom: Non-convergence or negative IV for deep OTM options
 * Fix: Use bounded bisection as fallback, clamp IV range
 */
TEST_F(FinancialGPURegressionTest, ImpliedVolDivergence) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Deep OTM option with very low price -> difficult IV calculation
    float S = 100.0f, K = 200.0f, r = 0.05f, T = 0.25f;
    float market_price = 0.001f;  // Almost worthless deep OTM call

    float implied_vol = 0.0f;
    bool success = fin_derivatives_gpu_implied_vol(
        ctx_, market_price, S, K, r, T, true, &implied_vol);

    if (success) {
        EXPECT_TRUE(std::isfinite(implied_vol))
            << "IV should be finite for deep OTM option";
        EXPECT_GT(implied_vol, 0.0f) << "IV should be positive";
        EXPECT_LT(implied_vol, 5.0f) << "IV should be reasonable (< 500%)";
    }
    // May fail for extreme cases, which is acceptable

    // Near-ATM with reasonable price should always converge
    float normal_price = 10.0f;  // ATM call with ~10% of spot
    success = fin_derivatives_gpu_implied_vol(
        ctx_, normal_price, 100.0f, 100.0f, 0.05f, 1.0f, true, &implied_vol);

    EXPECT_TRUE(success) << "IV should converge for ATM option";
    if (success) {
        EXPECT_GT(implied_vol, 0.05f);
        EXPECT_LT(implied_vol, 1.0f);
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #6: Binomial tree American option early exercise boundary
 * ============================================================================
 * Symptom: American put valued less than intrinsic value
 * Fix: Enforce early exercise check at every node
 */
TEST_F(FinancialGPURegressionTest, AmericanPutIntrinsicBound) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Deep ITM American put with high interest rate
    // Should be worth at least intrinsic value
    fin_derivatives_gpu_params_t params = {
        .spot_price = 80.0f,
        .strike_price = 100.0f,
        .risk_free_rate = 0.15f,  // High rate encourages early exercise
        .volatility = 0.20f,
        .time_to_maturity = 1.0f,
        .num_steps = 500,
        .is_call = false,
        .is_american = true
    };

    fin_derivatives_gpu_result_t result;
    memset(&result, 0, sizeof(result));

    bool success = fin_derivatives_gpu_binomial_tree(ctx_, &params, &result);

    if (success) {
        float intrinsic = params.strike_price - params.spot_price;  // 20
        EXPECT_GE(result.price, intrinsic - TOLERANCE)
            << "American put should be worth at least intrinsic: got "
            << result.price << ", intrinsic is " << intrinsic;
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #7: Negative volatility input handling
 * ============================================================================
 * Symptom: NaN or crash with negative volatility
 * Fix: Validate inputs and return error
 */
TEST_F(FinancialGPURegressionTest, NegativeVolatilityHandling) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    float spots[] = {100.0f};
    float strikes[] = {100.0f};
    float rates[] = {0.05f};
    float vols[] = {-0.2f};  // Invalid: negative volatility
    float times[] = {1.0f};
    bool is_call[] = {true};
    float prices[1] = {0.0f};

    bool success = fin_derivatives_gpu_black_scholes_batch(
        ctx_, spots, strikes, rates, vols, times, is_call, prices, 1);

    // Should either fail or return valid positive price (using abs(vol))
    if (success) {
        EXPECT_TRUE(std::isfinite(prices[0]));
        EXPECT_GE(prices[0], 0.0f);
    }
    // Failing gracefully is also acceptable
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #8: RNG state corruption with multiple streams
 * ============================================================================
 * Symptom: Correlated or repeated random numbers across streams
 * Fix: Proper stream initialization with different seeds
 */
TEST_F(FinancialGPURegressionTest, RNGStreamIndependence) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Create two RNGs with different seeds
    fin_gpu_rng_t* rng1 = fin_gpu_rng_create(ctx_, 1000, 11111);
    fin_gpu_rng_t* rng2 = fin_gpu_rng_create(ctx_, 1000, 22222);

    if (rng1 && rng2) {
        fin_monte_carlo_gpu_params_t params = {
            .initial_value = 100.0f,
            .drift = 0.05f,
            .volatility = 0.20f,
            .time_horizon = 1.0f,
            .num_steps = 252,
            .num_paths = 1000
        };

        fin_monte_carlo_gpu_result_t result1, result2;
        result1.terminal_values = new float[1000];
        result1.path_returns = new float[1000];
        result2.terminal_values = new float[1000];
        result2.path_returns = new float[1000];

        bool success1 = fin_monte_carlo_gpu_simulate(ctx_, rng1, &params, &result1);
        bool success2 = fin_monte_carlo_gpu_simulate(ctx_, rng2, &params, &result2);

        if (success1 && success2) {
            // Results should be different (different seeds)
            int same_count = 0;
            for (int i = 0; i < 1000; i++) {
                if (std::abs(result1.terminal_values[i] - result2.terminal_values[i]) < 1e-6f) {
                    same_count++;
                }
            }
            EXPECT_LT(same_count, 50)
                << "Different seeds should produce different results";
        }

        delete[] result1.terminal_values;
        delete[] result1.path_returns;
        delete[] result2.terminal_values;
        delete[] result2.path_returns;
    }

    if (rng1) fin_gpu_rng_destroy(rng1);
    if (rng2) fin_gpu_rng_destroy(rng2);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #9: CVaR calculation with sorted array ownership
 * ============================================================================
 * Symptom: Double-free or memory corruption when reusing sorted array
 * Fix: Make defensive copy or clearly document ownership
 */
TEST_F(FinancialGPURegressionTest, CVaRMemorySafety) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    std::vector<float> returns(1000);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& r : returns) {
        r = dist(gen);
    }

    fin_risk_gpu_params_t params = {
        .confidence_level = 0.95f,
        .time_horizon_days = 1
    };

    // Call multiple times with same data
    for (int i = 0; i < 3; i++) {
        fin_risk_gpu_result_t result;

        bool success = fin_risk_gpu_compute(ctx_, returns.data(), 1000, &params, &result);

        EXPECT_TRUE(success) << "Repeated CVaR calculation should succeed";
        if (success) {
            EXPECT_TRUE(std::isfinite(result.var_95));
            EXPECT_TRUE(std::isfinite(result.cvar_95));
        }
    }

    // Original array should be unchanged
    // (verify by checking if it's still a valid normal-ish distribution)
    float sum = 0.0f;
    for (const auto& r : returns) {
        EXPECT_TRUE(std::isfinite(r)) << "Original array corrupted";
        sum += r;
    }
    float mean = sum / returns.size();
    EXPECT_NEAR(mean, 0.0f, 0.01f) << "Original array mean should be ~0";
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * Bug #10: Greeks finite difference step size
 * ============================================================================
 * Symptom: Inaccurate Greeks with inappropriate bump size
 * Fix: Use adaptive step size based on spot price magnitude
 */
TEST_F(FinancialGPURegressionTest, GreeksFiniteDifferenceAccuracy) {
#ifdef NIMCP_ENABLE_CUDA
    ASSERT_NE(ctx_, nullptr) << "GPU context required for this test";

    // Test with different spot price magnitudes
    std::vector<float> spot_prices = {1.0f, 100.0f, 10000.0f};

    for (float S : spot_prices) {
        fin_derivatives_gpu_params_t params = {
            .spot_price = S,
            .strike_price = S,  // ATM
            .risk_free_rate = 0.05f,
            .volatility = 0.20f,
            .time_to_maturity = 1.0f,
            .num_steps = 200,
            .is_call = true,
            .is_american = false
        };

        fin_derivatives_gpu_greeks_t greeks;
        bool success = fin_derivatives_gpu_compute_greeks(ctx_, &params, &greeks);

        if (success) {
            // ATM call delta should be around 0.5-0.6 regardless of spot magnitude
            EXPECT_GT(greeks.delta, 0.4f)
                << "Delta too low for S=" << S;
            EXPECT_LT(greeks.delta, 0.7f)
                << "Delta too high for S=" << S;

            // Gamma should be positive and scale appropriately
            EXPECT_GT(greeks.gamma, 0.0f)
                << "Gamma should be positive for S=" << S;

            // All Greeks should be finite
            EXPECT_TRUE(std::isfinite(greeks.delta));
            EXPECT_TRUE(std::isfinite(greeks.gamma));
            EXPECT_TRUE(std::isfinite(greeks.theta));
            EXPECT_TRUE(std::isfinite(greeks.vega));
            EXPECT_TRUE(std::isfinite(greeks.rho));
        }
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

}  // namespace
