//=============================================================================
// test_financial_gpu_regression.cpp - Regression Tests for GPU Financial Module
//=============================================================================
/**
 * @file test_financial_gpu_regression.cpp
 * @brief Regression tests for GPU-accelerated financial computations
 *
 * Tests numerical stability, edge cases, known values, and ensures
 * consistent behavior across updates. Critical for financial applications
 * where precision is paramount.
 *
 * COVERAGE:
 *   - Black-Scholes known values
 *   - Monte Carlo statistical properties
 *   - Optimization convergence guarantees
 *   - Risk metric precision
 *   - Numerical stability with extreme inputs
 *   - Deterministic behavior (where applicable)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

extern "C" {
#include "gpu/financial/nimcp_financial_gpu.h"
#include "gpu/financial/nimcp_financial_risk_gpu.h"
#include "gpu/financial/nimcp_financial_derivatives_gpu.h"
#include "gpu/context/nimcp_gpu_context.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class FinancialGPURegressionTest : public ::testing::Test {
protected:
    nimcp_gpu_context_t* ctx = nullptr;
    fin_gpu_rng_t* rng = nullptr;

    void SetUp() override {
        nimcp_gpu_config_t config = nimcp_gpu_config_default();
        ctx = nimcp_gpu_context_create(&config);
        ASSERT_NE(ctx, nullptr) << "Failed to create GPU context";

        rng = fin_gpu_rng_create(ctx, 100000, 42);
        ASSERT_NE(rng, nullptr) << "Failed to create GPU RNG";
    }

    void TearDown() override {
        if (rng) {
            fin_gpu_rng_destroy(rng);
            rng = nullptr;
        }
        if (ctx) {
            nimcp_gpu_context_destroy(ctx);
            ctx = nullptr;
        }
    }

    float normCDF(float x) {
        return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }

    float blackScholesCall(float S, float K, float r, float sigma, float T) {
        float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * std::sqrt(T));
        float d2 = d1 - sigma * std::sqrt(T);
        return S * normCDF(d1) - K * std::exp(-r * T) * normCDF(d2);
    }

    float blackScholesPut(float S, float K, float r, float sigma, float T) {
        float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * std::sqrt(T));
        float d2 = d1 - sigma * std::sqrt(T);
        return K * std::exp(-r * T) * normCDF(-d2) - S * normCDF(-d1);
    }
};

//=============================================================================
// Black-Scholes Known Value Tests
//=============================================================================

TEST_F(FinancialGPURegressionTest, BlackScholesKnownValues) {
    // Test against known Black-Scholes prices
    // Source: Standard finance textbook examples

    struct TestCase {
        float S, K, r, sigma, T;
        fin_option_type_t type;
        float expected;
        float tolerance;
    };

    std::vector<TestCase> cases = {
        // ATM call, 1 year, 20% vol, 5% rate
        {100.0f, 100.0f, 0.05f, 0.20f, 1.0f, FIN_OPT_CALL, 10.45f, 0.1f},
        // ATM put
        {100.0f, 100.0f, 0.05f, 0.20f, 1.0f, FIN_OPT_PUT, 5.57f, 0.1f},
        // Deep ITM call
        {150.0f, 100.0f, 0.05f, 0.20f, 1.0f, FIN_OPT_CALL, 54.15f, 0.2f},
        // Deep OTM call
        {100.0f, 150.0f, 0.05f, 0.20f, 1.0f, FIN_OPT_CALL, 0.69f, 0.1f},
        // Short expiry ATM
        {100.0f, 100.0f, 0.05f, 0.20f, 0.1f, FIN_OPT_CALL, 3.16f, 0.1f},
        // High vol
        {100.0f, 100.0f, 0.05f, 0.50f, 1.0f, FIN_OPT_CALL, 22.13f, 0.2f},
    };

    for (size_t i = 0; i < cases.size(); i++) {
        const auto& tc = cases[i];
        float price = fin_deriv_gpu_black_scholes(ctx, tc.S, tc.K, tc.r, tc.sigma, tc.T, tc.type);

        // Verify against our CPU reference
        float cpu_price = (tc.type == FIN_OPT_CALL)
            ? blackScholesCall(tc.S, tc.K, tc.r, tc.sigma, tc.T)
            : blackScholesPut(tc.S, tc.K, tc.r, tc.sigma, tc.T);

        EXPECT_NEAR(price, cpu_price, 0.01f) << "Case " << i << ": GPU vs CPU mismatch";
        EXPECT_NEAR(price, tc.expected, tc.tolerance) << "Case " << i << ": Known value mismatch";
    }
}

TEST_F(FinancialGPURegressionTest, PutCallParity) {
    // Put-call parity: C - P = S - K*exp(-rT)
    std::vector<std::tuple<float, float, float, float, float>> params = {
        {100.0f, 100.0f, 0.05f, 0.20f, 1.0f},
        {100.0f, 90.0f, 0.03f, 0.30f, 0.5f},
        {100.0f, 110.0f, 0.08f, 0.15f, 2.0f},
    };

    for (const auto& [S, K, r, sigma, T] : params) {
        float call = fin_deriv_gpu_black_scholes(ctx, S, K, r, sigma, T, FIN_OPT_CALL);
        float put = fin_deriv_gpu_black_scholes(ctx, S, K, r, sigma, T, FIN_OPT_PUT);

        float lhs = call - put;
        float rhs = S - K * std::exp(-r * T);

        EXPECT_NEAR(lhs, rhs, 0.01f) << "Put-call parity violated";
    }
}

//=============================================================================
// Numerical Stability Tests
//=============================================================================

TEST_F(FinancialGPURegressionTest, BlackScholesExtremeValues) {
    // Test with extreme but valid inputs
    std::vector<std::tuple<float, float, float, float, float>> extreme_cases = {
        {0.01f, 100.0f, 0.05f, 0.20f, 1.0f},      // Very low spot (deep OTM)
        {10000.0f, 100.0f, 0.05f, 0.20f, 1.0f},   // Very high spot (deep ITM)
        {100.0f, 100.0f, 0.00001f, 0.20f, 1.0f},  // Very low rate
        {100.0f, 100.0f, 0.50f, 0.20f, 1.0f},     // High rate
        {100.0f, 100.0f, 0.05f, 0.01f, 1.0f},     // Very low vol
        {100.0f, 100.0f, 0.05f, 2.0f, 1.0f},      // Very high vol
        {100.0f, 100.0f, 0.05f, 0.20f, 0.001f},   // Very short expiry
        {100.0f, 100.0f, 0.05f, 0.20f, 10.0f},    // Very long expiry
    };

    for (size_t i = 0; i < extreme_cases.size(); i++) {
        const auto& [S, K, r, sigma, T] = extreme_cases[i];
        float price = fin_deriv_gpu_black_scholes(ctx, S, K, r, sigma, T, FIN_OPT_CALL);

        EXPECT_FALSE(std::isnan(price)) << "NaN at case " << i;
        EXPECT_FALSE(std::isinf(price)) << "Inf at case " << i;
        EXPECT_GE(price, 0.0f) << "Negative price at case " << i;
    }
}

TEST_F(FinancialGPURegressionTest, VaRNumericalStability) {
    // Test VaR computation with edge cases
    std::vector<float> returns(1000);

    // Normal returns
    for (int i = 0; i < 1000; i++) {
        returns[i] = 0.001f * (i - 500) / 500.0f;
    }

    fin_risk_gpu_params_t params = fin_risk_gpu_params_default();
    params.num_returns = 1000;

    fin_risk_gpu_result_t result = {0};
    bool ok = fin_risk_gpu_compute(ctx, returns.data(), &params, &result);
    ASSERT_TRUE(ok);

    EXPECT_FALSE(std::isnan(result.var_95));
    EXPECT_FALSE(std::isnan(result.cvar_95));
    EXPECT_GT(result.var_95, 0.0f);
    EXPECT_GE(result.cvar_95, result.var_95);  // CVaR >= VaR
}

TEST_F(FinancialGPURegressionTest, OptimizationNumericalStability) {
    // Test optimization with poorly conditioned covariance matrix
    const uint32_t N = 5;

    // Near-singular covariance (highly correlated assets)
    std::vector<float> cov(N * N);
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            if (i == j) {
                cov[i * N + j] = 0.04f;  // Variance
            } else {
                cov[i * N + j] = 0.039f; // Very high correlation ≈ 0.975
            }
        }
    }

    std::vector<float> returns(N, 0.10f);

    fin_optimization_gpu_params_t opt_params = fin_optimization_gpu_params_default();
    opt_params.n_assets = N;
    opt_params.regularization = 1e-4f;  // Add regularization

    fin_optimization_gpu_result_t result = {0};
    bool ok = fin_optimization_gpu_mean_variance(ctx, returns.data(), cov.data(), &opt_params, &result);
    ASSERT_TRUE(ok);

    // Check weights are valid
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FALSE(std::isnan(result.optimal_weights[i])) << "NaN weight at " << i;
        EXPECT_FALSE(std::isinf(result.optimal_weights[i])) << "Inf weight at " << i;
        weight_sum += result.optimal_weights[i];
    }
    EXPECT_NEAR(weight_sum, 1.0f, 0.1f) << "Weights should approximately sum to 1";

    fin_optimization_gpu_result_free(&result);
}

//=============================================================================
// Monte Carlo Statistical Properties
//=============================================================================

TEST_F(FinancialGPURegressionTest, MonteCarloExpectedValue) {
    // E[S_T] = S_0 * exp(μT) for GBM
    fin_monte_carlo_gpu_params_t params = fin_monte_carlo_gpu_params_default();
    params.initial_value = 100.0f;
    params.drift = 0.10f;
    params.volatility = 0.20f;
    params.time_horizon = 1.0f;
    params.num_paths = 100000;
    params.num_steps = 252;
    params.antithetic = true;

    fin_monte_carlo_gpu_result_t result = {0};
    bool ok = fin_monte_carlo_gpu_simulate(ctx, rng, &params, &result);
    ASSERT_TRUE(ok);

    // Expected terminal value under risk-neutral measure
    float expected_mean = params.initial_value * std::exp(params.drift * params.time_horizon);

    EXPECT_NEAR(result.mean_value, expected_mean, expected_mean * 0.05f)
        << "MC mean should be close to theoretical expectation";

    fin_monte_carlo_gpu_result_free(ctx, &result);
}

TEST_F(FinancialGPURegressionTest, MonteCarloVariance) {
    // Var[ln(S_T/S_0)] = σ²T for GBM
    fin_monte_carlo_gpu_params_t params = fin_monte_carlo_gpu_params_default();
    params.initial_value = 100.0f;
    params.drift = 0.05f;
    params.volatility = 0.25f;
    params.time_horizon = 1.0f;
    params.num_paths = 100000;
    params.num_steps = 252;

    fin_monte_carlo_gpu_result_t result = {0};
    bool ok = fin_monte_carlo_gpu_simulate(ctx, rng, &params, &result);
    ASSERT_TRUE(ok);

    // Standard deviation of returns should be close to σ√T
    float expected_std = params.volatility * std::sqrt(params.time_horizon);

    EXPECT_NEAR(result.std_return, expected_std, expected_std * 0.1f)
        << "MC std should be close to σ√T";

    fin_monte_carlo_gpu_result_free(ctx, &result);
}

TEST_F(FinancialGPURegressionTest, AntitheticVariateReduction) {
    // Antithetic variates should reduce variance
    fin_monte_carlo_gpu_params_t params = fin_monte_carlo_gpu_params_default();
    params.initial_value = 100.0f;
    params.num_paths = 50000;

    // Without antithetic
    params.antithetic = false;
    fin_monte_carlo_gpu_result_t result1 = {0};
    fin_monte_carlo_gpu_simulate(ctx, rng, &params, &result1);

    fin_gpu_rng_reseed(rng, 42);  // Reset RNG

    // With antithetic
    params.antithetic = true;
    fin_monte_carlo_gpu_result_t result2 = {0};
    fin_monte_carlo_gpu_simulate(ctx, rng, &params, &result2);

    // Antithetic should have smaller standard error (on average)
    // This is a statistical test, so use loose tolerance
    EXPECT_LE(result2.std_error, result1.std_error * 1.2f)
        << "Antithetic should generally reduce variance";

    fin_monte_carlo_gpu_result_free(ctx, &result1);
    fin_monte_carlo_gpu_result_free(ctx, &result2);
}

//=============================================================================
// Greeks Consistency Tests
//=============================================================================

TEST_F(FinancialGPURegressionTest, DeltaBounds) {
    // Call delta should be in [0, 1], Put delta in [-1, 0]
    for (float K = 80.0f; K <= 120.0f; K += 5.0f) {
        fin_derivatives_gpu_params_t params = fin_derivatives_gpu_params_default();
        params.spot = 100.0f;
        params.strike = K;
        params.compute_greeks = true;

        params.option_type = FIN_OPT_CALL;
        fin_derivatives_gpu_result_t call_result = {0};
        fin_derivatives_gpu_binomial_tree(ctx, &params, &call_result);

        EXPECT_GE(call_result.delta, 0.0f) << "Call delta >= 0 at K=" << K;
        EXPECT_LE(call_result.delta, 1.0f) << "Call delta <= 1 at K=" << K;

        params.option_type = FIN_OPT_PUT;
        fin_derivatives_gpu_result_t put_result = {0};
        fin_derivatives_gpu_binomial_tree(ctx, &params, &put_result);

        EXPECT_GE(put_result.delta, -1.0f) << "Put delta >= -1 at K=" << K;
        EXPECT_LE(put_result.delta, 0.0f) << "Put delta <= 0 at K=" << K;
    }
}

TEST_F(FinancialGPURegressionTest, GammaPositive) {
    // Gamma should always be positive for long options
    fin_derivatives_gpu_params_t params = fin_derivatives_gpu_params_default();
    params.compute_greeks = true;

    for (float K = 80.0f; K <= 120.0f; K += 10.0f) {
        params.strike = K;

        params.option_type = FIN_OPT_CALL;
        fin_derivatives_gpu_result_t result = {0};
        fin_derivatives_gpu_binomial_tree(ctx, &params, &result);

        EXPECT_GT(result.gamma, 0.0f) << "Gamma should be positive at K=" << K;
    }
}

TEST_F(FinancialGPURegressionTest, ThetaNegative) {
    // Theta should generally be negative for long options (time decay)
    fin_derivatives_gpu_params_t params = fin_derivatives_gpu_params_default();
    params.option_type = FIN_OPT_CALL;
    params.compute_greeks = true;

    fin_derivatives_gpu_result_t result = {0};
    fin_derivatives_gpu_binomial_tree(ctx, &params, &result);

    EXPECT_LT(result.theta, 0.0f) << "Theta should be negative for long call";
}

TEST_F(FinancialGPURegressionTest, VegaPositive) {
    // Vega should always be positive for long options
    fin_derivatives_gpu_params_t params = fin_derivatives_gpu_params_default();
    params.option_type = FIN_OPT_CALL;
    params.compute_greeks = true;

    fin_derivatives_gpu_result_t result = {0};
    fin_derivatives_gpu_binomial_tree(ctx, &params, &result);

    EXPECT_GT(result.vega, 0.0f) << "Vega should be positive";
}

//=============================================================================
// Binomial Tree Convergence
//=============================================================================

TEST_F(FinancialGPURegressionTest, BinomialTreeConvergence) {
    // More steps should give better convergence to BS
    fin_derivatives_gpu_params_t params = fin_derivatives_gpu_params_default();
    params.option_type = FIN_OPT_CALL;
    params.option_style = FIN_OPT_STYLE_EUROPEAN;

    float bs_price = blackScholesCall(params.spot, params.strike, params.risk_free_rate,
                                       params.volatility, params.time_to_expiry);

    float prev_error = 1000.0f;
    for (uint32_t steps : {50, 100, 500, 1000}) {
        params.tree_steps = steps;
        fin_derivatives_gpu_result_t result = {0};
        fin_derivatives_gpu_binomial_tree(ctx, &params, &result);

        float error = std::abs(result.price - bs_price);
        EXPECT_LT(error, prev_error + 0.01f) << "Error should decrease with more steps";
        prev_error = error;
    }
}

//=============================================================================
// Risk Metric Consistency
//=============================================================================

TEST_F(FinancialGPURegressionTest, VaROrdering) {
    // VaR(99%) should be >= VaR(95%)
    std::vector<float> returns(1000);
    for (int i = 0; i < 1000; i++) {
        float u1 = (static_cast<float>(rand()) / RAND_MAX);
        float u2 = (static_cast<float>(rand()) / RAND_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        returns[i] = 0.02f * z;
    }

    fin_risk_gpu_params_t params = fin_risk_gpu_params_default();
    params.num_returns = 1000;

    fin_risk_gpu_result_t result = {0};
    bool ok = fin_risk_gpu_compute(ctx, returns.data(), &params, &result);
    ASSERT_TRUE(ok);

    EXPECT_GE(result.var_99, result.var_95) << "VaR(99%) >= VaR(95%)";
    EXPECT_GE(result.cvar_99, result.cvar_95) << "CVaR(99%) >= CVaR(95%)";
}

TEST_F(FinancialGPURegressionTest, CVaRGreaterThanVaR) {
    // CVaR (Expected Shortfall) >= VaR by definition
    std::vector<float> returns(1000);
    for (int i = 0; i < 1000; i++) {
        returns[i] = 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }

    fin_risk_gpu_params_t params = fin_risk_gpu_params_default();
    params.num_returns = 1000;

    fin_risk_gpu_result_t result = {0};
    fin_risk_gpu_compute(ctx, returns.data(), &params, &result);

    EXPECT_GE(result.cvar_95, result.var_95) << "CVaR >= VaR at 95%";
    EXPECT_GE(result.cvar_99, result.var_99) << "CVaR >= VaR at 99%";
}

//=============================================================================
// Optimization Constraints
//=============================================================================

TEST_F(FinancialGPURegressionTest, LongOnlyConstraint) {
    const uint32_t N = 5;
    std::vector<float> returns = {0.08f, 0.10f, 0.12f, 0.06f, 0.14f};
    std::vector<float> cov(N * N);

    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            if (i == j) {
                cov[i * N + j] = 0.04f + 0.01f * i;
            } else {
                cov[i * N + j] = 0.01f;
            }
        }
    }

    fin_optimization_gpu_params_t params = fin_optimization_gpu_params_default();
    params.n_assets = N;
    params.long_only = true;

    fin_optimization_gpu_result_t result = {0};
    bool ok = fin_optimization_gpu_mean_variance(ctx, returns.data(), cov.data(), &params, &result);
    ASSERT_TRUE(ok);

    for (uint32_t i = 0; i < N; i++) {
        EXPECT_GE(result.optimal_weights[i], -0.001f) << "Long-only violated at " << i;
    }

    fin_optimization_gpu_result_free(&result);
}

TEST_F(FinancialGPURegressionTest, WeightsSumToOne) {
    const uint32_t N = 10;
    std::vector<float> returns(N);
    std::vector<float> cov(N * N);

    for (uint32_t i = 0; i < N; i++) {
        returns[i] = 0.05f + 0.02f * i / N;
        for (uint32_t j = 0; j < N; j++) {
            if (i == j) {
                cov[i * N + j] = 0.04f;
            } else {
                cov[i * N + j] = 0.01f;
            }
        }
    }

    fin_optimization_gpu_params_t params = fin_optimization_gpu_params_default();
    params.n_assets = N;

    fin_optimization_gpu_result_t result = {0};
    bool ok = fin_optimization_gpu_mean_variance(ctx, returns.data(), cov.data(), &params, &result);
    ASSERT_TRUE(ok);

    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum += result.optimal_weights[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f) << "Weights should sum to 1";

    fin_optimization_gpu_result_free(&result);
}

//=============================================================================
// Determinism Tests
//=============================================================================

TEST_F(FinancialGPURegressionTest, DeterministicBlackScholes) {
    // Black-Scholes should always give same result
    float price1 = fin_deriv_gpu_black_scholes(ctx, 100.0f, 100.0f, 0.05f, 0.2f, 1.0f, FIN_OPT_CALL);
    float price2 = fin_deriv_gpu_black_scholes(ctx, 100.0f, 100.0f, 0.05f, 0.2f, 1.0f, FIN_OPT_CALL);

    EXPECT_EQ(price1, price2) << "Black-Scholes should be deterministic";
}

TEST_F(FinancialGPURegressionTest, DeterministicOptimization) {
    const uint32_t N = 4;
    std::vector<float> returns = {0.08f, 0.10f, 0.12f, 0.09f};
    std::vector<float> cov = {
        0.04f, 0.01f, 0.01f, 0.01f,
        0.01f, 0.05f, 0.01f, 0.01f,
        0.01f, 0.01f, 0.06f, 0.01f,
        0.01f, 0.01f, 0.01f, 0.04f
    };

    fin_optimization_gpu_params_t params = fin_optimization_gpu_params_default();
    params.n_assets = N;

    fin_optimization_gpu_result_t result1 = {0}, result2 = {0};
    fin_optimization_gpu_mean_variance(ctx, returns.data(), cov.data(), &params, &result1);
    fin_optimization_gpu_mean_variance(ctx, returns.data(), cov.data(), &params, &result2);

    for (uint32_t i = 0; i < N; i++) {
        EXPECT_NEAR(result1.optimal_weights[i], result2.optimal_weights[i], 1e-6f)
            << "Optimization should be deterministic at " << i;
    }

    fin_optimization_gpu_result_free(&result1);
    fin_optimization_gpu_result_free(&result2);
}

//=============================================================================
// Known Issue Regressions
//=============================================================================

TEST_F(FinancialGPURegressionTest, ZeroVolatilityHandling) {
    // Zero volatility should not cause crash
    float price = fin_deriv_gpu_black_scholes(ctx, 100.0f, 95.0f, 0.05f, 0.0001f, 1.0f, FIN_OPT_CALL);

    EXPECT_FALSE(std::isnan(price));
    EXPECT_FALSE(std::isinf(price));
    // ITM call with ~0 vol should be close to intrinsic value
    float intrinsic = std::max(0.0f, 100.0f - 95.0f * std::exp(-0.05f));
    EXPECT_NEAR(price, intrinsic, 1.0f);
}

TEST_F(FinancialGPURegressionTest, VeryLongExpiryOption) {
    // Very long expiry should not overflow
    float price = fin_deriv_gpu_black_scholes(ctx, 100.0f, 100.0f, 0.05f, 0.2f, 30.0f, FIN_OPT_CALL);

    EXPECT_FALSE(std::isnan(price));
    EXPECT_FALSE(std::isinf(price));
    EXPECT_GT(price, 0.0f);
    EXPECT_LT(price, 100.0f);  // Call can't be worth more than stock
}

TEST_F(FinancialGPURegressionTest, SingleReturnVaR) {
    // Edge case: single return value
    std::vector<float> returns = {-0.05f};

    fin_risk_gpu_params_t params = fin_risk_gpu_params_default();
    params.num_returns = 1;

    fin_risk_gpu_result_t result = {0};
    bool ok = fin_risk_gpu_compute(ctx, returns.data(), &params, &result);
    ASSERT_TRUE(ok);

    // With single value, VaR should be that value
    EXPECT_NEAR(result.var_95, 0.05f, 0.01f);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
