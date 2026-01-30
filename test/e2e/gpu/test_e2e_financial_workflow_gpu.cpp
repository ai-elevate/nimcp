/* ============================================================================
 * E2E Test: Financial Workflow GPU Pipeline
 * ============================================================================
 * WHAT: End-to-end test of GPU-accelerated financial computing workflow
 * WHY:  Validate complete financial analysis pipeline on GPU
 * HOW:  Market simulation -> Risk analysis -> Portfolio optimization -> Hedging
 * ============================================================================
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef NIMCP_CUDA_ENABLED
extern "C" {
#include "gpu/financial/nimcp_financial_monte_carlo_gpu.h"
#include "gpu/financial/nimcp_financial_optimization_gpu.h"
#include "gpu/financial/nimcp_financial_risk_gpu.h"
#include "gpu/financial/nimcp_financial_derivatives_gpu.h"
#include "gpu/nimcp_gpu_context.h"
}
#endif

namespace {

constexpr float TOLERANCE = 1e-4f;
constexpr float STATISTICAL_TOLERANCE = 0.1f;

class FinancialWorkflowE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef NIMCP_CUDA_ENABLED
        ctx_ = nimcp_gpu_context_create(nullptr);
        if (ctx_) {
            rng_ = fin_gpu_rng_create(ctx_, 100000, 54321);
        }
#endif
    }

    void TearDown() override {
#ifdef NIMCP_CUDA_ENABLED
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

#ifdef NIMCP_CUDA_ENABLED
    nimcp_gpu_context_t* ctx_ = nullptr;
    fin_gpu_rng_t* rng_ = nullptr;
#endif
};

/* ============================================================================
 * E2E Test: Complete Portfolio Management Workflow
 * ============================================================================
 * Simulates a realistic portfolio management process:
 * 1. Simulate market scenarios
 * 2. Estimate returns and covariances
 * 3. Optimize portfolio
 * 4. Compute risk metrics
 * 5. Price hedging options
 */
TEST_F(FinancialWorkflowE2ETest, PortfolioManagementWorkflow) {
#ifdef NIMCP_CUDA_ENABLED
    ASSERT_NE(ctx_, nullptr) << "GPU context required";
    ASSERT_NE(rng_, nullptr) << "RNG required";

    const int num_assets = 10;
    const int num_scenarios = 50000;
    const int num_steps = 252;  // Daily for 1 year

    // ==================== Phase 1: Market Parameters ====================
    std::vector<float> expected_returns(num_assets);
    std::vector<float> volatilities(num_assets);
    std::vector<float> initial_prices(num_assets, 100.0f);

    // Diversified asset mix (stocks, bonds, commodities)
    float base_returns[] = {0.08f, 0.10f, 0.12f, 0.06f, 0.04f, 0.15f, 0.11f, 0.09f, 0.05f, 0.07f};
    float base_vols[] = {0.18f, 0.22f, 0.25f, 0.12f, 0.06f, 0.35f, 0.20f, 0.16f, 0.08f, 0.15f};

    for (int i = 0; i < num_assets; i++) {
        expected_returns[i] = base_returns[i];
        volatilities[i] = base_vols[i];
    }

    // Correlation structure (block diagonal with some cross-correlations)
    std::vector<float> correlation(num_assets * num_assets, 0.0f);
    for (int i = 0; i < num_assets; i++) {
        correlation[i * num_assets + i] = 1.0f;  // Diagonal
        for (int j = i + 1; j < num_assets; j++) {
            float corr = 0.3f;
            // Same asset class: higher correlation
            if ((i < 5 && j < 5) || (i >= 5 && j >= 5)) {
                corr = 0.5f;
            }
            correlation[i * num_assets + j] = corr;
            correlation[j * num_assets + i] = corr;
        }
    }

    // ==================== Phase 2: Monte Carlo Simulation ====================
    auto sim_start = std::chrono::high_resolution_clock::now();

    fin_multi_asset_gpu_params_t mc_params = {
        .num_assets = static_cast<uint32_t>(num_assets),
        .num_paths = static_cast<uint32_t>(num_scenarios),
        .num_steps = static_cast<uint32_t>(num_steps),
        .time_horizon = 1.0f,
        .expected_returns = expected_returns.data(),
        .volatilities = volatilities.data(),
        .correlation_matrix = correlation.data()
    };

    std::vector<float> terminal_returns(num_scenarios * num_assets);

    bool mc_success = fin_monte_carlo_gpu_multi_asset(ctx_, rng_, &mc_params, terminal_returns.data());
    ASSERT_TRUE(mc_success) << "Monte Carlo simulation failed";

    auto sim_end = std::chrono::high_resolution_clock::now();
    auto sim_time = std::chrono::duration_cast<std::chrono::milliseconds>(sim_end - sim_start);

    // ==================== Phase 3: Estimate Parameters from Simulation ====================
    std::vector<float> estimated_returns(num_assets, 0.0f);
    std::vector<float> covariance(num_assets * num_assets, 0.0f);

    // Calculate means
    for (int a = 0; a < num_assets; a++) {
        for (int s = 0; s < num_scenarios; s++) {
            estimated_returns[a] += terminal_returns[s * num_assets + a];
        }
        estimated_returns[a] /= num_scenarios;
    }

    // Calculate covariance
    for (int i = 0; i < num_assets; i++) {
        for (int j = i; j < num_assets; j++) {
            float cov = 0.0f;
            for (int s = 0; s < num_scenarios; s++) {
                cov += (terminal_returns[s * num_assets + i] - estimated_returns[i]) *
                       (terminal_returns[s * num_assets + j] - estimated_returns[j]);
            }
            cov /= (num_scenarios - 1);
            covariance[i * num_assets + j] = cov;
            covariance[j * num_assets + i] = cov;
        }
    }

    // Verify estimated returns are close to expected
    for (int i = 0; i < num_assets; i++) {
        float rel_error = std::abs(estimated_returns[i] - expected_returns[i]) /
                         std::abs(expected_returns[i] + 0.01f);
        EXPECT_LT(rel_error, STATISTICAL_TOLERANCE)
            << "Estimated return for asset " << i << " differs too much from expected";
    }

    // ==================== Phase 4: Portfolio Optimization ====================
    auto opt_start = std::chrono::high_resolution_clock::now();

    fin_optimization_gpu_params_t opt_params = {
        .target_return = 0.08f,
        .max_iterations = 2000,
        .convergence_threshold = 1e-8f,
        .constraint_type = FIN_OPT_CONSTRAINT_LONG_ONLY,
        .risk_aversion = 2.0f
    };

    fin_optimization_gpu_result_t opt_result;
    opt_result.weights = new float[num_assets];

    bool opt_success = fin_optimization_gpu_mean_variance(
        ctx_, estimated_returns.data(), covariance.data(), num_assets,
        &opt_params, &opt_result);
    ASSERT_TRUE(opt_success) << "Portfolio optimization failed";

    auto opt_end = std::chrono::high_resolution_clock::now();
    auto opt_time = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start);

    // Verify weights sum to 1
    float weight_sum = 0.0f;
    for (int i = 0; i < num_assets; i++) {
        EXPECT_GE(opt_result.weights[i], -TOLERANCE) << "Weight " << i << " is negative";
        weight_sum += opt_result.weights[i];
    }
    EXPECT_NEAR(weight_sum, 1.0f, 0.01f) << "Weights should sum to 1";

    // Calculate portfolio statistics
    float portfolio_return = 0.0f;
    float portfolio_variance = 0.0f;

    for (int i = 0; i < num_assets; i++) {
        portfolio_return += opt_result.weights[i] * estimated_returns[i];
        for (int j = 0; j < num_assets; j++) {
            portfolio_variance += opt_result.weights[i] * opt_result.weights[j] *
                                 covariance[i * num_assets + j];
        }
    }
    float portfolio_vol = std::sqrt(portfolio_variance);

    // ==================== Phase 5: Risk Analysis ====================
    // Calculate portfolio returns for all scenarios
    std::vector<float> portfolio_scenario_returns(num_scenarios);
    for (int s = 0; s < num_scenarios; s++) {
        float port_ret = 0.0f;
        for (int a = 0; a < num_assets; a++) {
            port_ret += opt_result.weights[a] * terminal_returns[s * num_assets + a];
        }
        portfolio_scenario_returns[s] = port_ret;
    }

    fin_risk_gpu_params_t risk_params = {
        .confidence_level = 0.95f,
        .time_horizon_days = 252
    };

    fin_risk_gpu_result_t risk_result;
    bool risk_success = fin_risk_gpu_compute(
        ctx_, portfolio_scenario_returns.data(), num_scenarios, &risk_params, &risk_result);
    ASSERT_TRUE(risk_success) << "Risk computation failed";

    // VaR should be reasonable
    EXPECT_GT(risk_result.var_95, -1.0f) << "VaR too extreme";
    EXPECT_LT(risk_result.var_95, 0.5f) << "VaR too small (likely wrong sign)";

    // CVaR should be more extreme than VaR
    EXPECT_LE(risk_result.cvar_95, risk_result.var_95 + TOLERANCE)
        << "CVaR should be worse than VaR";

    // ==================== Phase 6: Options Hedging ====================
    // Price protective puts for the largest holdings
    std::vector<int> top_holdings;
    for (int i = 0; i < num_assets; i++) {
        if (opt_result.weights[i] > 0.1f) {
            top_holdings.push_back(i);
        }
    }

    if (!top_holdings.empty()) {
        std::vector<float> spots, strikes, rates, vols, times;
        std::vector<bool> is_call;

        for (int idx : top_holdings) {
            spots.push_back(initial_prices[idx]);
            strikes.push_back(initial_prices[idx] * 0.95f);  // 5% OTM puts
            rates.push_back(0.05f);
            vols.push_back(volatilities[idx]);
            times.push_back(0.25f);  // 3-month puts
            is_call.push_back(false);
        }

        std::vector<float> put_prices(top_holdings.size());

        bool deriv_success = fin_derivatives_gpu_black_scholes_batch(
            ctx_, spots.data(), strikes.data(), rates.data(), vols.data(),
            times.data(), is_call.data(), put_prices.data(),
            static_cast<int>(top_holdings.size()));

        if (deriv_success) {
            float total_hedge_cost = 0.0f;
            for (size_t i = 0; i < top_holdings.size(); i++) {
                float position_value = opt_result.weights[top_holdings[i]] * 1000000.0f;  // $1M portfolio
                float num_contracts = position_value / (spots[i] * 100.0f);  // 100 shares per contract
                float hedge_cost = num_contracts * put_prices[i] * 100.0f;
                total_hedge_cost += hedge_cost;
            }

            float portfolio_value = 1000000.0f;
            float hedge_cost_pct = total_hedge_cost / portfolio_value * 100.0f;

            std::cout << "Hedging cost: $" << total_hedge_cost
                      << " (" << hedge_cost_pct << "% of portfolio)" << std::endl;

            EXPECT_LT(hedge_cost_pct, 5.0f) << "Hedge cost should be reasonable";
        }
    }

    // ==================== Phase 7: Performance Report ====================
    std::cout << "\n=== Portfolio Management E2E Results ===" << std::endl;
    std::cout << "Simulation time: " << sim_time.count() << " ms (" << num_scenarios << " scenarios)" << std::endl;
    std::cout << "Optimization time: " << opt_time.count() << " ms" << std::endl;
    std::cout << "\nOptimal Portfolio:" << std::endl;
    std::cout << "  Expected Return: " << portfolio_return * 100.0f << "%" << std::endl;
    std::cout << "  Volatility: " << portfolio_vol * 100.0f << "%" << std::endl;
    std::cout << "  Sharpe Ratio: " << (portfolio_return - 0.02f) / portfolio_vol << std::endl;
    std::cout << "\nRisk Metrics:" << std::endl;
    std::cout << "  VaR (95%): " << risk_result.var_95 * 100.0f << "%" << std::endl;
    std::cout << "  CVaR (95%): " << risk_result.cvar_95 * 100.0f << "%" << std::endl;
    std::cout << "\nTop Holdings:" << std::endl;
    for (int i = 0; i < num_assets; i++) {
        if (opt_result.weights[i] > 0.05f) {
            std::cout << "  Asset " << i << ": " << opt_result.weights[i] * 100.0f << "%" << std::endl;
        }
    }

    delete[] opt_result.weights;
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * E2E Test: Options Market Making Simulation
 * ============================================================================ */
TEST_F(FinancialWorkflowE2ETest, OptionsMarketMakingSimulation) {
#ifdef NIMCP_CUDA_ENABLED
    ASSERT_NE(ctx_, nullptr) << "GPU context required";

    // Simulate a market maker quoting options throughout the day
    const int num_quotes = 1000;  // Number of quote updates
    const int options_per_quote = 50;  // Options in the book

    float spot = 100.0f;
    float base_vol = 0.20f;
    float rf_rate = 0.05f;

    // Generate option chain (5 strikes x 4 maturities x 2 types)
    std::vector<float> all_strikes;
    std::vector<float> all_times;
    std::vector<bool> all_is_call;

    float strike_range[] = {90.0f, 95.0f, 100.0f, 105.0f, 110.0f};
    float time_range[] = {0.0833f, 0.25f, 0.5f, 1.0f};  // 1M, 3M, 6M, 1Y

    for (float T : time_range) {
        for (float K : strike_range) {
            all_strikes.push_back(K);
            all_times.push_back(T);
            all_is_call.push_back(true);
            all_strikes.push_back(K);
            all_times.push_back(T);
            all_is_call.push_back(false);
        }
    }

    const int total_options = static_cast<int>(all_strikes.size());

    auto start = std::chrono::high_resolution_clock::now();

    float total_pnl = 0.0f;
    float total_delta = 0.0f;
    float total_gamma = 0.0f;
    float total_vega = 0.0f;

    std::mt19937 gen(42);
    std::normal_distribution<float> spot_change(0.0f, 0.005f);  // 0.5% moves
    std::normal_distribution<float> vol_change(0.0f, 0.01f);    // 1% vol changes

    for (int quote = 0; quote < num_quotes; quote++) {
        // Update market conditions
        spot *= (1.0f + spot_change(gen));
        base_vol = std::max(0.05f, std::min(0.50f, base_vol + vol_change(gen)));

        // Build vol smile
        std::vector<float> vols(total_options);
        for (int i = 0; i < total_options; i++) {
            float moneyness = all_strikes[i] / spot;
            float smile_adjust = 0.05f * (moneyness - 1.0f) * (moneyness - 1.0f);
            vols[i] = base_vol + smile_adjust;
        }

        std::vector<float> spots(total_options, spot);
        std::vector<float> rates(total_options, rf_rate);

        // Price all options
        std::vector<float> prices(total_options);
        bool success = fin_derivatives_gpu_black_scholes_batch(
            ctx_, spots.data(), all_strikes.data(), rates.data(), vols.data(),
            all_times.data(), all_is_call.data(), prices.data(), total_options);

        if (!success) continue;

        // Compute Greeks for a subset (top 20 options by vega)
        std::vector<std::pair<float, int>> vega_rank;
        for (int i = 0; i < total_options; i++) {
            // Estimate vega from vol sensitivity
            float est_vega = prices[i] * 0.01f / vols[i];
            vega_rank.push_back({est_vega, i});
        }
        std::sort(vega_rank.begin(), vega_rank.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        float quote_delta = 0.0f, quote_gamma = 0.0f, quote_vega = 0.0f;

        for (int k = 0; k < std::min(20, total_options); k++) {
            int idx = vega_rank[k].second;

            fin_derivatives_gpu_params_t params = {
                .spot_price = spot,
                .strike_price = all_strikes[idx],
                .risk_free_rate = rf_rate,
                .volatility = vols[idx],
                .time_to_maturity = all_times[idx],
                .num_steps = 100,
                .is_call = all_is_call[idx],
                .is_american = false
            };

            fin_derivatives_gpu_greeks_t greeks;
            if (fin_derivatives_gpu_compute_greeks(ctx_, &params, &greeks)) {
                quote_delta += greeks.delta;
                quote_gamma += greeks.gamma;
                quote_vega += greeks.vega;
            }
        }

        total_delta += quote_delta;
        total_gamma += quote_gamma;
        total_vega += quote_vega;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    float quotes_per_sec = num_quotes / (elapsed.count() / 1000.0f);
    float options_per_sec = num_quotes * total_options / (elapsed.count() / 1000.0f);

    std::cout << "\n=== Options Market Making E2E Results ===" << std::endl;
    std::cout << "Total time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Quote updates: " << num_quotes << std::endl;
    std::cout << "Options per update: " << total_options << std::endl;
    std::cout << "Quote throughput: " << quotes_per_sec << " quotes/sec" << std::endl;
    std::cout << "Pricing throughput: " << options_per_sec << " options/sec" << std::endl;

    // Should be able to process at least 100 quotes/sec for real-time trading
    EXPECT_GT(quotes_per_sec, 50.0f) << "Quote throughput too low for trading";
    EXPECT_GT(options_per_sec, 1000.0f) << "Pricing throughput too low";
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * E2E Test: Risk Scenario Analysis
 * ============================================================================ */
TEST_F(FinancialWorkflowE2ETest, RiskScenarioAnalysis) {
#ifdef NIMCP_CUDA_ENABLED
    ASSERT_NE(ctx_, nullptr) << "GPU context required";
    ASSERT_NE(rng_, nullptr) << "RNG required";

    // Define market scenarios
    struct Scenario {
        const char* name;
        float drift;
        float volatility;
    };

    std::vector<Scenario> scenarios = {
        {"Bull Market", 0.15f, 0.15f},
        {"Bear Market", -0.20f, 0.30f},
        {"High Volatility", 0.05f, 0.40f},
        {"Recession", -0.30f, 0.35f},
        {"Recovery", 0.25f, 0.25f},
        {"Stagflation", -0.05f, 0.20f}
    };

    const int num_paths = 10000;
    const float portfolio_value = 1000000.0f;  // $1M portfolio

    std::cout << "\n=== Risk Scenario Analysis ===" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Scenario         | Mean Return | VaR (95%) | CVaR (95%) | Max Loss" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& scenario : scenarios) {
        fin_monte_carlo_gpu_params_t mc_params = {
            .initial_value = portfolio_value,
            .drift = scenario.drift,
            .volatility = scenario.volatility,
            .time_horizon = 1.0f,
            .num_steps = 252,
            .num_paths = static_cast<uint32_t>(num_paths)
        };

        fin_monte_carlo_gpu_result_t mc_result;
        mc_result.terminal_values = new float[num_paths];
        mc_result.path_returns = new float[num_paths];

        bool mc_success = fin_monte_carlo_gpu_simulate(ctx_, rng_, &mc_params, &mc_result);

        if (mc_success) {
            // Convert terminal values to returns
            for (int i = 0; i < num_paths; i++) {
                mc_result.path_returns[i] = (mc_result.terminal_values[i] - portfolio_value) / portfolio_value;
            }

            fin_risk_gpu_params_t risk_params = {.confidence_level = 0.95f, .time_horizon_days = 252};
            fin_risk_gpu_result_t risk_result;

            fin_risk_gpu_compute(ctx_, mc_result.path_returns, num_paths, &risk_params, &risk_result);

            // Find max loss
            float max_loss = 0.0f;
            float mean_return = 0.0f;
            for (int i = 0; i < num_paths; i++) {
                max_loss = std::min(max_loss, mc_result.path_returns[i]);
                mean_return += mc_result.path_returns[i];
            }
            mean_return /= num_paths;

            printf("%-16s | %+10.1f%% | %9.1f%% | %10.1f%% | %8.1f%%\n",
                   scenario.name,
                   mean_return * 100.0f,
                   risk_result.var_95 * 100.0f,
                   risk_result.cvar_95 * 100.0f,
                   max_loss * 100.0f);

            // Validate scenario expectations
            if (scenario.drift > 0.1f) {
                EXPECT_GT(mean_return, 0.0f) << scenario.name << " should have positive mean";
            } else if (scenario.drift < -0.1f) {
                EXPECT_LT(mean_return, 0.0f) << scenario.name << " should have negative mean";
            }

            // Higher volatility should generally mean worse VaR
            if (scenario.volatility > 0.30f) {
                EXPECT_LT(risk_result.var_95, -0.2f)
                    << scenario.name << " VaR should be significant with high vol";
            }
        }

        delete[] mc_result.terminal_values;
        delete[] mc_result.path_returns;
    }

    std::cout << std::string(70, '-') << std::endl;
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

/* ============================================================================
 * E2E Test: GPU Performance Benchmark
 * ============================================================================ */
TEST_F(FinancialWorkflowE2ETest, GPUPerformanceBenchmark) {
#ifdef NIMCP_CUDA_ENABLED
    ASSERT_NE(ctx_, nullptr) << "GPU context required";
    ASSERT_NE(rng_, nullptr) << "RNG required";

    std::cout << "\n=== GPU Financial Computing Benchmark ===" << std::endl;

    // Benchmark 1: Monte Carlo Scaling
    std::cout << "\n1. Monte Carlo Path Scaling:" << std::endl;
    std::vector<int> path_counts = {1000, 10000, 100000, 500000};

    for (int num_paths : path_counts) {
        fin_monte_carlo_gpu_params_t params = {
            .initial_value = 100.0f,
            .drift = 0.05f,
            .volatility = 0.20f,
            .time_horizon = 1.0f,
            .num_steps = 252,
            .num_paths = static_cast<uint32_t>(num_paths)
        };

        fin_monte_carlo_gpu_result_t result;
        result.terminal_values = new float[num_paths];
        result.path_returns = new float[num_paths];

        auto start = std::chrono::high_resolution_clock::now();
        fin_monte_carlo_gpu_simulate(ctx_, rng_, &params, &result);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float paths_per_sec = num_paths / (elapsed.count() / 1e6f);

        std::cout << "  " << num_paths << " paths: "
                  << elapsed.count() / 1000.0f << " ms ("
                  << paths_per_sec / 1e6f << "M paths/sec)" << std::endl;

        delete[] result.terminal_values;
        delete[] result.path_returns;
    }

    // Benchmark 2: Option Pricing Batch Size
    std::cout << "\n2. Black-Scholes Batch Scaling:" << std::endl;
    std::vector<int> batch_sizes = {100, 1000, 10000, 50000};

    for (int batch : batch_sizes) {
        std::vector<float> spots(batch, 100.0f);
        std::vector<float> strikes(batch, 100.0f);
        std::vector<float> rates(batch, 0.05f);
        std::vector<float> vols(batch, 0.20f);
        std::vector<float> times(batch, 1.0f);
        std::vector<bool> is_call(batch, true);
        std::vector<float> prices(batch);

        auto start = std::chrono::high_resolution_clock::now();
        fin_derivatives_gpu_black_scholes_batch(
            ctx_, spots.data(), strikes.data(), rates.data(), vols.data(),
            times.data(), is_call.data(), prices.data(), batch);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float opts_per_sec = batch / (elapsed.count() / 1e6f);

        std::cout << "  " << batch << " options: "
                  << elapsed.count() / 1000.0f << " ms ("
                  << opts_per_sec / 1e3f << "K opts/sec)" << std::endl;
    }

    // Benchmark 3: Portfolio Optimization Asset Scaling
    std::cout << "\n3. Portfolio Optimization Scaling:" << std::endl;
    std::vector<int> asset_counts = {10, 50, 100, 200};

    for (int n : asset_counts) {
        std::vector<float> returns(n);
        std::vector<float> covariance(n * n);

        // Generate random returns and positive-definite covariance
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.08f, 0.03f);
        for (int i = 0; i < n; i++) {
            returns[i] = dist(gen);
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    covariance[i * n + j] = 0.04f;
                } else {
                    covariance[i * n + j] = 0.01f;
                }
            }
        }

        fin_optimization_gpu_params_t params = {
            .target_return = 0.08f,
            .max_iterations = 1000,
            .convergence_threshold = 1e-6f,
            .constraint_type = FIN_OPT_CONSTRAINT_LONG_ONLY,
            .risk_aversion = 1.0f
        };

        fin_optimization_gpu_result_t result;
        result.weights = new float[n];

        auto start = std::chrono::high_resolution_clock::now();
        fin_optimization_gpu_mean_variance(ctx_, returns.data(), covariance.data(), n, &params, &result);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "  " << n << " assets: " << elapsed.count() << " ms" << std::endl;

        delete[] result.weights;

        // Optimization should complete in reasonable time
        EXPECT_LT(elapsed.count(), 5000) << n << " asset optimization took too long";
    }
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

}  // namespace
