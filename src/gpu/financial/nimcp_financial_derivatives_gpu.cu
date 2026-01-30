/**
 * @file nimcp_financial_derivatives_gpu.cu
 * @brief GPU Derivatives Pricing
 *
 * WHAT: GPU-accelerated option pricing and Greeks computation
 * WHY:  Fast binomial trees, batch pricing, implied volatility
 * HOW:  Level-parallel binomial tree, batch Black-Scholes on GPU
 *
 * Implements:
 *   - Binomial tree (CRR) for American/European options
 *   - Black-Scholes batch pricing
 *   - Greeks computation (Delta, Gamma, Theta, Vega, Rho)
 *   - Implied volatility (Newton-Raphson)
 *   - Option chain pricing
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <float.h>

#include "gpu/financial/nimcp_financial_gpu.h"
#include "gpu/financial/nimcp_financial_derivatives_gpu.h"
#include "gpu/common/nimcp_cuda_utils.h"
#include "gpu/recovery/nimcp_gpu_recovery.h"
#include "utils/exception/nimcp_exception_macros.h"

//=============================================================================
// Thread-Local Error Storage
//=============================================================================

static __thread char g_deriv_error[256] = {0};

static void set_deriv_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_deriv_error, sizeof(g_deriv_error), fmt, args);
    va_end(args);
}

//=============================================================================
// Device Constants
//=============================================================================

#define INV_SQRT_2PI 0.3989422804014327f

//=============================================================================
// Device Helper Functions
//=============================================================================

/**
 * @brief Standard normal PDF
 */
__device__ __forceinline__ float norm_pdf_device(float x) {
    return INV_SQRT_2PI * expf(-0.5f * x * x);
}

/**
 * @brief Standard normal CDF (Abramowitz and Stegun approximation)
 */
__device__ __forceinline__ float norm_cdf_device(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;

    float sign = 1.0f;
    if (x < 0.0f) {
        sign = -1.0f;
        x = -x;
    }

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * expf(-x*x/2.0f);

    return 0.5f * (1.0f + sign * y);
}

/**
 * @brief Black-Scholes d1 parameter
 */
__device__ __forceinline__ float bs_d1(
    float S, float K, float r, float q, float sigma, float T)
{
    return (logf(S / K) + (r - q + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
}

/**
 * @brief Black-Scholes d2 parameter
 */
__device__ __forceinline__ float bs_d2(
    float d1, float sigma, float T)
{
    return d1 - sigma * sqrtf(T);
}

/**
 * @brief Black-Scholes call price
 */
__device__ float bs_call_device(
    float S, float K, float r, float q, float sigma, float T)
{
    if (T <= 0.0f) {
        return fmaxf(S - K, 0.0f);
    }

    float d1 = bs_d1(S, K, r, q, sigma, T);
    float d2 = bs_d2(d1, sigma, T);

    return S * expf(-q * T) * norm_cdf_device(d1) -
           K * expf(-r * T) * norm_cdf_device(d2);
}

/**
 * @brief Black-Scholes put price
 */
__device__ float bs_put_device(
    float S, float K, float r, float q, float sigma, float T)
{
    if (T <= 0.0f) {
        return fmaxf(K - S, 0.0f);
    }

    float d1 = bs_d1(S, K, r, q, sigma, T);
    float d2 = bs_d2(d1, sigma, T);

    return K * expf(-r * T) * norm_cdf_device(-d2) -
           S * expf(-q * T) * norm_cdf_device(-d1);
}

/**
 * @brief Black-Scholes vega (same for call and put)
 */
__device__ float bs_vega_device(
    float S, float K, float r, float q, float sigma, float T)
{
    if (T <= 0.0f) return 0.0f;

    float d1 = bs_d1(S, K, r, q, sigma, T);
    return S * expf(-q * T) * norm_pdf_device(d1) * sqrtf(T);
}

//=============================================================================
// Binomial Tree Kernels
//=============================================================================

/**
 * @brief Initialize terminal payoffs for binomial tree
 */
__global__ void kernel_binomial_init_payoffs(
    float* __restrict__ values,
    float S0,
    float u,
    float d,
    float K,
    bool is_call,
    uint32_t num_steps)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_steps) return;

    // Price at node (num_steps, i) = S0 * u^(num_steps - i) * d^i
    float S = S0 * powf(u, (float)(num_steps - i)) * powf(d, (float)i);

    if (is_call) {
        values[i] = fmaxf(S - K, 0.0f);
    } else {
        values[i] = fmaxf(K - S, 0.0f);
    }
}

/**
 * @brief Backward induction step for binomial tree
 *
 * European option: just discounted expectation
 */
__global__ void kernel_binomial_european_step(
    float* __restrict__ values,
    const float* __restrict__ prev_values,
    float p,
    float discount,
    uint32_t level)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > level) return;

    values[i] = discount * (p * prev_values[i] + (1.0f - p) * prev_values[i + 1]);
}

/**
 * @brief Backward induction step for American option
 *
 * Compare hold value with exercise value
 */
__global__ void kernel_binomial_american_step(
    float* __restrict__ values,
    const float* __restrict__ prev_values,
    float S0,
    float u,
    float d,
    float K,
    float p,
    float discount,
    bool is_call,
    uint32_t level,
    uint32_t num_steps)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > level) return;

    // Hold value
    float hold = discount * (p * prev_values[i] + (1.0f - p) * prev_values[i + 1]);

    // Current stock price at this node
    float S = S0 * powf(u, (float)(level - i)) * powf(d, (float)i);

    // Exercise value
    float exercise;
    if (is_call) {
        exercise = fmaxf(S - K, 0.0f);
    } else {
        exercise = fmaxf(K - S, 0.0f);
    }

    // American option: max of hold and exercise
    values[i] = fmaxf(hold, exercise);
}

//=============================================================================
// Black-Scholes Batch Kernels
//=============================================================================

/**
 * @brief Batch Black-Scholes pricing
 */
__global__ void kernel_black_scholes_batch(
    const float* __restrict__ spots,
    const float* __restrict__ strikes,
    const float* __restrict__ rates,
    const float* __restrict__ vols,
    const float* __restrict__ times,
    const int* __restrict__ types,      // 0 = call, 1 = put
    float* __restrict__ prices,
    uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float S = spots[i];
    float K = strikes[i];
    float r = rates[i];
    float sigma = vols[i];
    float T = times[i];
    bool is_call = (types[i] == 0);

    if (is_call) {
        prices[i] = bs_call_device(S, K, r, 0.0f, sigma, T);
    } else {
        prices[i] = bs_put_device(S, K, r, 0.0f, sigma, T);
    }
}

/**
 * @brief Batch Greeks computation
 */
__global__ void kernel_greeks_batch(
    const float* __restrict__ spots,
    const float* __restrict__ strikes,
    const float* __restrict__ rates,
    const float* __restrict__ vols,
    const float* __restrict__ times,
    const int* __restrict__ types,
    float* __restrict__ deltas,
    float* __restrict__ gammas,
    float* __restrict__ thetas,
    float* __restrict__ vegas,
    float* __restrict__ rhos,
    uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float S = spots[i];
    float K = strikes[i];
    float r = rates[i];
    float sigma = vols[i];
    float T = times[i];
    bool is_call = (types[i] == 0);

    if (T <= 0.0f) {
        // At expiry
        if (deltas) deltas[i] = is_call ? (S > K ? 1.0f : 0.0f) : (S < K ? -1.0f : 0.0f);
        if (gammas) gammas[i] = 0.0f;
        if (thetas) thetas[i] = 0.0f;
        if (vegas) vegas[i] = 0.0f;
        if (rhos) rhos[i] = 0.0f;
        return;
    }

    float sqrt_T = sqrtf(T);
    float d1 = bs_d1(S, K, r, 0.0f, sigma, T);
    float d2 = bs_d2(d1, sigma, T);
    float nd1 = norm_cdf_device(d1);
    float nd2 = norm_cdf_device(d2);
    float npd1 = norm_pdf_device(d1);

    // Delta
    if (deltas) {
        deltas[i] = is_call ? nd1 : (nd1 - 1.0f);
    }

    // Gamma (same for call and put)
    if (gammas) {
        gammas[i] = npd1 / (S * sigma * sqrt_T);
    }

    // Theta
    if (thetas) {
        float term1 = -(S * npd1 * sigma) / (2.0f * sqrt_T);
        if (is_call) {
            thetas[i] = term1 - r * K * expf(-r * T) * nd2;
        } else {
            thetas[i] = term1 + r * K * expf(-r * T) * norm_cdf_device(-d2);
        }
        thetas[i] /= 365.0f;  // Per day
    }

    // Vega
    if (vegas) {
        vegas[i] = S * npd1 * sqrt_T / 100.0f;  // Per 1% vol change
    }

    // Rho
    if (rhos) {
        if (is_call) {
            rhos[i] = K * T * expf(-r * T) * nd2 / 100.0f;
        } else {
            rhos[i] = -K * T * expf(-r * T) * norm_cdf_device(-d2) / 100.0f;
        }
    }
}

//=============================================================================
// Implied Volatility Kernels
//=============================================================================

/**
 * @brief Newton-Raphson implied volatility
 */
__global__ void kernel_implied_vol_newton(
    const float* __restrict__ market_prices,
    const float* __restrict__ spots,
    const float* __restrict__ strikes,
    const float* __restrict__ rates,
    const float* __restrict__ times,
    const int* __restrict__ types,
    float* __restrict__ implied_vols,
    float initial_guess,
    float tolerance,
    uint32_t max_iterations,
    uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float target = market_prices[i];
    float S = spots[i];
    float K = strikes[i];
    float r = rates[i];
    float T = times[i];
    bool is_call = (types[i] == 0);

    float sigma = initial_guess;

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        float price;
        if (is_call) {
            price = bs_call_device(S, K, r, 0.0f, sigma, T);
        } else {
            price = bs_put_device(S, K, r, 0.0f, sigma, T);
        }

        float diff = price - target;

        if (fabsf(diff) < tolerance) {
            break;
        }

        float vega = bs_vega_device(S, K, r, 0.0f, sigma, T);

        if (vega < 1e-10f) {
            // Vega too small, can't converge
            sigma = NAN;
            break;
        }

        sigma -= diff / vega;

        // Bounds check
        if (sigma <= 0.001f) sigma = 0.001f;
        if (sigma > 5.0f) sigma = 5.0f;
    }

    implied_vols[i] = sigma;
}

//=============================================================================
// Host API Implementation
//=============================================================================

extern "C" {

bool fin_derivatives_gpu_binomial_tree(
    nimcp_gpu_context_t* ctx,
    const fin_derivatives_gpu_params_t* params,
    fin_derivatives_gpu_result_t* result)
{
    // Initialize GPU recovery if not already done
    if (!nimcp_gpu_recovery_is_initialized()) {
        nimcp_gpu_recovery_init(NULL);
    }

    if (!ctx || !nimcp_gpu_context_is_valid(ctx)) {
        set_deriv_error("Invalid GPU context");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid GPU context");
        return false;
    }
    if (!params || !result) {
        set_deriv_error("Invalid parameters");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid parameters");
        return false;
    }
    if (params->tree_steps == 0) {
        set_deriv_error("Zero tree steps");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Zero tree steps");
        return false;
    }

    uint32_t N = params->tree_steps;
    cudaStream_t stream = nimcp_gpu_get_compute_stream(ctx);
    uint32_t block_size = 256;

    // CRR parameters
    float dt = params->time_to_expiry / (float)N;
    float u = expf(params->volatility * sqrtf(dt));
    float d = 1.0f / u;
    float discount = expf(-params->risk_free_rate * dt);
    float p = (expf((params->risk_free_rate - params->dividend_yield) * dt) - d) / (u - d);

    bool is_call = (params->option_type == FIN_OPT_CALL);
    bool is_american = (params->option_style == FIN_OPT_STYLE_AMERICAN);

    // Allocate two buffers for ping-pong with recovery
    float* d_values_a = NULL;
    float* d_values_b = NULL;

    cudaError_t err;

    err = cudaMalloc(&d_values_a, (N + 1) * sizeof(float));
    if (err != cudaSuccess) {
        nimcp_gpu_recovery_result_t recovery_result = {0};
        if (nimcp_gpu_try_recover(NULL, GPU_ERROR_OUT_OF_MEMORY, err, &recovery_result)) {
            err = cudaMalloc(&d_values_a, (N + 1) * sizeof(float));
        }
        if (err != cudaSuccess) {
            set_deriv_error("Failed to allocate tree buffer A");
            NIMCP_THROW_GPU(NIMCP_ERROR_GPU, 0, err, "Failed to allocate tree buffer A");
            return false;
        }
    }

    err = cudaMalloc(&d_values_b, (N + 1) * sizeof(float));
    if (err != cudaSuccess) {
        nimcp_gpu_recovery_result_t recovery_result = {0};
        if (nimcp_gpu_try_recover(NULL, GPU_ERROR_OUT_OF_MEMORY, err, &recovery_result)) {
            err = cudaMalloc(&d_values_b, (N + 1) * sizeof(float));
        }
        if (err != cudaSuccess) {
            cudaFree(d_values_a);
            set_deriv_error("Failed to allocate tree buffer B");
            NIMCP_THROW_GPU(NIMCP_ERROR_GPU, 0, err, "Failed to allocate tree buffer B");
            return false;
        }
    }

    uint32_t grid_size = NIMCP_CUDA_GRID_SIZE(N + 1, block_size);

    // Initialize terminal payoffs
    kernel_binomial_init_payoffs<<<grid_size, block_size, 0, stream>>>(
        d_values_a, params->spot, u, d, params->strike, is_call, N);

    // Backward induction
    float* current = d_values_a;
    float* next = d_values_b;

    for (int level = N - 1; level >= 0; level--) {
        uint32_t level_grid = NIMCP_CUDA_GRID_SIZE(level + 1, block_size);

        if (is_american) {
            kernel_binomial_american_step<<<level_grid, block_size, 0, stream>>>(
                next, current, params->spot, u, d, params->strike,
                p, discount, is_call, level, N);
        } else {
            kernel_binomial_european_step<<<level_grid, block_size, 0, stream>>>(
                next, current, p, discount, level);
        }

        // Swap buffers
        float* tmp = current;
        current = next;
        next = tmp;
    }

    // Get result (root node)
    float h_price;
    cudaMemcpy(&h_price, current, sizeof(float), cudaMemcpyDeviceToHost);

    result->price = h_price;

    // Compute Greeks using finite difference on tree
    if (params->compute_greeks) {
        // Delta: dV/dS
        // Use values at first level to approximate
        float h_v0, h_v1;
        cudaMemcpy(&h_v0, &current[0], sizeof(float), cudaMemcpyDeviceToHost);

        // Re-run with slightly different spot to get delta
        float delta_spot = params->spot * 0.01f;
        float spot_up = params->spot + delta_spot;
        float spot_down = params->spot - delta_spot;

        // For simplicity, use Black-Scholes Greeks for now
        float sqrt_T = sqrtf(params->time_to_expiry);
        float d1 = (logf(params->spot / params->strike) +
                   (params->risk_free_rate - params->dividend_yield +
                    0.5f * params->volatility * params->volatility) * params->time_to_expiry) /
                   (params->volatility * sqrt_T);
        float d2 = d1 - params->volatility * sqrt_T;

        float nd1 = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
        float npd1 = INV_SQRT_2PI * expf(-0.5f * d1 * d1);

        result->delta = is_call ? nd1 : (nd1 - 1.0f);
        result->gamma = npd1 / (params->spot * params->volatility * sqrt_T);

        // Theta (simplified)
        float term1 = -(params->spot * npd1 * params->volatility) / (2.0f * sqrt_T);
        result->theta = term1 / 365.0f;

        // Vega
        result->vega = params->spot * npd1 * sqrt_T / 100.0f;

        // Rho
        float nd2 = 0.5f * (1.0f + erff(d2 / sqrtf(2.0f)));
        result->rho = is_call ?
            params->strike * params->time_to_expiry *
            expf(-params->risk_free_rate * params->time_to_expiry) * nd2 / 100.0f :
            -params->strike * params->time_to_expiry *
            expf(-params->risk_free_rate * params->time_to_expiry) * (1.0f - nd2) / 100.0f;
    }

    cudaFree(d_values_a);
    cudaFree(d_values_b);

    return true;
}

bool fin_derivatives_gpu_black_scholes_batch(
    nimcp_gpu_context_t* ctx,
    const fin_option_chain_t* chain,
    const float* volatilities,
    float* prices)
{
    // Initialize GPU recovery if not already done
    if (!nimcp_gpu_recovery_is_initialized()) {
        nimcp_gpu_recovery_init(NULL);
    }

    if (!ctx || !nimcp_gpu_context_is_valid(ctx)) {
        set_deriv_error("Invalid GPU context");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid GPU context");
        return false;
    }
    if (!chain || !volatilities || !prices) {
        set_deriv_error("Invalid parameters");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid parameters");
        return false;
    }
    if (chain->num_options == 0) {
        set_deriv_error("Empty option chain");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Empty option chain");
        return false;
    }

    uint32_t n = chain->num_options;
    cudaStream_t stream = nimcp_gpu_get_compute_stream(ctx);
    uint32_t block_size = 256;
    uint32_t grid_size = NIMCP_CUDA_GRID_SIZE(n, block_size);

    // Pre-declare host arrays to avoid goto initialization issues
    float* h_spots = NULL;
    float* h_rates = NULL;
    int* h_types = NULL;

    // Allocate device memory
    float* d_spots = NULL;
    float* d_strikes = NULL;
    float* d_rates = NULL;
    float* d_vols = NULL;
    float* d_times = NULL;
    int* d_types = NULL;
    float* d_prices = NULL;

    cudaError_t err;

    err = cudaMalloc(&d_spots, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_strikes, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_rates, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_vols, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_times, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_types, n * sizeof(int));
    if (err != cudaSuccess) goto cleanup_bs;
    err = cudaMalloc(&d_prices, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_bs;

    // Fill spot prices (all same)
    h_spots = (float*)malloc(n * sizeof(float));
    h_rates = (float*)malloc(n * sizeof(float));
    h_types = (int*)malloc(n * sizeof(int));

    for (uint32_t i = 0; i < n; i++) {
        h_spots[i] = chain->spot;
        h_rates[i] = chain->rate;
        h_types[i] = (chain->types[i] == FIN_OPT_CALL) ? 0 : 1;
    }

    cudaMemcpyAsync(d_spots, h_spots, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strikes, chain->strikes, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rates, h_rates, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vols, volatilities, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_times, chain->expiries, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_types, h_types, n * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Run kernel
    kernel_black_scholes_batch<<<grid_size, block_size, 0, stream>>>(
        d_spots, d_strikes, d_rates, d_vols, d_times, d_types, d_prices, n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);

    // Copy results
    cudaMemcpy(prices, d_prices, n * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_spots);
    free(h_rates);
    free(h_types);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_vols);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_prices);

    return true;

cleanup_bs:
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_vols);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_prices);
    set_deriv_error("Memory allocation failed");
    return false;
}

bool fin_derivatives_gpu_greeks_batch(
    nimcp_gpu_context_t* ctx,
    const fin_option_chain_t* chain,
    const float* volatilities,
    fin_option_chain_result_t* result)
{
    // Initialize GPU recovery if not already done
    if (!nimcp_gpu_recovery_is_initialized()) {
        nimcp_gpu_recovery_init(NULL);
    }

    if (!ctx || !nimcp_gpu_context_is_valid(ctx)) {
        set_deriv_error("Invalid GPU context");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid GPU context");
        return false;
    }
    if (!chain || !volatilities || !result) {
        set_deriv_error("Invalid parameters");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid parameters");
        return false;
    }

    uint32_t n = chain->num_options;
    cudaStream_t stream = nimcp_gpu_get_compute_stream(ctx);
    uint32_t block_size = 256;
    uint32_t grid_size = NIMCP_CUDA_GRID_SIZE(n, block_size);

    // Allocate result arrays if not already allocated
    if (!result->prices) result->prices = (float*)malloc(n * sizeof(float));
    if (!result->deltas) result->deltas = (float*)malloc(n * sizeof(float));
    if (!result->gammas) result->gammas = (float*)malloc(n * sizeof(float));
    if (!result->thetas) result->thetas = (float*)malloc(n * sizeof(float));
    if (!result->vegas) result->vegas = (float*)malloc(n * sizeof(float));

    result->num_options = n;

    // Pre-declare host arrays to avoid goto initialization issues
    float* h_spots = NULL;
    float* h_rates = NULL;
    int* h_types = NULL;

    // Device allocations
    float* d_spots = NULL;
    float* d_strikes = NULL;
    float* d_rates = NULL;
    float* d_vols = NULL;
    float* d_times = NULL;
    int* d_types = NULL;
    float* d_prices = NULL;
    float* d_deltas = NULL;
    float* d_gammas = NULL;
    float* d_thetas = NULL;
    float* d_vegas = NULL;
    float* d_rhos = NULL;

    cudaError_t err;

    err = cudaMalloc(&d_spots, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_strikes, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_rates, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_vols, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_times, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_types, n * sizeof(int));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_prices, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_deltas, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_gammas, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_thetas, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_vegas, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;
    err = cudaMalloc(&d_rhos, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_greeks;

    // Prepare input data
    h_spots = (float*)malloc(n * sizeof(float));
    h_rates = (float*)malloc(n * sizeof(float));
    h_types = (int*)malloc(n * sizeof(int));

    for (uint32_t i = 0; i < n; i++) {
        h_spots[i] = chain->spot;
        h_rates[i] = chain->rate;
        h_types[i] = (chain->types[i] == FIN_OPT_CALL) ? 0 : 1;
    }

    cudaMemcpyAsync(d_spots, h_spots, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strikes, chain->strikes, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rates, h_rates, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vols, volatilities, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_times, chain->expiries, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_types, h_types, n * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Prices
    kernel_black_scholes_batch<<<grid_size, block_size, 0, stream>>>(
        d_spots, d_strikes, d_rates, d_vols, d_times, d_types, d_prices, n);

    // Greeks
    kernel_greeks_batch<<<grid_size, block_size, 0, stream>>>(
        d_spots, d_strikes, d_rates, d_vols, d_times, d_types,
        d_deltas, d_gammas, d_thetas, d_vegas, d_rhos, n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);

    // Copy results back
    cudaMemcpy(result->prices, d_prices, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->deltas, d_deltas, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->gammas, d_gammas, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->thetas, d_thetas, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->vegas, d_vegas, n * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_spots);
    free(h_rates);
    free(h_types);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_vols);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_prices);
    cudaFree(d_deltas);
    cudaFree(d_gammas);
    cudaFree(d_thetas);
    cudaFree(d_vegas);
    cudaFree(d_rhos);

    return true;

cleanup_greeks:
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_vols);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_prices);
    cudaFree(d_deltas);
    cudaFree(d_gammas);
    cudaFree(d_thetas);
    cudaFree(d_vegas);
    cudaFree(d_rhos);
    set_deriv_error("Memory allocation failed");
    return false;
}

bool fin_derivatives_gpu_implied_vol_batch(
    nimcp_gpu_context_t* ctx,
    const float* market_prices,
    const fin_option_chain_t* chain,
    float* implied_vols)
{
    // Initialize GPU recovery if not already done
    if (!nimcp_gpu_recovery_is_initialized()) {
        nimcp_gpu_recovery_init(NULL);
    }

    if (!ctx || !nimcp_gpu_context_is_valid(ctx)) {
        set_deriv_error("Invalid GPU context");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid GPU context");
        return false;
    }
    if (!market_prices || !chain || !implied_vols) {
        set_deriv_error("Invalid parameters");
        NIMCP_THROW_GPU(NIMCP_ERROR_INVALID_PARAM, 0, 0, "Invalid parameters");
        return false;
    }

    uint32_t n = chain->num_options;
    cudaStream_t stream = nimcp_gpu_get_compute_stream(ctx);
    uint32_t block_size = 256;
    uint32_t grid_size = NIMCP_CUDA_GRID_SIZE(n, block_size);

    // Pre-declare host arrays to avoid goto initialization issues
    float* h_spots = NULL;
    float* h_rates = NULL;
    int* h_types = NULL;

    // Device allocations
    float* d_market = NULL;
    float* d_spots = NULL;
    float* d_strikes = NULL;
    float* d_rates = NULL;
    float* d_times = NULL;
    int* d_types = NULL;
    float* d_ivs = NULL;

    cudaError_t err;

    err = cudaMalloc(&d_market, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_spots, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_strikes, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_rates, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_times, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_types, n * sizeof(int));
    if (err != cudaSuccess) goto cleanup_iv;
    err = cudaMalloc(&d_ivs, n * sizeof(float));
    if (err != cudaSuccess) goto cleanup_iv;

    // Prepare inputs
    h_spots = (float*)malloc(n * sizeof(float));
    h_rates = (float*)malloc(n * sizeof(float));
    h_types = (int*)malloc(n * sizeof(int));

    for (uint32_t i = 0; i < n; i++) {
        h_spots[i] = chain->spot;
        h_rates[i] = chain->rate;
        h_types[i] = (chain->types[i] == FIN_OPT_CALL) ? 0 : 1;
    }

    cudaMemcpyAsync(d_market, market_prices, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_spots, h_spots, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strikes, chain->strikes, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rates, h_rates, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_times, chain->expiries, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_types, h_types, n * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Run Newton-Raphson
    kernel_implied_vol_newton<<<grid_size, block_size, 0, stream>>>(
        d_market, d_spots, d_strikes, d_rates, d_times, d_types, d_ivs,
        0.2f,    // Initial guess
        1e-6f,   // Tolerance
        100,     // Max iterations
        n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);

    cudaMemcpy(implied_vols, d_ivs, n * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_spots);
    free(h_rates);
    free(h_types);

    cudaFree(d_market);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_ivs);

    return true;

cleanup_iv:
    cudaFree(d_market);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_rates);
    cudaFree(d_times);
    cudaFree(d_types);
    cudaFree(d_ivs);
    set_deriv_error("Memory allocation failed");
    return false;
}

void fin_option_chain_result_free(fin_option_chain_result_t* result) {
    if (result) {
        free(result->prices);
        free(result->deltas);
        free(result->gammas);
        free(result->thetas);
        free(result->vegas);
        free(result->implied_vols);
        memset(result, 0, sizeof(*result));
    }
}

const char* fin_derivatives_gpu_get_last_error(void) {
    return g_deriv_error;
}

} // extern "C"

#else // !NIMCP_ENABLE_CUDA

//=============================================================================
// CPU Fallback Stubs
//=============================================================================

extern "C" {

bool fin_derivatives_gpu_binomial_tree(
    nimcp_gpu_context_t* ctx,
    const fin_derivatives_gpu_params_t* params,
    fin_derivatives_gpu_result_t* result)
{
    (void)ctx; (void)params; (void)result;
    return false;
}

bool fin_derivatives_gpu_black_scholes_batch(
    nimcp_gpu_context_t* ctx,
    const fin_option_chain_t* chain,
    const float* volatilities,
    float* prices)
{
    (void)ctx; (void)chain; (void)volatilities; (void)prices;
    return false;
}

bool fin_derivatives_gpu_greeks_batch(
    nimcp_gpu_context_t* ctx,
    const fin_option_chain_t* chain,
    const float* volatilities,
    fin_option_chain_result_t* result)
{
    (void)ctx; (void)chain; (void)volatilities; (void)result;
    return false;
}

bool fin_derivatives_gpu_implied_vol_batch(
    nimcp_gpu_context_t* ctx,
    const float* market_prices,
    const fin_option_chain_t* chain,
    float* implied_vols)
{
    (void)ctx; (void)market_prices; (void)chain; (void)implied_vols;
    return false;
}

void fin_option_chain_result_free(fin_option_chain_result_t* result) {
    (void)result;
}

const char* fin_derivatives_gpu_get_last_error(void) {
    return "GPU support not compiled";
}

} // extern "C"

#endif // NIMCP_ENABLE_CUDA
