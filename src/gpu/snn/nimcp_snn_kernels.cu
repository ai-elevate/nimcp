/**
 * @file nimcp_snn_kernels.cu
 * @brief GPU Spiking Neural Network CUDA Kernels
 *
 * WHAT: CUDA kernels for SNN neuron models and learning
 * WHY:  GPU acceleration for spiking neural network simulation
 * HOW:  Custom kernels for LIF, Izhikevich, AdEx, STDP, surrogate gradients
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include "gpu/snn/nimcp_snn_gpu.h"
#include "utils/logging/nimcp_logging.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define LOG_MODULE "SNN_GPU"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define WARP_SIZE 32

//=============================================================================
// LIF Neuron State Management
//=============================================================================

nimcp_lif_state_t* nimcp_lif_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_lif_params_t* params)
{
    if (!ctx || !params || n_neurons == 0) {
        LOG_ERROR("Invalid parameters for LIF state creation");
        return NULL;
    }

    nimcp_lif_state_t* state = (nimcp_lif_state_t*)calloc(1, sizeof(nimcp_lif_state_t));
    if (!state) return NULL;

    size_t dims[1] = {n_neurons};

    state->v = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->i_syn = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->spikes = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!state->v || !state->i_syn || !state->spikes) {
        nimcp_lif_state_destroy(state);
        return NULL;
    }

    // Initialize to resting potential
    nimcp_gpu_fill(ctx, state->v, params->v_rest);
    nimcp_gpu_zeros(ctx, state->i_syn);
    nimcp_gpu_zeros(ctx, state->spikes);

    state->params = *params;

    LOG_DEBUG("Created LIF state for %zu neurons", n_neurons);
    return state;
}

void nimcp_lif_state_destroy(nimcp_lif_state_t* state)
{
    if (!state) return;

    if (state->v) nimcp_gpu_tensor_destroy(state->v);
    if (state->i_syn) nimcp_gpu_tensor_destroy(state->i_syn);
    if (state->spikes) nimcp_gpu_tensor_destroy(state->spikes);
    free(state);
}

//=============================================================================
// LIF Forward Kernel
//=============================================================================

__global__ void kernel_lif_forward(
    float* v, float* i_syn, float* spikes, const float* input,
    float tau_mem, float tau_syn, float v_thresh, float v_reset, float v_rest,
    float dt, bool hard_reset, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Decay factors
    float alpha_mem = expf(-dt / tau_mem);
    float alpha_syn = expf(-dt / tau_syn);

    // Update synaptic current
    float i = i_syn[idx] * alpha_syn + input[idx];
    i_syn[idx] = i;

    // Update membrane potential
    float membrane = v[idx];
    membrane = alpha_mem * membrane + (1.0f - alpha_mem) * (v_rest + i);

    // Check for spike
    float spike = 0.0f;
    if (membrane >= v_thresh) {
        spike = 1.0f;
        if (hard_reset) {
            membrane = v_reset;
        } else {
            membrane = membrane - (v_thresh - v_reset);
        }
    }

    v[idx] = membrane;
    spikes[idx] = spike;
}

bool nimcp_gpu_lif_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    if (!ctx || !state || !input) return false;

    size_t n = state->v->numel;
    const nimcp_lif_params_t* p = &state->params;

    kernel_lif_forward<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->v->data, (float*)state->i_syn->data,
        (float*)state->spikes->data, (const float*)input->data,
        p->tau_mem, p->tau_syn, p->v_thresh, p->v_reset, p->v_rest,
        p->dt, p->hard_reset, n);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Surrogate Gradient Kernels
//=============================================================================

__device__ inline float surrogate_superspike(float x, float beta)
{
    // SuperSpike: 1 / (1 + beta * |x|)^2
    float abs_x = fabsf(x);
    float denom = 1.0f + beta * abs_x;
    return 1.0f / (denom * denom);
}

__device__ inline float surrogate_fast_sigmoid(float x, float beta)
{
    // Fast sigmoid: 0.5 * beta / (1 + beta * |x|)
    return 0.5f * beta / (1.0f + beta * fabsf(x));
}

__device__ inline float surrogate_arctan(float x, float beta)
{
    // Arctan: beta / (pi * (1 + (beta * x)^2))
    float bx = beta * x;
    return beta / (3.14159265f * (1.0f + bx * bx));
}

__device__ inline float surrogate_triangular(float x, float beta)
{
    // Triangular: max(0, 1 - beta * |x|)
    return fmaxf(0.0f, 1.0f - beta * fabsf(x));
}

__device__ inline float surrogate_gaussian(float x, float beta)
{
    // Gaussian: exp(-beta * x^2)
    return expf(-beta * x * x);
}

__global__ void kernel_surrogate_gradient(
    const float* v, float v_thresh, float* grad,
    int surrogate_type, float beta, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = v[idx] - v_thresh;
    float g;

    switch (surrogate_type) {
        case 0:  // SuperSpike
            g = surrogate_superspike(x, beta);
            break;
        case 1:  // Fast sigmoid
            g = surrogate_fast_sigmoid(x, beta);
            break;
        case 2:  // Arctan
            g = surrogate_arctan(x, beta);
            break;
        case 3:  // Triangular
            g = surrogate_triangular(x, beta);
            break;
        case 4:  // Gaussian
            g = surrogate_gaussian(x, beta);
            break;
        default:
            g = surrogate_superspike(x, beta);
    }

    grad[idx] = g;
}

bool nimcp_gpu_surrogate_gradient(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* v,
    float v_thresh,
    nimcp_gpu_tensor_t* grad,
    nimcp_surrogate_type_t surrogate_type,
    float beta)
{
    if (!ctx || !v || !grad) return false;

    kernel_surrogate_gradient<<<GRID_SIZE(v->numel), BLOCK_SIZE>>>(
        (const float*)v->data, v_thresh, (float*)grad->data,
        (int)surrogate_type, beta, v->numel);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// LIF Backward with Surrogate Gradient
//=============================================================================

__global__ void kernel_lif_backward(
    const float* v, float v_thresh, const float* grad_output, float* grad_input,
    int surrogate_type, float beta, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = v[idx] - v_thresh;
    float surrogate;

    switch (surrogate_type) {
        case 0:
            surrogate = surrogate_superspike(x, beta);
            break;
        case 1:
            surrogate = surrogate_fast_sigmoid(x, beta);
            break;
        case 2:
            surrogate = surrogate_arctan(x, beta);
            break;
        case 3:
            surrogate = surrogate_triangular(x, beta);
            break;
        case 4:
            surrogate = surrogate_gaussian(x, beta);
            break;
        default:
            surrogate = surrogate_superspike(x, beta);
    }

    grad_input[idx] = grad_output[idx] * surrogate;
}

bool nimcp_gpu_lif_backward(
    nimcp_gpu_context_t* ctx,
    const nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_surrogate_type_t surrogate_type,
    float beta)
{
    if (!ctx || !state || !grad_output || !grad_input) return false;

    kernel_lif_backward<<<GRID_SIZE(state->v->numel), BLOCK_SIZE>>>(
        (const float*)state->v->data, state->params.v_thresh,
        (const float*)grad_output->data, (float*)grad_input->data,
        (int)surrogate_type, beta, state->v->numel);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Izhikevich Neuron State Management
//=============================================================================

nimcp_izhikevich_state_t* nimcp_izhikevich_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_izhikevich_params_t* params)
{
    if (!ctx || !params || n_neurons == 0) return NULL;

    nimcp_izhikevich_state_t* state = (nimcp_izhikevich_state_t*)calloc(1, sizeof(nimcp_izhikevich_state_t));
    if (!state) return NULL;

    size_t dims[1] = {n_neurons};

    state->v = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->u = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->spikes = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!state->v || !state->u || !state->spikes) {
        nimcp_izhikevich_state_destroy(state);
        return NULL;
    }

    // Initialize to resting state (v=-65mV typical)
    nimcp_gpu_fill(ctx, state->v, -65.0f);
    nimcp_gpu_fill(ctx, state->u, params->b * (-65.0f));
    nimcp_gpu_zeros(ctx, state->spikes);

    state->params = *params;
    return state;
}

void nimcp_izhikevich_state_destroy(nimcp_izhikevich_state_t* state)
{
    if (!state) return;

    if (state->v) nimcp_gpu_tensor_destroy(state->v);
    if (state->u) nimcp_gpu_tensor_destroy(state->u);
    if (state->spikes) nimcp_gpu_tensor_destroy(state->spikes);
    free(state);
}

//=============================================================================
// Izhikevich Forward Kernel
//=============================================================================

__global__ void kernel_izhikevich_forward(
    float* v, float* u, float* spikes, const float* input,
    float a, float b, float c, float d, float v_thresh, float dt, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float membrane = v[idx];
    float recovery = u[idx];
    float I = input[idx];

    // Euler integration (can do 2 steps for stability)
    for (int substep = 0; substep < 2; substep++) {
        float dt_half = dt * 0.5f;

        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        float dv = 0.04f * membrane * membrane + 5.0f * membrane + 140.0f - recovery + I;

        // du/dt = a*(b*v - u)
        float du = a * (b * membrane - recovery);

        membrane += dt_half * dv;
        recovery += dt_half * du;
    }

    // Spike detection and reset
    float spike = 0.0f;
    if (membrane >= v_thresh) {
        spike = 1.0f;
        membrane = c;
        recovery = recovery + d;
    }

    v[idx] = membrane;
    u[idx] = recovery;
    spikes[idx] = spike;
}

bool nimcp_gpu_izhikevich_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_izhikevich_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    if (!ctx || !state || !input) return false;

    size_t n = state->v->numel;
    const nimcp_izhikevich_params_t* p = &state->params;

    kernel_izhikevich_forward<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->v->data, (float*)state->u->data,
        (float*)state->spikes->data, (const float*)input->data,
        p->a, p->b, p->c, p->d, p->v_thresh, p->dt, n);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// AdEx Neuron State Management
//=============================================================================

nimcp_adex_state_t* nimcp_adex_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_adex_params_t* params)
{
    if (!ctx || !params || n_neurons == 0) return NULL;

    nimcp_adex_state_t* state = (nimcp_adex_state_t*)calloc(1, sizeof(nimcp_adex_state_t));
    if (!state) return NULL;

    size_t dims[1] = {n_neurons};

    state->v = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->w = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->spikes = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!state->v || !state->w || !state->spikes) {
        nimcp_adex_state_destroy(state);
        return NULL;
    }

    nimcp_gpu_fill(ctx, state->v, params->v_rest);
    nimcp_gpu_zeros(ctx, state->w);
    nimcp_gpu_zeros(ctx, state->spikes);

    state->params = *params;
    return state;
}

void nimcp_adex_state_destroy(nimcp_adex_state_t* state)
{
    if (!state) return;

    if (state->v) nimcp_gpu_tensor_destroy(state->v);
    if (state->w) nimcp_gpu_tensor_destroy(state->w);
    if (state->spikes) nimcp_gpu_tensor_destroy(state->spikes);
    free(state);
}

//=============================================================================
// AdEx Forward Kernel
//=============================================================================

__global__ void kernel_adex_forward(
    float* v, float* w, float* spikes, const float* input,
    float tau_mem, float tau_w, float v_thresh, float v_reset, float v_rest,
    float v_rheo, float delta_T, float a_sub, float b_spike, float dt, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float membrane = v[idx];
    float adaptation = w[idx];
    float I = input[idx];

    // Exponential term
    float exp_term = delta_T * expf((membrane - v_rheo) / delta_T);

    // dv/dt = (-v + v_rest + exp_term + I - w) / tau_mem
    float dv = (-(membrane - v_rest) + exp_term - adaptation + I) / tau_mem;

    // dw/dt = (a*(v - v_rest) - w) / tau_w
    float dw = (a_sub * (membrane - v_rest) - adaptation) / tau_w;

    membrane += dt * dv;
    adaptation += dt * dw;

    // Spike detection
    float spike = 0.0f;
    if (membrane >= v_thresh) {
        spike = 1.0f;
        membrane = v_reset;
        adaptation += b_spike;
    }

    v[idx] = membrane;
    w[idx] = adaptation;
    spikes[idx] = spike;
}

bool nimcp_gpu_adex_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_adex_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    if (!ctx || !state || !input) return false;

    size_t n = state->v->numel;
    const nimcp_adex_params_t* p = &state->params;

    kernel_adex_forward<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->v->data, (float*)state->w->data,
        (float*)state->spikes->data, (const float*)input->data,
        p->tau_mem, p->tau_w, p->v_thresh, p->v_reset, p->v_rest,
        p->v_rheo, p->delta_T, p->a, p->b, p->dt, n);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Spike Propagation
//=============================================================================

__global__ void kernel_spike_propagate(
    const float* spikes, const float* weights, float* output,
    size_t n_pre, size_t n_post)
{
    size_t post_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (post_idx >= n_post) return;

    float sum = 0.0f;
    for (size_t pre_idx = 0; pre_idx < n_pre; pre_idx++) {
        if (spikes[pre_idx] > 0.0f) {
            sum += weights[pre_idx * n_post + post_idx];
        }
    }
    output[post_idx] += sum;
}

bool nimcp_gpu_spike_propagate(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    const nimcp_gpu_tensor_t* weights,
    nimcp_gpu_tensor_t* output)
{
    if (!ctx || !spikes || !weights || !output) return false;

    size_t n_pre = spikes->numel;
    size_t n_post = output->numel;

    kernel_spike_propagate<<<GRID_SIZE(n_post), BLOCK_SIZE>>>(
        (const float*)spikes->data, (const float*)weights->data,
        (float*)output->data, n_pre, n_post);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_spike_propagate_sparse(
    nimcp_gpu_context_t* ctx,
    const uint32_t* spike_indices,
    size_t n_spikes,
    const nimcp_gpu_tensor_t* weights,
    nimcp_gpu_tensor_t* output)
{
    // TODO: Implement sparse spike propagation for efficiency
    LOG_WARN("Sparse spike propagation not yet implemented");
    return false;
}

//=============================================================================
// STDP Learning
//=============================================================================

__global__ void kernel_eligibility_trace_update(
    float* trace, const float* spikes, float decay, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    trace[idx] = trace[idx] * decay + spikes[idx] * (1.0f - decay);
}

bool nimcp_gpu_eligibility_trace_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* trace,
    const nimcp_gpu_tensor_t* spikes,
    float decay)
{
    if (!ctx || !trace || !spikes) return false;

    kernel_eligibility_trace_update<<<GRID_SIZE(trace->numel), BLOCK_SIZE>>>(
        (float*)trace->data, (const float*)spikes->data, decay, trace->numel);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_stdp_pair(
    float* weights,
    const float* pre_spikes, const float* post_spikes,
    const float* pre_trace, const float* post_trace,
    float A_plus, float A_minus, float w_max, float w_min,
    size_t n_pre, size_t n_post)
{
    size_t pre_idx = blockIdx.x;
    size_t post_idx = threadIdx.x;

    if (pre_idx >= n_pre || post_idx >= n_post) return;

    size_t w_idx = pre_idx * n_post + post_idx;
    float w = weights[w_idx];

    float pre_spike = pre_spikes[pre_idx];
    float post_spike = post_spikes[post_idx];
    float pre_tr = pre_trace[pre_idx];
    float post_tr = post_trace[post_idx];

    // LTP: Post spike with pre trace (pre before post)
    float dw_ltp = A_plus * post_spike * pre_tr * (w_max - w);

    // LTD: Pre spike with post trace (post before pre)
    float dw_ltd = A_minus * pre_spike * post_tr * (w - w_min);

    w = fminf(fmaxf(w + dw_ltp - dw_ltd, w_min), w_max);
    weights[w_idx] = w;
}

bool nimcp_gpu_stdp_pair(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    const nimcp_gpu_tensor_t* pre_trace,
    const nimcp_gpu_tensor_t* post_trace,
    const nimcp_stdp_params_t* params)
{
    if (!ctx || !weights || !pre_spikes || !post_spikes || !pre_trace || !post_trace || !params) {
        return false;
    }

    size_t n_pre = pre_spikes->numel;
    size_t n_post = post_spikes->numel;

    // Launch with pre neurons as blocks, post neurons as threads
    dim3 grid(n_pre);
    dim3 block(n_post < 256 ? n_post : 256);

    kernel_stdp_pair<<<grid, block>>>(
        (float*)weights->data,
        (const float*)pre_spikes->data, (const float*)post_spikes->data,
        (const float*)pre_trace->data, (const float*)post_trace->data,
        params->A_plus, params->A_minus, params->w_max, params->w_min,
        n_pre, n_post);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_stdp_triplet(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    nimcp_gpu_tensor_t* pre_trace_fast,
    nimcp_gpu_tensor_t* pre_trace_slow,
    nimcp_gpu_tensor_t* post_trace_fast,
    nimcp_gpu_tensor_t* post_trace_slow,
    const nimcp_stdp_params_t* params)
{
    // TODO: Implement triplet STDP
    LOG_WARN("Triplet STDP not yet implemented, using pair-based");
    return nimcp_gpu_stdp_pair(ctx, weights, pre_spikes, post_spikes,
                               pre_trace_fast, post_trace_fast, params);
}

//=============================================================================
// Utility Functions
//=============================================================================

__global__ void kernel_reset_state(float* v, float v_rest, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = v_rest;
    }
}

bool nimcp_gpu_snn_reset_state(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* v,
    float v_rest)
{
    if (!ctx || !v) return false;

    kernel_reset_state<<<GRID_SIZE(v->numel), BLOCK_SIZE>>>(
        (float*)v->data, v_rest, v->numel);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

__device__ inline float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kernel_spike_count(const float* spikes, uint32_t* count, size_t n)
{
    __shared__ uint32_t shared[BLOCK_SIZE / WARP_SIZE];

    uint32_t sum = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (spikes[i] > 0.0f) sum++;
    }

    // Warp reduction
    sum = (uint32_t)warp_reduce_sum((float)sum);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
        sum = (uint32_t)warp_reduce_sum((float)sum);
        if (lane == 0) atomicAdd(count, sum);
    }
}

bool nimcp_gpu_spike_count(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    uint32_t* count)
{
    if (!ctx || !spikes || !count) return false;

    uint32_t* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

    int grid = GRID_SIZE(spikes->numel);
    grid = grid > 256 ? 256 : grid;
    kernel_spike_count<<<grid, BLOCK_SIZE>>>((const float*)spikes->data, d_count, spikes->numel);

    CUDA_CHECK(cudaMemcpy(count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_count);

    return true;
}

bool nimcp_gpu_spike_rate(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    size_t n_timesteps,
    nimcp_gpu_tensor_t* rates)
{
    if (!ctx || !spikes || !rates || n_timesteps == 0) return false;

    // Compute mean spikes per neuron over timesteps
    // Assuming spikes is (timesteps, neurons)
    size_t n_neurons = rates->numel;

    nimcp_gpu_mean(ctx, spikes, rates, 0, false);

    return true;
}

#else // !NIMCP_ENABLE_CUDA

#include "gpu/snn/nimcp_snn_gpu.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>

#define LOG_MODULE "SNN_GPU"

nimcp_lif_state_t* nimcp_lif_state_create(nimcp_gpu_context_t* ctx, size_t n_neurons,
    const nimcp_lif_params_t* params)
{
    LOG_WARN("CUDA not available - SNN requires GPU");
    return NULL;
}

void nimcp_lif_state_destroy(nimcp_lif_state_t* state)
{
    if (state) free(state);
}

bool nimcp_gpu_lif_forward(nimcp_gpu_context_t* ctx, nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    return false;
}

bool nimcp_gpu_lif_backward(nimcp_gpu_context_t* ctx, const nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input,
    nimcp_surrogate_type_t surrogate_type, float beta)
{
    return false;
}

nimcp_izhikevich_state_t* nimcp_izhikevich_state_create(nimcp_gpu_context_t* ctx,
    size_t n_neurons, const nimcp_izhikevich_params_t* params)
{
    return NULL;
}

void nimcp_izhikevich_state_destroy(nimcp_izhikevich_state_t* state)
{
    if (state) free(state);
}

bool nimcp_gpu_izhikevich_forward(nimcp_gpu_context_t* ctx, nimcp_izhikevich_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    return false;
}

nimcp_adex_state_t* nimcp_adex_state_create(nimcp_gpu_context_t* ctx, size_t n_neurons,
    const nimcp_adex_params_t* params)
{
    return NULL;
}

void nimcp_adex_state_destroy(nimcp_adex_state_t* state)
{
    if (state) free(state);
}

bool nimcp_gpu_adex_forward(nimcp_gpu_context_t* ctx, nimcp_adex_state_t* state,
    const nimcp_gpu_tensor_t* input)
{
    return false;
}

bool nimcp_gpu_surrogate_gradient(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* v,
    float v_thresh, nimcp_gpu_tensor_t* grad, nimcp_surrogate_type_t type, float beta)
{
    return false;
}

bool nimcp_gpu_spike_propagate(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* spikes,
    const nimcp_gpu_tensor_t* weights, nimcp_gpu_tensor_t* output)
{
    return false;
}

bool nimcp_gpu_spike_propagate_sparse(nimcp_gpu_context_t* ctx, const uint32_t* indices,
    size_t n_spikes, const nimcp_gpu_tensor_t* weights, nimcp_gpu_tensor_t* output)
{
    return false;
}

bool nimcp_gpu_eligibility_trace_update(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* trace,
    const nimcp_gpu_tensor_t* spikes, float decay)
{
    return false;
}

bool nimcp_gpu_stdp_pair(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes, const nimcp_gpu_tensor_t* post_spikes,
    const nimcp_gpu_tensor_t* pre_trace, const nimcp_gpu_tensor_t* post_trace,
    const nimcp_stdp_params_t* params)
{
    return false;
}

bool nimcp_gpu_stdp_triplet(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes, const nimcp_gpu_tensor_t* post_spikes,
    nimcp_gpu_tensor_t* pre_trace_fast, nimcp_gpu_tensor_t* pre_trace_slow,
    nimcp_gpu_tensor_t* post_trace_fast, nimcp_gpu_tensor_t* post_trace_slow,
    const nimcp_stdp_params_t* params)
{
    return false;
}

bool nimcp_gpu_snn_reset_state(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* v, float v_rest)
{
    return false;
}

bool nimcp_gpu_spike_count(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* spikes,
    uint32_t* count)
{
    return false;
}

bool nimcp_gpu_spike_rate(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* spikes,
    size_t n_timesteps, nimcp_gpu_tensor_t* rates)
{
    return false;
}

#endif // NIMCP_ENABLE_CUDA
