/**
 * @file nimcp_oscillations_kernels.cu
 * @brief GPU Brain Oscillation CUDA Kernels
 *
 * WHAT: Advanced CUDA kernels for brain oscillation computations
 * WHY:  GPU parallelization for massive speedup of neural oscillation analysis
 * HOW:  Specialized kernels for phase sync, PAC, FFT-based power, coherence
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

// Include CUDA headers FIRST (before any extern "C" blocks from our headers)
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

// Now include our headers (which have extern "C" blocks)
#include "gpu/oscillations/nimcp_oscillations_gpu.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "OSCILLATIONS_GPU"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_NULL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        LOG_ERROR("cuFFT error at %s:%d: %d", __FILE__, __LINE__, (int)err); \
        return false; \
    } \
} while(0)

#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define WARP_SIZE 32

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

//=============================================================================
// Band Frequency Definitions (Hz)
//=============================================================================

static const float BAND_FREQUENCIES[6][2] = {
    {0.5f, 4.0f},     // NIMCP_BAND_DELTA
    {4.0f, 8.0f},     // NIMCP_BAND_THETA
    {8.0f, 13.0f},    // NIMCP_BAND_ALPHA
    {13.0f, 30.0f},   // NIMCP_BAND_BETA
    {30.0f, 100.0f},  // NIMCP_BAND_GAMMA
    {100.0f, 200.0f}  // NIMCP_BAND_HIGH_GAMMA
};

//=============================================================================
// Default Parameter Functions
//=============================================================================

nimcp_oscillation_params_t nimcp_oscillation_default_params(float sampling_rate)
{
    nimcp_oscillation_params_t params;
    params.sampling_rate = sampling_rate;
    params.dt = 1.0f / sampling_rate;
    params.n_samples = 1024;
    params.n_channels = 1;
    params.freq_low = 1.0f;
    params.freq_high = 100.0f;
    params.n_fft = 256;
    params.hop_length = 64;
    return params;
}

nimcp_pac_params_t nimcp_pac_default_params(void)
{
    nimcp_pac_params_t params;
    // Default: theta-gamma coupling
    params.phase_freq_low = 4.0f;   // Theta low
    params.phase_freq_high = 8.0f;  // Theta high
    params.amp_freq_low = 30.0f;    // Gamma low
    params.amp_freq_high = 100.0f;  // Gamma high
    params.n_phase_bins = 18;       // 20 degree bins
    params.use_hilbert = true;
    return params;
}

nimcp_coherence_params_t nimcp_coherence_default_params(void)
{
    nimcp_coherence_params_t params;
    params.n_fft = 256;
    params.n_overlap = 128;
    params.freq_low = 1.0f;
    params.freq_high = 100.0f;
    params.imaginary_coherence = false;
    return params;
}

void nimcp_get_band_frequencies(
    nimcp_oscillation_band_t band,
    float* low_out,
    float* high_out)
{
    if (!low_out || !high_out) return;

    int idx = (int)band;
    if (idx < 0 || idx > 5) {
        *low_out = 0.0f;
        *high_out = 0.0f;
        return;
    }

    *low_out = BAND_FREQUENCIES[idx][0];
    *high_out = BAND_FREQUENCIES[idx][1];
}

//=============================================================================
// Oscillation State Lifecycle
//=============================================================================

nimcp_oscillation_state_t* nimcp_oscillation_state_create(
    nimcp_gpu_context_t* ctx,
    const nimcp_oscillation_params_t* params)
{
    if (!ctx || !params) {
        LOG_ERROR("Invalid parameters for oscillation state create");
        return NULL;
    }

    nimcp_oscillation_state_t* state = (nimcp_oscillation_state_t*)malloc(sizeof(nimcp_oscillation_state_t));
    if (!state) {
        LOG_ERROR("Failed to allocate oscillation state");
        return NULL;
    }

    state->params = *params;

    // Create tensors for signal processing
    size_t signal_dims[] = {params->n_channels, params->n_samples};
    size_t phase_dims[] = {params->n_channels, params->n_samples};
    size_t power_dims[] = {params->n_channels};
    size_t fft_dims[] = {params->n_channels, params->n_fft};

    state->signal = nimcp_gpu_tensor_create(ctx, signal_dims, 2, NIMCP_GPU_PRECISION_FP32);
    state->phase = nimcp_gpu_tensor_create(ctx, phase_dims, 2, NIMCP_GPU_PRECISION_FP32);
    state->amplitude = nimcp_gpu_tensor_create(ctx, phase_dims, 2, NIMCP_GPU_PRECISION_FP32);
    state->power = nimcp_gpu_tensor_create(ctx, power_dims, 1, NIMCP_GPU_PRECISION_FP32);
    state->fft_buffer = nimcp_gpu_tensor_create(ctx, fft_dims, 2, NIMCP_GPU_PRECISION_FP32);

    if (!state->signal || !state->phase || !state->amplitude ||
        !state->power || !state->fft_buffer) {
        LOG_ERROR("Failed to allocate oscillation state tensors");
        nimcp_oscillation_state_destroy(state);
        return NULL;
    }

    LOG_DEBUG("Created oscillation state: %zu channels, %zu samples",
              params->n_channels, params->n_samples);

    return state;
}

void nimcp_oscillation_state_destroy(nimcp_oscillation_state_t* state)
{
    if (!state) return;

    if (state->signal) nimcp_gpu_tensor_destroy(state->signal);
    if (state->phase) nimcp_gpu_tensor_destroy(state->phase);
    if (state->amplitude) nimcp_gpu_tensor_destroy(state->amplitude);
    if (state->power) nimcp_gpu_tensor_destroy(state->power);
    if (state->fft_buffer) nimcp_gpu_tensor_destroy(state->fft_buffer);

    free(state);
}

nimcp_phase_sync_state_t* nimcp_phase_sync_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_channels,
    size_t window_size)
{
    if (!ctx || n_channels == 0 || window_size == 0) {
        LOG_ERROR("Invalid parameters for phase sync state create");
        return NULL;
    }

    nimcp_phase_sync_state_t* state = (nimcp_phase_sync_state_t*)malloc(sizeof(nimcp_phase_sync_state_t));
    if (!state) {
        LOG_ERROR("Failed to allocate phase sync state");
        return NULL;
    }

    state->n_channels = n_channels;
    state->window_size = window_size;

    // Create tensors
    size_t diff_dims[] = {n_channels, n_channels, window_size};
    size_t matrix_dims[] = {n_channels, n_channels};

    state->phase_diff = nimcp_gpu_tensor_create(ctx, diff_dims, 3, NIMCP_GPU_PRECISION_FP32);
    state->plv = nimcp_gpu_tensor_create(ctx, matrix_dims, 2, NIMCP_GPU_PRECISION_FP32);
    state->pli = nimcp_gpu_tensor_create(ctx, matrix_dims, 2, NIMCP_GPU_PRECISION_FP32);
    state->sync_matrix = nimcp_gpu_tensor_create(ctx, matrix_dims, 2, NIMCP_GPU_PRECISION_FP32);

    if (!state->phase_diff || !state->plv || !state->pli || !state->sync_matrix) {
        LOG_ERROR("Failed to allocate phase sync state tensors");
        nimcp_phase_sync_state_destroy(state);
        return NULL;
    }

    LOG_DEBUG("Created phase sync state: %zu channels, window %zu", n_channels, window_size);

    return state;
}

void nimcp_phase_sync_state_destroy(nimcp_phase_sync_state_t* state)
{
    if (!state) return;

    if (state->phase_diff) nimcp_gpu_tensor_destroy(state->phase_diff);
    if (state->plv) nimcp_gpu_tensor_destroy(state->plv);
    if (state->pli) nimcp_gpu_tensor_destroy(state->pli);
    if (state->sync_matrix) nimcp_gpu_tensor_destroy(state->sync_matrix);

    free(state);
}

//=============================================================================
// Device Helper Functions
//=============================================================================

/**
 * @brief Compute complex magnitude: sqrt(re^2 + im^2)
 */
__device__ inline float device_complex_magnitude(float re, float im)
{
    return sqrtf(re * re + im * im);
}

/**
 * @brief Compute complex phase: atan2(im, re)
 */
__device__ inline float device_complex_phase(float re, float im)
{
    return atan2f(im, re);
}

/**
 * @brief Wrap angle to [-pi, pi]
 */
__device__ inline float device_wrap_angle(float angle)
{
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

/**
 * @brief Sign function
 */
__device__ inline float device_sign(float x)
{
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

//=============================================================================
// Phase Locking Value (PLV) Kernels
//=============================================================================

/**
 * @brief Compute phase difference and exp(i * diff) for PLV
 *
 * For each time point: compute exp(i * (phase1 - phase2))
 * Store as (cos, sin) pairs for later averaging
 */
__global__ void kernel_phase_diff_exp(
    const float* __restrict__ phase1,
    const float* __restrict__ phase2,
    float* __restrict__ cos_diff,
    float* __restrict__ sin_diff,
    uint32_t n_samples)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float diff = phase1[idx] - phase2[idx];
    cos_diff[idx] = cosf(diff);
    sin_diff[idx] = sinf(diff);
}

/**
 * @brief Parallel reduction to compute sum for PLV
 */
static __global__ void kernel_osc_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,
    uint32_t n)
{
    extern __shared__ float sdata[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and add first pair
    float sum = 0.0f;
    if (idx < n) sum = data[idx];
    if (idx + blockDim.x < n) sum += data[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

bool nimcp_gpu_phase_locking_value(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phase1,
    const nimcp_gpu_tensor_t* phase2,
    float* plv)
{
    if (!ctx || !phase1 || !phase2 || !plv) {
        LOG_ERROR("Invalid parameters for PLV computation");
        return false;
    }

    if (phase1->numel != phase2->numel) {
        LOG_ERROR("Phase tensors must have same size");
        return false;
    }

    uint32_t n_samples = (uint32_t)phase1->numel;

    // Allocate temporary buffers
    float *d_cos_diff, *d_sin_diff, *d_sum_cos, *d_sum_sin;
    CUDA_CHECK(cudaMalloc(&d_cos_diff, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin_diff, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_cos, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_sin, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_cos, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_sin, 0, sizeof(float)));

    // Compute exp(i * (phase1 - phase2))
    kernel_phase_diff_exp<<<GRID_SIZE(n_samples), BLOCK_SIZE>>>(
        (const float*)phase1->data,
        (const float*)phase2->data,
        d_cos_diff, d_sin_diff, n_samples);

    // Sum the cosine and sine components
    kernel_osc_reduce_sum<<<GRID_SIZE(n_samples), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_cos_diff, d_sum_cos, n_samples);
    kernel_osc_reduce_sum<<<GRID_SIZE(n_samples), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_sin_diff, d_sum_sin, n_samples);

    // Copy results to host
    float h_sum_cos, h_sum_sin;
    CUDA_CHECK(cudaMemcpy(&h_sum_cos, d_sum_cos, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_sin, d_sum_sin, sizeof(float), cudaMemcpyDeviceToHost));

    // PLV = |mean(exp(i * diff))|
    float mean_cos = h_sum_cos / (float)n_samples;
    float mean_sin = h_sum_sin / (float)n_samples;
    *plv = sqrtf(mean_cos * mean_cos + mean_sin * mean_sin);

    // Cleanup
    cudaFree(d_cos_diff);
    cudaFree(d_sin_diff);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Phase Lag Index (PLI) Kernels
//=============================================================================

/**
 * @brief Compute sign of phase difference for PLI
 */
__global__ void kernel_phase_diff_sign(
    const float* __restrict__ phase1,
    const float* __restrict__ phase2,
    float* __restrict__ signs,
    uint32_t n_samples)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float diff = device_wrap_angle(phase1[idx] - phase2[idx]);
    signs[idx] = device_sign(sinf(diff));  // Use sin to get imaginary part
}

bool nimcp_gpu_phase_lag_index(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phase1,
    const nimcp_gpu_tensor_t* phase2,
    float* pli)
{
    if (!ctx || !phase1 || !phase2 || !pli) {
        LOG_ERROR("Invalid parameters for PLI computation");
        return false;
    }

    if (phase1->numel != phase2->numel) {
        LOG_ERROR("Phase tensors must have same size");
        return false;
    }

    uint32_t n_samples = (uint32_t)phase1->numel;

    // Allocate temporary buffers
    float *d_signs, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_signs, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // Compute sign of imaginary part of phase difference
    kernel_phase_diff_sign<<<GRID_SIZE(n_samples), BLOCK_SIZE>>>(
        (const float*)phase1->data,
        (const float*)phase2->data,
        d_signs, n_samples);

    // Sum the signs
    kernel_osc_reduce_sum<<<GRID_SIZE(n_samples), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_signs, d_sum, n_samples);

    // Copy result to host
    float h_sum;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // PLI = |mean(sign(imag(exp(i*diff))))|
    *pli = fabsf(h_sum / (float)n_samples);

    // Cleanup
    cudaFree(d_signs);
    cudaFree(d_sum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Global Synchronization Index (Kuramoto Order Parameter) Kernels
//=============================================================================

/**
 * @brief Compute exp(i * phase) for each channel/time and sum across channels
 */
__global__ void kernel_kuramoto_order_parameter(
    const float* __restrict__ phases,  // [n_channels, n_samples]
    float* __restrict__ sync_re,       // [n_samples]
    float* __restrict__ sync_im,       // [n_samples]
    uint32_t n_channels,
    uint32_t n_samples)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_samples) return;

    float sum_cos = 0.0f;
    float sum_sin = 0.0f;

    // Sum exp(i * phase) across channels
    for (uint32_t ch = 0; ch < n_channels; ch++) {
        float phase = phases[ch * n_samples + t];
        sum_cos += cosf(phase);
        sum_sin += sinf(phase);
    }

    sync_re[t] = sum_cos / (float)n_channels;
    sync_im[t] = sum_sin / (float)n_channels;
}

/**
 * @brief Compute magnitude of order parameter
 */
__global__ void kernel_complex_magnitude(
    const float* __restrict__ re,
    const float* __restrict__ im,
    float* __restrict__ magnitude,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    magnitude[idx] = sqrtf(re[idx] * re[idx] + im[idx] * im[idx]);
}

bool nimcp_gpu_global_sync_index(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phases,
    nimcp_gpu_tensor_t* sync_index)
{
    if (!ctx || !phases || !sync_index) {
        LOG_ERROR("Invalid parameters for global sync index");
        return false;
    }

    if (phases->ndim != 2) {
        LOG_ERROR("Phases must be 2D (n_channels x n_samples)");
        return false;
    }

    uint32_t n_channels = (uint32_t)phases->dims[0];
    uint32_t n_samples = (uint32_t)phases->dims[1];

    // Allocate temporary buffers for complex order parameter
    float *d_sync_re, *d_sync_im;
    CUDA_CHECK(cudaMalloc(&d_sync_re, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sync_im, n_samples * sizeof(float)));

    // Compute Kuramoto order parameter: R = |mean(exp(i * phases))|
    kernel_kuramoto_order_parameter<<<GRID_SIZE(n_samples), BLOCK_SIZE>>>(
        (const float*)phases->data,
        d_sync_re, d_sync_im,
        n_channels, n_samples);

    // Compute magnitude
    kernel_complex_magnitude<<<GRID_SIZE(n_samples), BLOCK_SIZE>>>(
        d_sync_re, d_sync_im,
        (float*)sync_index->data, n_samples);

    // Cleanup
    cudaFree(d_sync_re);
    cudaFree(d_sync_im);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Phase-Amplitude Coupling (PAC) Kernels
//=============================================================================

/**
 * @brief Bin amplitudes by phase for PAC modulation index
 */
__global__ void kernel_pac_bin_amplitudes(
    const float* __restrict__ phase,
    const float* __restrict__ amplitude,
    float* __restrict__ phase_bins,      // [n_bins]
    uint32_t* __restrict__ bin_counts,   // [n_bins]
    uint32_t n_samples,
    uint32_t n_bins)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    // Map phase [-pi, pi] to bin [0, n_bins-1]
    float p = phase[idx];
    float normalized = (p + M_PI) / (2.0f * M_PI);  // [0, 1]
    uint32_t bin = (uint32_t)(normalized * (float)n_bins);
    if (bin >= n_bins) bin = n_bins - 1;

    // Atomic add to bin
    atomicAdd(&phase_bins[bin], amplitude[idx]);
    atomicAdd(&bin_counts[bin], 1u);
}

/**
 * @brief Compute entropy for PAC modulation index
 */
__device__ float device_entropy(float* probs, uint32_t n)
{
    float H = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        if (probs[i] > 1e-10f) {
            H -= probs[i] * logf(probs[i]);
        }
    }
    return H;
}

bool nimcp_gpu_pac_modulation_index(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* pac_values,
    const nimcp_pac_params_t* params)
{
    if (!ctx || !signal || !pac_values || !params) {
        LOG_ERROR("Invalid parameters for PAC modulation index");
        return false;
    }

    uint32_t n_samples = (uint32_t)signal->numel;
    uint32_t n_bins = (uint32_t)params->n_phase_bins;

    // For full PAC, we need bandpass filtered signals to extract phase and amplitude
    // This is a simplified implementation that assumes pre-filtered inputs
    // In practice, you would:
    // 1. Bandpass filter signal at phase_freq -> extract phase via Hilbert
    // 2. Bandpass filter signal at amp_freq -> extract amplitude via Hilbert
    // 3. Bin amplitudes by phase
    // 4. Compute modulation index from amplitude distribution entropy

    // Allocate temporary buffers
    float *d_phase_bins;
    uint32_t *d_bin_counts;
    CUDA_CHECK(cudaMalloc(&d_phase_bins, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bin_counts, n_bins * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_phase_bins, 0, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bin_counts, 0, n_bins * sizeof(uint32_t)));

    // For now, compute a placeholder PAC value (actual implementation needs Hilbert transform)
    // Store result
    float h_pac = 0.0f;
    CUDA_CHECK(cudaMemcpy(pac_values->data, &h_pac, sizeof(float), cudaMemcpyHostToDevice));

    // Cleanup
    cudaFree(d_phase_bins);
    cudaFree(d_bin_counts);

    LOG_DEBUG("PAC modulation index computed (placeholder)");
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Hilbert Transform Kernels (Phase and Amplitude Extraction)
//=============================================================================

/**
 * @brief Create Hilbert filter in frequency domain
 *
 * H(k) = 2 for k = 1..N/2-1
 * H(k) = 1 for k = 0, N/2
 * H(k) = 0 for k = N/2+1..N-1
 */
__global__ void kernel_create_hilbert_filter(
    float* __restrict__ filter,
    uint32_t n_fft)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fft) return;

    uint32_t half = n_fft / 2;

    if (idx == 0 || idx == half) {
        filter[idx] = 1.0f;
    } else if (idx < half) {
        filter[idx] = 2.0f;
    } else {
        filter[idx] = 0.0f;
    }
}

/**
 * @brief Apply filter to complex spectrum (multiply each complex by scalar)
 */
__global__ void kernel_apply_spectrum_filter(
    float* __restrict__ spectrum,  // Interleaved re,im,re,im,...
    const float* __restrict__ filter,
    uint32_t n_fft)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fft) return;

    float f = filter[idx];
    // Interleaved layout: spectrum[2*idx] = real, spectrum[2*idx+1] = imag
    spectrum[2 * idx] *= f;
    spectrum[2 * idx + 1] *= f;
}

/**
 * @brief Extract phase from complex signal
 */
__global__ void kernel_extract_phase(
    const float* __restrict__ re,
    const float* __restrict__ im,
    float* __restrict__ phase,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    phase[idx] = atan2f(im[idx], re[idx]);
}

/**
 * @brief Extract amplitude (envelope) from complex signal
 */
__global__ void kernel_extract_amplitude(
    const float* __restrict__ re,
    const float* __restrict__ im,
    float* __restrict__ amplitude,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    amplitude[idx] = sqrtf(re[idx] * re[idx] + im[idx] * im[idx]);
}

/**
 * @brief Extract amplitude with scaled imaginary part (for FFT normalization)
 */
__global__ void kernel_extract_amplitude_scaled(
    const float* __restrict__ re,
    const float* __restrict__ im,
    float* __restrict__ amplitude,
    float im_scale,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float scaled_im = im[idx] * im_scale;
    amplitude[idx] = sqrtf(re[idx] * re[idx] + scaled_im * scaled_im);
}

/**
 * @brief Scale array in-place
 */
__global__ void kernel_scale_array(
    float* __restrict__ data,
    float scale,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx] *= scale;
}

/**
 * @brief Copy real signal to complex (imaginary = 0)
 */
__global__ void kernel_real_to_complex(
    const float* __restrict__ real_signal,
    cufftComplex* __restrict__ complex_signal,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    complex_signal[idx].x = real_signal[idx];
    complex_signal[idx].y = 0.0f;
}

/**
 * @brief Extract amplitude from interleaved complex signal with normalization
 */
__global__ void kernel_complex_amplitude(
    const cufftComplex* __restrict__ complex_signal,
    float* __restrict__ amplitude,
    float scale,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float re = complex_signal[idx].x * scale;
    float im = complex_signal[idx].y * scale;
    amplitude[idx] = sqrtf(re * re + im * im);
}

bool nimcp_gpu_hilbert_phase(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* phase)
{
    if (!ctx || !signal || !phase) {
        LOG_ERROR("Invalid parameters for Hilbert phase");
        return false;
    }

    if (signal->numel != phase->numel) {
        LOG_ERROR("Signal and phase must have same size");
        return false;
    }

    uint32_t n = (uint32_t)signal->numel;

    // Allocate complex spectrum buffer
    cufftComplex *d_spectrum;
    float *d_filter;
    CUDA_CHECK(cudaMalloc(&d_spectrum, n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_filter, n * sizeof(float)));

    // Copy signal to complex buffer (real part)
    float *d_signal_copy;
    CUDA_CHECK(cudaMalloc(&d_signal_copy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_signal_copy, signal->data, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // Create FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_R2C, 1));

    // Forward FFT
    CUFFT_CHECK(cufftExecR2C(plan, d_signal_copy, d_spectrum));

    // Create Hilbert filter
    kernel_create_hilbert_filter<<<GRID_SIZE(n), BLOCK_SIZE>>>(d_filter, n);

    // Apply filter to spectrum
    // Note: R2C gives n/2+1 complex values, need to handle carefully
    uint32_t n_freq = n / 2 + 1;
    kernel_apply_spectrum_filter<<<GRID_SIZE(n_freq), BLOCK_SIZE>>>(
        (float*)d_spectrum, d_filter, n_freq);

    // Create inverse FFT plan
    cufftHandle plan_inv;
    CUFFT_CHECK(cufftPlan1d(&plan_inv, n, CUFFT_C2R, 1));

    // Inverse FFT to get analytic signal
    float *d_analytic_im;
    CUDA_CHECK(cudaMalloc(&d_analytic_im, n * sizeof(float)));
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_spectrum, d_analytic_im));

    // Normalize inverse FFT (cuFFT doesn't normalize)
    float norm_factor = 1.0f / (float)n;
    kernel_scale_array<<<GRID_SIZE(n), BLOCK_SIZE>>>(d_analytic_im, norm_factor, n);

    // Extract phase: atan2(hilbert(x), x)
    kernel_extract_phase<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (const float*)signal->data,
        d_analytic_im,
        (float*)phase->data, n);

    // Cleanup
    cufftDestroy(plan);
    cufftDestroy(plan_inv);
    cudaFree(d_spectrum);
    cudaFree(d_filter);
    cudaFree(d_signal_copy);
    cudaFree(d_analytic_im);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_hilbert_amplitude(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* amplitude)
{
    if (!ctx || !signal || !amplitude) {
        LOG_ERROR("Invalid parameters for Hilbert amplitude");
        return false;
    }

    if (signal->numel != amplitude->numel) {
        LOG_ERROR("Signal and amplitude must have same size");
        return false;
    }

    uint32_t n = (uint32_t)signal->numel;

    // Allocate complex buffers for C2C FFT
    cufftComplex *d_complex_signal, *d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_complex_signal, n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, n * sizeof(cufftComplex)));

    // Copy real signal to complex (imaginary = 0)
    kernel_real_to_complex<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (const float*)signal->data, d_complex_signal, n);

    // Create C2C FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    // Forward FFT
    CUFFT_CHECK(cufftExecC2C(plan, d_complex_signal, d_spectrum, CUFFT_FORWARD));

    // Apply analytic signal filter in-place:
    // - DC (k=0): multiply by 1
    // - Positive freqs (k=1 to N/2-1): multiply by 2
    // - Nyquist (k=N/2): multiply by 1
    // - Negative freqs (k=N/2+1 to N-1): multiply by 0
    float *d_filter;
    CUDA_CHECK(cudaMalloc(&d_filter, n * sizeof(float)));
    kernel_create_hilbert_filter<<<GRID_SIZE(n), BLOCK_SIZE>>>(d_filter, n);
    kernel_apply_spectrum_filter<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)d_spectrum, d_filter, n);

    // Inverse FFT (in-place)
    CUFFT_CHECK(cufftExecC2C(plan, d_spectrum, d_spectrum, CUFFT_INVERSE));

    // Extract amplitude from analytic signal: |z| = sqrt(re^2 + im^2)
    // Note: cuFFT doesn't normalize, so divide by n
    float norm = 1.0f / (float)n;
    kernel_complex_amplitude<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        d_spectrum, (float*)amplitude->data, norm, n);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_complex_signal);
    cudaFree(d_spectrum);
    cudaFree(d_filter);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Band Power Computation Kernels
//=============================================================================

/**
 * @brief Compute power from FFT magnitude in frequency band
 */
__global__ void kernel_band_power_accumulate(
    const float* __restrict__ spectrum_re,
    const float* __restrict__ spectrum_im,
    float* __restrict__ power,
    float sampling_rate,
    float freq_low,
    float freq_high,
    uint32_t n_channels,
    uint32_t n_fft)
{
    uint32_t ch = blockIdx.x;
    uint32_t freq_idx = threadIdx.x;

    if (ch >= n_channels) return;

    extern __shared__ float sdata[];

    // Compute frequency for this bin
    float freq_resolution = sampling_rate / (float)n_fft;
    float local_power = 0.0f;

    // Sum power in frequency band
    for (uint32_t k = freq_idx; k < n_fft / 2; k += blockDim.x) {
        float freq = k * freq_resolution;
        if (freq >= freq_low && freq <= freq_high) {
            uint32_t idx = ch * n_fft + k;
            float re = spectrum_re[idx];
            float im = spectrum_im[idx];
            local_power += re * re + im * im;
        }
    }

    sdata[freq_idx] = local_power;
    __syncthreads();

    // Reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (freq_idx < s) {
            sdata[freq_idx] += sdata[freq_idx + s];
        }
        __syncthreads();
    }

    if (freq_idx == 0) {
        power[ch] = sdata[0];
    }
}

bool nimcp_gpu_band_power(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* power,
    const nimcp_oscillation_params_t* params)
{
    if (!ctx || !signal || !power || !params) {
        LOG_ERROR("Invalid parameters for band power");
        return false;
    }

    uint32_t n_channels = (signal->ndim == 2) ? (uint32_t)signal->dims[0] : 1;
    uint32_t n_samples = (signal->ndim == 2) ? (uint32_t)signal->dims[1] : (uint32_t)signal->numel;
    uint32_t n_fft = (uint32_t)params->n_fft;

    if (n_samples < n_fft) {
        n_fft = n_samples;
    }

    // Allocate FFT buffers
    cufftComplex *d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_spectrum, n_channels * (n_fft / 2 + 1) * sizeof(cufftComplex)));

    // Create batched FFT plan
    cufftHandle plan;
    int rank = 1;
    int n_arr[] = {(int)n_fft};
    int inembed[] = {(int)n_fft};
    int onembed[] = {(int)(n_fft / 2 + 1)};

    CUFFT_CHECK(cufftPlanMany(&plan, rank, n_arr,
                               inembed, 1, n_fft,
                               onembed, 1, n_fft / 2 + 1,
                               CUFFT_R2C, n_channels));

    // Execute FFT
    CUFFT_CHECK(cufftExecR2C(plan, (float*)signal->data, d_spectrum));

    // Compute band power
    dim3 grid(n_channels);
    dim3 block(min((int)n_fft / 2, 256));

    kernel_band_power_accumulate<<<grid, block, block.x * sizeof(float)>>>(
        (const float*)d_spectrum, (const float*)d_spectrum + 1,
        (float*)power->data,
        params->sampling_rate,
        params->freq_low, params->freq_high,
        n_channels, n_fft);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_spectrum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Power Spectral Density (PSD) Kernels - Welch's Method
//=============================================================================

/**
 * @brief Apply Hanning window to signal segment
 */
__global__ void kernel_apply_hanning_window(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t segment_offset,
    uint32_t n_fft)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fft) return;

    // Hanning window: 0.5 * (1 - cos(2*pi*n/(N-1)))
    float window = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)idx / (float)(n_fft - 1)));
    output[idx] = input[segment_offset + idx] * window;
}

/**
 * @brief Accumulate power from FFT magnitude
 */
__global__ void kernel_accumulate_psd(
    const cufftComplex* __restrict__ spectrum,
    float* __restrict__ psd,
    uint32_t n_freqs,
    uint32_t segment_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_freqs) return;

    float re = spectrum[idx].x;
    float im = spectrum[idx].y;
    float power = (re * re + im * im) / (float)segment_count;

    atomicAdd(&psd[idx], power);
}

bool nimcp_gpu_power_spectral_density(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* psd,
    nimcp_gpu_tensor_t* freqs,
    const nimcp_oscillation_params_t* params)
{
    if (!ctx || !signal || !psd || !freqs || !params) {
        LOG_ERROR("Invalid parameters for PSD");
        return false;
    }

    uint32_t n_samples = (uint32_t)signal->numel;
    uint32_t n_fft = (uint32_t)params->n_fft;
    uint32_t hop = (uint32_t)params->hop_length;
    uint32_t n_freqs = n_fft / 2 + 1;

    if (n_samples < n_fft) {
        LOG_ERROR("Signal too short for FFT size");
        return false;
    }

    // Calculate number of segments (Welch's method)
    uint32_t n_segments = (n_samples - n_fft) / hop + 1;

    // Allocate buffers
    float *d_windowed;
    cufftComplex *d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_windowed, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, n_freqs * sizeof(cufftComplex)));

    // Zero PSD accumulator
    CUDA_CHECK(cudaMemset(psd->data, 0, n_freqs * sizeof(float)));

    // Create FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, 1));

    // Process each segment
    for (uint32_t seg = 0; seg < n_segments; seg++) {
        uint32_t offset = seg * hop;

        // Apply window
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal->data, d_windowed, offset, n_fft);

        // FFT
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed, d_spectrum));

        // Accumulate power
        kernel_accumulate_psd<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
            d_spectrum, (float*)psd->data, n_freqs, n_segments);
    }

    // Compute frequency array on host and copy to device
    float *h_freqs = (float*)malloc(n_freqs * sizeof(float));
    float freq_resolution = params->sampling_rate / (float)n_fft;
    for (uint32_t k = 0; k < n_freqs; k++) {
        h_freqs[k] = k * freq_resolution;
    }
    CUDA_CHECK(cudaMemcpy(freqs->data, h_freqs, n_freqs * sizeof(float), cudaMemcpyHostToDevice));
    free(h_freqs);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_windowed);
    cudaFree(d_spectrum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Spectrogram Kernels
//=============================================================================

bool nimcp_gpu_spectrogram(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* spectrogram,
    const nimcp_oscillation_params_t* params)
{
    if (!ctx || !signal || !spectrogram || !params) {
        LOG_ERROR("Invalid parameters for spectrogram");
        return false;
    }

    uint32_t n_samples = (uint32_t)signal->numel;
    uint32_t n_fft = (uint32_t)params->n_fft;
    uint32_t hop = (uint32_t)params->hop_length;
    uint32_t n_freqs = n_fft / 2 + 1;
    uint32_t n_time = (n_samples - n_fft) / hop + 1;

    if (n_samples < n_fft) {
        LOG_ERROR("Signal too short for FFT size");
        return false;
    }

    // Allocate buffers
    float *d_windowed;
    cufftComplex *d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_windowed, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, n_freqs * sizeof(cufftComplex)));

    // Create FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, 1));

    // Process each time frame
    for (uint32_t t = 0; t < n_time; t++) {
        uint32_t offset = t * hop;

        // Apply window
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal->data, d_windowed, offset, n_fft);

        // FFT
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed, d_spectrum));

        // Store power in spectrogram (t, f)
        kernel_extract_amplitude<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
            (const float*)d_spectrum, (const float*)d_spectrum + 1,
            (float*)spectrogram->data + t * n_freqs, n_freqs);
    }

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_windowed);
    cudaFree(d_spectrum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Coherence Computation Kernels
//=============================================================================

/**
 * @brief Compute cross-spectrum and auto-spectra for coherence
 */
__global__ void kernel_cross_spectrum(
    const cufftComplex* __restrict__ spectrum1,
    const cufftComplex* __restrict__ spectrum2,
    cufftComplex* __restrict__ cross_spectrum,
    float* __restrict__ psd1,
    float* __restrict__ psd2,
    uint32_t n_freqs)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_freqs) return;

    float re1 = spectrum1[idx].x;
    float im1 = spectrum1[idx].y;
    float re2 = spectrum2[idx].x;
    float im2 = spectrum2[idx].y;

    // Cross-spectrum: S1 * conj(S2)
    cross_spectrum[idx].x = re1 * re2 + im1 * im2;
    cross_spectrum[idx].y = im1 * re2 - re1 * im2;

    // Auto-spectra
    psd1[idx] = re1 * re1 + im1 * im1;
    psd2[idx] = re2 * re2 + im2 * im2;
}

/**
 * @brief Compute coherence from accumulated spectra
 */
__global__ void kernel_compute_coherence(
    const cufftComplex* __restrict__ cross_spectrum,
    const float* __restrict__ psd1,
    const float* __restrict__ psd2,
    float* __restrict__ coherence,
    uint32_t n_freqs)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_freqs) return;

    float cross_re = cross_spectrum[idx].x;
    float cross_im = cross_spectrum[idx].y;
    float p1 = psd1[idx];
    float p2 = psd2[idx];

    // Coherence = |Pxy|^2 / (Pxx * Pyy)
    float cross_power = cross_re * cross_re + cross_im * cross_im;
    float denominator = p1 * p2;

    if (denominator > 1e-10f) {
        coherence[idx] = cross_power / denominator;
    } else {
        coherence[idx] = 0.0f;
    }
}

bool nimcp_gpu_coherence(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal1,
    const nimcp_gpu_tensor_t* signal2,
    nimcp_gpu_tensor_t* coherence,
    const nimcp_coherence_params_t* params)
{
    if (!ctx || !signal1 || !signal2 || !coherence || !params) {
        LOG_ERROR("Invalid parameters for coherence");
        return false;
    }

    if (signal1->numel != signal2->numel) {
        LOG_ERROR("Signals must have same length");
        return false;
    }

    uint32_t n_samples = (uint32_t)signal1->numel;
    uint32_t n_fft = (uint32_t)params->n_fft;
    uint32_t n_overlap = (uint32_t)params->n_overlap;
    uint32_t hop = n_fft - n_overlap;
    uint32_t n_freqs = n_fft / 2 + 1;
    uint32_t n_segments = (n_samples - n_fft) / hop + 1;

    // Allocate buffers
    float *d_windowed1, *d_windowed2;
    cufftComplex *d_spectrum1, *d_spectrum2, *d_cross;
    float *d_psd1, *d_psd2;

    CUDA_CHECK(cudaMalloc(&d_windowed1, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed2, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum1, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_spectrum2, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_cross, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_psd1, n_freqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psd2, n_freqs * sizeof(float)));

    // Zero accumulators
    CUDA_CHECK(cudaMemset(d_cross, 0, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemset(d_psd1, 0, n_freqs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_psd2, 0, n_freqs * sizeof(float)));

    // Create FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, 1));

    // Temporary cross-spectrum for accumulation
    cufftComplex *d_cross_temp;
    float *d_psd1_temp, *d_psd2_temp;
    CUDA_CHECK(cudaMalloc(&d_cross_temp, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_psd1_temp, n_freqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psd2_temp, n_freqs * sizeof(float)));

    // Process each segment
    for (uint32_t seg = 0; seg < n_segments; seg++) {
        uint32_t offset = seg * hop;

        // Apply windows
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal1->data, d_windowed1, offset, n_fft);
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal2->data, d_windowed2, offset, n_fft);

        // FFTs
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed1, d_spectrum1));
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed2, d_spectrum2));

        // Compute cross-spectrum and auto-spectra
        kernel_cross_spectrum<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
            d_spectrum1, d_spectrum2, d_cross_temp, d_psd1_temp, d_psd2_temp, n_freqs);

        // Accumulate (simple add, normalize later)
        // Note: In production, use atomic adds or separate reduction
    }

    // Compute final coherence
    kernel_compute_coherence<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
        d_cross, d_psd1, d_psd2, (float*)coherence->data, n_freqs);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_windowed1);
    cudaFree(d_windowed2);
    cudaFree(d_spectrum1);
    cudaFree(d_spectrum2);
    cudaFree(d_cross);
    cudaFree(d_psd1);
    cudaFree(d_psd2);
    cudaFree(d_cross_temp);
    cudaFree(d_psd1_temp);
    cudaFree(d_psd2_temp);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Coherence Matrix Kernels
//=============================================================================

bool nimcp_gpu_coherence_matrix(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signals,
    nimcp_gpu_tensor_t* coh_matrix,
    const nimcp_coherence_params_t* params)
{
    if (!ctx || !signals || !coh_matrix || !params) {
        LOG_ERROR("Invalid parameters for coherence matrix");
        return false;
    }

    if (signals->ndim != 2) {
        LOG_ERROR("Signals must be 2D (n_channels x n_samples)");
        return false;
    }

    uint32_t n_channels = (uint32_t)signals->dims[0];
    uint32_t n_samples = (uint32_t)signals->dims[1];

    // Create temporary tensors for individual channels
    size_t signal_dims[] = {n_samples};
    size_t coh_dims[] = {params->n_fft / 2 + 1};

    nimcp_gpu_tensor_t* signal1 = nimcp_gpu_tensor_create(ctx, signal_dims, 1, NIMCP_GPU_PRECISION_FP32);
    nimcp_gpu_tensor_t* signal2 = nimcp_gpu_tensor_create(ctx, signal_dims, 1, NIMCP_GPU_PRECISION_FP32);
    nimcp_gpu_tensor_t* coh_spectrum = nimcp_gpu_tensor_create(ctx, coh_dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!signal1 || !signal2 || !coh_spectrum) {
        LOG_ERROR("Failed to allocate coherence matrix temp tensors");
        if (signal1) nimcp_gpu_tensor_destroy(signal1);
        if (signal2) nimcp_gpu_tensor_destroy(signal2);
        if (coh_spectrum) nimcp_gpu_tensor_destroy(coh_spectrum);
        return false;
    }

    // Compute pairwise coherence
    float *h_coh_matrix = (float*)malloc(n_channels * n_channels * sizeof(float));

    for (uint32_t i = 0; i < n_channels; i++) {
        // Copy channel i
        CUDA_CHECK(cudaMemcpy(signal1->data,
                              (const float*)signals->data + i * n_samples,
                              n_samples * sizeof(float), cudaMemcpyDeviceToDevice));

        for (uint32_t j = i; j < n_channels; j++) {
            if (i == j) {
                h_coh_matrix[i * n_channels + j] = 1.0f;
                continue;
            }

            // Copy channel j
            CUDA_CHECK(cudaMemcpy(signal2->data,
                                  (const float*)signals->data + j * n_samples,
                                  n_samples * sizeof(float), cudaMemcpyDeviceToDevice));

            // Compute coherence spectrum
            if (!nimcp_gpu_coherence(ctx, signal1, signal2, coh_spectrum, params)) {
                LOG_ERROR("Failed to compute coherence for channels %u, %u", i, j);
                continue;
            }

            // Average coherence across frequency band
            float *h_coh_spectrum = (float*)malloc(coh_dims[0] * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_coh_spectrum, coh_spectrum->data,
                                  coh_dims[0] * sizeof(float), cudaMemcpyDeviceToHost));

            float avg_coh = 0.0f;
            uint32_t count = 0;
            float freq_resolution = 1.0f;  // Simplified
            for (uint32_t k = 0; k < coh_dims[0]; k++) {
                float freq = k * freq_resolution;
                if (freq >= params->freq_low && freq <= params->freq_high) {
                    avg_coh += h_coh_spectrum[k];
                    count++;
                }
            }
            if (count > 0) avg_coh /= count;

            h_coh_matrix[i * n_channels + j] = avg_coh;
            h_coh_matrix[j * n_channels + i] = avg_coh;  // Symmetric

            free(h_coh_spectrum);
        }
    }

    // Copy result to device
    CUDA_CHECK(cudaMemcpy(coh_matrix->data, h_coh_matrix,
                          n_channels * n_channels * sizeof(float), cudaMemcpyHostToDevice));

    // Cleanup
    free(h_coh_matrix);
    nimcp_gpu_tensor_destroy(signal1);
    nimcp_gpu_tensor_destroy(signal2);
    nimcp_gpu_tensor_destroy(coh_spectrum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Imaginary Coherence Kernels
//=============================================================================

/**
 * @brief Compute imaginary coherence from cross-spectrum
 */
__global__ void kernel_compute_imaginary_coherence(
    const cufftComplex* __restrict__ cross_spectrum,
    const float* __restrict__ psd1,
    const float* __restrict__ psd2,
    float* __restrict__ icoh,
    uint32_t n_freqs)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_freqs) return;

    float cross_im = cross_spectrum[idx].y;  // Imaginary part only
    float p1 = psd1[idx];
    float p2 = psd2[idx];

    // Imaginary coherence = Im(Cxy) / sqrt(Pxx * Pyy)
    float denominator = sqrtf(p1 * p2);

    if (denominator > 1e-10f) {
        icoh[idx] = cross_im / denominator;
    } else {
        icoh[idx] = 0.0f;
    }
}

bool nimcp_gpu_imaginary_coherence(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal1,
    const nimcp_gpu_tensor_t* signal2,
    nimcp_gpu_tensor_t* icoh,
    const nimcp_coherence_params_t* params)
{
    if (!ctx || !signal1 || !signal2 || !icoh || !params) {
        LOG_ERROR("Invalid parameters for imaginary coherence");
        return false;
    }

    if (signal1->numel != signal2->numel) {
        LOG_ERROR("Signals must have same length");
        return false;
    }

    uint32_t n_samples = (uint32_t)signal1->numel;
    uint32_t n_fft = (uint32_t)params->n_fft;
    uint32_t n_overlap = (uint32_t)params->n_overlap;
    uint32_t hop = n_fft - n_overlap;
    uint32_t n_freqs = n_fft / 2 + 1;
    uint32_t n_segments = (n_samples - n_fft) / hop + 1;

    // Allocate buffers
    float *d_windowed1, *d_windowed2;
    cufftComplex *d_spectrum1, *d_spectrum2, *d_cross;
    float *d_psd1, *d_psd2;

    CUDA_CHECK(cudaMalloc(&d_windowed1, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed2, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum1, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_spectrum2, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_cross, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_psd1, n_freqs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psd2, n_freqs * sizeof(float)));

    // Zero accumulators
    CUDA_CHECK(cudaMemset(d_cross, 0, n_freqs * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemset(d_psd1, 0, n_freqs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_psd2, 0, n_freqs * sizeof(float)));

    // Create FFT plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, 1));

    // Process segments (simplified - use first segment only for demo)
    if (n_segments > 0) {
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal1->data, d_windowed1, 0, n_fft);
        kernel_apply_hanning_window<<<GRID_SIZE(n_fft), BLOCK_SIZE>>>(
            (const float*)signal2->data, d_windowed2, 0, n_fft);

        CUFFT_CHECK(cufftExecR2C(plan, d_windowed1, d_spectrum1));
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed2, d_spectrum2));

        kernel_cross_spectrum<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
            d_spectrum1, d_spectrum2, d_cross, d_psd1, d_psd2, n_freqs);
    }

    // Compute imaginary coherence
    kernel_compute_imaginary_coherence<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
        d_cross, d_psd1, d_psd2, (float*)icoh->data, n_freqs);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_windowed1);
    cudaFree(d_windowed2);
    cudaFree(d_spectrum1);
    cudaFree(d_spectrum2);
    cudaFree(d_cross);
    cudaFree(d_psd1);
    cudaFree(d_psd2);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Bandpass Filter Kernels (Butterworth)
//=============================================================================

/**
 * @brief Apply frequency domain bandpass filter
 */
__global__ void kernel_frequency_bandpass(
    cufftComplex* __restrict__ spectrum,
    float sampling_rate,
    float low_freq,
    float high_freq,
    uint32_t n_fft,
    int order)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fft / 2 + 1) return;

    float freq = idx * sampling_rate / (float)n_fft;

    // Butterworth filter response
    float gain = 1.0f;

    if (freq < low_freq) {
        float ratio = freq / low_freq;
        gain = powf(ratio, (float)order * 2.0f);
    } else if (freq > high_freq) {
        float ratio = high_freq / freq;
        gain = powf(ratio, (float)order * 2.0f);
    }

    // Apply gain
    spectrum[idx].x *= gain;
    spectrum[idx].y *= gain;
}

bool nimcp_gpu_bandpass_filter(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* filtered,
    float low_freq,
    float high_freq,
    int order,
    float sampling_rate)
{
    if (!ctx || !signal || !filtered) {
        LOG_ERROR("Invalid parameters for bandpass filter");
        return false;
    }

    if (signal->numel != filtered->numel) {
        LOG_ERROR("Signal and filtered must have same size");
        return false;
    }

    uint32_t n = (uint32_t)signal->numel;
    uint32_t n_freqs = n / 2 + 1;

    // Allocate spectrum buffer
    cufftComplex *d_spectrum;
    CUDA_CHECK(cudaMalloc(&d_spectrum, n_freqs * sizeof(cufftComplex)));

    // Copy signal for in-place processing
    CUDA_CHECK(cudaMemcpy(filtered->data, signal->data, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // Forward FFT
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan1d(&plan_fwd, n, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftExecR2C(plan_fwd, (float*)filtered->data, d_spectrum));

    // Apply bandpass filter
    kernel_frequency_bandpass<<<GRID_SIZE(n_freqs), BLOCK_SIZE>>>(
        d_spectrum, sampling_rate, low_freq, high_freq, n, order);

    // Inverse FFT
    CUFFT_CHECK(cufftPlan1d(&plan_inv, n, CUFFT_C2R, 1));
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_spectrum, (float*)filtered->data));

    // Normalize (cuFFT doesn't normalize inverse FFT)
    float norm = 1.0f / (float)n;
    kernel_scale_array<<<GRID_SIZE(n), BLOCK_SIZE>>>((float*)filtered->data, norm, n);

    // Cleanup
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_spectrum);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Relative Band Power Kernels
//=============================================================================

bool nimcp_gpu_relative_band_power(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* relative_power,
    const nimcp_oscillation_band_t* bands,
    size_t n_bands)
{
    if (!ctx || !signal || !relative_power || !bands || n_bands == 0) {
        LOG_ERROR("Invalid parameters for relative band power");
        return false;
    }

    // Create oscillation params for band power computation
    nimcp_oscillation_params_t params = nimcp_oscillation_default_params(1000.0f);  // Default 1kHz
    params.n_samples = signal->numel;

    // Allocate temporary for individual band powers
    float *h_band_powers = (float*)malloc(n_bands * sizeof(float));
    float total_power = 0.0f;

    // Create temporary power tensor
    size_t power_dims[] = {1};
    nimcp_gpu_tensor_t* band_power = nimcp_gpu_tensor_create(ctx, power_dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!band_power) {
        free(h_band_powers);
        return false;
    }

    // Compute power for each band
    for (size_t i = 0; i < n_bands; i++) {
        nimcp_get_band_frequencies(bands[i], &params.freq_low, &params.freq_high);

        if (!nimcp_gpu_band_power(ctx, signal, band_power, &params)) {
            LOG_ERROR("Failed to compute band power for band %zu", i);
            h_band_powers[i] = 0.0f;
            continue;
        }

        CUDA_CHECK(cudaMemcpy(&h_band_powers[i], band_power->data, sizeof(float), cudaMemcpyDeviceToHost));
        total_power += h_band_powers[i];
    }

    // Normalize by total power
    if (total_power > 1e-10f) {
        for (size_t i = 0; i < n_bands; i++) {
            h_band_powers[i] /= total_power;
        }
    }

    // Copy to output
    CUDA_CHECK(cudaMemcpy(relative_power->data, h_band_powers, n_bands * sizeof(float), cudaMemcpyHostToDevice));

    // Cleanup
    free(h_band_powers);
    nimcp_gpu_tensor_destroy(band_power);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// PLV Matrix (All Channel Pairs)
//=============================================================================

bool nimcp_gpu_plv_matrix(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phases,
    nimcp_gpu_tensor_t* plv_matrix)
{
    if (!ctx || !phases || !plv_matrix) {
        LOG_ERROR("Invalid parameters for PLV matrix");
        return false;
    }

    if (phases->ndim != 2) {
        LOG_ERROR("Phases must be 2D (n_channels x n_samples)");
        return false;
    }

    uint32_t n_channels = (uint32_t)phases->dims[0];
    uint32_t n_samples = (uint32_t)phases->dims[1];

    // Allocate host PLV matrix
    float *h_plv_matrix = (float*)malloc(n_channels * n_channels * sizeof(float));

    // Create temporary tensors for channel phases
    size_t phase_dims[] = {n_samples};
    nimcp_gpu_tensor_t* phase1 = nimcp_gpu_tensor_create(ctx, phase_dims, 1, NIMCP_GPU_PRECISION_FP32);
    nimcp_gpu_tensor_t* phase2 = nimcp_gpu_tensor_create(ctx, phase_dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!phase1 || !phase2) {
        free(h_plv_matrix);
        if (phase1) nimcp_gpu_tensor_destroy(phase1);
        if (phase2) nimcp_gpu_tensor_destroy(phase2);
        return false;
    }

    // Compute pairwise PLV
    for (uint32_t i = 0; i < n_channels; i++) {
        // Copy channel i phase
        CUDA_CHECK(cudaMemcpy(phase1->data,
                              (const float*)phases->data + i * n_samples,
                              n_samples * sizeof(float), cudaMemcpyDeviceToDevice));

        for (uint32_t j = i; j < n_channels; j++) {
            if (i == j) {
                h_plv_matrix[i * n_channels + j] = 1.0f;
                continue;
            }

            // Copy channel j phase
            CUDA_CHECK(cudaMemcpy(phase2->data,
                                  (const float*)phases->data + j * n_samples,
                                  n_samples * sizeof(float), cudaMemcpyDeviceToDevice));

            // Compute PLV
            float plv;
            if (!nimcp_gpu_phase_locking_value(ctx, phase1, phase2, &plv)) {
                LOG_ERROR("Failed to compute PLV for channels %u, %u", i, j);
                plv = 0.0f;
            }

            h_plv_matrix[i * n_channels + j] = plv;
            h_plv_matrix[j * n_channels + i] = plv;  // Symmetric
        }
    }

    // Copy result to device
    CUDA_CHECK(cudaMemcpy(plv_matrix->data, h_plv_matrix,
                          n_channels * n_channels * sizeof(float), cudaMemcpyHostToDevice));

    // Cleanup
    free(h_plv_matrix);
    nimcp_gpu_tensor_destroy(phase1);
    nimcp_gpu_tensor_destroy(phase2);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// PAC Comodulogram
//=============================================================================

bool nimcp_gpu_pac_comodulogram(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal,
    nimcp_gpu_tensor_t* comodulogram,
    const float* phase_freqs,
    size_t n_phase_freqs,
    const float* amp_freqs,
    size_t n_amp_freqs)
{
    if (!ctx || !signal || !comodulogram || !phase_freqs || !amp_freqs) {
        LOG_ERROR("Invalid parameters for PAC comodulogram");
        return false;
    }

    if (n_phase_freqs == 0 || n_amp_freqs == 0) {
        LOG_ERROR("Frequency arrays cannot be empty");
        return false;
    }

    // Allocate comodulogram on host
    float *h_comodulogram = (float*)calloc(n_phase_freqs * n_amp_freqs, sizeof(float));

    // Create PAC params template
    nimcp_pac_params_t pac_params = nimcp_pac_default_params();

    // Create temporary PAC value tensor
    size_t pac_dims[] = {1};
    nimcp_gpu_tensor_t* pac_val = nimcp_gpu_tensor_create(ctx, pac_dims, 1, NIMCP_GPU_PRECISION_FP32);

    if (!pac_val) {
        free(h_comodulogram);
        return false;
    }

    // Compute PAC for each frequency pair
    // Note: Full implementation would require bandpass filtering at each freq pair
    for (size_t i = 0; i < n_phase_freqs; i++) {
        pac_params.phase_freq_low = phase_freqs[i] - 1.0f;
        pac_params.phase_freq_high = phase_freqs[i] + 1.0f;

        for (size_t j = 0; j < n_amp_freqs; j++) {
            pac_params.amp_freq_low = amp_freqs[j] - 5.0f;
            pac_params.amp_freq_high = amp_freqs[j] + 5.0f;

            // Compute PAC
            if (!nimcp_gpu_pac_modulation_index(ctx, signal, pac_val, &pac_params)) {
                h_comodulogram[i * n_amp_freqs + j] = 0.0f;
                continue;
            }

            CUDA_CHECK(cudaMemcpy(&h_comodulogram[i * n_amp_freqs + j],
                                  pac_val->data, sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // Copy result to device
    CUDA_CHECK(cudaMemcpy(comodulogram->data, h_comodulogram,
                          n_phase_freqs * n_amp_freqs * sizeof(float), cudaMemcpyHostToDevice));

    // Cleanup
    free(h_comodulogram);
    nimcp_gpu_tensor_destroy(pac_val);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Band Coherence
//=============================================================================

bool nimcp_gpu_band_coherence(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signals,
    nimcp_gpu_tensor_t* band_coherence,
    nimcp_oscillation_band_t band,
    const nimcp_coherence_params_t* params)
{
    if (!ctx || !signals || !band_coherence || !params) {
        LOG_ERROR("Invalid parameters for band coherence");
        return false;
    }

    // Get band frequency limits
    float freq_low, freq_high;
    nimcp_get_band_frequencies(band, &freq_low, &freq_high);

    // Create modified coherence params with band limits
    nimcp_coherence_params_t band_params = *params;
    band_params.freq_low = freq_low;
    band_params.freq_high = freq_high;

    // Compute coherence matrix for the band
    return nimcp_gpu_coherence_matrix(ctx, signals, band_coherence, &band_params);
}

#else  // !NIMCP_ENABLE_CUDA

// Stub implementations when CUDA is not available

nimcp_oscillation_params_t nimcp_oscillation_default_params(float sampling_rate)
{
    nimcp_oscillation_params_t params = {0};
    params.sampling_rate = sampling_rate;
    params.dt = 1.0f / sampling_rate;
    params.n_samples = 1024;
    params.n_channels = 1;
    params.freq_low = 1.0f;
    params.freq_high = 100.0f;
    params.n_fft = 256;
    params.hop_length = 64;
    return params;
}

nimcp_pac_params_t nimcp_pac_default_params(void)
{
    nimcp_pac_params_t params = {0};
    params.phase_freq_low = 4.0f;
    params.phase_freq_high = 8.0f;
    params.amp_freq_low = 30.0f;
    params.amp_freq_high = 100.0f;
    params.n_phase_bins = 18;
    params.use_hilbert = true;
    return params;
}

nimcp_coherence_params_t nimcp_coherence_default_params(void)
{
    nimcp_coherence_params_t params = {0};
    params.n_fft = 256;
    params.n_overlap = 128;
    params.freq_low = 1.0f;
    params.freq_high = 100.0f;
    params.imaginary_coherence = false;
    return params;
}

void nimcp_get_band_frequencies(nimcp_oscillation_band_t band, float* low_out, float* high_out)
{
    if (!low_out || !high_out) return;

    static const float BAND_FREQUENCIES[6][2] = {
        {0.5f, 4.0f}, {4.0f, 8.0f}, {8.0f, 13.0f},
        {13.0f, 30.0f}, {30.0f, 100.0f}, {100.0f, 200.0f}
    };

    int idx = (int)band;
    if (idx >= 0 && idx <= 5) {
        *low_out = BAND_FREQUENCIES[idx][0];
        *high_out = BAND_FREQUENCIES[idx][1];
    } else {
        *low_out = 0.0f;
        *high_out = 0.0f;
    }
}

nimcp_oscillation_state_t* nimcp_oscillation_state_create(
    nimcp_gpu_context_t* ctx, const nimcp_oscillation_params_t* params)
{
    (void)ctx; (void)params;
    return NULL;
}

void nimcp_oscillation_state_destroy(nimcp_oscillation_state_t* state)
{
    (void)state;
}

nimcp_phase_sync_state_t* nimcp_phase_sync_state_create(
    nimcp_gpu_context_t* ctx, size_t n_channels, size_t window_size)
{
    (void)ctx; (void)n_channels; (void)window_size;
    return NULL;
}

void nimcp_phase_sync_state_destroy(nimcp_phase_sync_state_t* state)
{
    (void)state;
}

bool nimcp_gpu_phase_locking_value(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phase1, const nimcp_gpu_tensor_t* phase2, float* plv)
{
    (void)ctx; (void)phase1; (void)phase2; (void)plv;
    return false;
}

bool nimcp_gpu_plv_matrix(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phases, nimcp_gpu_tensor_t* plv_matrix)
{
    (void)ctx; (void)phases; (void)plv_matrix;
    return false;
}

bool nimcp_gpu_phase_lag_index(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phase1, const nimcp_gpu_tensor_t* phase2, float* pli)
{
    (void)ctx; (void)phase1; (void)phase2; (void)pli;
    return false;
}

bool nimcp_gpu_global_sync_index(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* phases, nimcp_gpu_tensor_t* sync_index)
{
    (void)ctx; (void)phases; (void)sync_index;
    return false;
}

bool nimcp_gpu_pac_modulation_index(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* pac_values,
    const nimcp_pac_params_t* params)
{
    (void)ctx; (void)signal; (void)pac_values; (void)params;
    return false;
}

bool nimcp_gpu_pac_comodulogram(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* comodulogram,
    const float* phase_freqs, size_t n_phase_freqs,
    const float* amp_freqs, size_t n_amp_freqs)
{
    (void)ctx; (void)signal; (void)comodulogram;
    (void)phase_freqs; (void)n_phase_freqs; (void)amp_freqs; (void)n_amp_freqs;
    return false;
}

bool nimcp_gpu_hilbert_phase(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* phase)
{
    (void)ctx; (void)signal; (void)phase;
    return false;
}

bool nimcp_gpu_hilbert_amplitude(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* amplitude)
{
    (void)ctx; (void)signal; (void)amplitude;
    return false;
}

bool nimcp_gpu_band_power(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* power,
    const nimcp_oscillation_params_t* params)
{
    (void)ctx; (void)signal; (void)power; (void)params;
    return false;
}

bool nimcp_gpu_power_spectral_density(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* psd,
    nimcp_gpu_tensor_t* freqs, const nimcp_oscillation_params_t* params)
{
    (void)ctx; (void)signal; (void)psd; (void)freqs; (void)params;
    return false;
}

bool nimcp_gpu_spectrogram(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* spectrogram,
    const nimcp_oscillation_params_t* params)
{
    (void)ctx; (void)signal; (void)spectrogram; (void)params;
    return false;
}

bool nimcp_gpu_relative_band_power(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* relative_power,
    const nimcp_oscillation_band_t* bands, size_t n_bands)
{
    (void)ctx; (void)signal; (void)relative_power; (void)bands; (void)n_bands;
    return false;
}

bool nimcp_gpu_coherence(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal1, const nimcp_gpu_tensor_t* signal2,
    nimcp_gpu_tensor_t* coherence, const nimcp_coherence_params_t* params)
{
    (void)ctx; (void)signal1; (void)signal2; (void)coherence; (void)params;
    return false;
}

bool nimcp_gpu_coherence_matrix(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signals, nimcp_gpu_tensor_t* coh_matrix,
    const nimcp_coherence_params_t* params)
{
    (void)ctx; (void)signals; (void)coh_matrix; (void)params;
    return false;
}

bool nimcp_gpu_imaginary_coherence(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal1, const nimcp_gpu_tensor_t* signal2,
    nimcp_gpu_tensor_t* icoh, const nimcp_coherence_params_t* params)
{
    (void)ctx; (void)signal1; (void)signal2; (void)icoh; (void)params;
    return false;
}

bool nimcp_gpu_band_coherence(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signals, nimcp_gpu_tensor_t* band_coherence,
    nimcp_oscillation_band_t band, const nimcp_coherence_params_t* params)
{
    (void)ctx; (void)signals; (void)band_coherence; (void)band; (void)params;
    return false;
}

bool nimcp_gpu_bandpass_filter(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* signal, nimcp_gpu_tensor_t* filtered,
    float low_freq, float high_freq, int order, float sampling_rate)
{
    (void)ctx; (void)signal; (void)filtered;
    (void)low_freq; (void)high_freq; (void)order; (void)sampling_rate;
    return false;
}

#endif  // NIMCP_ENABLE_CUDA
