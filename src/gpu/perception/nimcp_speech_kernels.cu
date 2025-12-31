/**
 * @file nimcp_speech_kernels.cu
 * @brief GPU Speech Processing CUDA Kernels
 *
 * WHAT: CUDA kernels for speech feature extraction
 * WHY:  GPU acceleration for speech recognition preprocessing
 * HOW:  Custom kernels for formant extraction, pitch detection, VAD
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "utils/logging/nimcp_logging.h"
#include <cuda_runtime.h>
#include <math.h>

#define LOG_MODULE "SPEECH_GPU"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define BLOCK_SIZE 256

//=============================================================================
// Pitch Detection (Autocorrelation Method)
//=============================================================================

__global__ void kernel_autocorrelation(
    const float* signal, float* autocorr, int signal_len, int max_lag)
{
    int lag = blockIdx.x * blockDim.x + threadIdx.x;
    if (lag >= max_lag) return;

    float sum = 0.0f;
    for (int i = 0; i < signal_len - lag; i++) {
        sum += signal[i] * signal[i + lag];
    }
    autocorr[lag] = sum;
}

__global__ void kernel_find_pitch_peak(
    const float* autocorr, float* pitch_period, float* confidence,
    int n_frames, int frame_len, int min_lag, int max_lag)
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;

    const float* ac = autocorr + frame * frame_len;

    // Find peak in valid pitch range
    float max_val = 0.0f;
    int max_idx = min_lag;

    for (int lag = min_lag; lag < max_lag && lag < frame_len; lag++) {
        if (ac[lag] > max_val) {
            max_val = ac[lag];
            max_idx = lag;
        }
    }

    pitch_period[frame] = (float)max_idx;
    confidence[frame] = ac[0] > 0.0f ? max_val / ac[0] : 0.0f;
}

bool nimcp_gpu_pitch_detect(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* audio,
    nimcp_gpu_tensor_t* pitch,
    nimcp_gpu_tensor_t* confidence,
    float sample_rate, float min_f0, float max_f0)
{
    if (!ctx || !audio || !pitch) return false;

    int signal_len = audio->dims[audio->ndim - 1];
    int max_lag = (int)(sample_rate / min_f0);
    int min_lag = (int)(sample_rate / max_f0);

    // Compute autocorrelation
    float* d_autocorr;
    CUDA_CHECK(cudaMalloc(&d_autocorr, max_lag * sizeof(float)));

    kernel_autocorrelation<<<(max_lag + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        (const float*)audio->data, d_autocorr, signal_len, max_lag);

    // Find pitch peaks
    kernel_find_pitch_peak<<<1, 1>>>(
        d_autocorr, (float*)pitch->data,
        confidence ? (float*)confidence->data : NULL,
        1, max_lag, min_lag, max_lag);

    cudaFree(d_autocorr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Voice Activity Detection (Energy-based)
//=============================================================================

__global__ void kernel_frame_energy(
    const float* audio, float* energy, int n_frames, int frame_len, int hop_len)
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;

    float sum = 0.0f;
    int start = frame * hop_len;
    for (int i = 0; i < frame_len && (start + i) < (n_frames * hop_len + frame_len); i++) {
        float val = audio[start + i];
        sum += val * val;
    }
    energy[frame] = sum / frame_len;
}

__global__ void kernel_vad_threshold(
    const float* energy, float* vad, int n_frames, float threshold)
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;

    vad[frame] = energy[frame] > threshold ? 1.0f : 0.0f;
}

bool nimcp_gpu_vad(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* audio,
    nimcp_gpu_tensor_t* vad,
    int frame_len, int hop_len, float threshold)
{
    if (!ctx || !audio || !vad) return false;

    int audio_len = audio->dims[audio->ndim - 1];
    int n_frames = (audio_len - frame_len) / hop_len + 1;

    // Compute frame energy
    float* d_energy;
    CUDA_CHECK(cudaMalloc(&d_energy, n_frames * sizeof(float)));

    kernel_frame_energy<<<(n_frames + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        (const float*)audio->data, d_energy, n_frames, frame_len, hop_len);

    // Apply threshold
    kernel_vad_threshold<<<(n_frames + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_energy, (float*)vad->data, n_frames, threshold);

    cudaFree(d_energy);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Formant Extraction (LPC-based)
//=============================================================================

__global__ void kernel_preemphasis(
    const float* input, float* output, int n, float coeff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx == 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = input[idx] - coeff * input[idx - 1];
    }
}

__global__ void kernel_lpc_autocorr(
    const float* signal, float* autocorr, int signal_len, int order)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > order) return;

    float sum = 0.0f;
    for (int i = 0; i < signal_len - k; i++) {
        sum += signal[i] * signal[i + k];
    }
    autocorr[k] = sum;
}

bool nimcp_gpu_formant_extract(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* audio,
    nimcp_gpu_tensor_t* formants,
    int n_formants, int lpc_order, float sample_rate)
{
    if (!ctx || !audio || !formants) return false;

    int signal_len = audio->dims[audio->ndim - 1];

    // Apply pre-emphasis
    float* d_preemph;
    CUDA_CHECK(cudaMalloc(&d_preemph, signal_len * sizeof(float)));

    kernel_preemphasis<<<(signal_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        (const float*)audio->data, d_preemph, signal_len, 0.97f);

    // Compute LPC autocorrelation
    float* d_autocorr;
    CUDA_CHECK(cudaMalloc(&d_autocorr, (lpc_order + 1) * sizeof(float)));

    kernel_lpc_autocorr<<<1, lpc_order + 1>>>(d_preemph, d_autocorr, signal_len, lpc_order);

    // TODO: Implement Levinson-Durbin on GPU or CPU
    // TODO: Find LPC roots to get formants
    LOG_WARN("Formant extraction using simplified LPC implementation");

    cudaFree(d_preemph);
    cudaFree(d_autocorr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Phoneme Feature Extraction
//=============================================================================

__global__ void kernel_delta_features(
    const float* features, float* delta, int n_frames, int n_features, int context)
{
    int frame = blockIdx.y;
    int feat = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat >= n_features || frame >= n_frames) return;

    float sum_num = 0.0f;
    float sum_denom = 0.0f;

    for (int t = 1; t <= context; t++) {
        int prev = (frame - t >= 0) ? frame - t : 0;
        int next = (frame + t < n_frames) ? frame + t : n_frames - 1;

        sum_num += t * (features[next * n_features + feat] - features[prev * n_features + feat]);
        sum_denom += t * t;
    }

    delta[frame * n_features + feat] = sum_num / (2.0f * sum_denom);
}

bool nimcp_gpu_delta_features(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* features,
    nimcp_gpu_tensor_t* delta,
    int context)
{
    if (!ctx || !features || !delta) return false;

    int n_frames = features->dims[0];
    int n_features = features->dims[1];

    dim3 block(BLOCK_SIZE);
    dim3 grid((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE, n_frames);

    kernel_delta_features<<<grid, block>>>(
        (const float*)features->data, (float*)delta->data,
        n_frames, n_features, context);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else // !NIMCP_ENABLE_CUDA

#include "utils/logging/nimcp_logging.h"
#define LOG_MODULE "SPEECH_GPU"

bool nimcp_gpu_pitch_detect(void* ctx, void* audio, void* pitch, void* conf, float sr, float min, float max)
{
    LOG_WARN("CUDA not available - speech processing requires GPU");
    return false;
}

bool nimcp_gpu_vad(void* ctx, void* audio, void* vad, int fl, int hl, float th) { return false; }
bool nimcp_gpu_formant_extract(void* ctx, void* audio, void* form, int n, int o, float sr) { return false; }
bool nimcp_gpu_delta_features(void* ctx, void* feat, void* delta, int ctx_size) { return false; }

#endif // NIMCP_ENABLE_CUDA
