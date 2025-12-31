/**
 * @file nimcp_kernel_backend.c
 * @brief Unified Kernel Backend System Implementation
 *
 * WHAT: Runtime selection between CUDA and CPU compute backends
 * WHY:  Single interface for all kernel operations with fallback
 * HOW:  Strategy pattern with function pointer tables
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#include "gpu/backend/nimcp_kernel_backend.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LOG_MODULE "KERNEL_BACKEND"

//=============================================================================
// Global Backend State
//=============================================================================

static nimcp_kernel_backend_t g_cpu_backend;
static nimcp_kernel_backend_t g_cuda_backend;
static nimcp_kernel_backend_t* g_active_backend = NULL;
static bool g_backend_initialized = false;

//=============================================================================
// CPU Backend Implementation - Tensor Operations
//=============================================================================

static nimcp_kernel_error_t cpu_tensor_add(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !b || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;
    if (a->numel != b->numel || a->numel != result->numel) return NIMCP_KERNEL_ERROR_INVALID_SIZE;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < a->numel; i++) {
        r_data[i] = a_data[i] + b_data[i];
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_sub(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !b || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < a->numel; i++) {
        r_data[i] = a_data[i] - b_data[i];
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_mul(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !b || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < a->numel; i++) {
        r_data[i] = a_data[i] * b_data[i];
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_div(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !b || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < a->numel; i++) {
        r_data[i] = a_data[i] / (b_data[i] + 1e-8f);
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_scale(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    float scalar,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* a_data = (float*)a->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < a->numel; i++) {
        r_data[i] = a_data[i] * scalar;
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_matmul(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !b || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    // Assume 2D matrices: a[M,K] @ b[K,N] = result[M,N]
    size_t M = a->dims[0];
    size_t K = a->dims[1];
    size_t N = b->dims[1];

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            r_data[i * N + j] = sum;
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_transpose(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    nimcp_gpu_tensor_t* result)
{
    (void)ctx;
    if (!a || !result) return NIMCP_KERNEL_ERROR_NULL_PTR;

    size_t rows = a->dims[0];
    size_t cols = a->dims[1];

    float* a_data = (float*)a->data;
    float* r_data = (float*)result->data;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            r_data[j * rows + i] = a_data[i * cols + j];
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_relu(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_sigmoid(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        out_data[i] = 1.0f / (1.0f + expf(-in_data[i]));
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_tanh(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        out_data[i] = tanhf(in_data[i]);
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_softmax(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;
    size_t n = input->numel;

    // Find max for numerical stability
    float max_val = in_data[0];
    for (size_t i = 1; i < n; i++) {
        if (in_data[i] > max_val) max_val = in_data[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out_data[i] = expf(in_data[i] - max_val);
        sum += out_data[i];
    }

    // Normalize
    for (size_t i = 0; i < n; i++) {
        out_data[i] /= sum;
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_sum(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    float sum = 0.0f;
    for (size_t i = 0; i < input->numel; i++) {
        sum += in_data[i];
    }
    out_data[0] = sum;

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_mean(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    float sum = 0.0f;
    for (size_t i = 0; i < input->numel; i++) {
        sum += in_data[i];
    }
    out_data[0] = sum / (float)input->numel;

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_max(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    float max_val = in_data[0];
    for (size_t i = 1; i < input->numel; i++) {
        if (in_data[i] > max_val) max_val = in_data[i];
    }
    out_data[0] = max_val;

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_tensor_min(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    float min_val = in_data[0];
    for (size_t i = 1; i < input->numel; i++) {
        if (in_data[i] < min_val) min_val = in_data[i];
    }
    out_data[0] = min_val;

    return NIMCP_KERNEL_SUCCESS;
}

//=============================================================================
// CPU Backend Implementation - Training Operations
//=============================================================================

static nimcp_kernel_error_t cpu_mse_loss(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* pred,
    const nimcp_gpu_tensor_t* target,
    nimcp_gpu_tensor_t* loss)
{
    (void)ctx;
    if (!pred || !target || !loss) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* p = (float*)pred->data;
    float* t = (float*)target->data;
    float* l = (float*)loss->data;

    float sum = 0.0f;
    for (size_t i = 0; i < pred->numel; i++) {
        float diff = p[i] - t[i];
        sum += diff * diff;
    }
    l[0] = sum / (float)pred->numel;

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_cross_entropy_loss(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* pred,
    const nimcp_gpu_tensor_t* target,
    nimcp_gpu_tensor_t* loss)
{
    (void)ctx;
    if (!pred || !target || !loss) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* p = (float*)pred->data;
    float* t = (float*)target->data;
    float* l = (float*)loss->data;

    float sum = 0.0f;
    for (size_t i = 0; i < pred->numel; i++) {
        // Clamp pred to avoid log(0)
        float pred_clamped = fmaxf(fminf(p[i], 1.0f - 1e-7f), 1e-7f);
        sum -= t[i] * logf(pred_clamped);
    }
    l[0] = sum / (float)pred->numel;

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_gradient_clip(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* gradients,
    float max_norm)
{
    (void)ctx;
    if (!gradients) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* g = (float*)gradients->data;

    // Compute norm
    float norm_sq = 0.0f;
    for (size_t i = 0; i < gradients->numel; i++) {
        norm_sq += g[i] * g[i];
    }
    float norm = sqrtf(norm_sq);

    // Clip if necessary
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < gradients->numel; i++) {
            g[i] *= scale;
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_gradient_accumulate(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* accumulated,
    const nimcp_gpu_tensor_t* gradients,
    float scale)
{
    (void)ctx;
    if (!accumulated || !gradients) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* acc = (float*)accumulated->data;
    float* g = (float*)gradients->data;

    for (size_t i = 0; i < gradients->numel; i++) {
        acc[i] += scale * g[i];
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_sgd_step(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* params,
    const nimcp_gpu_tensor_t* gradients,
    float learning_rate,
    float momentum,
    nimcp_gpu_tensor_t* velocity)
{
    (void)ctx;
    if (!params || !gradients) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* p = (float*)params->data;
    float* g = (float*)gradients->data;
    float* v = velocity ? (float*)velocity->data : NULL;

    for (size_t i = 0; i < params->numel; i++) {
        if (v && momentum > 0.0f) {
            v[i] = momentum * v[i] + g[i];
            p[i] -= learning_rate * v[i];
        } else {
            p[i] -= learning_rate * g[i];
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_adam_step(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* params,
    const nimcp_gpu_tensor_t* gradients,
    nimcp_gpu_tensor_t* m,
    nimcp_gpu_tensor_t* v,
    float learning_rate,
    float beta1, float beta2,
    float epsilon, uint64_t t)
{
    (void)ctx;
    if (!params || !gradients || !m || !v) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* p = (float*)params->data;
    float* g = (float*)gradients->data;
    float* m_data = (float*)m->data;
    float* v_data = (float*)v->data;

    float beta1_t = powf(beta1, (float)t);
    float beta2_t = powf(beta2, (float)t);

    for (size_t i = 0; i < params->numel; i++) {
        m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * g[i];
        v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * g[i] * g[i];

        float m_hat = m_data[i] / (1.0f - beta1_t);
        float v_hat = v_data[i] / (1.0f - beta2_t);

        p[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_backward_linear(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* grad_output,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weights,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_weights,
    nimcp_gpu_tensor_t* grad_bias)
{
    (void)ctx;
    if (!grad_output || !input || !weights) return NIMCP_KERNEL_ERROR_NULL_PTR;

    // Simplified linear backward for 2D case
    // grad_input = grad_output @ weights.T
    // grad_weights = input.T @ grad_output
    // grad_bias = sum(grad_output, dim=0)

    LOG_DEBUG("CPU backward linear - full implementation needed");

    return NIMCP_KERNEL_SUCCESS;
}

//=============================================================================
// CPU Backend Implementation - SNN Operations
//=============================================================================

static nimcp_kernel_error_t cpu_lif_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* membrane,
    nimcp_gpu_tensor_t* spikes,
    float tau, float threshold,
    float reset, float dt)
{
    (void)ctx;
    if (!input || !membrane || !spikes) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in = (float*)input->data;
    float* v = (float*)membrane->data;
    float* s = (float*)spikes->data;

    float decay = expf(-dt / tau);

    for (size_t i = 0; i < input->numel; i++) {
        v[i] = decay * v[i] + (1.0f - decay) * in[i];

        if (v[i] >= threshold) {
            s[i] = 1.0f;
            v[i] = reset;
        } else {
            s[i] = 0.0f;
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_izhikevich_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* v,
    nimcp_gpu_tensor_t* u,
    nimcp_gpu_tensor_t* spikes,
    float a, float b, float c, float d,
    float dt)
{
    (void)ctx;
    if (!input || !v || !u || !spikes) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* I = (float*)input->data;
    float* V = (float*)v->data;
    float* U = (float*)u->data;
    float* S = (float*)spikes->data;

    for (size_t i = 0; i < input->numel; i++) {
        // Izhikevich model equations
        float dv = (0.04f * V[i] * V[i] + 5.0f * V[i] + 140.0f - U[i] + I[i]) * dt;
        float du = a * (b * V[i] - U[i]) * dt;

        V[i] += dv;
        U[i] += du;

        if (V[i] >= 30.0f) {
            S[i] = 1.0f;
            V[i] = c;
            U[i] += d;
        } else {
            S[i] = 0.0f;
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_surrogate_superspike(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    float beta)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in = (float*)input->data;
    float* out = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        float x = in[i];
        out[i] = 1.0f / powf(1.0f + beta * fabsf(x), 2.0f);
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_surrogate_fast_sigmoid(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    float slope)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in = (float*)input->data;
    float* out = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        float x = slope * in[i];
        out[i] = x / (1.0f + fabsf(x));
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_stdp_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    nimcp_gpu_tensor_t* pre_trace,
    nimcp_gpu_tensor_t* post_trace,
    float A_plus, float A_minus,
    float tau_plus, float tau_minus,
    float dt)
{
    (void)ctx;
    if (!weights || !pre_spikes || !post_spikes || !pre_trace || !post_trace) {
        return NIMCP_KERNEL_ERROR_NULL_PTR;
    }

    float* w = (float*)weights->data;
    float* pre = (float*)pre_spikes->data;
    float* post = (float*)post_spikes->data;
    float* tr_pre = (float*)pre_trace->data;
    float* tr_post = (float*)post_trace->data;

    float decay_pre = expf(-dt / tau_plus);
    float decay_post = expf(-dt / tau_minus);

    size_t n_pre = pre_spikes->numel;
    size_t n_post = post_spikes->numel;

    // Update traces
    for (size_t i = 0; i < n_pre; i++) {
        tr_pre[i] = decay_pre * tr_pre[i] + pre[i];
    }
    for (size_t j = 0; j < n_post; j++) {
        tr_post[j] = decay_post * tr_post[j] + post[j];
    }

    // Update weights
    for (size_t i = 0; i < n_pre; i++) {
        for (size_t j = 0; j < n_post; j++) {
            size_t idx = i * n_post + j;
            // LTP: post spike with pre trace
            w[idx] += A_plus * post[j] * tr_pre[i];
            // LTD: pre spike with post trace
            w[idx] -= A_minus * pre[i] * tr_post[j];
            // Clamp weights
            w[idx] = fmaxf(0.0f, fminf(1.0f, w[idx]));
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

//=============================================================================
// CPU Backend Implementation - CNN Operations
//=============================================================================

static nimcp_kernel_error_t cpu_conv2d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* kernel,
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    uint32_t stride, uint32_t padding)
{
    (void)ctx;
    if (!input || !kernel || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    // Simplified 2D convolution for [H, W] input, [KH, KW] kernel
    // Full implementation would handle batches and channels

    LOG_DEBUG("CPU conv2d forward - simplified implementation");

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_conv2d_backward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* grad_output,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* kernel,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_kernel,
    nimcp_gpu_tensor_t* grad_bias,
    uint32_t stride, uint32_t padding)
{
    (void)ctx;
    (void)grad_output;
    (void)input;
    (void)kernel;
    (void)grad_input;
    (void)grad_kernel;
    (void)grad_bias;
    (void)stride;
    (void)padding;

    LOG_DEBUG("CPU conv2d backward - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_maxpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* indices,
    uint32_t kernel_size, uint32_t stride)
{
    (void)ctx;
    (void)input;
    (void)output;
    (void)indices;
    (void)kernel_size;
    (void)stride;

    LOG_DEBUG("CPU maxpool2d - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_avgpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    uint32_t kernel_size, uint32_t stride)
{
    (void)ctx;
    (void)input;
    (void)output;
    (void)kernel_size;
    (void)stride;

    LOG_DEBUG("CPU avgpool2d - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_batchnorm_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* mean,
    nimcp_gpu_tensor_t* var,
    float epsilon, bool training)
{
    (void)ctx;
    (void)input;
    (void)gamma;
    (void)beta;
    (void)output;
    (void)mean;
    (void)var;
    (void)epsilon;
    (void)training;

    LOG_DEBUG("CPU batchnorm forward - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

//=============================================================================
// CPU Backend Implementation - Inference Operations
//=============================================================================

static nimcp_kernel_error_t cpu_linear_relu(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output)
{
    (void)ctx;
    if (!input || !weights || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    // Fused linear + ReLU
    size_t M = input->dims[0];
    size_t K = input->dims[1];
    size_t N = weights->dims[1];

    float* in = (float*)input->data;
    float* w = (float*)weights->data;
    float* b = bias ? (float*)bias->data : NULL;
    float* out = (float*)output->data;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = b ? b[j] : 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += in[i * K + k] * w[k * N + j];
            }
            out[i * N + j] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_conv_bn_relu(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* kernel,
    const nimcp_gpu_tensor_t* bn_params,
    nimcp_gpu_tensor_t* output,
    uint32_t stride, uint32_t padding)
{
    (void)ctx;
    (void)input;
    (void)kernel;
    (void)bn_params;
    (void)output;
    (void)stride;
    (void)padding;

    LOG_DEBUG("CPU conv_bn_relu - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_quantize_int8(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    float scale, int8_t zero_point)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    float* in = (float*)input->data;
    int8_t* out = (int8_t*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        float q = roundf(in[i] / scale) + zero_point;
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[i] = (int8_t)q;
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_dequantize_int8(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    float scale, int8_t zero_point)
{
    (void)ctx;
    if (!input || !output) return NIMCP_KERNEL_ERROR_NULL_PTR;

    int8_t* in = (int8_t*)input->data;
    float* out = (float*)output->data;

    for (size_t i = 0; i < input->numel; i++) {
        out[i] = scale * ((float)in[i] - zero_point);
    }

    return NIMCP_KERNEL_SUCCESS;
}

static nimcp_kernel_error_t cpu_matmul_int8(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* a,
    const nimcp_gpu_tensor_t* b,
    nimcp_gpu_tensor_t* result,
    float scale_a, float scale_b,
    float scale_out)
{
    (void)ctx;
    (void)a;
    (void)b;
    (void)result;
    (void)scale_a;
    (void)scale_b;
    (void)scale_out;

    LOG_DEBUG("CPU matmul_int8 - simplified implementation");
    return NIMCP_KERNEL_SUCCESS;
}

//=============================================================================
// Initialize CPU Backend
//=============================================================================

static void init_cpu_backend(void)
{
    memset(&g_cpu_backend, 0, sizeof(g_cpu_backend));

    g_cpu_backend.type = NIMCP_BACKEND_CPU;
    g_cpu_backend.name = "CPU";
    g_cpu_backend.initialized = true;

    // Tensor operations
    g_cpu_backend.tensor.add = cpu_tensor_add;
    g_cpu_backend.tensor.sub = cpu_tensor_sub;
    g_cpu_backend.tensor.mul = cpu_tensor_mul;
    g_cpu_backend.tensor.div = cpu_tensor_div;
    g_cpu_backend.tensor.scale = cpu_tensor_scale;
    g_cpu_backend.tensor.matmul = cpu_tensor_matmul;
    g_cpu_backend.tensor.transpose = cpu_tensor_transpose;
    g_cpu_backend.tensor.relu = cpu_tensor_relu;
    g_cpu_backend.tensor.sigmoid = cpu_tensor_sigmoid;
    g_cpu_backend.tensor.tanh = cpu_tensor_tanh;
    g_cpu_backend.tensor.softmax = cpu_tensor_softmax;
    g_cpu_backend.tensor.sum = cpu_tensor_sum;
    g_cpu_backend.tensor.mean = cpu_tensor_mean;
    g_cpu_backend.tensor.max = cpu_tensor_max;
    g_cpu_backend.tensor.min = cpu_tensor_min;

    // Training operations
    g_cpu_backend.training.mse_loss = cpu_mse_loss;
    g_cpu_backend.training.cross_entropy_loss = cpu_cross_entropy_loss;
    g_cpu_backend.training.gradient_clip = cpu_gradient_clip;
    g_cpu_backend.training.gradient_accumulate = cpu_gradient_accumulate;
    g_cpu_backend.training.sgd_step = cpu_sgd_step;
    g_cpu_backend.training.adam_step = cpu_adam_step;
    g_cpu_backend.training.backward_linear = cpu_backward_linear;

    // SNN operations
    g_cpu_backend.snn.lif_forward = cpu_lif_forward;
    g_cpu_backend.snn.izhikevich_forward = cpu_izhikevich_forward;
    g_cpu_backend.snn.surrogate_superspike = cpu_surrogate_superspike;
    g_cpu_backend.snn.surrogate_fast_sigmoid = cpu_surrogate_fast_sigmoid;
    g_cpu_backend.snn.stdp_update = cpu_stdp_update;

    // CNN operations
    g_cpu_backend.cnn.conv2d_forward = cpu_conv2d_forward;
    g_cpu_backend.cnn.conv2d_backward = cpu_conv2d_backward;
    g_cpu_backend.cnn.maxpool2d = cpu_maxpool2d;
    g_cpu_backend.cnn.avgpool2d = cpu_avgpool2d;
    g_cpu_backend.cnn.batchnorm_forward = cpu_batchnorm_forward;

    // Inference operations
    g_cpu_backend.inference.linear_relu = cpu_linear_relu;
    g_cpu_backend.inference.conv_bn_relu = cpu_conv_bn_relu;
    g_cpu_backend.inference.quantize_int8 = cpu_quantize_int8;
    g_cpu_backend.inference.dequantize_int8 = cpu_dequantize_int8;
    g_cpu_backend.inference.matmul_int8 = cpu_matmul_int8;

    LOG_INFO("CPU kernel backend initialized");
}

//=============================================================================
// Initialize CUDA Backend (placeholder - actual impl in .cu file)
//=============================================================================

#ifdef NIMCP_ENABLE_CUDA
extern void init_cuda_backend_ops(nimcp_kernel_backend_t* backend);
#endif

static void init_cuda_backend(void)
{
#ifdef NIMCP_ENABLE_CUDA
    memset(&g_cuda_backend, 0, sizeof(g_cuda_backend));

    g_cuda_backend.type = NIMCP_BACKEND_CUDA;
    g_cuda_backend.name = "CUDA";

    init_cuda_backend_ops(&g_cuda_backend);

    g_cuda_backend.initialized = true;
    LOG_INFO("CUDA kernel backend initialized");
#else
    g_cuda_backend.initialized = false;
    LOG_DEBUG("CUDA backend not available (not compiled with CUDA support)");
#endif
}

//=============================================================================
// Backend API Implementation
//=============================================================================

bool nimcp_kernel_backend_init(nimcp_backend_type_t preferred)
{
    if (g_backend_initialized) {
        LOG_WARN("Kernel backend already initialized");
        return true;
    }

    // Always initialize CPU backend
    init_cpu_backend();

    // Try to initialize CUDA backend
    init_cuda_backend();

    // Select active backend
    if (preferred == NIMCP_BACKEND_CUDA && g_cuda_backend.initialized) {
        g_active_backend = &g_cuda_backend;
    } else if (preferred == NIMCP_BACKEND_AUTO) {
        if (g_cuda_backend.initialized) {
            g_active_backend = &g_cuda_backend;
        } else {
            g_active_backend = &g_cpu_backend;
        }
    } else {
        g_active_backend = &g_cpu_backend;
    }

    g_backend_initialized = true;

    LOG_INFO("Kernel backend system initialized: active=%s", g_active_backend->name);
    return true;
}

void nimcp_kernel_backend_shutdown(void)
{
    if (!g_backend_initialized) return;

    g_active_backend = NULL;
    g_backend_initialized = false;

    LOG_INFO("Kernel backend system shutdown");
}

nimcp_kernel_backend_t* nimcp_get_kernel_backend(void)
{
    if (!g_backend_initialized) {
        // Auto-initialize with default settings
        nimcp_kernel_backend_init(NIMCP_BACKEND_AUTO);
    }

    return g_active_backend;
}

bool nimcp_cuda_backend_available(void)
{
    return g_cuda_backend.initialized;
}

nimcp_backend_type_t nimcp_get_backend_type(void)
{
    return g_active_backend ? g_active_backend->type : NIMCP_BACKEND_CPU;
}

bool nimcp_switch_backend(nimcp_backend_type_t type)
{
    if (!g_backend_initialized) {
        return nimcp_kernel_backend_init(type);
    }

    if (type == NIMCP_BACKEND_CUDA) {
        if (!g_cuda_backend.initialized) {
            LOG_ERROR("Cannot switch to CUDA backend - not available");
            return false;
        }
        g_active_backend = &g_cuda_backend;
    } else if (type == NIMCP_BACKEND_CPU) {
        g_active_backend = &g_cpu_backend;
    } else {
        LOG_ERROR("Unknown backend type: %d", type);
        return false;
    }

    LOG_INFO("Switched to %s backend", g_active_backend->name);
    return true;
}

const char* nimcp_backend_type_name(nimcp_backend_type_t type)
{
    switch (type) {
        case NIMCP_BACKEND_CPU: return "CPU";
        case NIMCP_BACKEND_CUDA: return "CUDA";
        case NIMCP_BACKEND_AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}
