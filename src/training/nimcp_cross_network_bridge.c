/**
 * @file nimcp_cross_network_bridge.c
 * @brief Cross-network bridge encoding/decoding with differentiable gradient flow
 *
 * WHAT: Specialized encoding/decoding for cross-network-type boundaries
 * WHY:  SNN↔ANN and LNN↔SNN boundaries need domain-specific transforms that
 *       are differentiable for end-to-end gradient flow
 * HOW:  Rate-to-spike (sigmoid threshold + SuperSpike surrogate),
 *       spike-to-rate (exponential filter), continuous-to-spike (LNN ODE→spike)
 *
 * GRADIENT FLOW:
 *   All bridges use SuperSpike surrogate gradient: σ'(x) = 1/(β|x|+1)²
 *   This enables backpropagation across the spike/rate boundary.
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <math.h>
#include <string.h>

//=============================================================================
// SuperSpike Surrogate Gradient (Zenke & Ganguli 2018)
//=============================================================================

/** SuperSpike sharpness parameter */
#define BRIDGE_SURROGATE_BETA  1.0f

/**
 * @brief SuperSpike surrogate gradient: σ'(x) = 1 / (β|x| + 1)²
 *
 * This is the derivative of the spike threshold function used to make
 * the non-differentiable spike step function differentiable for backprop.
 */
static inline float superspike_surrogate(float x) {
    float denom = BRIDGE_SURROGATE_BETA * fabsf(x) + 1.0f;
    return 1.0f / (denom * denom);
}

//=============================================================================
// Rate-to-Spike Bridge (ANN → SNN)
//=============================================================================
// Forward: Convert continuous [0,1]-normalized activations to spike probabilities
//          using a sigmoid threshold. Deterministic (soft) spikes for differentiability.
//
// Math:    spike_rate[i] = σ(gain * (rate[i] - threshold))
//          where σ is the sigmoid function, gain controls sharpness
//
// Backward: dL/drate[i] = dL/dspike[i] * gain * σ(x) * (1 - σ(x))
//           Falls back to surrogate gradient near threshold
//=============================================================================

#define RATE_TO_SPIKE_GAIN      5.0f
#define RATE_TO_SPIKE_THRESHOLD 0.5f

/**
 * @brief Forward pass: continuous rates → soft spike rates
 */
void bridge_rate_to_spike_forward(const nimcp_cross_network_bridge_t* b,
                                   const float* source_output,
                                   float* target_input) {
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        float x = RATE_TO_SPIKE_GAIN * (source_output[i] - RATE_TO_SPIKE_THRESHOLD);
        /* Sigmoid: soft spike probability */
        float spike_rate = 1.0f / (1.0f + expf(-x));
        target_input[i] = spike_rate;
    }
    /* Zero-pad if target is larger */
    for (uint32_t i = dim; i < b->target_dim; i++) {
        target_input[i] = 0.0f;
    }
}

/**
 * @brief Backward pass: dL/dspike → dL/drate using sigmoid derivative + surrogate
 */
void bridge_rate_to_spike_backward(const nimcp_cross_network_bridge_t* b,
                                    const float* dl_dtarget,
                                    float* dl_dsource) {
    if (!dl_dsource) return;
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        float x = RATE_TO_SPIKE_GAIN * (b->last_source_output[i] - RATE_TO_SPIKE_THRESHOLD);
        float sig = 1.0f / (1.0f + expf(-x));
        /* Sigmoid derivative: σ(x)·(1-σ(x)) */
        float dsig = sig * (1.0f - sig);
        /* Chain rule: dL/drate = dL/dspike × gain × dsig */
        /* Blend with surrogate gradient for numerical stability near saturation */
        float grad = RATE_TO_SPIKE_GAIN * (dsig + 0.1f * superspike_surrogate(x));
        dl_dsource[i] = dl_dtarget[i] * grad;
    }
}

//=============================================================================
// Spike-to-Rate Bridge (SNN → ANN)
//=============================================================================
// Forward: Convert spike counts/rates to continuous activations
//          using exponential smoothing with learnable time constant.
//
// Math:    rate[i] = (1 - α) * rate_prev[i] + α * spike[i]
//          where α = dt/τ controls smoothing (τ = time constant)
//          For single-step: rate[i] = spike[i] (direct passthrough)
//
// Backward: dL/dspike[i] = dL/drate[i] * α * surrogate(spike[i] - threshold)
//           Surrogate gradient makes the spike→rate boundary differentiable
//=============================================================================

#define SPIKE_TO_RATE_ALPHA     0.3f   /* Smoothing factor (higher = more responsive) */
#define SPIKE_TO_RATE_THRESHOLD 0.5f

/**
 * @brief Forward pass: spikes → continuous rates via exponential smoothing
 */
void bridge_spike_to_rate_forward(const nimcp_cross_network_bridge_t* b,
                                   const float* source_output,
                                   float* target_input) {
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        /* Exponential moving average with previous cached state */
        float prev = (b->last_target_input && i < b->target_dim) ?
                      b->last_target_input[i] : 0.0f;
        float rate = (1.0f - SPIKE_TO_RATE_ALPHA) * prev +
                     SPIKE_TO_RATE_ALPHA * source_output[i];
        /* Clamp to [0, 1] for downstream ANN */
        if (rate < 0.0f) rate = 0.0f;
        if (rate > 1.0f) rate = 1.0f;
        target_input[i] = rate;
    }
    for (uint32_t i = dim; i < b->target_dim; i++) {
        target_input[i] = 0.0f;
    }
}

/**
 * @brief Backward pass: dL/drate → dL/dspike using surrogate gradient
 */
void bridge_spike_to_rate_backward(const nimcp_cross_network_bridge_t* b,
                                    const float* dl_dtarget,
                                    float* dl_dsource) {
    if (!dl_dsource) return;
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        /* The forward function is: rate = (1-α)*prev + α*spike
         * So drate/dspike = α
         * But spike is non-differentiable, so use surrogate: */
        float spike_val = b->last_source_output[i];
        float surrogate = superspike_surrogate(spike_val - SPIKE_TO_RATE_THRESHOLD);
        dl_dsource[i] = dl_dtarget[i] * SPIKE_TO_RATE_ALPHA * surrogate;
    }
}

/* Helper: safely read cached source output */
static inline float source_output_val(const nimcp_cross_network_bridge_t* b, uint32_t i) {
    return (b->last_source_output && i < b->source_dim) ? b->last_source_output[i] : 0.0f;
}

//=============================================================================
// Continuous-to-Spike Bridge (LNN → SNN)
//=============================================================================
// Forward: Convert continuous LNN ODE state to spike probabilities.
//          LNN outputs are unbounded continuous values from ODE integration.
//          We normalize via tanh → [0,1] → soft spike via sigmoid.
//
// Math:    normalized[i] = 0.5 * (tanh(scale * x[i]) + 1)     // map to [0,1]
//          spike_prob[i] = σ(gain * (normalized[i] - threshold))
//
// Backward: Chain rule through tanh + sigmoid + surrogate.
//=============================================================================

#define CONT_TO_SPIKE_SCALE     1.0f   /* tanh input scaling */
#define CONT_TO_SPIKE_GAIN      5.0f
#define CONT_TO_SPIKE_THRESHOLD 0.5f

/**
 * @brief Forward pass: continuous ODE states → soft spike probabilities
 */
void bridge_continuous_to_spike_forward(const nimcp_cross_network_bridge_t* b,
                                         const float* source_output,
                                         float* target_input) {
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        /* Step 1: Normalize unbounded ODE output to [0,1] via tanh */
        float t = tanhf(CONT_TO_SPIKE_SCALE * source_output[i]);
        float normalized = 0.5f * (t + 1.0f);

        /* Step 2: Sigmoid threshold for soft spike */
        float x = CONT_TO_SPIKE_GAIN * (normalized - CONT_TO_SPIKE_THRESHOLD);
        float spike_prob = 1.0f / (1.0f + expf(-x));
        target_input[i] = spike_prob;
    }
    for (uint32_t i = dim; i < b->target_dim; i++) {
        target_input[i] = 0.0f;
    }
}

/**
 * @brief Backward pass: dL/dspike → dL/d(ODE_state) through tanh + sigmoid + surrogate
 */
void bridge_continuous_to_spike_backward(const nimcp_cross_network_bridge_t* b,
                                          const float* dl_dtarget,
                                          float* dl_dsource) {
    if (!dl_dsource) return;
    uint32_t dim = (b->source_dim < b->target_dim) ? b->source_dim : b->target_dim;

    for (uint32_t i = 0; i < dim; i++) {
        float src = source_output_val(b, i);

        /* Recompute forward intermediates */
        float t = tanhf(CONT_TO_SPIKE_SCALE * src);
        float normalized = 0.5f * (t + 1.0f);
        float x = CONT_TO_SPIKE_GAIN * (normalized - CONT_TO_SPIKE_THRESHOLD);
        float sig = 1.0f / (1.0f + expf(-x));

        /* Chain rule: dL/dsrc = dL/dspike × dspike/dnorm × dnorm/dsrc */
        /* dspike/dnorm = gain * sig * (1 - sig) + surrogate blend */
        float dsig = sig * (1.0f - sig);
        float dspike_dnorm = CONT_TO_SPIKE_GAIN * (dsig + 0.1f * superspike_surrogate(x));

        /* dnorm/dsrc = 0.5 * scale * (1 - tanh²(scale * src)) */
        float dtanh = 1.0f - t * t;
        float dnorm_dsrc = 0.5f * CONT_TO_SPIKE_SCALE * dtanh;

        dl_dsource[i] = dl_dtarget[i] * dspike_dnorm * dnorm_dsrc;
    }
}
