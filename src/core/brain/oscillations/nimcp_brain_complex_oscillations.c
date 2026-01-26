//=============================================================================
// nimcp_brain_complex_oscillations.c - Complex Phasor Oscillation Tracking
//=============================================================================

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

#include "core/brain/oscillations/nimcp_brain_complex_oscillations.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LOG_MODULE "BRAIN_OSCILLATIONS"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_complex_oscillations module */
static nimcp_health_agent_t* g_brain_complex_oscillations_health_agent = NULL;

/**
 * @brief Set health agent for brain_complex_oscillations heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_complex_oscillations_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_complex_oscillations_health_agent = agent;
}

/** @brief Send heartbeat from brain_complex_oscillations module */
static inline void brain_complex_oscillations_heartbeat(const char* operation, float progress) {
    if (g_brain_complex_oscillations_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_complex_oscillations_health_agent, operation, progress);
    }
}


//=============================================================================
// Constants
//=============================================================================

#define DEFAULT_PHASE_UPDATE_RATE 0.1f   // radians per step
#define DEFAULT_AMPLITUDE_DECAY 0.95f    // decay factor per step
#define PAC_NUM_PHASE_BINS 18            // 20° bins for PAC

//=============================================================================
// Lifecycle Management
//=============================================================================

brain_complex_oscillation_state_t* brain_complex_oscillation_create(
    uint32_t num_neurons,
    float phase_update_rate,
    float amplitude_decay
) {
    // Guard: Validate inputs
    if (num_neurons == 0) {
        return NULL;
    }

    // Allocate state structure
    brain_complex_oscillation_state_t* state =
        (brain_complex_oscillation_state_t*)nimcp_calloc(1, sizeof(brain_complex_oscillation_state_t));
    if (!state) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "state is NULL");

        return NULL;
    }

    // Allocate phasor array
    state->neuron_phasors = (neural_phasor_t*)nimcp_calloc(num_neurons, sizeof(neural_phasor_t));
    if (!state->neuron_phasors) {
        nimcp_free(state);
        return NULL;
    }

    // Initialize state
    state->num_neurons = num_neurons;
    state->phase_update_rate = phase_update_rate > 0.0F ? phase_update_rate : DEFAULT_PHASE_UPDATE_RATE;
    state->amplitude_decay = (amplitude_decay >= 0.0F && amplitude_decay <= 1.0F) ?
                              amplitude_decay : DEFAULT_AMPLITUDE_DECAY;
    state->metrics_valid = false;
    state->last_coherence = 0.0F;
    state->last_mean_phase = 0.0F;
    state->last_phase_variance = 0.0F;

    return state;
}

void brain_complex_oscillation_destroy(
    brain_complex_oscillation_state_t* state
) {
    // Guard: NULL state
    if (!state) {
        return;
    }

    // Free phasor array
    if (state->neuron_phasors) {
        nimcp_free(state->neuron_phasors);
    }

    // Free state
    nimcp_free(state);
}

//=============================================================================
// Phasor Tracking
//=============================================================================

bool brain_complex_oscillation_update(
    brain_complex_oscillation_state_t* state,
    const float* activations
) {
    // Guard: Validate inputs
    if (!state || !activations) {
        return false;
    }

    // Update each neuron's phasor
    for (uint32_t i = 0; i < state->num_neurons; i++) {
        // Extract current phase
        float current_phase = phasor_phase(state->neuron_phasors[i]);

        // Increment phase
        float new_phase = current_phase + state->phase_update_rate;

        // Wrap phase to [-π, π]
        while (new_phase > M_PI) new_phase -= 2.0F * M_PI;
        while (new_phase < -M_PI) new_phase += 2.0F * M_PI;

        // Compute new amplitude with decay
        float new_amplitude = activations[i] * state->amplitude_decay;

        // Create new phasor
        state->neuron_phasors[i] = phasor_from_polar(new_amplitude, new_phase);
    }

    // Invalidate cached metrics
    state->metrics_valid = false;

    return true;
}

bool brain_complex_oscillation_get_phasor(
    const brain_complex_oscillation_state_t* state,
    uint32_t neuron_index,
    neural_phasor_t* phasor
) {
    // Guard: Validate inputs
    if (!state || !phasor) {
        return false;
    }

    // Guard: Check index bounds
    if (neuron_index >= state->num_neurons) {
        return false;
    }

    // Return phasor
    *phasor = state->neuron_phasors[neuron_index];
    return true;
}

bool brain_complex_oscillation_set_phasor(
    brain_complex_oscillation_state_t* state,
    uint32_t neuron_index,
    neural_phasor_t phasor
) {
    // Guard: Validate inputs
    if (!state) {
        return false;
    }

    // Guard: Check index bounds
    if (neuron_index >= state->num_neurons) {
        return false;
    }

    // Set phasor
    state->neuron_phasors[neuron_index] = phasor;

    // Invalidate cached metrics
    state->metrics_valid = false;

    return true;
}

//=============================================================================
// Phase Coherence Analysis
//=============================================================================

bool brain_complex_oscillation_compute_coherence(
    brain_complex_oscillation_state_t* state,
    phase_coherence_result_t* result
) {
    // Guard: Validate inputs
    if (!state || !result) {
        return false;
    }

    // Guard: Need at least 2 neurons
    if (state->num_neurons < 2) {
        return false;
    }

    // Compute coherence using complex math utility
    result->coherence = phasor_array_coherence(
        state->neuron_phasors,
        state->num_neurons
    );

    // Compute mean phase
    result->mean_phase = phasor_array_mean_phase(
        state->neuron_phasors,
        state->num_neurons
    );

    // Compute phase variance
    result->phase_variance = phasor_array_phase_variance(
        state->neuron_phasors,
        state->num_neurons
    );

    // Compute mean amplitude
    float total_amplitude = 0.0F;
    for (uint32_t i = 0; i < state->num_neurons; i++) {
        total_amplitude += phasor_amplitude(state->neuron_phasors[i]);
    }
    result->mean_amplitude = total_amplitude / state->num_neurons;
    result->num_neurons = state->num_neurons;

    // Cache results
    state->last_coherence = result->coherence;
    state->last_mean_phase = result->mean_phase;
    state->last_phase_variance = result->phase_variance;
    state->metrics_valid = true;

    return true;
}

bool brain_complex_oscillation_compute_coherence_subset(
    brain_complex_oscillation_state_t* state,
    const uint32_t* neuron_indices,
    uint32_t num_neurons,
    phase_coherence_result_t* result
) {
    // Guard: Validate inputs
    if (!state || !neuron_indices || !result) {
        return false;
    }

    // Guard: Need at least 2 neurons
    if (num_neurons < 2) {
        return false;
    }

    // Allocate temporary phasor array
    neural_phasor_t* subset_phasors =
        (neural_phasor_t*)nimcp_malloc(num_neurons * sizeof(neural_phasor_t));
    if (!subset_phasors) {
        return false;
    }

    // Extract subset phasors
    for (uint32_t i = 0; i < num_neurons; i++) {
        // Guard: Check index bounds
        if (neuron_indices[i] >= state->num_neurons) {
            nimcp_free(subset_phasors);
            return false;
        }
        subset_phasors[i] = state->neuron_phasors[neuron_indices[i]];
    }

    // Compute coherence
    result->coherence = phasor_array_coherence(subset_phasors, num_neurons);
    result->mean_phase = phasor_array_mean_phase(subset_phasors, num_neurons);
    result->phase_variance = phasor_array_phase_variance(subset_phasors, num_neurons);

    // Compute mean amplitude
    float total_amplitude = 0.0F;
    for (uint32_t i = 0; i < num_neurons; i++) {
        total_amplitude += phasor_amplitude(subset_phasors[i]);
    }
    result->mean_amplitude = total_amplitude / num_neurons;
    result->num_neurons = num_neurons;

    // Cleanup
    nimcp_free(subset_phasors);

    return true;
}

float brain_complex_oscillation_compute_synchrony(
    brain_complex_oscillation_state_t* state,
    const uint32_t* indices1,
    uint32_t num1,
    const uint32_t* indices2,
    uint32_t num2
) {
    // Guard: Validate inputs
    if (!state || !indices1 || !indices2) {
        return -1.0F;
    }

    // Guard: Need at least one neuron in each population
    if (num1 == 0 || num2 == 0) {
        return -1.0F;
    }

    // Allocate temporary arrays
    neural_phasor_t* phasors1 = (neural_phasor_t*)nimcp_malloc(num1 * sizeof(neural_phasor_t));
    neural_phasor_t* phasors2 = (neural_phasor_t*)nimcp_malloc(num2 * sizeof(neural_phasor_t));
    if (!phasors1 || !phasors2) {
        nimcp_free(phasors1);
        nimcp_free(phasors2);
        return -1.0F;
    }

    // Extract phasors for each population
    for (uint32_t i = 0; i < num1; i++) {
        if (indices1[i] >= state->num_neurons) {
            nimcp_free(phasors1);
            nimcp_free(phasors2);
            return -1.0F;
        }
        phasors1[i] = state->neuron_phasors[indices1[i]];
    }

    for (uint32_t i = 0; i < num2; i++) {
        if (indices2[i] >= state->num_neurons) {
            nimcp_free(phasors1);
            nimcp_free(phasors2);
            return -1.0F;
        }
        phasors2[i] = state->neuron_phasors[indices2[i]];
    }

    // Compute synchrony using complex math utility
    float synchrony = phasor_array_synchrony(phasors1, phasors2,
                                              num1 < num2 ? num1 : num2);

    // Cleanup
    nimcp_free(phasors1);
    nimcp_free(phasors2);

    return synchrony;
}

//=============================================================================
// Phase-Amplitude Coupling (PAC)
//=============================================================================

bool brain_complex_oscillation_compute_pac(
    brain_complex_oscillation_state_t* state,
    const uint32_t* phase_indices,
    uint32_t num_phase,
    const float* amplitude_values,
    uint32_t num_amplitude,
    pac_result_t* result
) {
    // Guard: Validate inputs
    if (!state || !phase_indices || !amplitude_values || !result) {
        return false;
    }

    // Guard: Need matching array sizes
    if (num_phase != num_amplitude || num_phase == 0) {
        return false;
    }

    // Allocate temporary phase phasor array
    neural_phasor_t* phase_phasors =
        (neural_phasor_t*)nimcp_malloc(num_phase * sizeof(neural_phasor_t));
    if (!phase_phasors) {
        return false;
    }

    // Extract phase phasors
    for (uint32_t i = 0; i < num_phase; i++) {
        if (phase_indices[i] >= state->num_neurons) {
            nimcp_free(phase_phasors);
            return false;
        }
        phase_phasors[i] = state->neuron_phasors[phase_indices[i]];
    }

    // Compute modulation index using complex math utility
    result->modulation_index = phasor_pac_modulation_index(
        phase_phasors,
        amplitude_values,
        num_phase
    );

    // Find preferred phase (phase with highest mean amplitude)
    float phase_bins[PAC_NUM_PHASE_BINS] = {0};
    uint32_t bin_counts[PAC_NUM_PHASE_BINS] = {0};

    for (uint32_t i = 0; i < num_phase; i++) {
        float phase = phasor_phase(phase_phasors[i]);
        // Convert phase [-π, π] to bin [0, PAC_NUM_PHASE_BINS)
        int bin = (int)((phase + M_PI) / (2.0F * M_PI) * PAC_NUM_PHASE_BINS);
        if (bin < 0) bin = 0;
        if (bin >= PAC_NUM_PHASE_BINS) bin = PAC_NUM_PHASE_BINS - 1;

        phase_bins[bin] += amplitude_values[i];
        bin_counts[bin]++;
    }

    // Find bin with highest mean amplitude
    int max_bin = 0;
    float max_mean_amp = 0.0F;
    for (int i = 0; i < PAC_NUM_PHASE_BINS; i++) {
        if (bin_counts[i] > 0) {
            float mean_amp = phase_bins[i] / bin_counts[i];
            if (mean_amp > max_mean_amp) {
                max_mean_amp = mean_amp;
                max_bin = i;
            }
        }
    }

    // Convert bin back to phase
    result->preferred_phase = -M_PI + (max_bin + 0.5F) * (2.0F * M_PI / PAC_NUM_PHASE_BINS);
    result->phase_bin_count = PAC_NUM_PHASE_BINS;

    // Compute significance (simple heuristic: modulation_index itself)
    result->significance = result->modulation_index;

    // Cleanup
    nimcp_free(phase_phasors);

    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

void brain_complex_oscillation_reset(
    brain_complex_oscillation_state_t* state
) {
    // Guard: NULL state
    if (!state) {
        return;
    }

    // Reset all phasors to zero
    memset(state->neuron_phasors, 0, state->num_neurons * sizeof(neural_phasor_t));

    // Invalidate cache
    state->metrics_valid = false;
}

void brain_complex_oscillation_invalidate_cache(
    brain_complex_oscillation_state_t* state
) {
    // Guard: NULL state
    if (!state) {
        return;
    }

    state->metrics_valid = false;
}

uint32_t brain_complex_oscillation_get_num_neurons(
    const brain_complex_oscillation_state_t* state
) {
    // Guard: NULL state
    if (!state) {
        return 0;
    }

    return state->num_neurons;
}

bool brain_complex_oscillation_is_enabled(brain_t brain) {
    // Guard: NULL brain
    if (!brain) {
        return false;
    }

    // Complex oscillations always available when compiled in
    return true;
}
