/**
 * @file nimcp_oscillation_api.c
 * @brief Complex oscillation and phasor analysis API
 *
 * This file contains complex number oscillation and phase-amplitude coupling APIs.
 * Extracted from nimcp.c (SRP refactoring - lines 1698-1969)
 */

#include "nimcp.h"
#include "api/nimcp_api_internal.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/oscillations/nimcp_brain_complex_oscillations.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "API.OSCILLATION"

// External declarations from nimcp.c
extern void set_error(const char* fmt, ...);

//=============================================================================
// Complex Number & Oscillation API Implementation
//=============================================================================

/**
 * @brief Enable or disable complex oscillation features
 */
NIMCP_EXPORT bool nimcp_enable_complex_oscillations(nimcp_brain_t brain, bool enable) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return false;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return false;
    }

    // Get brain's internal structure to access complex oscillation state
    // Note: This requires brain configuration function
    bool currently_enabled = brain_complex_oscillation_is_enabled(brain->internal_brain);

    if (enable == currently_enabled) {
        set_error("No error");
        return true;
    }

    // For actual enable/disable, we would need brain_config_t access
    set_error("Complex oscillation enable/disable requires brain reconfiguration");
    return false;
}

/**
 * @brief Check if complex oscillation features are enabled
 */
NIMCP_EXPORT bool nimcp_is_complex_oscillations_enabled(nimcp_brain_t brain) {
    if (!brain || !brain->internal_brain) {
        return false;
    }

    return brain_complex_oscillation_is_enabled(brain->internal_brain);
}

/**
 * @brief Get oscillation phasor for a specific neuron
 */
NIMCP_EXPORT nimcp_oscillation_phasor_t nimcp_get_oscillation_phasor(
    nimcp_brain_t brain,
    uint32_t neuron_id)
{
    nimcp_oscillation_phasor_t result = {0.0f, 0.0f};

    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_oscillation_phasor");
        return result;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return result;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled - call nimcp_enable_complex_oscillations first");
        return result;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return result;
    }

    // Access complex oscillation state from analyzer
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;

    // Get neuron phasor
    neural_phasor_t phasor;
    if (!brain_complex_oscillation_get_phasor(complex_state, neuron_id, &phasor)) {
        set_error("Invalid neuron ID or failed to get phasor");
        return result;
    }

    // Convert neural_phasor_t to nimcp_oscillation_phasor_t
    result.amplitude = phasor_amplitude(phasor);
    result.phase = phasor_phase(phasor);

    set_error("No error");
    return result;
}

/**
 * @brief Compute phase coherence across multiple neurons
 */
NIMCP_EXPORT float nimcp_get_phase_coherence(
    nimcp_brain_t brain,
    const uint32_t* neuron_ids,
    uint32_t count)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_phase_coherence");
        return 0.0f;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return 0.0f;
    }

    if (!neuron_ids) {
        set_error("NULL neuron_ids provided");
        return 0.0f;
    }

    if (count == 0) {
        set_error("Invalid count (0) provided");
        return 0.0f;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled");
        return 0.0f;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return 0.0f;
    }

    // Access complex oscillation state
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;

    // Compute phase coherence for neuron subset
    phase_coherence_result_t result;
    if (!brain_complex_oscillation_compute_coherence_subset(
            complex_state, neuron_ids, count, &result)) {
        set_error("Failed to compute phase coherence");
        return 0.0f;
    }

    set_error("No error");
    return result.coherence;
}

/**
 * @brief Compute phase-amplitude coupling (PAC) modulation index
 */
NIMCP_EXPORT float nimcp_get_pac_modulation(
    nimcp_brain_t brain,
    float theta_freq,
    float gamma_freq)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_pac_modulation");
        return 0.0f;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return 0.0f;
    }

    // Validate frequency ranges
    if (theta_freq < 4.0f || theta_freq > 8.0f) {
        set_error("Theta frequency should be in range 4-8 Hz");
        return 0.0f;
    }

    if (gamma_freq < 30.0f || gamma_freq > 100.0f) {
        set_error("Gamma frequency should be in range 30-100 Hz");
        return 0.0f;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled");
        return 0.0f;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return 0.0f;
    }

    // Access complex oscillation state
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;

    // Compute PAC using internal API
    uint32_t num_neurons = brain_complex_oscillation_get_num_neurons(complex_state);

    if (num_neurons == 0) {
        set_error("No neurons in complex oscillation state");
        return 0.0f;
    }

    // Allocate arrays for phase and amplitude
    uint32_t* phase_indices = (uint32_t*)nimcp_malloc(num_neurons * sizeof(uint32_t));
    float* amplitude_values = (float*)nimcp_malloc(num_neurons * sizeof(float));

    if (!phase_indices || !amplitude_values) {
        set_error("Failed to allocate memory for PAC computation");
        if (phase_indices) nimcp_free(phase_indices);
        if (amplitude_values) nimcp_free(amplitude_values);
        return 0.0f;
    }

    // Populate indices and extract amplitude values
    for (uint32_t i = 0; i < num_neurons; i++) {
        phase_indices[i] = i;

        neural_phasor_t phasor;
        if (brain_complex_oscillation_get_phasor(complex_state, i, &phasor)) {
            amplitude_values[i] = phasor_amplitude(phasor);
        } else {
            amplitude_values[i] = 0.0f;
        }
    }

    // Compute PAC
    pac_result_t pac_result;
    bool success = brain_complex_oscillation_compute_pac(
        complex_state,
        phase_indices,
        num_neurons,
        amplitude_values,
        num_neurons,
        &pac_result
    );

    // Cleanup
    nimcp_free(phase_indices);
    nimcp_free(amplitude_values);

    if (!success) {
        set_error("Failed to compute PAC modulation index");
        return 0.0f;
    }

    set_error("No error");
    return pac_result.modulation_index;
}
