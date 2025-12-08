//=============================================================================
// nimcp_neuralnet_homeostasis.h - Homeostatic Plasticity
//=============================================================================
/**
 * @file nimcp_neuralnet_homeostasis.h
 * @brief Homeostatic plasticity mechanisms
 *
 * RESPONSIBILITY: Activity regulation and stability maintenance
 *
 * This module provides:
 * - Homeostatic plasticity (maintains target firing rate)
 * - Calcium dynamics for metaplasticity
 * - Adaptive threshold adjustment
 * - Network-wide homeostasis maintenance
 */

#ifndef NIMCP_NEURALNET_HOMEOSTASIS_H
#define NIMCP_NEURALNET_HOMEOSTASIS_H

#include "core/neuralnet/nimcp_neuralnet.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Public API
//=============================================================================

/**
 * @brief Apply homeostatic plasticity to a neuron
 *
 * WHY: Maintains target firing rate, prevents runaway excitation/silence
 * COMPLEXITY: O(s) where s = num_synapses
 *
 * @param network Neural network
 * @param neuron_id Neuron to regulate
 * @param timestamp Current network time
 * @return true on success, false on error
 */
NIMCP_EXPORT bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id,
                                                   uint64_t timestamp);

/**
 * @brief Maintain homeostasis across entire network
 *
 * WHY: Periodic maintenance to prevent network instability
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param network Neural network
 * @param timestamp Current network time
 */
NIMCP_EXPORT void neural_network_maintain_homeostasis(neural_network_t network, uint64_t timestamp);

/**
 * @brief Adapt neuron threshold based on activity level
 *
 * WHY: Implements adaptive threshold for spiking neurons based on current activity
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuron_id Neuron to adapt
 * @param activity_level Current activity level
 * @return true on success, false on error
 */
NIMCP_EXPORT bool neural_network_adapt_threshold_by_activity(neural_network_t network, uint32_t neuron_id,
                                                              float activity_level);

/**
 * @brief Perform periodic network maintenance
 *
 * WHY: Weight normalization, cleanup, stability checks
 * COMPLEXITY: O(n * s) where n = neurons, s = synapses
 *
 * @param network Neural network
 * @param timestamp Current network time
 */
NIMCP_EXPORT void neural_network_maintain(neural_network_t network, uint64_t timestamp);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_NEURALNET_HOMEOSTASIS_H
