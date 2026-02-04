//=============================================================================
// nimcp_brain_shannon.c - Shannon Information Theory Implementation
//=============================================================================
/**
 * @file nimcp_brain_shannon.c
 * @brief Implementation of Shannon information theory functions for brain
 *
 * This module provides implementations for:
 * - Shannon information flow monitoring (Phase C4)
 * - Quantum-Shannon accelerated diffusion (Phase C4.1)
 * - Cross-modal information flow tracking (Phase C4.7)
 *
 * EXTRACTED FROM: nimcp_brain.c (lines 7339-7739)
 * DATE: 2025-11-19
 *
 * DESIGN DECISIONS:
 * - No nested ifs: All validation uses early returns (guard clauses)
 * - Thread-safe: Error handling uses thread-local storage
 * - Clean separation: Independent from main brain implementation
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

// Unified memory integration
#include "utils/memory/nimcp_unified_memory.h"

#include "core/brain/information/nimcp_brain_shannon.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/strategy/nimcp_brain_strategy.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "information/nimcp_shannon.h"
#include "utils/quantum/nimcp_quantum_shannon.h"
#include "information/nimcp_cross_modal.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INFO"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_shannon)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_shannon_mesh_id = 0;
static mesh_participant_registry_t* g_brain_shannon_mesh_registry = NULL;

nimcp_error_t brain_shannon_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_shannon_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_shannon", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_shannon";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_shannon_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_shannon_mesh_registry = registry;
    return err;
}

void brain_shannon_mesh_unregister(void) {
    if (g_brain_shannon_mesh_registry && g_brain_shannon_mesh_id != 0) {
        mesh_participant_unregister(g_brain_shannon_mesh_registry, g_brain_shannon_mesh_id);
        g_brain_shannon_mesh_id = 0;
        g_brain_shannon_mesh_registry = NULL;
    }
}


//=============================================================================
// Error Handling (External Linkage)
//=============================================================================

/**
 * ARCHITECTURE NOTE:
 * Error handling has been centralized to avoid duplicate thread-local storage.
 * All brain modules use the same error state via external linkage.
 *
 * Implementation: src/core/brain/strategy/nimcp_brain_strategy.c
 */
extern void set_error(const char* format, ...);

//=============================================================================
// Phase C4: Shannon Information Theory API
//=============================================================================

/**
 * @brief Enable Shannon information flow monitoring
 *
 * WHAT: Activate real-time Shannon metrics during learning/inference
 * WHY:  Monitor channel capacity, detect bottlenecks, optimize information flow
 * HOW:  Sets enable_shannon_monitoring flag in brain
 *
 * PERFORMANCE IMPACT: ~5-10% overhead during learning/inference
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 */
void brain_enable_shannon_monitoring(brain_t brain, bool enable)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return;
    }

    brain->enable_shannon_monitoring = enable;
    brain_clear_error();
}

/**
 * @brief Get last Shannon network metrics
 *
 * WHAT: Retrieve most recent Shannon analysis results
 * WHY:  Allow external monitoring of information flow characteristics
 * HOW:  Returns copy of last_shannon_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_shannon_metrics(brain_t brain, shannon_network_metrics_t* metrics)
{
    if (!brain || !metrics) {
        set_error("Invalid parameters");
        return false;
    }

    *metrics = brain->last_shannon_metrics;
    brain_clear_error();
    return true;
}

/**
 * @brief Set custom Shannon configuration
 *
 * WHAT: Override default Shannon analysis parameters
 * WHY:  Tune accuracy vs performance tradeoff
 * HOW:  Updates brain->shannon_config
 *
 * @param brain Brain handle
 * @param config Custom Shannon configuration
 */
void brain_set_shannon_config(brain_t brain, const shannon_config_t* config)
{
    if (!brain || !config) {
        set_error("Invalid parameters");
        return;
    }

    brain->shannon_config = *config;
    brain_clear_error();
}

//=============================================================================
// Phase C4.1: Quantum-Shannon Diffusion API
//=============================================================================

/**
 * @brief Enable quantum-Shannon accelerated diffusion
 *
 * WHAT: Activate √N speedup quantum walk diffusion with Shannon monitoring
 * WHY:  Quadratic speedup for neuromodulator propagation and real-time bottleneck detection
 * HOW:  Creates quantum_shannon_diffusion_t on brain network, enables in learning/inference
 *
 * PERFORMANCE IMPACT: 2-50x speedup (topology dependent), 3× memory overhead
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 * @param source_neuron_id Initial source neuron for diffusion (0 = auto-select middle neuron)
 * @param source_information_bits Initial information content (default: 10.0 bits)
 * @return true on success, false on error
 */
bool brain_enable_quantum_shannon_diffusion(brain_t brain, bool enable, uint32_t source_neuron_id, float source_information_bits)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("Invalid brain handle");
        return false;
    }

    // Disabling
    if (!enable) {
        brain->enable_quantum_shannon_diffusion = false;
        if (brain->quantum_shannon_diffusion) {
            quantum_shannon_destroy((quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion);
            brain->quantum_shannon_diffusion = NULL;
        }
        brain_clear_error();
        return true;
    }

    // Enabling
    brain->enable_quantum_shannon_diffusion = true;

    // Create quantum-Shannon diffusion system if not already created
    if (!brain->quantum_shannon_diffusion) {
        // Get base neural network
        neural_network_t network = adaptive_network_get_base_network(brain->network);
        if (!network) {
            set_error("Failed to get base network");
            brain->enable_quantum_shannon_diffusion = false;
            return false;
        }

        // Auto-select middle neuron if source_neuron_id = 0
        uint32_t num_neurons = neural_network_get_num_neurons(network);
        if (source_neuron_id == 0) {
            source_neuron_id = num_neurons / 2;  // Middle neuron for better connectivity
        }

        // Default information if not specified
        if (source_information_bits <= 0.0F) {
            source_information_bits = 10.0F;  // 10 bits default
        }

        // Create quantum-Shannon config
        quantum_shannon_config_t config = quantum_shannon_default_config();
        config.quantum_config.hybrid_mixing = brain->quantum_shannon_mixing_ratio;
        config.quantum_config.num_steps = brain->quantum_shannon_evolution_steps;

        // Create quantum-Shannon diffusion system
        brain->quantum_shannon_diffusion = quantum_shannon_create(
            network,
            source_neuron_id,
            source_information_bits,
            &config
        );

        if (!brain->quantum_shannon_diffusion) {
            set_error("Failed to create quantum-Shannon diffusion system");
            brain->enable_quantum_shannon_diffusion = false;
            return false;
        }
    }

    brain_clear_error();
    return true;
}

/**
 * @brief Set quantum-Shannon mixing ratio
 *
 * WHAT: Control quantum vs classical diffusion blend
 * WHY:  Tune performance vs accuracy tradeoff
 * HOW:  Sets mixing_ratio [0=pure quantum, 1=pure classical]
 *
 * @param brain Brain handle
 * @param mixing_ratio Mix ratio [0.0-1.0]
 */
void brain_set_quantum_shannon_mixing(brain_t brain, float mixing_ratio)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return;
    }

    // Clamp to [0, 1]
    if (mixing_ratio < 0.0F) mixing_ratio = 0.0F;
    if (mixing_ratio > 1.0F) mixing_ratio = 1.0F;

    brain->quantum_shannon_mixing_ratio = mixing_ratio;
    brain_clear_error();
}

/**
 * @brief Set quantum-Shannon evolution steps
 *
 * WHAT: Control how many quantum steps per diffusion update
 * WHY:  More steps = better spreading, but slower
 * HOW:  Sets evolution_steps parameter
 *
 * @param brain Brain handle
 * @param steps Number of steps (10-1000, default: 100)
 */
void brain_set_quantum_shannon_steps(brain_t brain, uint32_t steps)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return;
    }

    // Clamp to reasonable range
    if (steps < 10) steps = 10;
    if (steps > 1000) steps = 1000;

    brain->quantum_shannon_evolution_steps = steps;
    brain_clear_error();
}

/**
 * @brief Get last quantum-Shannon diffusion metrics
 *
 * WHAT: Retrieve most recent Shannon metrics from quantum diffusion
 * WHY:  Monitor speedup, bottlenecks, and information flow
 * HOW:  Returns last_quantum_shannon_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_quantum_shannon_metrics(brain_t brain, shannon_diffusion_metrics_t* metrics)
{
    if (!brain || !metrics) {
        set_error("Invalid parameters");
        return false;
    }

    if (!brain->enable_quantum_shannon_diffusion) {
        set_error("Quantum-Shannon diffusion not enabled");
        return false;
    }

    *metrics = brain->last_quantum_shannon_metrics;
    brain_clear_error();
    return true;
}

/**
 * @brief Evolve quantum-Shannon diffusion manually
 *
 * WHAT: Manually trigger quantum-Shannon evolution
 * WHY:  For fine-grained control or testing
 * HOW:  Calls quantum_shannon_evolve() with configured steps
 *
 * @param brain Brain handle
 * @param num_steps Number of evolution steps (0 = use configured value)
 * @return true on success, false on error
 */
bool brain_evolve_quantum_shannon(brain_t brain, uint32_t num_steps)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return false;
    }

    if (!brain->enable_quantum_shannon_diffusion || !brain->quantum_shannon_diffusion) {
        set_error("Quantum-Shannon diffusion not enabled");
        return false;
    }

    // Use configured steps if not specified
    if (num_steps == 0) {
        num_steps = brain->quantum_shannon_evolution_steps;
    }

    // Evolve
    quantum_shannon_diffusion_t* qsd = (quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion;
    bool success = quantum_shannon_evolve(qsd, num_steps);

    if (!success) {
        set_error("Quantum-Shannon evolution failed");
        return false;
    }

    // Update metrics
    quantum_shannon_get_metrics(qsd, &brain->last_quantum_shannon_metrics);

    brain_clear_error();
    return true;
}

//=============================================================================
// Phase C4.7: Cross-Modal Information Flow API
//=============================================================================

/**
 * @brief Enable cross-modal information flow monitoring
 *
 * WHAT: Activate real-time tracking of information flow between sensory modalities
 * WHY:  Monitor multi-sensory integration, detect bottlenecks, optimize routing
 * HOW:  Sets enable_cross_modal_monitoring flag and creates routing graph
 *
 * BIOLOGICAL BASIS: Superior temporal sulcus (audiovisual), superior colliculus (multisensory)
 * PERFORMANCE IMPACT: ~2-5% overhead during multimodal processing
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 */
void brain_enable_cross_modal_monitoring(brain_t brain, bool enable)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return;
    }

    brain->enable_cross_modal_monitoring = enable;

    // Create routing graph if enabled and not already created
    if (enable && !brain->cross_modal_graph) {
        // Create graph with visual, audio, speech modalities
        const char* modalities[] = {"visual", "audio", "speech"};
        brain->cross_modal_graph = cross_modal_create_routing_graph(modalities, 3);

        if (!brain->cross_modal_graph) {
            set_error("Failed to create cross-modal routing graph");
            brain->enable_cross_modal_monitoring = false;
            return;
        }
    }

    brain_clear_error();
}

/**
 * @brief Get cross-modal routing graph
 *
 * WHAT: Retrieve current cross-modal information routing graph
 * WHY:  Allow external analysis of multi-sensory integration pathways
 * HOW:  Returns pointer to brain's cross_modal_graph
 *
 * NOTE: Graph may be NULL if cross-modal monitoring not enabled
 *
 * @param brain Brain handle
 * @return Cross-modal routing graph (NULL if not enabled)
 */
cross_modal_routing_graph_t* brain_get_cross_modal_graph(brain_t brain)
{
    if (!brain) {
        set_error("Invalid brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }

    if (!brain->enable_cross_modal_monitoring) {
        set_error("Cross-modal monitoring not enabled");
        return NULL;
    }

    brain_clear_error();
    return brain->cross_modal_graph;
}

/**
 * @brief Get last multi-modal integration metrics
 *
 * WHAT: Retrieve most recent cross-modal integration metrics
 * WHY:  Monitor synergy, redundancy, and integration efficiency
 * HOW:  Returns copy of last_cross_modal_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_cross_modal_metrics(brain_t brain, multi_modal_integration_t* metrics)
{
    if (!brain || !metrics) {
        set_error("Invalid parameters");
        return false;
    }

    if (!brain->enable_cross_modal_monitoring) {
        set_error("Cross-modal monitoring not enabled");
        return false;
    }

    *metrics = brain->last_cross_modal_metrics;
    brain_clear_error();
    return true;
}

/**
 * @brief Set cross-modal bottleneck detection threshold
 *
 * WHAT: Configure threshold for identifying cross-modal bottlenecks
 * WHY:  Tune sensitivity of bottleneck detection
 * HOW:  Sets cross_modal_bottleneck_threshold in brain
 *
 * @param brain Brain handle
 * @param threshold Efficiency threshold [0.0-1.0] (default: 0.5)
 */
void brain_set_cross_modal_threshold(brain_t brain, float threshold)
{
    if (!brain) {
        set_error("Invalid brain handle");
        return;
    }

    // Clamp to valid range
    if (threshold < 0.0F) threshold = 0.0F;
    if (threshold > 1.0F) threshold = 1.0F;

    brain->cross_modal_bottleneck_threshold = threshold;
    brain_clear_error();
}
