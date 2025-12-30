//=============================================================================
// nimcp_brain_init_occipital.c - Occipital Cortex Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_occipital.c
 * @brief Occipital Cortex (Visual Cortex) Initialization Implementation
 *
 * WHAT: Initialization functions for Occipital Cortex (visual processing)
 * WHY:  Enable visual processing capabilities in the brain
 * HOW:  Creates occipital adapter and connects all integration bridges
 *
 * EXTRACTED FROM: Brain factory initialization
 * DATE: 2025-12-30
 *
 * @version Phase O1: Occipital Cortex Brain Integration
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_occipital.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#define LOG_MODULE "BRAIN_INIT_OCCIPITAL"

// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Occipital region includes
#include "core/brain/regions/occipital/nimcp_occipital_adapter.h"
#include "core/occipital/nimcp_occipital_substrate_bridge.h"
#include "core/occipital/nimcp_occipital_thalamic_bridge.h"
#include "core/brain/regions/occipital/nimcp_occipital_quantum_bridge.h"

#include <string.h>

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Connect occipital to bio-async messaging
 */
static bool connect_occipital_to_bio_async(brain_t brain) {
    if (!brain || !brain->occipital) return true; /* Non-fatal if not available */

    /* Register with bio-async if enabled */
    if (brain->bio_async_enabled && brain->bio_async_ctx) {
        /*
         * TODO: Register occipital message handlers
         * bio_router_register_module(router, BIO_MODULE_OCCIPITAL, brain->occipital);
         */
    }

    return true;
}

/**
 * @brief Connect occipital substrate bridge to neural substrate
 */
static bool connect_substrate_bridge(brain_t brain) {
    if (!brain || !brain->occipital_substrate_bridge) return true;

    /* Apply initial metabolic effects */
    if (occipital_substrate_bridge_update(brain->occipital_substrate_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Initial substrate bridge update failed");
    }

    return true;
}

/**
 * @brief Connect occipital thalamic bridge to thalamic router
 */
static bool connect_thalamic_bridge(brain_t brain) {
    if (!brain || !brain->occipital_thalamic_bridge) return true;

    /* Reset bridge to clean state */
    if (occipital_thalamic_bridge_reset(brain->occipital_thalamic_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Thalamic bridge reset failed");
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_occipital_subsystem(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->occipital) {
        return true;  /* Already initialized */
    }

    /* Check if occipital is enabled in config */
    /* Default to enabled for vision-capable brains */
    if (!brain->config.enable_visual_cortex &&
        !brain->config.enable_multimodal_integration) {
        brain->occipital_enabled = false;
        return true;  /* Not enabled, not an error */
    }

    /* Create occipital adapter with default configuration */
    occipital_config_t occipital_cfg = occipital_default_config();

    /* Scale configuration based on brain config */
    if (brain->config.num_inputs > 0) {
        /* Infer image dimensions from input size */
        uint32_t input_size = brain->config.num_inputs;
        uint32_t channels = 3;  /* Assume RGB */
        uint32_t pixels = input_size / channels;
        uint32_t side = (uint32_t)sqrtf((float)pixels);
        if (side > 0) {
            occipital_cfg.image_width = side;
            occipital_cfg.image_height = side;
            occipital_cfg.color_channels = channels;
        }
    }

    /* Enable training if brain training is enabled */
    occipital_cfg.enable_training = brain->enable_training_integration;
    occipital_cfg.enable_bio_async = brain->bio_async_enabled;

    brain->occipital = occipital_create(&occipital_cfg);
    if (!brain->occipital) {
        set_error("Failed to create occipital adapter");
        return false;
    }

    brain->occipital_enabled = true;
    brain->last_occipital_update_us = 0;

    /* Initialize integration bridges */
    if (!nimcp_brain_factory_init_occipital_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_occipital_thalamic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital thalamic bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_occipital_quantum_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital quantum bridge init failed (non-fatal)");
    }

    /* Connect to other subsystems */
    if (!nimcp_brain_factory_connect_occipital_to_parietal(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital-Parietal connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_occipital_to_temporal(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital-Temporal connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_occipital_to_dragonfly(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital-Dragonfly connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_occipital_to_training(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital-Training connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_occipital_to_immune(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital-Immune connection failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!connect_occipital_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Occipital bio-async connection failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Occipital cortex initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_occipital_substrate_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->occipital_substrate_bridge) {
        return true;
    }

    /* Need occipital adapter first */
    if (!brain->occipital) {
        return true;  /* Not ready yet, will be called again */
    }

    /* Get neural substrate - may be NULL in simple configurations */
    neural_substrate_t* substrate = NULL;
    /*
     * TODO: Get substrate from brain if available
     * substrate = brain->substrate;
     */

    /* Create substrate bridge with default config */
    occipital_substrate_config_t config = occipital_substrate_default_config();

    brain->occipital_substrate_bridge = occipital_substrate_bridge_create(
        brain->occipital, substrate, &config);

    if (!brain->occipital_substrate_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create occipital substrate bridge");
        return false;
    }

    /* Connect to substrate */
    connect_substrate_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Occipital substrate bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_occipital_thalamic_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->occipital_thalamic_bridge) {
        return true;
    }

    /* Need occipital adapter first */
    if (!brain->occipital) {
        return true;  /* Not ready yet */
    }

    /* Get thalamic router - may be NULL in simple configurations */
    void* router = NULL;
    /*
     * TODO: Get thalamic router from brain if available
     * router = brain->thalamic_router;
     */

    /* Create thalamic bridge with default config */
    occipital_thalamic_config_t thal_config = occipital_thalamic_default_config();

    brain->occipital_thalamic_bridge = occipital_thalamic_bridge_create(
        brain->occipital, router, &thal_config);

    if (!brain->occipital_thalamic_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create occipital thalamic bridge");
        return false;
    }

    /* Connect to thalamus */
    connect_thalamic_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Occipital thalamic bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_occipital_quantum_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->occipital_quantum_bridge) {
        return true;
    }

    /* Need occipital adapter first */
    if (!brain->occipital) {
        return true;  /* Not ready yet */
    }

    /* Check if quantum reasoning is enabled */
    if (!brain->quantum_reasoning_enabled) {
        return true;  /* Not enabled, not an error */
    }

    /* Create quantum bridge with default config */
    occipital_quantum_config_t config = occipital_quantum_default_config();

    /* Scale search depth based on visual field size */
    if (brain->config.num_inputs > 0) {
        config.visual_search_depth = brain->config.num_inputs / 3;  /* Approx pixels */
    }

    brain->occipital_quantum_bridge = occipital_quantum_bridge_create(
        brain->occipital, &config);

    if (!brain->occipital_quantum_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create occipital quantum bridge");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Occipital quantum bridge initialized");
    return true;
}

bool nimcp_brain_factory_connect_occipital_to_parietal(brain_t brain) {
    if (!brain || !brain->occipital) {
        return true;  /* Nothing to connect */
    }

    /* Check if parietal is available */
    if (!brain->parietal) {
        return true;  /* Parietal not initialized yet */
    }

    /*
     * Connect dorsal stream: V5/MT motion signals to parietal
     *
     * BIOLOGICAL: The dorsal "where" stream carries:
     * - Motion perception (V5/MT -> MST -> parietal)
     * - Spatial attention and awareness
     * - Visually-guided reaching and grasping
     */

    /* TODO: Register motion callbacks
     * occipital_set_motion_callback(brain->occipital,
     *     parietal_receive_motion, brain->parietal);
     */

    LOG_DEBUG(LOG_MODULE, "Occipital connected to parietal (dorsal stream)");
    return true;
}

bool nimcp_brain_factory_connect_occipital_to_temporal(brain_t brain) {
    if (!brain || !brain->occipital) {
        return true;  /* Nothing to connect */
    }

    /*
     * Connect ventral stream: V4 color/form signals to temporal
     *
     * BIOLOGICAL: The ventral "what" stream carries:
     * - Object identity (color, shape, texture)
     * - Face recognition (fusiform face area)
     * - Scene categorization (parahippocampal place area)
     */

    /* TODO: Register feature callbacks
     * occipital_set_feature_callback(brain->occipital,
     *     temporal_receive_features, brain->temporal);
     */

    LOG_DEBUG(LOG_MODULE, "Occipital connected to temporal (ventral stream)");
    return true;
}

bool nimcp_brain_factory_connect_occipital_to_dragonfly(brain_t brain) {
    if (!brain || !brain->occipital) {
        return true;  /* Nothing to connect */
    }

    /* Check if dragonfly is available */
    if (!brain->dragonfly_enabled || !brain->dragonfly) {
        return true;  /* Dragonfly not initialized */
    }

    /*
     * Connect visual motion to dragonfly target tracking
     *
     * BIOLOGICAL: Dragonfly visual processing:
     * - TSDN (Target-Selective Descending Neurons) receive V5/MT motion
     * - CSTMD1 (Centrifugal Small Target Motion Detector) for selective attention
     * - Enables high-speed target interception
     */

    /* TODO: Register motion callback for dragonfly
     * occipital_set_motion_callback(brain->occipital,
     *     dragonfly_receive_visual_motion, brain->dragonfly);
     */

    LOG_DEBUG(LOG_MODULE, "Occipital connected to dragonfly system");
    return true;
}

bool nimcp_brain_factory_connect_occipital_to_training(brain_t brain) {
    if (!brain || !brain->occipital) {
        return true;  /* Nothing to connect */
    }

    /* Check if training is enabled */
    if (!brain->enable_training_integration || !brain->training_ctx) {
        return true;  /* Training not enabled */
    }

    /*
     * Register occipital adapter with training context.
     * This allows visual processing learning through:
     * - Feature detector refinement (V1 Gabor filters)
     * - Contour integration learning (V2 association fields)
     * - Color constancy calibration (V4 adaptation)
     * - Motion estimation improvement (V5/MT temporal filters)
     */

    /* TODO: Register with training
     * nimcp_brain_training_register_module(brain->training_ctx,
     *     TRAIN_MODULE_OCCIPITAL, brain->occipital);
     */

    LOG_DEBUG(LOG_MODULE, "Occipital connected to training system");
    return true;
}

bool nimcp_brain_factory_connect_occipital_to_immune(brain_t brain) {
    if (!brain || !brain->occipital) {
        return true;  /* Nothing to connect */
    }

    /* Check if immune system is available */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled */
    }

    /*
     * Register for cytokine signals that affect visual processing.
     * Neuroinflammation (high IL-1beta, TNF-alpha) can cause:
     * - Reduced contrast sensitivity (V1 impairment)
     * - Color perception disturbances (V4 impairment)
     * - Motion perception deficits (V5/MT impairment)
     * - Visual fatigue and photophobia
     */

    /* TODO: Register with immune system
     * brain_immune_register_cytokine_callback(brain->immune_system,
     *     CYTOKINE_IL1B | CYTOKINE_TNF_A, occipital_inflammation_callback, brain->occipital);
     */

    LOG_DEBUG(LOG_MODULE, "Occipital connected to immune system");
    return true;
}
