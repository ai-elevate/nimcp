//=============================================================================
// nimcp_brain_init_insula.c - Insula Region Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_insula.c
 * @brief Insula Region Initialization Implementation
 *
 * WHAT: Initialization functions for Insula region (interoception, emotion, social)
 * WHY:  Enable body awareness and emotional processing in the brain
 * HOW:  Creates Insula adapter and connects all integration bridges
 *
 * EXTRACTED FROM: Brain factory initialization
 * DATE: 2025-12-30
 *
 * @version Phase I1: Insula Integration
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_insula.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#define LOG_MODULE "BRAIN_INIT_INSULA"

// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Insula region includes
#include "core/brain/regions/insula/nimcp_insula_adapter.h"
#include "core/brain/regions/insula/nimcp_insula_quantum_bridge.h"

#include <string.h>

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Connect Insula to bio-async messaging
 */
static bool connect_insula_to_bio_async(brain_t brain) {
    if (!brain || !brain->insula) return true; /* Non-fatal if not available */

    /* Register with bio-async if enabled */
    if (brain->bio_async_enabled && brain->bio_async_ctx) {
        /*
         * TODO: Register Insula message handlers
         * bio_router_register_module(router, BIO_MODULE_INSULA, brain->insula);
         */
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_insula_subsystem(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->insula) {
        return true;  /* Already initialized */
    }

    /* Check if Insula should be enabled */
    /* Default to enabled for emotionally-capable brains */
    if (!brain->config.enable_emotional_tagging &&
        !brain->config.enable_multimodal_integration) {
        brain->insula_enabled = false;
        return true;  /* Not enabled, not an error */
    }

    LOG_INFO(LOG_MODULE, "Initializing Insula subsystem");

    /* Create Insula adapter with default configuration */
    insula_config_t insula_cfg = insula_default_config();

    /* Scale configuration based on brain config */
    /* Insula sensitivity can be modulated by overall emotional sensitivity */

    brain->insula = insula_create(&insula_cfg);
    if (!brain->insula) {
        set_error("Failed to create Insula adapter");
        return false;
    }

    brain->insula_enabled = true;
    brain->last_insula_update_us = 0;

    /* Initialize quantum bridge */
    if (!nimcp_brain_factory_init_insula_quantum_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Insula quantum bridge init failed (non-fatal)");
    }

    /* Connect to other subsystems */
    if (!nimcp_brain_factory_connect_insula_to_limbic(brain)) {
        LOG_WARN(LOG_MODULE, "Insula-limbic connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_insula_to_somatosensory(brain)) {
        LOG_WARN(LOG_MODULE, "Insula-somatosensory connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_insula_to_emotional(brain)) {
        LOG_WARN(LOG_MODULE, "Insula-emotional connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_insula_to_immune(brain)) {
        LOG_WARN(LOG_MODULE, "Insula-immune connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_insula_to_theory_of_mind(brain)) {
        LOG_WARN(LOG_MODULE, "Insula-ToM connection failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!connect_insula_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Insula bio-async connection failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Insula subsystem initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_insula_quantum_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->insula_quantum_bridge) {
        return true;
    }

    /* Need Insula adapter first */
    if (!brain->insula) {
        return true;  /* Not ready yet */
    }

    /* Check if quantum reasoning is enabled */
    if (!brain->quantum_reasoning_enabled) {
        return true;  /* Not enabled, not an error */
    }

    LOG_DEBUG(LOG_MODULE, "Initializing Insula quantum bridge");

    /* Create quantum bridge with default config */
    insula_quantum_config_t config = insula_quantum_default_config();

    /* Scale based on brain configuration */
    /* More interoceptive channels for larger brains */
    /* Use num_outputs as proxy for brain size since num_hidden_neurons doesn't exist */
    if (brain->config.num_outputs > 100) {
        config.intero_channels = 32;
    }

    brain->insula_quantum_bridge = insula_quantum_bridge_create(
        brain->insula, &config);

    if (!brain->insula_quantum_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Insula quantum bridge");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Insula quantum bridge initialized");
    return true;
}

bool nimcp_brain_factory_connect_insula_to_limbic(brain_t brain) {
    if (!brain || !brain->insula) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting Insula to limbic system");

    /*
     * Connect Insula to limbic structures for emotional integration.
     *
     * BIOLOGICAL BASIS:
     * - Anterior insula → Amygdala (fear, threat valuation)
     * - Insula ↔ ACC (emotional conflict monitoring)
     * - Insula → Hippocampus (emotional memory context)
     *
     * Bidirectional connections allow:
     * - Body signals to influence emotional state (bottom-up)
     * - Emotional states to modulate interoception (top-down)
     */

    /* TODO: Connect to amygdala if available */
    /* TODO: Connect to anterior cingulate if available */
    /* TODO: Connect to hippocampal emotional memory */

    LOG_DEBUG(LOG_MODULE, "Insula connected to limbic system");
    return true;
}

bool nimcp_brain_factory_connect_insula_to_somatosensory(brain_t brain) {
    if (!brain || !brain->insula) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting Insula to somatosensory cortex");

    /*
     * Connect posterior insula to somatosensory areas for body awareness.
     *
     * PATHWAYS:
     * - S1 (Primary somatosensory) → Posterior insula (raw body signals)
     * - S2 (Secondary somatosensory) → Mid-insula (integrated body state)
     * - Posterior parietal → Insula (body schema, agency)
     *
     * SIGNALS:
     * - Tactile: Touch, pressure, texture
     * - Proprioceptive: Body position, movement
     * - Temperature: Warm/cold sensation
     * - Pain: Nociceptive signals
     */

    /* TODO: Register interoceptive callbacks with somatosensory cortex */
    /* TODO: Set up body map integration */

    LOG_DEBUG(LOG_MODULE, "Insula connected to somatosensory cortex");
    return true;
}

bool nimcp_brain_factory_connect_insula_to_emotional(brain_t brain) {
    if (!brain || !brain->insula) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting Insula to emotional system");

    /* Check if emotional system is available */
    if (!brain->emotional_system) {
        LOG_DEBUG(LOG_MODULE, "Emotional system not initialized yet");
        return true;  /* Not ready */
    }

    /*
     * Connect Insula to central emotional processing.
     *
     * INTEGRATION:
     * - Insula provides subjective feeling states
     * - Emotional system provides valence/arousal dimensions
     * - Bidirectional: body affects emotion, emotion affects body
     *
     * The insula is critical for:
     * - Emotional awareness (how do I feel?)
     * - Somatic markers (gut feelings)
     * - Emotional salience (what matters?)
     */

    /* TODO: Register bidirectional emotion callbacks */
    /* emotional_system_register_body_callback(brain->emotional_system,
     *     insula_process_emotion, brain->insula);
     * insula_set_emotion_callback(brain->insula,
     *     emotional_system_update_from_body, brain->emotional_system);
     */

    LOG_DEBUG(LOG_MODULE, "Insula connected to emotional system");
    return true;
}

bool nimcp_brain_factory_connect_insula_to_immune(brain_t brain) {
    if (!brain || !brain->insula) {
        return true;  /* Nothing to connect */
    }

    /* Check if immune system is available */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting Insula to immune system");

    /*
     * Connect Insula to brain immune system for sickness behavior.
     *
     * BIOLOGICAL BASIS:
     * When the immune system is activated, cytokines signal to the brain:
     * - IL-1β → Fever, fatigue, malaise
     * - IL-6 → Mood changes, cognitive fog
     * - TNF-α → Social withdrawal, anhedonia
     *
     * The insula receives these signals and:
     * - Updates body state (fatigue, discomfort)
     * - Modulates emotional state (decreased valence)
     * - Triggers avoidance behavior (social withdrawal)
     */

    /* TODO: Register for cytokine signals
     * brain_immune_register_cytokine_callback(brain->immune_system,
     *     CYTOKINE_IL1B | CYTOKINE_IL6 | CYTOKINE_TNF_A,
     *     insula_immune_callback, brain->insula);
     */

    LOG_DEBUG(LOG_MODULE, "Insula connected to immune system");
    return true;
}

bool nimcp_brain_factory_connect_insula_to_theory_of_mind(brain_t brain) {
    if (!brain || !brain->insula) {
        return true;  /* Nothing to connect */
    }

    /* Check if theory of mind is available */
    if (!brain->theory_of_mind) {
        return true;  /* ToM not initialized */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting Insula to Theory of Mind");

    /*
     * Connect Insula to Theory of Mind for empathy and social cognition.
     *
     * EMPATHY PATHWAY:
     * 1. Observe other's emotional expression
     * 2. Mirror neurons activate (TPJ, STS)
     * 3. Insula generates empathic resonance (shared feeling)
     * 4. ToM adds perspective-taking (cognitive empathy)
     *
     * The anterior insula is critical for:
     * - Empathic resonance (feeling what others feel)
     * - Disgust recognition in others
     * - Trust and fairness evaluation
     */

    /* TODO: Register social emotion callbacks
     * theory_of_mind_register_empathy_callback(brain->theory_of_mind,
     *     insula_process_empathy, brain->insula);
     * insula_set_social_callback(brain->insula,
     *     theory_of_mind_update_social, brain->theory_of_mind);
     */

    LOG_DEBUG(LOG_MODULE, "Insula connected to Theory of Mind");
    return true;
}
