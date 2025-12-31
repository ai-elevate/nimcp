/**
 * @file nimcp_brain_init_collective_cognition.c
 * @brief Factory initialization for collective cognition subsystem
 *
 * WHAT: Creates distributed consciousness system for multi-brain coordination
 * WHY:  Enables inter-instance synchronization, shared intentionality, and extended mind
 * HOW:  Creates collective cognition system and optionally connects to immune system
 *
 * COMPONENTS:
 * - Hyperscanning: Real-time neural synchronization using EEG-like frequency bands
 * - Extended Mind: External tools as cognitive extensions
 * - Collective Phi: Integrated Information Theory metrics for group consciousness
 * - Shared Intentionality: Joint goals and we-mode cognition
 *
 * THEORETICAL BASIS:
 * - Integrated Information Theory (Tononi, 2004, 2014)
 * - Shared Intentionality (Tomasello, 2005, 2009)
 * - Extended Mind Thesis (Clark & Chalmers, 1998)
 * - Inter-brain synchronization (Dumas et al., 2010)
 *
 * @author NIMCP Development Team
 * @date 2025-01-01
 */

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/collective_cognition/nimcp_collective_cognition.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "BRAIN_INIT_COLLECTIVE_COGNITION"

/*=============================================================================
 * Collective Cognition Subsystem Initialization
 *===========================================================================*/

/**
 * @brief Initialize collective cognition subsystem
 *
 * WHAT: Creates distributed consciousness system for multi-brain coordination
 * WHY:  Enables inter-instance synchronization, shared intentionality, and extended mind
 * HOW:  Creates collective cognition system and optionally connects to immune system
 */
bool nimcp_brain_factory_init_collective_cognition_subsystem(brain_t brain)
{
    /* Guard clause: Validate brain pointer */
    if (!brain) {
        LOG_ERROR("Null brain in init_collective_cognition_subsystem");
        return false;
    }

    /* Initialize collective cognition fields to defaults */
    brain->collective_cognition = NULL;
    brain->collective_cognition_enabled = false;

    /* Check if collective cognition is enabled in config */
    if (!brain->config.enable_collective_cognition) {
        LOG_DEBUG("Collective cognition system disabled in config");
        return true;  /* Success - collective cognition disabled */
    }

    /* Build collective cognition configuration from brain config */
    collective_cognition_config_t cc_config;
    cc_config = collective_cognition_default_config();

    /* Override defaults with brain config values */

    /* Sizing */
    cc_config.max_instances = brain->config.collective_max_instances > 0 ?
        brain->config.collective_max_instances : COLLECTIVE_MAX_INSTANCES;

    /* Hyperscanning config */
    cc_config.hyperscanning.max_instances = cc_config.max_instances;
    cc_config.hyperscanning.sync_threshold = brain->config.collective_sync_threshold > 0.0f ?
        brain->config.collective_sync_threshold : 0.7f;
    cc_config.hyperscanning.enable_leader_detection = brain->config.collective_enable_leader_detection;
    cc_config.hyperscanning.enable_bio_async = brain->config.collective_enable_bio_async;

    /* Extended mind config */
    cc_config.extended_mind.max_extensions = brain->config.collective_max_extensions > 0 ?
        brain->config.collective_max_extensions : 32;
    cc_config.extended_mind.enable_bio_async = brain->config.collective_enable_bio_async;

    /* Phi config */
    cc_config.phi.min_instances_for_phi = 2;
    cc_config.phi.coherence_weight = brain->config.collective_coherence_weight > 0.0f ?
        brain->config.collective_coherence_weight : 0.3f;

    /* Shared intentionality config */
    cc_config.intentionality.max_shared_goals = brain->config.collective_max_shared_goals > 0 ?
        brain->config.collective_max_shared_goals : 64;
    cc_config.intentionality.max_joint_attentions = brain->config.collective_max_joint_attentions > 0 ?
        brain->config.collective_max_joint_attentions : 32;
    cc_config.intentionality.commitment_threshold = brain->config.collective_commitment_threshold > 0.0f ?
        brain->config.collective_commitment_threshold : 0.5f;
    cc_config.intentionality.we_mode_threshold = brain->config.collective_we_mode_threshold > 0.0f ?
        brain->config.collective_we_mode_threshold : 0.6f;
    cc_config.intentionality.enable_bio_async = brain->config.collective_enable_bio_async;

    /* Global settings */
    cc_config.enable_bio_async = brain->config.collective_enable_bio_async;

    /* Create collective cognition system */
    collective_cognition_t* cc = collective_cognition_create(&cc_config);
    if (!cc) {
        LOG_ERROR("Failed to create collective cognition system for brain '%s'",
                  brain->config.task_name);
        return false;
    }

    brain->collective_cognition = cc;
    brain->collective_cognition_enabled = true;

    /* TODO: Connect to bio-async when brain bio-router infrastructure is implemented */
    if (brain->config.collective_enable_bio_async && brain->bio_async_enabled) {
        /* Bio-async connection will be implemented when brain has bio_router field */
        LOG_DEBUG("Collective cognition bio-async requested but brain router not available");
    }

    /* Connect to immune system if both are enabled */
    if (brain->config.collective_enable_immune_integration && brain->immune_enabled && brain->immune_system) {
        /* Register collective cognition as a threat pattern source */
        LOG_DEBUG("Collective cognition connected to brain immune system");
        /* TODO: Create a collective_cognition_immune_bridge when needed */
    }

    LOG_INFO("Collective cognition system enabled for brain '%s' (max_instances=%u)",
             brain->config.task_name, cc_config.max_instances);

    return true;
}

