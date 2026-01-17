//=============================================================================
// nimcp_brain_init_world_model.c - World Model Subsystem Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_world_model.c
 * @brief Implementation of World Model subsystem initialization
 * @version 1.0.0
 * @date 2026-01-16
 *
 * WHAT: Brain factory initialization function for world models
 * WHY:  Integrate generative world models into brain lifecycle
 * HOW:  Creates and configures omni and multimodal world models
 *
 * THEORETICAL BASIS:
 * - DreamerV3: RSSM world model with symlog rewards (Hafner et al.)
 * - JEPA: Predict in latent space, not pixel space (LeCun)
 * - Active Inference: Generative model for EFE minimization (Friston)
 *
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_world_model.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/omni/nimcp_omni_world_model.h"
#include "cognitive/extrapolation/nimcp_world_model_multimodal.h"
#include "cognitive/omni/nimcp_omni_active_inference.h"
#include "utils/logging/nimcp_logging.h"

/* World Model Integration Bridge Headers */
#include "cognitive/omni/bridges/nimcp_omni_wm_security_immune_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_logging_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_cognitive_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_parietal_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_hypothalamus_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_thalamic_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_substrate_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_memory_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_kg_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_tom_bridge.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_plasticity_bridge.h"

/* Forward declarations to avoid conflicting type definitions in imagination header */
struct imagination_engine;
int imagination_engine_set_world_model(struct imagination_engine* engine,
                                       struct omni_world_model* wm);

#define LOG_MODULE "BRAIN_INIT_WORLD_MODEL"

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Initialize Omni World Model
 */
static bool init_omni_world_model(struct brain_struct* brain) {
    if (!brain->config.enable_omni_world_model) {
        LOG_DEBUG(LOG_MODULE, "Omni world model disabled in config");
        return true;
    }

    LOG_DEBUG(LOG_MODULE, "Creating Omni World Model...");

    /* Build configuration from brain config */
    omni_wm_config_t config;
    if (omni_wm_get_default_config(&config) != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to get default omni WM config");
        return false;
    }

    /* Override with brain config values */
    if (brain->config.omni_wm_state_dim > 0) {
        config.state_dim = brain->config.omni_wm_state_dim;
    }
    if (brain->config.omni_wm_action_dim > 0) {
        config.action_dim = brain->config.omni_wm_action_dim;
    }
    if (brain->config.omni_wm_obs_dim > 0) {
        config.obs_dim = brain->config.omni_wm_obs_dim;
    }
    if (brain->config.omni_wm_latent_dim > 0) {
        config.latent_dim = brain->config.omni_wm_latent_dim;
    }
    if (brain->config.omni_wm_rssm_h_dim > 0) {
        config.rssm_h_dim = brain->config.omni_wm_rssm_h_dim;
    }
    if (brain->config.omni_wm_rssm_z_dim > 0) {
        config.rssm_z_dim = brain->config.omni_wm_rssm_z_dim;
    }
    if (brain->config.omni_wm_learning_rate > 0.0f) {
        config.learning_rate = brain->config.omni_wm_learning_rate;
    }
    if (brain->config.omni_wm_dream_horizon > 0) {
        config.dream_horizon = brain->config.omni_wm_dream_horizon;
    }
    config.enable_dreaming = brain->config.omni_wm_enable_dreaming;
    config.use_rssm = true;  /* Always use RSSM for DreamerV3-style dynamics */
    config.use_symlog_rewards = true;  /* DreamerV3 reward normalization */

    /* Create omni world model */
    brain->omni_world_model = omni_wm_create(&config);
    if (!brain->omni_world_model) {
        LOG_ERROR(LOG_MODULE, "Failed to create omni world model");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Omni World Model created:");
    LOG_DEBUG(LOG_MODULE, "  - State dim: %u", config.state_dim);
    LOG_DEBUG(LOG_MODULE, "  - Action dim: %u", config.action_dim);
    LOG_DEBUG(LOG_MODULE, "  - RSSM h_dim: %u, z_dim: %u", config.rssm_h_dim, config.rssm_z_dim);
    LOG_DEBUG(LOG_MODULE, "  - Dreaming: %s", config.enable_dreaming ? "enabled" : "disabled");

    return true;
}

/**
 * @brief Initialize Multimodal World Model
 */
static bool init_multimodal_world_model(struct brain_struct* brain) {
    if (!brain->config.enable_multimodal_world_model) {
        LOG_DEBUG(LOG_MODULE, "Multimodal world model disabled in config");
        return true;
    }

    LOG_DEBUG(LOG_MODULE, "Creating Multimodal World Model...");

    /* Get default configuration */
    wm_config_t config = wm_default_config();

    /* Override with brain config values */
    if (brain->config.mm_wm_latent_dim > 0) {
        config.latent_dim = brain->config.mm_wm_latent_dim;
    }
    if (brain->config.mm_wm_max_entities > 0) {
        config.max_entities = brain->config.mm_wm_max_entities;
    }
    if (brain->config.mm_wm_max_prediction_steps > 0) {
        config.max_prediction_steps = brain->config.mm_wm_max_prediction_steps;
    }
    if (brain->config.mm_wm_learning_rate > 0.0f) {
        config.learning_rate = brain->config.mm_wm_learning_rate;
    }
    config.enable_bio_async = brain->config.mm_wm_enable_bio_async;

    /* Create multimodal world model */
    brain->multimodal_world_model = wm_create(&config);
    if (!brain->multimodal_world_model) {
        LOG_ERROR(LOG_MODULE, "Failed to create multimodal world model");
        return false;
    }

    /* Initialize the world model */
    if (wm_init(brain->multimodal_world_model) != WM_OK) {
        LOG_ERROR(LOG_MODULE, "Failed to initialize multimodal world model");
        wm_destroy(brain->multimodal_world_model);
        brain->multimodal_world_model = NULL;
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Multimodal World Model created:");
    LOG_DEBUG(LOG_MODULE, "  - Latent dim: %u", config.latent_dim);
    LOG_DEBUG(LOG_MODULE, "  - Max entities: %u", config.max_entities);
    LOG_DEBUG(LOG_MODULE, "  - Max prediction steps: %u", config.max_prediction_steps);
    LOG_DEBUG(LOG_MODULE, "  - Bio-async: %s", config.enable_bio_async ? "enabled" : "disabled");

    return true;
}

//=============================================================================
// Subsystem Initialization
//=============================================================================

bool nimcp_brain_factory_init_world_model_subsystem(struct brain_struct* brain) {
    if (!brain) {
        LOG_ERROR(LOG_MODULE, "NULL brain pointer");
        return false;
    }

    /* Initialize world model fields to defaults */
    brain->omni_world_model = NULL;
    brain->multimodal_world_model = NULL;
    brain->world_model_enabled = false;
    brain->world_model_lazy_init = false;
    brain->last_world_model_update_us = 0;
    brain->world_model_update_interval_us = 10000;  /* Default: 10ms */

    /* Check if world model is enabled in config */
    if (!brain->config.enable_world_model) {
        LOG_DEBUG(LOG_MODULE, "World model disabled in config - skipping initialization");
        return true;  /* Not an error - just disabled */
    }

    /* Check for lazy initialization */
    if (brain->config.lazy_world_model_init) {
        LOG_DEBUG(LOG_MODULE, "World model set to lazy initialization - deferring");
        brain->world_model_lazy_init = true;
        return true;
    }

    LOG_INFO(LOG_MODULE, "Initializing World Model subsystem...");

    /* Initialize Omni World Model */
    if (!init_omni_world_model(brain)) {
        LOG_ERROR(LOG_MODULE, "Failed to initialize Omni World Model");
        return false;
    }

    /* Initialize Multimodal World Model */
    if (!init_multimodal_world_model(brain)) {
        LOG_ERROR(LOG_MODULE, "Failed to initialize Multimodal World Model");
        /* Cleanup omni world model on failure */
        if (brain->omni_world_model) {
            omni_wm_destroy(brain->omni_world_model);
            brain->omni_world_model = NULL;
        }
        return false;
    }

    /* Mark world model as enabled */
    brain->world_model_enabled = true;

    LOG_INFO(LOG_MODULE, "World Model subsystem initialized successfully");
    LOG_DEBUG(LOG_MODULE, "  - Omni World Model: %s",
              brain->omni_world_model ? "enabled" : "disabled");
    LOG_DEBUG(LOG_MODULE, "  - Multimodal World Model: %s",
              brain->multimodal_world_model ? "enabled" : "disabled");

    return true;
}

void nimcp_brain_factory_destroy_world_model_subsystem(struct brain_struct* brain) {
    if (!brain) {
        return;
    }

    LOG_DEBUG(LOG_MODULE, "Destroying World Model subsystem...");

    /* Destroy omni world model */
    if (brain->omni_world_model) {
        omni_wm_destroy(brain->omni_world_model);
        brain->omni_world_model = NULL;
    }

    /* Destroy multimodal world model */
    if (brain->multimodal_world_model) {
        wm_destroy(brain->multimodal_world_model);
        brain->multimodal_world_model = NULL;
    }

    brain->world_model_enabled = false;
    brain->world_model_lazy_init = false;

    LOG_DEBUG(LOG_MODULE, "World Model subsystem destroyed");
}

bool nimcp_brain_factory_wire_world_model_active_inference(struct brain_struct* brain) {
    if (!brain) {
        LOG_ERROR(LOG_MODULE, "NULL brain pointer");
        return false;
    }

    if (!brain->config.world_model_connect_active_inference) {
        LOG_DEBUG(LOG_MODULE, "World model -> active inference wiring disabled");
        return true;
    }

    if (!brain->world_model_enabled) {
        LOG_DEBUG(LOG_MODULE, "World model not enabled - skipping active inference wiring");
        return true;
    }

    /* Check if omni world model is available */
    if (!brain->omni_world_model) {
        LOG_DEBUG(LOG_MODULE, "Omni world model not initialized - skipping active inference wiring");
        return true;
    }

    /* TODO: Check if active inference system is available and wire them together
     * This requires accessing the active inference module from the brain
     * For now, we log that the connection would be made
     *
     * When active inference is initialized, it should call:
     *   omni_wm_connect_active_inference(brain->omni_world_model, brain->active_inference);
     */
    LOG_INFO(LOG_MODULE, "World model ready for active inference connection");
    LOG_DEBUG(LOG_MODULE, "  - Active inference will use world model for policy evaluation");
    LOG_DEBUG(LOG_MODULE, "  - EFE computation via simulated rollouts");

    return true;
}

bool nimcp_brain_factory_wire_world_model_imagination(struct brain_struct* brain) {
    if (!brain) {
        LOG_ERROR(LOG_MODULE, "NULL brain pointer");
        return false;
    }

    if (!brain->config.world_model_connect_imagination) {
        LOG_DEBUG(LOG_MODULE, "World model -> imagination wiring disabled");
        return true;
    }

    if (!brain->world_model_enabled) {
        LOG_DEBUG(LOG_MODULE, "World model not enabled - skipping imagination wiring");
        return true;
    }

    /* TODO: Check if imagination engine is available and wire them together
     * This requires accessing the imagination engine from the brain
     * For now, we log that the connection would be made
     *
     * When imagination engine is initialized, it should call:
     *   imagination_engine_set_world_model(brain->imagination_engine, brain->omni_world_model);
     */
    LOG_INFO(LOG_MODULE, "World model ready for imagination engine connection");
    LOG_DEBUG(LOG_MODULE, "  - Imagination will use world model for scene dynamics");
    LOG_DEBUG(LOG_MODULE, "  - Counterfactual reasoning via world model simulation");

    return true;
}

//=============================================================================
// World Model Bridge Wiring
//=============================================================================

bool nimcp_brain_factory_wire_world_model_bridges(struct brain_struct* brain) {
    if (!brain) {
        LOG_ERROR(LOG_MODULE, "NULL brain pointer");
        return false;
    }

    if (!brain->world_model_enabled) {
        LOG_DEBUG(LOG_MODULE, "World model not enabled - skipping bridge wiring");
        return true;
    }

    if (!brain->omni_world_model) {
        LOG_DEBUG(LOG_MODULE, "Omni world model not available - skipping bridge wiring");
        return true;
    }

    LOG_INFO(LOG_MODULE, "Wiring World Model integration bridges...");

    uint32_t bridges_created = 0;
    uint32_t bridges_failed = 0;

    /* 1. Security-Immune Bridge (0x0E63) */
    if (brain->config.enable_wm_security_immune_bridge) {
        omni_wm_security_immune_config_t sec_config;
        omni_wm_security_immune_bridge_default_config(&sec_config);
        brain->wm_security_immune_bridge = omni_wm_security_immune_bridge_create(&sec_config);
        if (brain->wm_security_immune_bridge) {
            if (omni_wm_security_immune_bridge_connect_world_model(brain->wm_security_immune_bridge,
                                                          brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Security-Immune Bridge (0x0E63) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Security-Immune Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Security-Immune Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 2. Logging Bridge (0x0E64) */
    if (brain->config.enable_wm_logging_bridge) {
        omni_wm_logging_bridge_config_t log_config;
        omni_wm_logging_bridge_default_config(&log_config);
        brain->wm_logging_bridge = omni_wm_logging_bridge_create(&log_config);
        if (brain->wm_logging_bridge) {
            if (omni_wm_logging_bridge_connect_world_model(brain->wm_logging_bridge,
                                                   brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Logging Bridge (0x0E64) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Logging Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Logging Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 3. Cognitive Bridge (0x0E65) */
    if (brain->config.enable_wm_cognitive_bridge) {
        omni_wm_cognitive_bridge_config_t cog_config = omni_wm_cognitive_bridge_default_config();
        brain->wm_cognitive_bridge = omni_wm_cognitive_bridge_create(&cog_config);
        if (brain->wm_cognitive_bridge) {
            if (omni_wm_cognitive_bridge_connect_world_model(brain->wm_cognitive_bridge,
                                                    brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Cognitive Bridge (0x0E65) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Cognitive Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Cognitive Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 4. Parietal Bridge (0x0E66) */
    if (brain->config.enable_wm_parietal_bridge) {
        omni_wm_parietal_bridge_config_t par_config;
        omni_wm_parietal_bridge_default_config(&par_config);
        brain->wm_parietal_bridge = omni_wm_parietal_bridge_create(&par_config);
        if (brain->wm_parietal_bridge) {
            if (omni_wm_parietal_bridge_connect_world_model(brain->wm_parietal_bridge,
                                                   brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Parietal Bridge (0x0E66) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Parietal Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Parietal Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 5. Hypothalamus Bridge (0x0E67) */
    if (brain->config.enable_wm_hypothalamus_bridge) {
        omni_wm_hypothalamus_bridge_config_t hypo_config;
        omni_wm_hypothalamus_bridge_default_config(&hypo_config);
        brain->wm_hypothalamus_bridge = omni_wm_hypothalamus_bridge_create(&hypo_config);
        if (brain->wm_hypothalamus_bridge) {
            if (omni_wm_hypothalamus_bridge_connect_world_model(brain->wm_hypothalamus_bridge,
                                                       brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Hypothalamus Bridge (0x0E67) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Hypothalamus Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Hypothalamus Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 6. Thalamic Bridge (0x0E68) */
    if (brain->config.enable_wm_thalamic_bridge) {
        omni_wm_thalamic_bridge_config_t thal_config;
        omni_wm_thalamic_bridge_default_config(&thal_config);
        brain->wm_thalamic_bridge = omni_wm_thalamic_bridge_create(&thal_config);
        if (brain->wm_thalamic_bridge) {
            if (omni_wm_thalamic_bridge_connect_world_model(brain->wm_thalamic_bridge,
                                                   brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Thalamic Bridge (0x0E68) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Thalamic Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Thalamic Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 7. Substrate Bridge (0x0E69) */
    if (brain->config.enable_wm_substrate_bridge) {
        omni_wm_substrate_bridge_config_t sub_config;
        omni_wm_substrate_bridge_default_config(&sub_config);
        brain->wm_substrate_bridge = omni_wm_substrate_bridge_create(&sub_config);
        if (brain->wm_substrate_bridge) {
            if (omni_wm_substrate_bridge_connect_world_model(brain->wm_substrate_bridge,
                                                    brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Substrate Bridge (0x0E69) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Substrate Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Substrate Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 8. Memory Bridge (0x0E6A) */
    if (brain->config.enable_wm_memory_bridge) {
        omni_wm_memory_bridge_config_t mem_config;
        omni_wm_memory_bridge_default_config(&mem_config);
        brain->wm_memory_bridge = omni_wm_memory_bridge_create(&mem_config);
        if (brain->wm_memory_bridge) {
            if (omni_wm_memory_bridge_connect_world_model(brain->wm_memory_bridge,
                                                  brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Memory Bridge (0x0E6A) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Memory Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Memory Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 9. KG Bridge (0x0E6B) */
    if (brain->config.enable_wm_kg_bridge) {
        omni_wm_kg_bridge_config_t kg_config;
        omni_wm_kg_bridge_default_config(&kg_config);
        brain->wm_kg_bridge = omni_wm_kg_bridge_create(&kg_config);
        if (brain->wm_kg_bridge) {
            if (omni_wm_kg_bridge_connect_world_model(brain->wm_kg_bridge,
                                             brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - KG Bridge (0x0E6B) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - KG Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - KG Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 10. Theory of Mind Bridge (0x0E6C) */
    if (brain->config.enable_wm_tom_bridge) {
        omni_wm_tom_bridge_config_t tom_config;
        omni_wm_tom_bridge_default_config(&tom_config);
        brain->wm_tom_bridge = omni_wm_tom_bridge_create(&tom_config);
        if (brain->wm_tom_bridge) {
            if (omni_wm_tom_bridge_connect_world_model(brain->wm_tom_bridge,
                                              brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - ToM Bridge (0x0E6C) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - ToM Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - ToM Bridge creation failed");
            bridges_failed++;
        }
    }

    /* 11. Plasticity Bridge (0x0E6D) */
    if (brain->config.enable_wm_plasticity_bridge) {
        omni_wm_plasticity_bridge_config_t plast_config;
        omni_wm_plasticity_bridge_default_config(&plast_config);
        brain->wm_plasticity_bridge = omni_wm_plasticity_bridge_create(&plast_config);
        if (brain->wm_plasticity_bridge) {
            if (omni_wm_plasticity_bridge_connect_world_model(brain->wm_plasticity_bridge,
                                                     brain->omni_world_model) == NIMCP_SUCCESS) {
                bridges_created++;
                LOG_DEBUG(LOG_MODULE, "  - Plasticity Bridge (0x0E6D) created");
            } else {
                LOG_WARN(LOG_MODULE, "  - Plasticity Bridge failed to connect");
                bridges_failed++;
            }
        } else {
            LOG_WARN(LOG_MODULE, "  - Plasticity Bridge creation failed");
            bridges_failed++;
        }
    }

    LOG_INFO(LOG_MODULE, "World Model bridge wiring complete: %u created, %u failed",
             bridges_created, bridges_failed);

    /* Return success even if some bridges failed - non-fatal */
    return true;
}

void nimcp_brain_factory_destroy_world_model_bridges(struct brain_struct* brain) {
    if (!brain) {
        return;
    }

    LOG_DEBUG(LOG_MODULE, "Destroying World Model integration bridges...");

    /* Destroy in reverse order of creation */

    /* 11. Plasticity Bridge */
    if (brain->wm_plasticity_bridge) {
        omni_wm_plasticity_bridge_destroy(brain->wm_plasticity_bridge);
        brain->wm_plasticity_bridge = NULL;
    }

    /* 10. ToM Bridge */
    if (brain->wm_tom_bridge) {
        omni_wm_tom_bridge_destroy(brain->wm_tom_bridge);
        brain->wm_tom_bridge = NULL;
    }

    /* 9. KG Bridge */
    if (brain->wm_kg_bridge) {
        omni_wm_kg_bridge_destroy(brain->wm_kg_bridge);
        brain->wm_kg_bridge = NULL;
    }

    /* 8. Memory Bridge */
    if (brain->wm_memory_bridge) {
        omni_wm_memory_bridge_destroy(brain->wm_memory_bridge);
        brain->wm_memory_bridge = NULL;
    }

    /* 7. Substrate Bridge */
    if (brain->wm_substrate_bridge) {
        omni_wm_substrate_bridge_destroy(brain->wm_substrate_bridge);
        brain->wm_substrate_bridge = NULL;
    }

    /* 6. Thalamic Bridge */
    if (brain->wm_thalamic_bridge) {
        omni_wm_thalamic_bridge_destroy(brain->wm_thalamic_bridge);
        brain->wm_thalamic_bridge = NULL;
    }

    /* 5. Hypothalamus Bridge */
    if (brain->wm_hypothalamus_bridge) {
        omni_wm_hypothalamus_bridge_destroy(brain->wm_hypothalamus_bridge);
        brain->wm_hypothalamus_bridge = NULL;
    }

    /* 4. Parietal Bridge */
    if (brain->wm_parietal_bridge) {
        omni_wm_parietal_bridge_destroy(brain->wm_parietal_bridge);
        brain->wm_parietal_bridge = NULL;
    }

    /* 3. Cognitive Bridge */
    if (brain->wm_cognitive_bridge) {
        omni_wm_cognitive_bridge_destroy(brain->wm_cognitive_bridge);
        brain->wm_cognitive_bridge = NULL;
    }

    /* 2. Logging Bridge */
    if (brain->wm_logging_bridge) {
        omni_wm_logging_bridge_destroy(brain->wm_logging_bridge);
        brain->wm_logging_bridge = NULL;
    }

    /* 1. Security-Immune Bridge */
    if (brain->wm_security_immune_bridge) {
        omni_wm_security_immune_bridge_destroy(brain->wm_security_immune_bridge);
        brain->wm_security_immune_bridge = NULL;
    }

    LOG_DEBUG(LOG_MODULE, "World Model integration bridges destroyed");
}
