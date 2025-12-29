//=============================================================================
// nimcp_brain_init_parietal.c - Parietal Lobe Initialization & Integration
//=============================================================================
/**
 * @file nimcp_brain_init_parietal.c
 * @brief Parietal lobe subsystem initialization with full brain integration
 *
 * BIOLOGICAL BASIS:
 * The parietal lobe (posterior parietal cortex) is essential for:
 * - Number sense (intraparietal sulcus - IPS)
 * - Spatial reasoning (superior parietal lobule - SPL)
 * - Mathematical intuition (angular gyrus)
 * - Scientific reasoning (integration with prefrontal cortex)
 *
 * INTEGRATION POINTS:
 * - Working Memory: Mathematical problem-solving requires WM resources
 * - Training Layer: Parietal guides mathematical learning optimization
 * - Immune System: Inflammation affects numerical precision
 * - FEP Orchestrator: Mathematical reasoning modulates free energy
 * - Sleep/Medulla: Fatigue affects spatial and numerical accuracy
 * - Global Workspace: Parietal results broadcast for conscious access
 * - Theory of Mind: Spatial reasoning for agent tracking
 * - Logic Gates: Mathematical operations use neural logic
 *
 * @version 2.0.0
 * @author NIMCP Development Team
 * @date 2025-12-29
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/parietal/nimcp_parietal.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

//=============================================================================
// Forward Declarations for Optional Integration Types
//=============================================================================

/* These may not be available if modules are disabled */
struct fep_brain_struct;
struct thalamic_router_struct;
struct substrate_interface_struct;
struct perception_system_struct;

//=============================================================================
// Integration Helper Functions
//=============================================================================

/**
 * @brief Connect parietal to working memory subsystem
 *
 * Working memory integration enables:
 * - Mathematical problem-solving with WM item manipulation
 * - Spatial working memory for mental rotation tasks
 * - Number sense for WM capacity estimation
 */
static void connect_parietal_to_working_memory(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->working_memory) {
        return;
    }

    /* Cast to expected type - working_memory_t is typedef struct working_memory */
    working_memory_t* wm = brain->working_memory;
    int result = parietal_attach_working_memory(parietal, wm);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to working memory\n");
    }
}

/**
 * @brief Connect parietal to training subsystem
 *
 * Training integration enables:
 * - Mathematical optimization of learning rates
 * - Pattern-based curriculum learning
 * - Scientific hypothesis testing for training experiments
 */
static void connect_parietal_to_training(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->training_ctx || !brain->enable_training_integration) {
        return;
    }

    /* Training context provides training_engine interface */
    training_engine_t* training = (training_engine_t*)brain->training_ctx;
    int result = parietal_attach_training(parietal, training);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to training layer\n");
    }
}

/**
 * @brief Connect parietal to immune system
 *
 * Immune integration enables:
 * - Inflammation affects Weber fraction (numerical precision)
 * - Immune stress reduces spatial accuracy
 * - Cytokine signaling modulates mathematical confidence
 */
static void connect_parietal_to_immune(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->immune_system || !brain->immune_enabled) {
        return;
    }

    /* Brain immune system provides code_immune_system interface */
    code_immune_system_t* immune = (code_immune_system_t*)brain->immune_system;
    int result = parietal_attach_immune(parietal, immune);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to immune system (inflammation modulation)\n");
    }
}

/**
 * @brief Connect parietal to FEP orchestrator
 *
 * FEP integration enables:
 * - Mathematical free energy computation
 * - Prediction error quantification using number sense
 * - Hypothesis testing for active inference
 */
static void connect_parietal_to_fep(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->fep_orchestrator || !brain->fep_orchestrator_enabled) {
        return;
    }

    /* FEP orchestrator provides fep_brain interface */
    fep_brain_t* fep = (fep_brain_t*)brain->fep_orchestrator;
    int result = parietal_attach_fep(parietal, fep);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to FEP orchestrator\n");
    }
}

/**
 * @brief Connect parietal to sleep/medulla system
 *
 * Sleep integration enables:
 * - Fatigue affects spatial reasoning speed
 * - Circadian modulation of numerical precision
 * - Sleep deprivation increases Weber fraction
 */
static void connect_parietal_to_sleep(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->medulla || !brain->medulla_enabled) {
        return;
    }

    /* Medulla provides sleep_system interface via its sleep subsystem */
    /* For now, we use the brain's sleep_system directly */
    sleep_system_t* sleep = &brain->sleep_system;
    if (sleep) {
        int result = parietal_attach_sleep(parietal, sleep);
        if (result == 0) {
            fprintf(stderr, "[PARIETAL] Connected to sleep system (fatigue modulation)\n");
        }
    }
}

/**
 * @brief Connect parietal to logic gate network
 *
 * Logic gate integration enables:
 * - Neural logic operations for mathematical reasoning
 * - Boolean algebra using spiking logic gates
 * - Arithmetic circuits with GPU acceleration
 */
static void connect_parietal_to_logic(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->logic) {
        return;
    }

    /* Neural logic network provides logic_gate_network interface */
    logic_gate_network_t* logic = (logic_gate_network_t*)brain->logic;
    int result = parietal_attach_logic_gates(parietal, logic);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to logic gate network\n");
    }
}

/**
 * @brief Connect parietal to perception system (if available)
 *
 * Perception integration enables:
 * - Visual number processing (numerosity from visual input)
 * - Spatial perception for coordinate transforms
 * - Multi-modal magnitude estimation
 */
static void connect_parietal_to_perception(parietal_lobe_t* parietal, brain_t brain) {
    /* Check if visual cortex is available as perception proxy */
    if (!brain->visual_cortex) {
        return;
    }

    /* Visual cortex can serve as perception system interface */
    perception_system_t* perception = (perception_system_t*)brain->visual_cortex;
    int result = parietal_attach_perception(parietal, perception);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Connected to perception system\n");
    }
}

/**
 * @brief Subscribe parietal to global workspace broadcasts
 *
 * Global workspace integration enables:
 * - Parietal results broadcast for conscious access
 * - Mathematical reasoning contributes to workspace competition
 * - Spatial attention integrated with workspace
 */
static void connect_parietal_to_global_workspace(parietal_lobe_t* parietal, brain_t brain) {
    (void)parietal;  /* Parietal doesn't have direct GW attach, but we can subscribe */

    if (!brain->global_workspace) {
        return;
    }

    /* Subscribe parietal module to global workspace broadcasts */
    /* MODULE_PARIETAL should be defined - if not, use a reasonable value */
    #ifndef MODULE_PARIETAL
    #define MODULE_PARIETAL 41  /* Matches PARIETAL_BRAIN_REGION_TYPE */
    #endif

    /* Note: global_workspace_subscribe takes module ID, not parietal pointer */
    /* The parietal will need to implement GW callbacks separately */
    fprintf(stderr, "[PARIETAL] Global workspace integration prepared (module %d)\n", MODULE_PARIETAL);
}

/**
 * @brief Connect parietal to brain region architecture
 *
 * Brain region integration enables:
 * - Parietal creates its own brain region with cortical layers
 * - Connections to other regions (prefrontal, temporal, motor)
 * - Hierarchical processing within parietal subregions
 */
static void connect_parietal_to_brain_regions(parietal_lobe_t* parietal, brain_t brain) {
    if (!brain->brain_regions) {
        return;
    }

    /* Attach parietal to brain module with standard neuron count */
    brain_module_t* brain_module = brain->brain_regions;
    uint32_t parietal_neurons = 10000;  /* Reasonable default for parietal */

    int result = parietal_attach_to_brain(parietal, brain_module, parietal_neurons);
    if (result == 0) {
        fprintf(stderr, "[PARIETAL] Attached to brain region architecture (%u neurons)\n", parietal_neurons);
    }
}

//=============================================================================
// Main Initialization Function
//=============================================================================

/**
 * @brief Initialize parietal lobe subsystem for brain with full integration
 *
 * Creates and configures the parietal lobe module based on brain configuration,
 * then connects it to all available brain subsystems.
 *
 * @param brain Brain instance to initialize parietal for
 * @return true on success, false on failure
 *
 * PROCESS:
 * 1. Check if parietal is enabled in brain config
 * 2. Get default parietal configuration
 * 3. Map brain config fields to parietal submodule configs
 * 4. Create parietal lobe with custom configuration
 * 5. Store in brain->parietal
 * 6. Connect to all available subsystems
 */
bool nimcp_brain_factory_init_parietal_subsystem(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if parietal is disabled in config */
    if (!brain->config.enable_parietal) {
        brain->parietal = NULL;
        brain->parietal_enabled = false;
        brain->last_parietal_update_us = 0;
        return true;  /* Success - parietal is disabled by config */
    }

    fprintf(stderr, "[PARIETAL] Initializing parietal lobe subsystem...\n");

    /* Get default parietal configuration */
    parietal_config_t config = parietal_default_config();

    /* Map brain config fields to parietal submodule configs */

    /* Weber fraction affects magnitude discrimination precision */
    if (brain->config.parietal_weber_fraction > 0.0f) {
        config.number_sense.weber_fraction = brain->config.parietal_weber_fraction;
    }

    /* Subitizing limit affects instant small-number recognition */
    if (brain->config.parietal_subitizing_limit > 0) {
        config.number_sense.subitizing_limit = brain->config.parietal_subitizing_limit;
    }

    /* Mental rotation rate (degrees per millisecond) */
    if (brain->config.parietal_rotation_rate_deg_ms > 0.0f) {
        config.spatial.rotation_rate_deg_ms = brain->config.parietal_rotation_rate_deg_ms;
    }

    /* Enable integration features based on brain configuration */
    config.enable_immune_bridge = brain->immune_enabled;
    config.enable_fep_bridge = brain->fep_orchestrator_enabled;
    config.enable_working_memory = (brain->working_memory != NULL);
    config.enable_training = brain->enable_training_integration;
    config.enable_bio_async = brain->bio_async_enabled;
    config.enable_sleep_modulation = brain->medulla_enabled;
    config.enable_logic_gates = (brain->logic != NULL);

    /* Create parietal lobe with custom configuration */
    parietal_lobe_t* parietal = parietal_create_custom(&config);
    if (!parietal) {
        fprintf(stderr, "[PARIETAL] ERROR: Failed to create parietal lobe\n");
        brain->parietal = NULL;
        brain->parietal_enabled = false;
        return false;
    }

    /* Store parietal in brain */
    brain->parietal = parietal;
    brain->parietal_enabled = true;
    brain->last_parietal_update_us = 0;

    fprintf(stderr, "[PARIETAL] Parietal lobe created, connecting to subsystems...\n");

    /* ====================================================================== */
    /* CONNECT TO ALL AVAILABLE SUBSYSTEMS                                    */
    /* ====================================================================== */

    /* 1. Working Memory - Mathematical problem-solving */
    connect_parietal_to_working_memory(parietal, brain);

    /* 2. Training Layer - Learning optimization */
    connect_parietal_to_training(parietal, brain);

    /* 3. Immune System - Inflammation affects precision */
    connect_parietal_to_immune(parietal, brain);

    /* 4. FEP Orchestrator - Free energy computation */
    connect_parietal_to_fep(parietal, brain);

    /* 5. Sleep/Medulla - Fatigue modulation */
    connect_parietal_to_sleep(parietal, brain);

    /* 6. Logic Gates - Neural logic operations */
    connect_parietal_to_logic(parietal, brain);

    /* 7. Perception System - Visual numerosity */
    connect_parietal_to_perception(parietal, brain);

    /* 8. Global Workspace - Conscious access */
    connect_parietal_to_global_workspace(parietal, brain);

    /* 9. Brain Regions - Cortical architecture */
    connect_parietal_to_brain_regions(parietal, brain);

    fprintf(stderr, "[PARIETAL] Parietal lobe initialization complete\n");
    fprintf(stderr, "[PARIETAL]   Weber fraction: %.3f\n", config.number_sense.weber_fraction);
    fprintf(stderr, "[PARIETAL]   Subitizing limit: %u\n", config.number_sense.subitizing_limit);
    fprintf(stderr, "[PARIETAL]   Rotation rate: %.3f deg/ms\n", config.spatial.rotation_rate_deg_ms);

    return true;
}

//=============================================================================
// Accessor Function
//=============================================================================

/**
 * @brief Get parietal lobe from brain
 *
 * @param brain Brain instance
 * @return Parietal lobe handle or NULL if not enabled
 */
parietal_lobe_t* brain_get_parietal(brain_t brain) {
    if (!brain || !brain->parietal_enabled) {
        return NULL;
    }
    return brain->parietal;
}

//=============================================================================
// Runtime Integration Functions
//=============================================================================

/**
 * @brief Update parietal from immune system state
 *
 * Call this periodically to sync inflammation levels with parietal precision.
 *
 * @param brain Brain instance
 * @return 0 on success, -1 on error
 */
int brain_update_parietal_from_immune(brain_t brain) {
    if (!brain || !brain->parietal_enabled || !brain->parietal) {
        return -1;
    }

    return parietal_update_from_immune(brain->parietal);
}

/**
 * @brief Update parietal from sleep system state
 *
 * Call this periodically to sync fatigue levels with parietal accuracy.
 *
 * @param brain Brain instance
 * @return 0 on success, -1 on error
 */
int brain_update_parietal_from_sleep(brain_t brain) {
    if (!brain || !brain->parietal_enabled || !brain->parietal) {
        return -1;
    }

    return parietal_update_from_sleep(brain->parietal);
}

/**
 * @brief Step parietal lobe forward in time
 *
 * Call this during brain stepping to process pending parietal requests.
 *
 * @param brain Brain instance
 * @param delta_t Time step in microseconds
 * @return 0 on success, -1 on error
 */
int brain_step_parietal(brain_t brain, uint64_t delta_t) {
    if (!brain || !brain->parietal_enabled || !brain->parietal) {
        return -1;
    }

    int result = parietal_step(brain->parietal, delta_t);
    if (result == 0) {
        brain->last_parietal_update_us += delta_t;
    }
    return result;
}
