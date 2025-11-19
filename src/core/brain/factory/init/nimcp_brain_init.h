//=============================================================================
// nimcp_brain_init.h - Brain Initialization and Subsystem Setup
//=============================================================================
/**
 * @file nimcp_brain_init.h
 * @brief Brain initialization and subsystem setup functions
 *
 * WHAT: All brain structure and subsystem initialization logic
 * WHY:  Separates initialization from creation orchestration
 * HOW:  Modular initialization functions for each brain subsystem
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

#ifndef NIMCP_BRAIN_INIT_H
#define NIMCP_BRAIN_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include <stdbool.h>
#include <stdint.h>

// Forward declarations for internal types
typedef struct task_strategy task_strategy_t;

//=============================================================================
// Note: All function declarations are in the main nimcp_brain_factory.h header
// This module contains the implementations
//=============================================================================

//=============================================================================
// Subsystem Initialization Functions (31 total)
//=============================================================================

bool nimcp_brain_factory_init_glial_subsystem(brain_t brain);
bool nimcp_brain_factory_init_multimodal_subsystems(brain_t brain);
bool nimcp_brain_factory_init_pink_noise_subsystem(brain_t brain);
bool nimcp_brain_factory_init_neuromodulator_system(brain_t brain);
bool nimcp_brain_factory_init_spatial_neuromod_system(brain_t brain);
bool nimcp_brain_factory_init_attention_subsystem(brain_t brain);
bool nimcp_brain_factory_init_brain_regions_subsystem(brain_t brain);
bool nimcp_brain_factory_init_symbolic_logic_subsystem(brain_t brain);
bool nimcp_brain_factory_init_symbolic_reasoning_subsystem(brain_t brain);
bool nimcp_brain_factory_init_epistemic_subsystem(brain_t brain);
bool nimcp_brain_factory_init_working_memory_subsystem(brain_t brain);
bool nimcp_brain_factory_init_executive_subsystem(brain_t brain);
bool nimcp_brain_factory_init_theory_of_mind_subsystem(brain_t brain);
bool nimcp_brain_factory_init_natural_explanations_subsystem(brain_t brain);
bool nimcp_brain_factory_init_meta_learning_subsystem(brain_t brain);
bool nimcp_brain_factory_init_mental_health_subsystem(brain_t brain);
bool nimcp_brain_factory_init_predictive_subsystem(brain_t brain);
bool nimcp_brain_factory_init_mirror_neurons(brain_t brain);
bool nimcp_brain_factory_init_consolidation_subsystem(brain_t brain);
bool nimcp_brain_factory_init_curiosity_subsystem(brain_t brain);
bool nimcp_brain_factory_init_salience_subsystem(brain_t brain);
bool nimcp_brain_factory_init_introspection_subsystem(brain_t brain);
bool nimcp_brain_factory_init_ethics_engine_subsystem(brain_t brain);
bool nimcp_brain_factory_init_empathy_network_subsystem(brain_t brain);
bool nimcp_brain_factory_init_empathetic_response_subsystem(brain_t brain);
bool nimcp_brain_factory_init_autobiographical_memory_subsystem(brain_t brain);
bool nimcp_brain_factory_init_self_model_subsystem(brain_t brain);
bool nimcp_brain_factory_init_global_workspace_subsystem(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_INIT_H
