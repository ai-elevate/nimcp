//=============================================================================
// nimcp_brain_init_subsystems.h - Brain Subsystem Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_subsystems.h
 * @brief Brain subsystem initialization functions
 *
 * WHAT: All brain subsystem initialization functions (37 functions)
 * WHY:  Separates subsystem initialization from core brain creation
 * HOW:  Each function initializes a specific brain subsystem
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#ifndef NIMCP_BRAIN_INIT_SUBSYSTEMS_H
#define NIMCP_BRAIN_INIT_SUBSYSTEMS_H

#include "core/brain/nimcp_brain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Glial and biological subsystems
bool nimcp_brain_factory_init_glial_subsystem(brain_t brain);
bool nimcp_brain_factory_init_multimodal_subsystems(brain_t brain);
bool nimcp_brain_factory_init_pink_noise_subsystem(brain_t brain);
bool nimcp_brain_factory_init_neuromodulator_system(brain_t brain);
bool nimcp_brain_factory_init_spatial_neuromod_system(brain_t brain);
bool nimcp_brain_factory_init_attention_subsystem(brain_t brain);
bool nimcp_brain_factory_init_brain_regions_subsystem(brain_t brain);

// Cognitive subsystems
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
bool nimcp_brain_factory_init_global_workspace_subsystem(brain_t brain);
bool nimcp_brain_factory_init_autobiographical_memory_subsystem(brain_t brain);
bool nimcp_brain_factory_init_self_model_subsystem(brain_t brain);

// Plasticity subsystems
bool nimcp_brain_factory_init_homeostatic_plasticity_subsystem(brain_t brain);
bool nimcp_brain_factory_init_dendritic_computation_subsystem(brain_t brain);
bool nimcp_brain_factory_init_biological_predictive_subsystem(brain_t brain);

// Training subsystem
bool nimcp_brain_factory_init_training_subsystem(brain_t brain);

// Higher-order cognitive functions
bool nimcp_brain_factory_init_consolidation_subsystem(brain_t brain);
bool nimcp_brain_factory_init_curiosity_subsystem(brain_t brain);
bool nimcp_brain_factory_init_salience_subsystem(brain_t brain);
bool nimcp_brain_factory_init_introspection_subsystem(brain_t brain);
bool nimcp_brain_factory_init_connectivity_health_subsystem(brain_t brain);
bool nimcp_brain_factory_init_middleware_controller_subsystem(brain_t brain);
bool nimcp_brain_factory_init_ethics_engine_subsystem(brain_t brain);
bool nimcp_brain_factory_init_empathy_network_subsystem(brain_t brain);
bool nimcp_brain_factory_init_empathetic_response_subsystem(brain_t brain);

// Structural subsystems
bool nimcp_brain_factory_init_axon_subsystem(brain_t brain);
bool nimcp_brain_factory_init_dendrite_subsystem(brain_t brain);
bool nimcp_brain_factory_init_cortical_columns_subsystem(brain_t brain);

// FEP Orchestrator (central coordination of all FEP bridges)
bool nimcp_brain_factory_init_fep_orchestrator_subsystem(brain_t brain);

// Core Directives (ethical foundation - Asimov's Laws, Golden Rule, Harm Prevention)
bool nimcp_brain_factory_init_core_directives_subsystem(brain_t brain);

// === COORDINATOR/ORCHESTRATOR SUBSYSTEMS ===
// These provide system-wide coordination across NIMCP
// Initialization order matters due to dependencies:
// 1. Bio-Async Orchestrator (foundation for inter-module messaging)
// 2. Plasticity Coordinator (depends on bio-async)
// 3. Immune Bridge Coordinator (depends on bio-async, brain immune)
// 4. Cognitive Meta-Controller (depends on plasticity, working memory, executive)
// 5. Security-Perception Bridge (depends on BBB, immune, perception cortices)
// 6. Swarm Module Registry (depends on all above, swarm_brain)

bool nimcp_brain_factory_init_bio_async_orchestrator_subsystem(brain_t brain);
bool nimcp_brain_factory_init_plasticity_coordinator_subsystem(brain_t brain);
bool nimcp_brain_factory_init_immune_bridge_coordinator_subsystem(brain_t brain);
bool nimcp_brain_factory_init_cognitive_meta_controller_subsystem(brain_t brain);
bool nimcp_brain_factory_init_security_perception_bridge_subsystem(brain_t brain);
bool nimcp_brain_factory_init_swarm_module_registry_subsystem(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_INIT_SUBSYSTEMS_H
