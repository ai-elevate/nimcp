//=============================================================================
// nimcp_brain_init.c - Brain Initialization Dispatcher (Refactored)
//=============================================================================
/**
 * @file nimcp_brain_init.c
 * @brief Brain initialization dispatcher - delegates to modular subsystems
 *
 * WHAT: Thin dispatcher layer that includes modular initialization subsystems
 * WHY:  Refactored from monolithic 4107-line file into 5 SRP-compliant modules
 * HOW:  All functionality moved to specialized modules:
 *       - nimcp_brain_init_config.c      (configuration builders)
 *       - nimcp_brain_init_validation.c  (BBB global system)
 *       - nimcp_brain_init_core.c        (core initialization)
 *       - nimcp_brain_init_subsystems.c  (37 subsystem init functions)
 *       - nimcp_brain_init_security.c    (security subsystem)
 *
 * REFACTORING COMPLETE: 2025-12-08
 * Original file: 4107 lines → 5 modular files (3973 lines total)
 * Reduction: 134 lines (header/includes overhead eliminated)
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Module Organization
//=============================================================================
//
// This file now serves as a dispatcher/facade. All implementation has been
// moved to the following specialized modules:
//
// 1. nimcp_brain_init_config.c (280 lines)
//    - nimcp_brain_factory_get_neuron_count()
//    - nimcp_brain_factory_get_default_sparsity()
//    - nimcp_brain_factory_build_spike_params()
//    - nimcp_brain_factory_build_base_network_config()
//    - nimcp_brain_factory_build_network_config()
//    - nimcp_brain_factory_init_brain_config()
//    - nimcp_brain_factory_init_brain_stats()
//
// 2. nimcp_brain_init_validation.c (112 lines)
//    - get_global_bbb_system() [static]
//    - nimcp_bbb_release_global_system()
//    - nimcp_bbb_get_global_system()
//
// 3. nimcp_brain_init_core.c (174 lines)
//    - nimcp_brain_factory_allocate_brain()
//    - nimcp_brain_factory_create_brain_network()
//    - nimcp_brain_factory_init_output_labels()
//    - nimcp_brain_factory_init_event_bus()
//
// 4. nimcp_brain_init_subsystems.c (3124 lines)
//    - nimcp_brain_factory_init_glial_subsystem()
//    - nimcp_brain_factory_init_multimodal_subsystems()
//    - nimcp_brain_factory_init_pink_noise_subsystem()
//    - nimcp_brain_factory_init_neuromodulator_system()
//    - nimcp_brain_factory_init_spatial_neuromod_system()
//    - nimcp_brain_factory_init_attention_subsystem()
//    - nimcp_brain_factory_init_brain_regions_subsystem()
//    - nimcp_brain_factory_init_symbolic_logic_subsystem()
//    - nimcp_brain_factory_init_symbolic_reasoning_subsystem()
//    - nimcp_brain_factory_init_epistemic_subsystem()
//    - nimcp_brain_factory_init_working_memory_subsystem()
//    - nimcp_brain_factory_init_executive_subsystem()
//    - nimcp_brain_factory_init_theory_of_mind_subsystem()
//    - nimcp_brain_factory_init_natural_explanations_subsystem()
//    - nimcp_brain_factory_init_meta_learning_subsystem()
//    - nimcp_brain_factory_init_mental_health_subsystem()
//    - nimcp_brain_factory_init_predictive_subsystem()
//    - nimcp_brain_factory_init_mirror_neurons()
//    - nimcp_brain_factory_init_homeostatic_plasticity_subsystem()
//    - nimcp_brain_factory_init_dendritic_computation_subsystem()
//    - nimcp_brain_factory_init_biological_predictive_subsystem()
//    - nimcp_brain_factory_init_training_subsystem()
//    - nimcp_brain_factory_init_consolidation_subsystem()
//    - nimcp_brain_factory_init_curiosity_subsystem()
//    - nimcp_brain_factory_init_salience_subsystem()
//    - nimcp_brain_factory_init_introspection_subsystem()
//    - nimcp_brain_factory_init_connectivity_health_subsystem()
//    - nimcp_brain_factory_init_middleware_controller_subsystem()
//    - nimcp_brain_factory_init_ethics_engine_subsystem()
//    - nimcp_brain_factory_init_empathy_network_subsystem()
//    - nimcp_brain_factory_init_empathetic_response_subsystem()
//    - nimcp_brain_factory_init_autobiographical_memory_subsystem()
//    - nimcp_brain_factory_init_self_model_subsystem()
//    - nimcp_brain_factory_init_global_workspace_subsystem()
//    - nimcp_brain_factory_init_axon_subsystem()
//    - nimcp_brain_factory_init_dendrite_subsystem()
//    - nimcp_brain_factory_init_cortical_columns_subsystem()
//
// 5. nimcp_brain_init_security.c (318 lines)
//    - nimcp_brain_factory_init_security_subsystem()
//
//=============================================================================

// This file is now intentionally minimal - all functions are defined in the
// modular files listed above. This design follows the Single Responsibility
// Principle and makes the codebase more maintainable.

// No implementation needed here - all functions are in the modular files
