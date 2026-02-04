//=============================================================================
// nimcp_brain_state.c - Brain State Accessors and COW Handling
//=============================================================================
/**
 * @file nimcp_brain_state.c
 * @brief Brain state accessors and copy-on-write network handling
 *
 * This module contains approximately 1200 lines extracted from nimcp_brain.c:
 * - brain_get_network() - Network accessor
 * - brain_get_neuromodulator_system() - Neuromodulator accessor
 * - brain_get_sleep_system() - Sleep system accessor
 * - brain_get_theory_of_mind() - Theory of mind accessor
 * - brain_get_explanation_generator() - Explanation generator accessor
 * - ensure_writable_network() - COW network cloning
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_state.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "api/nimcp_api_exception.h"
#include "utils/logging/nimcp_logging.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_explanations.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_STATE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_state)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_state_mesh_id = 0;
static mesh_participant_registry_t* g_brain_state_mesh_registry = NULL;

nimcp_error_t brain_state_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_state_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_state", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_state";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_state_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_state_mesh_registry = registry;
    return err;
}

void brain_state_mesh_unregister(void) {
    if (g_brain_state_mesh_registry && g_brain_state_mesh_id != 0) {
        mesh_participant_unregister(g_brain_state_mesh_registry, g_brain_state_mesh_id);
        g_brain_state_mesh_id = 0;
        g_brain_state_mesh_registry = NULL;
    }
}


// NOTE: Implementation functions are currently in nimcp_brain.c
// External declarations for linking
extern adaptive_network_t brain_get_network(brain_t brain);
extern neuromodulator_system_t brain_get_neuromodulator_system(brain_t brain);
extern sleep_system_t brain_get_sleep_system(brain_t brain);
extern theory_of_mind_t brain_get_theory_of_mind(brain_t brain);
extern explanation_generator_t brain_get_explanation_generator(brain_t brain);
extern bool ensure_writable_network(brain_t brain);
