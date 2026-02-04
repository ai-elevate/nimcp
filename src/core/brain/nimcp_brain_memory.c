//=============================================================================
// nimcp_brain_memory.c - Working Memory State Persistence
//=============================================================================
/**
 * @file nimcp_brain_memory.c
 * @brief Working memory save/load operations
 *
 * This module contains approximately 300 lines extracted from nimcp_brain.c:
 * - save_working_memory_state() - Serialize working memory to file
 * - load_working_memory_state() - Deserialize working memory from file
 * - load_working_memory_item() - Load individual memory items
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_memory.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "api/nimcp_api_exception.h"
#include "utils/logging/nimcp_logging.h"
#include "cognitive/nimcp_working_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>

#define LOG_MODULE "BRAIN_MEM"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_memory)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_memory_mesh_id = 0;
static mesh_participant_registry_t* g_brain_memory_mesh_registry = NULL;

nimcp_error_t brain_memory_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_memory_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_memory", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_memory";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_memory_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_memory_mesh_registry = registry;
    return err;
}

void brain_memory_mesh_unregister(void) {
    if (g_brain_memory_mesh_registry && g_brain_memory_mesh_id != 0) {
        mesh_participant_unregister(g_brain_memory_mesh_registry, g_brain_memory_mesh_id);
        g_brain_memory_mesh_id = 0;
        g_brain_memory_mesh_registry = NULL;
    }
}


// NOTE: Implementation functions are currently in nimcp_brain.c
// External declarations for linking
extern bool save_working_memory_state(working_memory_t* wm, FILE* file);
extern bool load_working_memory_item(working_memory_t* wm, FILE* file);
extern bool load_working_memory_state(brain_t brain, FILE* file);
