//=============================================================================
// nimcp_brain_io.c - Brain I/O and Persistence
//=============================================================================
/**
 * @file nimcp_brain_io.c
 * @brief Brain I/O, metadata persistence, and JSON serialization
 *
 * This module contains approximately 900 lines extracted from nimcp_brain.c:
 * - save_metadata() / load_metadata() - Brain metadata persistence
 * - brain_export_json() / brain_import_json() - JSON export/import
 * - brain_save_json() / brain_load_json() - JSON file operations
 * - ensure_snapshot_dir() - Snapshot directory management
 * - brain_get_memory_usage() - Memory usage calculation
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_io.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "api/nimcp_api_exception.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <cjson/cJSON.h>
#include <dirent.h>
#include <sys/stat.h>
#include "io/serialization/nimcp_serialization.h"

#define LOG_MODULE "BRAIN_IO"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_io)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_io_mesh_id = 0;
static mesh_participant_registry_t* g_brain_io_mesh_registry = NULL;

nimcp_error_t brain_io_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_io_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_io", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_io";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_io_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_io_mesh_registry = registry;
    return err;
}

void brain_io_mesh_unregister(void) {
    if (g_brain_io_mesh_registry && g_brain_io_mesh_id != 0) {
        mesh_participant_unregister(g_brain_io_mesh_registry, g_brain_io_mesh_id);
        g_brain_io_mesh_id = 0;
        g_brain_io_mesh_registry = NULL;
    }
}


// NOTE: Implementation functions are currently in nimcp_brain.c
// External declarations for linking
extern bool save_metadata(brain_t brain, const char* filepath);
extern bool load_metadata(brain_t brain, const char* filepath);
extern bool ensure_snapshot_dir(const char* snapshot_dir);
extern size_t brain_get_memory_usage(brain_t brain);
extern brain_t brain_import_json(const char* json_str);
extern bool brain_save_json(brain_t brain, const char* filepath, uint32_t flags);
extern brain_t brain_load_json(const char* filepath);
extern bool brain_resize_update_subsystems_internal(brain_t brain, neural_network_t new_base_network, uint32_t new_neuron_count);
