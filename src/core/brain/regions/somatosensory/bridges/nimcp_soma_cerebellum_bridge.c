/**
 * @file nimcp_soma_cerebellum_bridge.c
 * @brief Somatosensory-Cerebellum Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-12
 *
 * @author NIMCP Development Team
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brain/regions/somatosensory/bridges/nimcp_soma_cerebellum_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(soma_cerebellum_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_soma_cerebellum_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_soma_cerebellum_bridge_mesh_registry = NULL;

nimcp_error_t soma_cerebellum_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_soma_cerebellum_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "soma_cerebellum_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "soma_cerebellum_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_soma_cerebellum_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_soma_cerebellum_bridge_mesh_registry = registry;
    return err;
}

void soma_cerebellum_bridge_mesh_unregister(void) {
    if (g_soma_cerebellum_bridge_mesh_registry && g_soma_cerebellum_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_soma_cerebellum_bridge_mesh_registry, g_soma_cerebellum_bridge_mesh_id);
        g_soma_cerebellum_bridge_mesh_id = 0;
        g_soma_cerebellum_bridge_mesh_registry = NULL;
    }
}


#define LOG_MODULE "SOMA_CEREBELLUM_BRIDGE"


/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct soma_cereb_bridge_struct {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    soma_cereb_config_t config;
    nimcp_somatosensory_t* soma;
    void* cerebellum;

    bool is_connected;
    soma_cereb_status_t status;

    /* State tracking */
    soma_cereb_joint_state_t* joint_states;
    uint32_t num_joints;

    float* prediction_buffer;
    uint32_t prediction_dim;

    soma_cereb_stats_t stats;
};

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int soma_cereb_default_config(soma_cereb_config_t* config) {
    if (!config) return -1;

    memset(config, 0, sizeof(soma_cereb_config_t));

    config->max_joints = SOMA_CEREB_MAX_JOINTS;
    config->max_effectors = SOMA_CEREB_MAX_EFFECTORS;
    config->prediction_horizon_ms = SOMA_CEREB_PREDICTION_HORIZON;
    config->error_gain = 1.0f;
    config->adaptation_rate = 0.01f;
    config->enable_prediction = true;
    config->enable_adaptation = true;
    config->enable_logging = false;

    return 0;
}

soma_cereb_bridge_t* soma_cereb_bridge_create(const soma_cereb_config_t* config) {
    soma_cereb_bridge_t* bridge = (soma_cereb_bridge_t*)nimcp_calloc(1, sizeof(soma_cereb_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        memcpy(&bridge->config, config, sizeof(soma_cereb_config_t));
    } else {
        soma_cereb_default_config(&bridge->config);
    }

    bridge->joint_states = (soma_cereb_joint_state_t*)nimcp_calloc(
        bridge->config.max_joints, sizeof(soma_cereb_joint_state_t));
    if (!bridge->joint_states) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->prediction_dim = 64;
    bridge->prediction_buffer = (float*)nimcp_calloc(bridge->prediction_dim, sizeof(float));
    if (!bridge->prediction_buffer) {
        nimcp_free(bridge->joint_states);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->is_connected = false;
    bridge->status = SOMA_CEREB_STATUS_IDLE;

    NIMCP_LOGGING_INFO("Created %s bridge", "soma_cerebellum");
    return bridge;
}

void soma_cereb_bridge_destroy(soma_cereb_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "soma_cerebellum");

    nimcp_free(bridge->joint_states);
    nimcp_free(bridge->prediction_buffer);
    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API Implementation
 * ============================================================================ */

int soma_cereb_connect(soma_cereb_bridge_t* bridge, nimcp_somatosensory_t* soma, void* cerebellum) {
    if (!bridge || !soma) return -1;

    bridge->soma = soma;
    bridge->cerebellum = cerebellum;
    bridge->is_connected = true;
    bridge->status = SOMA_CEREB_STATUS_IDLE;

    return 0;
}

int soma_cereb_disconnect(soma_cereb_bridge_t* bridge) {
    if (!bridge) return -1;

    bridge->soma = NULL;
    bridge->cerebellum = NULL;
    bridge->is_connected = false;

    return 0;
}

bool soma_cereb_is_connected(const soma_cereb_bridge_t* bridge) {
    return bridge && bridge->is_connected;
}

/* ============================================================================
 * Proprioceptive API Implementation
 * ============================================================================ */

int soma_cereb_send_proprio(soma_cereb_bridge_t* bridge,
                            const soma_cereb_joint_state_t* joints,
                            uint32_t num_joints) {
    if (!bridge || !bridge->is_connected || !joints) return -1;
    if (num_joints > bridge->config.max_joints) return -1;

    memcpy(bridge->joint_states, joints, num_joints * sizeof(soma_cereb_joint_state_t));
    bridge->num_joints = num_joints;

    bridge->stats.proprio_updates++;

    return 0;
}

int soma_cereb_send_touch_during_move(soma_cereb_bridge_t* bridge,
                                      body_segment_t region,
                                      float intensity) {
    if (!bridge || !bridge->is_connected) return -1;
    (void)region;
    (void)intensity;

    bridge->stats.touch_events++;

    return 0;
}

int soma_cereb_send_balance_update(soma_cereb_bridge_t* bridge,
                                   float roll, float pitch, float yaw) {
    if (!bridge || !bridge->is_connected) return -1;
    (void)roll;
    (void)pitch;
    (void)yaw;

    return 0;
}

/* ============================================================================
 * Prediction API Implementation
 * ============================================================================ */

int soma_cereb_send_prediction_error(soma_cereb_bridge_t* bridge,
                                     const soma_cereb_prediction_error_t* error) {
    if (!bridge || !bridge->is_connected || !error) return -1;

    bridge->stats.prediction_errors++;
    bridge->stats.avg_prediction_error =
        bridge->stats.avg_prediction_error * 0.9f + error->error_magnitude * 0.1f;

    if (bridge->config.enable_adaptation) {
        bridge->stats.adaptation_progress += bridge->config.adaptation_rate;
        if (bridge->stats.adaptation_progress > 1.0f) {
            bridge->stats.adaptation_progress = 1.0f;
        }
    }

    return 0;
}

int soma_cereb_receive_efference(soma_cereb_bridge_t* bridge,
                                 soma_cereb_efference_t* efference) {
    if (!bridge || !bridge->is_connected || !efference) return -1;

    bridge->stats.efference_copies++;

    return 0;
}

int soma_cereb_compute_prediction(soma_cereb_bridge_t* bridge,
                                  const soma_cereb_efference_t* efference,
                                  float* prediction) {
    if (!bridge || !efference || !prediction) return -1;
    if (!bridge->config.enable_prediction) return -1;

    /* Simple forward model prediction */
    for (uint32_t i = 0; i < efference->prediction_dim && i < bridge->prediction_dim; i++) {
        prediction[i] = efference->predicted_sensory[i];
    }

    return 0;
}

/* ============================================================================
 * Coordination API Implementation
 * ============================================================================ */

int soma_cereb_request_coordination(soma_cereb_bridge_t* bridge, uint32_t task_id) {
    if (!bridge || !bridge->is_connected) return -1;
    (void)task_id;

    bridge->status = SOMA_CEREB_STATUS_COORDINATING;

    return 0;
}

int soma_cereb_report_movement_complete(soma_cereb_bridge_t* bridge,
                                        uint32_t task_id, bool success) {
    if (!bridge || !bridge->is_connected) return -1;
    (void)task_id;
    (void)success;

    bridge->status = SOMA_CEREB_STATUS_IDLE;

    return 0;
}

/* ============================================================================
 * Statistics API Implementation
 * ============================================================================ */

int soma_cereb_get_stats(const soma_cereb_bridge_t* bridge, soma_cereb_stats_t* stats) {
    if (!bridge || !stats) return -1;
    memcpy(stats, &bridge->stats, sizeof(soma_cereb_stats_t));
    return 0;
}

int soma_cereb_reset_stats(soma_cereb_bridge_t* bridge) {
    if (!bridge) return -1;
    memset(&bridge->stats, 0, sizeof(soma_cereb_stats_t));
    return 0;
}

void soma_cereb_print_summary(const soma_cereb_bridge_t* bridge) {
    if (!bridge) return;

    printf("=== Somatosensory-Cerebellum Bridge Summary ===\n");
    printf("Connected: %s\n", bridge->is_connected ? "Yes" : "No");
    printf("Proprio Updates: %lu\n", (unsigned long)bridge->stats.proprio_updates);
    printf("Touch Events: %lu\n", (unsigned long)bridge->stats.touch_events);
    printf("Prediction Errors: %lu\n", (unsigned long)bridge->stats.prediction_errors);
    printf("Avg Prediction Error: %.4f\n", bridge->stats.avg_prediction_error);
    printf("Adaptation Progress: %.2f%%\n", bridge->stats.adaptation_progress * 100.0f);
    printf("==============================================\n");
}
