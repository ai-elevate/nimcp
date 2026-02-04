/**
 * @file nimcp_information_geometry_bridge.c
 * @brief Information Geometry NIMCP Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-16
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "physics/geometry/nimcp_information_geometry_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <stdlib.h>
#include <string.h>

#include <stddef.h>  /* for NULL */
#include "utils/memory/nimcp_memory.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(information_geometry_bridge)

#define LOG_MODULE "INFORMATION_GEOMETRY_BRIDGE"


#define LOG_TAG "info_geom_bridge"

//=============================================================================
// Internal Structure
//=============================================================================

struct info_geom_bridge_struct {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    info_geom_bridge_config_t config;
    brain_kg_t* kg;
    void* exception_handler;
    void* bio_async_channel;
    bool initialized;
    brain_kg_node_id_t root_id;
    uint32_t node_count;
    uint32_t edge_count;
};

//=============================================================================
// Lifecycle Implementation
//=============================================================================

info_geom_bridge_config_t info_geom_bridge_default_config(void)
{
    info_geom_bridge_config_t config = {0};
    config.enable_kg_wiring = true;
    config.enable_exception_handling = true;
    config.enable_bio_async = true;
    config.enable_immune_presentation = true;
    config.enable_logging = true;
    return config;
}

info_geom_bridge_t info_geom_bridge_create(const info_geom_bridge_config_t* config)
{
    struct info_geom_bridge_struct* bridge = nimcp_calloc(1, sizeof(struct info_geom_bridge_struct));
    NIMCP_API_CHECK_ALLOC(bridge, "Failed to allocate information geometry bridge");

    if (config) {
        memcpy(&bridge->config, config, sizeof(info_geom_bridge_config_t));
    } else {
        bridge->config = info_geom_bridge_default_config();
    }

    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOG_INFO(LOG_TAG, "Information geometry bridge created");
    }

    return bridge;
}

void info_geom_bridge_destroy(info_geom_bridge_t bridge)
{
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "information_geometry");

    if (bridge->config.enable_logging) {
        NIMCP_LOG_INFO(LOG_TAG, "Destroying information geometry bridge");
    }

    nimcp_free(bridge);
}

int info_geom_bridge_register_kg(info_geom_bridge_t bridge, brain_kg_t* kg)
{
    if (!bridge || !kg) return -1;

    if (!bridge->config.enable_kg_wiring) {
        return 0;
    }

    bridge->kg = kg;

    /* Create root node */
    brain_kg_node_id_t root_id = brain_kg_add_node(kg, "information_geometry",
                                                    BRAIN_KG_NODE_INTEGRATION,
                                                    "Information geometry module");
    if (root_id == BRAIN_KG_INVALID_NODE) {
        return -1;
    }
    bridge->root_id = root_id;
    bridge->node_count++;

    /* Create Fisher information subsystem */
    brain_kg_node_id_t fisher_id = brain_kg_add_node(kg, "fisher_information",
                                                      BRAIN_KG_NODE_UTILITY,
                                                      "Fisher information matrix computation");
    if (fisher_id != BRAIN_KG_INVALID_NODE) {
        brain_kg_add_edge(kg, root_id, fisher_id, BRAIN_KG_EDGE_CONNECTS_TO,
                          "Contains Fisher info", 1.0f);
        bridge->node_count++;
        bridge->edge_count++;
    }

    /* Create natural gradient subsystem */
    brain_kg_node_id_t natgrad_id = brain_kg_add_node(kg, "natural_gradient",
                                                       BRAIN_KG_NODE_UTILITY,
                                                       "Natural gradient descent");
    if (natgrad_id != BRAIN_KG_INVALID_NODE) {
        brain_kg_add_edge(kg, root_id, natgrad_id, BRAIN_KG_EDGE_CONNECTS_TO,
                          "Contains natural gradient", 1.0f);
        bridge->node_count++;
        bridge->edge_count++;
    }

    /* Create neural manifold subsystem */
    brain_kg_node_id_t manifold_id = brain_kg_add_node(kg, "neural_manifold",
                                                        BRAIN_KG_NODE_UTILITY,
                                                        "Neural manifold analysis");
    if (manifold_id != BRAIN_KG_INVALID_NODE) {
        brain_kg_add_edge(kg, root_id, manifold_id, BRAIN_KG_EDGE_CONNECTS_TO,
                          "Contains manifold analysis", 1.0f);
        bridge->node_count++;
        bridge->edge_count++;
    }

    /* Create curvature subsystem */
    brain_kg_node_id_t curv_id = brain_kg_add_node(kg, "ricci_curvature",
                                                    BRAIN_KG_NODE_UTILITY,
                                                    "Ricci curvature computation");
    if (curv_id != BRAIN_KG_INVALID_NODE) {
        brain_kg_add_edge(kg, root_id, curv_id, BRAIN_KG_EDGE_CONNECTS_TO,
                          "Contains curvature analysis", 1.0f);
        bridge->node_count++;
        bridge->edge_count++;
    }

    /* Create KL divergence subsystem */
    brain_kg_node_id_t kl_id = brain_kg_add_node(kg, "kl_divergence",
                                                  BRAIN_KG_NODE_UTILITY,
                                                  "KL divergence computation");
    if (kl_id != BRAIN_KG_INVALID_NODE) {
        brain_kg_add_edge(kg, root_id, kl_id, BRAIN_KG_EDGE_CONNECTS_TO,
                          "Contains KL divergence", 1.0f);
        bridge->node_count++;
        bridge->edge_count++;
    }

    if (bridge->config.enable_logging) {
        NIMCP_LOG_INFO(LOG_TAG, "Registered %u nodes and %u edges in KG",
                       bridge->node_count, bridge->edge_count);
    }

    return 0;
}

int info_geom_bridge_register_exception(info_geom_bridge_t bridge, void* handler)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    bridge->exception_handler = handler;
    return 0;
}

int info_geom_bridge_register_bio_async(info_geom_bridge_t bridge, void* channel)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    bridge->bio_async_channel = channel;
    return 0;
}
