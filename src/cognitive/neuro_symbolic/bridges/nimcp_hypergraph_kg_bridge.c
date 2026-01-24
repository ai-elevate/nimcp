/**
 * @file nimcp_hypergraph_kg_bridge.c
 * @brief Hypergraph - Knowledge Graph Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_hypergraph_kg_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

NIMCP_API hypergraph_kg_bridge_t* hypergraph_kg_bridge_create(void) {
    hypergraph_kg_bridge_t* bridge = nimcp_calloc(1, sizeof(hypergraph_kg_bridge_t));
    if (!bridge) return NULL;

    bridge_base_init(&bridge->base, BIO_MODULE_HYPERGRAPH_KG_BRIDGE,
                     "hypergraph_kg_bridge");

    bridge->enable_bidirectional_sync = true;

    return bridge;
}

NIMCP_API void hypergraph_kg_bridge_destroy(hypergraph_kg_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}
