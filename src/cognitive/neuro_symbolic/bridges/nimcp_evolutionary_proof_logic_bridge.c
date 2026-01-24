/**
 * @file nimcp_evolutionary_proof_logic_bridge.c
 * @brief Evolutionary Proof - Logic Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_evolutionary_proof_logic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

NIMCP_API evoproof_logic_bridge_t* evoproof_logic_bridge_create(void) {
    evoproof_logic_bridge_t* bridge = nimcp_calloc(1, sizeof(evoproof_logic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_EVOPROOF_LOGIC_BRIDGE,
                     "evoproof_logic_bridge");

    bridge->enable_axiom_expansion = true;
    bridge->enable_lemma_caching = true;

    return bridge;
}

NIMCP_API void evoproof_logic_bridge_destroy(evoproof_logic_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}
