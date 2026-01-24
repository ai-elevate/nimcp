/**
 * @file nimcp_genius_game_theory_bridge.c
 * @brief Genius - Game Theory Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_genius_game_theory_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

NIMCP_API genius_gt_bridge_t* genius_gt_bridge_create(void) {
    genius_gt_bridge_t* bridge = nimcp_calloc(1, sizeof(genius_gt_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_GENIUS_GT_BRIDGE,
                     "genius_gt_bridge");

    bridge->enable_nash_equilibrium = true;
    bridge->enable_shapley_attribution = true;

    return bridge;
}

NIMCP_API void genius_gt_bridge_destroy(genius_gt_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}
