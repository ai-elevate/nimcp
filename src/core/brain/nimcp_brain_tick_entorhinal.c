/**
 * @file nimcp_brain_tick_entorhinal.c
 * @brief Implements brain_tick_entorhinal() — Wave 8B-c hot-path tick.
 *
 * STATUS: PARTIAL no-op shell. See header for rationale.
 *
 * The entorhinal cortex is not owned by brain_t — it is created by
 * downstream consumers (hippocampus, perirhinal) and stashed in their
 * local bridge structs. There is no brain->entorhinal field and no
 * brain_get_entorhinal() accessor. Without one, this driver cannot
 * safely reach the entorhinal_adapter_t* to call
 * entorhinal_adapter_update(adapter, dt) or
 * entorhinal_bidirectional_update(ec, dt).
 *
 * When a brain owner / accessor is added, the body should become:
 *
 *     entorhinal_adapter_t* ec_adapter = brain->entorhinal;
 *     if (ec_adapter && brain->entorhinal_enabled) {
 *         (void)entorhinal_adapter_update(ec_adapter, dt_ms);
 *     }
 *
 * Until then the driver fires harmlessly from the hot path / coordinator
 * and reserves its BRAIN_CYCLE_ENTORHINAL slot.
 *
 * @author NIMCP Development Team
 * @date 2026-04-24
 * @version 1.0.0
 */

#include "core/brain/nimcp_brain_tick_entorhinal.h"
#include "core/brain/nimcp_brain_internal.h"

void brain_tick_entorhinal(brain_t brain, float dt_ms)
{
    (void)dt_ms;

    if (!brain) {
        return;
    }

    /* No brain->entorhinal field exists yet — see file header for the
     * upgrade path. Intentional no-op (documented gap). */
}
