/**
 * @file nimcp_jepa_brain_bridges.h
 * @brief Brain-level wiring for the four JEPA bridges (Phase 4n-q).
 *
 * Allocates, destroys, and drives:
 *   brain->omni_jepa_bridge      — omni world model latent-transition JEPA
 *   brain->neuromod_jepa_bridge  — neuromodulator-concentration JEPA
 *   brain->audio_jepa_bridge     — audio cortex mel/MFCC JEPA
 *   brain->engram_jepa_bridge    — engram-trace transition JEPA
 *
 * Each bridge is independent — missing target subsystems leave their
 * slot NULL and all subsequent ops on that slot no-op. The tick function
 * drives every available bridge one training step.
 */
#ifndef NIMCP_JEPA_BRAIN_BRIDGES_H
#define NIMCP_JEPA_BRAIN_BRIDGES_H

#include <stdbool.h>
#include "core/brain/nimcp_brain.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create all available JEPA bridges on brain. Called once during
 *        brain init after target subsystems are up. Safe to call if any
 *        target is missing — that bridge slot stays NULL.
 */
void nimcp_jepa_brain_bridges_init(brain_t brain);

/**
 * @brief Destroy every bridge, null out pointers. Safe repeat call.
 */
void nimcp_jepa_brain_bridges_destroy(brain_t brain);

/**
 * @brief Drive one training step across every live bridge. Called per
 *        brain tick from brain_learn_vector right after the octopus tick.
 */
void nimcp_jepa_brain_bridges_tick(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif
