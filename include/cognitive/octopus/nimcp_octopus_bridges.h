/**
 * @file nimcp_octopus_bridges.h
 * @brief Phase 2a bridges that wire the octopus module to peer subsystems.
 *
 * The octopus module exposes six hook slots (ethics / swarm / world / fep /
 * bio / immune) via octopus_set_*_hook(). This header declares a single
 * install function that populates all of them with wrappers around the
 * corresponding brain subsystems.
 *
 * SOLID notes:
 *  - SRP: each hook is a thin wrapper, one responsibility
 *  - OCP: adding a new hook means adding a new wrapper fn, no changes
 *    to the octopus core module
 *  - DIP: octopus core depends on the hook signatures, not these wrappers
 *
 * Called from the brain factory's octopus init after the octopus itself
 * is created and brain->{ethics,bio_router,...} subsystems are available.
 * Order-sensitive: bridges must install AFTER their peer subsystems are up.
 */
#ifndef NIMCP_OCTOPUS_BRIDGES_H
#define NIMCP_OCTOPUS_BRIDGES_H

#include <stdbool.h>
#include <stdint.h>
#include "core/brain/nimcp_brain.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Install all octopus hooks on the given brain's octopus module.
 *
 * Safe to call multiple times (idempotent). Safe if any peer subsystem is
 * NULL — that hook is just left unbound (octopus treats NULL hooks as no-op).
 *
 * @param brain Brain whose octopus module should be wired up.
 * @return true if at least one hook was bound; false on invalid input.
 */
bool nimcp_octopus_install_bridges(brain_t brain);

/**
 * @brief Tear down any bio_router registration + hook state.
 *
 * Safe to call on a brain that never had bridges installed.
 * Must be called BEFORE the bridge state is freed; lifecycle.c handles
 * this ordering (call uninstall, then nimcp_free the state).
 */
void nimcp_octopus_uninstall_bridges(brain_t brain);

/**
 * @brief Phase 4b: sample current occipital/visual cortex activity into a
 *        fixed-length feature vector suitable for feeding to octopus_explore().
 *
 * Packs a compact 64-channel summary of current visual state:
 *   [0..7]   V1 orientation histogram (normalized)
 *   [8..15]  Per-area feature-count density (V1..V5, padded)
 *   [16..23] Top-8 visual feature strengths (any area)
 *   [24..31] Top-8 motion-vector magnitudes from V5/MT
 *   [32..47] Top-8 color-percept hues + saturations from V4 (interleaved)
 *   [48..55] Global motion vector (dx/dy) and processing-time proxies
 *   [56..63] Padding / reserved
 *
 * Non-populated channels are zeroed. Out-of-range out_dim > 64 is clamped.
 *
 * @param brain   Brain instance (must have non-NULL brain->occipital).
 * @param out_vec Caller-provided buffer of length >= out_dim.
 * @param out_dim Target number of channels (clamped to [1, 64]).
 * @return Number of channels written (0 if occipital unavailable).
 */
uint32_t nimcp_octopus_sample_occipital_vec(brain_t brain,
                                            float* out_vec,
                                            uint32_t out_dim);

/**
 * @brief Phase 4b: convenience — sample occipital features and call
 *        octopus_explore() on the packed vision vector. Increments the
 *        bridge's vision_samples counter.
 *
 * @param brain Brain instance. Must have brain->octopus and brain->occipital.
 * @return 0 on success; -1 if occipital or octopus unavailable.
 */
int nimcp_octopus_explore_from_occipital(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_OCTOPUS_BRIDGES_H */
