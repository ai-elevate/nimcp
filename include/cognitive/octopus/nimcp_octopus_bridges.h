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

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_OCTOPUS_BRIDGES_H */
