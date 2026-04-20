/**
 * @file nimcp_octopus_bridges.c
 * @brief Phase 2a: register octopus hooks against peer brain subsystems.
 *
 * Each wrapper is a thin adapter — it pulls the relevant subsystem handle
 * from `brain_t`, does a minimal transformation, and calls into the peer.
 * All wrappers NULL-guard their subsystem handle so the module degrades
 * gracefully if a peer isn't initialized.
 *
 * Wrappers are registered via octopus_set_*_hook(). The octopus core
 * module knows nothing about brain internals (DIP).
 */
#include "cognitive/octopus/nimcp_octopus_bridges.h"
#include "cognitive/octopus/nimcp_octopus.h"
#include "core/brain/nimcp_brain_internal.h"

#include "cognitive/ethics/nimcp_ethics.h"
#include "async/nimcp_bio_router.h"
#include "cognitive/free_energy/nimcp_fep_orchestrator.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#include <string.h>
#include <stdint.h>

/*============================================================================
 * Per-brain bridge state — allows the void* user pointer in each hook to
 * carry the brain reference to the peer subsystem without capturing globals.
 *==========================================================================*/

typedef struct octopus_bridge_state_s {
    brain_t brain;
    uint64_t bio_broadcast_counter;
    uint64_t ethics_eval_counter;
} octopus_bridge_state_t;

/*============================================================================
 * Hook implementations
 *==========================================================================*/

/* Ethics: convert arm latent → action_context_t, evaluate, return allowed. */
static bool _ethics_hook(const octopus_arm_t* arm, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !st->brain || !st->brain->ethics || !arm) return true;

    /* Build an action_context_t from the arm's latent. First 5 floats of
     * the arm latent map to the 5 specific violation metrics; the rest
     * are the general feature vector. This is a heuristic projection —
     * higher-fidelity mapping is Phase 2b work. */
    action_context_t ctx = {0};
    ctx.features = (float*)arm->latent;   /* ethics reads as const */
    ctx.num_features = 64;
    ctx.affected_agents = NULL;
    ctx.num_affected_agents = 0;
    ctx.predicted_harm      = arm->latent[0] > 0 ? arm->latent[0] : 0.0f;
    ctx.fairness_violation  = arm->latent[1] > 0 ? arm->latent[1] : 0.0f;
    ctx.deception_level     = arm->latent[2] > 0 ? arm->latent[2] : 0.0f;
    ctx.autonomy_violation  = arm->latent[3] > 0 ? arm->latent[3] : 0.0f;
    ctx.privacy_violation   = arm->latent[4] > 0 ? arm->latent[4] : 0.0f;
    ctx.consent_violation   = 0.0f;

    st->ethics_eval_counter++;
    ethics_evaluation_t eval = ethics_engine_evaluate_action(
        (ethics_engine_t)st->brain->ethics, &ctx);
    return eval.allowed;
}

/* Swarm delegation: for Phase 2a, log + count. Real edge delegation needs
 * the swarm master's submit_task API which isn't consistently available
 * at this init order. Phase 2b will promote this. */
static void _swarm_hook(const octopus_arm_t* arm, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !arm) return;
    NIMCP_LOGGING_DEBUG("[octopus-swarm] arm %u delegated (conf=%.3f)",
                        arm->id, arm->confidence);
}

/* World model: Phase 2a logs. Phase 2b will map aggregated latent to
 * world-model observation when omni_wm_submit_observation API is bound. */
static void _world_hook(const float* aggregated, uint32_t len, void* user) {
    (void)aggregated;
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !st->brain) return;
    NIMCP_LOGGING_DEBUG("[octopus-world] aggregated latent len=%u delivered",
                        len);
}

/* FEP: coherence is the inverse-surprise signal. fep_orchestrator_update
 * takes time_ms rather than a value; we log-and-store the coherence so a
 * higher-level cycle coordinator can pick it up via stats. */
static void _fep_hook(float coherence, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st) return;
    NIMCP_LOGGING_DEBUG("[octopus-fep] coherence=%.3f (higher = less surprise)",
                        coherence);
}

/* Bio-async: broadcast significant events (low coherence, arm vetoes) via
 * the router. The event string identifies the event type; value carries
 * a magnitude or arm-id depending on event. */
static void _bio_hook(const char* event, float value, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !event) return;
    st->bio_broadcast_counter++;
    /* Defer actual broadcast plumbing (msg_type allocation, payload packing)
     * to Phase 2b; Phase 2a logs so monitoring can observe firing rate. */
    NIMCP_LOGGING_DEBUG("[octopus-bio] event=%s value=%.3f (count=%llu)",
                        event, value,
                        (unsigned long long)st->bio_broadcast_counter);
}

/* Immune: called every explore(). A plain heartbeat signals liveness to
 * the immune system; if the heartbeat stops, the immune system can flag
 * octopus as dead-or-frozen. Bio-async broadcast is the natural channel. */
static void _immune_hook(void* user) {
    (void)user;
    /* Heartbeat via logging for now; Phase 2b will emit a real bio-async
     * heartbeat message that the immune system can subscribe to. */
    NIMCP_LOGGING_DEBUG("[octopus-immune] heartbeat");
}

/*============================================================================
 * Public install function
 *==========================================================================*/

bool nimcp_octopus_install_bridges(brain_t brain) {
    if (!brain || !brain->octopus) return false;

    /* Allocate bridge state once; leak-free because brain lifetime matches
     * octopus lifetime. The state is stored as a per-hook user pointer. */
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)nimcp_calloc(
        1, sizeof(octopus_bridge_state_t));
    if (!st) return false;
    st->brain = brain;

    octopus_system_t* ctx = (octopus_system_t*)brain->octopus;

    octopus_set_ethics_hook(ctx, _ethics_hook, st);
    octopus_set_swarm_hook (ctx, _swarm_hook,  st);
    octopus_set_world_hook (ctx, _world_hook,  st);
    octopus_set_fep_hook   (ctx, _fep_hook,    st);
    octopus_set_bio_hook   (ctx, _bio_hook,    st);
    octopus_set_immune_hook(ctx, _immune_hook, st);

    NIMCP_LOGGING_INFO("octopus_bridges: 6 hooks installed "
                       "(ethics=%s, swarm=log, world=log, fep=log, bio=log, immune=heartbeat)",
                       brain->ethics ? "live" : "no-peer");
    return true;
}
