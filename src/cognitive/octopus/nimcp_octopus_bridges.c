/**
 * @file nimcp_octopus_bridges.c
 * @brief Octopus → peer-subsystem bridges (Phase 2a + 2b + 3b).
 *
 * Phase 2a: install the 6 hooks, only ethics is live-gating.
 * Phase 2b: route the remaining 5 hooks through bio_router_broadcast so real
 *          subscribers (swarm / world / FEP / immune / bio-async) receive
 *          structured events. Octopus registers as BIO_MODULE_OCTOPUS and
 *          emits octopus_event_msg_t on each hook firing.
 * Phase 3b: swarm hook also adds rate-limited nodes to brain->internal_kg;
 *          bio hook also encodes rate-limited engrams on low-coherence
 *          events via brain->engram_system.
 *
 * Each wrapper is a thin adapter — pulls the subsystem handle from
 * `brain_t`, transforms arm/aggregate data into an event, broadcasts.
 * All wrappers NULL-guard their inputs and degrade gracefully if a peer
 * isn't initialized (e.g., router not up at install time).
 *
 * SOLID:
 *   SRP: each hook is a single-purpose adapter.
 *   OCP: adding a new event type = add enum + one emit call; no core change.
 *   DIP: octopus core only knows the hook signatures, not bridge contents.
 */
#include "cognitive/octopus/nimcp_octopus_bridges.h"
#include "cognitive/octopus/nimcp_octopus.h"
#include "core/brain/nimcp_brain_internal.h"

#include "cognitive/ethics/nimcp_ethics.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "cognitive/memory/nimcp_engram.h"
#include "core/brain/nimcp_brain_kg.h"
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_adapter.h"
#include "core/brain/subcortical/nimcp_amygdala.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Broadcast message schema: a small fixed-size struct the octopus emits
 * on significant events. Consumers read the `type` field to disambiguate.
 * Padded to 32 bytes for consistent router stats. */
typedef enum {
    OCTOPUS_EVT_HEARTBEAT      = 0x0000,  /* every explore() */
    OCTOPUS_EVT_ARM_VETO       = 0x0001,  /* ethics vetoed an arm */
    OCTOPUS_EVT_LOW_COHERENCE  = 0x0002,  /* aggregate coherence dropped */
    OCTOPUS_EVT_DELEGATE       = 0x0003,  /* high-confidence arm → swarm */
    OCTOPUS_EVT_WORLD_OBS      = 0x0004,  /* aggregated latent → world model */
    OCTOPUS_EVT_FEP_SURPRISE   = 0x0005,  /* coherence signal for FEP */
    OCTOPUS_EVT_STRESS_CONTEXT = 0x0006,  /* hypothalamus cortisol signal (Phase 3c) */
    OCTOPUS_EVT_FEAR_CONTEXT   = 0x0007,  /* amygdala fear intensity (Phase 3c) */
} octopus_event_type_t;

typedef struct {
    uint16_t type;       /* octopus_event_type_t */
    uint16_t arm_id;     /* 0xFFFF if not arm-specific */
    float    value;      /* event-specific magnitude (coherence, confidence, etc.) */
    uint32_t sequence;   /* monotonic per-context; detect drops */
    uint32_t _pad[5];    /* reserve 32-byte total, room for future fields */
} octopus_event_msg_t;

/*============================================================================
 * Per-brain bridge state — allows the void* user pointer in each hook to
 * carry the brain reference to the peer subsystem without capturing globals.
 *==========================================================================*/

typedef struct octopus_bridge_state_s {
    brain_t brain;
    uint64_t bio_broadcast_counter;
    bio_module_context_t bio_ctx; /* NULL if router not up at install time */
    uint32_t sequence;            /* monotonic per-context sequence */
    /* Phase 3b integrations. */
    uint64_t engram_encodings;    /* memory episodes encoded from octopus events */
    uint64_t kg_nodes_added;      /* KG nodes added for delegation events */
    /* Phase 3c: hypothalamus + amygdala. */
    float    last_cortisol;       /* most recently observed stress level */
    float    last_fear;           /* most recently observed fear intensity */
    uint64_t stress_broadcasts;   /* number of STRESS_CONTEXT events emitted */
    uint64_t fear_broadcasts;     /* number of FEAR_CONTEXT events emitted */
    uint64_t amygdala_steps;      /* number of amygdala_step() calls driven */
    uint64_t last_step_us;        /* monotonic timestamp of last amygdala_step */
} octopus_bridge_state_t;

/* Publish all bridge-owned counters onto the octopus core's stats struct so
 * a single octopus_get_stats() call sees both core and bridge activity.
 * Called at the end of the immune hook since that fires every explore. */
static void _publish_bridge_stats(octopus_bridge_state_t* st) {
    if (!st || !st->brain || !st->brain->octopus) return;
    octopus_set_bridge_stats(
        (octopus_system_t*)st->brain->octopus,
        st->engram_encodings,
        st->kg_nodes_added,
        st->stress_broadcasts,
        st->fear_broadcasts,
        st->amygdala_steps,
        st->last_cortisol,
        st->last_fear);
}

/* Lazy register against bio_router if install-time registration was
 * skipped (router wasn't up yet). Safe to call repeatedly — already
 * registered states are a no-op. */
static void _ensure_registered(octopus_bridge_state_t* st) {
    if (!st || st->bio_ctx) return;
    if (!bio_router_is_initialized()) return;
    bio_module_info_t info = {0};
    info.module_id = BIO_MODULE_OCTOPUS;
    info.module_name = "octopus";
    info.inbox_capacity = 0;
    info.user_data = st;
    st->bio_ctx = bio_router_register_module(&info);
    if (st->bio_ctx) {
        NIMCP_LOGGING_INFO("octopus_bridges: lazy-registered with bio_router");
    }
}

/* DRY helper: pack + broadcast one event. NULL-guards the router context
 * so degraded operation (router not initialized) is a no-op instead of a
 * crash. Returns true iff broadcast succeeded. */
static bool _emit(octopus_bridge_state_t* st,
                   octopus_event_type_t type,
                   uint16_t arm_id,
                   float value) {
    if (!st) return false;
    _ensure_registered(st);
    if (!st->bio_ctx) return false;
    octopus_event_msg_t msg = {0};
    msg.type     = (uint16_t)type;
    msg.arm_id   = arm_id;
    msg.value    = value;
    msg.sequence = ++st->sequence;
    nimcp_error_t rc = bio_router_broadcast(st->bio_ctx, &msg, sizeof(msg));
    if (rc == NIMCP_SUCCESS) st->bio_broadcast_counter++;
    return rc == NIMCP_SUCCESS;
}

/*============================================================================
 * Hook implementations
 *==========================================================================*/

/* Ethics: convert arm latent → action_context_t, evaluate, return allowed.
 * Copies the latent into a local writable buffer to avoid a cast-away-const
 * (action_context_t.features is `float*` by convention even though the
 * evaluator reads it as const; copying is the safe portable path). */
static bool _ethics_hook(const octopus_arm_t* arm, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !arm || !st->brain || !st->brain->ethics) return true;

    float features_copy[OCTOPUS_ARM_DIM];
    memcpy(features_copy, arm->latent, sizeof(features_copy));

    /* First 5 floats map to the 5 specific violation metrics (clamped to
     * [0,1] since they represent probability-like magnitudes); the rest
     * are the general feature vector. Heuristic projection; higher-fidelity
     * mapping is Phase 2b work. */
    action_context_t ctx = {0};
    ctx.features = features_copy;
    ctx.num_features = OCTOPUS_ARM_DIM;
    ctx.affected_agents = NULL;
    ctx.num_affected_agents = 0;
    ctx.predicted_harm      = features_copy[0] > 0 ? features_copy[0] : 0.0f;
    ctx.fairness_violation  = features_copy[1] > 0 ? features_copy[1] : 0.0f;
    ctx.deception_level     = features_copy[2] > 0 ? features_copy[2] : 0.0f;
    ctx.autonomy_violation  = features_copy[3] > 0 ? features_copy[3] : 0.0f;
    ctx.privacy_violation   = features_copy[4] > 0 ? features_copy[4] : 0.0f;
    ctx.consent_violation   = 0.0f;

    /* brain->ethics is declared ethics_engine_t in brain_internal.h — no cast needed. */
    ethics_evaluation_t eval = ethics_engine_evaluate_action(
        st->brain->ethics, &ctx);
    return eval.allowed;
}

/* Swarm delegation: (1) broadcast DELEGATE event through bio_router,
 * (2) add a node to the internal KG recording this high-confidence
 * arm-finding so downstream reasoners can query past delegations.
 * KG additions are rate-limited to every 64th delegation (power-of-two
 * mask avoids % so the check stays constant-time) to avoid graph bloat
 * from identical high-confidence steady states. */
static void _swarm_hook(const octopus_arm_t* arm, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !arm) return;
    _emit(st, OCTOPUS_EVT_DELEGATE, (uint16_t)arm->id, arm->confidence);

    /* KG integration: add a node for notable delegations. */
    if (st->brain && st->brain->internal_kg && st->brain->internal_kg_enabled) {
        if ((st->kg_nodes_added & 0x3F) == 0) {  /* every 64th */
            char name[64], desc[128];
            snprintf(name, sizeof(name), "octopus_delegate_%llu",
                     (unsigned long long)st->kg_nodes_added);
            snprintf(desc, sizeof(desc),
                     "Octopus arm %u delegated with confidence %.3f "
                     "(sequence=%u)",
                     arm->id, arm->confidence, st->sequence);
            brain_kg_add_node(st->brain->internal_kg, name,
                              BRAIN_KG_NODE_COGNITIVE, desc);
        }
        st->kg_nodes_added++;
    }
}

/* World model: broadcast WORLD_OBS with coherence as magnitude. Full
 * aggregated-latent delivery would require variable-sized messages; a
 * follow-up iteration can upgrade to latent-payload messages once the
 * subscriber side is defined. */
static void _world_hook(const float* aggregated, uint32_t len, void* user) {
    (void)aggregated; (void)len;
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    _emit(st, OCTOPUS_EVT_WORLD_OBS, 0xFFFF, 0.0f);
}

/* FEP: broadcast the coherence signal. Low coherence = high surprise =
 * free-energy proxy. FEP orchestrator can subscribe to this and feed it
 * into its surprise accumulator. */
static void _fep_hook(float coherence, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    _emit(st, OCTOPUS_EVT_FEP_SURPRISE, 0xFFFF, coherence);
}

/* Bio-async event broadcast: route octopus events (arm vetoes, low
 * coherence) through the real bio_router. Event string is mapped to one
 * of our typed enum values. Unrecognized events get dropped (safe-fail).
 *
 * Phase 3b: low-coherence events also trigger an episodic engram —
 * the brain remembers surprising-octopus moments so they can be replayed
 * during consolidation. Rate-limited so identical prolonged low-coherence
 * doesn't flood the engram store. */
static void _bio_hook(const char* event, float value, void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    if (!st || !event) return;
    octopus_event_type_t type;
    if (strcmp(event, "octopus_low_coherence") == 0) {
        type = OCTOPUS_EVT_LOW_COHERENCE;
    } else if (strcmp(event, "octopus_arm_vetoed") == 0) {
        type = OCTOPUS_EVT_ARM_VETO;
    } else {
        return;  /* unknown event — don't emit a spurious message */
    }
    /* For ARM_VETO events the octopus core passes `(float)arm_id` as
     * value; truncate to uint16_t to carry that id in the message header.
     * For LOW_COHERENCE, value IS the coherence, so arm_id is "n/a" (0xFFFF). */
    uint16_t arm_id = (type == OCTOPUS_EVT_ARM_VETO)
                    ? (uint16_t)value : (uint16_t)0xFFFF;
    _emit(st, type, arm_id, value);

    /* Phase 3b: engram encoding for significant episodes. Low coherence
     * means "arms are split" = surprising experience = worth remembering.
     * Only encode every 32nd such event to avoid flooding consolidation. */
    if (type == OCTOPUS_EVT_LOW_COHERENCE &&
        st->brain && st->brain->engram_system &&
        (st->engram_encodings & 0x1F) == 0) {
        /* Build a minimal engram pattern from the arm broadcast states.
         * n_arms neuron-ids (synthetic but stable), activations = the
         * arm's broadcast state at the moment of low coherence. */
        octopus_system_t* oc = (octopus_system_t*)st->brain->octopus;
        if (oc) {
            uint32_t n_arms = octopus_get_n_arms(oc);
            uint32_t ids[16];       /* max arms = 16 */
            float    acts[16];
            uint32_t count = 0;
            for (uint32_t a = 0; a < n_arms && count < 16; a++) {
                ids[count]  = 0x50000000u | a;  /* synthetic id space */
                acts[count] = octopus_get_broadcast_state(oc, a);
                count++;
            }
            emotional_tag_t emo = {0};
            emo.valence   = -0.3f;  /* low coherence → mildly negative */
            emo.arousal   = 1.0f - value;  /* lower coherence → higher arousal */
            emo.intensity = 1.0f - value;
            emo.timestamp_ms = 0;
            engram_encode(st->brain->engram_system, ids, acts, count,
                          MEMORY_TYPE_EPISODIC, emo);
        }
    }
    if (type == OCTOPUS_EVT_LOW_COHERENCE) st->engram_encodings++;
}

/* Immune: fires on every explore(). Emit a heartbeat, step the amygdala
 * forward (using real wall-clock dt), then on a 32x rate limit poll
 * hypothalamus/amygdala state and broadcast STRESS/FEAR context events.
 *
 * Driving amygdala_step() here is cheap (μs-scale for a small module)
 * and guarantees it advances every explore. Without this the amygdala's
 * internal temporal dynamics would stay frozen and fear responses would
 * be stale or all-zero. */
static void _immune_hook(void* user) {
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)user;
    _emit(st, OCTOPUS_EVT_HEARTBEAT, 0xFFFF, 0.0f);
    if (!st || !st->brain) return;

    /* === Drive amygdala_step per explore with real dt === */
    if (st->brain->amygdala && st->brain->amygdala_enabled) {
        uint64_t now_us = nimcp_time_get_us();
        float dt_ms = 0.0f;
        if (st->last_step_us != 0 && now_us > st->last_step_us) {
            dt_ms = (float)(now_us - st->last_step_us) / 1000.0f;
            /* Sanity cap: if the brain was paused, don't feed a huge dt. */
            if (dt_ms > 5000.0f) dt_ms = 5000.0f;
        }
        st->last_step_us = now_us;
        if (dt_ms > 0.0f) {
            (void)amygdala_step((amygdala_t*)st->brain->amygdala, dt_ms);
            st->amygdala_steps++;
        }
    }

    /* Publish bridge counters every explore so octopus_get_stats is fresh. */
    _publish_bridge_stats(st);

    /* Rate-limit the slower physiological-signal polls. */
    if ((st->stress_broadcasts & 0x1F) != 0) {
        st->stress_broadcasts++;
        return;
    }
    st->stress_broadcasts++;

    if (st->brain->hypothalamus && st->brain->hypothalamus_enabled) {
        hypothalamus_state_t hs = {0};
        if (hypothalamus_get_state(
                (const hypothalamus_adapter_t*)st->brain->hypothalamus, &hs)) {
            float cortisol = hs.hpa_axis.cortisol_level;
            st->last_cortisol = cortisol;
            _emit(st, OCTOPUS_EVT_STRESS_CONTEXT, 0xFFFF, cortisol);
        }
    }

    if (st->brain->amygdala && st->brain->amygdala_enabled) {
        amyg_fear_response_t resp = {0};
        if (amygdala_get_response(
                (const amygdala_t*)st->brain->amygdala, &resp) == 0) {
            st->last_fear = resp.fear_intensity;
            st->fear_broadcasts++;
            _emit(st, OCTOPUS_EVT_FEAR_CONTEXT, 0xFFFF, resp.fear_intensity);
        }
    }
}

/*============================================================================
 * Public install function
 *==========================================================================*/

bool nimcp_octopus_install_bridges(brain_t brain) {
    if (!brain || !brain->octopus) return false;
    if (brain->octopus_bridge_state) return true;  /* idempotent */

    /* Allocate bridge state and track it on the brain so destroy path
     * can free it (see nimcp_brain_part_lifecycle.c octopus teardown). */
    octopus_bridge_state_t* st = (octopus_bridge_state_t*)nimcp_calloc(
        1, sizeof(octopus_bridge_state_t));
    if (!st) return false;
    st->brain = brain;
    brain->octopus_bridge_state = (void*)st;

    /* Register with bio_router so broadcasts carry a real module identity.
     * Safe-fail: if the router isn't initialized yet, st->bio_ctx stays
     * NULL and _emit() becomes a no-op. The module can be re-installed
     * later once the router is up. */
    if (bio_router_is_initialized()) {
        bio_module_info_t info = {0};
        info.module_id = BIO_MODULE_OCTOPUS;
        info.module_name = "octopus";
        info.inbox_capacity = 0;   /* use router default */
        info.user_data = st;
        st->bio_ctx = bio_router_register_module(&info);
        if (!st->bio_ctx) {
            NIMCP_LOGGING_WARN("octopus_bridges: router registration failed");
        }
    }

    octopus_system_t* ctx = (octopus_system_t*)brain->octopus;

    octopus_set_ethics_hook(ctx, _ethics_hook, st);
    octopus_set_swarm_hook (ctx, _swarm_hook,  st);
    octopus_set_world_hook (ctx, _world_hook,  st);
    octopus_set_fep_hook   (ctx, _fep_hook,    st);
    octopus_set_bio_hook   (ctx, _bio_hook,    st);
    octopus_set_immune_hook(ctx, _immune_hook, st);

    NIMCP_LOGGING_INFO("octopus_bridges: 6 hooks installed "
                       "(ethics=%s, router=%s)",
                       brain->ethics ? "live" : "no-peer",
                       st->bio_ctx   ? "live" : "not-up-yet");
    return true;
}

/* Called from the brain lifecycle destroy path AFTER octopus_destroy().
 * Unregisters the bio_router module before the state is freed so the
 * router doesn't retain a dangling context reference. */
void nimcp_octopus_uninstall_bridges(brain_t brain) {
    if (!brain || !brain->octopus_bridge_state) return;
    octopus_bridge_state_t* st =
        (octopus_bridge_state_t*)brain->octopus_bridge_state;
    if (st->bio_ctx) {
        bio_router_unregister_module(st->bio_ctx);
        st->bio_ctx = NULL;
    }
}
