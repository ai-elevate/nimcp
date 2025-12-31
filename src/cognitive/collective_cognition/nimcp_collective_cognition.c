/**
 * @file nimcp_collective_cognition.c
 * @brief Implementation of unified collective cognition system
 *
 * WHAT: Integrates hyperscanning, extended mind, collective phi, and shared intentionality
 * WHY: Enable distributed consciousness across multiple NIMCP brain instances
 * HOW: Unified coordination of all subsystems with bio-async messaging
 */

#include "cognitive/collective_cognition/nimcp_collective_cognition.h"
#include "cognitive/collective_cognition/nimcp_hyperscanning.h"
#include "cognitive/collective_cognition/nimcp_extended_mind.h"
#include "cognitive/collective_cognition/nimcp_collective_phi.h"
#include "cognitive/collective_cognition/nimcp_shared_intentionality.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/*=============================================================================
 * Internal Structures
 *===========================================================================*/

/**
 * @brief Registered brain instance
 */
typedef struct {
    uint32_t instance_id;
    brain_t* brain;                 /* Optional direct brain reference */
    bool active;

    /* Per-instance state */
    float local_phi;                /* Individual phi */
    float atp_level;                /* Metabolic state */
    float fatigue_level;
    uint64_t last_heartbeat_us;
    uint32_t missed_heartbeats;

    /* Sync state per band */
    float band_power[SYNC_BAND_COUNT];
    float band_phase[SYNC_BAND_COUNT];
} registered_instance_t;

/**
 * @brief Internal state for collective cognition system
 */
struct collective_cognition {
    /* Configuration */
    collective_cognition_config_t config;

    /* Registered instances */
    registered_instance_t instances[COLLECTIVE_MAX_INSTANCES];
    uint32_t instance_count;

    /* Subsystem handles (embedded for now, will be separate modules) */
    /* TODO: Replace with actual subsystem handles when implemented */
    void* hyperscanning;     /* hyperscanning_t* */
    void* extended_mind;     /* extended_mind_t* */
    void* phi_system;        /* collective_phi_system_t* */
    void* intentionality;    /* shared_intentionality_t* */

    /* Cached state */
    collective_cognition_state_t state;

    /* Bio-async integration */
    bio_router_t* bio_router;
    bool bio_async_connected;

    /* Statistics */
    collective_cognition_stats_t stats;

    /* Synchronization tracking */
    float pair_plv[COLLECTIVE_MAX_INSTANCES][COLLECTIVE_MAX_INSTANCES][SYNC_BAND_COUNT];

    /* Internal flags */
    bool initialized;
    uint64_t last_update_us;
};

/*=============================================================================
 * Helper Functions - Time
 *===========================================================================*/

/**
 * @brief Get current wall-clock timestamp in microseconds
 *
 * Uses nimcp_time_get_us() for actual wall-clock time instead of
 * a monotonic counter. Handles potential overflow by using the full
 * 64-bit range which won't overflow for ~584,000 years from epoch.
 */
static uint64_t get_timestamp_us(void) {
    return nimcp_time_get_us();
}

/*=============================================================================
 * Helper Functions - Instance Management
 *===========================================================================*/

static registered_instance_t* find_instance(
    collective_cognition_t* cc,
    uint32_t instance_id
) {
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active && cc->instances[i].instance_id == instance_id) {
            return &cc->instances[i];
        }
    }
    return NULL;
}

static registered_instance_t* find_free_slot(collective_cognition_t* cc) {
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (!cc->instances[i].active) {
            return &cc->instances[i];
        }
    }
    return NULL;
}

static int find_instance_index(
    const collective_cognition_t* cc,
    uint32_t instance_id
) {
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active && cc->instances[i].instance_id == instance_id) {
            return (int)i;
        }
    }
    return -1;
}

/*=============================================================================
 * Helper Functions - Phi Computation
 *===========================================================================*/

/**
 * @brief Compute phase-locking value between two instances for a given band
 */
static float compute_plv(
    const registered_instance_t* a,
    const registered_instance_t* b,
    sync_band_t band
) {
    /* PLV = |mean(exp(i * (phase_a - phase_b)))| */
    float phase_diff = a->band_phase[band] - b->band_phase[band];
    /* Simplified: use cosine of phase difference as proxy */
    return fabsf(cosf(phase_diff));
}

/**
 * @brief Compute global synchronization across all instances
 */
static float compute_global_sync(collective_cognition_t* cc) {
    if (cc->instance_count < 2) return 0.0f;

    float total_sync = 0.0f;
    uint32_t pair_count = 0;

    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (!cc->instances[i].active) continue;

        for (uint32_t j = i + 1; j < COLLECTIVE_MAX_INSTANCES; j++) {
            if (!cc->instances[j].active) continue;

            /* Compute average PLV across bands */
            float pair_sync = 0.0f;
            for (int b = 0; b < SYNC_BAND_COUNT; b++) {
                float plv = compute_plv(&cc->instances[i], &cc->instances[j], (sync_band_t)b);
                cc->pair_plv[i][j][b] = plv;
                cc->pair_plv[j][i][b] = plv;
                pair_sync += plv;
            }
            pair_sync /= SYNC_BAND_COUNT;
            total_sync += pair_sync;
            pair_count++;
        }
    }

    return pair_count > 0 ? total_sync / pair_count : 0.0f;
}

/**
 * @brief Compute collective phi (integrated information)
 */
static void compute_collective_phi(collective_cognition_t* cc) {
    collective_phi_t* phi = &cc->state.phi;

    /* Sum local phis */
    phi->phi_local = 0.0f;
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active) {
            phi->phi_local += cc->instances[i].local_phi;
        }
    }

    /* Network phi based on synchronization */
    float global_sync = cc->state.hyperscanning.global_sync;
    phi->phi_network = global_sync * global_sync * cc->instance_count;

    /* Total phi with synergy coefficient */
    float synergy = cc->config.phi.synergy_coefficient;
    phi->phi_total = phi->phi_local + phi->phi_network * synergy;

    /* IIT decomposition (simplified) */
    phi->information = phi->phi_local;
    phi->integration = phi->phi_network;
    phi->exclusion = cc->instance_count > 1 ? 1.0f : 0.0f;

    /* Network topology metrics */
    phi->connectivity = cc->instance_count > 1 ?
        (float)(cc->instance_count - 1) / cc->instance_count : 0.0f;
    phi->modularity = 1.0f - global_sync;  /* Higher sync = lower modularity */
    phi->small_world_index = global_sync * phi->connectivity;

    /* Update statistics */
    cc->stats.avg_phi = (cc->stats.avg_phi * cc->stats.total_updates + phi->phi_total) /
                        (cc->stats.total_updates + 1);
    if (phi->phi_total > cc->stats.max_phi) {
        cc->stats.max_phi = phi->phi_total;
    }
}

/**
 * @brief Determine consciousness level from phi
 */
static collective_consciousness_level_t phi_to_level(float phi) {
    if (phi < 0.1f) return COLLECTIVE_CONSCIOUSNESS_NONE;
    if (phi < 0.3f) return COLLECTIVE_CONSCIOUSNESS_MINIMAL;
    if (phi < 0.5f) return COLLECTIVE_CONSCIOUSNESS_EMERGING;
    if (phi < 0.7f) return COLLECTIVE_CONSCIOUSNESS_PARTIAL;
    if (phi < 0.9f) return COLLECTIVE_CONSCIOUSNESS_UNIFIED;
    return COLLECTIVE_CONSCIOUSNESS_TRANSCENDENT;
}

/*=============================================================================
 * Helper Functions - Hyperscanning
 *===========================================================================*/

static void update_hyperscanning_state(collective_cognition_t* cc) {
    hyperscan_state_t* hs = &cc->state.hyperscanning;

    /* Compute global sync */
    hs->global_sync = compute_global_sync(cc);

    /* Compute band-specific sync */
    float gamma_total = 0.0f, theta_total = 0.0f, beta_total = 0.0f;
    uint32_t pair_count = 0;

    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (!cc->instances[i].active) continue;
        for (uint32_t j = i + 1; j < COLLECTIVE_MAX_INSTANCES; j++) {
            if (!cc->instances[j].active) continue;

            gamma_total += cc->pair_plv[i][j][SYNC_BAND_GAMMA];
            theta_total += cc->pair_plv[i][j][SYNC_BAND_THETA];
            beta_total += cc->pair_plv[i][j][SYNC_BAND_BETA];
            pair_count++;
        }
    }

    if (pair_count > 0) {
        hs->gamma_binding = gamma_total / pair_count;
        hs->theta_emotional = theta_total / pair_count;
        hs->beta_coordination = beta_total / pair_count;
    } else {
        hs->gamma_binding = 0.0f;
        hs->theta_emotional = 0.0f;
        hs->beta_coordination = 0.0f;
    }

    /* Check for entrainment */
    hs->is_entrained = hs->global_sync >= cc->config.hyperscanning.sync_threshold;

    if (hs->is_entrained && !cc->state.hyperscanning.is_entrained) {
        cc->stats.entrainment_events++;
    }

    /* Find leader (highest gamma power) */
    float max_gamma = -1.0f;
    uint32_t leader_id = 0;
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active &&
            cc->instances[i].band_power[SYNC_BAND_GAMMA] > max_gamma) {
            max_gamma = cc->instances[i].band_power[SYNC_BAND_GAMMA];
            leader_id = cc->instances[i].instance_id;
        }
    }
    hs->leader_instance_id = leader_id;
    hs->leader_influence = max_gamma > 0.0f ? max_gamma : 0.0f;

    /* Update average sync statistic */
    cc->stats.avg_sync = (cc->stats.avg_sync * cc->stats.total_updates + hs->global_sync) /
                         (cc->stats.total_updates + 1);
}

/*=============================================================================
 * Helper Functions - We-Mode
 *===========================================================================*/

static void update_we_mode_state(collective_cognition_t* cc) {
    we_mode_state_t* wm = &cc->state.we_mode;

    /* We-mode strength based on hyperscanning and phi */
    float sync = cc->state.hyperscanning.global_sync;
    float phi_norm = cc->state.phi.phi_total /
        (cc->instance_count > 0 ? cc->instance_count : 1.0f);

    wm->we_mode_strength = (sync + phi_norm) / 2.0f;
    wm->joint_commitment = sync;
    wm->mutual_responsiveness = cc->state.hyperscanning.theta_emotional;
    wm->role_understanding = cc->state.hyperscanning.beta_coordination;

    /* TODO: Track actual shared goals and joint attentions when subsystem implemented */
    wm->active_shared_goals = 0;
    wm->active_joint_attentions = 0;
}

/*=============================================================================
 * Helper Functions - Extended Mind
 *===========================================================================*/

static void update_extended_mind_state(collective_cognition_t* cc) {
    extended_mind_state_t* em = &cc->state.extended_mind;

    /* TODO: Implement when extended_mind subsystem is created */
    em->total_cognitive_capacity = 1.0f + (cc->instance_count * 0.1f);
    em->extended_ratio = 0.0f;
    em->integration_quality = 0.0f;
    em->active_extensions = 0;
    em->degraded_extensions = 0;
}

/*=============================================================================
 * Helper Functions - Aggregate State
 *===========================================================================*/

static void update_aggregate_state(collective_cognition_t* cc) {
    /* Collective capacity */
    cc->state.collective_capacity =
        cc->state.extended_mind.total_cognitive_capacity +
        (cc->state.hyperscanning.global_sync * cc->instance_count * 0.1f);

    /* Integration quality */
    cc->state.integration_quality =
        (cc->state.hyperscanning.global_sync +
         cc->state.phi.integration +
         cc->state.we_mode.we_mode_strength) / 3.0f;

    /* Overall consciousness level [0-1] */
    cc->state.consciousness_level = cc->state.phi.phi_total;
    if (cc->state.consciousness_level > 1.0f) {
        cc->state.consciousness_level = 1.0f;
    }

    /* Flow metrics */
    cc->state.information_flow_rate =
        cc->state.phi.information * cc->instance_count * 10.0f;  /* bits/sec estimate */
    cc->state.attention_coherence = cc->state.hyperscanning.gamma_binding;
    cc->state.goal_alignment = cc->state.we_mode.joint_commitment;

    /* Health indicators */
    cc->state.is_fragmented = cc->state.integration_quality <
                              cc->config.fragmentation_threshold;
    cc->state.is_overloaded = cc->state.collective_capacity > cc->config.overload_threshold;
    cc->state.active_instances = cc->instance_count;

    /* Update event stats */
    if (cc->state.is_fragmented) {
        cc->stats.fragmentation_events++;
    }
    if (cc->state.is_overloaded) {
        cc->stats.overload_events++;
    }

    cc->state.last_update_us = get_timestamp_us();
}

/*=============================================================================
 * Configuration API
 *===========================================================================*/

collective_cognition_config_t collective_cognition_default_config(void) {
    collective_cognition_config_t config = {
        .hyperscanning = hyperscanning_default_config(),
        .extended_mind = extended_mind_default_config(),
        .phi = collective_phi_default_config(),
        .intentionality = shared_intentionality_default_config(),

        .max_instances = COLLECTIVE_MAX_INSTANCES,
        .fragmentation_threshold = 0.3f,
        .overload_threshold = 1.5f,
        .enable_auto_balancing = true,
        .enable_bio_async = true,
        .update_interval_ms = 50
    };
    return config;
}

hyperscanning_config_t hyperscanning_default_config(void) {
    hyperscanning_config_t config = {
        .max_instances = COLLECTIVE_MAX_INSTANCES,
        .sync_threshold = 0.7f,
        .sample_rate_hz = 100,
        .enable_leader_detection = true,
        .enable_bio_async = true
    };
    return config;
}

extended_mind_config_t extended_mind_default_config(void) {
    extended_mind_config_t config = {
        .max_extensions = COLLECTIVE_MAX_EXTENSIONS,
        .trust_decay_rate = 0.1f,
        .integration_threshold = 0.8f,
        .enable_automatic_offload = true,
        .enable_bio_async = true
    };
    return config;
}

collective_phi_config_t collective_phi_default_config(void) {
    collective_phi_config_t config = {
        .aggregation_method = 3,  /* SYNERGISTIC */
        .synergy_coefficient = 0.5f,
        .min_instances_for_phi = 2,
        .coherence_weight = 0.3f,
        .enable_network_topology = true
    };
    return config;
}

shared_intentionality_config_t shared_intentionality_default_config(void) {
    shared_intentionality_config_t config = {
        .max_shared_goals = COLLECTIVE_MAX_SHARED_GOALS,
        .max_joint_attentions = COLLECTIVE_MAX_JOINT_ATTENTIONS,
        .commitment_threshold = 0.5f,
        .we_mode_threshold = 0.6f,
        .enable_role_negotiation = true,
        .enable_bio_async = true
    };
    return config;
}

/*=============================================================================
 * Lifecycle API
 *===========================================================================*/

collective_cognition_t* collective_cognition_create(
    const collective_cognition_config_t* config
) {
    collective_cognition_t* cc = nimcp_malloc(sizeof(collective_cognition_t));
    if (!cc) return NULL;

    memset(cc, 0, sizeof(collective_cognition_t));

    /* Apply configuration */
    if (config) {
        cc->config = *config;
    } else {
        cc->config = collective_cognition_default_config();
    }

    /* Initialize instances */
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        cc->instances[i].active = false;
        cc->instances[i].local_phi = 0.3f;  /* Default individual phi */
        cc->instances[i].atp_level = 1.0f;
        cc->instances[i].fatigue_level = 0.0f;

        /* Initialize band states */
        for (int b = 0; b < SYNC_BAND_COUNT; b++) {
            cc->instances[i].band_power[b] = 0.5f;
            cc->instances[i].band_phase[b] = 0.0f;
        }
    }

    /* Create subsystem handles */
    cc->hyperscanning = hyperscanning_create(&cc->config.hyperscanning);
    cc->extended_mind = extended_mind_create(&cc->config.extended_mind);
    cc->phi_system = collective_phi_create(&cc->config.phi);
    cc->intentionality = shared_intentionality_create(&cc->config.intentionality);

    /* Check for allocation failures - subsystems are optional but recommended */
    if (!cc->hyperscanning || !cc->extended_mind ||
        !cc->phi_system || !cc->intentionality) {
        /* Clean up any successfully created subsystems */
        if (cc->hyperscanning) hyperscanning_destroy(cc->hyperscanning);
        if (cc->extended_mind) extended_mind_destroy(cc->extended_mind);
        if (cc->phi_system) collective_phi_destroy(cc->phi_system);
        if (cc->intentionality) shared_intentionality_destroy(cc->intentionality);
        nimcp_free(cc);
        return NULL;
    }

    cc->initialized = true;
    cc->last_update_us = get_timestamp_us();

    return cc;
}

void collective_cognition_destroy(collective_cognition_t* cc) {
    if (!cc) return;

    /* Destroy subsystem handles */
    if (cc->hyperscanning) hyperscanning_destroy(cc->hyperscanning);
    if (cc->extended_mind) extended_mind_destroy(cc->extended_mind);
    if (cc->phi_system) collective_phi_destroy(cc->phi_system);
    if (cc->intentionality) shared_intentionality_destroy(cc->intentionality);

    /* Disconnect bio-async if connected */
    if (cc->bio_async_connected) {
        collective_cognition_disconnect_bio_async(cc);
    }

    nimcp_free(cc);
}

int collective_cognition_reset(collective_cognition_t* cc) {
    if (!cc) return -1;

    /* Reset all instances */
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        cc->instances[i].active = false;
    }
    cc->instance_count = 0;

    /* Reset state */
    memset(&cc->state, 0, sizeof(cc->state));
    memset(&cc->stats, 0, sizeof(cc->stats));
    memset(cc->pair_plv, 0, sizeof(cc->pair_plv));

    cc->last_update_us = get_timestamp_us();

    return 0;
}

/*=============================================================================
 * Instance Management API
 *===========================================================================*/

int collective_cognition_register_instance(
    collective_cognition_t* cc,
    uint32_t instance_id,
    brain_t* brain
) {
    if (!cc) return -1;

    /* Check if already registered */
    if (find_instance(cc, instance_id)) {
        return -1;  /* Already exists */
    }

    /* Find free slot */
    registered_instance_t* slot = find_free_slot(cc);
    if (!slot) {
        return -1;  /* No free slots */
    }

    /* Initialize instance */
    slot->instance_id = instance_id;
    slot->brain = brain;
    slot->active = true;
    slot->local_phi = 0.3f;
    slot->atp_level = 1.0f;
    slot->fatigue_level = 0.0f;
    slot->last_heartbeat_us = get_timestamp_us();
    slot->missed_heartbeats = 0;

    /* Initialize band states with some variation */
    for (int b = 0; b < SYNC_BAND_COUNT; b++) {
        slot->band_power[b] = 0.5f + (instance_id % 10) * 0.01f;
        slot->band_phase[b] = (instance_id * 0.5f);  /* Different starting phases */
    }

    cc->instance_count++;
    cc->stats.instances_joined++;

    return 0;
}

int collective_cognition_unregister_instance(
    collective_cognition_t* cc,
    uint32_t instance_id
) {
    if (!cc) return -1;

    registered_instance_t* inst = find_instance(cc, instance_id);
    if (!inst) {
        return -1;  /* Not found */
    }

    inst->active = false;
    inst->brain = NULL;
    cc->instance_count--;
    cc->stats.instances_left++;

    return 0;
}

uint32_t collective_cognition_instance_count(const collective_cognition_t* cc) {
    return cc ? cc->instance_count : 0;
}

bool collective_cognition_has_instance(
    const collective_cognition_t* cc,
    uint32_t instance_id
) {
    if (!cc) return false;

    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active && cc->instances[i].instance_id == instance_id) {
            return true;
        }
    }
    return false;
}

/*=============================================================================
 * Update API
 *===========================================================================*/

int collective_cognition_update(collective_cognition_t* cc) {
    if (!cc || !cc->initialized) return -1;

    /* Update all subsystem states */
    update_hyperscanning_state(cc);
    compute_collective_phi(cc);
    update_we_mode_state(cc);
    update_extended_mind_state(cc);

    /* Update aggregate state */
    update_aggregate_state(cc);

    /* Update statistics */
    cc->stats.total_updates++;
    cc->last_update_us = get_timestamp_us();

    /* Simulate phase evolution for testing */
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active) {
            for (int b = 0; b < SYNC_BAND_COUNT; b++) {
                /* Phase drift with some coupling */
                float base_freq = 2.0f + b * 5.0f;  /* Different freq per band */
                cc->instances[i].band_phase[b] += base_freq * 0.01f;
                if (cc->instances[i].band_phase[b] > 6.28f) {
                    cc->instances[i].band_phase[b] -= 6.28f;
                }
            }
        }
    }

    return 0;
}

/*=============================================================================
 * State Query API
 *===========================================================================*/

int collective_cognition_get_state(
    const collective_cognition_t* cc,
    collective_cognition_state_t* state
) {
    if (!cc || !state) return -1;

    *state = cc->state;
    return 0;
}

collective_consciousness_level_t collective_cognition_get_consciousness_level(
    const collective_cognition_t* cc
) {
    if (!cc) return COLLECTIVE_CONSCIOUSNESS_NONE;

    return phi_to_level(cc->state.phi.phi_total);
}

int collective_cognition_get_hyperscan_state(
    const collective_cognition_t* cc,
    hyperscan_state_t* state
) {
    if (!cc || !state) return -1;

    *state = cc->state.hyperscanning;
    return 0;
}

int collective_cognition_get_extended_mind_state(
    const collective_cognition_t* cc,
    extended_mind_state_t* state
) {
    if (!cc || !state) return -1;

    *state = cc->state.extended_mind;
    return 0;
}

int collective_cognition_get_phi(
    const collective_cognition_t* cc,
    collective_phi_t* phi
) {
    if (!cc || !phi) return -1;

    *phi = cc->state.phi;
    return 0;
}

int collective_cognition_get_we_mode(
    const collective_cognition_t* cc,
    we_mode_state_t* state
) {
    if (!cc || !state) return -1;

    *state = cc->state.we_mode;
    return 0;
}

/*=============================================================================
 * Component Access API
 *===========================================================================*/

hyperscanning_t* collective_cognition_get_hyperscanning(collective_cognition_t* cc) {
    return cc ? (hyperscanning_t*)cc->hyperscanning : NULL;
}

extended_mind_t* collective_cognition_get_extended_mind(collective_cognition_t* cc) {
    return cc ? (extended_mind_t*)cc->extended_mind : NULL;
}

collective_phi_system_t* collective_cognition_get_phi_system(collective_cognition_t* cc) {
    return cc ? (collective_phi_system_t*)cc->phi_system : NULL;
}

shared_intentionality_t* collective_cognition_get_intentionality(collective_cognition_t* cc) {
    return cc ? (shared_intentionality_t*)cc->intentionality : NULL;
}

/*=============================================================================
 * Bio-Async Integration API
 *===========================================================================*/

int collective_cognition_connect_bio_async(
    collective_cognition_t* cc,
    bio_router_t* router
) {
    if (!cc || !router) return -1;

    cc->bio_router = router;
    cc->bio_async_connected = true;

    /* TODO: Register message handlers when bio-async integration is implemented */

    return 0;
}

int collective_cognition_disconnect_bio_async(collective_cognition_t* cc) {
    if (!cc) return -1;

    /* TODO: Unregister message handlers */

    cc->bio_router = NULL;
    cc->bio_async_connected = false;

    return 0;
}

bool collective_cognition_is_bio_async_connected(const collective_cognition_t* cc) {
    return cc ? cc->bio_async_connected : false;
}

/*=============================================================================
 * Load Balancing API
 *===========================================================================*/

int collective_cognition_balance_load(collective_cognition_t* cc) {
    if (!cc) return -1;

    /* Find overloaded and underloaded instances */
    int overloaded_idx = -1;
    int underloaded_idx = -1;
    float max_load = 0.0f;
    float min_load = 2.0f;

    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (!cc->instances[i].active) continue;

        float load = cc->instances[i].fatigue_level;
        if (load > max_load) {
            max_load = load;
            overloaded_idx = (int)i;
        }
        if (load < min_load) {
            min_load = load;
            underloaded_idx = (int)i;
        }
    }

    /* Balance if there's significant imbalance */
    if (overloaded_idx >= 0 && underloaded_idx >= 0 &&
        max_load - min_load > 0.3f) {
        /* Simulate load transfer */
        float transfer = (max_load - min_load) * 0.25f;
        cc->instances[overloaded_idx].fatigue_level -= transfer;
        cc->instances[underloaded_idx].fatigue_level += transfer;
        return 1;  /* One task redistributed */
    }

    return 0;
}

int collective_cognition_offload_task(
    collective_cognition_t* cc,
    uint32_t from_instance,
    uint32_t to_instance,
    const void* task_data,
    size_t task_size
) {
    if (!cc || !task_data) return -1;

    registered_instance_t* from = find_instance(cc, from_instance);
    registered_instance_t* to = find_instance(cc, to_instance);

    if (!from || !to) return -1;

    /* Update load estimates */
    float task_load = task_size * 0.0001f;  /* Estimate load from size */
    from->fatigue_level -= task_load;
    to->fatigue_level += task_load;

    if (from->fatigue_level < 0.0f) from->fatigue_level = 0.0f;
    if (to->fatigue_level > 1.0f) to->fatigue_level = 1.0f;

    cc->stats.bytes_transferred += task_size;

    return 0;
}

/*=============================================================================
 * Statistics API
 *===========================================================================*/

int collective_cognition_get_stats(
    const collective_cognition_t* cc,
    collective_cognition_stats_t* stats
) {
    if (!cc || !stats) return -1;

    *stats = cc->stats;
    return 0;
}

void collective_cognition_reset_stats(collective_cognition_t* cc) {
    if (!cc) return;

    memset(&cc->stats, 0, sizeof(cc->stats));
}

/*=============================================================================
 * Utility API
 *===========================================================================*/

const char* collective_consciousness_level_name(collective_consciousness_level_t level) {
    switch (level) {
        case COLLECTIVE_CONSCIOUSNESS_NONE:        return "NONE";
        case COLLECTIVE_CONSCIOUSNESS_MINIMAL:     return "MINIMAL";
        case COLLECTIVE_CONSCIOUSNESS_EMERGING:    return "EMERGING";
        case COLLECTIVE_CONSCIOUSNESS_PARTIAL:     return "PARTIAL";
        case COLLECTIVE_CONSCIOUSNESS_UNIFIED:     return "UNIFIED";
        case COLLECTIVE_CONSCIOUSNESS_TRANSCENDENT: return "TRANSCENDENT";
        default: return "UNKNOWN";
    }
}

const char* sync_band_name(sync_band_t band) {
    switch (band) {
        case SYNC_BAND_DELTA: return "DELTA";
        case SYNC_BAND_THETA: return "THETA";
        case SYNC_BAND_ALPHA: return "ALPHA";
        case SYNC_BAND_BETA:  return "BETA";
        case SYNC_BAND_GAMMA: return "GAMMA";
        default: return "UNKNOWN";
    }
}

const char* extension_type_name(extension_type_t type) {
    switch (type) {
        case EXT_TYPE_MEMORY:        return "MEMORY";
        case EXT_TYPE_PERCEPTION:    return "PERCEPTION";
        case EXT_TYPE_REASONING:     return "REASONING";
        case EXT_TYPE_ACTION:        return "ACTION";
        case EXT_TYPE_COMMUNICATION: return "COMMUNICATION";
        default: return "UNKNOWN";
    }
}

void collective_cognition_dump(const collective_cognition_t* cc) {
    if (!cc) {
        printf("Collective Cognition: NULL\n");
        return;
    }

    printf("=== Collective Cognition State ===\n");
    printf("Initialized: %s\n", cc->initialized ? "yes" : "no");
    printf("Instances: %u / %d\n", cc->instance_count, COLLECTIVE_MAX_INSTANCES);

    /* List instances */
    printf("\nRegistered Instances:\n");
    for (uint32_t i = 0; i < COLLECTIVE_MAX_INSTANCES; i++) {
        if (cc->instances[i].active) {
            printf("  [%u] ID=%u phi=%.3f atp=%.3f fatigue=%.3f\n",
                   i, cc->instances[i].instance_id,
                   cc->instances[i].local_phi,
                   cc->instances[i].atp_level,
                   cc->instances[i].fatigue_level);
        }
    }

    /* Hyperscanning state */
    printf("\nHyperscanning:\n");
    printf("  Global sync: %.3f\n", cc->state.hyperscanning.global_sync);
    printf("  Gamma binding: %.3f\n", cc->state.hyperscanning.gamma_binding);
    printf("  Theta emotional: %.3f\n", cc->state.hyperscanning.theta_emotional);
    printf("  Beta coordination: %.3f\n", cc->state.hyperscanning.beta_coordination);
    printf("  Entrained: %s\n", cc->state.hyperscanning.is_entrained ? "yes" : "no");
    printf("  Leader: %u (influence: %.3f)\n",
           cc->state.hyperscanning.leader_instance_id,
           cc->state.hyperscanning.leader_influence);

    /* Phi metrics */
    printf("\nCollective Phi (IIT):\n");
    printf("  Local phi: %.3f\n", cc->state.phi.phi_local);
    printf("  Network phi: %.3f\n", cc->state.phi.phi_network);
    printf("  Total phi: %.3f\n", cc->state.phi.phi_total);
    printf("  Consciousness level: %s\n",
           collective_consciousness_level_name(
               phi_to_level(cc->state.phi.phi_total)));

    /* We-mode */
    printf("\nWe-Mode (Shared Intentionality):\n");
    printf("  Strength: %.3f\n", cc->state.we_mode.we_mode_strength);
    printf("  Joint commitment: %.3f\n", cc->state.we_mode.joint_commitment);
    printf("  Mutual responsiveness: %.3f\n", cc->state.we_mode.mutual_responsiveness);
    printf("  Role understanding: %.3f\n", cc->state.we_mode.role_understanding);

    /* Extended mind */
    printf("\nExtended Mind:\n");
    printf("  Total capacity: %.3f\n", cc->state.extended_mind.total_cognitive_capacity);
    printf("  Active extensions: %u\n", cc->state.extended_mind.active_extensions);

    /* Aggregate state */
    printf("\nAggregate State:\n");
    printf("  Collective capacity: %.3f\n", cc->state.collective_capacity);
    printf("  Integration quality: %.3f\n", cc->state.integration_quality);
    printf("  Consciousness level: %.3f\n", cc->state.consciousness_level);
    printf("  Fragmented: %s\n", cc->state.is_fragmented ? "yes" : "no");
    printf("  Overloaded: %s\n", cc->state.is_overloaded ? "yes" : "no");

    /* Statistics */
    printf("\nStatistics:\n");
    printf("  Total updates: %lu\n", (unsigned long)cc->stats.total_updates);
    printf("  Instances joined: %lu\n", (unsigned long)cc->stats.instances_joined);
    printf("  Instances left: %lu\n", (unsigned long)cc->stats.instances_left);
    printf("  Entrainment events: %lu\n", (unsigned long)cc->stats.entrainment_events);
    printf("  Avg phi: %.3f, Max phi: %.3f\n", cc->stats.avg_phi, cc->stats.max_phi);
    printf("  Avg sync: %.3f\n", cc->stats.avg_sync);

    printf("\nBio-Async: %s\n", cc->bio_async_connected ? "connected" : "disconnected");
}
