//=============================================================================
// nimcp_brain_pr_memory.c - Prime Resonant Memory Brain Integration
//=============================================================================
/**
 * @file nimcp_brain_pr_memory.c
 * @brief Implementation of brain-PR memory integration
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "core/brain/internal/nimcp_brain_pr_memory.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/memory/core/nimcp_z_ladder.h"
#include "cognitive/memory/core/nimcp_theta_gamma.h"
#include "cognitive/memory/core/nimcp_entanglement.h"
#include "cognitive/memory/core/nimcp_pr_memory_node.h"
#include "cognitive/memory/core/nimcp_prime_signature.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/logging/nimcp_logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Forward declarations for cycle coordinator — kept loose to avoid
 * dragging the full header into this TU. Values must match
 * brain_cycle_type_t values in nimcp_brain_cycle_coordinator.h. */
#define CYCLE_TYPE_LONG_TERM_MEMORY 9
extern int brain_cycle_coordinator_register(void* coord, int type,
                                            void* cycle_handle,
                                            void* health_fn);
extern int brain_cycle_coordinator_unregister(void* coord, int type);
extern int brain_cycle_coordinator_notify_tick(void* coord, int type,
                                               uint64_t duration_us);
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_pr_memory, MESH_ADAPTER_CATEGORY_COGNITIVE)


//=============================================================================
// Configuration
//=============================================================================

brain_pr_memory_config_t brain_pr_memory_config_default(void) {
    brain_pr_memory_config_t config;
    memset(&config, 0, sizeof(config));

    /* Z-Ladder defaults (biologically-inspired capacities) */
    config.z0_capacity = 9;              /* Miller's 7±2 working memory */
    config.z1_capacity = 100;            /* Short-term buffer */
    config.z2_capacity = 10000;          /* Long-term consolidation */
    config.z3_capacity = 100000;         /* Permanent semantic/procedural */

    /* Theta-gamma defaults (hippocampal rhythms) */
    config.theta_freq_hz = 6.0f;         /* 4-8 Hz theta band center */
    config.gamma_freq_hz = 40.0f;        /* 30-80 Hz gamma band center (not used directly) */
    config.enable_phase_gating = true;

    /* Entanglement defaults */
    config.max_entangle_nodes = 50000;
    config.max_entangle_edges = 200000;
    config.auto_link_threshold = 0.6f;

    /* Consolidation timing */
    config.consolidation_interval_us = 100000;  /* 100ms = 10 Hz update rate */
    config.enable_sleep_boost = true;

    return config;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

bool nimcp_brain_pr_memory_init(struct brain_struct* brain, const brain_pr_memory_config_t* config) {
    if (!brain) {
        fprintf(stderr, "[PR_MEMORY] Init failed: NULL brain\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_pr_memory_init: brain is NULL");
        return false;
    }

    /* Check if already initialized */
    if (brain->pr_z_ladder || brain->pr_theta_gamma || brain->pr_entanglement) {
        /* Already initialized - this is a no-op */
        return true;
    }

    /* Use defaults if no config provided */
    brain_pr_memory_config_t cfg = config ? *config : brain_pr_memory_config_default();

    /* Phase E6: create a unified_mem_manager for the pr_node_manager
     * so node data_handles get populated on insert — this is what makes
     * pr_memory_node_read return real bytes and lets full-fidelity
     * landmark checkpoints round-trip features, not just signatures. */
    unified_mem_config_t um_cfg = unified_mem_default_config();
    unified_mem_manager_t umm = unified_mem_create(&um_cfg);
    if (!umm) {
        fprintf(stderr, "[PR_MEMORY] Failed to create unified_mem_manager\n");
        goto cleanup;
    }
    brain->pr_memory_unified_mm = (void*)umm;

    /* Phase E2: create a node manager FIRST so z_ladder can reference it.
     * Without this, z_ladder is initialized with node_manager=NULL and
     * has no way to allocate the pr_memory_node_t instances it tracks. */
    pr_node_manager_config_t nm_cfg = pr_node_manager_default_config();
    nm_cfg.mem_manager = umm;  /* Phase E6: wire the unified memory manager */
    pr_node_manager_t nm = pr_node_manager_create(&nm_cfg);
    if (!nm) {
        fprintf(stderr, "[PR_MEMORY] Failed to create node manager\n");
        goto cleanup;
    }
    brain->pr_node_manager = (void*)nm;

    /* Initialize Z-Ladder */
    z_ladder_config_t z_config = z_ladder_default_config();
    /* Set per-tier capacities using tier_configs array */
    z_config.tier_configs[PR_MEMORY_TIER_Z0].capacity = cfg.z0_capacity;
    z_config.tier_configs[PR_MEMORY_TIER_Z1].capacity = cfg.z1_capacity;
    z_config.tier_configs[PR_MEMORY_TIER_Z2].capacity = cfg.z2_capacity;
    z_config.tier_configs[PR_MEMORY_TIER_Z3].capacity = cfg.z3_capacity;
    z_config.node_manager = nm;  /* Phase E2: give ladder a way to allocate nodes */

    brain->pr_z_ladder = z_ladder_create(&z_config);
    if (!brain->pr_z_ladder) {
        fprintf(stderr, "[PR_MEMORY] Failed to create Z-Ladder\n");
        goto cleanup;
    }

    /* Initialize Theta-Gamma coupling */
    theta_gamma_config_t tg_config = theta_gamma_config_default();
    tg_config.theta_freq_default = cfg.theta_freq_hz;
    /* Note: theta_gamma_config_t doesn't have gamma_freq_default or phase_gating_enabled
     * Gamma frequencies are configured via low/high bands instead */

    brain->pr_theta_gamma = theta_gamma_create(&tg_config);
    if (!brain->pr_theta_gamma) {
        fprintf(stderr, "[PR_MEMORY] Failed to create theta-gamma manager\n");
        goto cleanup;
    }

    /* Initialize Entanglement graph */
    entangle_config_t e_config = entangle_config_default();
    e_config.initial_node_capacity = cfg.max_entangle_nodes / 10;  /* Start smaller */
    e_config.initial_edge_capacity = cfg.max_entangle_edges / 10;
    e_config.auto_link_threshold = cfg.auto_link_threshold;

    brain->pr_entanglement = entangle_graph_create(&e_config);
    if (!brain->pr_entanglement) {
        fprintf(stderr, "[PR_MEMORY] Failed to create entanglement graph\n");
        goto cleanup;
    }

    /* Set brain-level configuration */
    brain->pr_memory_enabled = true;
    brain->pr_lazy_init = false;  /* Explicitly initialized */
    brain->last_pr_consolidation_us = 0;
    brain->pr_consolidation_interval_us = cfg.consolidation_interval_us;

    /* Phase E3: auto-insertion defaults — OFF, caller opts in via
     * brain.long_term_set_auto_insert(...). Reasonable thresholds match
     * typical training confidence distributions. */
    brain->pr_auto_insert_enabled      = false;
    brain->pr_auto_insert_confidence   = 0.7f;
    brain->pr_auto_landmark_confidence = 0.95f;
    brain->pr_auto_insert_count        = 0;
    brain->pr_auto_landmark_count      = 0;

    /* Phase E6: spin up the autonomous pr_memory driver thread. Runs at
     * 100ms cadence, calls pr_memory_tick and notifies the cycle
     * coordinator's BRAIN_CYCLE_LONG_TERM_MEMORY. This makes memory
     * consolidation + landmark hygiene independent of brain_learn_vector. */
    __atomic_store_n(&brain->pr_memory_driver_stop, 0, __ATOMIC_RELEASE);
    brain->pr_memory_driver_ticks = 0;
    extern void* nimcp_brain_pr_memory_driver_run(void* arg);  /* defined below */
    nimcp_thread_t* thr = (nimcp_thread_t*)calloc(1, sizeof(nimcp_thread_t));
    if (thr) {
        if (nimcp_thread_create(thr, nimcp_brain_pr_memory_driver_run,
                                brain, NULL) == NIMCP_OK) {
            brain->pr_memory_driver_thread = (void*)thr;
        } else {
            free(thr);
            NIMCP_LOGGING_WARN("pr_memory: driver thread failed to start — "
                               "memory will only evolve via brain_learn_vector");
        }
    }

    return true;

cleanup:
    /* Cleanup on failure — Z-Ladder before its node manager since ladder
     * holds the tracked nodes allocated by the manager. */
    if (brain->pr_z_ladder) {
        z_ladder_destroy(brain->pr_z_ladder);
        brain->pr_z_ladder = NULL;
    }
    if (brain->pr_node_manager) {
        pr_node_manager_destroy((pr_node_manager_t)brain->pr_node_manager);
        brain->pr_node_manager = NULL;
    }
    if (brain->pr_memory_unified_mm) {
        unified_mem_destroy((unified_mem_manager_t)brain->pr_memory_unified_mm);
        brain->pr_memory_unified_mm = NULL;
    }
    if (brain->pr_theta_gamma) {
        theta_gamma_destroy(brain->pr_theta_gamma);
        brain->pr_theta_gamma = NULL;
    }
    if (brain->pr_entanglement) {
        entangle_graph_destroy(brain->pr_entanglement);
        brain->pr_entanglement = NULL;
    }
    brain->pr_memory_enabled = false;
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_brain_pr_memory_init: validation failed");
    return false;
}

void nimcp_brain_pr_memory_destroy(struct brain_struct* brain) {
    if (!brain) {
        return;
    }

    /* Phase E6: stop the autonomous driver thread first so it can't
     * observe torn-down state. Signal → join → free. */
    if (brain->pr_memory_driver_thread) {
        __atomic_store_n(&brain->pr_memory_driver_stop, 1, __ATOMIC_RELEASE);
        nimcp_thread_t* thr = (nimcp_thread_t*)brain->pr_memory_driver_thread;
        nimcp_thread_join(*thr, NULL);
        free(thr);
        brain->pr_memory_driver_thread = NULL;
    }

    /* Destroy Z-Ladder first — its tracked nodes are allocated by the
     * node manager, so the manager must outlive the ladder. */
    if (brain->pr_z_ladder) {
        z_ladder_destroy(brain->pr_z_ladder);
        brain->pr_z_ladder = NULL;
    }
    if (brain->pr_node_manager) {
        pr_node_manager_destroy((pr_node_manager_t)brain->pr_node_manager);
        brain->pr_node_manager = NULL;
    }

    /* Phase E6: destroy unified_mem_manager AFTER all its allocations
     * are released (nodes freed by node_manager_destroy above). */
    if (brain->pr_memory_unified_mm) {
        unified_mem_destroy((unified_mem_manager_t)brain->pr_memory_unified_mm);
        brain->pr_memory_unified_mm = NULL;
    }

    /* Destroy Theta-Gamma manager */
    if (brain->pr_theta_gamma) {
        theta_gamma_destroy(brain->pr_theta_gamma);
        brain->pr_theta_gamma = NULL;
    }

    /* Destroy Entanglement graph */
    if (brain->pr_entanglement) {
        entangle_graph_destroy(brain->pr_entanglement);
        brain->pr_entanglement = NULL;
    }

    brain->pr_memory_enabled = false;
}

//=============================================================================
// Update Functions
//=============================================================================

bool nimcp_brain_pr_memory_tick(struct brain_struct* brain, uint64_t current_time_us) {
    if (!brain || !brain->pr_memory_enabled) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_pr_memory_tick: required parameter is NULL (brain, brain->pr_memory_enabled)");
        return false;
    }

    /* Advance theta-gamma phase */
    if (brain->pr_theta_gamma) {
        /* Calculate time delta in nanoseconds */
        uint64_t dt_us = current_time_us - brain->last_pr_consolidation_us;
        if (dt_us > 1000000) dt_us = 100000;  /* Clamp large jumps to 100ms */
        uint64_t dt_ns = dt_us * 1000;  /* Convert to nanoseconds */

        theta_gamma_update(brain->pr_theta_gamma, dt_ns);
    }

    /* Check if consolidation interval has elapsed */
    if (current_time_us - brain->last_pr_consolidation_us < brain->pr_consolidation_interval_us) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_brain_pr_memory_tick: validation failed");
        return false;  /* Not time for consolidation yet */
    }

    /* Trigger consolidation */
    bool consolidation_triggered = false;

    if (brain->pr_z_ladder) {
        /* Run full consolidation (applies decay, promotes/demotes, evicts) */
        z_ladder_consolidate(brain->pr_z_ladder);
        consolidation_triggered = true;

        /* Phase E4/E5: periodic landmark hygiene — every 100 consolidations
         * reclaim any ghost landmark slots (node evicted elsewhere, slot
         * still marked in_use). Bounded O(256) work, rare enough that the
         * cost is amortized. */
        brain->pr_consolidation_count++;
        if ((brain->pr_consolidation_count % 100u) == 0u) {
            size_t reclaimed = z_ladder_landmark_prune_stale(brain->pr_z_ladder);
            if (reclaimed > 0) {
                NIMCP_LOGGING_INFO("pr_memory_tick: pruned %zu stale landmark slot(s)",
                                   reclaimed);
            }
        }
    }

    /* Update timestamp */
    brain->last_pr_consolidation_us = current_time_us;

    return consolidation_triggered;
}

bool nimcp_brain_pr_memory_is_initialized(const struct brain_struct* brain) {
    if (!brain) {
        return false;
    }
    return brain->pr_memory_enabled &&
           brain->pr_z_ladder != NULL &&
           brain->pr_theta_gamma != NULL &&
           brain->pr_entanglement != NULL;
}

//=============================================================================
// Accessor Functions
//=============================================================================

struct z_ladder_struct* nimcp_brain_get_z_ladder(struct brain_struct* brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }
    return brain->pr_z_ladder;
}

struct theta_gamma_manager_internal* nimcp_brain_get_theta_gamma(struct brain_struct* brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }
    return brain->pr_theta_gamma;
}

struct entangle_graph_struct* nimcp_brain_get_entanglement(struct brain_struct* brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }
    return brain->pr_entanglement;
}

//=============================================================================
// Statistics
//=============================================================================

bool nimcp_brain_pr_memory_get_stats(const struct brain_struct* brain, brain_pr_memory_stats_t* stats) {
    if (!brain || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_pr_memory_get_stats: required parameter is NULL (brain, stats)");
        return false;
    }

    if (!brain->pr_memory_enabled) {
        memset(stats, 0, sizeof(*stats));
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_pr_memory_get_stats: brain->pr_memory_enabled is NULL");
        return false;
    }

    memset(stats, 0, sizeof(*stats));

    /* Z-Ladder statistics */
    if (brain->pr_z_ladder) {
        z_ladder_stats_t z_stats;
        if (z_ladder_get_stats(brain->pr_z_ladder, &z_stats)) {
            /* Use tier_counts array instead of z0_count etc. */
            stats->z0_count = (uint32_t)z_stats.tier_counts[PR_MEMORY_TIER_Z0];
            stats->z1_count = (uint32_t)z_stats.tier_counts[PR_MEMORY_TIER_Z1];
            stats->z2_count = (uint32_t)z_stats.tier_counts[PR_MEMORY_TIER_Z2];
            stats->z3_count = (uint32_t)z_stats.tier_counts[PR_MEMORY_TIER_Z3];
            /* Sum promotions and demotions across tiers */
            stats->total_promotions = z_stats.promotions[0] + z_stats.promotions[1] + z_stats.promotions[2];
            stats->total_demotions = z_stats.demotions[0] + z_stats.demotions[1] + z_stats.demotions[2];
            stats->total_evictions = z_stats.evictions[0] + z_stats.evictions[1] +
                                     z_stats.evictions[2] + z_stats.evictions[3];
        }
    }

    /* Theta-gamma statistics */
    if (brain->pr_theta_gamma) {
        stats->current_theta_phase = theta_gamma_get_theta_phase(brain->pr_theta_gamma);
        stats->current_gamma_amplitude = theta_gamma_get_gamma_phase(brain->pr_theta_gamma);

        /* Check encoding/retrieval windows (encoding: 0-90°, retrieval: 180-270°) */
        float phase = stats->current_theta_phase;
        stats->is_encoding_window = (phase >= 0.0f && phase < 90.0f);
        stats->is_retrieval_window = (phase >= 180.0f && phase < 270.0f);
    }

    /* Entanglement statistics */
    if (brain->pr_entanglement) {
        entangle_stats_t e_stats;
        if (entangle_get_stats(brain->pr_entanglement, &e_stats)) {
            stats->entangle_node_count = (uint32_t)e_stats.num_nodes;
            stats->entangle_edge_count = (uint32_t)e_stats.num_edges;
            stats->avg_node_degree = e_stats.avg_degree;
        }
    }

    /* Timing statistics */
    stats->last_consolidation_us = brain->last_pr_consolidation_us;

    return true;
}

/*============================================================================
 * Phase E6 — autonomous driver thread
 *
 * Runs at 100ms cadence. Drives pr_memory_tick (consolidation + landmark
 * hygiene) and notifies the cycle coordinator so health/timing
 * observability works even when brain_learn_vector is idle.
 *
 * The first call registers BRAIN_CYCLE_LONG_TERM_MEMORY with the
 * coordinator (if present). Unregisters on exit. Exits when
 * brain->pr_memory_driver_stop becomes non-zero.
 *==========================================================================*/
void* nimcp_brain_pr_memory_driver_run(void* arg) {
    struct brain_struct* brain = (struct brain_struct*)arg;
    if (!brain) return NULL;

    /* Lazy register with the cycle coordinator — harmless if absent. */
    bool registered = false;
    if (brain->cycle_coordinator_enabled && brain->cycle_coordinator) {
        int rc = brain_cycle_coordinator_register(
            (void*)brain->cycle_coordinator,
            CYCLE_TYPE_LONG_TERM_MEMORY,
            (void*)brain, NULL);
        registered = (rc == 0);
    }

    const long sleep_ns = 100L * 1000L * 1000L;  /* 100ms */
    while (__atomic_load_n(&brain->pr_memory_driver_stop,
                           __ATOMIC_ACQUIRE) == 0) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t now_us = (uint64_t)t0.tv_sec * 1000000ull +
                          (uint64_t)t0.tv_nsec / 1000ull;

        if (brain->pr_memory_enabled && brain->pr_z_ladder) {
            (void)nimcp_brain_pr_memory_tick(brain, now_us);
            brain->pr_memory_driver_ticks++;

            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (registered) {
                uint64_t dur_us = ((uint64_t)t1.tv_sec - (uint64_t)t0.tv_sec) * 1000000ull +
                                  ((uint64_t)t1.tv_nsec - (uint64_t)t0.tv_nsec) / 1000ull;
                brain_cycle_coordinator_notify_tick(
                    (void*)brain->cycle_coordinator,
                    CYCLE_TYPE_LONG_TERM_MEMORY, dur_us);
            }
        }

        struct timespec sleep_ts = { 0, sleep_ns };
        nanosleep(&sleep_ts, NULL);
    }

    if (registered) {
        brain_cycle_coordinator_unregister(
            (void*)brain->cycle_coordinator,
            CYCLE_TYPE_LONG_TERM_MEMORY);
    }
    return NULL;
}

/*============================================================================
 * Phase E3 — auto-insertion helper. Called from brain_learn_vector's
 * tick chain. No-op when pr_auto_insert_enabled is false.
 *==========================================================================*/

void nimcp_brain_pr_memory_auto_insert(struct brain_struct* brain,
                                        const float* features,
                                        uint32_t num_features,
                                        float confidence) {
    if (!brain || !features || num_features == 0) return;
    if (!brain->pr_auto_insert_enabled) return;
    if (!brain->pr_z_ladder || !brain->pr_node_manager) return;
    if (confidence < brain->pr_auto_insert_confidence) return;

    /* Cap features to 4096 (same bound as the Python insert path). */
    size_t n = num_features > 4096u ? 4096u : num_features;

    pr_node_config_t cfg = pr_memory_node_default_config();
    cfg.initial_tier       = PR_MEMORY_TIER_Z0;
    cfg.initial_strength   = confidence;
    cfg.compute_signature  = true;

    pr_memory_node_t* node = pr_memory_node_create(
        (pr_node_manager_t)brain->pr_node_manager,
        (const void*)features, n * sizeof(float), &cfg);
    if (!node) return;

    z_ladder_error_t ins = z_ladder_insert(
        brain->pr_z_ladder, node, PR_MEMORY_TIER_Z0);
    if (ins != Z_LADDER_SUCCESS) {
        pr_memory_node_destroy(node);
        return;
    }
    brain->pr_auto_insert_count++;

    /* Promote to landmark on very-high-confidence training events. */
    if (confidence >= brain->pr_auto_landmark_confidence) {
        uint64_t id = pr_memory_node_get_id(node);
        if (z_ladder_mark_landmark(brain->pr_z_ladder, id,
                                    "auto_high_confidence") == Z_LADDER_SUCCESS) {
            brain->pr_auto_landmark_count++;
            NIMCP_LOGGING_DEBUG("pr_memory_auto_insert: landmarked node %llu "
                                "(confidence=%.3f)",
                                (unsigned long long)id, (double)confidence);
        }
    }
}

/*============================================================================
 * Phase E4 #4 — event-triggered landmark promotion.
 *==========================================================================*/
uint64_t nimcp_brain_pr_memory_promote_event(struct brain_struct* brain,
                                              const float* features,
                                              uint32_t num_features,
                                              const char* reason) {
    if (!brain || !features || num_features == 0) return 0;
    if (!brain->pr_z_ladder || !brain->pr_node_manager) return 0;

    size_t n = num_features > 4096u ? 4096u : num_features;

    pr_node_config_t cfg = pr_memory_node_default_config();
    cfg.initial_tier      = PR_MEMORY_TIER_Z0;
    cfg.initial_strength  = 1.0f;
    cfg.compute_signature = true;

    pr_memory_node_t* node = pr_memory_node_create(
        (pr_node_manager_t)brain->pr_node_manager,
        (const void*)features, n * sizeof(float), &cfg);
    if (!node) return 0;

    if (z_ladder_insert(brain->pr_z_ladder, node,
                        PR_MEMORY_TIER_Z0) != Z_LADDER_SUCCESS) {
        pr_memory_node_destroy(node);
        return 0;
    }

    uint64_t id = pr_memory_node_get_id(node);
    const char* label = (reason && reason[0]) ? reason : "event";
    if (z_ladder_mark_landmark(brain->pr_z_ladder, id,
                                label) != Z_LADDER_SUCCESS) {
        /* Leave the node in the ladder — it just doesn't become a
         * landmark. Return 0 to signal the caller that landmark
         * promotion didn't take. */
        return 0;
    }

    NIMCP_LOGGING_DEBUG("pr_memory_promote_event: landmarked node %llu (reason='%s')",
                        (unsigned long long)id, label);
    return id;
}

/*============================================================================
 * Phase E4 #2 / E6 — landmark checkpoint save / load.
 *
 * Phase E6 wired a unified_mem_manager into pr_node_manager, so node
 * data_handles are now populated. Landmarks therefore have both a
 * prime_signature (fast oracle retrieval) AND their raw feature bytes
 * (full-fidelity restore).
 *
 * Save always writes v3 (signature + data). Load accepts v2 and v3 for
 * backwards compatibility — v2 records restore signature-only nodes.
 *
 * File format (little-endian, packed):
 *   header:
 *     char     magic[8]   = "NIMCPELM"
 *     uint32_t version    = 2 or 3
 *     uint32_t n_records
 *   record v2 (208 bytes):
 *     char              reason[64]
 *     prime_signature_t signature    (144 bytes)
 *   record v3 (208 bytes + data):
 *     char              reason[64]
 *     prime_signature_t signature    (144 bytes)
 *     uint32_t          data_size
 *     uint8_t           data[data_size]
 *==========================================================================*/
#define LANDMARK_CKPT_MAGIC       "NIMCPELM"
#define LANDMARK_CKPT_VERSION_V2  2u
#define LANDMARK_CKPT_VERSION_V3  3u
#define LANDMARK_CKPT_VERSION     LANDMARK_CKPT_VERSION_V3
#define LANDMARK_CKPT_MAX_DATA    (16u * 1024u * 1024u)

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t n_records;
} landmark_ckpt_header_t;

bool nimcp_brain_pr_memory_save_landmarks(struct brain_struct* brain,
                                           const char* path) {
    if (!brain || !path || !brain->pr_z_ladder) return false;

    uint64_t ids[Z_LADDER_MAX_LANDMARKS];
    char     reasons[Z_LADDER_MAX_LANDMARKS][64];
    size_t n = z_ladder_landmark_list(brain->pr_z_ladder, ids, reasons,
                                       Z_LADDER_MAX_LANDMARKS);

    /* Build a temp path: path + ".tmp" for atomic replace. */
    char tmp[1024];
    int tmp_len = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    if (tmp_len <= 0 || (size_t)tmp_len >= sizeof(tmp)) return false;

    FILE* fp = fopen(tmp, "wb");
    if (!fp) return false;

    landmark_ckpt_header_t hdr;
    memcpy(hdr.magic, LANDMARK_CKPT_MAGIC, 8);
    hdr.version   = LANDMARK_CKPT_VERSION;
    hdr.n_records = 0;  /* patched after write */

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); remove(tmp); return false; }

    uint32_t written_records = 0;
    for (size_t i = 0; i < n; i++) {
        pr_memory_node_t* node = z_ladder_find(brain->pr_z_ladder, ids[i]);
        if (!node) continue;
        const prime_signature_t* sig = pr_memory_node_get_signature(node);
        if (!sig) continue;

        /* Phase E6: attempt to read the data payload. If data_handle is
         * NULL (older pr_memory layouts or zero-size nodes), emit a v3
         * record with data_size=0 — still round-trips correctly. */
        const void* data = NULL;
        size_t data_size = pr_memory_node_get_data_size(node);
        if (data_size > 0 && data_size <= LANDMARK_CKPT_MAX_DATA) {
            data = pr_memory_node_read(node);
            if (!data) data_size = 0;  /* null handle — emit empty payload */
        } else {
            data_size = 0;
        }

        if (fwrite(reasons[i], 64, 1, fp) != 1) break;
        if (fwrite(sig, sizeof(prime_signature_t), 1, fp) != 1) break;
        uint32_t ds32 = (uint32_t)data_size;
        if (fwrite(&ds32, sizeof(ds32), 1, fp) != 1) break;
        if (data_size > 0) {
            if (fwrite(data, 1, data_size, fp) != data_size) break;
        }
        written_records++;
    }

    /* Patch header count. */
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); remove(tmp); return false; }
    hdr.n_records = written_records;
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); remove(tmp); return false; }
    if (fflush(fp) != 0 || fclose(fp) != 0) { remove(tmp); return false; }

    /* Atomic replace. */
    if (rename(tmp, path) != 0) { remove(tmp); return false; }

    NIMCP_LOGGING_INFO("pr_memory_save_landmarks: wrote %u landmarks → %s",
                       written_records, path);
    return true;
}

bool nimcp_brain_pr_memory_load_landmarks(struct brain_struct* brain,
                                           const char* path) {
    if (!brain || !path || !brain->pr_z_ladder || !brain->pr_node_manager) return false;

    FILE* fp = fopen(path, "rb");
    if (!fp) return false;

    landmark_ckpt_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); return false; }
    if (memcmp(hdr.magic, LANDMARK_CKPT_MAGIC, 8) != 0 ||
        (hdr.version != LANDMARK_CKPT_VERSION_V2 &&
         hdr.version != LANDMARK_CKPT_VERSION_V3)) {
        fclose(fp);
        NIMCP_LOGGING_WARN("pr_memory_load_landmarks: bad magic/version in %s "
                           "(version=%u)", path, hdr.version);
        return false;
    }
    const bool v3 = (hdr.version == LANDMARK_CKPT_VERSION_V3);

    uint32_t restored = 0;
    for (uint32_t i = 0; i < hdr.n_records; i++) {
        char reason[64];
        prime_signature_t sig;

        if (fread(reason, 64, 1, fp) != 1) break;
        reason[63] = '\0';
        if (fread(&sig, sizeof(sig), 1, fp) != 1) break;

        /* v3 appends a data payload; v2 does not. */
        uint32_t data_size = 0;
        uint8_t* data_buf = NULL;
        if (v3) {
            if (fread(&data_size, sizeof(data_size), 1, fp) != 1) break;
            if (data_size > LANDMARK_CKPT_MAX_DATA) break;
            if (data_size > 0) {
                data_buf = (uint8_t*)malloc(data_size);
                if (!data_buf) break;
                if (fread(data_buf, 1, data_size, fp) != data_size) {
                    free(data_buf);
                    break;
                }
            }
        }

        pr_node_config_t cfg = pr_memory_node_default_config();
        cfg.initial_tier      = PR_MEMORY_TIER_Z0;
        cfg.initial_strength  = 1.0f;
        cfg.compute_signature = false;  /* we bring our own signature */

        pr_memory_node_t* node = pr_memory_node_create_with_signature(
            (pr_node_manager_t)brain->pr_node_manager,
            data_buf, (size_t)data_size, &sig, &cfg);
        free(data_buf);
        if (!node) continue;

        if (z_ladder_insert(brain->pr_z_ladder, node,
                            PR_MEMORY_TIER_Z0) != Z_LADDER_SUCCESS) {
            pr_memory_node_destroy(node);
            continue;
        }
        uint64_t id = pr_memory_node_get_id(node);
        if (z_ladder_mark_landmark(brain->pr_z_ladder, id, reason) == Z_LADDER_SUCCESS) {
            restored++;
        }
    }

    fclose(fp);
    NIMCP_LOGGING_INFO("pr_memory_load_landmarks: restored %u/%u landmarks from %s",
                       restored, hdr.n_records, path);
    return true;
}
