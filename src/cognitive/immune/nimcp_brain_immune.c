/**
 * @file nimcp_brain_immune.c
 * @brief Brain Immune System - Adaptive Defense Coordination Layer Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Implements the brain immune coordination layer
 * WHY:  Unified threat response via BBB, BFT, swarm immune coordination
 * HOW:  Maps biological immune concepts to existing module operations
 *
 * @author NIMCP Development Team
 */

#include "cognitive/immune/nimcp_brain_immune.h"
#include "cognitive/imagination/nimcp_imagination_callbacks.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_wiring_helpers.h"
#include "nimcp.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/fault_tolerance/nimcp_hierarchical_recovery.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_immune)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_immune_mesh_id = 0;
static mesh_participant_registry_t* g_brain_immune_mesh_registry = NULL;

nimcp_error_t brain_immune_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_immune_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_immune", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SECURITY);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_immune";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_immune_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_immune_mesh_registry = registry;
    return err;
}

void brain_immune_mesh_unregister(void) {
    if (g_brain_immune_mesh_registry && g_brain_immune_mesh_id != 0) {
        mesh_participant_unregister(g_brain_immune_mesh_registry, g_brain_immune_mesh_id);
        g_brain_immune_mesh_id = 0;
        g_brain_immune_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from brain_immune module (instance-level) */
static inline void brain_immune_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_brain_immune_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_immune_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_brain_immune_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



/* Mutex convenience macros */
#define nimcp_mutex_create() nimcp_platform_mutex_create()
#define nimcp_mutex_lock(m) nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)(m))
#define nimcp_mutex_unlock(m) nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)(m))
#define nimcp_mutex_destroy(m) do { \
    nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)(m)); \
    nimcp_free(m); \
} while(0)

/* ============================================================================
 * Internal Helpers - Forward Declarations
 * ============================================================================ */

static uint64_t get_timestamp_ms(void);
static brain_antigen_t* find_antigen_by_id(brain_immune_system_t* system, uint32_t id);
static brain_b_cell_t* find_b_cell_by_id(brain_immune_system_t* system, uint32_t id);
static brain_t_cell_t* find_t_cell_by_id(brain_immune_system_t* system, uint32_t id);
static brain_antibody_t* find_antibody_by_id(brain_immune_system_t* system, uint32_t id);
static brain_inflammation_site_t* find_inflammation_by_id(brain_immune_system_t* system, uint32_t id);
static void update_immune_phase(brain_immune_system_t* system);
static void process_pending_antigens(brain_immune_system_t* system);
static void decay_antibodies(brain_immune_system_t* system, uint64_t delta_ms);
static void update_inflammation_sites(brain_immune_system_t* system, uint64_t delta_ms);
static void send_imagination_modulation_unlocked(brain_immune_system_t* system);

/* ============================================================================
 * String Conversion
 * ============================================================================ */

/**
 * @brief Convert phase to string
 */
const char* brain_immune_phase_to_string(brain_immune_phase_t phase) {
    switch (phase) {
        case IMMUNE_PHASE_SURVEILLANCE: return "SURVEILLANCE";
        case IMMUNE_PHASE_RECOGNITION:  return "RECOGNITION";
        case IMMUNE_PHASE_ACTIVATION:   return "ACTIVATION";
        case IMMUNE_PHASE_EFFECTOR:     return "EFFECTOR";
        case IMMUNE_PHASE_RESOLUTION:   return "RESOLUTION";
        case IMMUNE_PHASE_MEMORY:       return "MEMORY";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert B cell state to string
 */
const char* brain_immune_b_cell_state_to_string(brain_b_cell_state_t state) {
    switch (state) {
        case B_CELL_NAIVE:     return "NAIVE";
        case B_CELL_ACTIVATED: return "ACTIVATED";
        case B_CELL_PLASMA:    return "PLASMA";
        case B_CELL_MEMORY:    return "MEMORY";
        case B_CELL_APOPTOTIC: return "APOPTOTIC";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert T cell type to string
 */
const char* brain_immune_t_cell_type_to_string(brain_t_cell_type_t type) {
    switch (type) {
        case T_CELL_NAIVE:      return "NAIVE";
        case T_CELL_HELPER:     return "HELPER";
        case T_CELL_KILLER:     return "KILLER";
        case T_CELL_REGULATORY: return "REGULATORY";
        case T_CELL_MEMORY:     return "MEMORY";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert cytokine type to string
 */
const char* brain_immune_cytokine_to_string(brain_cytokine_type_t type) {
    switch (type) {
        case CYTOKINE_IL1B:       return "IL-1";
        case CYTOKINE_IL6:       return "IL-6";
        case CYTOKINE_IL10:      return "IL-10";
        case CYTOKINE_TNFA: return "TNF-alpha";
        case BRAIN_CYTOKINE_IFN_GAMMA: return "IFN-gamma";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert inflammation level to string
 */
const char* brain_immune_inflammation_to_string(brain_inflammation_level_t level) {
    switch (level) {
        case INFLAMMATION_NONE:     return "NONE";
        case INFLAMMATION_LOCAL:    return "LOCAL";
        case INFLAMMATION_REGIONAL: return "REGIONAL";
        case INFLAMMATION_SYSTEMIC: return "SYSTEMIC";
        case INFLAMMATION_STORM:    return "CYTOKINE_STORM";
        default: return "UNKNOWN";
    }
}

/* ============================================================================
 * Internal Helpers - Implementation
 * ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_timestamp_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    /* P1 fix: Cast to uint64_t before multiplication to prevent overflow on 32-bit time_t */
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/**
 * @brief Find antigen by ID
 */
static brain_antigen_t* find_antigen_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    for (size_t i = 0; i < system->antigen_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antigen_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antigen_count);
        }

        if (system->antigens[i].id == id) {
            return &system->antigens[i];
        }
    }
    return NULL;
}

/**
 * @brief Find B cell by ID
 */
static brain_b_cell_t* find_b_cell_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    for (size_t i = 0; i < system->b_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->b_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->b_cell_count);
        }

        if (system->b_cells[i].id == id) {
            return &system->b_cells[i];
        }
    }
    return NULL;
}

/**
 * @brief Find T cell by ID
 */
static brain_t_cell_t* find_t_cell_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    for (size_t i = 0; i < system->t_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->t_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->t_cell_count);
        }

        if (system->t_cells[i].id == id) {
            return &system->t_cells[i];
        }
    }
    return NULL;
}

/**
 * @brief Find antibody by ID
 */
static brain_antibody_t* find_antibody_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    for (size_t i = 0; i < system->antibody_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antibody_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antibody_count);
        }

        if (system->antibodies[i].id == id) {
            return &system->antibodies[i];
        }
    }
    return NULL;
}

/**
 * @brief Find inflammation site by ID
 */
static brain_inflammation_site_t* find_inflammation_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    for (size_t i = 0; i < system->inflammation_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->inflammation_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->inflammation_count);
        }

        if (system->inflammation_sites[i].id == id) {
            return &system->inflammation_sites[i];
        }
    }
    return NULL;
}

/**
 * @brief Update immune phase based on system state
 */
static void update_immune_phase(brain_immune_system_t* system) {
    if (!system) return;

    size_t active_antigens = 0;
    size_t neutralized = 0;

    for (size_t i = 0; i < system->antigen_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antigen_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antigen_count);
        }

        if (!system->antigens[i].neutralized) {
            active_antigens++;
        } else {
            neutralized++;
        }
    }

    /* Determine phase based on activity */
    if (active_antigens == 0 && system->inflammation_count == 0) {
        system->phase = IMMUNE_PHASE_SURVEILLANCE;
    } else if (active_antigens > 0 && system->antibody_count == 0) {
        system->phase = IMMUNE_PHASE_RECOGNITION;
    } else if (system->b_cell_count > 0 || system->t_cell_count > 0) {
        if (system->antibody_count > 0) {
            system->phase = IMMUNE_PHASE_EFFECTOR;
        } else {
            system->phase = IMMUNE_PHASE_ACTIVATION;
        }
    } else if (neutralized > 0 && active_antigens == 0) {
        system->phase = IMMUNE_PHASE_RESOLUTION;
    }
}

/**
 * @brief Process pending antigens and auto-activate responses
 */
static void process_pending_antigens(brain_immune_system_t* system) {
    if (!system) return;

    /* NOTE: This is called while mutex is already held by brain_immune_update()
     * So we do NOT call public API functions here - we do the work directly */

    for (size_t i = 0; i < system->antigen_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antigen_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antigen_count);
        }

        brain_antigen_t* antigen = &system->antigens[i];
        if (antigen->neutralized || antigen->processed) continue;

        /* Check if danger signal exceeds threshold */
        if (antigen->danger_signal >= system->config.activation_threshold) {
            /* Auto-activate B cell (inline, no mutex) */
            if (system->b_cell_count < system->b_cell_capacity) {
                brain_b_cell_t* b_cell = &system->b_cells[system->b_cell_count];
                memset(b_cell, 0, sizeof(*b_cell));

                b_cell->id = system->next_b_cell_id++;
                b_cell->state = B_CELL_ACTIVATED;
                b_cell->receptor_len = antigen->epitope_len;
                memcpy(b_cell->receptor, antigen->epitope, b_cell->receptor_len);
                b_cell->affinity = antigen->confidence;
                b_cell->bound_antigen_id = antigen->id;
                b_cell->activation_time = get_timestamp_ms();

                system->b_cell_count++;
                system->stats.active_b_cells++;

                /* If severe, also activate killer T */
                if (antigen->severity >= 7 && system->t_cell_count < system->t_cell_capacity) {
                    brain_t_cell_t* t_cell = &system->t_cells[system->t_cell_count];
                    memset(t_cell, 0, sizeof(*t_cell));

                    t_cell->id = system->next_t_cell_id++;
                    t_cell->type = T_CELL_KILLER;
                    t_cell->receptor_len = antigen->epitope_len;
                    memcpy(t_cell->receptor, antigen->epitope, t_cell->receptor_len);
                    t_cell->recognized_antigen_id = antigen->id;
                    t_cell->activation_level = 1.0f;
                    t_cell->activation_time = get_timestamp_ms();

                    system->t_cell_count++;
                    system->stats.active_t_cells++;
                }
            }
            antigen->processed = true;
        }
    }
}

/**
 * @brief Decay antibodies based on half-life
 *
 * Uses delta_ms (simulated time) rather than real timestamps
 * to allow predictable testing behavior.
 */
static void decay_antibodies(brain_immune_system_t* system, uint64_t delta_ms) {
    if (!system || delta_ms == 0) return;

    float half_life = (float)system->config.antibody_half_life_ms;
    if (half_life <= 0) return;

    /* Calculate decay factor based on delta time passed */
    float decay_factor = powf(0.5f, (float)delta_ms / half_life);

    for (size_t i = 0; i < system->antibody_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antibody_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antibody_count);
        }

        brain_antibody_t* ab = &system->antibodies[i];
        if (!ab->active) continue;

        ab->effectiveness *= decay_factor;

        /* Deactivate if too weak */
        if (ab->effectiveness < 0.1f) {
            ab->active = false;
            system->stats.active_antibodies--;
        }
    }
}

/**
 * @brief Update inflammation sites (progress resolution)
 *
 * WHAT: Progress inflammation resolution and notify imagination engine
 * WHY:  Inflammation affects imagination vividness/coherence (sickness behavior)
 * HOW:  Track level changes and send modulation when inflammation changes
 */
static void update_inflammation_sites(brain_immune_system_t* system, uint64_t delta_ms) {
    if (!system) return;

    bool level_changed = false;

    for (size_t i = 0; i < system->inflammation_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->inflammation_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->inflammation_count);
        }

        brain_inflammation_site_t* site = &system->inflammation_sites[i];

        /* Progress resolution if active */
        if (site->resolution_progress > 0 && site->resolution_progress < 1.0f) {
            brain_inflammation_level_t old_level = site->level;
            site->resolution_progress += 0.001f * delta_ms;
            if (site->resolution_progress >= 1.0f) {
                site->resolution_progress = 1.0f;
                site->level = INFLAMMATION_NONE;
            }
            if (site->level != old_level) {
                level_changed = true;
            }
        }
    }

    /* Notify imagination engine of inflammation changes */
    if (level_changed) {
        send_imagination_modulation_unlocked(system);
    }
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 */
int brain_immune_default_config(brain_immune_config_t* config) {
    if (!config) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_default_config", 0.0f);


    memset(config, 0, sizeof(*config));

    /* Population limits */
    config->max_antigens = BRAIN_IMMUNE_MAX_ANTIGENS;
    config->max_b_cells = BRAIN_IMMUNE_MAX_B_CELLS;
    config->max_t_cells = BRAIN_IMMUNE_MAX_T_CELLS;
    config->max_antibodies = BRAIN_IMMUNE_MAX_ANTIBODIES;

    /* Timing */
    config->activation_delay_ms = 100;
    config->memory_formation_delay_ms = 5000;
    config->antibody_half_life_ms = 30000;

    /* Thresholds */
    config->recognition_threshold = 0.6f;
    config->activation_threshold = 0.5f;
    config->inflammation_threshold = 0.7f;
    config->cytokine_storm_threshold = 0.95f;

    /* Response tuning */
    config->memory_response_multiplier = 2.0f;
    config->helper_amplification = 1.5f;

    /* Integration enables - all on by default */
    config->enable_bbb_integration = true;
    config->enable_bft_integration = true;
    config->enable_swarm_integration = true;
    config->enable_bio_async = true;
    config->enable_logging = true;

    return 0;
}

/**
 * @brief Create brain immune system
 */
brain_immune_system_t* brain_immune_create(const brain_immune_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_create", 0.0f);


    brain_immune_system_t* system = nimcp_calloc(1, sizeof(brain_immune_system_t));
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate system");

        return NULL;

    }

    /* Apply configuration */
    if (config) {
        system->config = *config;
    } else {
        brain_immune_default_config(&system->config);
    }

    /* Allocate antigen pool */
    system->antigen_capacity = system->config.max_antigens;
    system->antigens = nimcp_calloc(system->antigen_capacity, sizeof(brain_antigen_t));
    if (!system->antigens) goto cleanup;

    /* Allocate B cells */
    system->b_cell_capacity = system->config.max_b_cells;
    system->b_cells = nimcp_calloc(system->b_cell_capacity, sizeof(brain_b_cell_t));
    if (!system->b_cells) goto cleanup;

    /* Allocate T cells */
    system->t_cell_capacity = system->config.max_t_cells;
    system->t_cells = nimcp_calloc(system->t_cell_capacity, sizeof(brain_t_cell_t));
    if (!system->t_cells) goto cleanup;

    /* Allocate antibodies */
    system->antibody_capacity = system->config.max_antibodies;
    system->antibodies = nimcp_calloc(system->antibody_capacity, sizeof(brain_antibody_t));
    if (!system->antibodies) goto cleanup;

    /* Allocate cytokines */
    system->cytokine_capacity = BRAIN_IMMUNE_MAX_CYTOKINES;
    system->cytokines = nimcp_calloc(system->cytokine_capacity, sizeof(brain_cytokine_t));
    if (!system->cytokines) goto cleanup;

    /* Allocate inflammation sites */
    system->inflammation_capacity = BRAIN_IMMUNE_MAX_INFLAMMATION;
    system->inflammation_sites = nimcp_calloc(system->inflammation_capacity, sizeof(brain_inflammation_site_t));
    if (!system->inflammation_sites) goto cleanup;

    /* Initialize mutex */
    system->mutex = nimcp_mutex_create();
    if (!system->mutex) goto cleanup;

    /* Set initial state */
    system->phase = IMMUNE_PHASE_SURVEILLANCE;
    system->next_antigen_id = 1;
    system->next_b_cell_id = 1;
    system->next_t_cell_id = 1;
    system->next_antibody_id = 1;
    system->next_cytokine_id = 1;
    system->next_inflammation_id = 1;
    system->start_time = get_timestamp_ms();

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Brain immune system created");
    }

    return system;

cleanup:
    brain_immune_destroy(system);
    return NULL;
}

/**
 * @brief Destroy brain immune system
 */
void brain_immune_destroy(brain_immune_system_t* system) {
    if (!system) return;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_destroy", 0.0f);


    brain_immune_stop(system);

    if (system->mutex) {
        nimcp_mutex_free(system->mutex);
    }

    nimcp_free(system->antigens);
    nimcp_free(system->b_cells);
    nimcp_free(system->t_cells);
    nimcp_free(system->antibodies);
    nimcp_free(system->cytokines);
    nimcp_free(system->inflammation_sites);
    nimcp_free(system);
}

/**
 * @brief Start immune system
 */
int brain_immune_start(brain_immune_system_t* system) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_start", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->running = true;
    system->start_time = get_timestamp_ms();
    system->phase = IMMUNE_PHASE_SURVEILLANCE;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Brain immune system started");
    }

    return 0;
}

/**
 * @brief Stop immune system
 */
int brain_immune_stop(brain_immune_system_t* system) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_stop", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->running = false;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Brain immune system stopped");
    }

    return 0;
}

/* ============================================================================
 * Integration API
 * ============================================================================ */

/**
 * @brief Connect to BBB security
 */
int brain_immune_connect_bbb(brain_immune_system_t* system, bbb_system_t bbb_system) {
    if (!system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_immune_connect_bbb: system is NULL");
        return -1;
    }
    if (!bbb_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_immune_connect_bbb: bbb_system is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_connect_bbb", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->bbb_system = bbb_system;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to BBB security");
    }

    return 0;
}

/* ============================================================================
 * BFT Callback Handlers (static)
 * ============================================================================ */

/**
 * @brief BFT accusation callback handler
 *
 * WHAT: Callback invoked by BFT on accusations
 * WHY:  Enable automatic antigen presentation
 * HOW:  Forward to immune handler
 */
static void bft_accusation_cb(
    uint32_t accuser_id,
    uint32_t accused_id,
    bft_behavior_t behavior,
    const bft_evidence_t* evidence,
    uint32_t evidence_count,
    void* user_data
) {
    brain_immune_system_t* system = (brain_immune_system_t*)user_data;
    if (system) {
        brain_immune_handle_bft_accusation(
            system, accuser_id, accused_id, behavior, evidence, evidence_count
        );
    }
}

/**
 * @brief BFT quarantine callback handler
 */
static void bft_quarantine_cb(
    uint32_t node_id,
    uint64_t duration_ms,
    float trust_score,
    void* user_data
) {
    brain_immune_system_t* system = (brain_immune_system_t*)user_data;
    if (system) {
        brain_immune_handle_bft_quarantine(system, node_id, duration_ms, trust_score);
    }
}

/**
 * @brief BFT trust recovery callback handler
 */
static void bft_trust_recovery_cb(
    uint32_t node_id,
    float old_trust,
    float new_trust,
    void* user_data
) {
    brain_immune_system_t* system = (brain_immune_system_t*)user_data;
    if (system) {
        brain_immune_handle_bft_trust_recovery(system, node_id, old_trust, new_trust);
    }
}

/**
 * @brief Connect to BFT with enhanced callbacks
 */
int brain_immune_connect_bft(brain_immune_system_t* system, bft_context_t* bft_context) {
    if (!system || !bft_context) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_connect_bft", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->bft_context = bft_context;
    nimcp_mutex_unlock(system->mutex);

    /* Register callbacks for automatic integration, check return values */
    bool accusation_ok = bft_register_accusation_callback(bft_context, bft_accusation_cb, system);
    if (!accusation_ok) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Failed to register BFT accusation callback");
    }

    bool quarantine_ok = bft_register_quarantine_callback(bft_context, bft_quarantine_cb, system);
    if (!quarantine_ok) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Failed to register BFT quarantine callback");
    }

    bool trust_ok = bft_register_trust_recovery_callback(bft_context, bft_trust_recovery_cb, system);
    if (!trust_ok) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Failed to register BFT trust recovery callback");
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to BFT with callbacks (accusation=%d, quarantine=%d, trust=%d)",
            accusation_ok, quarantine_ok, trust_ok);
    }

    return 0;
}

/**
 * @brief Connect to swarm immune with bidirectional sync
 *
 * WHAT: Connect brain immune to swarm immune with automatic threat/response sync
 * WHY:  Enable distributed immune response across swarm nodes
 * HOW:  Link systems and enable auto-sync of threats, memory cells, and responses
 */
int brain_immune_connect_swarm(brain_immune_system_t* system, NimcpSwarmImmuneSystem* swarm_immune) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_connect_swarm", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->swarm_immune = swarm_immune;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to swarm immune with bidirectional sync");
    }

    return 0;
}

/**
 * @brief Hierarchical recovery completion callback
 *
 * WHAT: Release IL-10 on successful recovery
 * WHY:  Anti-inflammatory response to recovery completion
 * HOW:  Triggered by HR success, releases IL-10 cytokine
 */
static void hr_completion_cb(
    const hr_recovery_request_t* request,
    const hr_recovery_response_t* response,
    void* user_data
) {
    brain_immune_system_t* system = (brain_immune_system_t*)user_data;
    if (!system) return;

    (void)request;   /* Unused */
    (void)response;  /* Unused */

    /* Release anti-inflammatory IL-10 cytokine */
    uint32_t cytokine_id = 0;
    brain_immune_release_cytokine(
        system,
        CYTOKINE_IL10,
        0,       /* source: recovery system */
        0.8f,    /* high concentration */
        0,       /* broadcast */
        &cytokine_id
    );

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Recovery completed -> IL-10 anti-inflammatory release (cytokine %u)", cytokine_id);
    }
}

/**
 * @brief Connect to hierarchical recovery
 */
int brain_immune_connect_hierarchical_recovery(brain_immune_system_t* system, void* hr_context) {
    if (!system || !hr_context) return -1;

    /* Register completion callback for IL-10 release */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_connect_hierarchical", 0.0f);


    hr_register_completion_callback((hr_context_t*)hr_context, hr_completion_cb, system);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to hierarchical recovery with IL-10 callback");
    }

    return 0;
}

/**
 * @brief Connect to bio-async router
 */
int brain_immune_connect_bio_async(brain_immune_system_t* system) {
    if (!system) return -1;
    if (!bio_router_is_initialized()) {
        if (system->config.enable_logging) {
            LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Bio-async router not available, skipping registration");
        }
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_connect_bio_async", 0.0f);


    bio_module_info_t info = {
        .module_id = BIO_MODULE_INTROSPECTION + 0x50,  /* Offset for immune */
        .module_name = BRAIN_IMMUNE_MODULE_NAME,
        .inbox_capacity = 64,
        .user_data = system
    };

    nimcp_mutex_lock(system->mutex);
    system->bio_context = bio_router_register_module(&info);
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to bio-async router");
    }

    return 0;
}

/* ============================================================================
 * Enhanced Swarm Integration API
 * ============================================================================ */

/**
 * @brief Automatically sync swarm threat to brain immune antigen
 *
 * WHAT: Auto-present swarm-detected threats as brain immune antigens
 * WHY:  Ensure all swarm threats are processed by brain immune system
 * HOW:  Called automatically when swarm detects threat
 */
int brain_immune_auto_sync_swarm_threat(
    brain_immune_system_t* system,
    const NimcpSwarmThreat* threat
) {
    if (!system || !threat || !system->swarm_immune) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_auto_sync_swarm_thre", 0.0f);


    uint32_t antigen_id = 0;
    int result = brain_immune_present_swarm_threat(system, threat, &antigen_id);

    if (result == 0 && system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Auto-synced swarm threat %u -> brain antigen %u",
            threat->id, antigen_id);
    }

    return result;
}

/**
 * @brief Sync brain immune memory cell to swarm immune memory
 *
 * WHAT: Create swarm immune memory cell from brain immune B cell memory
 * WHY:  Share learned threat patterns across swarm
 * HOW:  Convert B cell receptor pattern to swarm threat signature
 */
int brain_immune_sync_memory_to_swarm(
    brain_immune_system_t* system,
    uint32_t b_cell_id
) {
    if (!system || !system->swarm_immune) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_sync_memory_to_swarm", 0.0f);


    brain_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell || b_cell->state != B_CELL_MEMORY) return -1;

    /* Find corresponding antigen to get response type */
    brain_antigen_t* antigen = find_antigen_by_id(system, b_cell->bound_antigen_id);
    if (!antigen) return -1;

    /* Create swarm threat signature from B cell receptor */
    NimcpSwarmThreatSignature signature;
    memset(&signature, 0, sizeof(signature));

    signature.pattern_len = b_cell->receptor_len;
    memcpy(signature.pattern, b_cell->receptor,
           signature.pattern_len > 64 ? 64 : signature.pattern_len);
    signature.match_threshold = system->config.recognition_threshold;
    signature.type = THREAT_BYZANTINE;  /* Default threat type */
    signature.detection_count = 1;
    signature.last_seen = get_timestamp_ms();

    /* Determine response type based on antigen severity */
    NimcpSwarmResponseType response = RESPONSE_ISOLATION;
    if (antigen->severity >= 8) {
        response = RESPONSE_COUNTER_ATTACK;
    } else if (antigen->severity >= 5) {
        response = RESPONSE_ISOLATION;
    } else {
        response = RESPONSE_ALERT;
    }

    /* Add memory cell to swarm immune */
    uint32_t swarm_cell_id = 0;
    nimcp_result_t res = nimcp_swarm_immune_add_memory_cell(
        system->swarm_immune,
        &signature,
        response,
        b_cell->affinity,
        &swarm_cell_id
    );

    if (res == NIMCP_SUCCESS) {
        nimcp_mutex_lock(system->mutex);
        b_cell->swarm_memory_cell_id = swarm_cell_id;
        nimcp_mutex_unlock(system->mutex);

        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Synced brain B cell %u -> swarm memory cell %u",
                b_cell_id, swarm_cell_id);
        }
        return 0;
    }

    return -1;
}

/**
 * @brief Trigger swarm response from brain antibody
 *
 * WHAT: Execute swarm immune response when brain antibody is activated
 * WHY:  Translate brain immune action to swarm-level coordinated response
 * HOW:  Map antibody class to swarm response type and execute
 */
int brain_immune_trigger_swarm_response(
    brain_immune_system_t* system,
    uint32_t antibody_id
) {
    if (!system || !system->swarm_immune) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_trigger_swarm_respon", 0.0f);


    brain_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody || !antibody->active) return -1;

    brain_antigen_t* antigen = find_antigen_by_id(system, antibody->target_antigen_id);
    if (!antigen) return -1;

    /* If antibody already has swarm response, execute it */
    if (antibody->swarm_response_id != 0) {
        return nimcp_swarm_immune_execute_response(
            system->swarm_immune,
            antibody->swarm_response_id
        ) == NIMCP_SUCCESS ? 0 : -1;
    }

    /* Generate new swarm response based on antibody class */
    uint32_t threat_id = antigen->source_node_id;  /* Use source node as threat ID */
    uint32_t response_id = 0;

    nimcp_result_t res = nimcp_swarm_immune_generate_response(
        system->swarm_immune,
        threat_id,
        &response_id
    );

    if (res == NIMCP_SUCCESS) {
        nimcp_mutex_lock(system->mutex);
        antibody->swarm_response_id = response_id;
        nimcp_mutex_unlock(system->mutex);

        /* Execute the response */
        nimcp_swarm_immune_execute_response(system->swarm_immune, response_id);

        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Triggered swarm response %u from brain antibody %u",
                response_id, antibody_id);
        }
        return 0;
    }

    return -1;
}

/**
 * @brief Broadcast collective inflammation state to swarm
 *
 * WHAT: Share inflammation level across swarm nodes via consensus
 * WHY:  Enable swarm-wide coordinated inflammatory response
 * HOW:  Send cytokine message with inflammation severity, use consensus to agree
 */
int brain_immune_broadcast_inflammation_state(
    brain_immune_system_t* system,
    uint32_t site_id
) {
    if (!system || !system->swarm_immune) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_broadcast_inflammati", 0.0f);


    brain_inflammation_site_t* site = find_inflammation_by_id(system, site_id);
    if (!site) return -1;

    /* Map inflammation level to swarm severity */
    NimcpSwarmSeverity severity;
    switch (site->level) {
        case INFLAMMATION_LOCAL:    severity = SWARM_SEVERITY_LOW; break;
        case INFLAMMATION_REGIONAL: severity = SWARM_SEVERITY_MEDIUM; break;
        case INFLAMMATION_SYSTEMIC: severity = SWARM_SEVERITY_HIGH; break;
        case INFLAMMATION_STORM:    severity = SWARM_SEVERITY_CRITICAL; break;
        default:                    severity = SWARM_SEVERITY_LOW; break;
    }

    /* Broadcast via swarm immune alert */
    nimcp_result_t res = nimcp_swarm_immune_broadcast_alert(
        system->swarm_immune,
        site->triggering_antigen_id,
        severity
    );

    /* Also release pro-inflammatory cytokine */
    if (res == NIMCP_SUCCESS && system->config.enable_bio_async) {
        uint32_t cytokine_id;
        brain_cytokine_type_t type = (site->level >= INFLAMMATION_SYSTEMIC)
            ? CYTOKINE_TNFA : CYTOKINE_IL6;

        brain_immune_release_cytokine(
            system, type, 0,
            site->resource_allocation,
            0,  /* Broadcast to all */
            &cytokine_id
        );
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Broadcast inflammation state: site %u, level %s",
            site_id, brain_immune_inflammation_to_string(site->level));
    }

    return res == NIMCP_SUCCESS ? 0 : -1;
}

/**
 * @brief Request consensus on threat severity via swarm
 *
 * WHAT: Use swarm consensus to assess threat severity collectively
 * WHY:  Prevent false positives, ensure distributed agreement on threats
 * HOW:  Each node votes on severity, weighted by confidence, use cytokine messaging
 */
int brain_immune_consensus_threat_severity(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    float* agreed_severity_out
) {
    if (!system || !system->swarm_immune || !agreed_severity_out) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_consensus_threat_sev", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    /* Confirm threat via swarm consensus */
    uint32_t threat_id = antigen->id;
    uint32_t confirming_drone = system->swarm_immune->self_drone_id;

    nimcp_result_t res = nimcp_swarm_immune_confirm_threat(
        system->swarm_immune,
        threat_id,
        confirming_drone
    );

    if (res == NIMCP_SUCCESS) {
        /* Get confirmed threat info */
        const NimcpSwarmThreat* threat = NULL;
        res = nimcp_swarm_immune_get_threat(
            system->swarm_immune,
            threat_id,
            &threat
        );

        if (res == NIMCP_SUCCESS && threat) {
            /* Update antigen with consensus information */
            nimcp_mutex_lock(system->mutex);
            /* Note: brain_antigen_t uses 'processed' rather than 'confirmed' */
            antigen->processed = threat->confirmed;
            antigen->confidence = threat->confidence;

            /* Map swarm confirming drones to severity adjustment */
            float severity_factor = (float)threat->confirming_drones / 10.0f;
            if (severity_factor > 1.0f) severity_factor = 1.0f;

            *agreed_severity_out = antigen->severity * severity_factor;
            nimcp_mutex_unlock(system->mutex);

            if (system->config.enable_logging) {
                LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                    "Consensus on antigen %u: confirmed=%d, drones=%u, severity=%.2f",
                    antigen_id, threat->confirmed, threat->confirming_drones,
                    *agreed_severity_out);
            }
            return 0;
        }
    }

    return -1;
}

/**
 * @brief Propagate secondary response across swarm when memory cell recognizes threat
 *
 * WHAT: When any node recognizes learned threat, trigger swarm-wide secondary response
 * WHY:  Collective memory - if one node remembers, entire swarm benefits
 * HOW:  Share memory cell activation, broadcast rapid response to all nodes
 */
int brain_immune_propagate_secondary_response(
    brain_immune_system_t* system,
    uint32_t memory_b_cell_id,
    uint32_t antigen_id
) {
    if (!system || !system->swarm_immune) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_propagate_secondary_", 0.0f);


    brain_b_cell_t* b_cell = find_b_cell_by_id(system, memory_b_cell_id);
    if (!b_cell || b_cell->state != B_CELL_MEMORY) return -1;

    /* Share memory cell with swarm if not already shared */
    if (b_cell->swarm_memory_cell_id == 0) {
        brain_immune_sync_memory_to_swarm(system, memory_b_cell_id);
    }

    /* Share memory cell with entire swarm */
    if (b_cell->swarm_memory_cell_id != 0) {
        nimcp_result_t res = nimcp_swarm_immune_share_memory_cell(
            system->swarm_immune,
            b_cell->swarm_memory_cell_id
        );

        if (res == NIMCP_SUCCESS && system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Propagated secondary response: memory cell %u shared across swarm",
                b_cell->swarm_memory_cell_id);
        }
    }

    /* Broadcast rapid response alert with high priority */
    brain_immune_broadcast_alert(system, antigen_id, INFLAMMATION_REGIONAL);

    /* Release coordinating cytokines */
    uint32_t cytokine_id;
    brain_immune_release_cytokine(
        system, BRAIN_CYTOKINE_IFN_GAMMA,
        memory_b_cell_id,
        system->config.memory_response_multiplier,
        0,  /* Broadcast */
        &cytokine_id
    );

    return 0;
}

/* ============================================================================
 * Antigen Presentation API
 * ============================================================================ */

/**
 * @brief Present generic antigen
 */
int brain_immune_present_antigen(
    brain_immune_system_t* system,
    brain_antigen_source_t source,
    const uint8_t* epitope,
    size_t epitope_len,
    uint32_t severity,
    uint32_t source_node,
    uint32_t* antigen_id
) {
    if (!system || !epitope || epitope_len == 0) return -1;
    if (system->antigen_count >= system->antigen_capacity) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_present_antigen", 0.0f);


    nimcp_mutex_lock(system->mutex);

    brain_antigen_t* antigen = &system->antigens[system->antigen_count];
    memset(antigen, 0, sizeof(*antigen));

    antigen->id = system->next_antigen_id++;
    antigen->source = source;
    antigen->epitope_len = (epitope_len > BRAIN_IMMUNE_EPITOPE_SIZE)
        ? BRAIN_IMMUNE_EPITOPE_SIZE : epitope_len;
    memcpy(antigen->epitope, epitope, antigen->epitope_len);

    antigen->source_node_id = source_node;
    antigen->severity = severity;
    antigen->confidence = 1.0f;
    antigen->danger_signal = severity / 10.0f;
    antigen->detection_time = get_timestamp_ms();
    antigen->processed = false;
    antigen->neutralized = false;

    uint32_t new_antigen_id = antigen->id;
    if (antigen_id) *antigen_id = new_antigen_id;
    system->antigen_count++;
    system->stats.antigens_processed++;

    /* Transition to recognition phase */
    if (system->phase == IMMUNE_PHASE_SURVEILLANCE) {
        system->phase = IMMUNE_PHASE_RECOGNITION;
    }

    /* Copy antigen data before unlocking for safe callback invocation */
    brain_antigen_t antigen_copy = *antigen;
    void* callback_user_data = system->callback_user_data;
    brain_immune_antigen_cb_t antigen_callback = system->on_antigen;

    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback with copied data (safe after unlock) */
    if (antigen_callback) {
        antigen_callback(system, &antigen_copy, callback_user_data);
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Antigen presented: id=%u, source=%d, severity=%u",
            new_antigen_id, source, severity);
    }

    /* AUTO-RECOGNITION: Check if we've seen this threat before */
    uint32_t memory_b_cell_id = 0;
    if (brain_immune_check_memory(system, new_antigen_id, &memory_b_cell_id) == 0) {
        /* Memory match found - trigger automatic secondary response */
        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "LEARNED THREAT RECOGNIZED: Antigen %u matches memory B cell %u - triggering secondary response",
                new_antigen_id, memory_b_cell_id);
        }
        brain_immune_secondary_response(system, new_antigen_id, memory_b_cell_id);
    }

    return 0;
}

/**
 * @brief Present BBB threat as antigen
 */
int brain_immune_present_bbb_threat(
    brain_immune_system_t* system,
    bbb_threat_type_t threat_type,
    bbb_severity_t severity,
    const uint8_t* threat_data,
    size_t data_len,
    uint32_t* antigen_id
) {
    if (!system) return -1;

    /* Map BBB severity to 1-10 scale */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_present_bbb_threat", 0.0f);


    uint32_t immune_severity;
    switch (severity) {
        case SWARM_SEVERITY_LOW:      immune_severity = 3; break;
        case SWARM_SEVERITY_MEDIUM:   immune_severity = 5; break;
        case SWARM_SEVERITY_HIGH:     immune_severity = 7; break;
        case SWARM_SEVERITY_CRITICAL: immune_severity = 10; break;
        default: immune_severity = 1; break;
    }

    int result = brain_immune_present_antigen(
        system, ANTIGEN_SOURCE_BBB,
        threat_data, data_len,
        immune_severity, 0, antigen_id
    );

    if (result == 0 && antigen_id) {
        /* Set BBB-specific fields */
        brain_antigen_t* ag = find_antigen_by_id(system, *antigen_id);
        if (ag) {
            ag->bbb_threat_type = threat_type;
        }
        system->stats.bbb_threats_processed++;
    }

    return result;
}

/**
 * @brief Present Byzantine node as antigen
 */
int brain_immune_present_byzantine(
    brain_immune_system_t* system,
    uint32_t node_id,
    bft_behavior_t behavior,
    const bft_evidence_t* evidence,
    size_t evidence_len,
    uint32_t* antigen_id
) {
    if (!system) return -1;

    /* Create epitope from node ID and behavior */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_present_byzantine", 0.0f);


    uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
    memset(epitope, 0, sizeof(epitope));
    memcpy(epitope, &node_id, sizeof(node_id));
    memcpy(epitope + sizeof(node_id), &behavior, sizeof(behavior));

    /* Byzantine nodes are always severe */
    uint32_t severity = 8;
    if (behavior == BFT_BEHAV_COLLUSION) severity = 10;

    int result = brain_immune_present_antigen(
        system, ANTIGEN_SOURCE_BFT,
        epitope, sizeof(node_id) + sizeof(behavior),
        severity, node_id, antigen_id
    );

    if (result == 0 && antigen_id) {
        brain_antigen_t* ag = find_antigen_by_id(system, *antigen_id);
        if (ag) {
            ag->bft_behavior = behavior;
        }
        system->stats.bft_byzantines_handled++;
    }

    return result;
}

/**
 * @brief Present swarm threat as antigen
 */
int brain_immune_present_swarm_threat(
    brain_immune_system_t* system,
    const NimcpSwarmThreat* threat,
    uint32_t* antigen_id
) {
    if (!system || !threat) return -1;

    /* Map swarm severity */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_present_swarm_threat", 0.0f);


    uint32_t severity;
    switch (threat->severity) {
        case SWARM_SEVERITY_LOW:      severity = 3; break;
        case SWARM_SEVERITY_MEDIUM:   severity = 5; break;
        case SWARM_SEVERITY_HIGH:     severity = 7; break;
        case SWARM_SEVERITY_CRITICAL: severity = 10; break;
        default: severity = 5; break;
    }

    int result = brain_immune_present_antigen(
        system, ANTIGEN_SOURCE_SWARM,
        threat->data, threat->data_len,
        severity, threat->source_drone_id, antigen_id
    );

    if (result == 0) {
        system->stats.swarm_alerts_processed++;
    }

    return result;
}

/* ============================================================================
 * B Cell API
 * ============================================================================ */

/**
 * @brief Activate B cell for antigen
 */
int brain_immune_activate_b_cell(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t* b_cell_id
) {
    if (!system) return -1;
    if (system->b_cell_count >= system->b_cell_capacity) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_activate_b_cell", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    nimcp_mutex_lock(system->mutex);

    /* Create new B cell */
    brain_b_cell_t* b_cell = &system->b_cells[system->b_cell_count];
    memset(b_cell, 0, sizeof(*b_cell));

    b_cell->id = system->next_b_cell_id++;
    b_cell->state = B_CELL_ACTIVATED;
    b_cell->receptor_len = antigen->epitope_len;
    memcpy(b_cell->receptor, antigen->epitope, b_cell->receptor_len);
    b_cell->affinity = antigen->confidence;
    b_cell->bound_antigen_id = antigen_id;
    b_cell->activation_time = get_timestamp_ms();

    if (b_cell_id) *b_cell_id = b_cell->id;
    system->b_cell_count++;
    system->stats.active_b_cells++;

    /* Update phase */
    if (system->phase == IMMUNE_PHASE_RECOGNITION) {
        system->phase = IMMUNE_PHASE_ACTIVATION;
    }

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
            "B cell activated: id=%u for antigen=%u", b_cell->id, antigen_id);
    }

    return 0;
}

/**
 * @brief Convert B cell to memory
 */
int brain_immune_b_cell_to_memory(brain_immune_system_t* system, uint32_t b_cell_id) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_b_cell_to_memory", 0.0f);


    brain_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell) return -1;

    nimcp_mutex_lock(system->mutex);

    b_cell->state = B_CELL_MEMORY;
    system->stats.memory_cells++;

    /* If connected to swarm immune, create swarm memory cell */
    if (system->swarm_immune) {
        NimcpSwarmThreatSignature sig;
        memset(&sig, 0, sizeof(sig));
        memcpy(sig.pattern, b_cell->receptor, b_cell->receptor_len);
        sig.pattern_len = b_cell->receptor_len;
        sig.match_threshold = system->config.recognition_threshold;

        NimcpSwarmMemoryCell mem_cell;
        memset(&mem_cell, 0, sizeof(mem_cell));
        mem_cell.signature = sig;
        mem_cell.effectiveness = b_cell->affinity;
        mem_cell.created_time = get_timestamp_ms();

        /* Note: Would call nimcp_swarm_immune_add_memory_cell if available */
        b_cell->swarm_memory_cell_id = mem_cell.id;
    }

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "B cell %u converted to memory", b_cell_id);
    }

    return 0;
}

/* ============================================================================
 * T Cell API
 * ============================================================================ */

/**
 * @brief Activate helper T cell
 */
int brain_immune_activate_helper_t(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t* t_cell_id
) {
    if (!system) return -1;
    if (system->t_cell_count >= system->t_cell_capacity) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_activate_helper_t", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    nimcp_mutex_lock(system->mutex);

    brain_t_cell_t* t_cell = &system->t_cells[system->t_cell_count];
    memset(t_cell, 0, sizeof(*t_cell));

    t_cell->id = system->next_t_cell_id++;
    t_cell->type = T_CELL_HELPER;
    t_cell->receptor_len = antigen->epitope_len;
    memcpy(t_cell->receptor, antigen->epitope, t_cell->receptor_len);
    t_cell->recognized_antigen_id = antigen_id;
    t_cell->activation_level = antigen->danger_signal;
    t_cell->activation_time = get_timestamp_ms();

    if (t_cell_id) *t_cell_id = t_cell->id;
    system->t_cell_count++;
    system->stats.active_t_cells++;

    nimcp_mutex_unlock(system->mutex);

    /* Helper T cells release cytokines on activation */
    uint32_t cytokine_id;
    brain_immune_release_cytokine(system, CYTOKINE_IL6, t_cell->id,
                                   t_cell->activation_level, 0, &cytokine_id);

    if (system->config.enable_logging) {
        LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
            "Helper T cell activated: id=%u", t_cell->id);
    }

    return 0;
}

/**
 * @brief Activate killer T cell
 */
int brain_immune_activate_killer_t(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t* t_cell_id
) {
    if (!system) return -1;
    if (system->t_cell_count >= system->t_cell_capacity) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_activate_killer_t", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    nimcp_mutex_lock(system->mutex);

    brain_t_cell_t* t_cell = &system->t_cells[system->t_cell_count];
    memset(t_cell, 0, sizeof(*t_cell));

    t_cell->id = system->next_t_cell_id++;
    t_cell->type = T_CELL_KILLER;
    t_cell->receptor_len = antigen->epitope_len;
    memcpy(t_cell->receptor, antigen->epitope, t_cell->receptor_len);
    t_cell->recognized_antigen_id = antigen_id;
    t_cell->activation_level = antigen->danger_signal;
    t_cell->activation_time = get_timestamp_ms();

    if (t_cell_id) *t_cell_id = t_cell->id;
    system->t_cell_count++;
    system->stats.active_t_cells++;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
            "Killer T cell activated: id=%u", t_cell->id);
    }

    return 0;
}

/**
 * @brief Execute killer T cell action
 */
int brain_immune_t_cell_kill(
    brain_immune_system_t* system,
    uint32_t t_cell_id,
    uint32_t target_node
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_t_cell_kill", 0.0f);


    brain_t_cell_t* t_cell = find_t_cell_by_id(system, t_cell_id);
    if (!t_cell || t_cell->type != T_CELL_KILLER) return -1;

    /* Execute quarantine via BFT if connected */
    if (system->bft_context) {
        bft_quarantine_node(system->bft_context, target_node,
                           system->config.antibody_half_life_ms);
    }

    /* Capture callback and data under mutex to prevent race condition */
    brain_immune_kill_cb_t kill_callback = NULL;
    void* callback_user_data = NULL;
    brain_t_cell_t t_cell_copy;

    nimcp_mutex_lock(system->mutex);
    t_cell->kills++;
    kill_callback = system->on_kill;
    callback_user_data = system->callback_user_data;
    t_cell_copy = *t_cell;  /* Copy for safe callback invocation */
    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback with copied data (safe after unlock) */
    if (kill_callback) {
        kill_callback(system, &t_cell_copy, target_node, callback_user_data);
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Killer T cell %u eliminated node %u", t_cell_id, target_node);
    }

    return 0;
}

/**
 * @brief Helper T provides help to B cell
 */
int brain_immune_t_help_b(
    brain_immune_system_t* system,
    uint32_t helper_id,
    uint32_t b_cell_id
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_t_help_b", 0.0f);


    brain_t_cell_t* helper = find_t_cell_by_id(system, helper_id);
    if (!helper || helper->type != T_CELL_HELPER) return -1;

    brain_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell) return -1;

    nimcp_mutex_lock(system->mutex);

    b_cell->received_t_help = true;
    b_cell->affinity *= system->config.helper_amplification;
    if (b_cell->affinity > 1.0f) b_cell->affinity = 1.0f;

    helper->help_given++;

    /* Transition B cell to plasma state */
    if (b_cell->state == B_CELL_ACTIVATED) {
        b_cell->state = B_CELL_PLASMA;
    }

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
            "Helper T %u helped B cell %u", helper_id, b_cell_id);
    }

    return 0;
}

/* ============================================================================
 * Antibody API
 * ============================================================================ */

/**
 * @brief Produce antibody from B cell
 */
int brain_immune_produce_antibody(
    brain_immune_system_t* system,
    uint32_t b_cell_id,
    brain_antibody_class_t ab_class,
    uint32_t* antibody_id
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_produce_antibody", 0.0f);


    nimcp_mutex_lock(system->mutex);

    /* Check capacity inside lock to avoid race */
    if (system->antibody_count >= system->antibody_capacity) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Find and validate B cell inside critical section */
    brain_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell || b_cell->state != B_CELL_PLASMA) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    brain_antibody_t* antibody = &system->antibodies[system->antibody_count];
    memset(antibody, 0, sizeof(*antibody));

    antibody->id = system->next_antibody_id++;
    antibody->ab_class = ab_class;
    antibody->target_antigen_id = b_cell->bound_antigen_id;
    antibody->producer_b_cell_id = b_cell_id;
    antibody->effectiveness = b_cell->affinity;
    antibody->creation_time = get_timestamp_ms();
    antibody->active = true;

    /* Map to swarm response based on class */
    switch (ab_class) {
        case ANTIBODY_IGM:
            antibody->swarm_response = RESPONSE_ALERT;
            break;
        case ANTIBODY_IGG:
            antibody->swarm_response = RESPONSE_ISOLATION;
            break;
        case ANTIBODY_IGE:
            antibody->swarm_response = RESPONSE_COUNTER_ATTACK;
            break;
    }

    if (antibody_id) *antibody_id = antibody->id;
    system->antibody_count++;
    system->stats.active_antibodies++;
    system->stats.responses_generated++;
    b_cell->antibodies_produced++;

    /* Update phase to effector */
    if (system->phase == IMMUNE_PHASE_ACTIVATION) {
        system->phase = IMMUNE_PHASE_EFFECTOR;
    }

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
            "Antibody produced: id=%u, class=%d", antibody->id, ab_class);
    }

    return 0;
}

/**
 * @brief Execute antibody response
 */
int brain_immune_execute_antibody(brain_immune_system_t* system, uint32_t antibody_id) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_execute_antibody", 0.0f);


    nimcp_mutex_lock(system->mutex);

    /* Find and validate antibody inside critical section */
    brain_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody || !antibody->active) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Get associated antigen for BBB coordination */
    brain_antigen_t* antigen = find_antigen_by_id(system, antibody->target_antigen_id);

    /* Execute swarm response if connected */
    if (system->swarm_immune) {
        /* Would call nimcp_swarm_immune_generate_response */
        antibody->swarm_response_id = 1;  /* Placeholder */
    }

    /* Copy data needed after unlock */
    bbb_action_t bbb_action = antibody->bbb_action;
    brain_antibody_class_t ab_class = antibody->ab_class;
    NimcpSwarmResponseType swarm_response = antibody->swarm_response;
    uint32_t source_node_id = antigen ? antigen->source_node_id : 0;
    bool antigen_from_bbb = antigen && antigen->source == ANTIGEN_SOURCE_BBB;
    void* bbb_system = system->bbb_system;
    bool enable_logging = system->config.enable_logging;

    nimcp_mutex_unlock(system->mutex);

    /* Coordinate BBB action if antigen came from BBB (outside lock - BBB has own locking) */
    if (antigen_from_bbb && bbb_system) {
        /* BBB action is already stored in antibody->bbb_action */
        /* Execute coordinated BBB action based on antibody class */
        if (ab_class == ANTIBODY_IGG || ab_class == ANTIBODY_IGE) {
            /* For mature/emergency antibodies, ensure BBB takes strong action */
            if (bbb_action == BBB_ACTION_LOG || bbb_action == BBB_ACTION_BLOCK) {
                /* Escalate to quarantine for high-affinity antibody responses */
                if (source_node_id != 0) {
                    void* threat_addr = (void*)(uintptr_t)source_node_id;
                    bbb_quarantine_region(bbb_system, threat_addr, 1);
                }
            }
        }
    }

    if (enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Antibody %u executed response type %d with BBB action %d",
            antibody_id, swarm_response, bbb_action);
    }

    return 0;
}

/**
 * @brief Mark antigen as neutralized
 */
int brain_immune_neutralize(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t antibody_id
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_neutralize", 0.0f);


    nimcp_mutex_lock(system->mutex);

    /* Find and validate inside critical section */
    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    brain_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    antigen->neutralized = true;
    antigen->response_count++;
    antibody->neutralizations++;
    system->stats.threats_neutralized++;

    /* Copy IDs before unlock for safe access after critical section */
    uint32_t producer_b_cell_id = antibody->producer_b_cell_id;
    brain_antibody_t antibody_copy = *antibody;

    /* AUTO-LEARNING: Convert producing B cell to memory for future recognition */
    /* Done inside critical section to safely access B cell state */
    brain_b_cell_t* producer = find_b_cell_by_id(system, producer_b_cell_id);
    bool should_convert_to_memory = (producer && producer->state != B_CELL_MEMORY);

    /* Copy callback info before unlock */
    void* callback_user_data = system->callback_user_data;
    brain_immune_neutralize_cb_t neutralize_callback = system->on_neutralize;
    bool enable_logging = system->config.enable_logging;

    nimcp_mutex_unlock(system->mutex);

    /* Perform memory conversion outside lock (function handles its own locking) */
    if (should_convert_to_memory) {
        brain_immune_b_cell_to_memory(system, producer_b_cell_id);

        if (enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Auto-learning: B cell %u converted to memory after neutralizing antigen %u",
                producer_b_cell_id, antigen_id);
        }
    }

    /* Trigger callback with copied data (safe after unlock) */
    if (neutralize_callback) {
        neutralize_callback(system, antigen_id, &antibody_copy, callback_user_data);
    }

    if (enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Antigen %u neutralized by antibody %u", antigen_id, antibody_id);
    }

    return 0;
}

/* ============================================================================
 * Cytokine Signaling API
 * ============================================================================ */

/**
 * @brief Release cytokine signal
 */
int brain_immune_release_cytokine(
    brain_immune_system_t* system,
    brain_cytokine_type_t type,
    uint32_t source_cell,
    float concentration,
    uint32_t target_region,
    uint32_t* cytokine_id
) {
    if (!system) return -1;
    if (!system->mutex) return -1;
    if (!system->cytokines) return -1;
    if (system->cytokine_count >= system->cytokine_capacity) return -1;
    /* Validate concentration is in expected range [0.0, 1.0] */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_release_cytokine", 0.0f);


    if (concentration < 0.0f || !isfinite(concentration)) concentration = 0.0f;
    if (concentration > 1.0f) concentration = 1.0f;

    nimcp_mutex_lock(system->mutex);

    brain_cytokine_t* cytokine = &system->cytokines[system->cytokine_count];
    memset(cytokine, 0, sizeof(*cytokine));

    cytokine->id = system->next_cytokine_id++;
    cytokine->type = type;
    cytokine->source_cell_id = source_cell;
    cytokine->target_region = target_region;
    cytokine->concentration = concentration;
    cytokine->release_time = get_timestamp_ms();

    /* Determine if pro-inflammatory */
    cytokine->pro_inflammatory = (type != CYTOKINE_IL10);

    /* Map to bio-async message type */
    switch (type) {
        case CYTOKINE_IL1B:
        case CYTOKINE_IL6:
        case CYTOKINE_TNFA:
            cytokine->message_type = BIO_MSG_SECURITY_ALERT;
            break;
        case BRAIN_CYTOKINE_IFN_GAMMA:
            cytokine->message_type = BIO_MSG_SWARM_IMMUNE_ALERT;
            break;
        case CYTOKINE_IL10:
            cytokine->message_type = BIO_MSG_HEALTH_RESPONSE;
            break;
        default:
            cytokine->message_type = BIO_MSG_SECURITY_EVENT;
    }

    if (cytokine_id) *cytokine_id = cytokine->id;
    system->cytokine_count++;
    system->stats.cytokines_released++;

    /* Check for cytokine storm */
    float total_pro_inflammatory = 0;
    for (size_t i = 0; i < system->cytokine_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->cytokine_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->cytokine_count);
        }

        if (system->cytokines[i].pro_inflammatory) {
            total_pro_inflammatory += system->cytokines[i].concentration;
        }
    }
    if (total_pro_inflammatory >= system->config.cytokine_storm_threshold) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Cytokine storm risk detected!");
    }

    /* Copy callback data before unlock to prevent race condition */
    brain_immune_cytokine_cb_t callback = system->on_cytokine;
    void* callback_data = system->callback_user_data;
    bio_module_context_t bio_ctx = system->bio_context;

    /* Set delivered flag BEFORE unlock to prevent race condition */
    /* (checking target_region == 0 means broadcast) */
    if (bio_ctx && target_region == 0) {
        cytokine->delivered = true;
    }

    brain_cytokine_t cytokine_copy = *cytokine;

    nimcp_mutex_unlock(system->mutex);

    /* Bio-async message sending would happen here (outside lock) */
    /* The delivered flag was already set inside the critical section */

    /* Trigger callback with copied data */
    if (callback) {
        callback(system, &cytokine_copy, callback_data);
    }

    return 0;
}

/**
 * @brief Broadcast immune alert
 */
int brain_immune_broadcast_alert(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    brain_inflammation_level_t severity
) {
    if (!system) return -1;

    /* Release appropriate cytokine for severity */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_broadcast_alert", 0.0f);


    brain_cytokine_type_t type;
    float concentration;

    switch (severity) {
        case INFLAMMATION_LOCAL:
            type = CYTOKINE_IL1B;
            concentration = 0.3f;
            break;
        case INFLAMMATION_REGIONAL:
            type = CYTOKINE_IL6;
            concentration = 0.5f;
            break;
        case INFLAMMATION_SYSTEMIC:
            type = CYTOKINE_TNFA;
            concentration = 0.7f;
            break;
        case INFLAMMATION_STORM:
            type = CYTOKINE_TNFA;
            concentration = 0.9f;
            break;
        default:
            return 0;
    }

    uint32_t cytokine_id;
    return brain_immune_release_cytokine(system, type, 0, concentration, 0, &cytokine_id);
}

/* ============================================================================
 * Inflammation API
 * ============================================================================ */

/**
 * @brief Initiate inflammation
 */
int brain_immune_initiate_inflammation(
    brain_immune_system_t* system,
    uint32_t region_id,
    uint32_t antigen_id,
    uint32_t* site_id
) {
    if (!system) return -1;
    if (system->inflammation_count >= system->inflammation_capacity) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_initiate_inflammatio", 0.0f);


    nimcp_mutex_lock(system->mutex);

    brain_inflammation_site_t* site = &system->inflammation_sites[system->inflammation_count];
    memset(site, 0, sizeof(*site));

    site->id = system->next_inflammation_id++;
    site->region_id = region_id;
    site->triggering_antigen_id = antigen_id;
    site->level = INFLAMMATION_LOCAL;
    site->start_time = get_timestamp_ms();
    site->resource_allocation = 0.2f;
    site->resolution_progress = 0.0f;

    if (site_id) *site_id = site->id;
    system->inflammation_count++;
    system->stats.inflammation_sites++;

    /* Capture callback and data under mutex to prevent race condition */
    brain_immune_inflammation_cb_t inflammation_callback = system->on_inflammation;
    void* callback_user_data = system->callback_user_data;
    brain_inflammation_site_t site_copy = *site;  /* Copy for safe callback invocation */
    bool enable_logging = system->config.enable_logging;

    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback with copied data (safe after unlock) */
    if (inflammation_callback) {
        inflammation_callback(system, &site_copy, callback_user_data);
    }

    if (enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Inflammation initiated at region %u", region_id);
    }

    return 0;
}

/**
 * @brief Escalate inflammation
 */
int brain_immune_escalate_inflammation(brain_immune_system_t* system, uint32_t site_id) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_escalate_inflammatio", 0.0f);


    brain_inflammation_site_t* site = find_inflammation_by_id(system, site_id);
    if (!site) return -1;

    nimcp_mutex_lock(system->mutex);

    if (site->level < INFLAMMATION_STORM) {
        site->level++;
        site->resource_allocation += 0.2f;
        if (site->resource_allocation > 1.0f) site->resource_allocation = 1.0f;
    }

    nimcp_mutex_unlock(system->mutex);

    /* Alert on escalation */
    brain_immune_broadcast_alert(system, site->triggering_antigen_id, site->level);

    if (system->config.enable_logging) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME,
            "Inflammation escalated at site %u to level %s",
            site_id, brain_immune_inflammation_to_string(site->level));
    }

    return 0;
}

/**
 * @brief Resolve inflammation
 */
int brain_immune_resolve_inflammation(brain_immune_system_t* system, uint32_t site_id) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_resolve_inflammation", 0.0f);


    brain_inflammation_site_t* site = find_inflammation_by_id(system, site_id);
    if (!site) return -1;

    nimcp_mutex_lock(system->mutex);
    site->resolution_progress = 0.01f;  /* Start resolution */
    nimcp_mutex_unlock(system->mutex);

    /* Release anti-inflammatory cytokine */
    uint32_t cytokine_id;
    brain_immune_release_cytokine(system, CYTOKINE_IL10, 0, 0.5f,
                                   site->region_id, &cytokine_id);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Resolution started for inflammation site %u", site_id);
    }

    return 0;
}

/* ============================================================================
 * Memory Response API
 * ============================================================================ */

/**
 * @brief Check for memory cell match
 *
 * WHAT: Search memory B cells for matching antigen
 * WHY:  Enable faster secondary response to known threats
 * HOW:  Uses fuzzy affinity matching to recognize variants
 *
 * CROSS-REACTIVE IMMUNITY:
 * Even if exact match isn't found, partial matches are useful.
 * Returns the best matching memory cell above threshold.
 */
int brain_immune_check_memory(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t* b_cell_id
) {
    if (!system || !b_cell_id) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_check_memory", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    float best_affinity = 0.0f;
    uint32_t best_match_id = 0;

    /* Search memory B cells for best match */
    for (size_t i = 0; i < system->b_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->b_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->b_cell_count);
        }

        brain_b_cell_t* b_cell = &system->b_cells[i];
        if (b_cell->state != B_CELL_MEMORY) continue;

        float affinity = brain_immune_compute_affinity(
            b_cell->receptor, b_cell->receptor_len,
            antigen->epitope, antigen->epitope_len
        );

        /* Track best match */
        if (affinity > best_affinity) {
            best_affinity = affinity;
            best_match_id = b_cell->id;
        }
    }

    /* Return best match if above threshold */
    if (best_affinity >= system->config.recognition_threshold) {
        *b_cell_id = best_match_id;

        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Memory match found: B cell %u, affinity %.2f for antigen %u",
                best_match_id, best_affinity, antigen_id);
        }
        return 0;
    }

    /* Check for cross-reactive immunity (lower threshold) */
    float cross_reactive_threshold = system->config.recognition_threshold * 0.7f;
    if (best_affinity >= cross_reactive_threshold) {
        *b_cell_id = best_match_id;

        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Cross-reactive match found: B cell %u, affinity %.2f (variant of known threat)",
                best_match_id, best_affinity);
        }
        return 0;
    }

    return -1;  /* No memory match */
}

/**
 * @brief Trigger secondary response
 */
int brain_immune_secondary_response(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    uint32_t memory_b_cell_id
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_secondary_response", 0.0f);


    brain_b_cell_t* memory = find_b_cell_by_id(system, memory_b_cell_id);
    if (!memory || memory->state != B_CELL_MEMORY) return -1;

    /* Memory response is faster and stronger */
    nimcp_mutex_lock(system->mutex);

    /* Reactivate memory cell to plasma */
    memory->state = B_CELL_PLASMA;
    memory->bound_antigen_id = antigen_id;
    memory->affinity *= system->config.memory_response_multiplier;
    if (memory->affinity > 1.0f) memory->affinity = 1.0f;
    memory->activation_time = get_timestamp_ms();

    nimcp_mutex_unlock(system->mutex);

    /* Immediately produce antibodies */
    uint32_t antibody_id;
    brain_immune_produce_antibody(system, memory_b_cell_id, ANTIBODY_IGG, &antibody_id);
    brain_immune_execute_antibody(system, antibody_id);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Secondary response triggered for antigen %u", antigen_id);
    }

    return 0;
}

/* ============================================================================
 * Callback Registration
 * ============================================================================ */

int brain_immune_set_antigen_callback(
    brain_immune_system_t* system,
    brain_immune_antigen_cb_t callback,
    void* user_data
) {
    if (!system) return -1;

    /* THREAD SAFETY: Acquire mutex to prevent race with callback invocation */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_antigen_callback", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->on_antigen = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

int brain_immune_set_neutralize_callback(
    brain_immune_system_t* system,
    brain_immune_neutralize_cb_t callback,
    void* user_data
) {
    if (!system) return -1;

    /* THREAD SAFETY: Acquire mutex to prevent race with callback invocation */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_neutralize_callb", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->on_neutralize = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

int brain_immune_set_cytokine_callback(
    brain_immune_system_t* system,
    brain_immune_cytokine_cb_t callback,
    void* user_data
) {
    if (!system) return -1;

    /* THREAD SAFETY: Acquire mutex to prevent race with callback invocation */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_cytokine_callbac", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->on_cytokine = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

int brain_immune_set_inflammation_callback(
    brain_immune_system_t* system,
    brain_immune_inflammation_cb_t callback,
    void* user_data
) {
    if (!system) return -1;

    /* THREAD SAFETY: Acquire mutex to prevent race with callback invocation */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_inflammation_cal", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->on_inflammation = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

int brain_immune_set_kill_callback(
    brain_immune_system_t* system,
    brain_immune_kill_cb_t callback,
    void* user_data
) {
    if (!system) return -1;

    /* THREAD SAFETY: Acquire mutex to prevent race with callback invocation */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_kill_callback", 0.0f);


    nimcp_mutex_lock(system->mutex);
    system->on_kill = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/* ============================================================================
 * Update and Query API
 * ============================================================================ */

/**
 * @brief Update immune system state
 */
int brain_immune_update(brain_immune_system_t* system, uint64_t delta_ms) {
    if (!system || !system->running) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_update", 0.0f);


    nimcp_mutex_lock(system->mutex);

    /* Process pending antigens */
    process_pending_antigens(system);

    /* Decay antibodies */
    decay_antibodies(system, delta_ms);

    /* Update inflammation sites */
    update_inflammation_sites(system, delta_ms);

    /* Update phase */
    update_immune_phase(system);

    /* Calculate health metric */
    float health = 1.0f;
    if (system->antigen_count > 0) {
        size_t neutralized = 0;
        for (size_t i = 0; i < system->antigen_count; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && system->antigen_count > 256) {
                brain_immune_heartbeat("brain_immune_loop",
                                 (float)(i + 1) / (float)system->antigen_count);
            }

            if (system->antigens[i].neutralized) neutralized++;
        }
        health = (float)neutralized / system->antigen_count;
    }
    system->stats.system_health = health;

    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Get statistics
 */
int brain_immune_get_stats(brain_immune_system_t* system, brain_immune_stats_t* stats) {
    if (!system || !stats) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_stats", 0.0f);


    nimcp_mutex_lock(system->mutex);
    *stats = system->stats;

    /* Compute cytokine levels from active cytokines */
    stats->cytokine_il1 = 0.0f;
    stats->cytokine_il6 = 0.0f;
    stats->cytokine_il10 = 0.0f;
    stats->cytokine_tnf = 0.0f;
    stats->cytokine_ifn_gamma = 0.0f;

    for (size_t i = 0; i < system->cytokine_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->cytokine_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->cytokine_count);
        }

        const brain_cytokine_t* cyt = &system->cytokines[i];
        if (!cyt->delivered) continue;  /* Only count active cytokines */

        switch (cyt->type) {
            case BRAIN_CYTOKINE_IL1:
                stats->cytokine_il1 += cyt->concentration;
                break;
            case BRAIN_CYTOKINE_IL6:
                stats->cytokine_il6 += cyt->concentration;
                break;
            case BRAIN_CYTOKINE_IL10:
                stats->cytokine_il10 += cyt->concentration;
                break;
            case BRAIN_CYTOKINE_TNF:
                stats->cytokine_tnf += cyt->concentration;
                break;
            case BRAIN_CYTOKINE_IFN_GAMMA:
                stats->cytokine_ifn_gamma += cyt->concentration;
                break;
            default:
                break;
        }
    }

    /* Clamp cytokine levels to [0, 1] */
    if (stats->cytokine_il1 > 1.0f) stats->cytokine_il1 = 1.0f;
    if (stats->cytokine_il6 > 1.0f) stats->cytokine_il6 = 1.0f;
    if (stats->cytokine_il10 > 1.0f) stats->cytokine_il10 = 1.0f;
    if (stats->cytokine_tnf > 1.0f) stats->cytokine_tnf = 1.0f;
    if (stats->cytokine_ifn_gamma > 1.0f) stats->cytokine_ifn_gamma = 1.0f;

    /* Compute inflammation level from inflammation sites */
    if (system->inflammation_count == 0) {
        stats->inflammation_level = (brain_inflammation_level_t)INFLAMMATION_NONE;
    } else if (system->inflammation_count == 1) {
        stats->inflammation_level = (brain_inflammation_level_t)INFLAMMATION_LOCAL;
    } else if (system->inflammation_count <= 3) {
        stats->inflammation_level = (brain_inflammation_level_t)INFLAMMATION_REGIONAL;
    } else if (system->inflammation_count <= 6) {
        stats->inflammation_level = (brain_inflammation_level_t)INFLAMMATION_SYSTEMIC;
    } else {
        stats->inflammation_level = (brain_inflammation_level_t)INFLAMMATION_STORM;
    }

    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Get immune state snapshot for checkpointing
 *
 * WHAT: Extract immune state for fault tolerance checkpoints
 * WHY:  Include immune health in BFT/recovery checkpoints
 * HOW:  Copy key metrics to BFT-compatible structure
 */
int brain_immune_get_checkpoint_state(brain_immune_system_t* system, void* state) {
    if (!system || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_checkpoint_state", 0.0f);


    #include "utils/fault_tolerance/nimcp_byzantine_fault_tolerance.h"
    bft_immune_state_t* immune_state = (bft_immune_state_t*)state;

    nimcp_mutex_lock(system->mutex);

    /* Count active antigens (unprocessed) */
    uint32_t active_antigens = 0;
    for (size_t i = 0; i < system->antigen_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antigen_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antigen_count);
        }

        if (!system->antigens[i].neutralized) active_antigens++;
    }

    /* Count memory cells */
    uint32_t memory_cells = 0;
    for (size_t i = 0; i < system->b_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->b_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->b_cell_count);
        }

        if (system->b_cells[i].state == B_CELL_MEMORY) memory_cells++;
    }

    /* Count active antibodies */
    uint32_t active_antibodies = 0;
    for (size_t i = 0; i < system->antibody_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antibody_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antibody_count);
        }

        if (system->antibodies[i].active) active_antibodies++;
    }

    /* Fill state */
    immune_state->active_antigens = active_antigens;
    immune_state->active_antibodies = active_antibodies;
    immune_state->memory_cells = memory_cells;
    immune_state->inflammation_sites = (uint32_t)system->inflammation_count;
    immune_state->system_health = system->stats.system_health;
    immune_state->immune_phase = (uint8_t)system->phase;

    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Get current phase
 */
brain_immune_phase_t brain_immune_get_phase(brain_immune_system_t* system) {
    if (!system) return IMMUNE_PHASE_SURVEILLANCE;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_phase", 0.0f);


    return system->phase;
}

/**
 * @brief Check if antigen is neutralized
 */
bool brain_immune_is_neutralized(brain_immune_system_t* system, uint32_t antigen_id) {
    if (!system) return false;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_is_neutralized", 0.0f);


    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    return antigen && antigen->neutralized;
}

/**
 * @brief Get antigen by ID
 */
const brain_antigen_t* brain_immune_get_antigen(brain_immune_system_t* system, uint32_t antigen_id) {
    if (!system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "system is NULL");

        return NULL;

    }
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_antigen", 0.0f);


    return find_antigen_by_id(system, antigen_id);
}

/* ============================================================================
 * BFT Integration Handlers
 * ============================================================================ */

/**
 * @brief Handle BFT accusation event
 *
 * WHAT: Auto-present Byzantine accusation as immune antigen
 * WHY:  Trigger immune response for Byzantine threats
 * HOW:  Create antigen from evidence, activate immune cells
 */
int brain_immune_handle_bft_accusation(
    brain_immune_system_t* system,
    uint32_t accuser_id,
    uint32_t accused_id,
    bft_behavior_t behavior,
    const bft_evidence_t* evidence,
    uint32_t evidence_count
) {
    if (!system || !evidence) return -1;

    // Create antigen from BFT accusation
    uint32_t antigen_id = 0;
    int result = brain_immune_present_byzantine(
        system, accused_id, behavior, evidence, evidence_count, &antigen_id
    );

    if (result != 0) return result;

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "BFT accusation: node %u accused %u of %s -> antigen %u",
            accuser_id, accused_id, bft_behavior_to_string(behavior), antigen_id);
    }

    return 0;
}

/**
 * @brief Handle BFT quarantine action
 *
 * WHAT: Coordinate killer T cell with BFT quarantine
 * WHY:  Unified immune-BFT threat isolation
 * HOW:  Activate killer T, track in inflammation system
 */
int brain_immune_handle_bft_quarantine(
    brain_immune_system_t* system,
    uint32_t node_id,
    uint64_t duration_ms,
    float trust_score
) {
    if (!system) return -1;

    /* Find antigen for this node if exists */
    nimcp_mutex_lock(system->mutex);
    uint32_t antigen_id = 0;
    for (size_t i = 0; i < system->antigen_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->antigen_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->antigen_count);
        }

        if (system->antigens[i].source_node_id == node_id &&
            system->antigens[i].source == ANTIGEN_SOURCE_BFT) {
            antigen_id = system->antigens[i].id;
            break;
        }
    }
    nimcp_mutex_unlock(system->mutex);

    /* If we have an antigen for this node, activate killer T cell */
    if (antigen_id > 0) {
        uint32_t t_cell_id = 0;
        brain_immune_activate_killer_t(system, antigen_id, &t_cell_id);
        brain_immune_t_cell_kill(system, t_cell_id, node_id);

        /* Initiate inflammation for severe cases (low trust) */
        if (trust_score < 20.0f) {
            uint32_t site_id = 0;
            brain_immune_initiate_inflammation(system, node_id, antigen_id, &site_id);
        }
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "BFT quarantine: node %u (trust %.1f%%) -> killer T cell activated",
            node_id, trust_score);
    }

    return 0;
}

/**
 * @brief Handle BFT trust recovery
 *
 * WHAT: Form immune memory on trust restoration
 * WHY:  Map BFT trust recovery to immune memory
 * HOW:  Convert B cells to memory, release IL-10
 */
int brain_immune_handle_bft_trust_recovery(
    brain_immune_system_t* system,
    uint32_t node_id,
    float old_trust,
    float new_trust
) {
    if (!system) return -1;

    /* Find B cells associated with this node */
    nimcp_mutex_lock(system->mutex);
    for (size_t i = 0; i < system->b_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->b_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->b_cell_count);
        }

        if (system->b_cells[i].state == B_CELL_PLASMA ||
            system->b_cells[i].state == B_CELL_ACTIVATED) {
            /* Check if B cell is bound to antigen from this node */
            brain_antigen_t* antigen = find_antigen_by_id(system, system->b_cells[i].bound_antigen_id);
            if (antigen && antigen->source_node_id == node_id &&
                antigen->source == ANTIGEN_SOURCE_BFT) {
                /* Convert to memory B cell */
                uint32_t b_cell_id = system->b_cells[i].id;
                nimcp_mutex_unlock(system->mutex);
                brain_immune_b_cell_to_memory(system, b_cell_id);
                nimcp_mutex_lock(system->mutex);
            }
        }
    }
    nimcp_mutex_unlock(system->mutex);

    /* Release anti-inflammatory cytokine IL-10 (recovery signal) */
    uint32_t cytokine_id = 0;
    brain_immune_release_cytokine(
        system,
        CYTOKINE_IL10,
        node_id,
        0.7f,  /* moderate concentration */
        0,     /* broadcast */
        &cytokine_id
    );

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "BFT trust recovery: node %u (%.1f%% -> %.1f%%) -> memory formation + IL-10 release",
            node_id, old_trust, new_trust);
    }

    return 0;
}

/**
 * @brief Compute affinity between patterns
 *
 * WHAT: Calculate pattern similarity using fuzzy matching
 * WHY:  Determine receptor-epitope binding strength for threat recognition
 * HOW:  Multi-factor scoring: exact matches, partial matches, bit similarity
 *
 * BIOLOGICAL BASIS:
 * Real antibodies don't require exact epitope matches. They can recognize
 * mutated/variant antigens with reduced but non-zero affinity. This enables
 * cross-reactive immunity against threat variants.
 *
 * SCORING COMPONENTS:
 * 1. Exact byte matches (weight: 0.5) - strongest signal
 * 2. Bit similarity for non-exact bytes (weight: 0.3) - catches mutations
 * 3. Length similarity penalty (weight: 0.2) - size matters
 */
float brain_immune_compute_affinity(
    const uint8_t* pattern1,
    size_t len1,
    const uint8_t* pattern2,
    size_t len2
) {
    if (!pattern1 || !pattern2 || len1 == 0 || len2 == 0) return 0.0f;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_compute_affinity", 0.0f);


    size_t min_len = (len1 < len2) ? len1 : len2;
    size_t max_len = (len1 > len2) ? len1 : len2;

    /* Component 1: Exact byte matches */
    size_t exact_matches = 0;
    for (size_t i = 0; i < min_len; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_len > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)min_len);
        }

        if (pattern1[i] == pattern2[i]) {
            exact_matches++;
        }
    }
    float exact_score = (float)exact_matches / (float)max_len;

    /* Component 2: Bit similarity for non-exact bytes (detects mutations) */
    size_t total_bits = min_len * 8;
    size_t matching_bits = 0;
    for (size_t i = 0; i < min_len; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_len > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)min_len);
        }

        uint8_t xor_result = pattern1[i] ^ pattern2[i];
        /* Count matching bits (bits that are NOT different) */
        for (int bit = 0; bit < 8; bit++) {
            /* Phase 8: Loop progress heartbeat */
            if ((bit & 0xFF) == 0 && 8 > 256) {
                brain_immune_heartbeat("brain_immune_loop",
                                 (float)(bit + 1) / (float)8);
            }

            if (!(xor_result & (1 << bit))) {
                matching_bits++;
            }
        }
    }
    float bit_score = (total_bits > 0) ? (float)matching_bits / (float)total_bits : 0.0f;

    /* Component 3: Length similarity (penalize size mismatch) */
    float length_score = (float)min_len / (float)max_len;

    /* Weighted combination */
    float affinity = (0.5f * exact_score) + (0.3f * bit_score) + (0.2f * length_score);

    /* Ensure result is in [0, 1] range */
    if (affinity > 1.0f) affinity = 1.0f;
    if (affinity < 0.0f) affinity = 0.0f;

    return affinity;
}

/* ============================================================================
 * Cytokine and Inflammation Getters
 * ============================================================================ */

float brain_immune_get_cytokine_level(
    brain_immune_system_t* system,
    brain_cytokine_type_t type
) {
    if (!system) return 0.0f;
    if (!system->mutex) return 0.0f;

    /* Lock for thread safety */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_cytokine_level", 0.0f);


    nimcp_platform_mutex_lock(system->mutex);

    float level = 0.0f;
    switch (type) {
        case BRAIN_CYTOKINE_IL1:
            level = system->stats.cytokine_il1;
            break;
        case BRAIN_CYTOKINE_IL6:
            level = system->stats.cytokine_il6;
            break;
        case BRAIN_CYTOKINE_IL10:
            level = system->stats.cytokine_il10;
            break;
        case BRAIN_CYTOKINE_TNF:
            level = system->stats.cytokine_tnf;
            break;
        case BRAIN_CYTOKINE_IFN_GAMMA:
            level = system->stats.cytokine_ifn_gamma;
            break;
        default:
            level = 0.0f;
            break;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return level;
}

brain_inflammation_level_t brain_immune_get_inflammation_level(
    brain_immune_system_t* system
) {
    if (!system) return INFLAMMATION_NONE;
    if (!system->mutex) return INFLAMMATION_NONE;

    /* Lock for thread safety */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_inflammation_lev", 0.0f);


    nimcp_platform_mutex_lock(system->mutex);

    /* Find max inflammation level across all active sites */
    brain_inflammation_level_t max_level = INFLAMMATION_NONE;
    for (size_t i = 0; i < system->inflammation_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->inflammation_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->inflammation_count);
        }

        /* Skip sites that are resolving */
        if (system->inflammation_sites[i].resolution_progress > 0.0f) {
            continue;
        }
        if (system->inflammation_sites[i].level > max_level) {
            max_level = system->inflammation_sites[i].level;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);

    return max_level;
}

/* ============================================================================
 * Imagination Engine Integration
 * ============================================================================ */

/**
 * @brief Compute inflammation level as normalized float
 *
 * WHAT: Convert inflammation level enum to [0.0-1.0] scale
 * WHY:  Needed for modulation calculations
 * HOW:  Map enum values to float range
 *
 * NOTE: Called while mutex is held - assumes system is valid
 */
static float compute_inflammation_float_unlocked(brain_immune_system_t* system) {
    brain_inflammation_level_t max_level = INFLAMMATION_NONE;

    for (size_t i = 0; i < system->inflammation_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->inflammation_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->inflammation_count);
        }

        if (system->inflammation_sites[i].resolution_progress > 0.0f) {
            continue;  /* Skip resolving sites */
        }
        if (system->inflammation_sites[i].level > max_level) {
            max_level = system->inflammation_sites[i].level;
        }
    }

    /* Map enum to [0.0, 1.0] */
    switch (max_level) {
        case INFLAMMATION_NONE:     return 0.0f;
        case INFLAMMATION_LOCAL:    return 0.25f;
        case INFLAMMATION_REGIONAL: return 0.5f;
        case INFLAMMATION_SYSTEMIC: return 0.75f;
        case INFLAMMATION_STORM:    return 1.0f;
        default:                    return 0.0f;
    }
}

/**
 * @brief Send imagination modulation message (internal, mutex already held)
 *
 * WHAT: Compute and send vividness/coherence modifiers to imagination engine
 * WHY:  Sickness behavior includes reduced imaginative capacity
 * HOW:  Higher inflammation = lower vividness/coherence, via bio-async message
 *
 * BIOLOGICAL BASIS:
 * Inflammation triggers "sickness behavior" including cognitive changes:
 * - Reduced creativity and imagination vividness
 * - Lower working memory capacity
 * - Impaired cognitive flexibility
 * Pro-inflammatory cytokines (IL-1, IL-6, TNF-alpha) cross BBB and
 * affect hippocampus and prefrontal cortex, impairing imagination.
 *
 * NOTE: Called from update_inflammation_sites while mutex is held
 */
static void send_imagination_modulation_unlocked(brain_immune_system_t* system) {
    if (!system) return;
    if (!system->bio_context) return;  /* Bio-async not connected */

    /* Compute inflammation level as float [0.0, 1.0] */
    float inflammation = compute_inflammation_float_unlocked(system);

    /* Compute modifiers: higher inflammation = lower vividness/coherence
     * Using exponential decay: modifier = e^(-k * inflammation)
     * k=2 gives: inflammation=0 -> modifier=1.0
     *            inflammation=0.5 -> modifier=0.37
     *            inflammation=1.0 -> modifier=0.14
     */
    float vividness_modifier = expf(-2.0f * inflammation);
    float coherence_modifier = expf(-1.5f * inflammation);  /* Coherence degrades slower */

    /* Clamp to reasonable bounds */
    if (vividness_modifier < 0.1f) vividness_modifier = 0.1f;
    if (coherence_modifier < 0.2f) coherence_modifier = 0.2f;

    /* Build modulation message */
    bio_msg_imagination_modulation_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.type = BIO_MSG_IMAGINATION_VIVIDNESS_UPDATE;
    msg.header.source_module = BIO_MODULE_IMMUNE_BRAIN;
    msg.header.target_module = BIO_MODULE_IMAGINATION;
    msg.header.payload_size = sizeof(msg) - sizeof(bio_message_header_t);
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Slow, modulatory signal */

    msg.modulation_type = 0;  /* 0 = vividness modulation */
    msg.modifier = vividness_modifier;
    msg.source_level = inflammation;
    msg.secondary_level = coherence_modifier;  /* Coherence in secondary field */

    /* Send via bio-async (non-blocking) */
    bio_router_send(system->bio_context, &msg, sizeof(msg), 0);
}

/**
 * @brief Send imagination modulation based on current inflammation
 *
 * WHAT: Public API to send imagination modulation message
 * WHY:  Allow external triggers to update imagination engine
 * HOW:  Lock mutex, compute modulation, send via bio-async
 *
 * @param system Brain immune system
 * @return 0 on success, -1 on error
 */
int brain_immune_send_imagination_modulation(brain_immune_system_t* system) {
    if (!system) return -1;
    if (!system->mutex) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_send_imagination_mod", 0.0f);


    nimcp_mutex_lock(system->mutex);
    send_imagination_modulation_unlocked(system);
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Handler for imagination-related bio-async messages
 *
 * WHAT: Process incoming imagination engine messages
 * WHY:  Bidirectional communication between immune and imagination
 * HOW:  Dispatch based on message type
 *
 * @param msg Incoming message
 * @param msg_size Message size
 * @param response_promise Promise for response (may be NULL)
 * @param user_data Brain immune system pointer
 * @return NIMCP_SUCCESS on success, error code on failure
 */
static nimcp_error_t imagination_message_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data
) {
    (void)response_promise;  /* Unused - no response needed */

    if (!msg || msg_size < sizeof(bio_message_header_t) || !user_data) {
        return -1;
    }

    brain_immune_system_t* system = (brain_immune_system_t*)user_data;
    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    /* Handle imagination engine queries about immune state */
    switch (header->type) {
        case BIO_MSG_IMAGINATION_REQUEST:
            /* Imagination engine requesting current immune state */
            /* Send current modulation values */
            brain_immune_send_imagination_modulation(system);
            break;

        default:
            /* Unknown message type - ignore */
            break;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief KG-driven wiring callback for brain immune module
 */
static int brain_immune_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data)
{
    (void)user_data;

    int registered = 0;
    for (uint32_t i = 0; i < message_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && message_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)message_count);
        }

        switch (message_types[i]) {
            case BIO_MSG_IMAGINATION_REQUEST:
                bio_router_register_handler(ctx, message_types[i], imagination_message_handler);
                registered++;
                break;
            default:
                LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
                    "brain_immune: unknown message type %d in wiring callback", message_types[i]);
                break;
        }
    }

    LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
        "Brain immune: registered %d handlers via KG wiring", registered);
    return 0;
}

/**
 * @brief Register imagination handler with bio-async router
 *
 * WHAT: Register handler for imagination-related messages
 * WHY:  Enable bidirectional communication with imagination engine
 * HOW:  Register handler for imagination message types
 *
 * NOTE: User data comes from bio_module_info_t.user_data set during
 *       bio_router_register_module() call in brain_immune_connect_bio_async()
 *
 * @param system Brain immune system
 * @return 0 on success, -1 on error
 */
int brain_immune_register_imagination_handler(brain_immune_system_t* system) {
    if (!system) return -1;
    if (!system->bio_context) return -1;

    /* Module ID for brain immune */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_register_imagination", 0.0f);


    bio_module_id_t module_id = BIO_MODULE_INTROSPECTION + 0x50;

    /* Try KG-driven wiring callback registration first */
    nimcp_error_t wiring_result = bio_router_register_wiring_callback(
        module_id,
        (void*)brain_immune_wiring_handler_callback,
        system
    );

    if (wiring_result == NIMCP_SUCCESS) {
        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Brain immune: KG-driven wiring callback registered");
        }
        return 0;
    }

    /* Legacy fallback - register handlers directly */
    LEGACY_HANDLER_REGISTRATION(
        nimcp_error_t result = bio_router_register_handler(
            system->bio_context,
            BIO_MSG_IMAGINATION_REQUEST,
            imagination_message_handler
        )
    );

    if (result != NIMCP_SUCCESS) {
        if (system->config.enable_logging) {
            LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME,
                "Failed to register imagination handler");
        }
        return -1;
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Registered imagination engine handler (legacy)");
    }

    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Query KG for module self-awareness information
 * WHY:  Enable introspective self-knowledge about brain immune system
 * HOW:  Look up entity and relations in KG
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge found, 0 otherwise
 */
int brain_immune_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Brain_Immune_System");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                brain_immune_heartbeat("brain_immune_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Brain immune system self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Brain_Immune_System");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Brain_Immune_System");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Exception System Integration
 * ============================================================================ */

/* Exception callbacks - stored separately from other callbacks */
static brain_immune_exception_cb_t g_exception_callback = NULL;
static brain_immune_recovery_cb_t g_recovery_callback = NULL;
static void* g_exception_callback_data = NULL;
static void* g_recovery_callback_data = NULL;

/**
 * @brief Present exception as antigen to immune system
 */
int brain_immune_present_exception(
    brain_immune_system_t* system,
    const nimcp_exception_t* exception,
    uint32_t* antigen_id_out
) {
    if (!system || !exception) return -1;

    /* Use direct field access now that we include nimcp_exception.h */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_present_exception", 0.0f);


    uint32_t severity = (uint32_t)exception->severity;

    /* Get epitope directly from exception struct */
    const uint8_t* epitope = exception->epitope;
    size_t epitope_len = exception->epitope_len;

    /* If epitope not computed, use minimal epitope from error code */
    uint8_t fallback_epitope[NIMCP_EXCEPTION_EPITOPE_SIZE];
    if (epitope_len == 0) {
        memset(fallback_epitope, 0, sizeof(fallback_epitope));
        memcpy(fallback_epitope, &exception->code, sizeof(exception->code));
        memcpy(fallback_epitope + 4, &exception->category, sizeof(exception->category));
        memcpy(fallback_epitope + 8, &exception->severity, sizeof(exception->severity));
        epitope = fallback_epitope;
        epitope_len = 12;
    }

    /* Validate and clamp severity to 1-10 range */
    uint32_t immune_severity = severity;
    if (immune_severity > 10) immune_severity = 10;
    if (immune_severity < 1) immune_severity = 1;

    /* Present as generic antigen */
    int result = brain_immune_present_antigen(
        system,
        ANTIGEN_SOURCE_MANUAL,  /* Exceptions are manually presented */
        epitope,
        epitope_len,
        immune_severity,
        0,  /* source_node = 0 for exceptions */
        antigen_id_out
    );

    /* Trigger exception callback if registered */
    if (result == 0 && g_exception_callback && antigen_id_out) {
        g_exception_callback(system, exception, *antigen_id_out, g_exception_callback_data);
    }

    return result;
}

/**
 * @brief Set exception presentation callback
 */
int brain_immune_set_exception_callback(
    brain_immune_system_t* system,
    brain_immune_exception_cb_t callback,
    void* user_data
) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_exception_callba", 0.0f);


    (void)system;  /* Callback stored globally for simplicity */
    g_exception_callback = callback;
    g_exception_callback_data = user_data;
    return 0;
}

/**
 * @brief Set recovery completion callback
 */
int brain_immune_set_recovery_callback(
    brain_immune_system_t* system,
    brain_immune_recovery_cb_t callback,
    void* user_data
) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_set_recovery_callbac", 0.0f);


    (void)system;  /* Callback stored globally for simplicity */
    g_recovery_callback = callback;
    g_recovery_callback_data = user_data;
    return 0;
}

/**
 * @brief Notify immune system of recovery result
 */
int brain_immune_notify_recovery_result(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    int recovery_action,
    bool success
) {
    if (!system) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_notify_recovery_resu", 0.0f);


    LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
        "Recovery result: antigen=%u, action=%d, success=%s",
        antigen_id, recovery_action, success ? "true" : "false");

    nimcp_mutex_lock(system->mutex);

    /* Find antigen */
    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (antigen) {
        if (success) {
            antigen->neutralized = true;
            system->stats.threats_neutralized++;

            /* Release anti-inflammatory cytokine on success */
            nimcp_mutex_unlock(system->mutex);
            uint32_t cytokine_id;
            brain_immune_release_cytokine(system, BRAIN_CYTOKINE_IL10, 0, 0.5f, 0, &cytokine_id);
        } else {
            nimcp_mutex_unlock(system->mutex);
        }
    } else {
        nimcp_mutex_unlock(system->mutex);
    }

    /* Trigger callback */
    if (g_recovery_callback) {
        g_recovery_callback(system, antigen_id, recovery_action, success, g_recovery_callback_data);
    }

    return 0;
}

/**
 * @brief Get recommended recovery action for antigen
 */
int brain_immune_get_recovery_recommendation(
    brain_immune_system_t* system,
    uint32_t antigen_id,
    int* action_out
) {
    if (!system || !action_out) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_heartbeat("brain_immune_get_recovery_recomme", 0.0f);


    nimcp_mutex_lock(system->mutex);

    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Check if we have a memory cell that matches this antigen */
    for (size_t i = 0; i < system->b_cell_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->b_cell_count > 256) {
            brain_immune_heartbeat("brain_immune_loop",
                             (float)(i + 1) / (float)system->b_cell_count);
        }

        brain_b_cell_t* b_cell = &system->b_cells[i];
        if (b_cell->state == B_CELL_MEMORY) {
            float affinity = brain_immune_compute_affinity(
                b_cell->receptor,
                b_cell->receptor_len,
                antigen->epitope,
                antigen->epitope_len
            );

            if (affinity >= system->config.recognition_threshold) {
                /* Found matching memory cell - recommend based on source */
                switch (antigen->source) {
                    case ANTIGEN_SOURCE_BBB:
                        *action_out = 8;  /* RECOVERY_ACTION_QUARANTINE */
                        break;
                    case ANTIGEN_SOURCE_BFT:
                        *action_out = 5;  /* RECOVERY_ACTION_RESTART_THREAD */
                        break;
                    case ANTIGEN_SOURCE_ANOMALY:
                        *action_out = 2;  /* RECOVERY_ACTION_GC */
                        break;
                    default:
                        *action_out = 1;  /* RECOVERY_ACTION_RETRY */
                        break;
                }

                nimcp_mutex_unlock(system->mutex);
                LOG_MODULE_DEBUG(BRAIN_IMMUNE_MODULE_NAME,
                    "Memory-based recovery recommendation: action=%d for antigen=%u",
                    *action_out, antigen_id);
                return 0;
            }
        }
    }

    /* Store severity before unlocking */
    uint32_t severity = antigen->severity;

    nimcp_mutex_unlock(system->mutex);

    /* No memory match - return default recommendation based on severity */
    if (severity >= 7) {
        *action_out = 4;  /* RECOVERY_ACTION_ROLLBACK */
    } else if (severity >= 5) {
        *action_out = 2;  /* RECOVERY_ACTION_GC */
    } else {
        *action_out = 1;  /* RECOVERY_ACTION_RETRY */
    }

    return 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void brain_immune_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_brain_immune_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int brain_immune_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "brain_immune_training_begin: NULL argument");
        return -1;
    }
    brain_immune_heartbeat_instance(NULL, "brain_immune_training_begin", 0.0f);
    return 0;
}

int brain_immune_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "brain_immune_training_end: NULL argument");
        return -1;
    }
    brain_immune_heartbeat_instance(NULL, "brain_immune_training_end", 1.0f);
    return 0;
}

int brain_immune_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "brain_immune_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    brain_immune_heartbeat_instance(NULL, "brain_immune_training_step", progress);
    return 0;
}
