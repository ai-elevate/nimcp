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
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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
        case CYTOKINE_IL1:       return "IL-1";
        case CYTOKINE_IL6:       return "IL-6";
        case CYTOKINE_IL10:      return "IL-10";
        case CYTOKINE_TNF_ALPHA: return "TNF-alpha";
        case CYTOKINE_IFN_GAMMA: return "IFN-gamma";
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
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

/**
 * @brief Find antigen by ID
 */
static brain_antigen_t* find_antigen_by_id(brain_immune_system_t* system, uint32_t id) {
    if (!system) return NULL;
    for (size_t i = 0; i < system->antigen_count; i++) {
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
    if (!system) return NULL;
    for (size_t i = 0; i < system->b_cell_count; i++) {
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
    if (!system) return NULL;
    for (size_t i = 0; i < system->t_cell_count; i++) {
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
    if (!system) return NULL;
    for (size_t i = 0; i < system->antibody_count; i++) {
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
    if (!system) return NULL;
    for (size_t i = 0; i < system->inflammation_count; i++) {
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
 */
static void update_inflammation_sites(brain_immune_system_t* system, uint64_t delta_ms) {
    if (!system) return;

    for (size_t i = 0; i < system->inflammation_count; i++) {
        brain_inflammation_site_t* site = &system->inflammation_sites[i];

        /* Progress resolution if active */
        if (site->resolution_progress > 0 && site->resolution_progress < 1.0f) {
            site->resolution_progress += 0.001f * delta_ms;
            if (site->resolution_progress >= 1.0f) {
                site->resolution_progress = 1.0f;
                site->level = INFLAMMATION_NONE;
            }
        }
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
    brain_immune_system_t* system = nimcp_calloc(1, sizeof(brain_immune_system_t));
    if (!system) return NULL;

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

    brain_immune_stop(system);

    if (system->mutex) {
        nimcp_mutex_destroy(system->mutex);
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
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);
    system->bbb_system = bbb_system;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to BBB security");
    }

    return 0;
}

/**
 * @brief Connect to BFT
 */
int brain_immune_connect_bft(brain_immune_system_t* system, bft_context_t* bft_context) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);
    system->bft_context = bft_context;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to BFT");
    }

    return 0;
}

/**
 * @brief Connect to swarm immune
 */
int brain_immune_connect_swarm(brain_immune_system_t* system, NimcpSwarmImmuneSystem* swarm_immune) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);
    system->swarm_immune = swarm_immune;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME, "Connected to swarm immune");
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

    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback */
    if (system->on_antigen) {
        system->on_antigen(system, antigen, system->callback_user_data);
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
    uint32_t immune_severity;
    switch (severity) {
        case BBB_SEVERITY_LOW:      immune_severity = 3; break;
        case BBB_SEVERITY_MEDIUM:   immune_severity = 5; break;
        case BBB_SEVERITY_HIGH:     immune_severity = 7; break;
        case BBB_SEVERITY_CRITICAL: immune_severity = 10; break;
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
    uint32_t severity;
    switch (threat->severity) {
        case SEVERITY_LOW:      severity = 3; break;
        case SEVERITY_MEDIUM:   severity = 5; break;
        case SEVERITY_HIGH:     severity = 7; break;
        case SEVERITY_CRITICAL: severity = 10; break;
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

    brain_t_cell_t* t_cell = find_t_cell_by_id(system, t_cell_id);
    if (!t_cell || t_cell->type != T_CELL_KILLER) return -1;

    /* Execute quarantine via BFT if connected */
    if (system->bft_context) {
        bft_quarantine_node(system->bft_context, target_node,
                           system->config.antibody_half_life_ms);
    }

    nimcp_mutex_lock(system->mutex);
    t_cell->kills++;
    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback */
    if (system->on_kill) {
        system->on_kill(system, t_cell, target_node, system->callback_user_data);
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
    if (system->antibody_count >= system->antibody_capacity) return -1;

    brain_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell || b_cell->state != B_CELL_PLASMA) return -1;

    nimcp_mutex_lock(system->mutex);

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

    brain_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody || !antibody->active) return -1;

    /* Execute swarm response if connected */
    if (system->swarm_immune) {
        /* Would call nimcp_swarm_immune_generate_response */
        antibody->swarm_response_id = 1;  /* Placeholder */
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
            "Antibody %u executed response type %d",
            antibody_id, antibody->swarm_response);
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

    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    brain_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) return -1;

    nimcp_mutex_lock(system->mutex);

    antigen->neutralized = true;
    antigen->response_count++;
    antibody->neutralizations++;
    system->stats.threats_neutralized++;

    nimcp_mutex_unlock(system->mutex);

    /* AUTO-LEARNING: Convert producing B cell to memory for future recognition */
    brain_b_cell_t* producer = find_b_cell_by_id(system, antibody->producer_b_cell_id);
    if (producer && producer->state != B_CELL_MEMORY) {
        /* Successful neutralization triggers memory formation */
        brain_immune_b_cell_to_memory(system, antibody->producer_b_cell_id);

        if (system->config.enable_logging) {
            LOG_MODULE_INFO(BRAIN_IMMUNE_MODULE_NAME,
                "Auto-learning: B cell %u converted to memory after neutralizing antigen %u",
                antibody->producer_b_cell_id, antigen_id);
        }
    }

    /* Trigger callback */
    if (system->on_neutralize) {
        system->on_neutralize(system, antigen_id, antibody, system->callback_user_data);
    }

    if (system->config.enable_logging) {
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
    if (system->cytokine_count >= system->cytokine_capacity) return -1;

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
        case CYTOKINE_IL1:
        case CYTOKINE_IL6:
        case CYTOKINE_TNF_ALPHA:
            cytokine->message_type = BIO_MSG_SECURITY_ALERT;
            break;
        case CYTOKINE_IFN_GAMMA:
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
        if (system->cytokines[i].pro_inflammatory) {
            total_pro_inflammatory += system->cytokines[i].concentration;
        }
    }
    if (total_pro_inflammatory >= system->config.cytokine_storm_threshold) {
        LOG_MODULE_WARN(BRAIN_IMMUNE_MODULE_NAME, "Cytokine storm risk detected!");
    }

    nimcp_mutex_unlock(system->mutex);

    /* Send bio-async message if connected */
    if (system->bio_context && target_region == 0) {
        /* Broadcast - would construct and send bio message here */
        cytokine->delivered = true;
    }

    /* Trigger callback */
    if (system->on_cytokine) {
        system->on_cytokine(system, cytokine, system->callback_user_data);
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
    brain_cytokine_type_t type;
    float concentration;

    switch (severity) {
        case INFLAMMATION_LOCAL:
            type = CYTOKINE_IL1;
            concentration = 0.3f;
            break;
        case INFLAMMATION_REGIONAL:
            type = CYTOKINE_IL6;
            concentration = 0.5f;
            break;
        case INFLAMMATION_SYSTEMIC:
            type = CYTOKINE_TNF_ALPHA;
            concentration = 0.7f;
            break;
        case INFLAMMATION_STORM:
            type = CYTOKINE_TNF_ALPHA;
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

    nimcp_mutex_unlock(system->mutex);

    /* Trigger callback */
    if (system->on_inflammation) {
        system->on_inflammation(system, site, system->callback_user_data);
    }

    if (system->config.enable_logging) {
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

    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) return -1;

    float best_affinity = 0.0f;
    uint32_t best_match_id = 0;

    /* Search memory B cells for best match */
    for (size_t i = 0; i < system->b_cell_count; i++) {
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
    system->on_antigen = callback;
    system->callback_user_data = user_data;
    return 0;
}

int brain_immune_set_neutralize_callback(
    brain_immune_system_t* system,
    brain_immune_neutralize_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    system->on_neutralize = callback;
    system->callback_user_data = user_data;
    return 0;
}

int brain_immune_set_cytokine_callback(
    brain_immune_system_t* system,
    brain_immune_cytokine_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    system->on_cytokine = callback;
    system->callback_user_data = user_data;
    return 0;
}

int brain_immune_set_inflammation_callback(
    brain_immune_system_t* system,
    brain_immune_inflammation_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    system->on_inflammation = callback;
    system->callback_user_data = user_data;
    return 0;
}

int brain_immune_set_kill_callback(
    brain_immune_system_t* system,
    brain_immune_kill_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    system->on_kill = callback;
    system->callback_user_data = user_data;
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

    nimcp_mutex_lock(system->mutex);
    *stats = system->stats;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Get current phase
 */
brain_immune_phase_t brain_immune_get_phase(brain_immune_system_t* system) {
    if (!system) return IMMUNE_PHASE_SURVEILLANCE;
    return system->phase;
}

/**
 * @brief Check if antigen is neutralized
 */
bool brain_immune_is_neutralized(brain_immune_system_t* system, uint32_t antigen_id) {
    if (!system) return false;

    brain_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    return antigen && antigen->neutralized;
}

/**
 * @brief Get antigen by ID
 */
const brain_antigen_t* brain_immune_get_antigen(brain_immune_system_t* system, uint32_t antigen_id) {
    if (!system) return NULL;
    return find_antigen_by_id(system, antigen_id);
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

    size_t min_len = (len1 < len2) ? len1 : len2;
    size_t max_len = (len1 > len2) ? len1 : len2;

    /* Component 1: Exact byte matches */
    size_t exact_matches = 0;
    for (size_t i = 0; i < min_len; i++) {
        if (pattern1[i] == pattern2[i]) {
            exact_matches++;
        }
    }
    float exact_score = (float)exact_matches / (float)max_len;

    /* Component 2: Bit similarity for non-exact bytes (detects mutations) */
    size_t total_bits = min_len * 8;
    size_t matching_bits = 0;
    for (size_t i = 0; i < min_len; i++) {
        uint8_t xor_result = pattern1[i] ^ pattern2[i];
        /* Count matching bits (bits that are NOT different) */
        for (int bit = 0; bit < 8; bit++) {
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
