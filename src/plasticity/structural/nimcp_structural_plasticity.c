/**
 * @file nimcp_structural_plasticity.c
 * @brief Structural Plasticity Implementation
 */

#include "plasticity/structural/nimcp_structural_plasticity.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct structural_plasticity_system {
    /* Configuration */
    structural_plasticity_config_t config;

    /* Spine tracking */
    synapse_structural_state_t* spines;
    uint32_t num_spines;
    uint32_t max_spines;
    uint32_t next_synapse_id;

    /* Callbacks */
    structural_change_callback_t callback;
    void* callback_user_data;

    /* Statistics */
    uint64_t total_formations;
    uint64_t total_eliminations;
    uint64_t total_stabilizations;
    uint64_t total_potentiations;
    uint64_t total_pruning_starts;

    /* Thread safety */
    nimcp_platform_mutex_t* mutex;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * WHAT: Find spine by synapse ID
 * WHY:  Need to locate spine in array
 * HOW:  Linear search (could optimize with hash table)
 */
static synapse_structural_state_t* find_spine(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system || !system->spines) {
        return NULL;
    }

    for (uint32_t i = 0; i < system->num_spines; i++) {
        if (system->spines[i].synapse_id == synapse_id &&
            system->spines[i].state != SYNAPSE_STATE_ELIMINATED) {
            return &system->spines[i];
        }
    }
    return NULL;
}

/**
 * @brief Find spine by ID including eliminated spines (for state queries)
 */
static synapse_structural_state_t* find_spine_any_state(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system || !system->spines) {
        return NULL;
    }

    for (uint32_t i = 0; i < system->num_spines; i++) {
        if (system->spines[i].synapse_id == synapse_id) {
            return &system->spines[i];
        }
    }
    return NULL;
}

/**
 * WHAT: Initialize spine morphology for nascent state
 * WHY:  New spines start as thin, unstable structures
 * HOW:  Set small volume, PSD, high motility
 */
static void init_nascent_morphology(spine_morphology_t* morph) {
    if (!morph) return;

    morph->spine_volume = STRUCTURAL_VOLUME_NASCENT_MIN +
        (STRUCTURAL_VOLUME_NASCENT_MAX - STRUCTURAL_VOLUME_NASCENT_MIN) * 0.5f;
    morph->psd_size = STRUCTURAL_PSD_NASCENT_MIN +
        (STRUCTURAL_PSD_NASCENT_MAX - STRUCTURAL_PSD_NASCENT_MIN) * 0.5f;
    morph->actin_dynamics = STRUCTURAL_ACTIN_GROWTH_RATE;
    morph->spine_stability = 0.2f;  /* Low stability */
    morph->spine_motility = 0.8f;   /* High motility */
    morph->camkii_concentration = 0.1f;
    morph->ampar_count = 5.0f;      /* Few receptors */
    morph->nmdar_count = 3.0f;
}

/**
 * WHAT: Update morphology to stable state
 * WHY:  Stabilized spines are larger, more stable
 * HOW:  Increase volume, PSD, stability
 */
static void update_to_stable_morphology(spine_morphology_t* morph) {
    if (!morph) return;

    morph->spine_volume = STRUCTURAL_VOLUME_STABLE_MIN +
        (STRUCTURAL_VOLUME_STABLE_MAX - STRUCTURAL_VOLUME_STABLE_MIN) * 0.7f;
    morph->psd_size = STRUCTURAL_PSD_STABLE_MIN +
        (STRUCTURAL_PSD_STABLE_MAX - STRUCTURAL_PSD_STABLE_MIN) * 0.7f;
    morph->actin_dynamics = 0.02f;  /* Slower dynamics */
    morph->spine_stability = 0.8f;
    morph->spine_motility = 0.2f;
    morph->camkii_concentration = 0.6f;
    morph->ampar_count = 30.0f;
    morph->nmdar_count = 15.0f;
}

/**
 * WHAT: Update morphology to potentiated state
 * WHY:  LTP enlarges spines
 * HOW:  Maximize volume, PSD, receptors
 */
static void update_to_potentiated_morphology(spine_morphology_t* morph) {
    if (!morph) return;

    morph->spine_volume = STRUCTURAL_VOLUME_POTENTIATED_MIN +
        (STRUCTURAL_VOLUME_POTENTIATED_MAX - STRUCTURAL_VOLUME_POTENTIATED_MIN) * 0.8f;
    morph->psd_size = STRUCTURAL_PSD_POTENTIATED_MIN +
        (STRUCTURAL_PSD_POTENTIATED_MAX - STRUCTURAL_PSD_POTENTIATED_MIN) * 0.8f;
    morph->actin_dynamics = 0.01f;
    morph->spine_stability = 0.95f;
    morph->spine_motility = 0.05f;
    morph->camkii_concentration = 0.9f;
    morph->ampar_count = 80.0f;
    morph->nmdar_count = 30.0f;
}

/**
 * WHAT: Invoke structural change callback
 * WHY:  Notify listeners of state transitions
 * HOW:  Call registered callback if present
 */
static void invoke_callback(
    structural_plasticity_system_t* system,
    structural_event_t event,
    uint32_t synapse_id,
    synapse_state_t old_state,
    synapse_state_t new_state
) {
    if (!system || !system->callback) {
        return;
    }

    system->callback(event, synapse_id, old_state, new_state,
                    system->callback_user_data);
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int structural_plasticity_default_config(structural_plasticity_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config pointer");
        return -1;
    }

    config->formation_threshold_hz = STRUCTURAL_FORMATION_THRESHOLD_MAX;
    config->formation_rate = 0.01f;
    config->maturation_time_sec = STRUCTURAL_MATURATION_TIME_TYPICAL;
    config->stabilization_threshold = 10.0f;
    config->require_sleep_consolidation = true;
    config->pruning_threshold_hz = 0.5f;  /* Lower threshold to avoid premature pruning */
    config->pruning_rate = 0.005f;
    config->inactivity_timeout_sec = 3600.0f * 24.0f;  /* 24 hours */
    config->ltp_potentiation_threshold = 10.0f;
    config->potentiation_decay_rate = 0.001f;
    config->enable_immune_pruning = true;
    config->complement_sensitivity = 1.0f;
    config->enable_sleep_consolidation = true;
    config->sleep_consolidation_boost = 2.0f;
    config->max_spines = STRUCTURAL_MAX_SPINES;
    config->spine_density_limit = 1000.0f;

    return 0;
}

structural_plasticity_system_t* structural_plasticity_create(
    const structural_plasticity_config_t* config
) {
    structural_plasticity_system_t* system =
        (structural_plasticity_system_t*)nimcp_malloc(sizeof(*system));
    if (!system) {
        NIMCP_LOGGING_ERROR("Failed to allocate structural plasticity system");
        return NULL;
    }

    memset(system, 0, sizeof(*system));

    /* Apply configuration */
    if (config) {
        system->config = *config;
    } else {
        structural_plasticity_default_config(&system->config);
    }

    /* Allocate spine array */
    system->max_spines = system->config.max_spines;
    system->spines = (synapse_structural_state_t*)nimcp_malloc(
        sizeof(synapse_structural_state_t) * system->max_spines);
    if (!system->spines) {
        NIMCP_LOGGING_ERROR("Failed to allocate spine array");
        nimcp_free(system);
        return NULL;
    }

    memset(system->spines, 0,
           sizeof(synapse_structural_state_t) * system->max_spines);

    /* Create mutex */
    system->mutex = nimcp_platform_mutex_create();
    if (!system->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(system->spines);
        nimcp_free(system);
        return NULL;
    }

    system->next_synapse_id = 1;

    NIMCP_LOGGING_INFO("Structural plasticity system created");
    return system;
}

void structural_plasticity_destroy(structural_plasticity_system_t* system) {
    if (!system) return;

    if (system->mutex) {
        nimcp_platform_mutex_destroy(system->mutex);
    }

    if (system->spines) {
        nimcp_free(system->spines);
    }

    nimcp_free(system);
    NIMCP_LOGGING_INFO("Structural plasticity system destroyed");
}

/* ============================================================================
 * Formation and Elimination Implementation
 * ============================================================================ */

int structural_plasticity_form_synapse(
    structural_plasticity_system_t* system,
    uint32_t pre_neuron_id,
    uint32_t post_neuron_id,
    float activity_hz,
    uint32_t* synapse_id
) {
    if (!system || !synapse_id) {
        NIMCP_LOGGING_ERROR("NULL parameter");
        return -1;
    }

    /* Validate activity_hz to prevent NaN propagation */
    if (isnan(activity_hz) || isinf(activity_hz)) {
        activity_hz = 0.0f;
    }
    if (activity_hz < 0.0f) activity_hz = 0.0f;

    nimcp_platform_mutex_lock(system->mutex);

    /* BOUNDS VALIDATION: Check spine array limits
     * WHAT: Verify we have room in the spine array
     * WHY:  Prevent array overflow and memory corruption
     * HOW:  Check both num_spines < max_spines and spines array exists
     */
    if (!system->spines) {
        NIMCP_LOGGING_ERROR("Spine array not initialized");
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    if (system->num_spines >= system->max_spines) {
        NIMCP_LOGGING_WARN("Spine limit reached");
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    /* Find free slot with explicit bounds check */
    uint32_t slot = system->num_spines;
    if (slot >= system->max_spines) {
        NIMCP_LOGGING_ERROR("Slot index exceeds max_spines");
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    synapse_structural_state_t* spine = &system->spines[slot];

    /* Initialize spine */
    memset(spine, 0, sizeof(*spine));
    spine->synapse_id = system->next_synapse_id++;
    spine->pre_neuron_id = pre_neuron_id;
    spine->post_neuron_id = post_neuron_id;
    spine->state = SYNAPSE_STATE_NASCENT;
    init_nascent_morphology(&spine->morphology);
    spine->recent_activity_hz = activity_hz;
    spine->formation_time = 0;  /* Will be set by caller */
    spine->maturation_progress = 0.0f;
    spine->consolidation_tagged = false;
    spine->complement_tagged = false;
    spine->pruning_urgency = 0.0f;
    spine->formation_events = 1;

    system->num_spines++;
    system->total_formations++;
    *synapse_id = spine->synapse_id;

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;
    uint32_t cb_synapse_id = spine->synapse_id;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Notify callback after releasing mutex to avoid deadlock */
    if (callback) {
        callback(STRUCTURAL_EVENT_FORMATION, cb_synapse_id,
                SYNAPSE_STATE_ELIMINATED, SYNAPSE_STATE_NASCENT,
                callback_user_data);
    }

    NIMCP_LOGGING_DEBUG("Formed synapse %u (pre=%u, post=%u)",
                       spine->synapse_id, pre_neuron_id, post_neuron_id);
    return 0;
}

int structural_plasticity_eliminate_synapse(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        NIMCP_LOGGING_WARN("Synapse %u not found", synapse_id);
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    synapse_state_t old_state = spine->state;
    spine->state = SYNAPSE_STATE_ELIMINATED;
    system->total_eliminations++;

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Notify callback after releasing mutex to avoid deadlock */
    if (callback) {
        callback(STRUCTURAL_EVENT_ELIMINATION, synapse_id,
                old_state, SYNAPSE_STATE_ELIMINATED, callback_user_data);
    }

    NIMCP_LOGGING_DEBUG("Eliminated synapse %u", synapse_id);
    return 0;
}

bool structural_plasticity_should_form(
    const structural_plasticity_system_t* system,
    float activity_hz
) {
    if (!system) return false;
    return activity_hz >= system->config.formation_threshold_hz;
}

bool structural_plasticity_should_prune(
    const structural_plasticity_system_t* system,
    const synapse_structural_state_t* synapse
) {
    if (!system || !synapse) return false;

    /* Complement-tagged synapses should be pruned */
    if (system->config.enable_immune_pruning && synapse->complement_tagged) {
        return true;
    }

    /* Low activity triggers pruning */
    if (synapse->recent_activity_hz < system->config.pruning_threshold_hz) {
        return true;
    }

    return false;
}

/* ============================================================================
 * Stabilization and Potentiation Implementation
 * ============================================================================ */

int structural_plasticity_stabilize_synapse(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    if (spine->state != SYNAPSE_STATE_NASCENT) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;  /* Can only stabilize nascent spines */
    }

    synapse_state_t old_state = spine->state;
    spine->state = SYNAPSE_STATE_STABLE;
    update_to_stable_morphology(&spine->morphology);
    spine->maturation_progress = 1.0f;
    system->total_stabilizations++;

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Notify callback after releasing mutex to avoid deadlock */
    if (callback) {
        callback(STRUCTURAL_EVENT_STABILIZATION, synapse_id,
                old_state, SYNAPSE_STATE_STABLE, callback_user_data);
    }

    NIMCP_LOGGING_DEBUG("Stabilized synapse %u", synapse_id);
    return 0;
}

int structural_plasticity_potentiate_synapse(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    if (spine->state != SYNAPSE_STATE_STABLE) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;  /* Can only potentiate stable spines */
    }

    synapse_state_t old_state = spine->state;
    spine->state = SYNAPSE_STATE_POTENTIATED;
    update_to_potentiated_morphology(&spine->morphology);
    spine->potentiation_events++;
    system->total_potentiations++;

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Notify callback after releasing mutex to avoid deadlock */
    if (callback) {
        callback(STRUCTURAL_EVENT_POTENTIATION, synapse_id,
                old_state, SYNAPSE_STATE_POTENTIATED, callback_user_data);
    }

    NIMCP_LOGGING_DEBUG("Potentiated synapse %u", synapse_id);
    return 0;
}

int structural_plasticity_tag_for_consolidation(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    spine->consolidation_tagged = true;

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

/* ============================================================================
 * Activity Tracking Implementation
 * ============================================================================ */

int structural_plasticity_update_activity(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    uint64_t current_time
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    /* Update activity with exponential moving average */
    float alpha = 0.1f;  /* Smoothing factor */
    if (spine->last_active_time > 0) {
        float dt_sec = (current_time - spine->last_active_time) / 1000.0f;

        /* Guard against division by very small dt (cap at 1ms = 1000Hz max) */
        const float MIN_DT_SEC = 0.001f;  /* 1 millisecond */
        if (dt_sec >= MIN_DT_SEC) {
            float instant_rate = 1.0f / dt_sec;

            /* Cap maximum instantaneous rate to prevent overflow (max 10kHz) */
            const float MAX_RATE_HZ = 10000.0f;
            if (instant_rate > MAX_RATE_HZ) {
                instant_rate = MAX_RATE_HZ;
            }

            spine->recent_activity_hz =
                alpha * instant_rate + (1.0f - alpha) * spine->recent_activity_hz;
        }
    }

    spine->last_active_time = current_time;

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

int structural_plasticity_record_ltp(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    float ltp_magnitude
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    /* PARAMETER VALIDATION: Validate LTP magnitude
     * WHAT: Ensure ltp_magnitude is finite and non-negative
     * WHY:  Invalid values cause accumulator corruption
     * HOW:  Check for NaN/Inf and clamp to reasonable range
     */
    if (isnan(ltp_magnitude) || isinf(ltp_magnitude)) {
        return 0;  /* Skip invalid LTP event */
    }
    if (ltp_magnitude < 0.0f) ltp_magnitude = 0.0f;
    if (ltp_magnitude > 100.0f) ltp_magnitude = 100.0f;  /* Cap at reasonable max */

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    /* Accumulate with overflow protection */
    spine->ltp_accumulator += ltp_magnitude;
    if (spine->ltp_accumulator > 10000.0f) {
        spine->ltp_accumulator = 10000.0f;  /* Cap accumulator */
    }
    if (isnan(spine->ltp_accumulator)) {
        spine->ltp_accumulator = 0.0f;  /* Reset on NaN */
    }

    /* Check potentiation threshold */
    bool should_notify = false;
    synapse_state_t old_state = spine->state;
    if (spine->state == SYNAPSE_STATE_STABLE &&
        spine->ltp_accumulator >= system->config.ltp_potentiation_threshold) {
        /* Auto-potentiate */
        spine->state = SYNAPSE_STATE_POTENTIATED;
        update_to_potentiated_morphology(&spine->morphology);
        spine->potentiation_events++;
        system->total_potentiations++;
        should_notify = true;
    }

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Notify callback after releasing mutex to avoid deadlock */
    if (should_notify && callback) {
        callback(STRUCTURAL_EVENT_POTENTIATION, synapse_id,
                old_state, SYNAPSE_STATE_POTENTIATED, callback_user_data);
    }

    return 0;
}

int structural_plasticity_record_ltd(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    float ltd_magnitude
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    /* PARAMETER VALIDATION: Validate LTD magnitude
     * WHAT: Ensure ltd_magnitude is finite and non-negative
     * WHY:  Invalid values cause accumulator corruption
     * HOW:  Check for NaN/Inf and clamp to reasonable range
     */
    if (isnan(ltd_magnitude) || isinf(ltd_magnitude)) {
        return 0;  /* Skip invalid LTD event */
    }
    if (ltd_magnitude < 0.0f) ltd_magnitude = 0.0f;
    if (ltd_magnitude > 100.0f) ltd_magnitude = 100.0f;  /* Cap at reasonable max */

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    /* Accumulate with overflow protection */
    spine->ltd_accumulator += ltd_magnitude;
    if (spine->ltd_accumulator > 10000.0f) {
        spine->ltd_accumulator = 10000.0f;  /* Cap accumulator */
    }
    if (isnan(spine->ltd_accumulator)) {
        spine->ltd_accumulator = 0.0f;  /* Reset on NaN */
    }

    /* LTD increases pruning urgency */
    float urgency_increase = ltd_magnitude * 0.1f;
    if (!isnan(urgency_increase) && urgency_increase > 0.0f) {
        spine->pruning_urgency += urgency_increase;
    }
    if (spine->pruning_urgency > 1.0f) {
        spine->pruning_urgency = 1.0f;
    }
    if (spine->pruning_urgency < 0.0f) {
        spine->pruning_urgency = 0.0f;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

/* ============================================================================
 * Update Implementation
 * ============================================================================ */

/* Max deferred callbacks during update */
#define MAX_DEFERRED_CALLBACKS 64

typedef struct {
    structural_event_t event;
    uint32_t synapse_id;
    synapse_state_t old_state;
    synapse_state_t new_state;
} deferred_callback_t;

int structural_plasticity_update(
    structural_plasticity_system_t* system,
    float delta_sec
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    /* Validate delta_sec to prevent numerical issues */
    if (isnan(delta_sec) || isinf(delta_sec) || delta_sec < 0.0f) {
        return 0;  /* Skip update with invalid time step */
    }
    /* Cap delta_sec to prevent extreme values */
    if (delta_sec > 3600.0f) {  /* 1 hour max */
        delta_sec = 3600.0f;
    }

    /* Deferred callbacks to invoke after releasing mutex */
    deferred_callback_t deferred[MAX_DEFERRED_CALLBACKS];
    uint32_t num_deferred = 0;

    nimcp_platform_mutex_lock(system->mutex);

    /* BOUNDS VALIDATION: Verify spine array before iteration */
    if (!system->spines || system->num_spines > system->max_spines) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    for (uint32_t i = 0; i < system->num_spines; i++) {
        /* Defensive bounds check per iteration */
        if (i >= system->max_spines) break;
        synapse_structural_state_t* spine = &system->spines[i];

        if (spine->state == SYNAPSE_STATE_ELIMINATED) {
            continue;
        }

        /* Decay activity rate during inactivity (exponential decay with time constant ~20s) */
        float decay_rate = 0.05f;  /* 1/tau where tau = 20 seconds (accelerated for simulations) */
        float decay_factor = expf(-decay_rate * delta_sec);
        float old_activity = spine->recent_activity_hz;
        (void)old_activity;  /* Suppress unused warning */
        spine->recent_activity_hz *= decay_factor;

        /* Update maturation progress for nascent spines */
        if (spine->state == SYNAPSE_STATE_NASCENT) {
            float maturation_rate = 1.0f / system->config.maturation_time_sec;
            spine->maturation_progress += delta_sec * maturation_rate;

            /* Check for auto-stabilization */
            if (spine->maturation_progress >= 1.0f &&
                spine->recent_activity_hz >= system->config.stabilization_threshold) {

                /* Require sleep consolidation if enabled */
                bool can_stabilize = !system->config.require_sleep_consolidation ||
                                    spine->consolidation_tagged;

                if (can_stabilize) {
                    synapse_state_t old_state = spine->state;
                    spine->state = SYNAPSE_STATE_STABLE;
                    update_to_stable_morphology(&spine->morphology);
                    system->total_stabilizations++;

                    /* Defer callback */
                    if (num_deferred < MAX_DEFERRED_CALLBACKS) {
                        deferred[num_deferred].event = STRUCTURAL_EVENT_STABILIZATION;
                        deferred[num_deferred].synapse_id = spine->synapse_id;
                        deferred[num_deferred].old_state = old_state;
                        deferred[num_deferred].new_state = SYNAPSE_STATE_STABLE;
                        num_deferred++;
                    }
                }
            }
        }

        /* Check for pruning */
        if (structural_plasticity_should_prune(system, spine) &&
            spine->state != SYNAPSE_STATE_PRUNING) {

            synapse_state_t old_state = spine->state;
            spine->state = SYNAPSE_STATE_PRUNING;
            spine->pruning_start_time = 0;  /* Set by caller with real time */
            system->total_pruning_starts++;

            /* Defer callback */
            if (num_deferred < MAX_DEFERRED_CALLBACKS) {
                deferred[num_deferred].event = STRUCTURAL_EVENT_PRUNING_START;
                deferred[num_deferred].synapse_id = spine->synapse_id;
                deferred[num_deferred].old_state = old_state;
                deferred[num_deferred].new_state = SYNAPSE_STATE_PRUNING;
                num_deferred++;
            }
        }

        /* Update pruning spines */
        if (spine->state == SYNAPSE_STATE_PRUNING) {
            spine->pruning_urgency += delta_sec * system->config.pruning_rate;

            /* Shrink spine */
            spine->morphology.spine_volume -=
                delta_sec * STRUCTURAL_ACTIN_SHRINK_RATE;
            if (spine->morphology.spine_volume < 0.0f) {
                spine->morphology.spine_volume = 0.0f;
            }

            /* Eliminate if urgency threshold reached */
            if (spine->pruning_urgency >= 1.0f ||
                spine->morphology.spine_volume <= 0.01f) {

                synapse_state_t old_state = spine->state;
                spine->state = SYNAPSE_STATE_ELIMINATED;
                system->total_eliminations++;

                /* Defer callback */
                if (num_deferred < MAX_DEFERRED_CALLBACKS) {
                    deferred[num_deferred].event = STRUCTURAL_EVENT_ELIMINATION;
                    deferred[num_deferred].synapse_id = spine->synapse_id;
                    deferred[num_deferred].old_state = old_state;
                    deferred[num_deferred].new_state = SYNAPSE_STATE_ELIMINATED;
                    num_deferred++;
                }
            }
        }

        /* Decay potentiation back to stable */
        if (spine->state == SYNAPSE_STATE_POTENTIATED) {
            spine->ltp_accumulator -=
                delta_sec * system->config.potentiation_decay_rate;

            if (spine->ltp_accumulator < system->config.ltp_potentiation_threshold * 0.5f) {
                /* Decay back to stable */
                synapse_state_t old_state = spine->state;
                spine->state = SYNAPSE_STATE_STABLE;
                update_to_stable_morphology(&spine->morphology);

                /* Defer callback */
                if (num_deferred < MAX_DEFERRED_CALLBACKS) {
                    deferred[num_deferred].event = STRUCTURAL_EVENT_STABILIZATION;
                    deferred[num_deferred].synapse_id = spine->synapse_id;
                    deferred[num_deferred].old_state = old_state;
                    deferred[num_deferred].new_state = SYNAPSE_STATE_STABLE;
                    num_deferred++;
                }
            }
        }
    }

    /* Copy callback data before releasing mutex */
    structural_change_callback_t callback = system->callback;
    void* callback_user_data = system->callback_user_data;

    nimcp_platform_mutex_unlock(system->mutex);

    /* Invoke deferred callbacks after releasing mutex to avoid deadlock */
    if (callback) {
        for (uint32_t i = 0; i < num_deferred; i++) {
            callback(deferred[i].event, deferred[i].synapse_id,
                    deferred[i].old_state, deferred[i].new_state,
                    callback_user_data);
        }
    }

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int structural_plasticity_get_synapse_state(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    synapse_structural_state_t* state
) {
    if (!system || !state) {
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Use find_spine_any_state to allow querying eliminated spines */
    synapse_structural_state_t* spine =
        find_spine_any_state((structural_plasticity_system_t*)system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    *state = *spine;

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

int structural_plasticity_get_morphology(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    spine_morphology_t* morphology
) {
    if (!system || !morphology) {
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Use find_spine_any_state to allow querying eliminated spines */
    synapse_structural_state_t* spine =
        find_spine_any_state((structural_plasticity_system_t*)system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    *morphology = spine->morphology;

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

uint32_t structural_plasticity_get_spine_count(
    structural_plasticity_system_t* system,
    synapse_state_t state
) {
    if (!system) return 0;

    nimcp_platform_mutex_lock(system->mutex);

    uint32_t count = 0;
    for (uint32_t i = 0; i < system->num_spines; i++) {
        if (system->spines[i].state == state) {
            count++;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return count;
}

uint32_t structural_plasticity_get_total_spines(
    structural_plasticity_system_t* system
) {
    if (!system) return 0;

    nimcp_platform_mutex_lock(system->mutex);
    uint32_t count = 0;
    for (uint32_t i = 0; i < system->num_spines; i++) {
        if (system->spines[i].state != SYNAPSE_STATE_ELIMINATED) {
            count++;
        }
    }
    nimcp_platform_mutex_unlock(system->mutex);
    return count;
}

/* ============================================================================
 * Callback Implementation
 * ============================================================================ */

int structural_plasticity_register_callback(
    structural_plasticity_system_t* system,
    structural_change_callback_t callback,
    void* user_data
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("NULL system");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);
    system->callback = callback;
    system->callback_user_data = user_data;
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* ============================================================================
 * Immune Integration Implementation
 * ============================================================================ */

int structural_plasticity_tag_complement(
    structural_plasticity_system_t* system,
    uint32_t synapse_id,
    const uint8_t* tag,
    size_t tag_len
) {
    if (!system || !tag) {
        NIMCP_LOGGING_ERROR("NULL parameter");
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine = find_spine(system, synapse_id);
    if (!spine) {
        nimcp_platform_mutex_unlock(system->mutex);
        return -1;
    }

    spine->complement_tagged = true;

    /* Copy tag (truncate if necessary) */
    size_t copy_len = tag_len < STRUCTURAL_EPITOPE_SIZE ?
                     tag_len : STRUCTURAL_EPITOPE_SIZE;
    memcpy(spine->complement_tag, tag, copy_len);
    if (copy_len < STRUCTURAL_EPITOPE_SIZE) {
        memset(spine->complement_tag + copy_len, 0,
              STRUCTURAL_EPITOPE_SIZE - copy_len);
    }

    /* Increase pruning urgency */
    spine->pruning_urgency += 0.5f * system->config.complement_sensitivity;
    if (spine->pruning_urgency > 1.0f) {
        spine->pruning_urgency = 1.0f;
    }

    nimcp_platform_mutex_unlock(system->mutex);

    NIMCP_LOGGING_DEBUG("Tagged synapse %u with complement", synapse_id);
    return 0;
}

bool structural_plasticity_is_complement_tagged(
    structural_plasticity_system_t* system,
    uint32_t synapse_id
) {
    if (!system) return false;

    nimcp_platform_mutex_lock(system->mutex);

    synapse_structural_state_t* spine =
        find_spine((structural_plasticity_system_t*)system, synapse_id);
    bool tagged = spine ? spine->complement_tagged : false;

    nimcp_platform_mutex_unlock(system->mutex);
    return tagged;
}

int structural_plasticity_get_complement_tagged(
    structural_plasticity_system_t* system,
    uint32_t* synapse_ids,
    uint32_t max_count,
    uint32_t* count
) {
    if (!system || !synapse_ids || !count) {
        return -1;
    }

    nimcp_platform_mutex_lock(system->mutex);

    *count = 0;
    for (uint32_t i = 0; i < system->num_spines && *count < max_count; i++) {
        if (system->spines[i].complement_tagged &&
            system->spines[i].state != SYNAPSE_STATE_ELIMINATED) {
            synapse_ids[(*count)++] = system->spines[i].synapse_id;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}
