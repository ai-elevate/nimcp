/**
 * @file nimcp_stdp.c
 * @brief Implementation of dopamine-modulated STDP
 *
 * NIMCP Phase: Option 2.1 (Integration with Phase C2.2)
 * Date: 2025-11-12
 */

#include "plasticity/stdp/nimcp_stdp.h"
#include "plasticity/stdp/nimcp_stdp_sleep_bridge.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"
#include "api/nimcp_api_exception.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_security.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/fault_tolerance/nimcp_state_manager.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define LOG_MODULE "plasticity_stdp"

/* ============================================================================
 * Health Agent Forward Declarations (Phase 8: Heartbeat for Long Operations)
 * ============================================================================ */

struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/* Global health agent for STDP operations */
static nimcp_health_agent_t* g_stdp_health_agent = NULL;

void stdp_set_health_agent(nimcp_health_agent_t* agent) {
    g_stdp_health_agent = agent;
}

static inline void stdp_heartbeat(const char* operation, float progress) {
    if (g_stdp_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_stdp_health_agent, operation, progress);
    }
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

stdp_config_t stdp_config_default(void) {
    stdp_config_t config = {
        .w_max = 1.0F,
        .learning_rate = 0.01F,
        .a_plus = 0.005F,           /* LTP amplitude (Bi & Poo, 1998) */
        .a_minus = 0.00525F,        /* LTD amplitude (slightly larger for balance) */
        .tau_plus = 0.020F,         /* 20 ms (seconds) */
        .tau_minus = 0.020F,        /* 20 ms */
        .enable_da_modulation = true,
        .da_modulation_gain = 100.0F,   /* DA concentration [0-1] → LR multiplier */
        .burst_amplification = 3.0F     /* 3x learning during bursts */
    };
    return config;
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

void stdp_synapse_init(stdp_synapse_t* synapse) {
    stdp_config_t config = stdp_config_default();
    stdp_synapse_init_with_config(synapse, &config);
}

void stdp_synapse_init_with_config(stdp_synapse_t* synapse, const stdp_config_t* config) {
    LOG_DEBUG("Initializing STDP synapse with config");

    if (!synapse || !config) {
        LOG_ERROR("NULL pointer passed to stdp_synapse_init_with_config");
        return;
    }

    memset(synapse, 0, sizeof(stdp_synapse_t));

    synapse->weight = 0.5F;  /* Initialize to 50% of max */
    synapse->w_max = config->w_max;
    synapse->w_min = 0.0F;

    synapse->learning_rate = config->learning_rate;
    synapse->a_plus = config->a_plus;
    synapse->a_minus = config->a_minus;

    /* TIMING WINDOW VALIDATION: Ensure tau values are positive and finite
     * WHAT: Validate timing constants to prevent division issues
     * WHY:  tau values are used as divisors in exponential decay (dt/tau)
     * HOW:  Clamp to reasonable biological range [1ms, 1000ms]
     * NUMERICAL STABILITY: Zero or negative tau causes division by zero/NaN
     */
    float tau_plus = config->tau_plus;
    float tau_minus = config->tau_minus;

    /* Validate tau_plus */
    if (isnan(tau_plus) || isinf(tau_plus) || tau_plus <= 0.0F) {
        LOG_WARN("Invalid tau_plus (%.4f), using default 20ms", (double)config->tau_plus);
        tau_plus = 0.020F;  /* Default 20ms */
    }
    if (tau_plus < 0.001F) tau_plus = 0.001F;  /* Min 1ms */
    if (tau_plus > 1.0F) tau_plus = 1.0F;      /* Max 1000ms */

    /* Validate tau_minus */
    if (isnan(tau_minus) || isinf(tau_minus) || tau_minus <= 0.0F) {
        LOG_WARN("Invalid tau_minus (%.4f), using default 20ms", (double)config->tau_minus);
        tau_minus = 0.020F;  /* Default 20ms */
    }
    if (tau_minus < 0.001F) tau_minus = 0.001F;  /* Min 1ms */
    if (tau_minus > 1.0F) tau_minus = 1.0F;      /* Max 1000ms */

    synapse->tau_plus = tau_plus;
    synapse->tau_minus = tau_minus;

    synapse->enable_da_modulation = config->enable_da_modulation;
    synapse->da_modulation_gain = config->da_modulation_gain;
    synapse->burst_amplification = config->burst_amplification;

    synapse->current_sleep_state = SLEEP_STATE_AWAKE;  /* Default to awake */

    /* Initialize spinlock for thread safety */
    nimcp_spinlock_init(&synapse->lock);

    synapse->pre_trace = 0.0F;
    synapse->post_trace = 0.0F;

    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0F;
    synapse->total_ltd = 0.0F;
}

/* ============================================================================
 * Trace Updates
 * ============================================================================ */

void stdp_update_traces(stdp_synapse_t* synapse, float dt) {
    /* Guard clause: NULL synapse */
    if (!synapse) return;

    /* Validate dt parameter
     * WHAT: Ensure time step is positive and finite
     * WHY:  Negative or invalid dt causes incorrect decay
     * HOW:  Skip update if dt is invalid
     */
    if (isnan(dt) || isinf(dt) || dt < 0.0F) {
        return;
    }

    /* Validate tau values before division
     * NUMERICAL STABILITY: Zero or near-zero tau causes division overflow
     */
    float tau_plus = synapse->tau_plus;
    float tau_minus = synapse->tau_minus;

    if (tau_plus <= 0.0F || isnan(tau_plus)) tau_plus = 0.020F;
    if (tau_minus <= 0.0F || isnan(tau_minus)) tau_minus = 0.020F;

    /* Exponential decay of traces with validated parameters */
    float decay_factor_plus = expf(-dt / tau_plus);
    float decay_factor_minus = expf(-dt / tau_minus);

    /* Validate decay factors (should be in (0, 1]) */
    if (isnan(decay_factor_plus) || decay_factor_plus <= 0.0F) {
        decay_factor_plus = 0.0F;  /* Full decay on invalid */
    }
    if (isnan(decay_factor_minus) || decay_factor_minus <= 0.0F) {
        decay_factor_minus = 0.0F;  /* Full decay on invalid */
    }

    synapse->pre_trace *= decay_factor_plus;
    synapse->post_trace *= decay_factor_minus;

    /* NUMERICAL STABILITY: Flush subnormal traces to zero to prevent denormal slowdown.
     * WHY:  Subnormal floats (< ~1e-38) cause 10-100x performance degradation on most CPUs.
     *       This is the same fix applied in triplet_stdp.c for consistency.
     * HOW:  Threshold at 1e-10f which is well above denormal range but small enough
     *       that trace contribution to plasticity is negligible. */
    if (synapse->pre_trace < 1e-10F) synapse->pre_trace = 0.0F;
    if (synapse->post_trace < 1e-10F) synapse->post_trace = 0.0F;
}

/* ============================================================================
 * Classic STDP (No Neuromodulation)
 * ============================================================================ */

float stdp_pre_spike(stdp_synapse_t* synapse, float current_time) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* LTD: post trace indicates recent postsynaptic spike */
    /* Apply sleep modulation: LR factor and ratio factor (ratio affects A-) */
    float modulated_a_minus = synapse->a_minus / ratio_factor;  /* Lower ratio = higher A- */
    float weight_change = -modulated_a_minus * synapse->learning_rate * lr_factor * synapse->post_trace;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(weight_change) || isinf(weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    if (weight_change < 0.0F) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(weight_change);
    }

    /* Apply weight change with saturation tracking and clamping
     * WHAT: Update weight and detect saturation
     * WHY:  Frequent saturation indicates potential learning issues
     * HOW:  Track when weights hit bounds, log periodic warnings
     */
    float new_weight = synapse->weight + weight_change;
    bool saturated = false;
    if (new_weight < synapse->w_min) {
        new_weight = synapse->w_min;
        synapse->num_saturate_min_events++;
        saturated = true;
    } else if (new_weight > synapse->w_max) {
        new_weight = synapse->w_max;
        synapse->num_saturate_max_events++;
        saturated = true;
    }
    synapse->weight = new_weight;

    /* Log warning if saturation is frequent (every 100 events) */
    if (saturated) {
        uint64_t total_sat = synapse->num_saturate_min_events + synapse->num_saturate_max_events;
        if (total_sat % 100 == 0) {
            printf("STDP weight saturation warning: min=%lu max=%lu events "
                   "(w=%.4f, w_min=%.4f, w_max=%.4f)\n",
                   (unsigned long)synapse->num_saturate_min_events,
                   (unsigned long)synapse->num_saturate_max_events,
                   synapse->weight, synapse->w_min, synapse->w_max);
        }
    }

    /* Increment presynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->pre_trace = fminf(synapse->pre_trace + 1.0F, 10.0F);

    nimcp_spinlock_unlock(&synapse->lock);

    return weight_change;
}

float stdp_post_spike(stdp_synapse_t* synapse, float current_time) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* LTP: pre trace indicates recent presynaptic spike */
    /* Apply sleep modulation: LR factor and ratio factor (ratio affects A+) */
    float modulated_a_plus = synapse->a_plus * ratio_factor;  /* Higher ratio = higher A+ */
    float weight_change = modulated_a_plus * synapse->learning_rate * lr_factor * synapse->pre_trace;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(weight_change) || isinf(weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    if (weight_change > 0.0F) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += weight_change;
    }

    /* Apply weight change with saturation tracking and clamping
     * WHAT: Update weight and detect saturation
     * WHY:  Frequent saturation indicates potential learning issues
     * HOW:  Track when weights hit bounds, log periodic warnings
     */
    float new_weight = synapse->weight + weight_change;
    bool saturated = false;
    if (new_weight < synapse->w_min) {
        new_weight = synapse->w_min;
        synapse->num_saturate_min_events++;
        saturated = true;
    } else if (new_weight > synapse->w_max) {
        new_weight = synapse->w_max;
        synapse->num_saturate_max_events++;
        saturated = true;
    }
    synapse->weight = new_weight;

    /* Log warning if saturation is frequent (every 100 events) */
    if (saturated) {
        uint64_t total_sat = synapse->num_saturate_min_events + synapse->num_saturate_max_events;
        if (total_sat % 100 == 0) {
            printf("STDP weight saturation warning: min=%lu max=%lu events "
                   "(w=%.4f, w_min=%.4f, w_max=%.4f)\n",
                   (unsigned long)synapse->num_saturate_min_events,
                   (unsigned long)synapse->num_saturate_max_events,
                   synapse->weight, synapse->w_min, synapse->w_max);
        }
    }

    /* Increment postsynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->post_trace = fminf(synapse->post_trace + 1.0F, 10.0F);

    nimcp_spinlock_unlock(&synapse->lock);

    return weight_change;
}

/* ============================================================================
 * Dopamine-Modulated STDP (THREE-FACTOR LEARNING)
 * ============================================================================ */

float stdp_get_da_modulation_factor(const stdp_synapse_t* synapse,
                                     neuromodulator_system_t neuromod) {
    if (!synapse->enable_da_modulation || neuromod == NULL) {
        return 1.0F;  /* No modulation */
    }

    /* Get dopamine concentration (tonic + phasic) */
    float da_concentration = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);

    /* Base modulation: DA concentration → LR multiplier */
    float modulation_factor = 1.0F + da_concentration * synapse->da_modulation_gain;

    /* Burst amplification: High DA concentration indicates burst */
    /* Baseline DA ≈ 0.05 in [0,1] range, burst ≈ 0.8+ */
    if (da_concentration > 0.3F) {  /* Above 30% = likely in burst */
        modulation_factor *= synapse->burst_amplification;
    }

    return modulation_factor;
}

float stdp_apply_modulated_weight_change(stdp_synapse_t* synapse,
                                         float base_weight_change,
                                         neuromodulator_system_t neuromod) {
    /* Get dopamine modulation factor */
    float modulation = stdp_get_da_modulation_factor(synapse, neuromod);

    /* Apply modulation */
    float modulated_weight_change = base_weight_change * modulation;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(modulated_weight_change) || isinf(modulated_weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    /* Update statistics */
    if (modulated_weight_change > 0.0F) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += modulated_weight_change;
    } else if (modulated_weight_change < 0.0F) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(modulated_weight_change);
    }

    /* Apply weight change with saturation tracking and clamping
     * WHAT: Update weight and detect saturation
     * WHY:  Frequent saturation indicates potential learning issues
     * HOW:  Track when weights hit bounds, log periodic warnings
     */
    float new_weight = synapse->weight + modulated_weight_change;
    bool saturated = false;
    if (new_weight < synapse->w_min) {
        new_weight = synapse->w_min;
        synapse->num_saturate_min_events++;
        saturated = true;
    } else if (new_weight > synapse->w_max) {
        new_weight = synapse->w_max;
        synapse->num_saturate_max_events++;
        saturated = true;
    }
    synapse->weight = new_weight;

    /* Log warning if saturation is frequent (every 100 events) */
    if (saturated) {
        uint64_t total_sat = synapse->num_saturate_min_events + synapse->num_saturate_max_events;
        if (total_sat % 100 == 0) {
            printf("STDP (modulated) weight saturation warning: min=%lu max=%lu events "
                   "(w=%.4f, w_min=%.4f, w_max=%.4f)\n",
                   (unsigned long)synapse->num_saturate_min_events,
                   (unsigned long)synapse->num_saturate_max_events,
                   synapse->weight, synapse->w_min, synapse->w_max);
        }
    }

    nimcp_spinlock_unlock(&synapse->lock);

    return modulated_weight_change;
}

float stdp_pre_spike_modulated(stdp_synapse_t* synapse,
                                float current_time,
                                neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* Compute base weight change (LTD from post trace) with sleep modulation */
    float modulated_a_minus = synapse->a_minus / ratio_factor;
    float base_weight_change = -modulated_a_minus * synapse->learning_rate * lr_factor * synapse->post_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment presynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->pre_trace = fminf(synapse->pre_trace + 1.0F, 10.0F);

    return modulated_weight_change;
}

float stdp_post_spike_modulated(stdp_synapse_t* synapse,
                                 float current_time,
                                 neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* Compute base weight change (LTP from pre trace) with sleep modulation */
    float modulated_a_plus = synapse->a_plus * ratio_factor;
    float base_weight_change = modulated_a_plus * synapse->learning_rate * lr_factor * synapse->pre_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment postsynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->post_trace = fminf(synapse->post_trace + 1.0F, 10.0F);

    return modulated_weight_change;
}

/* ============================================================================
 * Utilities
 * ============================================================================ */

void stdp_set_sleep_state(stdp_synapse_t* synapse, sleep_state_t state) {
    /* WHAT: Update sleep state for synapse
     * WHY:  Sleep state modulates learning rate, LTP/LTD ratio, timing windows
     * HOW:  Set current_sleep_state field, will be queried during next spike
     */
    if (!synapse) return;

    /* P2 fix: Validate sleep state bounds to prevent array index overflow
     * WHY:  Sleep bridge uses state as array index; invalid state = UB
     */
    if (state < SLEEP_STATE_AWAKE || state > SLEEP_STATE_REM) {
        LOG_WARN("Invalid sleep state %d, defaulting to AWAKE", (int)state);
        state = SLEEP_STATE_AWAKE;
    }
    synapse->current_sleep_state = state;
}

void stdp_synapse_reset(stdp_synapse_t* synapse) {
    /* Defensive null check */
    if (!synapse) return;

    /* Reset traces */
    synapse->pre_trace = 0.0F;
    synapse->post_trace = 0.0F;

    /* Reset statistics */
    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0F;
    synapse->total_ltd = 0.0F;

    /* Reset saturation tracking */
    synapse->num_saturate_max_events = 0;
    synapse->num_saturate_min_events = 0;

    /* Fix corrupted weight (NaN/Inf or out of bounds) */
    if (!isfinite(synapse->weight) ||
        synapse->weight < synapse->w_min ||
        synapse->weight > synapse->w_max) {
        synapse->weight = (synapse->w_min + synapse->w_max) / 2.0F;  /* Reset to middle */
    }
}

void stdp_synapse_print_stats(const stdp_synapse_t* synapse) {
    /* Defensive null check */
    if (!synapse) {
        printf("STDP Synapse Statistics: NULL synapse\n");
        return;
    }

    /* THREAD SAFETY: Acquire spinlock before reading synapse fields.
     * Cast away const for locking (lock is mutable state, not logical state).
     * This ensures consistent reads of weight, traces, and statistics
     * that could be modified concurrently by stdp_update/record functions. */
    stdp_synapse_t* mutable_synapse = (stdp_synapse_t*)synapse;
    nimcp_spinlock_lock(&mutable_synapse->lock);

    /* Copy all values while holding lock to ensure consistent snapshot */
    float weight = synapse->weight;
    float w_max = synapse->w_max;
    uint64_t num_pot = synapse->num_potentiation_events;
    float total_ltp = synapse->total_ltp;
    uint64_t num_dep = synapse->num_depression_events;
    float total_ltd = synapse->total_ltd;
    float pre_trace = synapse->pre_trace;
    float post_trace = synapse->post_trace;
    bool enable_da = synapse->enable_da_modulation;
    float da_gain = synapse->da_modulation_gain;
    float burst_amp = synapse->burst_amplification;
    uint64_t sat_max = synapse->num_saturate_max_events;
    uint64_t sat_min = synapse->num_saturate_min_events;

    nimcp_spinlock_unlock(&mutable_synapse->lock);

    /* Print using copied values (no locks held during I/O) */
    printf("STDP Synapse Statistics:\n");
    printf("  Weight: %.4f / %.4f\n", weight, w_max);
    printf("  Potentiation events: %lu (total LTP: %.6f)\n", (unsigned long)num_pot, total_ltp);
    printf("  Depression events: %lu (total LTD: %.6f)\n", (unsigned long)num_dep, total_ltd);
    printf("  Net change: %.6f\n", total_ltp - total_ltd);
    printf("  Weight saturation: max=%lu min=%lu\n", (unsigned long)sat_max, (unsigned long)sat_min);
    printf("  Pre trace: %.6f, Post trace: %.6f\n", pre_trace, post_trace);
    printf("  DA modulation: %s (gain: %.1f, burst amp: %.1fx)\n",
           enable_da ? "ENABLED" : "DISABLED", da_gain, burst_amp);
}

/* ============================================================================
 * KG Reader Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Allow STDP module to introspect its own capabilities and connections
 * WHY:  Self-awareness enables adaptive behavior and system introspection
 * HOW:  Query KG for STDP_Module entity and its relations
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge found, 0 if not found or error
 */
int stdp_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Query our own entity from the knowledge graph */
    const kg_entity_t* self = kg_reader_get_entity(kg, "STDP_Module");
    if (self) {
        /* Module now knows its own capabilities from KG */
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("STDP self-knowledge: %s", self->observations[i]);
        }
    }

    /* Query connections to understand integration points */
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "STDP_Module");
    if (connections) {
        LOG_DEBUG("STDP has %u outgoing connections", connections->count);
        kg_relation_list_destroy(connections);
    }

    /* Query incoming connections */
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "STDP_Module");
    if (incoming) {
        LOG_DEBUG("STDP has %u incoming connections", incoming->count);
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: State Manager Integration for Fault Tolerance
 * ============================================================================
 *
 * WHAT: Enable checkpointing and recovery for STDP synapse state
 * WHY:  Support system-wide resilience through consistent state management
 * HOW:  Implement serialize/deserialize/validate/reset/get_size operations
 */

/** Magic number for STDP state validation */
#define STDP_STATE_MAGIC 0x53544450  /* "STDP" */

/** Version for state format compatibility */
#define STDP_STATE_VERSION 1

/**
 * @brief Serialized STDP synapse state header
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t checksum;
    uint32_t reserved;
} stdp_state_header_t;

/**
 * @brief Compute simple checksum for state validation
 */
static uint32_t stdp_compute_checksum(const uint8_t* data, size_t size) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < size; i++) {
        checksum = (checksum >> 1) | (checksum << 31);
        checksum ^= data[i];
    }
    return checksum;
}

/**
 * @brief Get serialized state size for STDP synapse
 *
 * @param module_state Pointer to stdp_synapse_t
 * @return Size in bytes required for serialization
 */
size_t stdp_state_get_size(void* module_state) {
    (void)module_state;  /* Size is fixed for stdp_synapse_t */

    /* Header + core state fields (excluding spinlock) */
    size_t size = sizeof(stdp_state_header_t);

    /* Weight and bounds */
    size += sizeof(float) * 3;  /* weight, w_max, w_min */

    /* Learning parameters */
    size += sizeof(float) * 5;  /* learning_rate, a_plus, a_minus, tau_plus, tau_minus */

    /* Traces */
    size += sizeof(float) * 2;  /* pre_trace, post_trace */

    /* DA modulation */
    size += sizeof(uint8_t);    /* enable_da_modulation (as byte) */
    size += sizeof(float) * 2;  /* da_modulation_gain, burst_amplification */

    /* Sleep state */
    size += sizeof(uint32_t);   /* current_sleep_state */

    /* Statistics */
    size += sizeof(uint64_t) * 4;  /* num_potentiation_events, num_depression_events,
                                      num_saturate_max_events, num_saturate_min_events */
    size += sizeof(float) * 2;     /* total_ltp, total_ltd */

    return size;
}

/**
 * @brief Serialize STDP synapse state to buffer
 *
 * @param module_state Pointer to stdp_synapse_t
 * @param buffer Output buffer (NULL to query size only)
 * @param size In: buffer size, Out: bytes written or required
 * @return 0 on success, -1 on error, -2 if buffer too small
 */
int stdp_state_serialize(void* module_state, uint8_t* buffer, size_t* size) {
    if (!size) return -1;

    size_t required_size = stdp_state_get_size(module_state);

    /* Size query mode */
    if (!buffer) {
        *size = required_size;
        return 0;
    }

    /* Check buffer size */
    if (*size < required_size) {
        *size = required_size;
        return -2;
    }

    if (!module_state) return -1;

    stdp_synapse_t* synapse = (stdp_synapse_t*)module_state;

    /* Acquire lock for consistent read */
    nimcp_spinlock_lock(&synapse->lock);

    uint8_t* ptr = buffer;

    /* Write header (will update checksum later) */
    stdp_state_header_t header = {
        .magic = STDP_STATE_MAGIC,
        .version = STDP_STATE_VERSION,
        .checksum = 0,
        .reserved = 0
    };
    memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    /* Weight and bounds */
    memcpy(ptr, &synapse->weight, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->w_max, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->w_min, sizeof(float)); ptr += sizeof(float);

    /* Learning parameters */
    memcpy(ptr, &synapse->learning_rate, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->a_plus, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->a_minus, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->tau_plus, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->tau_minus, sizeof(float)); ptr += sizeof(float);

    /* Traces */
    memcpy(ptr, &synapse->pre_trace, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->post_trace, sizeof(float)); ptr += sizeof(float);

    /* DA modulation */
    uint8_t da_enabled = synapse->enable_da_modulation ? 1 : 0;
    memcpy(ptr, &da_enabled, sizeof(uint8_t)); ptr += sizeof(uint8_t);
    memcpy(ptr, &synapse->da_modulation_gain, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->burst_amplification, sizeof(float)); ptr += sizeof(float);

    /* Sleep state */
    uint32_t sleep_state = (uint32_t)synapse->current_sleep_state;
    memcpy(ptr, &sleep_state, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    /* Statistics */
    memcpy(ptr, &synapse->num_potentiation_events, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &synapse->num_depression_events, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &synapse->total_ltp, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->total_ltd, sizeof(float)); ptr += sizeof(float);
    memcpy(ptr, &synapse->num_saturate_max_events, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &synapse->num_saturate_min_events, sizeof(uint64_t)); ptr += sizeof(uint64_t);

    nimcp_spinlock_unlock(&synapse->lock);

    /* Compute and update checksum (over data after header) */
    uint32_t checksum = stdp_compute_checksum(
        buffer + sizeof(stdp_state_header_t),
        required_size - sizeof(stdp_state_header_t)
    );
    memcpy(buffer + offsetof(stdp_state_header_t, checksum), &checksum, sizeof(uint32_t));

    *size = required_size;
    LOG_DEBUG("STDP state serialized: %zu bytes", required_size);
    return 0;
}

/**
 * @brief Deserialize STDP synapse state from buffer
 *
 * @param module_state Pointer to stdp_synapse_t
 * @param buffer Input buffer containing serialized state
 * @param size Size of input buffer
 * @return 0 on success, negative on error
 */
int stdp_state_deserialize(void* module_state, const uint8_t* buffer, size_t size) {
    if (!module_state || !buffer) return -1;

    size_t required_size = stdp_state_get_size(module_state);
    if (size < required_size) {
        LOG_ERROR("STDP deserialize: buffer too small (%zu < %zu)", size, required_size);
        return -1;
    }

    stdp_synapse_t* synapse = (stdp_synapse_t*)module_state;
    const uint8_t* ptr = buffer;

    /* Read and validate header */
    stdp_state_header_t header;
    memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    if (header.magic != STDP_STATE_MAGIC) {
        LOG_ERROR("STDP deserialize: invalid magic (0x%08X)", header.magic);
        return -1;
    }

    if (header.version != STDP_STATE_VERSION) {
        LOG_ERROR("STDP deserialize: version mismatch (%u != %u)", header.version, STDP_STATE_VERSION);
        return -1;
    }

    /* Verify checksum */
    uint32_t computed_checksum = stdp_compute_checksum(
        buffer + sizeof(stdp_state_header_t),
        required_size - sizeof(stdp_state_header_t)
    );
    if (computed_checksum != header.checksum) {
        LOG_ERROR("STDP deserialize: checksum mismatch (0x%08X != 0x%08X)",
                  computed_checksum, header.checksum);
        return -1;
    }

    /* Acquire lock for consistent write */
    nimcp_spinlock_lock(&synapse->lock);

    /* Weight and bounds */
    memcpy(&synapse->weight, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->w_max, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->w_min, ptr, sizeof(float)); ptr += sizeof(float);

    /* Learning parameters */
    memcpy(&synapse->learning_rate, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->a_plus, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->a_minus, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->tau_plus, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->tau_minus, ptr, sizeof(float)); ptr += sizeof(float);

    /* Traces */
    memcpy(&synapse->pre_trace, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->post_trace, ptr, sizeof(float)); ptr += sizeof(float);

    /* DA modulation */
    uint8_t da_enabled;
    memcpy(&da_enabled, ptr, sizeof(uint8_t)); ptr += sizeof(uint8_t);
    synapse->enable_da_modulation = (da_enabled != 0);
    memcpy(&synapse->da_modulation_gain, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->burst_amplification, ptr, sizeof(float)); ptr += sizeof(float);

    /* Sleep state */
    uint32_t sleep_state;
    memcpy(&sleep_state, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    synapse->current_sleep_state = (sleep_state_t)sleep_state;

    /* Statistics */
    memcpy(&synapse->num_potentiation_events, ptr, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(&synapse->num_depression_events, ptr, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(&synapse->total_ltp, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->total_ltd, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&synapse->num_saturate_max_events, ptr, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(&synapse->num_saturate_min_events, ptr, sizeof(uint64_t)); ptr += sizeof(uint64_t);

    nimcp_spinlock_unlock(&synapse->lock);

    LOG_DEBUG("STDP state deserialized successfully");
    return 0;
}

/**
 * @brief Validate STDP synapse state integrity
 *
 * @param module_state Pointer to stdp_synapse_t
 * @return 0 if valid, negative error code if invalid
 */
int stdp_state_validate(void* module_state) {
    if (!module_state) return -1;

    stdp_synapse_t* synapse = (stdp_synapse_t*)module_state;

    nimcp_spinlock_lock(&synapse->lock);

    int result = 0;

    /* Validate weight is finite and within bounds */
    if (!isfinite(synapse->weight) ||
        synapse->weight < synapse->w_min || synapse->weight > synapse->w_max) {
        LOG_WARN("STDP validate: weight out of bounds (%.4f not in [%.4f, %.4f])",
                 synapse->weight, synapse->w_min, synapse->w_max);
        result = -1;
    }

    /* Validate learning parameters are positive */
    if (synapse->learning_rate < 0.0F || synapse->a_plus < 0.0F ||
        synapse->a_minus < 0.0F || synapse->tau_plus <= 0.0F ||
        synapse->tau_minus <= 0.0F) {
        LOG_WARN("STDP validate: invalid learning parameters");
        result = -2;
    }

    /* Validate traces are finite */
    if (isnan(synapse->pre_trace) || isinf(synapse->pre_trace) ||
        isnan(synapse->post_trace) || isinf(synapse->post_trace)) {
        LOG_WARN("STDP validate: invalid traces (NaN/Inf detected)");
        result = -3;
    }

    /* Validate DA modulation parameters */
    if (synapse->da_modulation_gain < 0.0F || synapse->burst_amplification < 0.0F) {
        LOG_WARN("STDP validate: invalid DA modulation parameters");
        result = -4;
    }

    /* Validate sleep state enum */
    if (synapse->current_sleep_state < SLEEP_STATE_AWAKE ||
        synapse->current_sleep_state > SLEEP_STATE_REM) {
        LOG_WARN("STDP validate: invalid sleep state (%d)", (int)synapse->current_sleep_state);
        result = -5;
    }

    nimcp_spinlock_unlock(&synapse->lock);

    if (result == 0) {
        LOG_DEBUG("STDP state validation passed");
    }

    return result;
}

/**
 * @brief Reset STDP synapse state to defaults
 *
 * @param module_state Pointer to stdp_synapse_t
 * @return 0 on success, negative on error
 */
int stdp_state_reset(void* module_state) {
    if (!module_state) return -1;

    stdp_synapse_t* synapse = (stdp_synapse_t*)module_state;

    /* Use existing reset function which handles locking */
    stdp_synapse_reset(synapse);

    LOG_DEBUG("STDP state reset to defaults");
    return 0;
}

/**
 * @brief Get STDP state operations for state manager registration
 *
 * @return Pointer to static state ops structure
 */
const nimcp_module_state_ops_t* stdp_get_state_ops(void) {
    static nimcp_module_state_ops_t ops = {
        .serialize = stdp_state_serialize,
        .deserialize = stdp_state_deserialize,
        .validate = stdp_state_validate,
        .reset = stdp_state_reset,
        .get_size = stdp_state_get_size
    };
    return &ops;
}
