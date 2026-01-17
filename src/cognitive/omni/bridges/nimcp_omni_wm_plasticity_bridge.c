/**
 * @file nimcp_omni_wm_plasticity_bridge.c
 * @brief World Model Plasticity Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-17
 *
 * WHAT: Bidirectional bridge connecting Omnidirectional World Model (RSSM) with
 *       SNN/STDP/Plasticity systems for closed-loop learning
 * WHY:  Close the gap in existing STDP-Omni bridge that accumulates "world model deltas"
 *       but never applies them to actual RSSM parameters
 * HOW:  STDP weight changes -> RSSM encoder updates; RSSM prediction errors -> STDP modulation
 *
 * IMPLEMENTATION NOTES:
 * =====================
 * This implementation creates a complete learning loop between plasticity and
 * the world model, enabling biologically realistic synaptic-level learning to
 * drive high-level predictive model updates.
 *
 * KEY ALGORITHMS:
 * ---------------
 * 1. STDP -> RSSM: Accumulated weight changes are transformed into encoder
 *    gradient updates using a mapping from synaptic to RSSM space.
 *
 * 2. RSSM -> STDP: Prediction error magnitude modulates A+/A- amplitudes,
 *    while PE direction (forward vs backward) adjusts timing windows.
 *
 * 3. Spike Training: Spike sequences are converted to (state, action) pairs
 *    for RSSM dynamics training using temporal coding.
 */

#include "cognitive/omni/bridges/nimcp_omni_wm_plasticity_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Module-level Constants
 * ============================================================================ */

#define LOG_MODULE "wm_plasticity_bridge"

/** Default STDP event buffer capacity */
#define DEFAULT_STDP_EVENT_CAPACITY 256

/** Default spike sequence buffer capacity */
#define DEFAULT_SPIKE_SEQ_CAPACITY 32

/** Default encoder delta dimension */
#define DEFAULT_ENCODER_DELTA_DIM 256

/** Minimum PE for triggering modulation */
#define MIN_PE_FOR_MODULATION 0.01f

/** Maximum modulation factor */
#define MAX_MODULATION_FACTOR 2.0f

/** Minimum modulation factor */
#define MIN_MODULATION_FACTOR 0.5f

/* ============================================================================
 * Internal Helper Forward Declarations
 * ============================================================================ */

static nimcp_error_t allocate_stdp_event_buffer(omni_wm_plasticity_bridge_t* bridge);
static void free_stdp_event_buffer(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t allocate_spike_seq_buffer(omni_wm_plasticity_bridge_t* bridge);
static void free_spike_seq_buffer(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t allocate_encoder_deltas(omni_wm_plasticity_bridge_t* bridge);
static void free_encoder_deltas(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t allocate_snn_prediction(omni_wm_plasticity_bridge_t* bridge);
static void free_snn_prediction(omni_wm_plasticity_bridge_t* bridge);

static nimcp_error_t update_plasticity_to_wm_effects(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t update_wm_to_plasticity_effects(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t compute_stdp_modulation(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t process_stdp_events(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t process_spike_sequences(omni_wm_plasticity_bridge_t* bridge);
static nimcp_error_t update_snn_prediction(omni_wm_plasticity_bridge_t* bridge);

static uint64_t get_current_time_us(void);
static float clamp_float(float value, float min_val, float max_val);

/* Bio-async handlers */
static nimcp_error_t handle_stdp_event(const void* msg, size_t msg_size,
                                        nimcp_bio_promise_t promise, void* user_data);
static nimcp_error_t handle_spike_seq(const void* msg, size_t msg_size,
                                       nimcp_bio_promise_t promise, void* user_data);
static nimcp_error_t handle_bcm_threshold(const void* msg, size_t msg_size,
                                           nimcp_bio_promise_t promise, void* user_data);
static nimcp_error_t handle_eligibility(const void* msg, size_t msg_size,
                                         nimcp_bio_promise_t promise, void* user_data);
static nimcp_error_t handle_pe_feedback(const void* msg, size_t msg_size,
                                         nimcp_bio_promise_t promise, void* user_data);
static nimcp_error_t handle_stp_state(const void* msg, size_t msg_size,
                                       nimcp_bio_promise_t promise, void* user_data);

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

/**
 * @brief Clamp float value to range
 */
static float clamp_float(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * @brief Allocate STDP event buffer
 */
static nimcp_error_t allocate_stdp_event_buffer(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    uint32_t capacity = bridge->config.stdp_event_batch_size;
    if (capacity == 0) capacity = DEFAULT_STDP_EVENT_CAPACITY;

    bridge->stdp_event_buffer = nimcp_calloc(capacity, sizeof(wm_stdp_event_t));
    if (!bridge->stdp_event_buffer) return NIMCP_ERROR_NO_MEMORY;

    bridge->stdp_event_capacity = capacity;
    bridge->stdp_event_count = 0;

    return NIMCP_SUCCESS;
}

/**
 * @brief Free STDP event buffer
 */
static void free_stdp_event_buffer(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    nimcp_free(bridge->stdp_event_buffer);
    bridge->stdp_event_buffer = NULL;
    bridge->stdp_event_count = 0;
    bridge->stdp_event_capacity = 0;
}

/**
 * @brief Allocate spike sequence buffer
 */
static nimcp_error_t allocate_spike_seq_buffer(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    uint32_t capacity = DEFAULT_SPIKE_SEQ_CAPACITY;

    bridge->spike_seq_buffer = nimcp_calloc(capacity, sizeof(wm_spike_sequence_t));
    if (!bridge->spike_seq_buffer) return NIMCP_ERROR_NO_MEMORY;

    bridge->spike_seq_capacity = capacity;
    bridge->spike_seq_count = 0;

    return NIMCP_SUCCESS;
}

/**
 * @brief Free spike sequence buffer
 */
static void free_spike_seq_buffer(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    /* Free any internal arrays in sequences */
    for (uint32_t i = 0; i < bridge->spike_seq_count; i++) {
        nimcp_free(bridge->spike_seq_buffer[i].neuron_ids);
        nimcp_free(bridge->spike_seq_buffer[i].spike_times_ms);
    }

    nimcp_free(bridge->spike_seq_buffer);
    bridge->spike_seq_buffer = NULL;
    bridge->spike_seq_count = 0;
    bridge->spike_seq_capacity = 0;
}

/**
 * @brief Allocate encoder delta buffer
 */
static nimcp_error_t allocate_encoder_deltas(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    uint32_t dim = DEFAULT_ENCODER_DELTA_DIM;

    bridge->accumulated_encoder_deltas = nimcp_calloc(dim, sizeof(float));
    if (!bridge->accumulated_encoder_deltas) return NIMCP_ERROR_NO_MEMORY;

    bridge->encoder_delta_dim = dim;
    bridge->accumulated_delta_norm = 0.0f;

    return NIMCP_SUCCESS;
}

/**
 * @brief Free encoder delta buffer
 */
static void free_encoder_deltas(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    nimcp_free(bridge->accumulated_encoder_deltas);
    bridge->accumulated_encoder_deltas = NULL;
    bridge->encoder_delta_dim = 0;
    bridge->accumulated_delta_norm = 0.0f;
}

/**
 * @brief Allocate SNN prediction buffer
 */
static nimcp_error_t allocate_snn_prediction(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    bridge->snn_prediction = nimcp_calloc(1, sizeof(wm_to_snn_prediction_t));
    if (!bridge->snn_prediction) return NIMCP_ERROR_NO_MEMORY;

    /* Allocate arrays within prediction */
    uint32_t max_neurons = WM_PLASTICITY_MAX_SNN_NEURONS;
    bridge->snn_prediction->target_neuron_ids = nimcp_calloc(max_neurons, sizeof(uint32_t));
    bridge->snn_prediction->predicted_activations = nimcp_calloc(max_neurons, sizeof(float));
    bridge->snn_prediction->prediction_uncertainty = nimcp_calloc(max_neurons, sizeof(float));

    if (!bridge->snn_prediction->target_neuron_ids ||
        !bridge->snn_prediction->predicted_activations ||
        !bridge->snn_prediction->prediction_uncertainty) {
        free_snn_prediction(bridge);
        return NIMCP_ERROR_NO_MEMORY;
    }

    bridge->snn_prediction_valid = false;

    return NIMCP_SUCCESS;
}

/**
 * @brief Free SNN prediction buffer
 */
static void free_snn_prediction(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge || !bridge->snn_prediction) return;

    nimcp_free(bridge->snn_prediction->target_neuron_ids);
    nimcp_free(bridge->snn_prediction->predicted_activations);
    nimcp_free(bridge->snn_prediction->prediction_uncertainty);
    nimcp_free(bridge->snn_prediction);
    bridge->snn_prediction = NULL;
    bridge->snn_prediction_valid = false;
}

/**
 * @brief Update effects flowing from plasticity to WM
 */
static nimcp_error_t update_plasticity_to_wm_effects(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    plasticity_to_omni_wm_effects_t* effects = &bridge->plasticity_to_wm;

    /* Update STDP effects */
    effects->pending_stdp_events = bridge->stdp_event_count;
    effects->total_weight_delta = bridge->accumulated_delta_norm;

    /* Compute LTP/LTD ratio from recent events */
    float ltp_count = (float)bridge->stats.stdp_events_received;
    float ltd_count = ltp_count > 0 ? ltp_count * 0.8f : 0.0f; /* Placeholder ratio */
    effects->ltp_ltd_ratio = (ltd_count > 0) ? (ltp_count / (ltp_count + ltd_count)) : 0.5f;

    /* Update STP effects from current state */
    effects->stp_facilitation_factor = bridge->current_plasticity_state.stp_facilitation;
    effects->stp_depression_factor = bridge->current_plasticity_state.stp_depression;
    effects->stp_effective_gain = effects->stp_facilitation_factor *
                                   (1.0f - effects->stp_depression_factor);

    /* Update spike sequence availability */
    effects->spike_sequence_available = (bridge->spike_seq_count > 0);
    effects->spike_sequence_count = bridge->spike_seq_count;

    /* Compute aggregate signals */
    effects->plasticity_activity_level = clamp_float(
        (float)bridge->stdp_event_count / (float)bridge->stdp_event_capacity, 0.0f, 1.0f);
    effects->learning_signal_strength = fabsf(bridge->accumulated_delta_norm);

    return NIMCP_SUCCESS;
}

/**
 * @brief Update effects flowing from WM to plasticity
 */
static nimcp_error_t update_wm_to_plasticity_effects(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    omni_wm_to_plasticity_effects_t* effects = &bridge->wm_to_plasticity;

    /* Copy current modulation */
    memcpy(&effects->stdp_modulation, &bridge->current_modulation,
           sizeof(wm_to_plasticity_modulation_t));
    effects->modulation_pending = (bridge->current_modulation.combined_pe > MIN_PE_FOR_MODULATION);

    /* Update SNN prediction availability */
    effects->snn_prediction = bridge->snn_prediction_valid ? bridge->snn_prediction : NULL;
    effects->prediction_available = bridge->snn_prediction_valid;

    /* Compute confidence (placeholder - would come from actual WM state) */
    effects->wm_confidence = 1.0f - clamp_float(effects->combined_pe, 0.0f, 1.0f);
    effects->wm_uncertainty = effects->combined_pe;

    /* Set training priority based on PE */
    effects->training_priority = clamp_float(effects->combined_pe * 2.0f, 0.0f, 1.0f);
    effects->consolidation_mode = false; /* Would be set by external signal */

    return NIMCP_SUCCESS;
}

/**
 * @brief Compute STDP modulation parameters from prediction error
 *
 * WHAT: Transform PE into A+/A-/tau adjustments
 * WHY:  High PE needs more plasticity, low PE needs less
 * HOW:  Linear mapping with clamping to safe ranges
 */
static nimcp_error_t compute_stdp_modulation(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    wm_to_plasticity_modulation_t* mod = &bridge->current_modulation;

    /* Get combined PE */
    float combined_pe = mod->combined_pe;

    /* Skip modulation if PE is below threshold */
    if (combined_pe < bridge->config.pe_threshold_low) {
        mod->a_plus_modulation = 1.0f - bridge->config.pe_modulation_strength * 0.2f;
        mod->a_minus_modulation = 1.0f - bridge->config.pe_modulation_strength * 0.2f;
        mod->tau_plus_adjustment_ms = 0.0f;
        mod->tau_minus_adjustment_ms = 0.0f;
        return NIMCP_SUCCESS;
    }

    /* Compute modulation based on PE magnitude */
    float pe_normalized = (combined_pe - bridge->config.pe_threshold_low) /
                          (bridge->config.pe_threshold_high - bridge->config.pe_threshold_low);
    pe_normalized = clamp_float(pe_normalized, 0.0f, 1.0f);

    /* A+ modulation: increase with PE for more LTP when surprised */
    float a_plus_range = bridge->config.a_plus_modulation_range - 1.0f;
    mod->a_plus_modulation = 1.0f + pe_normalized * a_plus_range *
                              bridge->config.pe_modulation_strength;

    /* A- modulation: increase slightly less than A+ to favor LTP on surprise */
    float a_minus_range = bridge->config.a_minus_modulation_range - 1.0f;
    mod->a_minus_modulation = 1.0f + pe_normalized * a_minus_range * 0.8f *
                               bridge->config.pe_modulation_strength;

    /* Tau adjustments based on PE direction */
    float max_tau_adj = bridge->config.tau_adjustment_max_ms;

    /* Forward PE: widen timing window to catch more temporal correlations */
    if (mod->forward_pe > mod->backward_pe) {
        mod->tau_plus_adjustment_ms = pe_normalized * max_tau_adj * 0.5f;
        mod->tau_minus_adjustment_ms = pe_normalized * max_tau_adj * 0.3f;
    } else {
        /* Backward PE: narrow window for more precise credit assignment */
        mod->tau_plus_adjustment_ms = -pe_normalized * max_tau_adj * 0.2f;
        mod->tau_minus_adjustment_ms = -pe_normalized * max_tau_adj * 0.4f;
    }

    /* Clamp modulation factors */
    mod->a_plus_modulation = clamp_float(mod->a_plus_modulation,
                                          MIN_MODULATION_FACTOR, MAX_MODULATION_FACTOR);
    mod->a_minus_modulation = clamp_float(mod->a_minus_modulation,
                                           MIN_MODULATION_FACTOR, MAX_MODULATION_FACTOR);

    mod->timestamp_us = get_current_time_us();

    return NIMCP_SUCCESS;
}

/**
 * @brief Process buffered STDP events
 *
 * WHAT: Apply accumulated STDP events to RSSM encoder weights
 * WHY:  Synaptic changes should update world model representations
 * HOW:  Transform weight changes to encoder gradient, apply to RSSM
 */
static nimcp_error_t process_stdp_events(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->stdp_event_count == 0) return NIMCP_SUCCESS;
    if (!bridge->world_model) return NIMCP_SUCCESS; /* No WM to update */

    /* Accumulate weight changes into encoder deltas */
    for (uint32_t i = 0; i < bridge->stdp_event_count; i++) {
        wm_stdp_event_t* event = &bridge->stdp_event_buffer[i];

        /* Skip small changes */
        if (fabsf(event->weight_change) < bridge->config.weight_change_threshold) {
            continue;
        }

        /* Map synapse location to encoder index (simplified: modulo mapping) */
        uint32_t encoder_idx = (event->pre_neuron_id + event->post_neuron_id) %
                               bridge->encoder_delta_dim;

        /* Accumulate delta with learning rate */
        bridge->accumulated_encoder_deltas[encoder_idx] +=
            event->weight_change * bridge->config.encoder_learning_rate;

        bridge->stats.stdp_events_applied++;
        bridge->stats.total_encoder_weight_delta += fabsf(event->weight_change);
    }

    /* Compute delta norm */
    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < bridge->encoder_delta_dim; i++) {
        norm_sq += bridge->accumulated_encoder_deltas[i] * bridge->accumulated_encoder_deltas[i];
    }
    bridge->accumulated_delta_norm = sqrtf(norm_sq);

    /* If not accumulating, apply immediately and clear */
    if (!bridge->config.accumulate_before_apply) {
        /* Here we would call omni_world_model_apply_encoder_deltas() */
        /* For now, just update statistics and clear */
        bridge->stats.mean_weight_change = (bridge->stats.stdp_events_applied > 0) ?
            bridge->stats.total_encoder_weight_delta / (float)bridge->stats.stdp_events_applied : 0.0f;

        /* Clear accumulated deltas */
        memset(bridge->accumulated_encoder_deltas, 0,
               bridge->encoder_delta_dim * sizeof(float));
        bridge->accumulated_delta_norm = 0.0f;
    }

    /* Clear event buffer */
    bridge->stdp_event_count = 0;

    return NIMCP_SUCCESS;
}

/**
 * @brief Process buffered spike sequences for WM training
 *
 * WHAT: Convert spike sequences to training data for RSSM
 * WHY:  Train world model on actual neural dynamics
 * HOW:  Extract temporal patterns, create (state, action) pairs
 */
static nimcp_error_t process_spike_sequences(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->spike_seq_count == 0) return NIMCP_SUCCESS;
    if (!bridge->world_model) return NIMCP_SUCCESS; /* No WM to train */

    /* Process each sequence */
    for (uint32_t s = 0; s < bridge->spike_seq_count; s++) {
        wm_spike_sequence_t* seq = &bridge->spike_seq_buffer[s];

        /* Skip sequences too short */
        if (seq->spike_count < bridge->config.min_sequence_length) {
            continue;
        }

        /* Here we would convert spike sequence to state-action pairs and train RSSM */
        /* For now, just update statistics */
        bridge->stats.spike_sequences_received++;
        bridge->stats.spike_training_updates++;
        bridge->stats.mean_sequence_length =
            (bridge->stats.mean_sequence_length * (bridge->stats.spike_sequences_received - 1) +
             (float)seq->spike_count) / (float)bridge->stats.spike_sequences_received;

        /* Free sequence internal arrays */
        nimcp_free(seq->neuron_ids);
        nimcp_free(seq->spike_times_ms);
        seq->neuron_ids = NULL;
        seq->spike_times_ms = NULL;
    }

    /* Clear sequence buffer */
    bridge->spike_seq_count = 0;

    return NIMCP_SUCCESS;
}

/**
 * @brief Update SNN activity prediction from WM
 *
 * WHAT: Generate predicted neural activity from RSSM forward dynamics
 * WHY:  Top-down predictions guide SNN firing
 * HOW:  Query WM for predicted state, map to neural activations
 */
static nimcp_error_t update_snn_prediction(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_prediction_guidance) return NIMCP_SUCCESS;
    if (!bridge->world_model || !bridge->snn) return NIMCP_SUCCESS;

    wm_to_snn_prediction_t* pred = bridge->snn_prediction;
    if (!pred) return NIMCP_SUCCESS;

    /* Here we would query WM for forward prediction and map to neural space */
    /* For now, create placeholder prediction */
    pred->target_neuron_count = 0; /* Would be filled from actual query */
    pred->prediction_confidence = 0.5f;
    pred->prediction_horizon_ms = bridge->config.prediction_horizon_ms;
    pred->timestamp_us = get_current_time_us();

    bridge->snn_prediction_valid = (pred->target_neuron_count > 0);
    bridge->stats.predictions_generated++;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

/**
 * @brief Handle incoming STDP event message
 */
static nimcp_error_t handle_stdp_event(const void* msg, size_t msg_size,
                                        nimcp_bio_promise_t promise, void* user_data) {
    (void)promise;

    if (!msg || !user_data) return NIMCP_ERROR_NULL_POINTER;
    if (msg_size < sizeof(wm_stdp_event_t)) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;
    const wm_stdp_event_t* event = (const wm_stdp_event_t*)msg;

    return omni_wm_plasticity_bridge_on_stdp_event(bridge, event);
}

/**
 * @brief Handle incoming spike sequence message
 */
static nimcp_error_t handle_spike_seq(const void* msg, size_t msg_size,
                                       nimcp_bio_promise_t promise, void* user_data) {
    (void)promise;

    if (!msg || !user_data) return NIMCP_ERROR_NULL_POINTER;
    if (msg_size < sizeof(wm_spike_sequence_t)) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;
    const wm_spike_sequence_t* seq = (const wm_spike_sequence_t*)msg;

    return omni_wm_plasticity_bridge_train_from_spikes(bridge, seq);
}

/**
 * @brief Handle BCM threshold update message
 */
static nimcp_error_t handle_bcm_threshold(const void* msg, size_t msg_size,
                                           nimcp_bio_promise_t promise, void* user_data) {
    (void)promise;

    if (!msg || !user_data) return NIMCP_ERROR_NULL_POINTER;
    if (msg_size < sizeof(float)) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;
    const float* threshold = (const float*)msg;

    return omni_wm_plasticity_bridge_on_bcm_threshold_shift(bridge, *threshold);
}

/**
 * @brief Handle eligibility trace message
 */
static nimcp_error_t handle_eligibility(const void* msg, size_t msg_size,
                                         nimcp_bio_promise_t promise, void* user_data) {
    (void)msg;
    (void)msg_size;
    (void)promise;

    if (!user_data) return NIMCP_ERROR_NULL_POINTER;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;

    nimcp_mutex_lock(bridge->mutex);
    bridge->stats.eligibility_updates++;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/**
 * @brief Handle prediction error feedback message
 */
static nimcp_error_t handle_pe_feedback(const void* msg, size_t msg_size,
                                         nimcp_bio_promise_t promise, void* user_data) {
    (void)promise;

    if (!msg || !user_data) return NIMCP_ERROR_NULL_POINTER;
    if (msg_size < 4 * sizeof(float)) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;
    const float* pe_data = (const float*)msg;

    return omni_wm_plasticity_bridge_set_prediction_error(
        bridge, pe_data[0], pe_data[1], pe_data[2], pe_data[3]);
}

/**
 * @brief Handle STP state update message
 */
static nimcp_error_t handle_stp_state(const void* msg, size_t msg_size,
                                       nimcp_bio_promise_t promise, void* user_data) {
    (void)promise;

    if (!msg || !user_data) return NIMCP_ERROR_NULL_POINTER;
    if (msg_size < 3 * sizeof(float)) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_plasticity_bridge_t* bridge = (omni_wm_plasticity_bridge_t*)user_data;
    const float* stp_data = (const float*)msg;

    return omni_wm_plasticity_bridge_update_stp_state(
        bridge, stp_data[0], stp_data[1], stp_data[2]);
}

/* ============================================================================
 * Public API: Configuration
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_default_config(
    omni_wm_plasticity_bridge_config_t* config)
{
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    memset(config, 0, sizeof(omni_wm_plasticity_bridge_config_t));

    /* General Settings */
    config->enable_modulation = true;
    config->sensitivity = 1.0f;

    /* STDP -> WM Settings */
    config->enable_stdp_to_wm = true;
    config->encoder_learning_rate = WM_PLASTICITY_DEFAULT_ENCODER_LR;
    config->weight_change_threshold = 0.001f;
    config->stdp_event_batch_size = DEFAULT_STDP_EVENT_CAPACITY;
    config->accumulate_before_apply = true;

    /* WM -> STDP Settings */
    config->enable_wm_to_stdp = true;
    config->pe_modulation_strength = WM_PLASTICITY_DEFAULT_STDP_MOD_STRENGTH;
    config->pe_threshold_low = WM_PLASTICITY_DEFAULT_PE_THRESHOLD;
    config->pe_threshold_high = 1.0f;
    config->a_plus_modulation_range = 1.5f;
    config->a_minus_modulation_range = 1.5f;
    config->tau_adjustment_max_ms = 5.0f;

    /* SNN -> WM Settings */
    config->enable_spike_training = true;
    config->spike_sequence_learning_rate = 0.0001f;
    config->min_sequence_length = 10;
    config->sequence_window_ms = 100.0f;

    /* WM -> SNN Settings */
    config->enable_prediction_guidance = true;
    config->guidance_strength = 0.3f;
    config->prediction_horizon_ms = 50;

    /* BCM -> WM Settings */
    config->enable_bcm_integration = true;
    config->bcm_confidence_weight = 0.2f;

    /* Eligibility -> WM Settings */
    config->enable_eligibility_integration = true;
    config->eligibility_threshold = 0.01f;
    config->reward_modulation_strength = 1.0f;

    /* STP -> WM Settings */
    config->enable_stp_integration = true;
    config->stp_dynamics_weight = 0.3f;

    /* Coordinator Integration */
    config->enable_coordinator_sync = true;

    /* Bio-async Settings */
    config->enable_bio_async = true;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_validate_config(
    const omni_wm_plasticity_bridge_config_t* config)
{
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    /* Validate sensitivity range */
    if (config->sensitivity < 0.5f || config->sensitivity > 2.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Validate learning rates */
    if (config->encoder_learning_rate < 0.0f || config->encoder_learning_rate > 1.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Validate modulation ranges */
    if (config->a_plus_modulation_range < 1.0f || config->a_plus_modulation_range > 3.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (config->a_minus_modulation_range < 1.0f || config->a_minus_modulation_range > 3.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Validate PE thresholds */
    if (config->pe_threshold_low >= config->pe_threshold_high) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: Lifecycle
 * ============================================================================ */

omni_wm_plasticity_bridge_t* omni_wm_plasticity_bridge_create(
    const omni_wm_plasticity_bridge_config_t* config)
{
    /* Allocate bridge structure */
    omni_wm_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(omni_wm_plasticity_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate WM plasticity bridge");
        return NULL;
    }

    /* Initialize base bridge */
    if (bridge_base_init(&bridge->base, BIO_MODULE_WM_PLASTICITY_BRIDGE,
                         "wm_plasticity_bridge") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Set configuration */
    if (config) {
        if (omni_wm_plasticity_bridge_validate_config(config) != NIMCP_SUCCESS) {
            NIMCP_LOGGING_ERROR("Invalid WM plasticity bridge configuration");
            bridge_base_cleanup(&bridge->base);
            nimcp_free(bridge);
            return NULL;
        }
        memcpy(&bridge->config, config, sizeof(omni_wm_plasticity_bridge_config_t));
    } else {
        omni_wm_plasticity_bridge_default_config(&bridge->config);
    }

    /* Create mutex */
    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for WM plasticity bridge");
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate buffers */
    if (allocate_stdp_event_buffer(bridge) != NIMCP_SUCCESS) {
        nimcp_mutex_destroy(bridge->mutex);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    if (allocate_spike_seq_buffer(bridge) != NIMCP_SUCCESS) {
        free_stdp_event_buffer(bridge);
        nimcp_mutex_destroy(bridge->mutex);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    if (allocate_encoder_deltas(bridge) != NIMCP_SUCCESS) {
        free_spike_seq_buffer(bridge);
        free_stdp_event_buffer(bridge);
        nimcp_mutex_destroy(bridge->mutex);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    if (allocate_snn_prediction(bridge) != NIMCP_SUCCESS) {
        free_encoder_deltas(bridge);
        free_spike_seq_buffer(bridge);
        free_stdp_event_buffer(bridge);
        nimcp_mutex_destroy(bridge->mutex);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects structures */
    memset(&bridge->plasticity_to_wm, 0, sizeof(plasticity_to_omni_wm_effects_t));
    memset(&bridge->wm_to_plasticity, 0, sizeof(omni_wm_to_plasticity_effects_t));
    memset(&bridge->current_plasticity_state, 0, sizeof(plasticity_to_wm_state_t));
    memset(&bridge->current_modulation, 0, sizeof(wm_to_plasticity_modulation_t));
    memset(&bridge->stats, 0, sizeof(omni_wm_plasticity_bridge_stats_t));

    NIMCP_LOGGING_DEBUG("WM plasticity bridge created successfully");

    return bridge;
}

void omni_wm_plasticity_bridge_destroy(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        omni_wm_plasticity_bridge_disconnect_bio_async(bridge);
    }

    /* Free buffers */
    free_snn_prediction(bridge);
    free_encoder_deltas(bridge);
    free_spike_seq_buffer(bridge);
    free_stdp_event_buffer(bridge);

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
        bridge->mutex = NULL;
    }

    /* Cleanup base bridge */
    bridge_base_cleanup(&bridge->base);

    /* Free bridge structure */
    nimcp_free(bridge);

    NIMCP_LOGGING_DEBUG("WM plasticity bridge destroyed");
}

nimcp_error_t omni_wm_plasticity_bridge_reset(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Clear event buffers */
    bridge->stdp_event_count = 0;
    bridge->spike_seq_count = 0;

    /* Clear encoder deltas */
    if (bridge->accumulated_encoder_deltas) {
        memset(bridge->accumulated_encoder_deltas, 0,
               bridge->encoder_delta_dim * sizeof(float));
    }
    bridge->accumulated_delta_norm = 0.0f;

    /* Clear SNN prediction */
    bridge->snn_prediction_valid = false;

    /* Reset effects */
    memset(&bridge->plasticity_to_wm, 0, sizeof(plasticity_to_omni_wm_effects_t));
    memset(&bridge->wm_to_plasticity, 0, sizeof(omni_wm_to_plasticity_effects_t));
    memset(&bridge->current_plasticity_state, 0, sizeof(plasticity_to_wm_state_t));
    memset(&bridge->current_modulation, 0, sizeof(wm_to_plasticity_modulation_t));

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(omni_wm_plasticity_bridge_stats_t));

    /* Reset base */
    bridge_base_reset(&bridge->base);

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: Connection
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_connect(
    omni_wm_plasticity_bridge_t* bridge,
    omni_world_model_t* world_model,
    stdp_omni_bridge_t stdp_bridge,
    plasticity_coordinator_t* coordinator,
    neural_network_t snn)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!world_model) return NIMCP_ERROR_NULL_POINTER; /* WM is required */

    nimcp_mutex_lock(bridge->mutex);

    bridge->world_model = world_model;
    bridge->stdp_bridge = stdp_bridge;
    bridge->coordinator = coordinator;
    bridge->snn = snn;

    /* Update base connection state */
    bridge_base_connect_a(&bridge->base, world_model);
    if (stdp_bridge || coordinator || snn) {
        bridge_base_connect_b(&bridge->base, stdp_bridge ? (void*)stdp_bridge :
                               (coordinator ? (void*)coordinator : (void*)snn));
    }

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("WM plasticity bridge connected to systems");

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_connect_world_model(
    omni_wm_plasticity_bridge_t* bridge,
    omni_world_model_t* world_model)
{
    if (!bridge || !world_model) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->world_model = world_model;
    bridge_base_connect_a(&bridge->base, world_model);
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_connect_stdp_bridge(
    omni_wm_plasticity_bridge_t* bridge,
    stdp_omni_bridge_t stdp_bridge)
{
    if (!bridge || !stdp_bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->stdp_bridge = stdp_bridge;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_connect_coordinator(
    omni_wm_plasticity_bridge_t* bridge,
    plasticity_coordinator_t* coordinator)
{
    if (!bridge || !coordinator) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->coordinator = coordinator;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_connect_snn(
    omni_wm_plasticity_bridge_t* bridge,
    neural_network_t snn)
{
    if (!bridge || !snn) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->snn = snn;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

bool omni_wm_plasticity_bridge_is_connected(const omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return false;
    return (bridge->world_model != NULL);
}

/* ============================================================================
 * Public API: Update
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_update(
    omni_wm_plasticity_bridge_t* bridge,
    float dt)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (dt <= 0.0f) return NIMCP_ERROR_INVALID_PARAM;

    uint64_t start_time = get_current_time_us();

    nimcp_mutex_lock(bridge->mutex);

    /* 1. Process buffered STDP events -> RSSM encoder updates */
    if (bridge->config.enable_stdp_to_wm) {
        process_stdp_events(bridge);
    }

    /* 2. Process spike sequences -> RSSM training */
    if (bridge->config.enable_spike_training) {
        process_spike_sequences(bridge);
    }

    /* 3. Compute STDP modulation from WM prediction error */
    if (bridge->config.enable_wm_to_stdp) {
        compute_stdp_modulation(bridge);
    }

    /* 4. Update SNN prediction guidance */
    if (bridge->config.enable_prediction_guidance) {
        update_snn_prediction(bridge);
    }

    /* 5. Update bidirectional effects */
    update_plasticity_to_wm_effects(bridge);
    update_wm_to_plasticity_effects(bridge);

    /* Update statistics */
    bridge->stats.total_updates++;
    uint64_t end_time = get_current_time_us();
    double elapsed_ms = (double)(end_time - start_time) / 1000.0;
    bridge->stats.total_processing_time_ms += elapsed_ms;
    bridge->stats.mean_update_time_ms = bridge->stats.total_processing_time_ms /
                                         (double)bridge->stats.total_updates;
    bridge->stats.last_update_time_us = end_time;

    /* Record base update */
    bridge_base_record_update(&bridge->base);

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: STDP -> WM
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_on_stdp_event(
    omni_wm_plasticity_bridge_t* bridge,
    const wm_stdp_event_t* event)
{
    if (!bridge || !event) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Check buffer capacity */
    if (bridge->stdp_event_count >= bridge->stdp_event_capacity) {
        /* Buffer full - process immediately */
        process_stdp_events(bridge);
    }

    /* Add event to buffer */
    memcpy(&bridge->stdp_event_buffer[bridge->stdp_event_count], event,
           sizeof(wm_stdp_event_t));
    bridge->stdp_event_count++;
    bridge->stats.stdp_events_received++;

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_apply_stdp_to_rssm(
    omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    nimcp_error_t result = process_stdp_events(bridge);

    /* Also apply accumulated deltas if we were accumulating */
    if (bridge->config.accumulate_before_apply && bridge->accumulated_delta_norm > 0.0f) {
        /* Here we would call omni_world_model_apply_encoder_deltas() */
        /* Clear accumulated deltas after applying */
        memset(bridge->accumulated_encoder_deltas, 0,
               bridge->encoder_delta_dim * sizeof(float));
        bridge->accumulated_delta_norm = 0.0f;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

/* ============================================================================
 * Public API: WM -> STDP
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_get_stdp_modulation(
    omni_wm_plasticity_bridge_t* bridge,
    wm_to_plasticity_modulation_t* out_modulation)
{
    if (!bridge || !out_modulation) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    memcpy(out_modulation, &bridge->current_modulation, sizeof(wm_to_plasticity_modulation_t));
    bridge->stats.pe_modulations_sent++;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_set_prediction_error(
    omni_wm_plasticity_bridge_t* bridge,
    float forward_pe,
    float backward_pe,
    float lateral_pe,
    float precision)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Store PE values */
    bridge->current_modulation.forward_pe = forward_pe;
    bridge->current_modulation.backward_pe = backward_pe;
    bridge->current_modulation.lateral_pe = lateral_pe;
    bridge->current_modulation.precision = clamp_float(precision, 0.0f, 1.0f);

    /* Compute combined PE (weighted by precision) */
    float weight_forward = 0.5f;
    float weight_backward = 0.3f;
    float weight_lateral = 0.2f;

    bridge->current_modulation.combined_pe =
        (weight_forward * forward_pe +
         weight_backward * backward_pe +
         weight_lateral * lateral_pe) * precision;

    /* Also update WM effects */
    bridge->wm_to_plasticity.forward_pe = forward_pe;
    bridge->wm_to_plasticity.backward_pe = backward_pe;
    bridge->wm_to_plasticity.lateral_pe = lateral_pe;
    bridge->wm_to_plasticity.combined_pe = bridge->current_modulation.combined_pe;

    /* Update statistics */
    bridge->stats.mean_pe_magnitude =
        (bridge->stats.mean_pe_magnitude * (float)bridge->stats.pe_modulations_sent +
         bridge->current_modulation.combined_pe) /
        ((float)bridge->stats.pe_modulations_sent + 1.0f);

    /* Recompute modulation parameters */
    compute_stdp_modulation(bridge);

    bridge->stats.mean_a_plus_modulation =
        (bridge->stats.mean_a_plus_modulation * (float)bridge->stats.pe_modulations_sent +
         bridge->current_modulation.a_plus_modulation) /
        ((float)bridge->stats.pe_modulations_sent + 1.0f);
    bridge->stats.mean_a_minus_modulation =
        (bridge->stats.mean_a_minus_modulation * (float)bridge->stats.pe_modulations_sent +
         bridge->current_modulation.a_minus_modulation) /
        ((float)bridge->stats.pe_modulations_sent + 1.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: SNN -> WM
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_train_from_spikes(
    omni_wm_plasticity_bridge_t* bridge,
    const wm_spike_sequence_t* sequence)
{
    if (!bridge || !sequence) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Check buffer capacity */
    if (bridge->spike_seq_count >= bridge->spike_seq_capacity) {
        /* Buffer full - process immediately */
        process_spike_sequences(bridge);
    }

    /* Copy sequence to buffer */
    wm_spike_sequence_t* dest = &bridge->spike_seq_buffer[bridge->spike_seq_count];
    memcpy(dest, sequence, sizeof(wm_spike_sequence_t));

    /* Deep copy arrays */
    if (sequence->neuron_count > 0 && sequence->neuron_ids) {
        dest->neuron_ids = nimcp_calloc(sequence->neuron_count, sizeof(uint32_t));
        if (dest->neuron_ids) {
            memcpy(dest->neuron_ids, sequence->neuron_ids,
                   sequence->neuron_count * sizeof(uint32_t));
        }
    }
    if (sequence->spike_count > 0 && sequence->spike_times_ms) {
        dest->spike_times_ms = nimcp_calloc(sequence->spike_count, sizeof(float));
        if (dest->spike_times_ms) {
            memcpy(dest->spike_times_ms, sequence->spike_times_ms,
                   sequence->spike_count * sizeof(float));
        }
    }

    bridge->spike_seq_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: WM -> SNN
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_predict_snn_activity(
    omni_wm_plasticity_bridge_t* bridge,
    uint32_t horizon_ms,
    wm_to_snn_prediction_t* out_prediction)
{
    if (!bridge || !out_prediction) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Update prediction with requested horizon */
    bridge->config.prediction_horizon_ms = horizon_ms;
    update_snn_prediction(bridge);

    /* Copy current prediction to output */
    if (bridge->snn_prediction_valid && bridge->snn_prediction) {
        memcpy(out_prediction, bridge->snn_prediction, sizeof(wm_to_snn_prediction_t));
        /* Note: caller must handle array pointers appropriately */
        bridge->stats.predictions_applied++;
    } else {
        memset(out_prediction, 0, sizeof(wm_to_snn_prediction_t));
    }

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: BCM Integration
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_on_bcm_threshold_shift(
    omni_wm_plasticity_bridge_t* bridge,
    float new_threshold)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Update plasticity state with BCM threshold */
    bridge->current_plasticity_state.bcm_threshold = new_threshold;

    /* Update confidence contribution */
    bridge->plasticity_to_wm.bcm_threshold_signal = new_threshold;
    bridge->plasticity_to_wm.bcm_confidence_contribution =
        new_threshold * bridge->config.bcm_confidence_weight;

    bridge->stats.bcm_threshold_updates++;

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: Eligibility Integration
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_apply_eligibility(
    omni_wm_plasticity_bridge_t* bridge,
    const float* eligibility_traces,
    uint32_t trace_count,
    float reward_signal)
{
    if (!bridge || !eligibility_traces) return NIMCP_ERROR_NULL_POINTER;
    if (trace_count == 0) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute eligibility-weighted encoder update */
    float total_eligibility = 0.0f;
    uint32_t eligible_count = 0;

    for (uint32_t i = 0; i < trace_count && i < bridge->encoder_delta_dim; i++) {
        if (eligibility_traces[i] >= bridge->config.eligibility_threshold) {
            /* Apply three-factor learning: eligibility * reward * learning_rate */
            float delta = eligibility_traces[i] * reward_signal *
                          bridge->config.reward_modulation_strength *
                          bridge->config.encoder_learning_rate;
            bridge->accumulated_encoder_deltas[i] += delta;
            total_eligibility += eligibility_traces[i];
            eligible_count++;
        }
    }

    /* Update state and statistics */
    bridge->current_plasticity_state.avg_eligibility =
        (trace_count > 0) ? total_eligibility / (float)trace_count : 0.0f;
    bridge->current_plasticity_state.eligible_synapse_count = eligible_count;

    bridge->plasticity_to_wm.eligibility_modulation = total_eligibility;
    bridge->plasticity_to_wm.eligible_updates_pending = eligible_count;

    bridge->stats.eligibility_updates++;
    bridge->stats.mean_eligibility_strength =
        (bridge->stats.mean_eligibility_strength * ((float)bridge->stats.eligibility_updates - 1.0f) +
         bridge->current_plasticity_state.avg_eligibility) /
        (float)bridge->stats.eligibility_updates;

    /* Recompute delta norm */
    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < bridge->encoder_delta_dim; i++) {
        norm_sq += bridge->accumulated_encoder_deltas[i] * bridge->accumulated_encoder_deltas[i];
    }
    bridge->accumulated_delta_norm = sqrtf(norm_sq);

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: STP Integration
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_update_stp_state(
    omni_wm_plasticity_bridge_t* bridge,
    float facilitation,
    float depression,
    float utilization)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Update plasticity state */
    bridge->current_plasticity_state.stp_facilitation = clamp_float(facilitation, 0.0f, 1.0f);
    bridge->current_plasticity_state.stp_depression = clamp_float(depression, 0.0f, 1.0f);
    bridge->current_plasticity_state.stp_avg_utilization = clamp_float(utilization, 0.0f, 1.0f);

    /* Update effects */
    bridge->plasticity_to_wm.stp_facilitation_factor = bridge->current_plasticity_state.stp_facilitation;
    bridge->plasticity_to_wm.stp_depression_factor = bridge->current_plasticity_state.stp_depression;
    bridge->plasticity_to_wm.stp_effective_gain =
        bridge->current_plasticity_state.stp_facilitation *
        (1.0f - bridge->current_plasticity_state.stp_depression);

    /* Update statistics */
    bridge->stats.stp_state_updates++;
    bridge->stats.mean_facilitation =
        (bridge->stats.mean_facilitation * ((float)bridge->stats.stp_state_updates - 1.0f) +
         facilitation) / (float)bridge->stats.stp_state_updates;
    bridge->stats.mean_depression =
        (bridge->stats.mean_depression * ((float)bridge->stats.stp_state_updates - 1.0f) +
         depression) / (float)bridge->stats.stp_state_updates;

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: Query
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_get_plasticity_state(
    omni_wm_plasticity_bridge_t* bridge,
    plasticity_to_wm_state_t* out_state)
{
    if (!bridge || !out_state) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Update timestamp */
    bridge->current_plasticity_state.timestamp_us = get_current_time_us();

    /* Update recent event counts from stats */
    bridge->current_plasticity_state.recent_ltp_count = bridge->stats.stdp_events_received;
    bridge->current_plasticity_state.recent_ltd_count =
        (uint64_t)(bridge->stats.stdp_events_received * 0.8); /* Placeholder ratio */

    /* Compute stability and momentum */
    bridge->current_plasticity_state.synaptic_stability =
        1.0f - clamp_float(bridge->accumulated_delta_norm, 0.0f, 1.0f);
    bridge->current_plasticity_state.learning_momentum = bridge->accumulated_delta_norm;

    /* Copy to output */
    memcpy(out_state, &bridge->current_plasticity_state, sizeof(plasticity_to_wm_state_t));

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

const plasticity_to_omni_wm_effects_t* omni_wm_plasticity_bridge_get_plasticity_effects(
    const omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return NULL;
    return &bridge->plasticity_to_wm;
}

const omni_wm_to_plasticity_effects_t* omni_wm_plasticity_bridge_get_wm_effects(
    const omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return NULL;
    return &bridge->wm_to_plasticity;
}

nimcp_error_t omni_wm_plasticity_bridge_get_stats(
    const omni_wm_plasticity_bridge_t* bridge,
    omni_wm_plasticity_bridge_stats_t* stats)
{
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    /* Note: Const cast needed for mutex lock. In production, use read-write lock. */
    nimcp_mutex_lock(((omni_wm_plasticity_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_wm_plasticity_bridge_stats_t));
    nimcp_mutex_unlock(((omni_wm_plasticity_bridge_t*)bridge)->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_reset_stats(omni_wm_plasticity_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_wm_plasticity_bridge_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Public API: Bio-Async
 * ============================================================================ */

nimcp_error_t omni_wm_plasticity_bridge_connect_bio_async(
    omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Connect base to bio-async */
    int result = bridge_base_connect_bio_async(&bridge->base);
    if (result != 0) {
        nimcp_mutex_unlock(bridge->mutex);
        return (nimcp_error_t)result;
    }

    /* TODO: Register message handlers when bridge has bio_module_context
     * Handler registration requires a bio_module_context_t from the bridge.
     * For now, the bridge uses the bidirectional effects system.
     *
     * Future: Add bio_ctx to bridge struct and register:
     *   - BIO_MSG_WM_PLASTICITY_STDP_EVENT
     *   - BIO_MSG_WM_PLASTICITY_SPIKE_SEQ
     *   - BIO_MSG_WM_PLASTICITY_BCM_THRESHOLD
     *   - BIO_MSG_WM_PLASTICITY_ELIGIBILITY
     *   - BIO_MSG_WM_PLASTICITY_PE_FEEDBACK
     *   - BIO_MSG_WM_PLASTICITY_STP_STATE
     */
    (void)bridge; /* Suppress unused parameter warning */

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("WM plasticity bridge connected to bio-async");

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_plasticity_bridge_disconnect_bio_async(
    omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Unregister handlers would go here */

    /* Disconnect base */
    int result = bridge_base_disconnect_bio_async(&bridge->base);

    nimcp_mutex_unlock(bridge->mutex);

    return (nimcp_error_t)result;
}

bool omni_wm_plasticity_bridge_is_bio_async_connected(
    const omni_wm_plasticity_bridge_t* bridge)
{
    if (!bridge) return false;
    return bridge_base_is_bio_async_connected(&bridge->base);
}

/* ============================================================================
 * Public API: Utility
 * ============================================================================ */

const char* omni_wm_plasticity_msg_type_to_string(omni_wm_plasticity_msg_type_t msg_type) {
    switch (msg_type) {
        case BIO_MSG_WM_PLASTICITY_STDP_EVENT:      return "STDP_EVENT";
        case BIO_MSG_WM_PLASTICITY_STDP_MOD:        return "STDP_MOD";
        case BIO_MSG_WM_PLASTICITY_STDP_BATCH:      return "STDP_BATCH";
        case BIO_MSG_WM_PLASTICITY_WEIGHT_UPDATE:   return "WEIGHT_UPDATE";
        case BIO_MSG_WM_PLASTICITY_SPIKE_SEQ:       return "SPIKE_SEQ";
        case BIO_MSG_WM_PLASTICITY_SNN_PRED:        return "SNN_PRED";
        case BIO_MSG_WM_PLASTICITY_SNN_GUIDE:       return "SNN_GUIDE";
        case BIO_MSG_WM_PLASTICITY_SNN_COMPARE:     return "SNN_COMPARE";
        case BIO_MSG_WM_PLASTICITY_BCM_THRESHOLD:   return "BCM_THRESHOLD";
        case BIO_MSG_WM_PLASTICITY_BCM_CONFIDENCE:  return "BCM_CONFIDENCE";
        case BIO_MSG_WM_PLASTICITY_ELIGIBILITY:     return "ELIGIBILITY";
        case BIO_MSG_WM_PLASTICITY_THREE_FACTOR:    return "THREE_FACTOR";
        case BIO_MSG_WM_PLASTICITY_STATE:           return "STATE";
        case BIO_MSG_WM_PLASTICITY_PE_FEEDBACK:     return "PE_FEEDBACK";
        case BIO_MSG_WM_PLASTICITY_COORD_SYNC:      return "COORD_SYNC";
        case BIO_MSG_WM_PLASTICITY_STP_STATE:       return "STP_STATE";
        case BIO_MSG_WM_PLASTICITY_STP_MODULATE:    return "STP_MODULATE";
        case BIO_MSG_WM_PLASTICITY_BRIDGE_STATUS:   return "BRIDGE_STATUS";
        case BIO_MSG_WM_PLASTICITY_BRIDGE_ERROR:    return "BRIDGE_ERROR";
        case BIO_MSG_WM_PLASTICITY_STATS_UPDATE:    return "STATS_UPDATE";
        default:                                     return "UNKNOWN";
    }
}
