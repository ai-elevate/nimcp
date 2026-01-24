/**
 * @file nimcp_mirror_visual_bridge.c
 * @brief Mirror Neurons - Visual Cortex Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-01-05
 *
 * WHAT: Implements bidirectional integration between mirror neurons and visual cortex
 * WHY:  Mirror neurons need visual input for action observation; visual cortex needs
 *       social salience feedback for attention prioritization
 * HOW:  Models STS pathway connecting visual processing to social cognition
 */

#include "cognitive/mirror_neurons/nimcp_mirror_visual_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define VISUAL_FEATURES_BUFFER_SIZE  256
#define BIO_MOTION_COHERENCE_WEIGHT  0.6f
#define FACE_EXPRESSION_WEIGHT       0.7f
#define STS_INTEGRATION_DECAY        0.95f

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp float value to range
 */
static inline float clamp_f(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * @brief Compute sigmoid function
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Check if visual features indicate agent presence (heuristic)
 */
static bool detect_agent_from_features(
    const float* features,
    uint32_t num_features,
    float threshold,
    float* confidence
) {
    if (!features || num_features == 0) {
        if (confidence) *confidence = 0.0f;
        return false;
    }

    /* Simple heuristic: look for high-variance regions typical of agents */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < num_features && i < 64; i++) {
        sum += features[i];
        sum_sq += features[i] * features[i];
    }
    uint32_t n = (num_features < 64) ? num_features : 64;
    float mean = sum / (float)n;
    float variance = (sum_sq / (float)n) - (mean * mean);

    /* High variance suggests complex object (possibly agent) */
    float agent_score = sigmoid(variance * 2.0f - 0.5f);
    if (confidence) *confidence = agent_score;
    return agent_score >= threshold;
}

/**
 * @brief Analyze motion vectors for biological motion signatures
 */
static void analyze_bio_motion(
    const float* motion_vectors,
    uint32_t num_vectors,
    bio_motion_analysis_t* analysis
) {
    if (!motion_vectors || num_vectors == 0 || !analysis) return;

    memset(analysis, 0, sizeof(bio_motion_analysis_t));

    /* Compute motion statistics */
    float dx_sum = 0.0f, dy_sum = 0.0f;
    float magnitude_sum = 0.0f;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i + 1 < num_vectors; i += 2) {
        float dx = motion_vectors[i];
        float dy = motion_vectors[i + 1];
        float mag = sqrtf(dx * dx + dy * dy);
        if (mag > 0.01f) {
            dx_sum += dx;
            dy_sum += dy;
            magnitude_sum += mag;
            valid_count++;
        }
    }

    if (valid_count == 0) return;

    float mean_dx = dx_sum / (float)valid_count;
    float mean_dy = dy_sum / (float)valid_count;
    float mean_mag = magnitude_sum / (float)valid_count;

    /* Coherence: how aligned are the motion vectors */
    float coherence = 0.0f;
    for (uint32_t i = 0; i + 1 < num_vectors; i += 2) {
        float dx = motion_vectors[i];
        float dy = motion_vectors[i + 1];
        float mag = sqrtf(dx * dx + dy * dy);
        if (mag > 0.01f) {
            float dot = (dx * mean_dx + dy * mean_dy) / (mag * sqrtf(mean_dx * mean_dx + mean_dy * mean_dy) + 1e-6f);
            coherence += (dot + 1.0f) / 2.0f;
        }
    }
    coherence /= (float)valid_count;

    /* Biological motion tends to have moderate coherence (not too uniform) */
    float bio_score = 1.0f - fabsf(coherence - 0.5f) * 2.0f;
    bio_score *= sigmoid(mean_mag * 5.0f);

    analysis->bio_motion_present = bio_score > MIRROR_VISUAL_BIO_MOTION_THRESHOLD;
    analysis->bio_motion_score = bio_score;
    analysis->motion_direction = atan2f(mean_dy, mean_dx);
    analysis->motion_speed = mean_mag;
    analysis->coherence = coherence;
}

/**
 * @brief Extract face-like features from visual features (simple heuristic)
 */
static void extract_face_features(
    const float* features,
    uint32_t num_features,
    float threshold,
    face_expression_t* face
) {
    if (!features || num_features == 0 || !face) return;

    memset(face, 0, sizeof(face_expression_t));

    /* Simple heuristic: look for symmetric high-contrast regions */
    float symmetry_score = 0.0f;
    uint32_t half = num_features / 2;
    for (uint32_t i = 0; i < half && i < 32; i++) {
        float diff = fabsf(features[i] - features[half + i]);
        symmetry_score += 1.0f - diff;
    }
    symmetry_score /= (half < 32 ? half : 32);

    /* High symmetry suggests face */
    float face_score = symmetry_score * sigmoid(features[0] * 2.0f);

    face->face_detected = face_score >= threshold;
    face->face_confidence = face_score;
    face->face_x = 0.5f;  /* Default center */
    face->face_y = 0.5f;
    face->expression_valence = 0.0f;  /* Neutral */
    face->expression_arousal = 0.5f;
    face->gaze_at_observer = false;
}

/**
 * @brief Compute social salience from mirror neuron state
 */
static void compute_social_salience_unlocked(
    mirror_visual_bridge_t* bridge,
    social_salience_t* salience
) {
    if (!bridge || !salience) return;

    memset(salience, 0, sizeof(social_salience_t));

    /* Get mirror neuron social salience */
    float mirror_salience = 0.0f;
    if (bridge->mirror_neurons) {
        mirror_salience = mirror_neurons_get_social_salience(bridge->mirror_neurons);
    }

    /* Get observation activity */
    float observation_activity = 0.0f;
    if (bridge->mirror_neurons && mirror_neurons_has_recent_observations(bridge->mirror_neurons)) {
        observation_activity = 0.7f;
    }

    /* Combine into overall salience */
    salience->observation_activity = observation_activity;
    salience->empathic_resonance = bridge->state.last_face_detection.face_detected ?
                                   bridge->state.last_face_detection.face_confidence * 0.5f : 0.0f;
    salience->action_prediction_strength = bridge->effects.action_recognition_input;
    salience->overall_salience = clamp_f(
        mirror_salience * 0.4f +
        observation_activity * 0.3f +
        salience->empathic_resonance * 0.2f +
        salience->action_prediction_strength * 0.1f,
        0.0f, 1.0f
    );

    /* Compute attention boost factor */
    salience->attention_boost_factor = MIRROR_VISUAL_SALIENCE_BOOST_MIN +
        salience->overall_salience * (MIRROR_VISUAL_SALIENCE_BOOST_MAX - MIRROR_VISUAL_SALIENCE_BOOST_MIN);

    /* Set focus point based on agent detection */
    if (bridge->state.last_agent_detection.agent_detected) {
        salience->focus_x = bridge->state.last_agent_detection.position_x;
        salience->focus_y = bridge->state.last_agent_detection.position_y;
    } else {
        salience->focus_x = 0.5f;
        salience->focus_y = 0.5f;
    }
}

/**
 * @brief Update STS integration state (unlocked version)
 */
static void update_sts_state_unlocked(mirror_visual_bridge_t* bridge) {
    if (!bridge) return;

    sts_state_t* sts = &bridge->effects.sts_state;

    /* Decay previous state */
    sts->integration_level *= STS_INTEGRATION_DECAY;
    sts->visual_input_strength *= STS_INTEGRATION_DECAY;
    sts->mirror_input_strength *= STS_INTEGRATION_DECAY;

    /* Add visual inputs */
    if (bridge->state.last_agent_detection.agent_detected) {
        sts->visual_input_strength += bridge->state.last_agent_detection.confidence * 0.3f;
        sts->agent_representation += bridge->state.last_agent_detection.confidence * 0.2f;
    }
    if (bridge->state.last_bio_motion.bio_motion_present) {
        sts->visual_input_strength += bridge->state.last_bio_motion.bio_motion_score * 0.3f;
        sts->action_understanding += bridge->state.last_bio_motion.action_confidence * 0.2f;
    }
    if (bridge->state.last_face_detection.face_detected) {
        sts->visual_input_strength += bridge->state.last_face_detection.face_confidence * 0.2f;
        sts->social_context += bridge->state.last_face_detection.face_confidence * 0.2f;
    }

    /* Add mirror neuron inputs */
    sts->mirror_input_strength += bridge->state.current_salience.overall_salience * 0.4f;

    /* Clamp values */
    sts->visual_input_strength = clamp_f(sts->visual_input_strength, 0.0f, 1.0f);
    sts->mirror_input_strength = clamp_f(sts->mirror_input_strength, 0.0f, 1.0f);
    sts->agent_representation = clamp_f(sts->agent_representation, 0.0f, 1.0f);
    sts->action_understanding = clamp_f(sts->action_understanding, 0.0f, 1.0f);
    sts->social_context = clamp_f(sts->social_context, 0.0f, 1.0f);

    /* Compute integration level */
    sts->integration_level = (sts->visual_input_strength + sts->mirror_input_strength) / 2.0f;
}

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int mirror_visual_bridge_default_config(mirror_visual_config_t* config) {
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;

    }

    memset(config, 0, sizeof(mirror_visual_config_t));

    /* Agent detection */
    config->agent_detection_threshold = MIRROR_VISUAL_AGENT_CONFIDENCE_MIN;
    config->agent_attention_boost = 1.5f;
    config->enable_agent_detection = true;

    /* Biological motion */
    config->bio_motion_threshold = MIRROR_VISUAL_BIO_MOTION_THRESHOLD;
    config->bio_motion_sensitivity = 1.0f;
    config->enable_bio_motion = true;

    /* Face/expression processing */
    config->face_detection_threshold = MIRROR_VISUAL_FACE_CONFIDENCE_MIN;
    config->empathy_gain = 1.0f;
    config->enable_face_processing = true;

    /* Social salience feedback */
    config->salience_feedback_gain = 1.0f;
    config->attention_modulation_rate = 0.1f;
    config->enable_salience_feedback = true;

    /* STS pathway modeling */
    config->sts_integration_weight = 0.5f;
    config->sts_latency_ms = MIRROR_VISUAL_STS_LATENCY_MS;
    config->enable_sts_modeling = true;

    /* Action prediction */
    config->prediction_visual_gain = 1.2f;
    config->enable_action_prediction = true;

    return 0;
}

mirror_visual_bridge_t* mirror_visual_bridge_create(
    const mirror_visual_config_t* config
) {
    mirror_visual_bridge_t* bridge = nimcp_calloc(1, sizeof(mirror_visual_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate mirror-visual bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    /* Initialize base bridge infrastructure */
    if (bridge_base_init(&bridge->base, BIO_MODULE_MIRROR_VISUAL_BRIDGE, "mirror_visual_bridge") != 0) {
        nimcp_free(bridge);
        NIMCP_LOGGING_ERROR("Failed to initialize bridge base");
        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        mirror_visual_bridge_default_config(&bridge->config);
    }

    /* Allocate visual features buffer */
    bridge->visual_features_buffer = nimcp_calloc(VISUAL_FEATURES_BUFFER_SIZE, sizeof(float));
    if (!bridge->visual_features_buffer) {
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        NIMCP_LOGGING_ERROR("Failed to allocate visual features buffer");
        return NULL;
    }
    bridge->visual_features_size = VISUAL_FEATURES_BUFFER_SIZE;

    /* Initialize state */
    memset(&bridge->effects, 0, sizeof(mirror_visual_effects_t));
    memset(&bridge->state, 0, sizeof(mirror_visual_state_t));
    memset(&bridge->stats, 0, sizeof(mirror_visual_stats_t));

    NIMCP_LOGGING_INFO("Mirror-visual bridge created successfully");
    return bridge;
}

void mirror_visual_bridge_destroy(mirror_visual_bridge_t* bridge) {
    if (!bridge) return;

    NIMCP_LOGGING_DEBUG("Destroying mirror-visual bridge");

    /* Free visual features buffer */
    if (bridge->visual_features_buffer) {
        nimcp_free(bridge->visual_features_buffer);
        bridge->visual_features_buffer = NULL;
    }

    /* Cleanup base infrastructure */
    bridge_base_cleanup(&bridge->base);

    /* Free bridge */
    nimcp_free(bridge);
}

int mirror_visual_bridge_reset(mirror_visual_bridge_t* bridge) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset state and effects, preserve connections */
    memset(&bridge->effects, 0, sizeof(mirror_visual_effects_t));
    memset(&bridge->state, 0, sizeof(mirror_visual_state_t));
    memset(&bridge->stats, 0, sizeof(mirror_visual_stats_t));

    /* Reset base stats */
    bridge_base_reset(&bridge->base);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Mirror-visual bridge reset");
    return 0;
}

/* ============================================================================
 * Connection API Implementation
 * ============================================================================ */

int mirror_visual_bridge_connect_mirror_neurons(
    mirror_visual_bridge_t* bridge,
    mirror_neurons_t mirror_neurons
) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!mirror_neurons) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "mirror_neurons is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->mirror_neurons = mirror_neurons;
    bridge->base.system_a = mirror_neurons;
    bridge->base.system_a_connected = true;
    bridge->base.bridge_active = bridge->base.system_a_connected && bridge->base.system_b_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Mirror neurons connected to bridge");
    return 0;
}

int mirror_visual_bridge_connect_visual_cortex(
    mirror_visual_bridge_t* bridge,
    visual_cortex_t* visual_cortex
) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!visual_cortex) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "visual_cortex is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->visual_cortex = visual_cortex;
    bridge->base.system_b = visual_cortex;
    bridge->base.system_b_connected = true;
    bridge->base.bridge_active = bridge->base.system_a_connected && bridge->base.system_b_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Visual cortex connected to bridge");
    return 0;
}

bool mirror_visual_bridge_is_connected(const mirror_visual_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->base.bridge_active;
}

/* ============================================================================
 * Visual -> Mirror Neurons Direction Implementation
 * ============================================================================ */

int mirror_visual_process_agent_detection(
    mirror_visual_bridge_t* bridge,
    const float* features,
    uint32_t num_features,
    agent_detection_t* detection
) {
    if (!bridge || !features || !detection) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Detect agent from features */
    float confidence = 0.0f;
    bool detected = detect_agent_from_features(
        features,
        num_features,
        bridge->config.agent_detection_threshold,
        &confidence
    );

    /* Fill detection result */
    detection->agent_detected = detected;
    detection->confidence = confidence;
    detection->position_x = 0.5f;  /* Default center, would need localization */
    detection->position_y = 0.5f;
    detection->bounding_size = 0.3f;
    detection->timestamp_us = nimcp_time_monotonic_us();

    /* Update state */
    bridge->state.last_agent_detection = *detection;

    /* Update statistics */
    if (detected) {
        bridge->stats.agent_detections++;
        bridge->stats.avg_agent_confidence =
            (bridge->stats.avg_agent_confidence * (bridge->stats.agent_detections - 1) + confidence) /
            bridge->stats.agent_detections;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_process_bio_motion(
    mirror_visual_bridge_t* bridge,
    const float* motion_vectors,
    uint32_t num_vectors,
    bio_motion_analysis_t* analysis
) {
    if (!bridge || !motion_vectors || !analysis) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Analyze biological motion */
    analyze_bio_motion(motion_vectors, num_vectors, analysis);

    /* Update state */
    bridge->state.last_bio_motion = *analysis;

    /* Update effects */
    if (analysis->bio_motion_present) {
        bridge->effects.action_recognition_input = analysis->bio_motion_score *
            bridge->config.bio_motion_sensitivity;
    }

    /* Update statistics */
    if (analysis->bio_motion_present) {
        bridge->stats.bio_motion_detections++;
        bridge->stats.avg_bio_motion_score =
            (bridge->stats.avg_bio_motion_score * (bridge->stats.bio_motion_detections - 1) +
             analysis->bio_motion_score) / bridge->stats.bio_motion_detections;

        if (analysis->detected_action_id > 0) {
            bridge->stats.action_recognitions++;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_process_face(
    mirror_visual_bridge_t* bridge,
    const float* features,
    uint32_t num_features,
    face_expression_t* face
) {
    if (!bridge || !features || !face) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Extract face features */
    extract_face_features(
        features,
        num_features,
        bridge->config.face_detection_threshold,
        face
    );

    /* Update state */
    bridge->state.last_face_detection = *face;

    /* Update effects */
    if (face->face_detected) {
        bridge->effects.empathy_input = face->face_confidence *
            bridge->config.empathy_gain;
    }

    /* Update statistics */
    if (face->face_detected) {
        bridge->stats.face_detections++;
        if (face->face_confidence >= MIRROR_VISUAL_EMPATHY_THRESHOLD) {
            bridge->stats.empathy_activations++;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_trigger_observation_mode(
    mirror_visual_bridge_t* bridge,
    const agent_detection_t* detection
) {
    if (!bridge || !detection) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Only trigger if agent detected with sufficient confidence */
    if (!detection->agent_detected ||
        detection->confidence < bridge->config.agent_detection_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    /* Activate observation mode in mirror neurons */
    if (bridge->mirror_neurons) {
        mirror_neurons_activate_observation_mode(bridge->mirror_neurons);
        bridge->effects.observation_mode_triggered = true;
        bridge->stats.observation_mode_triggers++;
        NIMCP_LOGGING_DEBUG("Observation mode triggered (confidence=%.2f)", detection->confidence);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Mirror Neurons -> Visual Cortex Direction Implementation
 * ============================================================================ */

int mirror_visual_get_social_salience(
    mirror_visual_bridge_t* bridge,
    social_salience_t* salience
) {
    if (!bridge || !salience) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    compute_social_salience_unlocked(bridge, salience);
    bridge->state.current_salience = *salience;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mirror_visual_apply_attention_modulation(
    mirror_visual_bridge_t* bridge,
    const social_salience_t* salience
) {
    if (!bridge || !salience) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply attention boost to visual cortex */
    if (bridge->visual_cortex && salience->overall_salience > 0.1f) {
        visual_cortex_boost_region_attention(
            bridge->visual_cortex,
            salience->focus_x,
            salience->focus_y,
            salience->attention_boost_factor
        );

        bridge->effects.attention_boost_applied = salience->attention_boost_factor;
        bridge->stats.attention_boosts++;
        bridge->stats.avg_attention_boost =
            (bridge->stats.avg_attention_boost * (bridge->stats.attention_boosts - 1) +
             salience->attention_boost_factor) / bridge->stats.attention_boosts;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_apply_action_prediction(
    mirror_visual_bridge_t* bridge,
    uint32_t predicted_action_id,
    float prediction_confidence
) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply motion priming based on predicted action */
    if (bridge->visual_cortex && prediction_confidence > 0.3f) {
        /* In a full implementation, this would prime V5/MT for expected motion */
        bridge->effects.motion_priming_gain = 1.0f +
            (prediction_confidence * (bridge->config.prediction_visual_gain - 1.0f));
        bridge->stats.motion_priming_events++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_boost_face_attention(
    mirror_visual_bridge_t* bridge,
    float empathy_level
) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Boost face region attention based on empathy level */
    if (bridge->visual_cortex && empathy_level > MIRROR_VISUAL_EMPATHY_THRESHOLD) {
        float gain = 1.0f + empathy_level * 0.5f;
        bridge->effects.face_attention_gain = gain;
        bridge->stats.face_attention_events++;

        /* Boost attention at face location if known */
        if (bridge->state.last_face_detection.face_detected) {
            visual_cortex_boost_region_attention(
                bridge->visual_cortex,
                bridge->state.last_face_detection.face_x,
                bridge->state.last_face_detection.face_y,
                gain
            );
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * STS Pathway Modeling Implementation
 * ============================================================================ */

int mirror_visual_update_sts(mirror_visual_bridge_t* bridge) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    update_sts_state_unlocked(bridge);

    /* Update statistics */
    bridge->stats.avg_sts_integration =
        (bridge->stats.avg_sts_integration * bridge->stats.total_updates +
         bridge->effects.sts_state.integration_level) /
        (bridge->stats.total_updates + 1);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_visual_get_sts_state(
    const mirror_visual_bridge_t* bridge,
    sts_state_t* sts_state
) {
    if (!bridge || !sts_state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *sts_state = bridge->effects.sts_state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Update Cycle Implementation
 * ============================================================================ */

int mirror_visual_bridge_update(
    mirror_visual_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    (void)delta_ms;  /* Used for time-dependent effects */

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if both systems connected */
    if (!bridge->base.bridge_active) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    uint64_t start_time = nimcp_time_monotonic_us();

    /* 1. Process visual -> mirror direction */
    if (bridge->config.enable_agent_detection && bridge->visual_cortex) {
        /* In full implementation, get features from visual cortex */
        agent_detection_t detection;
        memset(&detection, 0, sizeof(detection));

        /* Check for agent using visual cortex API */
        if (bridge->visual_features_buffer && bridge->visual_features_size > 0) {
            bool detected = visual_cortex_detect_agent(
                bridge->visual_cortex,
                bridge->visual_features_buffer,
                bridge->visual_features_size
            );
            if (detected) {
                detection.agent_detected = true;
                detection.confidence = 0.7f;
                detection.position_x = 0.5f;
                detection.position_y = 0.5f;
                detection.timestamp_us = start_time;

                bridge->state.last_agent_detection = detection;

                /* Trigger observation mode */
                if (bridge->mirror_neurons) {
                    mirror_visual_trigger_observation_mode(bridge, &detection);
                }
            }
        }
    }

    /* 2. Update STS integration */
    if (bridge->config.enable_sts_modeling) {
        update_sts_state_unlocked(bridge);
    }

    /* 3. Process mirror -> visual direction */
    if (bridge->config.enable_salience_feedback) {
        social_salience_t salience;
        compute_social_salience_unlocked(bridge, &salience);
        bridge->state.current_salience = salience;

        /* Apply attention modulation */
        if (bridge->visual_cortex && salience.overall_salience > 0.1f) {
            visual_cortex_boost_region_attention(
                bridge->visual_cortex,
                salience.focus_x,
                salience.focus_y,
                salience.attention_boost_factor
            );
            bridge->effects.attention_boost_applied = salience.attention_boost_factor;
        }
    }

    /* 4. Update timing and statistics */
    uint64_t end_time = nimcp_time_monotonic_us();
    float processing_time_ms = (float)(end_time - start_time) / 1000.0f;

    bridge->stats.total_updates++;
    bridge->stats.avg_processing_time_ms =
        (bridge->stats.avg_processing_time_ms * (bridge->stats.total_updates - 1) +
         processing_time_ms) / bridge->stats.total_updates;
    bridge->stats.avg_salience_level =
        (bridge->stats.avg_salience_level * (bridge->stats.total_updates - 1) +
         bridge->state.current_salience.overall_salience) / bridge->stats.total_updates;

    bridge->state.frames_processed++;
    bridge->state.systems_synchronized = true;

    /* Record update in base */
    bridge_base_record_update(&bridge->base);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * State/Stats API Implementation
 * ============================================================================ */

int mirror_visual_bridge_get_state(
    const mirror_visual_bridge_t* bridge,
    mirror_visual_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mirror_visual_bridge_get_effects(
    const mirror_visual_bridge_t* bridge,
    mirror_visual_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mirror_visual_bridge_get_stats(
    const mirror_visual_bridge_t* bridge,
    mirror_visual_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

int mirror_visual_bridge_connect_bio_async(mirror_visual_bridge_t* bridge) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    return bridge_base_connect_bio_async(&bridge->base);
}

int mirror_visual_bridge_disconnect_bio_async(mirror_visual_bridge_t* bridge) {
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    return bridge_base_disconnect_bio_async(&bridge->base);
}

bool mirror_visual_bridge_is_bio_async_connected(
    const mirror_visual_bridge_t* bridge
) {
    if (!bridge) return false;
    return bridge_base_is_bio_async_connected(&bridge->base);
}

int mirror_visual_broadcast_agent_detection(
    mirror_visual_bridge_t* bridge,
    const agent_detection_t* detection
) {
    if (!bridge || !detection) return -1;

    if (!bridge->base.bio_async_enabled) return 0;

    /* In full implementation, would broadcast via bio_router_send */
    NIMCP_LOGGING_DEBUG("Broadcasting agent detection: detected=%d, conf=%.2f",
                        detection->agent_detected, detection->confidence);

    return 0;
}

int mirror_visual_broadcast_social_salience(
    mirror_visual_bridge_t* bridge,
    const social_salience_t* salience
) {
    if (!bridge || !salience) return -1;

    if (!bridge->base.bio_async_enabled) return 0;

    /* In full implementation, would broadcast via bio_router_send */
    NIMCP_LOGGING_DEBUG("Broadcasting social salience: level=%.2f, boost=%.2f",
                        salience->overall_salience, salience->attention_boost_factor);

    return 0;
}
