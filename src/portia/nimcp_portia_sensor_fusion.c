/**
 * @file nimcp_portia_sensor_fusion.c
 * @brief Lightweight sensor fusion implementation
 */

#include "portia/nimcp_portia_sensor_fusion.h"
#include "async/nimcp_bio_messages.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include <string.h>
#include <math.h>
#include <float.h>

// Constants
#define MAX_SENSOR_HISTORY 10
#define MIN_CONFIDENCE 0.01f
#define MAX_CONFIDENCE 1.0f
#define DEFAULT_PROCESS_NOISE 0.1f
#define DEFAULT_OUTLIER_THRESHOLD 3.0f
#define KALMAN_STATE_DIM 9  // x, y, z, vx, vy, vz, ax, ay, az

/**
 * Sensor history entry
 */
typedef struct {
    sensor_reading_t reading;
    uint64_t last_update_ms;
    uint32_t dropout_count;
    float running_mean;
    float running_variance;
    uint32_t sample_count;
} sensor_history_t;

/**
 * Extended Kalman Filter state
 */
typedef struct {
    float state[KALMAN_STATE_DIM];      // State vector
    float covariance[KALMAN_STATE_DIM][KALMAN_STATE_DIM];  // Covariance matrix
    float process_noise_matrix[KALMAN_STATE_DIM][KALMAN_STATE_DIM];
    uint64_t last_update_ms;
} kalman_state_t;

/**
 * Fusion context structure
 */
struct portia_fusion_ctx {
    portia_fusion_config_t config;
    nimcp_bio_ctx_t* bio_ctx;

    // Current state
    fused_state_t current_state;
    kalman_state_t kalman;

    // Sensor data
    sensor_history_t sensors[SENSOR_TYPE_COUNT];
    sensor_reading_t latest_readings[SENSOR_TYPE_COUNT];

    // Statistics
    portia_fusion_stats_t stats;

    // Thread safety
    nimcp_platform_mutex_t mutex;

    // Temporal tracking
    uint64_t last_fusion_ms;
    uint32_t active_sensors_mask;

    // Validation
    uint32_t magic;
};

#define FUSION_CTX_MAGIC 0x46555300  // "FUS\0"

// Forward declarations
static bool validate_fusion_ctx(const portia_fusion_ctx_t* ctx);
static bool process_weighted_average(portia_fusion_ctx_t* ctx);
static bool process_kalman_filter(portia_fusion_ctx_t* ctx);
static void initialize_kalman(kalman_state_t* kalman, float process_noise);
static void kalman_predict(kalman_state_t* kalman, float dt);
static void kalman_update(kalman_state_t* kalman, const sensor_reading_t* reading, float noise_variance);
static bool is_outlier(const portia_fusion_ctx_t* ctx, const sensor_reading_t* reading);
static void update_sensor_statistics(sensor_history_t* history, const sensor_reading_t* reading);
static float calculate_overall_confidence(const portia_fusion_ctx_t* ctx);
static void broadcast_fusion_event(portia_fusion_ctx_t* ctx, const char* event_type);

/**
 * Get sensor name string
 */
const char* portia_fusion_sensor_name(sensor_type_t type) {
    static const char* names[] = {
        "VISUAL", "AUDIO", "VIBRATION", "CHEMICAL",
        "THERMAL", "PROXIMITY", "IMU", "GPS"
    };

    if (type >= 0 && type < SENSOR_TYPE_COUNT) {
        return names[type];
    }
    return "UNKNOWN";
}

/**
 * Create default fusion configuration
 */
portia_fusion_config_t portia_fusion_default_config(void) {
    portia_fusion_config_t config;
    memset(&config, 0, sizeof(config));

    // Initialize all sensors
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        config.sensors[i].type = i;
        config.sensors[i].weight = 1.0f / SENSOR_TYPE_COUNT;
        config.sensors[i].noise_variance = 0.1f;
        config.sensors[i].update_rate_hz = 10;
        config.sensors[i].enabled = true;
        config.sensors[i].required = false;
    }

    config.process_noise = DEFAULT_PROCESS_NOISE;
    config.fusion_rate_hz = 20;
    config.enable_kalman = false;  // Start with simple mode
    config.enable_fallback = true;
    config.outlier_threshold = DEFAULT_OUTLIER_THRESHOLD;
    config.min_sensors = 1;

    return config;
}

/**
 * Validate fusion context
 */
static bool validate_fusion_ctx(const portia_fusion_ctx_t* ctx) {
    if (!bbb_validate_pointer(ctx, sizeof(portia_fusion_ctx_t))) {
        LOG_ERROR("Invalid fusion context pointer");
        return false;
    }

    if (ctx->magic != FUSION_CTX_MAGIC) {
        LOG_ERROR("Invalid fusion context magic: 0x%08X", ctx->magic);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid fusion context magic");
        return false;
    }

    return true;
}

/**
 * Initialize Kalman filter
 */
static void initialize_kalman(kalman_state_t* kalman, float process_noise) {
    memset(kalman, 0, sizeof(kalman_state_t));

    // Initialize covariance matrix with high uncertainty
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        kalman->covariance[i][i] = 1.0f;
    }

    // Initialize process noise matrix
    float noise_factor = process_noise * process_noise;
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        kalman->process_noise_matrix[i][i] = noise_factor;
    }

    kalman->last_update_ms = 0;

    LOG_DEBUG("Initialized Kalman filter with process_noise=%.4f", process_noise);
}

/**
 * Kalman filter prediction step
 */
static void kalman_predict(kalman_state_t* kalman, float dt) {
    if (dt <= 0.0f || dt > 1.0f) {
        dt = 0.01f;  // Default 10ms
    }

    // State transition: position += velocity * dt, velocity += acceleration * dt
    kalman->state[0] += kalman->state[3] * dt;  // x += vx * dt
    kalman->state[1] += kalman->state[4] * dt;  // y += vy * dt
    kalman->state[2] += kalman->state[5] * dt;  // z += vz * dt
    kalman->state[3] += kalman->state[6] * dt;  // vx += ax * dt
    kalman->state[4] += kalman->state[7] * dt;  // vy += ay * dt
    kalman->state[5] += kalman->state[8] * dt;  // vz += az * dt

    // Predict covariance: P = F * P * F^T + Q
    // Simplified for diagonal Q matrix
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        for (int j = 0; j < KALMAN_STATE_DIM; j++) {
            kalman->covariance[i][j] += kalman->process_noise_matrix[i][j];
        }
    }
}

/**
 * Kalman filter update step
 */
static void kalman_update(kalman_state_t* kalman, const sensor_reading_t* reading, float noise_variance) {
    // Measurement residual
    float innovation = reading->value - kalman->state[0];  // Simplified: measure position

    // Innovation covariance
    float S = kalman->covariance[0][0] + noise_variance;

    if (S < 1e-6f) {
        S = 1e-6f;  // Avoid division by zero
    }

    // Kalman gain
    float K[KALMAN_STATE_DIM];
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        K[i] = kalman->covariance[i][0] / S;
    }

    // Update state
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        kalman->state[i] += K[i] * innovation;
    }

    // Update covariance: P = (I - K*H) * P
    for (int i = 0; i < KALMAN_STATE_DIM; i++) {
        for (int j = 0; j < KALMAN_STATE_DIM; j++) {
            kalman->covariance[i][j] -= K[i] * kalman->covariance[0][j];
        }
    }
}

/**
 * Check if reading is an outlier
 */
static bool is_outlier(const portia_fusion_ctx_t* ctx, const sensor_reading_t* reading) {
    if (reading->type >= SENSOR_TYPE_COUNT) {
        return true;
    }

    const sensor_history_t* history = &ctx->sensors[reading->type];

    // Need at least 3 samples to detect outliers
    if (history->sample_count < 3) {
        return false;
    }

    float std_dev = sqrtf(history->running_variance);
    float z_score = fabsf(reading->value - history->running_mean) / (std_dev + 1e-6f);

    bool outlier = z_score > ctx->config.outlier_threshold;

    if (outlier) {
        LOG_DEBUG("Outlier detected for %s: value=%.3f, mean=%.3f, std=%.3f, z=%.2f",
                  portia_fusion_sensor_name(reading->type),
                  reading->value, history->running_mean, std_dev, z_score);
    }

    return outlier;
}

/**
 * Update sensor statistics
 */
static void update_sensor_statistics(sensor_history_t* history, const sensor_reading_t* reading) {
    history->last_update_ms = reading->timestamp_ms;
    history->reading = *reading;

    // Update running statistics
    uint32_t n = history->sample_count;
    float delta = reading->value - history->running_mean;

    history->running_mean += delta / (n + 1);
    history->running_variance = (n * history->running_variance + delta * (reading->value - history->running_mean)) / (n + 1);
    history->sample_count++;

    history->dropout_count = 0;  // Reset dropout counter
}

/**
 * Calculate overall confidence
 */
static float calculate_overall_confidence(const portia_fusion_ctx_t* ctx) {
    float total_weight = 0.0f;
    float weighted_confidence = 0.0f;
    uint32_t active_count = 0;

    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        if (!ctx->config.sensors[i].enabled) {
            continue;
        }

        const sensor_history_t* history = &ctx->sensors[i];
        if (history->reading.valid && history->reading.confidence > 0.0f) {
            float weight = ctx->config.sensors[i].weight;
            weighted_confidence += weight * history->reading.confidence;
            total_weight += weight;
            active_count++;
        }
    }

    if (total_weight < 1e-6f || active_count == 0) {
        return MIN_CONFIDENCE;
    }

    // Normalize and apply sensor count penalty
    float confidence = weighted_confidence / total_weight;

    // Reduce confidence if fewer sensors are active
    if (active_count < ctx->config.min_sensors) {
        confidence *= 0.5f;
    }

    return fminf(fmaxf(confidence, MIN_CONFIDENCE), MAX_CONFIDENCE);
}

/**
 * Broadcast fusion event via bio-async
 */
static void broadcast_fusion_event(portia_fusion_ctx_t* ctx, const char* event_type) {
    if (!ctx->bio_ctx) {
        return;
    }

    // Create message payload
    char payload[256];
    snprintf(payload, sizeof(payload),
             "fusion_event:type=%s,confidence=%.3f,sensors=%u,timestamp=%lu",
             event_type,
             ctx->current_state.confidence,
             ctx->active_sensors_mask,
             (unsigned long)ctx->current_state.timestamp_ms);

    // Broadcast via bio-async
    nimcp_bio_message_t msg = {
        .type = BIO_MSG_CUSTOM,
        .priority = BIO_MSG_PRIORITY_NORMAL,
        .timestamp_ms = ctx->current_state.timestamp_ms
    };

    strncpy(msg.source_id, "portia_fusion", sizeof(msg.source_id) - 1);
    strncpy(msg.data, payload, sizeof(msg.data) - 1);
    msg.data_size = strlen(payload);

    nimcp_bio_send_message(ctx->bio_ctx, &msg);
}

/**
 * Process weighted average fusion
 */
static bool process_weighted_average(portia_fusion_ctx_t* ctx) {
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    float sum_weights = 0.0f;
    uint32_t active_sensors = 0;

    uint64_t current_time_ms = nimcp_platform_get_time_ms();

    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        if (!ctx->config.sensors[i].enabled) {
            continue;
        }

        const sensor_history_t* history = &ctx->sensors[i];
        if (!history->reading.valid || history->reading.confidence < MIN_CONFIDENCE) {
            continue;
        }

        // Check for stale data
        uint64_t age_ms = current_time_ms - history->last_update_ms;
        uint32_t expected_period_ms = 1000 / ctx->config.sensors[i].update_rate_hz;
        if (age_ms > expected_period_ms * 5) {
            LOG_DEBUG("Stale data for %s: age=%lu ms", portia_fusion_sensor_name(i), (unsigned long)age_ms);
            continue;
        }

        float weight = ctx->config.sensors[i].weight * history->reading.confidence;

        // For simplicity, map sensor value to position
        sum_x += weight * history->reading.value;
        sum_y += weight * history->reading.value * 0.5f;  // Simplified mapping
        sum_z += weight * history->reading.value * 0.25f;
        sum_weights += weight;
        active_sensors++;

        ctx->active_sensors_mask |= (1u << i);
    }

    if (sum_weights < 1e-6f) {
        LOG_WARN("No valid sensor data for fusion");
        return false;
    }

    // Update state
    ctx->current_state.x = sum_x / sum_weights;
    ctx->current_state.y = sum_y / sum_weights;
    ctx->current_state.z = sum_z / sum_weights;

    // Simple velocity estimation
    if (ctx->last_fusion_ms > 0) {
        float dt = (current_time_ms - ctx->last_fusion_ms) / 1000.0f;
        if (dt > 0.0f && dt < 1.0f) {
            ctx->current_state.vx = (ctx->current_state.x - ctx->kalman.state[0]) / dt;
            ctx->current_state.vy = (ctx->current_state.y - ctx->kalman.state[1]) / dt;
            ctx->current_state.vz = (ctx->current_state.z - ctx->kalman.state[2]) / dt;
        }
    }

    ctx->current_state.heading = atan2f(ctx->current_state.vy, ctx->current_state.vx);
    ctx->current_state.confidence = calculate_overall_confidence(ctx);
    ctx->current_state.timestamp_ms = current_time_ms;
    ctx->current_state.contributing_sensors = ctx->active_sensors_mask;

    ctx->last_fusion_ms = current_time_ms;

    LOG_DEBUG("Weighted fusion: pos(%.3f,%.3f,%.3f) vel(%.3f,%.3f,%.3f) conf=%.3f sensors=%u",
              ctx->current_state.x, ctx->current_state.y, ctx->current_state.z,
              ctx->current_state.vx, ctx->current_state.vy, ctx->current_state.vz,
              ctx->current_state.confidence, active_sensors);

    return true;
}

/**
 * Process Kalman filter fusion
 */
static bool process_kalman_filter(portia_fusion_ctx_t* ctx) {
    uint64_t current_time_ms = nimcp_platform_get_time_ms();

    // Prediction step
    if (ctx->kalman.last_update_ms > 0) {
        float dt = (current_time_ms - ctx->kalman.last_update_ms) / 1000.0f;
        kalman_predict(&ctx->kalman, dt);
    }

    ctx->kalman.last_update_ms = current_time_ms;

    // Update step with each valid sensor
    uint32_t update_count = 0;
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        if (!ctx->config.sensors[i].enabled) {
            continue;
        }

        const sensor_history_t* history = &ctx->sensors[i];
        if (!history->reading.valid || history->reading.confidence < MIN_CONFIDENCE) {
            continue;
        }

        // Check for stale data
        uint64_t age_ms = current_time_ms - history->last_update_ms;
        uint32_t expected_period_ms = 1000 / ctx->config.sensors[i].update_rate_hz;
        if (age_ms > expected_period_ms * 5) {
            continue;
        }

        kalman_update(&ctx->kalman, &history->reading, ctx->config.sensors[i].noise_variance);
        update_count++;
        ctx->active_sensors_mask |= (1u << i);
    }

    if (update_count == 0) {
        LOG_WARN("No valid sensor data for Kalman update");
        return false;
    }

    // Copy Kalman state to fused state
    ctx->current_state.x = ctx->kalman.state[0];
    ctx->current_state.y = ctx->kalman.state[1];
    ctx->current_state.z = ctx->kalman.state[2];
    ctx->current_state.vx = ctx->kalman.state[3];
    ctx->current_state.vy = ctx->kalman.state[4];
    ctx->current_state.vz = ctx->kalman.state[5];
    ctx->current_state.heading = atan2f(ctx->kalman.state[4], ctx->kalman.state[3]);
    ctx->current_state.confidence = calculate_overall_confidence(ctx);
    ctx->current_state.timestamp_ms = current_time_ms;
    ctx->current_state.contributing_sensors = ctx->active_sensors_mask;

    ctx->last_fusion_ms = current_time_ms;

    LOG_DEBUG("Kalman fusion: pos(%.3f,%.3f,%.3f) vel(%.3f,%.3f,%.3f) conf=%.3f updates=%u",
              ctx->current_state.x, ctx->current_state.y, ctx->current_state.z,
              ctx->current_state.vx, ctx->current_state.vy, ctx->current_state.vz,
              ctx->current_state.confidence, update_count);

    return true;
}

/**
 * Initialize sensor fusion system
 */
portia_fusion_ctx_t* portia_fusion_init(
    const portia_fusion_config_t* config,
    nimcp_bio_ctx_t* bio_ctx
) {
    // Validate input
    if (!bbb_validate_pointer(config, sizeof(portia_fusion_config_t))) {
        LOG_ERROR("Invalid config pointer");
        return NULL;
    }

    // Validate configuration ranges
    if (config->fusion_rate_hz < 1 || config->fusion_rate_hz > 1000) {
        LOG_ERROR("Invalid fusion rate: %u Hz", config->fusion_rate_hz);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid fusion rate");
        return NULL;
    }

    if (config->outlier_threshold < 1.0f || config->outlier_threshold > 10.0f) {
        LOG_ERROR("Invalid outlier threshold: %.2f", config->outlier_threshold);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid outlier threshold");
        return NULL;
    }

    // Allocate context
    portia_fusion_ctx_t* ctx = nimcp_calloc(1, sizeof(portia_fusion_ctx_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate fusion context");
        return NULL;
    }

    // Initialize context
    ctx->config = *config;
    ctx->bio_ctx = bio_ctx;
    ctx->magic = FUSION_CTX_MAGIC;

    // Initialize mutex
    if (nimcp_platform_mutex_init(&ctx->mutex) != 0) {
        LOG_ERROR("Failed to initialize fusion mutex");
        nimcp_free(ctx);
        return NULL;
    }

    // Initialize sensor histories
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        ctx->sensors[i].reading.type = i;
        ctx->sensors[i].reading.valid = false;
        ctx->sensors[i].dropout_count = 0;
        ctx->sensors[i].sample_count = 0;
    }

    // Initialize Kalman filter if enabled
    if (config->enable_kalman) {
        initialize_kalman(&ctx->kalman, config->process_noise);
    }

    // Initialize state
    memset(&ctx->current_state, 0, sizeof(fused_state_t));
    ctx->current_state.confidence = MIN_CONFIDENCE;

    // Initialize statistics
    memset(&ctx->stats, 0, sizeof(portia_fusion_stats_t));

    LOG_INFO("Initialized Portia sensor fusion: kalman=%d, rate=%u Hz, outlier_threshold=%.2f",
             config->enable_kalman, config->fusion_rate_hz, config->outlier_threshold);

    bbb_audit_log(BBB_AUDIT_SYSTEM_INIT, "Portia sensor fusion initialized");

    if (bio_ctx) {
        broadcast_fusion_event(ctx, "init");
    }

    return ctx;
}

/**
 * Destroy fusion system
 */
void portia_fusion_destroy(portia_fusion_ctx_t* ctx) {
    if (!validate_fusion_ctx(ctx)) {
        return;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    LOG_INFO("Destroying Portia sensor fusion: fusions=%lu, outliers=%lu, dropouts=%lu",
             (unsigned long)ctx->stats.successful_fusions,
             (unsigned long)ctx->stats.outliers_rejected,
             (unsigned long)ctx->stats.sensor_dropouts);

    if (ctx->bio_ctx) {
        broadcast_fusion_event(ctx, "destroy");
    }

    ctx->magic = 0;

    nimcp_platform_mutex_unlock(&ctx->mutex);
    nimcp_platform_mutex_destroy(&ctx->mutex);

    nimcp_free(ctx);

    bbb_audit_log(BBB_AUDIT_SYSTEM_SHUTDOWN, "Portia sensor fusion destroyed");
}

/**
 * Update sensor reading
 */
bool portia_fusion_update_sensor(
    portia_fusion_ctx_t* ctx,
    const sensor_reading_t* reading
) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    if (!bbb_validate_pointer(reading, sizeof(sensor_reading_t))) {
        LOG_ERROR("Invalid reading pointer");
        return false;
    }

    if (reading->type >= SENSOR_TYPE_COUNT) {
        LOG_ERROR("Invalid sensor type: %d", reading->type);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid sensor type");
        return false;
    }

    if (!bbb_validate_range(reading->confidence, 0.0f, 1.0f)) {
        LOG_ERROR("Invalid confidence: %.3f", reading->confidence);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid sensor confidence");
        return false;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Check if sensor is enabled
    if (!ctx->config.sensors[reading->type].enabled) {
        LOG_DEBUG("Ignoring disabled sensor: %s", portia_fusion_sensor_name(reading->type));
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return false;
    }

    // Check for outliers
    if (is_outlier(ctx, reading)) {
        ctx->stats.outliers_rejected++;
        LOG_DEBUG("Rejected outlier from %s", portia_fusion_sensor_name(reading->type));
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return false;
    }

    // Update sensor history
    update_sensor_statistics(&ctx->sensors[reading->type], reading);
    ctx->latest_readings[reading->type] = *reading;

    ctx->stats.total_updates++;

    LOG_DEBUG("Updated sensor %s: value=%.3f, confidence=%.3f, timestamp=%lu",
              portia_fusion_sensor_name(reading->type),
              reading->value, reading->confidence,
              (unsigned long)reading->timestamp_ms);

    nimcp_platform_mutex_unlock(&ctx->mutex);

    return true;
}

/**
 * Process fusion algorithm
 */
bool portia_fusion_process(portia_fusion_ctx_t* ctx) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    ctx->active_sensors_mask = 0;

    // Count active sensors
    uint32_t active_count = 0;
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        if (ctx->config.sensors[i].enabled && ctx->sensors[i].reading.valid) {
            active_count++;
        }
    }

    ctx->stats.active_sensor_count = active_count;

    // Check minimum sensor requirement
    if (active_count < ctx->config.min_sensors) {
        LOG_WARN("Insufficient active sensors: %u < %u", active_count, ctx->config.min_sensors);
        ctx->current_state.confidence = MIN_CONFIDENCE;
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return false;
    }

    // Run appropriate fusion algorithm
    bool success;
    if (ctx->config.enable_kalman) {
        success = process_kalman_filter(ctx);
    } else {
        success = process_weighted_average(ctx);
    }

    if (success) {
        ctx->stats.successful_fusions++;
        ctx->stats.average_confidence =
            (ctx->stats.average_confidence * (ctx->stats.successful_fusions - 1) +
             ctx->current_state.confidence) / ctx->stats.successful_fusions;

        if (ctx->bio_ctx) {
            broadcast_fusion_event(ctx, "fusion_update");
        }
    }

    nimcp_platform_mutex_unlock(&ctx->mutex);

    return success;
}

/**
 * Get current fused state
 */
bool portia_fusion_get_state(
    const portia_fusion_ctx_t* ctx,
    fused_state_t* state
) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    if (!bbb_validate_pointer(state, sizeof(fused_state_t))) {
        LOG_ERROR("Invalid state pointer");
        return false;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&ctx->mutex);
    *state = ctx->current_state;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&ctx->mutex);

    return true;
}

/**
 * Set sensor weight
 */
bool portia_fusion_set_weight(
    portia_fusion_ctx_t* ctx,
    sensor_type_t type,
    float weight
) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    if (type >= SENSOR_TYPE_COUNT) {
        LOG_ERROR("Invalid sensor type: %d", type);
        return false;
    }

    if (!bbb_validate_range(weight, 0.0f, 1.0f)) {
        LOG_ERROR("Invalid weight: %.3f", weight);
        bbb_audit_log(BBB_AUDIT_VALIDATION_FAILED, "Invalid sensor weight");
        return false;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);
    ctx->config.sensors[type].weight = weight;
    nimcp_platform_mutex_unlock(&ctx->mutex);

    LOG_INFO("Set %s weight to %.3f", portia_fusion_sensor_name(type), weight);

    return true;
}

/**
 * Enable or disable sensor
 */
bool portia_fusion_enable_sensor(
    portia_fusion_ctx_t* ctx,
    sensor_type_t type,
    bool enabled
) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    if (type >= SENSOR_TYPE_COUNT) {
        LOG_ERROR("Invalid sensor type: %d", type);
        return false;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);
    ctx->config.sensors[type].enabled = enabled;
    nimcp_platform_mutex_unlock(&ctx->mutex);

    LOG_INFO("%s sensor %s", enabled ? "Enabled" : "Disabled", portia_fusion_sensor_name(type));

    bbb_audit_log(BBB_AUDIT_CONFIG_CHANGE, enabled ? "Sensor enabled" : "Sensor disabled");

    return true;
}

/**
 * Get fusion confidence
 */
float portia_fusion_get_confidence(const portia_fusion_ctx_t* ctx) {
    if (!validate_fusion_ctx(ctx)) {
        return 0.0f;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&ctx->mutex);
    float confidence = ctx->current_state.confidence;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&ctx->mutex);

    return confidence;
}

/**
 * Get fusion statistics
 */
bool portia_fusion_get_stats(
    const portia_fusion_ctx_t* ctx,
    portia_fusion_stats_t* stats
) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    if (!bbb_validate_pointer(stats, sizeof(portia_fusion_stats_t))) {
        LOG_ERROR("Invalid stats pointer");
        return false;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&ctx->mutex);
    *stats = ctx->stats;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&ctx->mutex);

    return true;
}

/**
 * Reset fusion state
 */
bool portia_fusion_reset(portia_fusion_ctx_t* ctx) {
    if (!validate_fusion_ctx(ctx)) {
        return false;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Reset state
    memset(&ctx->current_state, 0, sizeof(fused_state_t));
    ctx->current_state.confidence = MIN_CONFIDENCE;

    // Reset Kalman filter if enabled
    if (ctx->config.enable_kalman) {
        initialize_kalman(&ctx->kalman, ctx->config.process_noise);
    }

    // Reset sensor histories
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        ctx->sensors[i].reading.valid = false;
        ctx->sensors[i].dropout_count = 0;
        ctx->sensors[i].sample_count = 0;
        ctx->sensors[i].running_mean = 0.0f;
        ctx->sensors[i].running_variance = 0.0f;
    }

    // Reset statistics
    memset(&ctx->stats, 0, sizeof(portia_fusion_stats_t));

    ctx->last_fusion_ms = 0;
    ctx->active_sensors_mask = 0;

    nimcp_platform_mutex_unlock(&ctx->mutex);

    LOG_INFO("Reset Portia sensor fusion state");

    if (ctx->bio_ctx) {
        broadcast_fusion_event(ctx, "reset");
    }

    bbb_audit_log(BBB_AUDIT_CONFIG_CHANGE, "Fusion state reset");

    return true;
}
