/**
 * @file nimcp_dragonfly_visual_bridge.c
 * @brief Visual Cortex Bridge Implementation
 *
 * Connects dragonfly interception system to visual cortex perception.
 */

#include "dragonfly/nimcp_dragonfly_visual_bridge.h"
#include "utils/thread/nimcp_thread.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Previous frame blob for motion tracking
 */
typedef struct {
    float x;
    float y;
    float size;
    bool valid;
    uint32_t track_id;
} prev_blob_t;

/**
 * @brief Visual bridge state
 */
struct dragonfly_visual_bridge_s {
    /* Configuration */
    visual_bridge_config_t config;

    /* Connected systems */
    dragonfly_system_t* dragonfly;
    visual_cortex_t* visual_cortex;

    /* State */
    visual_motion_result_t current_result;
    uint32_t frame_count;
    uint32_t next_track_id;

    /* Previous frame for motion detection */
    prev_blob_t prev_blobs[VISUAL_BRIDGE_MAX_DETECTIONS];
    uint32_t num_prev_blobs;

    /* Statistics */
    visual_bridge_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

//=============================================================================
// Helper Functions
//=============================================================================

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static float distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

//=============================================================================
// Configuration Functions
//=============================================================================

visual_bridge_config_t visual_bridge_default_config(void) {
    visual_bridge_config_t config;
    memset(&config, 0, sizeof(config));

    /* Motion detection */
    config.min_motion_speed = 2.0f;       /* 2 pixels/frame */
    config.min_blob_size = 4.0f;          /* 4 pixel radius */
    config.max_blob_size = 200.0f;        /* 200 pixel radius */
    config.contrast_threshold = 0.1f;

    /* Depth estimation */
    config.estimate_depth = true;
    config.assumed_target_size_m = 0.05f; /* 5cm target */

    /* Attention */
    config.use_attention_filter = true;
    config.attention_threshold = 0.2f;

    /* Default calibration (640x480 camera) */
    config.calibration.focal_length = 500.0f;
    config.calibration.principal_x = 320.0f;
    config.calibration.principal_y = 240.0f;
    config.calibration.image_width = 640;
    config.calibration.image_height = 480;
    config.calibration.baseline_distance = 0.0f;

    /* Update rate */
    config.frame_dt_s = 1.0f / 30.0f;  /* 30 FPS */

    return config;
}

bool visual_bridge_validate_config(const visual_bridge_config_t* config) {
    if (!config) return false;

    if (config->min_motion_speed < 0.0f) return false;
    if (config->min_blob_size <= 0.0f) return false;
    if (config->max_blob_size <= config->min_blob_size) return false;
    if (config->contrast_threshold < 0.0f || config->contrast_threshold > 1.0f) return false;
    if (config->assumed_target_size_m <= 0.0f) return false;
    if (config->attention_threshold < 0.0f || config->attention_threshold > 1.0f) return false;
    if (config->calibration.focal_length <= 0.0f) return false;
    if (config->calibration.image_width == 0) return false;
    if (config->calibration.image_height == 0) return false;
    if (config->frame_dt_s <= 0.0f) return false;

    return true;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

dragonfly_visual_bridge_t* dragonfly_visual_bridge_create(
    dragonfly_system_t* dragonfly,
    visual_cortex_t* visual_cortex,
    const visual_bridge_config_t* config
) {
    visual_bridge_config_t default_config;
    if (!config) {
        default_config = visual_bridge_default_config();
        config = &default_config;
    }

    if (!visual_bridge_validate_config(config)) {
        return NULL;
    }

    dragonfly_visual_bridge_t* bridge =
        (dragonfly_visual_bridge_t*)calloc(1, sizeof(dragonfly_visual_bridge_t));
    if (!bridge) return NULL;

    bridge->config = *config;
    bridge->dragonfly = dragonfly;
    bridge->visual_cortex = visual_cortex;
    bridge->next_track_id = 1;

    /* Create mutex */
    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        free(bridge);
        return NULL;
    }

    return bridge;
}

void dragonfly_visual_bridge_destroy(dragonfly_visual_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    free(bridge);
}

int dragonfly_visual_bridge_reset(dragonfly_visual_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    memset(&bridge->current_result, 0, sizeof(bridge->current_result));
    memset(bridge->prev_blobs, 0, sizeof(bridge->prev_blobs));
    bridge->num_prev_blobs = 0;
    bridge->frame_count = 0;
    bridge->next_track_id = 1;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Processing Functions
//=============================================================================

int dragonfly_visual_bridge_process_frame(
    dragonfly_visual_bridge_t* bridge,
    const float* image,
    uint32_t width,
    uint32_t height,
    uint32_t channels
) {
    if (!bridge || !image) return -1;
    if (width == 0 || height == 0) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t start_time = get_time_us();
    bridge->frame_count++;

    /* Clear current result */
    memset(&bridge->current_result, 0, sizeof(bridge->current_result));
    bridge->current_result.timestamp_us = start_time;
    bridge->current_result.frame_number = bridge->frame_count;

    /* TODO: If visual_cortex is available, use it for feature extraction
     * For now, implement simple motion detection directly on image */

    /* Simple intensity-based blob detection (placeholder) */
    /* In a full implementation, this would use visual_cortex_process */

    /* Update statistics */
    uint64_t elapsed = get_time_us() - start_time;
    bridge->stats.frames_processed++;
    bridge->stats.avg_process_time_us =
        (bridge->stats.avg_process_time_us * (bridge->stats.frames_processed - 1) + (float)elapsed)
        / (float)bridge->stats.frames_processed;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_visual_bridge_process_features(
    dragonfly_visual_bridge_t* bridge,
    const float* features,
    uint32_t feature_dim,
    const attention_map_t* attention
) {
    if (!bridge || !features) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* TODO: Extract motion information from visual cortex features */
    /* This would analyze the feature vector for motion-related activations */

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_visual_bridge_inject_blob(
    dragonfly_visual_bridge_t* bridge,
    const motion_blob_t* blob
) {
    if (!bridge || !blob) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t now = get_time_us();

    /* Add to current result */
    if (bridge->current_result.num_blobs < VISUAL_BRIDGE_MAX_DETECTIONS) {
        uint32_t idx = bridge->current_result.num_blobs;
        bridge->current_result.blobs[idx] = *blob;

        /* Assign track ID if not set */
        if (bridge->current_result.blobs[idx].track_id == 0) {
            bridge->current_result.blobs[idx].track_id = bridge->next_track_id++;
        }

        bridge->current_result.num_blobs++;
        bridge->stats.blobs_detected++;

        /* Update attention peak if this is most salient */
        if (blob->salience > bridge->current_result.peak_salience) {
            bridge->current_result.attention_peak_x = blob->center_x;
            bridge->current_result.attention_peak_y = blob->center_y;
            bridge->current_result.peak_salience = blob->salience;
        }
    }

    /* If dragonfly is connected, send detection */
    if (bridge->dragonfly) {
        /* Convert blob to dragonfly detection */
        dragonfly_detection_t det;
        memset(&det, 0, sizeof(det));

        det.id = blob->track_id;

        /* Convert pixel to 3D position */
        float depth = dragonfly_visual_bridge_estimate_depth(bridge, blob->size_pixels);
        float position[3];
        dragonfly_visual_bridge_pixel_to_3d(bridge, blob->center_x, blob->center_y,
                                            depth, position);
        memcpy(det.position, position, sizeof(det.position));

        /* Compute motion direction and speed */
        float motion_speed = sqrtf(blob->velocity_x * blob->velocity_x +
                                   blob->velocity_y * blob->velocity_y);
        det.motion_direction_rad = atan2f(blob->velocity_y, blob->velocity_x);
        det.motion_speed = dragonfly_visual_bridge_pixel_velocity_to_angular(
            bridge, motion_speed);

        det.size = blob->size_pixels / bridge->config.calibration.focal_length;
        det.contrast = blob->contrast;
        det.timestamp_us = now;

        /* Send to dragonfly */
        dragonfly_process_detection(bridge->dragonfly, &det);
        bridge->stats.detections_sent++;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_visual_bridge_get_result(
    const dragonfly_visual_bridge_t* bridge,
    visual_motion_result_t* result
) {
    if (!bridge || !result) return -1;

    nimcp_mutex_lock(((dragonfly_visual_bridge_t*)bridge)->mutex);
    *result = bridge->current_result;
    nimcp_mutex_unlock(((dragonfly_visual_bridge_t*)bridge)->mutex);

    return 0;
}

//=============================================================================
// Utility Functions
//=============================================================================

int dragonfly_visual_bridge_pixel_to_3d(
    const dragonfly_visual_bridge_t* bridge,
    float pixel_x,
    float pixel_y,
    float depth_m,
    float position[3]
) {
    if (!bridge || !position || depth_m <= 0.0f) return -1;

    const visual_calibration_t* cal = &bridge->config.calibration;

    /* Convert from pixel to normalized image coordinates */
    float nx = (pixel_x - cal->principal_x) / cal->focal_length;
    float ny = (pixel_y - cal->principal_y) / cal->focal_length;

    /* Project to 3D at given depth */
    position[0] = nx * depth_m;
    position[1] = ny * depth_m;
    position[2] = depth_m;

    return 0;
}

float dragonfly_visual_bridge_estimate_depth(
    const dragonfly_visual_bridge_t* bridge,
    float blob_size_pixels
) {
    if (!bridge || blob_size_pixels <= 0.0f) return -1.0f;

    /* Estimate depth from angular size:
     * angular_size = actual_size / depth
     * pixel_size = angular_size * focal_length
     * depth = actual_size * focal_length / pixel_size
     */
    float depth = (bridge->config.assumed_target_size_m *
                   bridge->config.calibration.focal_length) / blob_size_pixels;

    return depth;
}

float dragonfly_visual_bridge_pixel_velocity_to_angular(
    const dragonfly_visual_bridge_t* bridge,
    float vel_pixels_per_frame
) {
    if (!bridge) return 0.0f;

    /* Convert pixels/frame to radians/second */
    float radians_per_pixel = 1.0f / bridge->config.calibration.focal_length;
    float radians_per_frame = vel_pixels_per_frame * radians_per_pixel;
    float radians_per_second = radians_per_frame / bridge->config.frame_dt_s;

    return radians_per_second;
}

//=============================================================================
// Statistics Functions
//=============================================================================

int dragonfly_visual_bridge_get_stats(
    const dragonfly_visual_bridge_t* bridge,
    visual_bridge_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(((dragonfly_visual_bridge_t*)bridge)->mutex);

    *stats = bridge->stats;

    /* Calculate derived stats */
    if (stats->frames_processed > 0) {
        stats->avg_blobs_per_frame =
            (float)stats->blobs_detected / (float)stats->frames_processed;
    }

    nimcp_mutex_unlock(((dragonfly_visual_bridge_t*)bridge)->mutex);
    return 0;
}

int dragonfly_visual_bridge_reset_stats(dragonfly_visual_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Configuration Functions
//=============================================================================

int dragonfly_visual_bridge_set_config(
    dragonfly_visual_bridge_t* bridge,
    const visual_bridge_config_t* config
) {
    if (!bridge || !config) return -1;
    if (!visual_bridge_validate_config(config)) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int dragonfly_visual_bridge_get_config(
    const dragonfly_visual_bridge_t* bridge,
    visual_bridge_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(((dragonfly_visual_bridge_t*)bridge)->mutex);
    *config = bridge->config;
    nimcp_mutex_unlock(((dragonfly_visual_bridge_t*)bridge)->mutex);

    return 0;
}

int dragonfly_visual_bridge_set_calibration(
    dragonfly_visual_bridge_t* bridge,
    const visual_calibration_t* calibration
) {
    if (!bridge || !calibration) return -1;
    if (calibration->focal_length <= 0.0f) return -1;
    if (calibration->image_width == 0 || calibration->image_height == 0) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->config.calibration = *calibration;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}
