/**
 * @file nimcp_working_memory_adapter.c
 * @brief Working memory adapter implementation (stub for testing)
 */

#include "middleware/cognitive/nimcp_working_memory_adapter.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct working_memory_adapter_struct {
    brain_temporal_buffer_t* buffer;
    brain_feature_normalizer_t* normalizer;
    brain_spike_feature_extractor_t spike_extractor;
    working_memory_adapter_config_t config;
};

working_memory_adapter_config_t working_memory_adapter_default_config(void) {
    return (working_memory_adapter_config_t){
        .num_channels = 100,
        .buffer_size = BUFFER_SIZE_100MS,
        .norm_type = NORMALIZE_ZSCORE,
        .max_features = 50,
        .enable_spike_features = false,
        .enable_oscillations = true
    };
}

working_memory_adapter_t working_memory_adapter_create(
    const working_memory_adapter_config_t* config
) {
    if (!config || config->num_channels == 0) return NULL;

    struct working_memory_adapter_struct* adapter =
        nimcp_calloc(1, sizeof(struct working_memory_adapter_struct));
    if (!adapter) return NULL;

    adapter->config = *config;

    adapter->buffer = brain_create_temporal_buffer(
        config->num_channels, config->buffer_size
    );
    if (!adapter->buffer) {
        working_memory_adapter_destroy(adapter);
        return NULL;
    }

    // Only create normalizer if max_features > 0
    if (config->max_features > 0) {
        adapter->normalizer = brain_create_feature_normalizer(
            config->max_features, config->norm_type
        );
        if (!adapter->normalizer) {
            working_memory_adapter_destroy(adapter);
            return NULL;
        }
    }

    if (config->enable_spike_features) {
        adapter->spike_extractor = brain_create_spike_feature_extractor(
            config->num_channels, config->enable_oscillations, true
        );
        if (!adapter->spike_extractor) {
            working_memory_adapter_destroy(adapter);
            return NULL;
        }
    }

    return adapter;
}

void working_memory_adapter_destroy(working_memory_adapter_t adapter) {
    if (!adapter) return;
    brain_destroy_temporal_buffer(adapter->buffer);
    brain_destroy_feature_normalizer(adapter->normalizer);
    brain_destroy_spike_feature_extractor(adapter->spike_extractor);
    nimcp_free(adapter);
}

uint32_t working_memory_adapter_update(
    working_memory_adapter_t adapter,
    const float* activity,
    uint32_t num_channels,
    uint64_t timestamp,
    float* features_out
) {
    if (!adapter || !activity || !features_out) return 0;
    if (num_channels != adapter->config.num_channels) return 0;

    // Buffer activity
    if (!brain_buffer_activity(adapter->buffer, activity, num_channels, timestamp)) {
        return 0;
    }

    // If no normalizer (max_features == 0), return 0 features
    if (!adapter->normalizer || adapter->config.max_features == 0) {
        return 0;
    }

    // Extract and normalize features
    return brain_extract_and_normalize(
        adapter->buffer, adapter->normalizer, features_out, adapter->config.max_features
    );
}
