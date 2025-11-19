//=============================================================================
// nimcp_middleware_context.c - Middleware Context Implementation
//=============================================================================

#include "middleware/pipeline/nimcp_middleware_context.h"
#include "middleware/events/nimcp_event_types.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

middleware_context_t* middleware_context_create(brain_t brain,
                                                 uint32_t max_features,
                                                 uint32_t max_patterns,
                                                 uint32_t event_history_size,
                                                 uint32_t num_stages) {
    middleware_context_t* ctx = nimcp_calloc(1, sizeof(middleware_context_t));
    if (!ctx) return NULL;

    ctx->brain = brain;
    ctx->cached_features = nimcp_calloc(max_features, sizeof(float));
    ctx->num_cached_features = max_features;
    ctx->detected_patterns = nimcp_calloc(max_patterns, sizeof(uint32_t));
    ctx->pattern_confidences = nimcp_calloc(max_patterns, sizeof(float));
    ctx->num_detected_patterns = max_patterns;
    ctx->recent_events = nimcp_calloc(event_history_size, sizeof(event_t));
    ctx->recent_event_capacity = event_history_size;
    ctx->stage_timings_us = nimcp_calloc(num_stages, sizeof(uint64_t));
    ctx->num_stages = num_stages;

    return ctx;
}

void middleware_context_destroy(middleware_context_t* context) {
    if (!context) return;

    nimcp_free(context->active_neurons);
    nimcp_free(context->cached_features);
    nimcp_free(context->detected_patterns);
    nimcp_free(context->pattern_confidences);

    // Free events in history
    for (uint32_t i = 0; i < context->recent_event_count; i++) {
        event_free(&context->recent_events[i]);
    }
    nimcp_free(context->recent_events);

    nimcp_free(context->stage_timings_us);
    nimcp_free(context);
}

void middleware_context_set_active_neurons(middleware_context_t* context,
                                           uint32_t* neurons, uint32_t count) {
    if (!context) return;

    nimcp_free(context->active_neurons);
    context->active_neurons = NULL;
    context->num_active_neurons = 0;

    // Handle empty neuron list
    if (count == 0 || !neurons) return;

    context->active_neurons = nimcp_malloc(count * sizeof(uint32_t));
    if (context->active_neurons) {
        memcpy(context->active_neurons, neurons, count * sizeof(uint32_t));
        context->num_active_neurons = count;
    }
}

void middleware_context_cache_features(middleware_context_t* context,
                                       float* features, uint32_t count) {
    if (!context || count > context->num_cached_features) return;

    memcpy(context->cached_features, features, count * sizeof(float));
    context->features_valid = true;
}

bool middleware_context_get_cached_features(middleware_context_t* context,
                                            float** features, uint32_t* count) {
    if (!context || !context->features_valid) return false;

    *features = context->cached_features;
    *count = context->num_cached_features;
    return true;
}

void middleware_context_invalidate_cache(middleware_context_t* context) {
    if (context) context->features_valid = false;
}

void middleware_context_add_event(middleware_context_t* context, const event_t* event) {
    if (!context || !event) return;

    // Free old event at write position
    if (context->recent_event_count >= context->recent_event_capacity) {
        event_free(&context->recent_events[context->recent_event_head]);
    }

    // Copy new event
    event_copy(&context->recent_events[context->recent_event_head], event);

    // Update counters
    context->recent_event_head = (context->recent_event_head + 1) % context->recent_event_capacity;
    if (context->recent_event_count < context->recent_event_capacity) {
        context->recent_event_count++;
    }
}

uint32_t middleware_context_get_recent_events(middleware_context_t* context, event_t** events) {
    if (!context) return 0;

    *events = context->recent_events;
    return context->recent_event_count;
}

void middleware_context_record_stage_time(middleware_context_t* context,
                                          uint32_t stage_index, uint64_t time_us) {
    if (context && stage_index < context->num_stages) {
        context->stage_timings_us[stage_index] = time_us;
    }
}

bool middleware_context_get_stage_timings(middleware_context_t* context,
                                          uint64_t** timings, uint32_t* count) {
    if (!context) return false;

    *timings = context->stage_timings_us;
    *count = context->num_stages;
    return true;
}
