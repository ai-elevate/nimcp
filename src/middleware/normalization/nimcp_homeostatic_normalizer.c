#include "middleware/normalization/nimcp_homeostatic_normalizer.h"
#include "utils/memory/nimcp_memory.h"
#include <math.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"



#define LOG_MODULE "nimcp_homeostatic_normalizer"
#define LOG_MODULE_ID 0x0521

typedef struct {
    float current_activity;
    float scaling_factor;
    float accumulated_error;
} homeostatic_channel_t;

struct homeostatic_normalizer {
    size_t num_channels;
    float target_activity;
    float time_constant;
    homeostatic_channel_t* channels;
};

homeostatic_normalizer_t* homeostatic_normalizer_create(
    size_t num_channels,
    float target_activity,
    float time_constant
) {
    if (num_channels == 0 || time_constant <= 0.0f) return NULL;

    homeostatic_normalizer_t* norm = nimcp_calloc(1, sizeof(homeostatic_normalizer_t));
    if (!norm) return NULL;

    norm->num_channels = num_channels;
    norm->target_activity = target_activity;
    norm->time_constant = time_constant;

    norm->channels = nimcp_calloc(num_channels, sizeof(homeostatic_channel_t));
    if (!norm->channels) {
        nimcp_free(norm);
        return NULL;
    }

    for (size_t i = 0; i < num_channels; i++) {
        norm->channels[i].current_activity = 0.0f;
        norm->channels[i].scaling_factor = 1.0f;
        norm->channels[i].accumulated_error = 0.0f;
    }

    return norm;
}

void homeostatic_normalizer_destroy(homeostatic_normalizer_t* normalizer) {
    if (!normalizer) return;
    nimcp_free(normalizer->channels);
    nimcp_free(normalizer);
}

bool homeostatic_normalizer_update(
    homeostatic_normalizer_t* normalizer,
    size_t channel,
    float activity,
    float dt
) {
    if (!normalizer || channel >= normalizer->num_channels || dt <= 0.0f) return false;

    homeostatic_channel_t* ch = &normalizer->channels[channel];

    // Update activity with exponential smoothing
    float alpha = dt / normalizer->time_constant;
    ch->current_activity = (1.0f - alpha) * ch->current_activity + alpha * activity;

    // Calculate error
    float error = normalizer->target_activity - ch->current_activity;
    ch->accumulated_error += error * dt;

    // Update scaling factor (homeostatic plasticity)
    ch->scaling_factor = 1.0f + 0.1f * ch->accumulated_error;

    // Clamp scaling factor
    if (ch->scaling_factor < 0.1f) ch->scaling_factor = 0.1f;
    if (ch->scaling_factor > 10.0f) ch->scaling_factor = 10.0f;

    return true;
}

float homeostatic_normalizer_get_scaling(
    const homeostatic_normalizer_t* normalizer,
    size_t channel
) {
    if (!normalizer || channel >= normalizer->num_channels) return 1.0f;
    return normalizer->channels[channel].scaling_factor;
}

float homeostatic_normalizer_apply(
    const homeostatic_normalizer_t* normalizer,
    size_t channel,
    float value
) {
    if (!normalizer || channel >= normalizer->num_channels) return value;
    return value * normalizer->channels[channel].scaling_factor;
}

bool homeostatic_normalizer_reset_channel(
    homeostatic_normalizer_t* normalizer,
    size_t channel
) {
    if (!normalizer || channel >= normalizer->num_channels) return false;

    homeostatic_channel_t* ch = &normalizer->channels[channel];
    ch->current_activity = 0.0f;
    ch->scaling_factor = 1.0f;
    ch->accumulated_error = 0.0f;

    return true;
}

void homeostatic_normalizer_reset_all(homeostatic_normalizer_t* normalizer) {
    if (!normalizer) return;
    for (size_t i = 0; i < normalizer->num_channels; i++) {
        homeostatic_normalizer_reset_channel(normalizer, i);
    }
}

size_t homeostatic_normalizer_num_channels(const homeostatic_normalizer_t* normalizer) {
    if (!normalizer) return 0;
    return normalizer->num_channels;
}
