#include "middleware/normalization/nimcp_adaptive_normalizer.h"
#include "utils/memory/nimcp_memory.h"
#include <math.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"



#define LOG_MODULE "nimcp_adaptive_normalizer"
#define LOG_MODULE_ID 0x0520

typedef struct {
    float mean;
    float variance;
    float learning_rate;
    size_t count;
} adaptive_channel_t;

struct adaptive_normalizer {
    size_t num_channels;
    float initial_learning_rate;
    float adaptation_rate;
    adaptive_channel_t* channels;
};

adaptive_normalizer_t* adaptive_normalizer_create(
    size_t num_channels,
    float initial_learning_rate,
    float adaptation_rate
) {
    if (num_channels == 0) return NULL;

    adaptive_normalizer_t* norm = nimcp_calloc(1, sizeof(adaptive_normalizer_t));
    if (!norm) return NULL;

    norm->num_channels = num_channels;
    norm->initial_learning_rate = initial_learning_rate;
    norm->adaptation_rate = adaptation_rate;

    norm->channels = nimcp_calloc(num_channels, sizeof(adaptive_channel_t));
    if (!norm->channels) {
        nimcp_free(norm);
        return NULL;
    }

    for (size_t i = 0; i < num_channels; i++) {
        norm->channels[i].mean = 0.0f;
        norm->channels[i].variance = 1.0f;
        norm->channels[i].learning_rate = initial_learning_rate;
        norm->channels[i].count = 0;
    }

    return norm;
}

void adaptive_normalizer_destroy(adaptive_normalizer_t* normalizer) {
    if (!normalizer) return;
    nimcp_free(normalizer->channels);
    nimcp_free(normalizer);
}

bool adaptive_normalizer_fit(
    adaptive_normalizer_t* normalizer,
    size_t channel,
    float value
) {
    if (!normalizer || channel >= normalizer->num_channels) return false;

    adaptive_channel_t* ch = &normalizer->channels[channel];

    float lr = ch->learning_rate;
    ch->mean = (1.0f - lr) * ch->mean + lr * value;

    float diff = value - ch->mean;
    ch->variance = (1.0f - lr) * ch->variance + lr * diff * diff;

    // Adapt learning rate based on variance change
    ch->learning_rate *= (1.0f + normalizer->adaptation_rate * fabsf(diff));
    if (ch->learning_rate > 0.5f) ch->learning_rate = 0.5f;
    if (ch->learning_rate < 0.001f) ch->learning_rate = 0.001f;

    ch->count++;
    return true;
}

float adaptive_normalizer_transform(
    const adaptive_normalizer_t* normalizer,
    size_t channel,
    float value
) {
    if (!normalizer || channel >= normalizer->num_channels) return value;

    const adaptive_channel_t* ch = &normalizer->channels[channel];
    if (ch->count == 0 || ch->variance < 0.0001f) return 0.0f;

    return (value - ch->mean) / sqrtf(ch->variance);
}

float adaptive_normalizer_fit_transform(
    adaptive_normalizer_t* normalizer,
    size_t channel,
    float value
) {
    adaptive_normalizer_fit(normalizer, channel, value);
    return adaptive_normalizer_transform(normalizer, channel, value);
}

bool adaptive_normalizer_reset_channel(
    adaptive_normalizer_t* normalizer,
    size_t channel
) {
    if (!normalizer || channel >= normalizer->num_channels) return false;

    adaptive_channel_t* ch = &normalizer->channels[channel];
    ch->mean = 0.0f;
    ch->variance = 1.0f;
    ch->learning_rate = normalizer->initial_learning_rate;
    ch->count = 0;

    return true;
}

void adaptive_normalizer_reset_all(adaptive_normalizer_t* normalizer) {
    if (!normalizer) return;
    for (size_t i = 0; i < normalizer->num_channels; i++) {
        adaptive_normalizer_reset_channel(normalizer, i);
    }
}

size_t adaptive_normalizer_num_channels(const adaptive_normalizer_t* normalizer) {
    if (!normalizer) return 0;
    return normalizer->num_channels;
}
