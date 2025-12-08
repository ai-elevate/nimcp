//=============================================================================
// nimcp_sequence_detector.c - Temporal Sequence Detection
//=============================================================================

#include "middleware/patterns/nimcp_sequence_detector.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"



#define LOG_MODULE "nimcp_sequence_detector"
#define LOG_MODULE_ID 0x0527

// ============================================================================
// CONSTANTS
// ============================================================================

#define SPIKE_BUFFER_SIZE 10000
#define NGRAM_HASH_SIZE 1000

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

typedef struct {
    uint32_t neuron_id;
    double timestamp_ms;
} spike_record_t;

typedef struct {
    spike_record_t* spikes;
    uint32_t capacity;
    uint32_t count;
    uint32_t head;
} spike_buffer_t;

typedef struct ngram_node {
    ngram_pattern_t pattern;
    struct ngram_node* next;
} ngram_node_t;

struct sequence_detector {
    // Configuration
    sequence_detector_config_t config;

    // Spike buffer
    spike_buffer_t buffer;

    // Template library
    sequence_template_t* templates;
    uint32_t num_templates;
    uint32_t next_template_id;

    // N-gram hash table
    ngram_node_t* ngram_table[NGRAM_HASH_SIZE];
    uint32_t total_ngrams;

    // Statistics
    uint64_t total_detections;
    double sum_strength;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static bool init_spike_buffer(spike_buffer_t* buffer, uint32_t capacity) {
    buffer->spikes = (spike_record_t*)nimcp_calloc(capacity, sizeof(spike_record_t));
    if (!buffer->spikes) return false;
    buffer->capacity = capacity;
    buffer->count = 0;
    buffer->head = 0;
    return true;
}

static void free_spike_buffer(spike_buffer_t* buffer) {
    if (buffer && buffer->spikes) {
        nimcp_free(buffer->spikes);
        buffer->spikes = NULL;
    }
}

static void buffer_add_spike(spike_buffer_t* buffer, uint32_t neuron_id,
                            double timestamp_ms) {
    buffer->spikes[buffer->head].neuron_id = neuron_id;
    buffer->spikes[buffer->head].timestamp_ms = timestamp_ms;
    buffer->head = (buffer->head + 1) % buffer->capacity;
    if (buffer->count < buffer->capacity) {
        buffer->count++;
    }
}

static uint32_t hash_ngram(const uint32_t* neurons, uint32_t length) {
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < length; i++) {
        hash = ((hash << 5) + hash) + neurons[i];
    }
    return hash % NGRAM_HASH_SIZE;
}

static void add_ngram(sequence_detector_t* detector, const uint32_t* neurons,
                     uint32_t length, float interval_ms) {
    if (length < 2 || length > detector->config.max_ngram) return;

    uint32_t hash = hash_ngram(neurons, length);

    // Search for existing N-gram
    ngram_node_t* node = detector->ngram_table[hash];
    while (node) {
        if (node->pattern.length == length) {
            bool match = true;
            for (uint32_t i = 0; i < length; i++) {
                if (node->pattern.neurons[i] != neurons[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // Update existing
                node->pattern.count++;
                float alpha = 0.1f;
                node->pattern.avg_interval_ms =
                    (1.0f - alpha) * node->pattern.avg_interval_ms + alpha * interval_ms;
                return;
            }
        }
        node = node->next;
    }

    // Create new N-gram
    ngram_node_t* new_node = (ngram_node_t*)nimcp_calloc(1, sizeof(ngram_node_t));
    if (!new_node) return;

    new_node->pattern.length = length;
    for (uint32_t i = 0; i < length; i++) {
        new_node->pattern.neurons[i] = neurons[i];
    }
    new_node->pattern.count = 1;
    new_node->pattern.avg_interval_ms = interval_ms;

    new_node->next = detector->ngram_table[hash];
    detector->ngram_table[hash] = new_node;
    detector->total_ngrams++;
}

static void extract_ngrams(sequence_detector_t* detector) {
    if (!detector->config.enable_ngram_learning || detector->buffer.count < 2) {
        return;
    }

    // Extract N-grams from recent spikes
    for (uint32_t n = 2; n <= detector->config.max_ngram; n++) {
        if (detector->buffer.count < n) continue;

        for (uint32_t i = 0; i + n <= detector->buffer.count; i++) {
            uint32_t neurons[SEQUENCE_MAX_NGRAM];
            double start_time = 0.0;
            double end_time = 0.0;

            for (uint32_t j = 0; j < n; j++) {
                uint32_t idx = (detector->buffer.head + detector->buffer.capacity -
                              detector->buffer.count + i + j) % detector->buffer.capacity;
                neurons[j] = detector->buffer.spikes[idx].neuron_id;

                if (j == 0) start_time = detector->buffer.spikes[idx].timestamp_ms;
                if (j == n - 1) end_time = detector->buffer.spikes[idx].timestamp_ms;
            }

            float interval = (float)(end_time - start_time) / (float)(n - 1);
            add_ngram(detector, neurons, n, interval);
        }
    }
}

static float match_template(const spike_buffer_t* buffer,
                           const sequence_template_t* seq_template,
                           float tolerance_ms,
                           bool* is_forward, bool* is_backward,
                           float* compression) {
    if (!buffer || !seq_template || buffer->count < seq_template->length) {
        return 0.0f;
    }

    uint32_t matched = 0;
    double first_spike_time = 0.0;
    double last_spike_time = 0.0;

    // Try forward match
    for (uint32_t i = 0; i < seq_template->length; i++) {
        uint32_t target_neuron = seq_template->elements[i].neuron_id;
        float target_time = seq_template->elements[i].relative_time_ms;

        // Search in buffer for matching spike
        bool found = false;
        for (uint32_t j = 0; j < buffer->count; j++) {
            uint32_t idx = (buffer->head + buffer->capacity - buffer->count + j) %
                          buffer->capacity;

            if (buffer->spikes[idx].neuron_id == target_neuron) {
                if (i == 0) {
                    first_spike_time = buffer->spikes[idx].timestamp_ms;
                    found = true;
                    matched++;
                } else {
                    double expected_time = first_spike_time + target_time;
                    double actual_time = buffer->spikes[idx].timestamp_ms;
                    double error = fabs(actual_time - expected_time);

                    if (error <= tolerance_ms) {
                        found = true;
                        matched++;
                        if (i == seq_template->length - 1) {
                            last_spike_time = actual_time;
                        }
                    }
                }
                break;
            }
        }

        if (!found) break;
    }

    float strength = (float)matched / (float)seq_template->length;

    if (strength > 0.5f) {
        *is_forward = true;
        *is_backward = false;

        if (matched >= 2) {
            double actual_duration = last_spike_time - first_spike_time;
            *compression = (float)(actual_duration / (seq_template->duration_ms + 1e-6));
        } else {
            *compression = 1.0f;
        }
    }

    return strength;
}

// ============================================================================
// PUBLIC API
// ============================================================================

sequence_detector_config_t sequence_detector_default_config(void) {
    sequence_detector_config_t config;
    config.max_templates = SEQUENCE_MAX_TEMPLATES;
    config.max_sequence_length = SEQUENCE_MAX_LENGTH;
    config.temporal_tolerance_ms = SEQUENCE_TEMPORAL_TOLERANCE_MS;
    config.min_strength_threshold = SEQUENCE_MIN_STRENGTH;
    config.max_ngram = SEQUENCE_MAX_NGRAM;
    config.enable_replay_detection = true;
    config.enable_ngram_learning = true;
    config.enable_compression = true;
    return config;
}

sequence_detector_t* sequence_detector_create(const sequence_detector_config_t* config) {
    if (!config || config->max_templates == 0 || config->max_sequence_length == 0) {
        return NULL;
    }

    sequence_detector_t* detector = (sequence_detector_t*)nimcp_calloc(1, sizeof(sequence_detector_t));
    if (!detector) return NULL;

    detector->config = *config;

    // Initialize spike buffer
    if (!init_spike_buffer(&detector->buffer, SPIKE_BUFFER_SIZE)) {
        sequence_detector_destroy(detector);
        return NULL;
    }

    // Allocate seq_template storage
    detector->templates = (sequence_template_t*)nimcp_calloc(config->max_templates,
                                                       sizeof(sequence_template_t));
    if (!detector->templates) {
        sequence_detector_destroy(detector);
        return NULL;
    }

    detector->num_templates = 0;
    detector->next_template_id = 1;

    // Initialize N-gram table
    memset(detector->ngram_table, 0, sizeof(detector->ngram_table));
    detector->total_ngrams = 0;

    // Initialize statistics
    detector->total_detections = 0;
    detector->sum_strength = 0.0;

    return detector;
}

void sequence_detector_destroy(sequence_detector_t* detector) {
    if (!detector) return;

    free_spike_buffer(&detector->buffer);

    // Free templates
    if (detector->templates) {
        for (uint32_t i = 0; i < detector->num_templates; i++) {
            nimcp_free(detector->templates[i].elements);
        }
        nimcp_free(detector->templates);
    }

    // Free N-grams
    for (uint32_t i = 0; i < NGRAM_HASH_SIZE; i++) {
        ngram_node_t* node = detector->ngram_table[i];
        while (node) {
            ngram_node_t* next = node->next;
            nimcp_free(node);
            node = next;
        }
    }

    nimcp_free(detector);
}

bool sequence_detector_add_spike(sequence_detector_t* detector,
                                  uint32_t neuron_id,
                                  double timestamp_ms) {
    if (!detector) return false;

    buffer_add_spike(&detector->buffer, neuron_id, timestamp_ms);

    // Extract N-grams periodically
    if (detector->config.enable_ngram_learning && detector->buffer.count >= 2) {
        extract_ngrams(detector);
    }

    return true;
}

bool sequence_detector_learn_template(sequence_detector_t* detector,
                                       const sequence_element_t* elements,
                                       uint32_t length,
                                       uint32_t* template_id) {
    if (!detector || !elements || length == 0 ||
        length > detector->config.max_sequence_length ||
        detector->num_templates >= detector->config.max_templates) {
        return false;
    }

    sequence_template_t* tmpl = &detector->templates[detector->num_templates];

    tmpl->elements = (sequence_element_t*)nimcp_malloc(length * sizeof(sequence_element_t));
    if (!tmpl->elements) return false;

    memcpy(tmpl->elements, elements, length * sizeof(sequence_element_t));
    tmpl->length = length;
    tmpl->duration_ms = elements[length - 1].relative_time_ms;
    tmpl->observations = 0;
    tmpl->avg_strength = 0.0f;
    tmpl->template_id = detector->next_template_id++;

    if (template_id) *template_id = tmpl->template_id;

    detector->num_templates++;

    return true;
}

bool sequence_detector_detect(sequence_detector_t* detector,
                               sequence_detection_t* detections,
                               uint32_t max_detections,
                               uint32_t* num_detected) {
    if (!detector || !detections || !num_detected || max_detections == 0) {
        return false;
    }

    *num_detected = 0;

    // Match against all templates
    for (uint32_t i = 0; i < detector->num_templates && *num_detected < max_detections; i++) {
        bool is_forward = false;
        bool is_backward = false;
        float compression = 1.0f;

        float strength = match_template(&detector->buffer,
                                       &detector->templates[i],
                                       detector->config.temporal_tolerance_ms,
                                       &is_forward, &is_backward,
                                       &compression);

        if (strength >= detector->config.min_strength_threshold) {
            sequence_detection_t* det = &detections[*num_detected];

            det->template_id = detector->templates[i].template_id;
            det->strength = strength;
            det->is_forward = is_forward;
            det->is_backward = is_backward;
            det->compression_factor = compression;
            det->matched_elements = (uint32_t)(strength * detector->templates[i].length);
            det->total_elements = detector->templates[i].length;

            // Estimate start/end times from buffer
            if (detector->buffer.count > 0) {
                uint32_t first = (detector->buffer.head + detector->buffer.capacity -
                                detector->buffer.count) % detector->buffer.capacity;
                uint32_t last = (detector->buffer.head + detector->buffer.capacity - 1) %
                               detector->buffer.capacity;

                det->start_time_ms = detector->buffer.spikes[first].timestamp_ms;
                det->end_time_ms = detector->buffer.spikes[last].timestamp_ms;
            }

            // Update seq_template statistics
            detector->templates[i].observations++;
            float alpha = 0.1f;
            detector->templates[i].avg_strength =
                (1.0f - alpha) * detector->templates[i].avg_strength + alpha * strength;

            (*num_detected)++;
            detector->total_detections++;
            detector->sum_strength += strength;
        }
    }

    return true;
}

bool sequence_detector_get_template(const sequence_detector_t* detector,
                                     uint32_t template_id,
                                     sequence_template_t* seq_template) {
    if (!detector || !seq_template) return false;

    for (uint32_t i = 0; i < detector->num_templates; i++) {
        if (detector->templates[i].template_id == template_id) {
            *seq_template = detector->templates[i];
            return true;
        }
    }

    return false;
}

bool sequence_detector_get_ngrams(const sequence_detector_t* detector,
                                   ngram_pattern_t* ngrams,
                                   uint32_t max_ngrams,
                                   uint32_t* num_ngrams) {
    if (!detector || !ngrams || !num_ngrams || max_ngrams == 0) {
        return false;
    }

    *num_ngrams = 0;

    // Collect all N-grams
    for (uint32_t i = 0; i < NGRAM_HASH_SIZE && *num_ngrams < max_ngrams; i++) {
        ngram_node_t* node = detector->ngram_table[i];
        while (node && *num_ngrams < max_ngrams) {
            ngrams[*num_ngrams] = node->pattern;
            (*num_ngrams)++;
            node = node->next;
        }
    }

    // Sort by count (simple bubble sort for small arrays)
    for (uint32_t i = 0; i < *num_ngrams; i++) {
        for (uint32_t j = i + 1; j < *num_ngrams; j++) {
            if (ngrams[j].count > ngrams[i].count) {
                ngram_pattern_t temp = ngrams[i];
                ngrams[i] = ngrams[j];
                ngrams[j] = temp;
            }
        }
    }

    return true;
}

void sequence_detector_reset(sequence_detector_t* detector) {
    if (!detector) return;

    detector->buffer.count = 0;
    detector->buffer.head = 0;
}

void sequence_detector_clear_templates(sequence_detector_t* detector) {
    if (!detector) return;

    for (uint32_t i = 0; i < detector->num_templates; i++) {
        nimcp_free(detector->templates[i].elements);
    }

    detector->num_templates = 0;
    detector->next_template_id = 1;
}

bool sequence_detector_get_stats(const sequence_detector_t* detector,
                                  uint32_t* num_templates,
                                  uint64_t* total_detections,
                                  float* avg_strength) {
    if (!detector) return false;

    if (num_templates) *num_templates = detector->num_templates;
    if (total_detections) *total_detections = detector->total_detections;

    if (avg_strength) {
        *avg_strength = (detector->total_detections > 0) ?
                       (float)(detector->sum_strength / detector->total_detections) : 0.0f;
    }

    return true;
}
