//=============================================================================
// nimcp_snn_language_bridge.c - SNN ↔ Language Spike-Driven Bridge
//=============================================================================
/**
 * @file nimcp_snn_language_bridge.c
 * @brief Implementation of spike-to-word decoder and STDP word-concept binding
 *
 * Phases 1-7 of the SNN-Language-Creative integration plan.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "language/nimcp_grounded_language.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define LOG_MODULE "SNN_LANG_BRIDGE"

//=============================================================================
// Internal: Sparse binding hash map
//=============================================================================

#define BINDING_HASH_BUCKETS 8192
#define BINDING_HASH_MASK    (BINDING_HASH_BUCKETS - 1)

typedef struct binding_node {
    snn_lang_binding_t binding;
    struct binding_node* next;
} binding_node_t;

//=============================================================================
// Internal: Population registration
//=============================================================================

typedef struct {
    uint64_t concept_id;
    bool     registered;
    float    last_spike_ms;
    float    activation;       // Accumulated activation
} concept_pop_info_t;

typedef struct {
    char     word_form[64];
    bool     registered;
    float    last_spike_ms;
    float    activation;
} word_pop_info_t;

//=============================================================================
// Bridge structure
//=============================================================================

struct snn_language_bridge {
    uint32_t magic;
    snn_lang_config_t config;

    // Population registrations
    concept_pop_info_t* concept_pops;
    uint32_t num_concept_pops;
    uint32_t concept_pops_capacity;

    word_pop_info_t* word_pops;
    uint32_t num_word_pops;
    uint32_t word_pops_capacity;

    // Sparse binding hash map
    binding_node_t* binding_buckets[BINDING_HASH_BUCKETS];
    uint32_t num_bindings;

    // Connected subsystems
    struct grounded_language* grounded_lang;
    struct imagination_snn_bridge* imagination;
    struct curiosity_snn_bridge* curiosity;
    struct neuromodulator_system_struct* neuromod;

    // Current time
    float current_time_ms;

    // Statistics
    snn_lang_stats_t stats;
};

//=============================================================================
// Hash function for binding lookup
//=============================================================================

static inline uint32_t binding_hash(uint32_t concept_pop, uint32_t word_pop)
{
    uint32_t h = concept_pop * 2654435761u + word_pop * 40503u;
    return h & BINDING_HASH_MASK;
}

static binding_node_t* binding_find(snn_language_bridge_t* bridge,
                                     uint32_t concept_pop, uint32_t word_pop)
{
    uint32_t bucket = binding_hash(concept_pop, word_pop);
    binding_node_t* node = bridge->binding_buckets[bucket];
    while (node) {
        if (node->binding.concept_pop == concept_pop &&
            node->binding.word_pop == word_pop) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

static binding_node_t* binding_insert(snn_language_bridge_t* bridge,
                                       uint32_t concept_pop, uint32_t word_pop,
                                       float initial_weight)
{
    binding_node_t* existing = binding_find(bridge, concept_pop, word_pop);
    if (existing) {
        existing->binding.weight = fmaxf(existing->binding.weight, initial_weight);
        return existing;
    }

    binding_node_t* node = nimcp_calloc(1, sizeof(binding_node_t));
    if (!node) return NULL;

    node->binding.concept_pop = concept_pop;
    node->binding.word_pop = word_pop;
    node->binding.weight = initial_weight;
    node->binding.last_pre_spike_ms = -1000.0f;
    node->binding.last_post_spike_ms = -1000.0f;

    uint32_t bucket = binding_hash(concept_pop, word_pop);
    node->next = bridge->binding_buckets[bucket];
    bridge->binding_buckets[bucket] = node;
    bridge->num_bindings++;

    return node;
}

//=============================================================================
// Configuration
//=============================================================================

snn_lang_config_t snn_lang_config_default(void)
{
    snn_lang_config_t config = {
        .max_concept_pops = 512,
        .max_word_pops = 4096,
        .neurons_per_pop = SNN_LANG_NEURONS_PER_POP,
        .stdp_tau_plus = SNN_LANG_DEFAULT_STDP_TAU,
        .stdp_tau_minus = SNN_LANG_DEFAULT_STDP_TAU,
        .stdp_a_plus = SNN_LANG_DEFAULT_STDP_A_PLUS,
        .stdp_a_minus = SNN_LANG_DEFAULT_STDP_A_MINUS,
        .stdp_learning_rate = 0.01f,
        .binding_w_max = SNN_LANG_BINDING_W_MAX,
        .decode_window_ms = SNN_LANG_DECODE_WINDOW_MS,
        .decay_rate = SNN_LANG_DECAY_RATE,
        .spike_blend = SNN_LANG_SPIKE_BLEND_DEFAULT,
        .enable_da_modulation = true,
        .da_modulation_gain = 50.0f,
        .enable_imagination = true,
        .enable_curiosity = true,
        .enable_sleep_consolidation = true,
        .prune_threshold = 0.005f
    };
    return config;
}

//=============================================================================
// Lifecycle
//=============================================================================

snn_language_bridge_t* snn_language_bridge_create(const snn_lang_config_t* config)
{
    if (!config) return NULL;

    snn_language_bridge_t* bridge = nimcp_calloc(1, sizeof(snn_language_bridge_t));
    if (!bridge) return NULL;

    bridge->magic = SNN_LANG_MAGIC;
    bridge->config = *config;

    // Allocate concept populations
    bridge->concept_pops_capacity = config->max_concept_pops;
    bridge->concept_pops = nimcp_calloc(bridge->concept_pops_capacity,
                                        sizeof(concept_pop_info_t));
    if (!bridge->concept_pops) {
        nimcp_free(bridge);
        return NULL;
    }

    // Allocate word populations
    bridge->word_pops_capacity = config->max_word_pops;
    bridge->word_pops = nimcp_calloc(bridge->word_pops_capacity,
                                     sizeof(word_pop_info_t));
    if (!bridge->word_pops) {
        nimcp_free(bridge->concept_pops);
        nimcp_free(bridge);
        return NULL;
    }

    LOG_INFO(LOG_MODULE, "SNN language bridge created (concepts=%u, words=%u, blend=%.2f)",
             config->max_concept_pops, config->max_word_pops, config->spike_blend);

    return bridge;
}

void snn_language_bridge_destroy(snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return;

    // Free binding hash map
    for (uint32_t i = 0; i < BINDING_HASH_BUCKETS; i++) {
        binding_node_t* node = bridge->binding_buckets[i];
        while (node) {
            binding_node_t* next = node->next;
            nimcp_free(node);
            node = next;
        }
    }

    nimcp_free(bridge->concept_pops);
    nimcp_free(bridge->word_pops);

    bridge->magic = 0;
    nimcp_free(bridge);
}

int snn_language_bridge_reset(snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    // Reset activations, keep bindings and registrations
    for (uint32_t i = 0; i < bridge->num_concept_pops; i++) {
        bridge->concept_pops[i].activation = 0.0f;
        bridge->concept_pops[i].last_spike_ms = -1000.0f;
    }
    for (uint32_t i = 0; i < bridge->num_word_pops; i++) {
        bridge->word_pops[i].activation = 0.0f;
        bridge->word_pops[i].last_spike_ms = -1000.0f;
    }

    bridge->current_time_ms = 0.0f;
    return 0;
}

//=============================================================================
// Connection
//=============================================================================

int snn_language_bridge_connect_grounded(snn_language_bridge_t* bridge,
                                          struct grounded_language* gl)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    bridge->grounded_lang = gl;
    return 0;
}

int snn_language_bridge_connect_imagination(snn_language_bridge_t* bridge,
                                             struct imagination_snn_bridge* imagination)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    bridge->imagination = imagination;
    return 0;
}

int snn_language_bridge_connect_curiosity(snn_language_bridge_t* bridge,
                                           struct curiosity_snn_bridge* curiosity)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    bridge->curiosity = curiosity;
    return 0;
}

int snn_language_bridge_connect_neuromod(snn_language_bridge_t* bridge,
                                          struct neuromodulator_system_struct* neuromod)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    bridge->neuromod = neuromod;
    return 0;
}

//=============================================================================
// Phase 1: Population Registration + Spike-to-Word Decoding
//=============================================================================

int snn_language_bridge_register_concept(snn_language_bridge_t* bridge,
                                          uint32_t concept_pop,
                                          uint64_t concept_id)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (concept_pop >= bridge->concept_pops_capacity) return -1;

    bridge->concept_pops[concept_pop].concept_id = concept_id;
    bridge->concept_pops[concept_pop].registered = true;
    bridge->concept_pops[concept_pop].last_spike_ms = -1000.0f;
    if (concept_pop >= bridge->num_concept_pops) {
        bridge->num_concept_pops = concept_pop + 1;
    }
    return 0;
}

int snn_language_bridge_register_word(snn_language_bridge_t* bridge,
                                       uint32_t word_pop,
                                       const char* word_form)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !word_form) return -1;
    if (word_pop >= bridge->word_pops_capacity) return -1;

    strncpy(bridge->word_pops[word_pop].word_form, word_form, 63);
    bridge->word_pops[word_pop].word_form[63] = '\0';
    bridge->word_pops[word_pop].registered = true;
    bridge->word_pops[word_pop].last_spike_ms = -1000.0f;
    if (word_pop >= bridge->num_word_pops) {
        bridge->num_word_pops = word_pop + 1;
    }
    return 0;
}

int snn_language_bridge_decode_spikes(snn_language_bridge_t* bridge,
                                       const float* concept_rates,
                                       uint32_t num_concept_pops,
                                       snn_lang_word_result_t* results,
                                       uint32_t max_results,
                                       uint32_t* num_results)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !concept_rates ||
        !results || !num_results) {
        return -1;
    }

    bridge->stats.total_decode_calls++;
    *num_results = 0;

    // Compute word activations through binding matrix
    // word_activation[w] = sum_c(concept_rates[c] * binding_weight[c,w])
    uint32_t n_words = bridge->num_word_pops;
    float* word_acts = nimcp_calloc(n_words, sizeof(float));
    if (!word_acts) return -1;

    // Iterate all bindings (sparse traversal)
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            uint32_t c = node->binding.concept_pop;
            uint32_t w = node->binding.word_pop;
            if (c < num_concept_pops && w < n_words) {
                word_acts[w] += concept_rates[c] * node->binding.weight;
            }
            node = node->next;
        }
    }

    // Find top-k words by activation
    for (uint32_t k = 0; k < max_results && k < n_words; k++) {
        float best_act = -FLT_MAX;
        uint32_t best_w = 0;
        bool found = false;

        for (uint32_t w = 0; w < n_words; w++) {
            if (!bridge->word_pops[w].registered) continue;
            if (word_acts[w] > best_act) {
                // Check not already in results
                bool duplicate = false;
                for (uint32_t j = 0; j < *num_results; j++) {
                    if (results[j].word_pop == w) { duplicate = true; break; }
                }
                if (!duplicate) {
                    best_act = word_acts[w];
                    best_w = w;
                    found = true;
                }
            }
        }

        if (!found || best_act <= 0.0f) break;

        results[*num_results].word_pop = best_w;
        results[*num_results].word_form = bridge->word_pops[best_w].word_form;
        results[*num_results].activation = best_act;
        // Normalize confidence: activation relative to sum
        float sum = 0.0f;
        for (uint32_t w = 0; w < n_words; w++) {
            if (word_acts[w] > 0.0f) sum += word_acts[w];
        }
        results[*num_results].confidence = (sum > 0.0f) ? best_act / sum : 0.0f;
        (*num_results)++;
    }

    nimcp_free(word_acts);
    return 0;
}

int snn_language_bridge_encode_word(snn_language_bridge_t* bridge,
                                     uint32_t word_pop,
                                     float* concept_activations,
                                     uint32_t num_concept_pops)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !concept_activations) return -1;
    if (word_pop >= bridge->num_word_pops) return -1;

    bridge->stats.total_encode_calls++;
    memset(concept_activations, 0, num_concept_pops * sizeof(float));

    // Reverse lookup: concept_activation[c] = binding_weight[c, word_pop]
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            if (node->binding.word_pop == word_pop &&
                node->binding.concept_pop < num_concept_pops) {
                concept_activations[node->binding.concept_pop] = node->binding.weight;
            }
            node = node->next;
        }
    }

    return 0;
}

//=============================================================================
// Phase 2: STDP-Driven Word-Concept Binding
//=============================================================================

int snn_language_bridge_concept_spike(snn_language_bridge_t* bridge,
                                       uint32_t concept_pop,
                                       float spike_time_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (concept_pop >= bridge->num_concept_pops) return -1;

    bridge->concept_pops[concept_pop].last_spike_ms = spike_time_ms;
    bridge->concept_pops[concept_pop].activation += 1.0f;
    return 0;
}

int snn_language_bridge_word_spike(snn_language_bridge_t* bridge,
                                    uint32_t word_pop,
                                    float spike_time_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (word_pop >= bridge->num_word_pops) return -1;

    bridge->word_pops[word_pop].last_spike_ms = spike_time_ms;
    bridge->word_pops[word_pop].activation += 1.0f;
    return 0;
}

int snn_language_bridge_apply_stdp(snn_language_bridge_t* bridge,
                                    float current_time_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    bridge->current_time_ms = current_time_ms;
    float tau_plus = bridge->config.stdp_tau_plus;
    float tau_minus = bridge->config.stdp_tau_minus;
    float a_plus = bridge->config.stdp_a_plus;
    float a_minus = bridge->config.stdp_a_minus;
    float lr = bridge->config.stdp_learning_rate;
    float w_max = bridge->config.binding_w_max;

    // Iterate all bindings
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            snn_lang_binding_t* b = &node->binding;
            uint32_t c = b->concept_pop;
            uint32_t w = b->word_pop;

            float t_pre = bridge->concept_pops[c].last_spike_ms;
            float t_post = bridge->word_pops[w].last_spike_ms;

            // Only process if both spiked recently (within 2x window)
            float window = fmaxf(tau_plus, tau_minus) * 3.0f;
            if (t_pre < current_time_ms - window &&
                t_post < current_time_ms - window) {
                // Decay eligibility trace
                float dt = current_time_ms - fmaxf(t_pre, t_post);
                b->eligibility *= expf(-dt / (tau_plus * 5.0f));
                node = node->next;
                continue;
            }

            // Update traces
            float dt_pre = current_time_ms - t_pre;
            float dt_post = current_time_ms - t_post;
            b->pre_trace = (dt_pre < window) ?
                expf(-dt_pre / tau_plus) : 0.0f;
            b->post_trace = (dt_post < window) ?
                expf(-dt_post / tau_minus) : 0.0f;

            // STDP weight update
            float dw = 0.0f;
            float dt_spike = t_post - t_pre;

            if (dt_spike > 0.0f && dt_spike < window) {
                // Post after pre → LTP (concept before word → strengthen binding)
                dw = a_plus * expf(-dt_spike / tau_plus);
                b->ltp_count++;
                bridge->stats.total_ltp_events++;
            } else if (dt_spike < 0.0f && dt_spike > -window) {
                // Pre after post → LTD (word before concept → weaken binding)
                dw = -a_minus * expf(dt_spike / tau_minus);
                b->ltd_count++;
                bridge->stats.total_ltd_events++;
            }

            if (dw != 0.0f) {
                // Update eligibility trace
                b->eligibility += fabsf(dw);

                // Apply weight change with learning rate
                float weight_change = lr * dw;

                // Soft bounds: weight-dependent scaling
                if (weight_change > 0.0f) {
                    weight_change *= (w_max - b->weight) / w_max;
                } else {
                    weight_change *= b->weight / w_max;
                }

                b->weight += weight_change;
                b->weight = fmaxf(SNN_LANG_BINDING_W_MIN,
                            fminf(w_max, b->weight));

                bridge->stats.total_stdp_updates++;
            }

            // Decay activations
            bridge->concept_pops[c].activation *= bridge->config.decay_rate;
            bridge->word_pops[w].activation *= bridge->config.decay_rate;

            node = node->next;
        }
    }

    return 0;
}

int snn_language_bridge_bind(snn_language_bridge_t* bridge,
                              uint32_t concept_pop, uint32_t word_pop,
                              float initial_weight)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (concept_pop >= bridge->num_concept_pops) return -1;
    if (word_pop >= bridge->num_word_pops) return -1;

    binding_node_t* node = binding_insert(bridge, concept_pop, word_pop,
                                          initial_weight);
    return node ? 0 : -1;
}

int snn_language_bridge_prune(snn_language_bridge_t* bridge, float threshold)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    uint32_t pruned = 0;

    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t** pp = &bridge->binding_buckets[bucket];
        while (*pp) {
            if ((*pp)->binding.weight < threshold) {
                binding_node_t* dead = *pp;
                *pp = dead->next;
                nimcp_free(dead);
                bridge->num_bindings--;
                pruned++;
            } else {
                pp = &(*pp)->next;
            }
        }
    }

    bridge->stats.bindings_pruned += pruned;
    return (int)pruned;
}

//=============================================================================
// Phase 3: Spike-Driven Language Production (Broca pathway)
//=============================================================================

int snn_language_bridge_produce_word(snn_language_bridge_t* bridge,
                                      const float* concept_activations,
                                      uint32_t num_concepts,
                                      snn_lang_word_result_t* result)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC ||
        !concept_activations || !result) {
        return -1;
    }

    snn_lang_word_result_t top;
    uint32_t num_out = 0;
    int rc = snn_language_bridge_decode_spikes(bridge, concept_activations,
                                               num_concepts, &top, 1, &num_out);
    if (rc != 0 || num_out == 0) return -1;

    *result = top;
    return 0;
}

int snn_language_bridge_produce(snn_language_bridge_t* bridge,
                                 const float* semantic_intent,
                                 uint32_t intent_dim,
                                 snn_lang_production_result_t* result)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC ||
        !semantic_intent || !result) {
        return -1;
    }

    bridge->stats.total_produce_calls++;
    memset(result, 0, sizeof(*result));

    // Map semantic intent to concept activations
    // Use first num_concept_pops dimensions of intent (or zero-pad)
    uint32_t n_concepts = bridge->num_concept_pops;
    float* concept_acts = nimcp_calloc(n_concepts, sizeof(float));
    if (!concept_acts) return -1;

    uint32_t copy_dim = (intent_dim < n_concepts) ? intent_dim : n_concepts;
    for (uint32_t i = 0; i < copy_dim; i++) {
        concept_acts[i] = fmaxf(0.0f, semantic_intent[i]); // ReLU activation
    }

    // Iterative word production with refractory inhibition
    char text_buf[2048] = {0};
    uint32_t text_pos = 0;
    uint32_t max_words = 32;
    float total_confidence = 0.0f;
    uint32_t word_count = 0;

    // Track used words for refractory
    uint32_t* used_words = nimcp_calloc(max_words, sizeof(uint32_t));
    if (!used_words) {
        nimcp_free(concept_acts);
        return -1;
    }

    for (uint32_t w = 0; w < max_words; w++) {
        // Get top word
        snn_lang_word_result_t word_result;
        uint32_t num_out = 0;
        snn_lang_word_result_t top5[5];
        int rc = snn_language_bridge_decode_spikes(bridge, concept_acts,
                                                    n_concepts, top5, 5, &num_out);
        if (rc != 0 || num_out == 0) break;

        // Find best word not already used
        bool found = false;
        for (uint32_t k = 0; k < num_out; k++) {
            bool refractory = false;
            for (uint32_t u = 0; u < word_count; u++) {
                if (used_words[u] == top5[k].word_pop) {
                    refractory = true;
                    break;
                }
            }
            if (!refractory) {
                word_result = top5[k];
                found = true;
                break;
            }
        }
        if (!found) break;

        // Stop if confidence too low
        if (word_result.confidence < 0.01f && word_count > 0) break;

        // Append word to text
        const char* word = word_result.word_form;
        size_t wlen = strlen(word);
        if (text_pos + wlen + 2 >= sizeof(text_buf)) break;

        if (text_pos > 0) {
            text_buf[text_pos++] = ' ';
        }
        memcpy(text_buf + text_pos, word, wlen);
        text_pos += wlen;

        used_words[word_count] = word_result.word_pop;
        total_confidence += word_result.confidence;
        word_count++;

        // Feed selected word back into concept activations (recurrent)
        float* word_concepts = nimcp_calloc(n_concepts, sizeof(float));
        if (word_concepts) {
            snn_language_bridge_encode_word(bridge, word_result.word_pop,
                                            word_concepts, n_concepts);
            // Blend: 70% original intent, 30% word feedback
            for (uint32_t c = 0; c < n_concepts; c++) {
                concept_acts[c] = 0.7f * concept_acts[c] + 0.3f * word_concepts[c];
            }
            nimcp_free(word_concepts);
        }
    }

    nimcp_free(used_words);
    nimcp_free(concept_acts);

    if (word_count == 0) return -1;

    text_buf[text_pos] = '\0';
    result->text = nimcp_malloc(text_pos + 1);
    if (!result->text) return -1;
    memcpy(result->text, text_buf, text_pos + 1);

    result->word_count = word_count;
    result->spike_confidence = total_confidence / word_count;
    result->fluency = fminf(1.0f, (float)word_count / 8.0f);
    result->creativity = 0.0f;  // Set by creative_produce

    return 0;
}

void snn_lang_production_result_cleanup(snn_lang_production_result_t* result)
{
    if (!result) return;
    if (result->text) {
        nimcp_free(result->text);
        result->text = NULL;
    }
}

//=============================================================================
// Phase 4: Spike-Driven Comprehension (Wernicke pathway)
//=============================================================================

int snn_language_bridge_comprehend(snn_language_bridge_t* bridge,
                                    const char* text,
                                    float* concept_activations,
                                    uint32_t max_concepts,
                                    uint32_t* num_activated,
                                    float* comprehension_confidence)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC ||
        !text || !concept_activations || !num_activated) {
        return -1;
    }

    bridge->stats.total_comprehend_calls++;
    memset(concept_activations, 0, max_concepts * sizeof(float));
    *num_activated = 0;

    // Tokenize text into words (simple whitespace split)
    char text_copy[2048];
    strncpy(text_copy, text, sizeof(text_copy) - 1);
    text_copy[sizeof(text_copy) - 1] = '\0';

    float total_activation = 0.0f;
    uint32_t word_count = 0;

    char* saveptr = NULL;
    char* token = strtok_r(text_copy, " \t\n,.;:!?\"'()-", &saveptr);

    while (token) {
        // Find word population
        uint32_t word_pop = UINT32_MAX;
        for (uint32_t w = 0; w < bridge->num_word_pops; w++) {
            if (bridge->word_pops[w].registered &&
                strcasecmp(bridge->word_pops[w].word_form, token) == 0) {
                word_pop = w;
                break;
            }
        }

        if (word_pop != UINT32_MAX) {
            // Cascade word activation through transposed binding matrix
            float* word_activations = nimcp_calloc(1, sizeof(float));
            if (!word_activations) { token = strtok_r(NULL, " \t\n,.;:!?\"'()-", &saveptr); continue; }
            nimcp_free(word_activations);

            // Direct reverse lookup
            for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
                binding_node_t* node = bridge->binding_buckets[bucket];
                while (node) {
                    if (node->binding.word_pop == word_pop &&
                        node->binding.concept_pop < max_concepts) {
                        concept_activations[node->binding.concept_pop] +=
                            node->binding.weight;
                        total_activation += node->binding.weight;
                    }
                    node = node->next;
                }
            }
            word_count++;
        }

        token = strtok_r(NULL, " \t\n,.;:!?\"'()-", &saveptr);
    }

    // Count activated concepts
    for (uint32_t c = 0; c < max_concepts; c++) {
        if (concept_activations[c] > 0.01f) {
            (*num_activated)++;
        }
    }

    if (comprehension_confidence) {
        *comprehension_confidence = (word_count > 0) ?
            total_activation / (float)word_count : 0.0f;
    }

    return 0;
}

//=============================================================================
// Phase 5: Creative/Imagination Integration
//=============================================================================

int snn_language_bridge_creative_produce(snn_language_bridge_t* bridge,
                                          const float* imagination_activations,
                                          uint32_t num_dims,
                                          float creativity_level,
                                          snn_lang_production_result_t* result)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC ||
        !imagination_activations || !result) {
        return -1;
    }

    bridge->stats.imagination_contributions++;

    // Map imagination activations to concept space
    // Imagination dims are creativity/vividness/coherence signals
    // Scale them by creativity_level to modulate word selection
    uint32_t n_concepts = bridge->num_concept_pops;
    float* concept_acts = nimcp_calloc(n_concepts, sizeof(float));
    if (!concept_acts) return -1;

    // Distribute imagination activations across concept populations
    // with creativity-scaled noise for exploration
    uint32_t copy_dim = (num_dims < n_concepts) ? num_dims : n_concepts;
    for (uint32_t i = 0; i < copy_dim; i++) {
        concept_acts[i] = imagination_activations[i] * (1.0f + creativity_level);
    }

    // Add stochastic exploration proportional to creativity
    if (creativity_level > 0.1f) {
        for (uint32_t i = 0; i < n_concepts; i++) {
            // Simple hash-based pseudo-random noise
            uint32_t h = (i * 2654435761u + bridge->stats.imagination_contributions * 40503u);
            float noise = ((float)(h & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
            concept_acts[i] += noise * creativity_level * 0.1f;
            if (concept_acts[i] < 0.0f) concept_acts[i] = 0.0f;
        }
    }

    // Produce via standard spike cascade
    int rc = snn_language_bridge_produce(bridge, concept_acts, n_concepts, result);
    if (rc == 0) {
        result->creativity = creativity_level;
    }

    nimcp_free(concept_acts);
    return rc;
}

int snn_language_bridge_curiosity_modulate(snn_language_bridge_t* bridge,
                                            float novelty_level,
                                            float exploration_drive)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    bridge->stats.curiosity_contributions++;

    // Boost binding learning rate for novel stimuli
    // Higher novelty → faster binding formation
    float novelty_boost = 1.0f + novelty_level * 0.5f;
    bridge->config.stdp_learning_rate *= novelty_boost;

    // Clamp to reasonable range
    if (bridge->config.stdp_learning_rate > 0.1f) {
        bridge->config.stdp_learning_rate = 0.1f;
    }

    // Exploration drive lowers the prune threshold (keep more bindings)
    if (exploration_drive > 0.5f) {
        bridge->config.prune_threshold *= 0.5f;
        if (bridge->config.prune_threshold < 0.001f) {
            bridge->config.prune_threshold = 0.001f;
        }
    }

    return 0;
}

//=============================================================================
// Phase 6: Sleep Consolidation
//=============================================================================

int snn_language_bridge_sleep_consolidate(snn_language_bridge_t* bridge,
                                           float consolidation_strength)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    bridge->stats.sleep_consolidation_cycles++;

    // Strengthen high-weight bindings, weaken low-weight ones
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            snn_lang_binding_t* b = &node->binding;

            // Replay: bindings with high eligibility get strengthened
            if (b->eligibility > 0.1f) {
                float replay_boost = consolidation_strength * b->eligibility * 0.1f;
                b->weight += replay_boost;
                if (b->weight > bridge->config.binding_w_max) {
                    b->weight = bridge->config.binding_w_max;
                }
            }

            // Decay eligibility traces during sleep
            b->eligibility *= 0.5f;

            node = node->next;
        }
    }

    // Prune weak bindings
    snn_language_bridge_prune(bridge, bridge->config.prune_threshold);

    return 0;
}

//=============================================================================
// Statistics & Introspection
//=============================================================================

int snn_language_bridge_get_stats(const snn_language_bridge_t* bridge,
                                   snn_lang_stats_t* stats)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !stats) return -1;

    *stats = bridge->stats;
    stats->active_bindings = bridge->num_bindings;
    stats->spike_blend_current = bridge->config.spike_blend;

    // Compute average binding weight
    float total_weight = 0.0f;
    uint32_t count = 0;
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            total_weight += node->binding.weight;
            count++;
            node = node->next;
        }
    }
    stats->avg_binding_weight = (count > 0) ? total_weight / count : 0.0f;

    return 0;
}

int snn_language_bridge_reset_stats(snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

float snn_language_bridge_get_blend(const snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return 0.0f;
    return bridge->config.spike_blend;
}

void snn_language_bridge_set_blend(snn_language_bridge_t* bridge, float blend)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return;
    bridge->config.spike_blend = fmaxf(0.0f, fminf(1.0f, blend));
}

//=============================================================================
// Serialization
//=============================================================================

int snn_language_bridge_save(const snn_language_bridge_t* bridge, const char* path)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Magic + config
    fwrite(&bridge->magic, sizeof(uint32_t), 1, f);
    fwrite(&bridge->config, sizeof(snn_lang_config_t), 1, f);
    fwrite(&bridge->num_concept_pops, sizeof(uint32_t), 1, f);
    fwrite(&bridge->num_word_pops, sizeof(uint32_t), 1, f);

    // Concept populations
    fwrite(bridge->concept_pops, sizeof(concept_pop_info_t),
           bridge->num_concept_pops, f);

    // Word populations
    fwrite(bridge->word_pops, sizeof(word_pop_info_t),
           bridge->num_word_pops, f);

    // Bindings (count first, then each binding)
    fwrite(&bridge->num_bindings, sizeof(uint32_t), 1, f);
    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            fwrite(&node->binding, sizeof(snn_lang_binding_t), 1, f);
            node = node->next;
        }
    }

    fclose(f);
    return 0;
}

snn_language_bridge_t* snn_language_bridge_load(const char* path)
{
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != SNN_LANG_MAGIC) {
        fclose(f);
        return NULL;
    }

    snn_lang_config_t config;
    if (fread(&config, sizeof(snn_lang_config_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    snn_language_bridge_t* bridge = snn_language_bridge_create(&config);
    if (!bridge) {
        fclose(f);
        return NULL;
    }

    uint32_t num_concepts, num_words;
    if (fread(&num_concepts, sizeof(uint32_t), 1, f) != 1 ||
        fread(&num_words, sizeof(uint32_t), 1, f) != 1) {
        snn_language_bridge_destroy(bridge);
        fclose(f);
        return NULL;
    }

    // Load populations
    if (num_concepts <= bridge->concept_pops_capacity) {
        if (fread(bridge->concept_pops, sizeof(concept_pop_info_t), num_concepts, f)
            != num_concepts) {
            snn_language_bridge_destroy(bridge);
            fclose(f);
            return NULL;
        }
        bridge->num_concept_pops = num_concepts;
    }

    if (num_words <= bridge->word_pops_capacity) {
        if (fread(bridge->word_pops, sizeof(word_pop_info_t), num_words, f)
            != num_words) {
            snn_language_bridge_destroy(bridge);
            fclose(f);
            return NULL;
        }
        bridge->num_word_pops = num_words;
    }

    // Load bindings
    uint32_t num_bindings;
    if (fread(&num_bindings, sizeof(uint32_t), 1, f) != 1) {
        snn_language_bridge_destroy(bridge);
        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < num_bindings; i++) {
        snn_lang_binding_t b;
        if (fread(&b, sizeof(snn_lang_binding_t), 1, f) != 1) break;
        binding_node_t* node = binding_insert(bridge, b.concept_pop,
                                               b.word_pop, b.weight);
        if (node) {
            node->binding = b;
        }
    }

    fclose(f);
    return bridge;
}

//=============================================================================
// Phase 8.5: Top-Down Binding -> Perception Attention Feedback
//=============================================================================

/**
 * WHAT: Generate attention weights from active concept bindings
 * WHY:  Language understanding guides sensory attention (top-down)
 * HOW:  Iterate all bindings; sum weights per concept population;
 *        normalize to [0, 1] attention weights
 */
int snn_language_bridge_generate_attention_feedback(
    snn_language_bridge_t* bridge,
    float* attention_weights,
    uint32_t num_weights)
{
    if (!bridge || !attention_weights || num_weights == 0) return -1;

    /* Zero output */
    memset(attention_weights, 0, num_weights * sizeof(float));

    /* Accumulate binding weights per concept population */
    float max_weight = 0.0f;
    for (uint32_t b = 0; b < BINDING_HASH_BUCKETS; b++) {
        binding_node_t* node = bridge->binding_buckets[b];
        while (node) {
            uint32_t cp = node->binding.concept_pop;
            if (cp < num_weights) {
                /* Weight contribution scaled by concept activation */
                float activation = 0.0f;
                if (cp < bridge->num_concept_pops) {
                    activation = bridge->concept_pops[cp].activation;
                }
                float contrib = node->binding.weight * activation;
                attention_weights[cp] += contrib;
                if (attention_weights[cp] > max_weight) {
                    max_weight = attention_weights[cp];
                }
            }
            node = node->next;
        }
    }

    /* Normalize to [0, 1] */
    if (max_weight > 1e-6f) {
        float inv_max = 1.0f / max_weight;
        for (uint32_t i = 0; i < num_weights; i++) {
            attention_weights[i] *= inv_max;
        }
    }

    return 0;
}

/**
 * WHAT: Generate predicted sensory pattern from active concepts
 * WHY:  Predictive coding — reduce prediction error at sensory level
 * HOW:  For each active concept, sum binding weights to word populations;
 *        output predicted sensory pattern as weighted concept-to-sensory map
 */
int snn_language_bridge_predict_sensory(
    snn_language_bridge_t* bridge,
    const float* concept_activations,
    uint32_t num_concepts,
    float* predicted_sensory,
    uint32_t sensory_dim)
{
    if (!bridge || !concept_activations || !predicted_sensory) return -1;
    if (num_concepts == 0 || sensory_dim == 0) return -1;

    memset(predicted_sensory, 0, sensory_dim * sizeof(float));

    /* For each active concept, lookup bound word populations and
     * generate expected sensory pattern from binding weights */
    float total_activation = 0.0f;
    for (uint32_t c = 0; c < num_concepts && c < bridge->num_concept_pops; c++) {
        float act = concept_activations[c];
        if (act < 0.01f) continue;  /* Skip inactive concepts */

        total_activation += act;

        /* Iterate bindings for this concept */
        for (uint32_t b = 0; b < BINDING_HASH_BUCKETS; b++) {
            binding_node_t* node = bridge->binding_buckets[b];
            while (node) {
                if (node->binding.concept_pop == c) {
                    uint32_t wp = node->binding.word_pop;
                    /* Map word population to sensory dimension via modular index */
                    uint32_t sensory_idx = wp % sensory_dim;
                    predicted_sensory[sensory_idx] +=
                        act * node->binding.weight;
                }
                node = node->next;
            }
        }
    }

    /* Normalize by total activation */
    if (total_activation > 1e-6f) {
        float inv = 1.0f / total_activation;
        for (uint32_t i = 0; i < sensory_dim; i++) {
            predicted_sensory[i] *= inv;
            /* Clamp to [0, 1] */
            if (predicted_sensory[i] > 1.0f) predicted_sensory[i] = 1.0f;
        }
    }

    return 0;
}
