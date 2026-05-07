//=============================================================================
// nimcp_snn_language_bridge.h - SNN ↔ Language Spike-Driven Bridge
//=============================================================================
/**
 * @file nimcp_snn_language_bridge.h
 * @brief Spike-to-word decoder and STDP-driven word-concept binding
 *
 * Bridges the gap between SNN spike patterns and language generation.
 * Concept neuron populations fire in response to semantic content;
 * word neuron populations represent vocabulary entries. STDP strengthens
 * bindings when concept spikes precede word spikes (e.g., seeing a dog
 * before hearing "dog"). Production cascades concept activations through
 * binding weights to select words via winner-take-all.
 *
 * Integrates with:
 * - Grounded Language System (production/comprehension fallback + dual-path)
 * - Imagination SNN (creative spike patterns → word selection)
 * - Curiosity SNN (novelty-driven lexical exploration)
 * - STDP module (spike-timing-dependent plasticity for binding learning)
 * - Neuromodulator system (dopamine gating for three-factor learning)
 * - Sleep consolidation (binding replay and pruning)
 */

#ifndef NIMCP_SNN_LANGUAGE_BRIDGE_H
#define NIMCP_SNN_LANGUAGE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Constants
//=============================================================================

#define SNN_LANG_MAX_CONCEPT_POPS     4096
#define SNN_LANG_MAX_WORD_POPS        16384
#define SNN_LANG_NEURONS_PER_POP      8
#define SNN_LANG_DEFAULT_STDP_TAU     50.0f   // ms (wider than standard 20ms)
#define SNN_LANG_DEFAULT_STDP_A_PLUS  0.008f
#define SNN_LANG_DEFAULT_STDP_A_MINUS 0.0084f // Slight LTD bias for sparsity
#define SNN_LANG_BINDING_W_MAX        1.0f
#define SNN_LANG_BINDING_W_MIN        0.0f
#define SNN_LANG_DECODE_WINDOW_MS     100.0f
#define SNN_LANG_DECAY_RATE           0.95f
#define SNN_LANG_SPIKE_BLEND_DEFAULT  0.1f    // 10% spike, 90% vector initially
#define SNN_LANG_MAGIC                0x534C4247  // "SLBG"

//=============================================================================
// Forward declarations
//=============================================================================

struct snn_network_s;
struct grounded_language;
struct imagination_snn_bridge;
struct curiosity_snn_bridge;
struct neuromodulator_system_struct;

//=============================================================================
// Types
//=============================================================================

/** Sparse binding entry (concept_id → word_id with STDP weight) */
typedef struct {
    uint32_t concept_pop;       // Concept population index
    uint32_t word_pop;          // Word population index
    float    weight;            // STDP-modifiable binding strength [0, 1]
    float    eligibility;       // Eligibility trace for three-factor learning
    float    pre_trace;         // Presynaptic (concept) trace
    float    post_trace;        // Postsynaptic (word) trace
    float    last_pre_spike_ms; // Last concept spike time
    float    last_post_spike_ms;// Last word spike time
    uint32_t ltp_count;         // Potentiation events
    uint32_t ltd_count;         // Depression events
} snn_lang_binding_t;

/** Configuration for SNN language bridge */
typedef struct {
    uint32_t max_concept_pops;     // Max concept populations
    uint32_t max_word_pops;        // Max word populations
    uint32_t neurons_per_pop;      // Neurons per population
    float    stdp_tau_plus;        // LTP time constant (ms)
    float    stdp_tau_minus;       // LTD time constant (ms)
    float    stdp_a_plus;          // LTP amplitude
    float    stdp_a_minus;         // LTD amplitude
    float    stdp_learning_rate;   // Base learning rate
    float    binding_w_max;        // Max binding weight
    float    decode_window_ms;     // Spike integration window
    float    decay_rate;           // Per-step activation decay
    float    spike_blend;          // Spike vs vector blend [0=all vector, 1=all spike]
    bool     enable_da_modulation; // Dopamine-gated three-factor learning
    float    da_modulation_gain;   // DA → LR scaling
    bool     enable_imagination;   // Wire imagination SNN output to word selection
    bool     enable_curiosity;     // Wire curiosity SNN to lexical exploration
    bool     enable_sleep_consolidation; // Enable binding replay during sleep
    float    prune_threshold;      // Binding weights below this are pruned
} snn_lang_config_t;

/** Word decode result */
typedef struct {
    uint32_t    word_pop;      // Word population index
    const char* word_form;     // Word string (borrowed pointer)
    float       activation;    // Accumulated activation level
    float       confidence;    // Decode confidence [0, 1]
} snn_lang_word_result_t;

/** Production result from spike-driven generation */
typedef struct {
    char*    text;             // Generated text (heap-allocated)
    uint32_t word_count;       // Number of words produced
    float    fluency;          // Fluency score [0, 1]
    float    spike_confidence; // Average spike-based word confidence
    float    creativity;       // Creativity contribution from imagination SNN
} snn_lang_production_result_t;

/** Bridge statistics */
typedef struct {
    uint64_t total_decode_calls;
    uint64_t total_encode_calls;
    uint64_t total_produce_calls;
    uint64_t total_comprehend_calls;
    uint64_t total_stdp_updates;
    uint64_t total_ltp_events;
    uint64_t total_ltd_events;
    uint32_t active_bindings;
    float    avg_binding_weight;
    float    avg_word_confidence;
    float    spike_blend_current;
    uint64_t imagination_contributions;
    uint64_t curiosity_contributions;
    uint64_t sleep_consolidation_cycles;
    uint64_t bindings_pruned;
} snn_lang_stats_t;

/** Opaque bridge type */
typedef struct snn_language_bridge snn_language_bridge_t;

//=============================================================================
// Configuration
//=============================================================================

/** Get default configuration */
snn_lang_config_t snn_lang_config_default(void);

//=============================================================================
// Lifecycle
//=============================================================================

/** Create SNN language bridge */
snn_language_bridge_t* snn_language_bridge_create(
    const snn_lang_config_t* config);

/** Destroy and free */
void snn_language_bridge_destroy(snn_language_bridge_t* bridge);

/** Reset state (clear activations, keep bindings) */
int snn_language_bridge_reset(snn_language_bridge_t* bridge);

//=============================================================================
// Connection (wire to existing subsystems)
//=============================================================================

/** Connect to grounded language system (required for word lookup) */
int snn_language_bridge_connect_grounded(
    snn_language_bridge_t* bridge,
    struct grounded_language* gl);

/** Connect to imagination SNN for creative word selection */
int snn_language_bridge_connect_imagination(
    snn_language_bridge_t* bridge,
    struct imagination_snn_bridge* imagination);

/** Connect to curiosity SNN for novelty-driven lexical exploration */
int snn_language_bridge_connect_curiosity(
    snn_language_bridge_t* bridge,
    struct curiosity_snn_bridge* curiosity);

/** Connect neuromodulator system for dopamine-gated STDP */
int snn_language_bridge_connect_neuromod(
    snn_language_bridge_t* bridge,
    struct neuromodulator_system_struct* neuromod);

//=============================================================================
// Phase 1: Spike-to-Word Decoding
//=============================================================================

/** Register a concept population (maps concept_id to neuron population) */
int snn_language_bridge_register_concept(
    snn_language_bridge_t* bridge,
    uint32_t concept_pop,
    uint64_t concept_id);

/** Register a word population (maps word string to neuron population) */
int snn_language_bridge_register_word(
    snn_language_bridge_t* bridge,
    uint32_t word_pop,
    const char* word_form);

/** Decode spike patterns to word activations (population vector decoding) */
int snn_language_bridge_decode_spikes(
    snn_language_bridge_t* bridge,
    const float* concept_rates,    // Firing rates per concept pop [num_concept_pops]
    uint32_t num_concept_pops,
    snn_lang_word_result_t* results, // Output: top-k words
    uint32_t max_results,
    uint32_t* num_results);

/** Encode a word as concept neuron activation pattern */
int snn_language_bridge_encode_word(
    snn_language_bridge_t* bridge,
    uint32_t word_pop,
    float* concept_activations,    // Output: concept activations [num_concept_pops]
    uint32_t num_concept_pops);

//=============================================================================
// Phase 2: STDP-Driven Word-Concept Binding
//=============================================================================

/** Record a concept population spike event */
int snn_language_bridge_concept_spike(
    snn_language_bridge_t* bridge,
    uint32_t concept_pop,
    float spike_time_ms);

/** Record a word population spike event */
int snn_language_bridge_word_spike(
    snn_language_bridge_t* bridge,
    uint32_t word_pop,
    float spike_time_ms);

/** Apply STDP updates to all bindings with recent spikes */
int snn_language_bridge_apply_stdp(
    snn_language_bridge_t* bridge,
    float current_time_ms);

/** Create or strengthen a binding between concept and word populations */
int snn_language_bridge_bind(
    snn_language_bridge_t* bridge,
    uint32_t concept_pop,
    uint32_t word_pop,
    float initial_weight);

/** Prune weak bindings below threshold */
int snn_language_bridge_prune(
    snn_language_bridge_t* bridge,
    float threshold);

//=============================================================================
// Phase 3: Spike-Driven Language Production (Broca pathway)
//=============================================================================

/** Produce text from semantic intent via spike cascade */
int snn_language_bridge_produce(
    snn_language_bridge_t* bridge,
    const float* semantic_intent,
    uint32_t intent_dim,
    snn_lang_production_result_t* result);

/** Produce a single word from current concept activations */
int snn_language_bridge_produce_word(
    snn_language_bridge_t* bridge,
    const float* concept_activations,
    uint32_t num_concepts,
    snn_lang_word_result_t* result);

/** Cleanup production result */
void snn_lang_production_result_cleanup(snn_lang_production_result_t* result);

//=============================================================================
// Phase 4: Spike-Driven Comprehension (Wernicke pathway)
//=============================================================================

/** Comprehend text by cascading word spikes to concept activations */
int snn_language_bridge_comprehend(
    snn_language_bridge_t* bridge,
    const char* text,
    float* concept_activations,    // Output: activated concepts
    uint32_t max_concepts,
    uint32_t* num_activated,
    float* comprehension_confidence);

//=============================================================================
// Phase 5: Creative/Imagination Integration
//=============================================================================

/** Feed imagination SNN output through binding matrix for creative words */
int snn_language_bridge_creative_produce(
    snn_language_bridge_t* bridge,
    const float* imagination_activations,
    uint32_t num_dims,
    float creativity_level,
    snn_lang_production_result_t* result);

/** Feed curiosity drive into lexical exploration (bias toward novel words) */
int snn_language_bridge_curiosity_modulate(
    snn_language_bridge_t* bridge,
    float novelty_level,
    float exploration_drive);

//=============================================================================
// Phase 6: Sleep Consolidation
//=============================================================================

/** Consolidate bindings during sleep (replay + prune) */
int snn_language_bridge_sleep_consolidate(
    snn_language_bridge_t* bridge,
    float consolidation_strength);

//=============================================================================
// Statistics & Introspection
//=============================================================================

/** Get bridge statistics */
int snn_language_bridge_get_stats(
    const snn_language_bridge_t* bridge,
    snn_lang_stats_t* stats);

/** Reset statistics counters */
int snn_language_bridge_reset_stats(snn_language_bridge_t* bridge);

/** Get current spike blend factor */
float snn_language_bridge_get_blend(const snn_language_bridge_t* bridge);

/** Get word form string for a given word population index (NULL if invalid/unregistered) */
const char* snn_language_bridge_get_word_form(
    const snn_language_bridge_t* bridge,
    uint32_t word_pop_index);

/** Set spike blend factor [0=all vector, 1=all spike] */
void snn_language_bridge_set_blend(snn_language_bridge_t* bridge, float blend);

//=============================================================================
// Serialization
//=============================================================================

/** Save binding weights and configuration */
int snn_language_bridge_save(const snn_language_bridge_t* bridge, const char* path);

/** Load binding weights and configuration */
snn_language_bridge_t* snn_language_bridge_load(const char* path);

/** Recompute per-word_pop binding-weight L2 norm cache (Σ weight² per word_pop)
 * from the current binding state. Used after bulk binding changes and on
 * bridge load. Cosine-normalized decode_spikes consults this cache to remove
 * binding-density rank-1 bias. Cheap O(num_bindings); idempotent. */
int snn_language_bridge_recompute_norms(snn_language_bridge_t* bridge);

//=============================================================================
// Phase 8.5: Top-Down Binding -> Perception Attention Feedback
//=============================================================================

/**
 * @brief Generate top-down attention signal from concept bindings
 * WHAT: Convert active concept bindings into perception attention weights
 * WHY:  Enable language understanding to guide visual/auditory attention
 * HOW:  Strong bindings -> high attention for associated sensory populations
 */
int snn_language_bridge_generate_attention_feedback(
    snn_language_bridge_t* bridge,
    float* attention_weights,
    uint32_t num_weights
);

/**
 * @brief Apply top-down prediction to sensory input
 * WHAT: Use concept predictions to modulate expected sensory patterns
 * WHY:  Predictive coding - reduce prediction error at sensory level
 * HOW:  Active concepts generate expected spike patterns, compared to actual
 */
int snn_language_bridge_predict_sensory(
    snn_language_bridge_t* bridge,
    const float* concept_activations,
    uint32_t num_concepts,
    float* predicted_sensory,
    uint32_t sensory_dim
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SNN_LANGUAGE_BRIDGE_H */
