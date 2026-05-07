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
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "security/nimcp_bbb_helpers.h"
#include "utils/geometry/nimcp_hyperbolic.h"
#include "utils/quantum/nimcp_quantum_monte_carlo.h"

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

#define LOG_MODULE "SNN_LANG_BRIDGE"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(snn_language_bridge)

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

    /* Patch A: per-word_pop sum of squared binding weights, for cosine
     * normalization in decode_spikes. Sized to word_pops_capacity. Maintained
     * incrementally on every binding mutation (binding_insert new + max-merge,
     * STDP weight update). decode_spikes divides word_acts[w] by sqrt(this+ε)
     * to remove rank-1 binding-density bias. Without this, words that have
     * accumulated more bindings (curriculum-frequent tokens) win every rank
     * regardless of input semantic vector. */
    float* word_norm_sq;

    /* PA-6: xorshift64* RNG state for produce-time softmax sampling. Seeded
     * once at create() with time XOR pointer mix. Only consulted when
     * config.temperature > 0; deterministic argmax path bypasses it. */
    uint64_t rng_state;

    /* PA-5: GloVe lookup callback + lazy per-word cache. emb_dim and the
     * cache buffers are NULL/zero until set_embedding_lookup() runs.
     *   word_emb_cache:   word_pops_capacity * emb_dim floats, row-major.
     *   word_emb_cached:  bool flag per word_pop — true once filled.
     *   word_emb_norm:    cached ‖emb[w]‖ per word_pop for cosine.
     * Cache is invalidated by snn_language_bridge_invalidate_emb_cache()
     * (e.g. after embedding-table retrain) and by detaching the lookup. */
    snn_lang_word_emb_fn emb_lookup_fn;
    void*               emb_lookup_ctx;
    uint32_t            emb_dim;
    float*              word_emb_cache;
    uint8_t*            word_emb_cached;
    float*              word_emb_norm;

    /* PA-3: SNN-spike routing — table of attached SNN pops + their roles.
     * Drain loop iterates this table once per tick. snn_pop_id < 0 marks
     * an empty slot. */
    struct {
        int                 snn_pop_id;
        uint32_t            n_neurons;
        snn_lang_pop_role_t role;
    } attached_pops[SNN_LANG_MAX_ATTACHED_POPS];
    uint32_t                n_attached_pops;

    /* Walkthrough round 2: PA-2 × PA-5 interaction fix. The autoregressive
     * decoder evolves concept_acts into binding space (state drifts toward
     * encode_word(picked) bindings). The GloVe cosine in decode_spikes uses
     * concept_rates[0:emb_dim] as the embedding query — which is corrupt
     * once state diverges from intent. produce sets emb_query_override to
     * the immutable original intent before each decode_spikes call, and
     * resets it on exit. NULL = use concept_rates (legacy / no autoreg). */
    const float*            emb_query_override;
    uint32_t                emb_query_override_dim;

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

/* PA-5 forward decl: lazy embedding cache filler. Defined alongside the
 * other PA-5 helpers below; declared up here because decode_spikes uses
 * it before its definition. */
static inline int emb_cache_ensure(snn_language_bridge_t* bridge, uint32_t w);

/* Patch A: maintain word_norm_sq[word_pop] = Σ w² across all bindings
 * touching word_pop. Δ(Σw²) = new² − old² for any single weight mutation. */
static inline void norm_update(snn_language_bridge_t* bridge,
                                uint32_t word_pop,
                                float old_w, float new_w)
{
    if (!bridge->word_norm_sq || word_pop >= bridge->word_pops_capacity) return;
    float delta = (new_w * new_w) - (old_w * old_w);
    bridge->word_norm_sq[word_pop] += delta;
    if (bridge->word_norm_sq[word_pop] < 0.0f) {
        bridge->word_norm_sq[word_pop] = 0.0f;  /* fp drift floor */
    }
}

static binding_node_t* binding_insert(snn_language_bridge_t* bridge,
                                       uint32_t concept_pop, uint32_t word_pop,
                                       float initial_weight)
{
    binding_node_t* existing = binding_find(bridge, concept_pop, word_pop);
    if (existing) {
        float old_w = existing->binding.weight;
        float new_w = fmaxf(old_w, initial_weight);
        if (new_w != old_w) {
            existing->binding.weight = new_w;
            norm_update(bridge, word_pop, old_w, new_w);
        }
        return existing;
    }

    binding_node_t* node = nimcp_calloc(1, sizeof(binding_node_t));
    if (!node) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "binding_insert: failed to allocate binding_node");
        return NULL;
    }

    node->binding.concept_pop = concept_pop;
    node->binding.word_pop = word_pop;
    node->binding.weight = initial_weight;
    node->binding.last_pre_spike_ms = -1000.0f;
    node->binding.last_post_spike_ms = -1000.0f;

    uint32_t bucket = binding_hash(concept_pop, word_pop);
    node->next = bridge->binding_buckets[bucket];
    bridge->binding_buckets[bucket] = node;
    bridge->num_bindings++;

    norm_update(bridge, word_pop, 0.0f, initial_weight);

    return node;
}

//=============================================================================
// Configuration
//=============================================================================

snn_lang_config_t snn_lang_config_default(void)
{
    /* These two MUST match the SNN_LANG_MAX_*_POPS macros in the
     * public header — grounded_language's mirror_binding_to_bridge
     * hashes form_hash modulo SNN_LANG_MAX_WORD_POPS (16384) and
     * concept_id modulo SNN_LANG_MAX_CONCEPT_POPS (4096). If the
     * runtime capacity is smaller, register_word/register_concept
     * silently reject any pop beyond the smaller cap (line 330+346
     * of this file: `if (pop >= bridge->*_pops_capacity) return -1`)
     * and ~75% of bindings are dropped on the floor. Bug verified
     * 2026-05-06 — bridge_active_bindings was effectively zero for
     * months because of this mismatch. */
    snn_lang_config_t config = {
        .max_concept_pops = SNN_LANG_MAX_CONCEPT_POPS,
        .max_word_pops    = SNN_LANG_MAX_WORD_POPS,
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
        .prune_threshold = 0.005f,
        /* PA-6: defaults preserve legacy argmax behavior. Callers explicitly
         * set temperature > 0 to opt into sampling. produce_topk = 8 gives
         * sampling some headroom over the legacy 5 without much cost. */
        .temperature = 0.0f,
        .top_p = 1.0f,
        .produce_topk = 8,
        /* PA-5: GloVe blend off by default. Caller sets >0 after attaching
         * an embedding lookup with set_embedding_lookup(). */
        .glove_blend = 0.0f,
        /* PA-2: autoregressive defaults match legacy 0.7/0.3 in-place blend
         * (intent_persistence = 0 → state-driven; word_feedback = 0.3). */
        .intent_persistence = 0.0f,
        .word_feedback      = 0.3f,
        /* PA-3: spike routing OFF by default — explicit opt-in required to
         * avoid recreating the prior sparsity-collapse failure mode.
         * activation_tau_ms = 200 is a safe non-zero value; ignored when
         * the master flag is off. */
        .enable_snn_spike_routing = false,
        .activation_tau_ms        = 200.0f,
        /* PA-5+: hyperbolic distance OFF by default. Switching to true
         * makes decode_spikes use 1/(1+d_H(.,.)) instead of cosine. */
        .use_hyperbolic_embeddings = false,
        /* PA-6+: 0 = legacy auto-dispatch (argmax / softmax+top-p driven
         * by temperature). Modes 1 (softmax-only) and 2 (q-MC) are
         * explicit opt-ins. */
        .sampling_mode = 0
    };
    return config;
}

//=============================================================================
// Lifecycle
//=============================================================================

snn_language_bridge_t* snn_language_bridge_create(const snn_lang_config_t* config)
{
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_create: config is NULL");
        return NULL;
    }

    snn_language_bridge_t* bridge = nimcp_calloc(1, sizeof(snn_language_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_create: failed to allocate bridge");
        return NULL;
    }

    bridge->magic = SNN_LANG_MAGIC;
    bridge->config = *config;

    // Allocate concept populations
    bridge->concept_pops_capacity = config->max_concept_pops;
    bridge->concept_pops = nimcp_calloc(bridge->concept_pops_capacity,
                                        sizeof(concept_pop_info_t));
    if (!bridge->concept_pops) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_create: failed to allocate concept_pops");
        nimcp_free(bridge);
        return NULL;
    }

    // Allocate word populations
    bridge->word_pops_capacity = config->max_word_pops;
    bridge->word_pops = nimcp_calloc(bridge->word_pops_capacity,
                                     sizeof(word_pop_info_t));
    if (!bridge->word_pops) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_create: failed to allocate word_pops");
        nimcp_free(bridge->concept_pops);
        nimcp_free(bridge);
        return NULL;
    }

    /* Patch A: per-word_pop binding-weight L2 norm cache. Calloc → all zeros,
     * which is the correct initial state (no bindings yet). */
    bridge->word_norm_sq = nimcp_calloc(bridge->word_pops_capacity,
                                         sizeof(float));
    if (!bridge->word_norm_sq) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_create: failed to allocate word_norm_sq");
        nimcp_free(bridge->word_pops);
        nimcp_free(bridge->concept_pops);
        nimcp_free(bridge);
        return NULL;
    }

    /* PA-6: seed sampling RNG. xorshift64* requires nonzero seed. */
    bridge->rng_state = (uint64_t)time(NULL) ^ ((uintptr_t)bridge * 0x9E3779B97F4A7C15ULL);
    if (bridge->rng_state == 0) bridge->rng_state = 0xDEADBEEF;

    /* PA-3: empty attached-pops table (snn_pop_id < 0 = unused slot). */
    for (uint32_t i = 0; i < SNN_LANG_MAX_ATTACHED_POPS; i++) {
        bridge->attached_pops[i].snn_pop_id = -1;
    }
    bridge->n_attached_pops = 0;

    bbb_register_module("snn_language_bridge", BBB_MODULE_TYPE_COGNITIVE);

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
    nimcp_free(bridge->word_norm_sq);
    nimcp_free(bridge->word_emb_cache);
    nimcp_free(bridge->word_emb_cached);
    nimcp_free(bridge->word_emb_norm);

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
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_connect_grounded: bridge is NULL or invalid");
        return -1;
    }
    bridge->grounded_lang = gl;
    return 0;
}

int snn_language_bridge_connect_imagination(snn_language_bridge_t* bridge,
                                             struct imagination_snn_bridge* imagination)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_connect_imagination: bridge is NULL or invalid");
        return -1;
    }
    bridge->imagination = imagination;
    return 0;
}

int snn_language_bridge_connect_curiosity(snn_language_bridge_t* bridge,
                                           struct curiosity_snn_bridge* curiosity)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_connect_curiosity: bridge is NULL or invalid");
        return -1;
    }
    bridge->curiosity = curiosity;
    return 0;
}

int snn_language_bridge_connect_neuromod(snn_language_bridge_t* bridge,
                                          struct neuromodulator_system_struct* neuromod)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_connect_neuromod: bridge is NULL or invalid");
        return -1;
    }
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_decode_spikes: bridge, concept_rates, results, or num_results is NULL");
        return -1;
    }

    bridge->stats.total_decode_calls++;
    *num_results = 0;

    // Compute word activations through binding matrix
    // word_activation[w] = sum_c(concept_rates[c] * binding_weight[c,w])
    uint32_t n_words = bridge->num_word_pops;
    float* word_acts = nimcp_calloc(n_words, sizeof(float));
    if (!word_acts) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_decode_spikes: failed to allocate word_acts");
        return -1;
    }

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

    /* Patch A: cosine-normalize. score[w] = (concept_rates · weight[*, w]) /
     *                                       ||weight[*, w]||₂
     * Without this, words with more bindings (or larger total binding mass)
     * dominate top-K regardless of input direction — diagnosed live as a
     * rank-1 collapse where the same 4 words win every produce call.
     * concept_rates is treated as already-comparable across calls (caller
     * passes l2-normalized intent); we only normalize the binding column. */
    const float eps = 1e-6f;
    float* word_scores = nimcp_calloc(n_words, sizeof(float));
    if (!word_scores) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_decode_spikes: failed to allocate word_scores");
        nimcp_free(word_acts);
        return -1;
    }

    /* PA-5: precompute the intent-side embedding norm once if the GloVe
     * blend is active. The embedding query is the leading emb_dim coords
     * of concept_rates. Skip lookups entirely when blend is 0 or when no
     * embedding callback has been attached.
     *
     * Walkthrough round 2 fix (PA-2 × PA-5): when the autoregressive
     * decoder evolves concept_acts into binding space, the GloVe cosine
     * sees a corrupted embedding query. produce sets emb_query_override
     * to the immutable original intent so the GloVe term remains
     * embedding-space-coherent across the loop. NULL = no override
     * (legacy callers + non-autoregressive produce). */
    const float glove_blend = bridge->config.glove_blend;
    const bool emb_active = (glove_blend > 0.0f && bridge->emb_lookup_fn &&
                              bridge->word_emb_cache && bridge->emb_dim > 0);
    const bool hyper_mode = bridge->config.use_hyperbolic_embeddings;
    const float* emb_query = bridge->emb_query_override
                               ? bridge->emb_query_override
                               : concept_rates;
    const uint32_t emb_query_dim = bridge->emb_query_override
                                     ? bridge->emb_query_override_dim
                                     : num_concept_pops;
    float intent_emb_norm = 0.0f;
    /* PA-5+: hyperbolic mode also needs the projected query inside the
     * Poincaré ball. Project via tanh(‖v‖)·v/‖v‖ — this maps any finite
     * Euclidean vector into B^d while preserving direction, and clips
     * to ‖.‖ < 1 even for unit-norm inputs. d_used is capped at the
     * smaller of emb_dim and emb_query_dim, max 64 (more than enough
     * for any GloVe slice we ever pass through here). */
    float intent_hyper[64] = {0};
    uint32_t d_used = 0;
    if (emb_active) {
        d_used = (bridge->emb_dim < emb_query_dim)
                   ? bridge->emb_dim : emb_query_dim;
        if (d_used > 64) d_used = 64;
        for (uint32_t d = 0; d < d_used; d++) {
            intent_emb_norm += emb_query[d] * emb_query[d];
        }
        intent_emb_norm = sqrtf(intent_emb_norm + eps);
        if (hyper_mode) {
            /* Project: q' = tanh(‖q‖) · q / ‖q‖. */
            float scale = tanhf(intent_emb_norm) / intent_emb_norm;
            for (uint32_t d = 0; d < d_used; d++) {
                intent_hyper[d] = emb_query[d] * scale;
            }
        }
    }

    for (uint32_t w = 0; w < n_words; w++) {
        if (!bridge->word_pops[w].registered) {
            word_scores[w] = -FLT_MAX;
            continue;
        }
        float ns = bridge->word_norm_sq ? bridge->word_norm_sq[w] : 1.0f;
        float norm = sqrtf(ns + eps);
        float binding_score = word_acts[w] / norm;

        /* PA-5: blend in cosine(intent_emb, word_emb[w]) when active.
         * Walkthrough round 2 fix: read embedding query from emb_query
         * (override-aware) so PA-2 autoregressive blending doesn't
         * corrupt the GloVe term.
         * PA-5+: hyper_mode swaps the cosine for 1/(1+d_H(.,.)). */
        if (emb_active) {
            int got = emb_cache_ensure(bridge, w);
            if (got == 1) {
                const float* emb = bridge->word_emb_cache + (size_t)w * bridge->emb_dim;
                float glove_score = 0.0f;
                if (hyper_mode) {
                    /* Project word emb the same way: e' = tanh(‖e‖)·e/‖e‖.
                     * word_emb_norm[w] already holds sqrt(Σe² + eps) from
                     * emb_cache_ensure, so reuse it. */
                    float wnorm = bridge->word_emb_norm[w];
                    if (wnorm < eps) wnorm = eps;
                    float wscale = tanhf(wnorm) / wnorm;
                    /* Inline Poincaré distance:
                     *   d = acosh(1 + 2 ‖x-y‖² / ((1-‖x‖²)(1-‖y‖²)))
                     */
                    float diff_sq = 0.0f;
                    float xn_sq = 0.0f;
                    float yn_sq = 0.0f;
                    for (uint32_t d = 0; d < d_used; d++) {
                        float xv = intent_hyper[d];
                        float yv = emb[d] * wscale;
                        float dv = xv - yv;
                        diff_sq += dv * dv;
                        xn_sq += xv * xv;
                        yn_sq += yv * yv;
                    }
                    /* Clip to ball interior. */
                    float one_minus_x = 1.0f - xn_sq;
                    float one_minus_y = 1.0f - yn_sq;
                    if (one_minus_x < POINCARE_EPSILON) one_minus_x = POINCARE_EPSILON;
                    if (one_minus_y < POINCARE_EPSILON) one_minus_y = POINCARE_EPSILON;
                    float arg = 1.0f + 2.0f * diff_sq / (one_minus_x * one_minus_y);
                    /* Numerically-safe acosh. arg ≥ 1 by construction; floor
                     * at 1.0 in case of fp drift. */
                    if (arg < 1.0f) arg = 1.0f;
                    float d_h = acoshf(arg);
                    if (!isfinite(d_h)) d_h = 0.0f;
                    glove_score = 1.0f / (1.0f + d_h);
                } else {
                    uint32_t d_cos = (bridge->emb_dim < emb_query_dim)
                                       ? bridge->emb_dim : emb_query_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < d_cos; d++) {
                        dot += emb_query[d] * emb[d];
                    }
                    glove_score = dot / (intent_emb_norm * bridge->word_emb_norm[w]);
                }
                word_scores[w] = (1.0f - glove_blend) * binding_score
                                  + glove_blend * glove_score;
            } else {
                /* No embedding for this word — fall back to binding-only,
                 * scaled by (1−blend) so words with embeddings still get a
                 * fair comparison if their glove_score is small. */
                word_scores[w] = (1.0f - glove_blend) * binding_score;
            }
        } else {
            word_scores[w] = binding_score;
        }
    }

    /* Sum of positive cosine scores for confidence normalization (denominator
     * of softmax-style attribution). Computed once across the registered set. */
    float sum_pos = 0.0f;
    for (uint32_t w = 0; w < n_words; w++) {
        if (word_scores[w] > 0.0f) sum_pos += word_scores[w];
    }

    // Find top-k words by cosine score
    for (uint32_t k = 0; k < max_results && k < n_words; k++) {
        float best_score = -FLT_MAX;
        uint32_t best_w = 0;
        bool found = false;

        for (uint32_t w = 0; w < n_words; w++) {
            if (!bridge->word_pops[w].registered) continue;
            if (word_scores[w] > best_score) {
                // Check not already in results
                bool duplicate = false;
                for (uint32_t j = 0; j < *num_results; j++) {
                    if (results[j].word_pop == w) { duplicate = true; break; }
                }
                if (!duplicate) {
                    best_score = word_scores[w];
                    best_w = w;
                    found = true;
                }
            }
        }

        if (!found || best_score <= 0.0f) break;

        results[*num_results].word_pop = best_w;
        results[*num_results].word_form = bridge->word_pops[best_w].word_form;
        results[*num_results].activation = best_score;
        results[*num_results].confidence = (sum_pos > 0.0f) ? best_score / sum_pos : 0.0f;
        (*num_results)++;
    }

    nimcp_free(word_scores);
    nimcp_free(word_acts);
    return 0;
}

int snn_language_bridge_encode_word(snn_language_bridge_t* bridge,
                                     uint32_t word_pop,
                                     float* concept_activations,
                                     uint32_t num_concept_pops)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !concept_activations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_encode_word: bridge or concept_activations is NULL");
        return -1;
    }
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

                /* Patch A: keep word_norm_sq cache consistent with STDP write. */
                float old_w = b->weight;
                b->weight += weight_change;
                b->weight = fmaxf(SNN_LANG_BINDING_W_MIN,
                            fminf(w_max, b->weight));
                norm_update(bridge, b->word_pop, old_w, b->weight);

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

/* PA-4: additive weight update on a binding. delta may be negative (LTD).
 * Creates the binding if it didn't exist (with weight = max(0, delta)).
 * Maintains the cosine norm cache via norm_update. */
int snn_language_bridge_strengthen_binding(snn_language_bridge_t* bridge,
                                            uint32_t concept_pop,
                                            uint32_t word_pop,
                                            float delta)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (concept_pop >= bridge->num_concept_pops) return -1;
    if (word_pop >= bridge->num_word_pops) return -1;
    if (!isfinite(delta)) return -1;

    float w_max = bridge->config.binding_w_max > 0.0f
                    ? bridge->config.binding_w_max : SNN_LANG_BINDING_W_MAX;

    binding_node_t* existing = binding_find(bridge, concept_pop, word_pop);
    if (existing) {
        float old_w = existing->binding.weight;
        float new_w = old_w + delta;
        if (new_w < SNN_LANG_BINDING_W_MIN) new_w = SNN_LANG_BINDING_W_MIN;
        if (new_w > w_max)                  new_w = w_max;
        if (new_w != old_w) {
            existing->binding.weight = new_w;
            norm_update(bridge, word_pop, old_w, new_w);
        }
        return 0;
    }

    /* No existing binding; only create one for positive delta. Negative
     * delta on a non-existent binding is a no-op — there is nothing to
     * weaken, and creating a zero-weight binding would just leak memory. */
    if (delta <= 0.0f) return 0;
    float new_w = (delta > w_max) ? w_max : delta;
    binding_node_t* node = binding_insert(bridge, concept_pop, word_pop, new_w);
    return node ? 0 : -1;
}

/* PA-4+: sigmoid reparameterization helpers for Riemannian binding update.
 * w = σ(u) ∈ (0, 1); σ'(u) = w*(1-w). Inlined — no new utils. */
static inline float sigmoid(float u)
{
    /* Numerically stable, branch-free for typical |u| < 30. */
    if (u >= 0.0f) {
        float z = expf(-u);
        return 1.0f / (1.0f + z);
    }
    float z = expf(u);
    return z / (1.0f + z);
}

static inline float sigmoid_prime(float w)
{
    /* d/du σ(u) expressed in terms of w = σ(u). Diagonal Fisher entry
     * for a Bernoulli-like binding. Vanishes at w∈{0,1} — that's the
     * boundary damping we want. */
    return w * (1.0f - w);
}

/* PA-4+: Riemannian / sigmoid-reparameterized binding update.
 *
 * Treats `w = σ(u)` for an unconstrained latent u and applies a Fisher-
 * preconditioned natural-gradient step in u-space, then projects back
 * through σ:
 *
 *     F_uu(w) = σ'(u) = w*(1-w)         (diag Fisher for Bernoulli-like w)
 *     Δu      = grad / (F_uu(w) + eps)  (`grad` already absorbs lr from caller)
 *     w'      = σ( σ⁻¹(w) + Δu )        (exact, not linearized)
 *
 * In mid-range (w≈0.5, F_uu=0.25) the linearization
 *   Δw ≈ σ'(u) * Δu = F_uu * (grad/F_uu) = grad
 * recovers the flat PA-4 step `lr * grad` exactly to first order. Near
 * the [0, 1] boundaries F_uu shrinks, so Δu blows up — but σ saturates
 * the projection back into the valid range, so the *effective Δw stays
 * bounded* and we never waste a step on truncation. This is the natural
 * "boundary damping" that the flat additive path lacks.
 *
 * Bridge config bound `binding_w_max` still applies post-projection so
 * an operator-tightened range is respected. */
int snn_language_bridge_strengthen_binding_riemannian(snn_language_bridge_t* bridge,
                                                       uint32_t concept_pop,
                                                       uint32_t word_pop,
                                                       float grad)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (concept_pop >= bridge->num_concept_pops) return -1;
    if (word_pop >= bridge->num_word_pops) return -1;
    if (!isfinite(grad)) return -1;

    float w_max = bridge->config.binding_w_max > 0.0f
                    ? bridge->config.binding_w_max : SNN_LANG_BINDING_W_MAX;

    /* Floor for the diagonal Fisher metric in u-space. eps keeps the
     * preconditioner finite when w is at a boundary; w_clamp keeps logit
     * away from ±inf. Both are needed and not redundant. */
    const float fisher_eps = 1e-6f;
    const float w_clamp_eps = 1e-6f;

    binding_node_t* existing = binding_find(bridge, concept_pop, word_pop);
    if (existing) {
        float old_w = existing->binding.weight;
        float w_clamped = old_w;
        if (w_clamped < w_clamp_eps)         w_clamped = w_clamp_eps;
        if (w_clamped > 1.0f - w_clamp_eps)  w_clamped = 1.0f - w_clamp_eps;

        float u       = logf(w_clamped / (1.0f - w_clamped));
        float fisher  = sigmoid_prime(w_clamped);            /* w*(1-w) */
        float du      = grad / (fisher + fisher_eps);
        float new_u   = u + du;
        float new_w   = sigmoid(new_u);

        if (new_w < SNN_LANG_BINDING_W_MIN) new_w = SNN_LANG_BINDING_W_MIN;
        if (new_w > w_max)                  new_w = w_max;
        if (new_w != old_w) {
            existing->binding.weight = new_w;
            norm_update(bridge, word_pop, old_w, new_w);
        }
        return 0;
    }

    /* No existing binding: only create one for positive grad. To match
     * the flat PA-4 bootstrap (weight = grad) — and avoid materializing
     * a half-strength binding from a single small step — we seed at
     * w_init ≈ grad clamped to w_max, identical to the flat path. The
     * Riemannian step kicks in on the *next* call when the binding
     * already exists. Negative grad on a non-existent binding is a no-op. */
    if (grad <= 0.0f) return 0;
    float new_w = (grad > w_max) ? w_max : grad;
    binding_node_t* node = binding_insert(bridge, concept_pop, word_pop, new_w);
    return node ? 0 : -1;
}

/* PA-6: xorshift64* — small period (~2^64) but more than enough for
 * per-word sampling. Returns a uniform float in [0, 1). */
static inline uint64_t bridge_rng_u64(snn_language_bridge_t* bridge)
{
    uint64_t x = bridge->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    bridge->rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float bridge_rng_unit(snn_language_bridge_t* bridge)
{
    /* 24 mantissa bits → uniform [0, 1). */
    uint32_t bits = (uint32_t)(bridge_rng_u64(bridge) >> 40);
    return (float)bits * (1.0f / 16777216.0f);
}

int snn_language_bridge_set_sampling(snn_language_bridge_t* bridge,
                                      float temperature, float top_p)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!isfinite(temperature) || temperature < 0.0f) return -1;
    if (!isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) return -1;
    bridge->config.temperature = temperature;
    bridge->config.top_p = top_p;
    return 0;
}

/* PA-6+: select sampling mode dispatch.
 *   0 = legacy (argmax / softmax+top-p auto-dispatch by temperature).
 *   1 = force softmax+top-p (PA-6).
 *   2 = quantum-Monte-Carlo MCMC sampling. */
int snn_language_bridge_set_sampling_mode(snn_language_bridge_t* bridge,
                                            int mode)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (mode < 0 || mode > 2) return -1;
    bridge->config.sampling_mode = mode;
    return 0;
}

/* PA-5 helpers — embedding cache lifecycle. */
static void emb_cache_free(snn_language_bridge_t* bridge)
{
    nimcp_free(bridge->word_emb_cache);
    nimcp_free(bridge->word_emb_cached);
    nimcp_free(bridge->word_emb_norm);
    bridge->word_emb_cache  = NULL;
    bridge->word_emb_cached = NULL;
    bridge->word_emb_norm   = NULL;
}

static int emb_cache_alloc(snn_language_bridge_t* bridge, uint32_t emb_dim)
{
    size_t cap = (size_t)bridge->word_pops_capacity;
    bridge->word_emb_cache  = nimcp_calloc(cap * emb_dim, sizeof(float));
    bridge->word_emb_cached = nimcp_calloc(cap, sizeof(uint8_t));
    bridge->word_emb_norm   = nimcp_calloc(cap, sizeof(float));
    if (!bridge->word_emb_cache || !bridge->word_emb_cached ||
        !bridge->word_emb_norm) {
        emb_cache_free(bridge);
        return -1;
    }
    return 0;
}

int snn_language_bridge_set_embedding_lookup(snn_language_bridge_t* bridge,
                                              snn_lang_word_emb_fn fn,
                                              void* ctx,
                                              uint32_t emb_dim)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;

    /* Detach: NULL fn frees the cache and zeros the lookup. */
    if (!fn) {
        emb_cache_free(bridge);
        bridge->emb_lookup_fn = NULL;
        bridge->emb_lookup_ctx = NULL;
        bridge->emb_dim = 0;
        return 0;
    }

    if (emb_dim == 0) return -1;

    /* Reattach with different dim → realloc cache. */
    if (bridge->word_emb_cache && bridge->emb_dim != emb_dim) {
        emb_cache_free(bridge);
    }
    if (!bridge->word_emb_cache) {
        if (emb_cache_alloc(bridge, emb_dim) != 0) {
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
                "set_embedding_lookup: failed to allocate emb cache");
            return -1;
        }
    }
    bridge->emb_lookup_fn  = fn;
    bridge->emb_lookup_ctx = ctx;
    bridge->emb_dim        = emb_dim;
    return 0;
}

int snn_language_bridge_set_glove_blend(snn_language_bridge_t* bridge,
                                         float blend)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!isfinite(blend) || blend < 0.0f || blend > 1.0f) return -1;
    bridge->config.glove_blend = blend;
    return 0;
}

int snn_language_bridge_invalidate_emb_cache(snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (bridge->word_emb_cached) {
        memset(bridge->word_emb_cached, 0,
               bridge->word_pops_capacity * sizeof(uint8_t));
    }
    if (bridge->word_emb_norm) {
        memset(bridge->word_emb_norm, 0,
               bridge->word_pops_capacity * sizeof(float));
    }
    return 0;
}

/* PA-5+: toggle Poincaré-ball hyperbolic distance for the GloVe term. The
 * raw Euclidean emb cache is shared between cosine and hyperbolic paths
 * (we project on the fly in decode_spikes) so no cache invalidation is
 * strictly required, but we do it anyway to be safe — callers may have
 * their own cache state predicated on which metric was active. */
int snn_language_bridge_set_hyperbolic_embeddings(snn_language_bridge_t* bridge,
                                                    bool enabled)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    bool was = bridge->config.use_hyperbolic_embeddings;
    bridge->config.use_hyperbolic_embeddings = enabled;
    if (was != enabled) {
        snn_language_bridge_invalidate_emb_cache(bridge);
    }
    return 0;
}

int snn_language_bridge_set_autoregressive(snn_language_bridge_t* bridge,
                                            float intent_persistence,
                                            float word_feedback)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!isfinite(intent_persistence) ||
        intent_persistence < 0.0f || intent_persistence > 1.0f) return -1;
    if (!isfinite(word_feedback) ||
        word_feedback < 0.0f || word_feedback > 1.0f) return -1;
    bridge->config.intent_persistence = intent_persistence;
    bridge->config.word_feedback      = word_feedback;
    return 0;
}

/* ============================================================================
 * PA-3: SNN-spike routing.
 *
 * The decode/comprehend paths previously relied on synthesized spike events
 * issued at lexicon-bind time, which caused SNN sparsity collapse — without
 * decay on concept_pops[].activation, every accumulated spike summed forever
 * and the attention-feedback path broadcast "attend everything" to the
 * sensory bridges. This API re-introduces real spike routing with three
 * mandatory safeguards:
 *
 *   1. Master flag (config.enable_snn_spike_routing) defaults to false.
 *   2. activation_tau_ms must be > 0 when the flag is true (rejected here).
 *   3. snn_language_bridge_tick() applies exponential decay every global
 *      tick, so accumulators cannot drift unbounded even at high spike
 *      rates.
 * ============================================================================ */
int snn_language_bridge_set_snn_spike_routing(snn_language_bridge_t* bridge,
                                                bool enabled, float tau_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (enabled) {
        if (!isfinite(tau_ms) || tau_ms <= 0.0f) return -1;
    }
    bridge->config.enable_snn_spike_routing = enabled;
    if (isfinite(tau_ms) && tau_ms > 0.0f) {
        bridge->config.activation_tau_ms = tau_ms;
    }
    return 0;
}

int snn_language_bridge_attach_snn_pop(snn_language_bridge_t* bridge,
                                        int snn_pop_id, uint32_t n_neurons,
                                        snn_lang_pop_role_t role)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (snn_pop_id < 0 || n_neurons == 0) return -1;

    /* PA-3 walkthrough fix: warn loudly if SNN pop is wider than the
     * bridge's matching cap. neuron_idx % cap aliases distinct neurons
     * onto the same bridge slot — at high collision counts each bridge
     * slot accumulates spikes from many sources, which together with
     * activation += 1.0 per spike approaches the runaway regime that
     * already destroyed sparsity once (commit 5d47666ae). The decay in
     * snn_language_bridge_tick keeps the activation bounded but only if
     * tau_ms stays sane; logging the collision factor up front lets
     * operators see the budget. */
    {
        uint32_t cap = (role == SNN_LANG_POP_ROLE_WORD)
                         ? bridge->word_pops_capacity
                         : bridge->concept_pops_capacity;
        if (cap > 0 && n_neurons > cap) {
            uint32_t collision_factor = (n_neurons + cap - 1u) / cap;
            LOG_WARN(LOG_MODULE,
                     "attach_snn_pop: pop_id=%d n_neurons=%u > bridge cap=%u "
                     "(role=%s, collision_factor=%u). decay_tau must stay "
                     "low enough that activation stays bounded.",
                     snn_pop_id, n_neurons, cap,
                     role == SNN_LANG_POP_ROLE_WORD ? "WORD" : "CONCEPT",
                     collision_factor);
            bridge->stats.attach_collision_warnings++;
        }
    }

    /* Update existing slot if pop_id is already attached. */
    for (uint32_t i = 0; i < SNN_LANG_MAX_ATTACHED_POPS; i++) {
        if (bridge->attached_pops[i].snn_pop_id == snn_pop_id) {
            bridge->attached_pops[i].n_neurons = n_neurons;
            bridge->attached_pops[i].role      = role;
            return 0;
        }
    }
    /* Otherwise grab the first free slot. */
    for (uint32_t i = 0; i < SNN_LANG_MAX_ATTACHED_POPS; i++) {
        if (bridge->attached_pops[i].snn_pop_id < 0) {
            bridge->attached_pops[i].snn_pop_id = snn_pop_id;
            bridge->attached_pops[i].n_neurons  = n_neurons;
            bridge->attached_pops[i].role       = role;
            bridge->n_attached_pops++;
            return 0;
        }
    }
    return -1;  /* table full */
}

int snn_language_bridge_drain_pop_spikes(snn_language_bridge_t* bridge,
                                           int snn_pop_id,
                                           const float* spike_output,
                                           uint32_t n_neurons,
                                           float current_time_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!spike_output) return -1;
    if (!bridge->config.enable_snn_spike_routing) return 0;  /* gated off */

    /* Look up role for this pop_id. */
    snn_lang_pop_role_t role = SNN_LANG_POP_ROLE_CONCEPT;
    bool found = false;
    for (uint32_t i = 0; i < SNN_LANG_MAX_ATTACHED_POPS; i++) {
        if (bridge->attached_pops[i].snn_pop_id == snn_pop_id) {
            role = bridge->attached_pops[i].role;
            found = true;
            break;
        }
    }
    if (!found) return -1;

    bridge->current_time_ms = current_time_ms;

    /* Walk spikes. neuron_idx → bridge pop index via modulo of bridge cap.
     * Multiple SNN neurons can map to the same bridge pop (synonyms-by-
     * collision; same pattern as the existing form_hash mirror). STDP
     * trace updates merge near-time spikes naturally. */
    if (role == SNN_LANG_POP_ROLE_WORD) {
        const uint32_t cap = bridge->word_pops_capacity > 0
                              ? bridge->word_pops_capacity
                              : SNN_LANG_MAX_WORD_POPS;
        for (uint32_t n = 0; n < n_neurons; n++) {
            if (spike_output[n] > 0.5f) {
                snn_language_bridge_word_spike(bridge, n % cap, current_time_ms);
            }
        }
    } else {
        const uint32_t cap = bridge->concept_pops_capacity > 0
                              ? bridge->concept_pops_capacity
                              : SNN_LANG_MAX_CONCEPT_POPS;
        for (uint32_t n = 0; n < n_neurons; n++) {
            if (spike_output[n] > 0.5f) {
                snn_language_bridge_concept_spike(bridge, n % cap, current_time_ms);
            }
        }
    }
    return 0;
}

int snn_language_bridge_tick(snn_language_bridge_t* bridge, float dt_ms)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!isfinite(dt_ms) || dt_ms < 0.0f) return -1;

    /* Always-on decay (independent of spike-routing flag) — cheap and
     * idempotent; keeps activations bounded even when callers synthesize
     * spikes through other paths. */
    float tau = bridge->config.activation_tau_ms > 0.0f
                  ? bridge->config.activation_tau_ms : 200.0f;
    float decay = expf(-dt_ms / tau);

    for (uint32_t c = 0; c < bridge->concept_pops_capacity; c++) {
        bridge->concept_pops[c].activation *= decay;
    }
    for (uint32_t w = 0; w < bridge->word_pops_capacity; w++) {
        bridge->word_pops[w].activation *= decay;
    }
    return 0;
}

int snn_language_bridge_get_attached_pop(const snn_language_bridge_t* bridge,
                                          uint32_t index,
                                          int* out_pop_id,
                                          uint32_t* out_n_neurons,
                                          snn_lang_pop_role_t* out_role)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (index >= SNN_LANG_MAX_ATTACHED_POPS) return -1;
    if (out_pop_id)     *out_pop_id     = bridge->attached_pops[index].snn_pop_id;
    if (out_n_neurons)  *out_n_neurons  = bridge->attached_pops[index].n_neurons;
    if (out_role)       *out_role       = bridge->attached_pops[index].role;
    return 0;
}

/* Lazy fill: ensure word w's embedding is cached. Returns 1 if cached
 * (success or already filled), 0 if no embedding for this word, -1 on
 * setup failure. After return 1, word_emb_cache[w] and word_emb_norm[w]
 * are populated. Inline to keep the decode hot path tight. */
static inline int emb_cache_ensure(snn_language_bridge_t* bridge, uint32_t w)
{
    if (!bridge->emb_lookup_fn || !bridge->word_emb_cache) return -1;
    if (w >= bridge->word_pops_capacity) return -1;
    if (bridge->word_emb_cached[w]) {
        /* 1 = embedding present and cached, 2 = looked up but missing. */
        return (bridge->word_emb_cached[w] == 1) ? 1 : 0;
    }
    const char* word = bridge->word_pops[w].word_form;
    if (!word || !word[0]) {
        bridge->word_emb_cached[w] = 2;  /* mark as missing, don't retry */
        return 0;
    }
    float* row = bridge->word_emb_cache + (size_t)w * bridge->emb_dim;
    int rc = bridge->emb_lookup_fn(bridge->emb_lookup_ctx,
                                    word, row, bridge->emb_dim);
    if (rc != 0) {
        bridge->word_emb_cached[w] = 2;
        return 0;
    }
    /* Compute and cache the norm so cosine in decode is one division. */
    float n2 = 0.0f;
    for (uint32_t d = 0; d < bridge->emb_dim; d++) n2 += row[d] * row[d];
    bridge->word_emb_norm[w] = sqrtf(n2 + 1e-6f);
    bridge->word_emb_cached[w] = 1;
    return 1;
}

/* Patch A: rebuild word_norm_sq[] from current binding state. Called after
 * binding load (where node->binding = b overwrites the weight that
 * binding_insert just norm-accounted), and exposed via the public API for
 * mid-flight live salvage of brains that pre-date Patch A. */
int snn_language_bridge_recompute_norms(snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return -1;
    if (!bridge->word_norm_sq) return -1;

    memset(bridge->word_norm_sq, 0,
           bridge->word_pops_capacity * sizeof(float));

    for (uint32_t bucket = 0; bucket < BINDING_HASH_BUCKETS; bucket++) {
        binding_node_t* node = bridge->binding_buckets[bucket];
        while (node) {
            uint32_t w = node->binding.word_pop;
            float weight = node->binding.weight;
            if (w < bridge->word_pops_capacity) {
                bridge->word_norm_sq[w] += weight * weight;
            }
            node = node->next;
        }
    }
    return 0;
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_produce_word: bridge, concept_activations, or result is NULL");
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

/* Tier-4 #16: produce-loop latency telemetry — public function timed at
 * entry/exit, work delegated to bridge_produce_impl. NULL/magic guards run
 * BEFORE clock_gettime so timing only counts work that actually happened.
 * Every entry to the public function (regardless of success/failure inside
 * the impl) increments produce_call_count and contributes elapsed wall-
 * clock micros to produce_total_us. */
static int bridge_produce_impl(snn_language_bridge_t* bridge,
                                const float* semantic_intent,
                                uint32_t intent_dim,
                                snn_lang_production_result_t* result)
{
    bridge->stats.total_produce_calls++;
    memset(result, 0, sizeof(*result));

    // Map semantic intent to concept activations
    // Use first num_concept_pops dimensions of intent (or zero-pad)
    uint32_t n_concepts = bridge->num_concept_pops;

    /* PA-2: separate intent / state / concept_acts buffers. Intent stays
     * constant across the produce loop. State evolves recurrently from
     * picked-word reverse-encodings. concept_acts (the actual decode input)
     * is recomputed each step as a blend of the two. The legacy in-place
     * 70/30 update was equivalent to the special case
     * intent_persistence = 0, word_feedback = 0.3 — which is the default. */
    float* intent       = nimcp_calloc(n_concepts, sizeof(float));
    float* state        = nimcp_calloc(n_concepts, sizeof(float));
    float* concept_acts = nimcp_calloc(n_concepts, sizeof(float));
    if (!intent || !state || !concept_acts) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_produce: failed to allocate decode buffers");
        nimcp_free(intent); nimcp_free(state); nimcp_free(concept_acts);
        return -1;
    }

    uint32_t copy_dim = (intent_dim < n_concepts) ? intent_dim : n_concepts;
    for (uint32_t i = 0; i < copy_dim; i++) {
        intent[i] = fmaxf(0.0f, semantic_intent[i]); // ReLU activation
    }
    /* state_0 = intent (so the first decode sees the same input as legacy
     * regardless of intent_persistence). */
    memcpy(state, intent, n_concepts * sizeof(float));
    memcpy(concept_acts, intent, n_concepts * sizeof(float));

    // Iterative word production with refractory inhibition
    char text_buf[2048] = {0};
    uint32_t text_pos = 0;
    uint32_t max_words = 32;
    float total_confidence = 0.0f;
    uint32_t word_count = 0;

    // Track used words for refractory
    uint32_t* used_words = nimcp_calloc(max_words, sizeof(uint32_t));
    if (!used_words) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_produce: failed to allocate used_words");
        nimcp_free(intent); nimcp_free(state); nimcp_free(concept_acts);
        return -1;
    }

    /* PA-2: read the recurrent knobs once. Clamp defensively. */
    float ip = bridge->config.intent_persistence;
    float wf = bridge->config.word_feedback;
    if (!isfinite(ip) || ip < 0.0f) ip = 0.0f;
    if (ip > 1.0f) ip = 1.0f;
    if (!isfinite(wf) || wf < 0.0f) wf = 0.0f;
    if (wf > 1.0f) wf = 1.0f;

    /* Walkthrough round 2 (PA-2 × PA-5): pin the embedding query to the
     * immutable original intent for every decode_spikes call inside this
     * produce loop. Without this, PA-5's GloVe cosine reads concept_acts
     * which PA-2 evolves into binding space — and the GloVe term would
     * see a corrupted query. Reset on every exit path below. */
    bridge->emb_query_override     = intent;
    bridge->emb_query_override_dim = n_concepts;

    /* PA-6: pull config sampling knobs once per produce call.
     * PA-6+: sampling_mode dispatch — 0 = auto, 1 = force softmax+top-p,
     * 2 = quantum-Monte-Carlo MCMC. */
    const float temperature = bridge->config.temperature;
    const float top_p = (bridge->config.top_p > 0.0f) ? bridge->config.top_p : 1.0f;
    uint32_t topk = bridge->config.produce_topk;
    if (topk == 0)  topk = 5;
    if (topk > 32)  topk = 32;
    const int sampling_mode = bridge->config.sampling_mode;

    /* DK-A+: accumulator for the average per-step entropy_confidence. */
    float total_entropy_conf = 0.0f;
    uint32_t entropy_steps = 0;

    for (uint32_t w = 0; w < max_words; w++) {
        // Get top word
        snn_lang_word_result_t word_result;
        uint32_t num_out = 0;
        snn_lang_word_result_t topK[32];
        int rc = snn_language_bridge_decode_spikes(bridge, concept_acts,
                                                    n_concepts, topK, topk, &num_out);
        if (rc != 0 || num_out == 0) break;

        /* Filter out refractory (already-used) candidates. */
        uint32_t valid_idx[32];
        uint32_t n_valid = 0;
        for (uint32_t k = 0; k < num_out; k++) {
            bool refractory = false;
            for (uint32_t u = 0; u < word_count; u++) {
                if (used_words[u] == topK[k].word_pop) {
                    refractory = true;
                    break;
                }
            }
            if (!refractory) valid_idx[n_valid++] = k;
        }
        if (n_valid == 0) break;

        /* PA-6+: when sampling_mode == 0 (auto) and temperature == 0, stay
         * on argmax. Otherwise compute the softmax posterior (used by both
         * mode 1 / 2). Mode 1 & 2 require T > 0 — if caller set the mode
         * with T = 0, fall back to argmax to avoid div-by-zero. */
        const bool need_posterior =
            (sampling_mode == 1 || sampling_mode == 2) ||
            (sampling_mode == 0 && temperature > 0.0f);

        if (!need_posterior || n_valid == 1) {
            /* Legacy hard-argmax path: pick the first valid (highest cosine). */
            word_result = topK[valid_idx[0]];
        } else {
            /* PA-6: softmax sample over valid candidates. Subtract max before
             * exp() for numerical stability. Then top-p (nucleus) truncation:
             * sort probs descending, keep until cumulative ≥ top_p, renormalize.
             * Finally inverse-CDF sample. */
            float scores[32];
            float max_score = -FLT_MAX;
            for (uint32_t i = 0; i < n_valid; i++) {
                scores[i] = topK[valid_idx[i]].activation;  /* cosine score */
                if (scores[i] > max_score) max_score = scores[i];
            }
            float probs[32];
            float sum = 0.0f;
            float T = (temperature > 0.0f) ? temperature : 1.0f;
            for (uint32_t i = 0; i < n_valid; i++) {
                probs[i] = expf((scores[i] - max_score) / T);
                sum += probs[i];
            }
            if (sum <= 0.0f) {
                word_result = topK[valid_idx[0]];
            } else {
                for (uint32_t i = 0; i < n_valid; i++) probs[i] /= sum;

                /* DK-A+: quantum-Shannon entropy confidence over the
                 * candidate posterior. 1 − H(p)/log2(K) — peaked → 1,
                 * flat → 0. Computed in nats then normalized via Hmax in
                 * nats too, so the ratio is unit-free in [0, 1]. */
                if (n_valid > 1) {
                    float H = 0.0f;
                    for (uint32_t i = 0; i < n_valid; i++) {
                        float p = probs[i];
                        if (p > 1e-12f) H -= p * logf(p);
                    }
                    float Hmax = logf((float)n_valid);  /* nats */
                    float ec = (Hmax > 0.0f) ? (1.0f - H / Hmax) : 1.0f;
                    if (ec < 0.0f) ec = 0.0f;
                    if (ec > 1.0f) ec = 1.0f;
                    total_entropy_conf += ec;
                    entropy_steps++;
                }

                /* PA-6+: q-MC route. Feed sqrt(probs) as quantum
                 * amplitudes — the quantum-MC importance sampler then
                 * draws |amp|² == probs. Single quantum measurement. */
                if (sampling_mode == 2) {
                    float amps[32];
                    for (uint32_t i = 0; i < n_valid; i++) {
                        amps[i] = sqrtf(probs[i]);
                    }
                    uint32_t qseed = (uint32_t)(bridge_rng_u64(bridge) & 0xFFFFFFFFu);
                    if (qseed == 0) qseed = 1;
                    uint32_t chosen = qmc_measure_importance(amps, n_valid,
                                                              /*proposal=*/NULL,
                                                              &qseed);
                    if (chosen >= n_valid) chosen = 0;
                    word_result = topK[valid_idx[chosen]];
                    goto sample_done;
                }

                if (top_p < 1.0f) {
                    /* Sort indices by descending prob (insertion sort, n≤32). */
                    uint32_t order[32];
                    for (uint32_t i = 0; i < n_valid; i++) order[i] = i;
                    for (uint32_t i = 1; i < n_valid; i++) {
                        uint32_t key = order[i];
                        int32_t j = (int32_t)i - 1;
                        while (j >= 0 && probs[order[j]] < probs[key]) {
                            order[j + 1] = order[j];
                            j--;
                        }
                        order[j + 1] = key;
                    }
                    /* Truncate tail past cumulative top_p. */
                    float cum = 0.0f;
                    uint32_t keep = n_valid;
                    for (uint32_t i = 0; i < n_valid; i++) {
                        cum += probs[order[i]];
                        if (cum >= top_p) { keep = i + 1; break; }
                    }
                    /* Zero-out tail and renormalize the kept head. */
                    float new_sum = 0.0f;
                    for (uint32_t i = 0; i < n_valid; i++) {
                        if (i >= keep) probs[order[i]] = 0.0f;
                        else new_sum += probs[order[i]];
                    }
                    if (new_sum > 0.0f) {
                        for (uint32_t i = 0; i < n_valid; i++) probs[i] /= new_sum;
                    } else {
                        /* PA-6 fix (walkthrough round 1): pathological top_p
                         * (e.g. < smallest prob mass) zeroes everything. The
                         * inverse-CDF below would deterministically pick
                         * `n_valid - 1` (silent argmin). Fall back to argmax
                         * — the most-likely candidate still respects the
                         * caller's intent under "keep only the highest mass". */
                        word_result = topK[valid_idx[order[0]]];
                        goto sample_done;
                    }
                }

                /* Inverse-CDF sample. */
                float u = bridge_rng_unit(bridge);
                float cum = 0.0f;
                uint32_t chosen = n_valid - 1;  /* fallback to last */
                for (uint32_t i = 0; i < n_valid; i++) {
                    cum += probs[i];
                    if (u < cum) { chosen = i; break; }
                }
                word_result = topK[valid_idx[chosen]];
            }
        }
sample_done:;

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

        /* PA-2: recurrent update. Evolve state from the just-picked word's
         * reverse-encoding, then rebuild concept_acts as the per-step blend
         * of constant intent + evolving state. With intent_persistence = 0
         * and word_feedback = 0.3, this reproduces the legacy single-buffer
         * 70/30 update bit-for-bit. */
        float* word_concepts = nimcp_calloc(n_concepts, sizeof(float));
        if (word_concepts) {
            snn_language_bridge_encode_word(bridge, word_result.word_pop,
                                            word_concepts, n_concepts);
            float keep = 1.0f - wf;
            for (uint32_t c = 0; c < n_concepts; c++) {
                state[c] = keep * state[c] + wf * word_concepts[c];
            }
            nimcp_free(word_concepts);
        }
        for (uint32_t c = 0; c < n_concepts; c++) {
            concept_acts[c] = ip * intent[c] + (1.0f - ip) * state[c];
        }
    }

    nimcp_free(used_words);
    nimcp_free(intent);
    nimcp_free(state);
    nimcp_free(concept_acts);

    /* Walkthrough round 2: clear the embedding-query override so any
     * subsequent direct caller of decode_spikes (outside this produce
     * call) reverts to using concept_rates as the GloVe query. */
    bridge->emb_query_override     = NULL;
    bridge->emb_query_override_dim = 0;

    if (word_count == 0) return -1;

    text_buf[text_pos] = '\0';
    result->text = nimcp_malloc(text_pos + 1);
    if (!result->text) return -1;
    memcpy(result->text, text_buf, text_pos + 1);

    result->word_count = word_count;
    result->spike_confidence = total_confidence / word_count;
    result->fluency = fminf(1.0f, (float)word_count / 8.0f);
    result->creativity = 0.0f;  // Set by creative_produce
    /* DK-A+: average per-step entropy_confidence over the steps where we
     * actually had a posterior. 0 when only argmax fired (no posterior to
     * measure). */
    result->entropy_confidence = (entropy_steps > 0)
                                   ? (total_entropy_conf / (float)entropy_steps)
                                   : 0.0f;

    /* Running average of per-call mean word confidence. EMA over the
     * total_produce_calls counter so the diagnostic stays bounded and
     * tracks recent behavior without unbounded growth. alpha=1/N gives
     * a true running mean; we use 0.05 floor so the metric stays
     * responsive once N is large. */
    {
        float alpha = (bridge->stats.total_produce_calls > 0)
            ? 1.0f / (float)bridge->stats.total_produce_calls : 1.0f;
        if (alpha < 0.05f) alpha = 0.05f;
        bridge->stats.avg_word_confidence =
            (1.0f - alpha) * bridge->stats.avg_word_confidence
            + alpha * result->spike_confidence;
    }

    return 0;
}

/* Tier-4 #16: public wrapper — timed entry/exit. NULL guards run before
 * clock_gettime so timing only counts work that actually happened. */
int snn_language_bridge_produce(snn_language_bridge_t* bridge,
                                 const float* semantic_intent,
                                 uint32_t intent_dim,
                                 snn_lang_production_result_t* result)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC ||
        !semantic_intent || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_produce: bridge, semantic_intent, or result is NULL");
        return -1;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int rc = bridge_produce_impl(bridge, semantic_intent, intent_dim, result);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    /* Saturating-conservative: if the wall clock somehow steps backward
     * (CLOCK_MONOTONIC shouldn't but defensively guard) skip the bump
     * rather than wrap into a huge unsigned. */
    int64_t elapsed_us =
        (int64_t)(t_end.tv_sec  - t_start.tv_sec ) * 1000000LL +
        (int64_t)(t_end.tv_nsec - t_start.tv_nsec) / 1000LL;
    if (elapsed_us < 0) elapsed_us = 0;
    bridge->stats.produce_total_us  += (uint64_t)elapsed_us;
    bridge->stats.produce_call_count++;

    return rc;
}

/* Tier-4 #16: derived getter for ops dashboards. */
float snn_language_bridge_get_avg_produce_us(const snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return 0.0f;
    if (bridge->stats.produce_call_count == 0) return 0.0f;
    return (float)bridge->stats.produce_total_us /
           (float)bridge->stats.produce_call_count;
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_comprehend: bridge, text, concept_activations, or num_activated is NULL");
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
            // Direct reverse lookup of word_pop → concept_pop bindings.
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_creative_produce: bridge, imagination_activations, or result is NULL");
        return -1;
    }

    bridge->stats.imagination_contributions++;

    // Map imagination activations to concept space
    // Imagination dims are creativity/vividness/coherence signals
    // Scale them by creativity_level to modulate word selection
    uint32_t n_concepts = bridge->num_concept_pops;
    float* concept_acts = nimcp_calloc(n_concepts, sizeof(float));
    if (!concept_acts) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "snn_language_bridge_creative_produce: failed to allocate concept_acts");
        return -1;
    }

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

            // Replay: bindings with high eligibility get strengthened.
            // PA-1 fix (walkthrough round 1): keep word_norm_sq cache
            // consistent with weight updates, otherwise post-sleep cosine
            // scores in decode_spikes go stale.
            if (b->eligibility > 0.1f) {
                float replay_boost = consolidation_strength * b->eligibility * 0.1f;
                float old_w = b->weight;
                b->weight += replay_boost;
                if (b->weight > bridge->config.binding_w_max) {
                    b->weight = bridge->config.binding_w_max;
                }
                norm_update(bridge, b->word_pop, old_w, b->weight);
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
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_get_stats: bridge or stats is NULL");
        return -1;
    }

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

/* Tier-4 #15: copy entire bridge config out for introspection. */
int snn_language_bridge_get_config(const snn_language_bridge_t* bridge,
                                    snn_lang_config_t* out)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !out) return -1;
    *out = bridge->config;
    return 0;
}

float snn_language_bridge_get_blend(const snn_language_bridge_t* bridge)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return 0.0f;
    return bridge->config.spike_blend;
}

const char* snn_language_bridge_get_word_form(
    const snn_language_bridge_t* bridge,
    uint32_t word_pop_index)
{
    if (!bridge || bridge->magic != SNN_LANG_MAGIC) return NULL;
    if (word_pop_index >= bridge->num_word_pops) return NULL;
    if (!bridge->word_pops[word_pop_index].registered) return NULL;
    return bridge->word_pops[word_pop_index].word_form;
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
    if (!bridge || bridge->magic != SNN_LANG_MAGIC || !path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_save: bridge or path is NULL");
        return -1;
    }

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
    if (!path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_load: path is NULL");
        return NULL;
    }

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

    /* Patch A: binding_insert's incremental norm tracking races with the
     * `node->binding = b` overwrite (the on-disk weight may differ from the
     * initial value just inserted). Rebuild from final state. */
    snn_language_bridge_recompute_norms(bridge);

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
    if (!bridge || !attention_weights || num_weights == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_language_bridge_generate_attention_feedback: bridge or attention_weights is NULL");
        return -1;
    }

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

    /* Single pass over all binding buckets — check concept activation per node.
     * O(num_bindings) instead of O(num_concepts * BINDING_HASH_BUCKETS). */
    float total_activation = 0.0f;
    uint32_t max_concept = num_concepts < bridge->num_concept_pops
                         ? num_concepts : bridge->num_concept_pops;

    for (uint32_t b = 0; b < BINDING_HASH_BUCKETS; b++) {
        binding_node_t* node = bridge->binding_buckets[b];
        while (node) {
            uint32_t cp = node->binding.concept_pop;
            if (cp < max_concept) {
                float act = concept_activations[cp];
                if (act >= 0.01f) {
                    total_activation += act;
                    uint32_t wp = node->binding.word_pop;
                    uint32_t sensory_idx = wp % sensory_dim;
                    predicted_sensory[sensory_idx] +=
                        act * node->binding.weight;
                }
            }
            node = node->next;
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
