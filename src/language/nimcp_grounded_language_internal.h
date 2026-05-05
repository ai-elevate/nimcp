/**
 * @file nimcp_grounded_language_internal.h
 * @brief Internal layout of grounded_language_t — shared between the
 *        primary implementation and the persistence sidecar.
 * @date 2026-05-04
 *
 * WHAT: Exposes `struct grounded_language` and `gl_context_t` so that
 *       sibling files in src/language/ can serialize / restore lexicon
 *       state without going through the public opaque-handle API.
 * WHY:  The public API (grounded_language.h) keeps the struct opaque
 *       on purpose — external callers shouldn't depend on the layout.
 *       But the persistence sidecar needs to walk every lexicon entry
 *       and template field-by-field, which the public API doesn't allow.
 *       Pulling the struct into an *internal* header keeps the opaque
 *       contract for the rest of the codebase while letting persistence
 *       and the primary impl share the layout via the same #include.
 * HOW:  Mirrors the orchestrator-internal pattern already established
 *       in src/language/nimcp_language_orchestrator_internal.h.
 *
 * NOTE: Do NOT include this header from anywhere outside src/language/.
 *       It is not installed; relying on it from another module would
 *       reintroduce the layout coupling we deliberately avoid.
 */

#ifndef NIMCP_GROUNDED_LANGUAGE_INTERNAL_H
#define NIMCP_GROUNDED_LANGUAGE_INTERNAL_H

#include "language/nimcp_grounded_language.h"
#include "cognitive/memory/nimcp_semantic_memory.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Context buffer for tracking conversation state.
 *
 * Mirrors the (formerly file-local) typedef in nimcp_grounded_language.c.
 */
typedef struct {
    float*   context_vector;          /**< Running context [semantic_dim] */
    float    context_strength;        /**< How strong current context is */
    uint32_t words_in_context;        /**< Words seen in current context */
    uint64_t last_concepts[16];       /**< Recent concept activations */
    uint32_t last_concept_count;      /**< Number of recent concepts */
} gl_context_t;

/**
 * @brief Internal state of the grounded language system.
 *
 * Layout previously lived in nimcp_grounded_language.c. Moved here so the
 * persistence sidecar can iterate the lexicon and templates directly.
 */
struct grounded_language {
    /* Lexicon: word forms -> concept bindings */
    gl_lexicon_entry_t** lexicon;       /**< Hash table of lexicon entries */
    uint32_t             lexicon_size;  /**< Hash table size */
    uint32_t             vocab_count;   /**< Number of words */
    gl_lexicon_entry_t** vocab_list;    /**< Linear list for iteration */

    /* Syntactic templates */
    gl_template_t*       templates;     /**< Array of syntactic templates */
    uint32_t             template_count;
    uint32_t             template_capacity;

    /* Semantic integration */
    uint32_t             semantic_dim;  /**< Dimension of semantic vectors */
    semantic_memory_system_t* semantic_memory; /**< Brain's concept store (not owned, not serialized) */

    /* Context state */
    gl_context_t         context;       /**< Current conversation context */

    /* Cross-modal connections */
    void*                visual_ctx;
    void*                auditory_ctx;
    void*                speech_ctx;
    void*                column_pool;
    void*                emotional_ctx;

    /* Learning parameters */
    float                hebbian_lr;    /**< Hebbian learning rate */
    float                decay_rate;    /**< Association decay rate */

    /* Statistics */
    gl_stats_t           stats;

    /* RNG state for sampling */
    uint64_t             rng_state;

    /* SNN language bridge (optional) */
    struct snn_language_bridge* snn_bridge;

    /* Memory-system attachments (optional, all opaque pointers).
     * Wired at brain init via grounded_language_connect_*; consulted on
     * every successful grounding event so the trained vocabulary
     * participates in working/episodic/long-term memory rather than
     * sitting in an isolated lexicon. None are owned or serialized. */
    void*                working_memory;     /**< working_memory_t* */
    void*                episodic_replay;    /**< nimcp_episodic_replay_t* */
    void*                hippocampus;        /**< hippocampus_adapter_t* */

    /* Region-adapter attachments — when connected, every newly-created
     * lexicon entry is mirrored into the broca production lexicon and
     * the wernicke comprehension lexicon. This keeps the three lexicons
     * in sync with grounded_language as the single source of truth. */
    void*                broca_adapter;      /**< broca_adapter_t* */
    void*                wernicke_adapter;   /**< wernicke_adapter_t* */

    /* NLP frontend attachments — embedding layer + BPE tokenizer used
     * to enrich semantic_vector and recover from totally-OOV words.
     * All optional; comprehend skips the corresponding stages when NULL. */
    void*                embeddings;         /**< embedding_layer_t* */
    uint32_t             emb_dim;            /**< embedding output dim */
    void*                tokenizer;          /**< tokenizer_t* (BPE) */
    gl_word_to_id_fn     word_to_id_fn;      /**< word→id mapper */
    void*                word_to_id_ctx;     /**< opaque ctx for fn */

    /* Per-network bridges — comprehend's semantic_vector gets broadcast
     * to each attached network's forward pass; the response magnitudes
     * feed back as confidence modulation (parallel to cortex modulation). */
    void*                lnn_layer;          /**< lnn_layer_t* */
    void*                cortex_cnn_proc;    /**< cortex_cnn_processor_t* */
    void*                fno_proc;           /**< fno_audio_processor_t* */
    gl_ann_predict_fn    ann_predict_fn;     /**< caller-owned ANN predictor */
    void*                ann_ctx;            /**< opaque ctx for ann_predict_fn */
    /* Last broadcast results — read by comprehend modulation hook. */
    float                last_lnn_mag;
    float                last_cnn_mag;
    float                last_fno_mag;
    float                last_ann_mag;

    /* Cognitive + region subscriber bus — flat array, ctx is the dedup
     * key. Cap covers ~10 cognitive modules + 5 anatomical regions +
     * the hemispheric language bridge, with headroom for future waves.
     * Sorted by descending priority on insert so fire walks once.
     * type_mask gates per-event-type delivery (#3 priority + #4 mask). */
    #define GL_MAX_SUBSCRIBERS 24
    gl_event_callback_t  subscribers[GL_MAX_SUBSCRIBERS];
    void*                subscriber_ctxs[GL_MAX_SUBSCRIBERS];
    uint32_t             subscriber_masks[GL_MAX_SUBSCRIBERS];
    int8_t               subscriber_priorities[GL_MAX_SUBSCRIBERS];
    uint32_t             subscriber_count;

    /* Re-entry guard (#11). Set true while gl_fire_event is iterating;
     * a subscriber that calls back into a function that would re-fire
     * an event hits this and the inner fire becomes a no-op (with a
     * dropped-events counter increment). Prevents infinite recursion
     * if a wrapper accidentally drives ground/comprehend/produce. */
    bool                 in_fire_event;
    uint64_t             events_dropped_reentry;

    /* Forgetting-curve telemetry (#15). Bumped by sleep_consolidate
     * each time an entry is decayed below retention. The ring rotates
     * once per call to gl_forgetting_telemetry_rotate (called by the
     * sleep_consolidate path on hour rollover). */
    uint32_t             decayed_ring[24];
    uint8_t              decayed_ring_head;          /* index of current bucket */
    uint64_t             decayed_all_time;

    /* Dialect/accent context (#14). Empty string = dialect-agnostic
     * mode. Setter truncates to GL_MAX_DIALECT_LEN-1 + NUL. Used by
     * fuzzy lookup as a tie-break preference and surfaced via probe
     * metrics for observability. */
    char                 context_dialect[GL_MAX_DIALECT_LEN];

    /* Compositional phrases (#9). Fixed-capacity table accumulated
     * lazily by learn_from_text. semantic_vec is gl-owned; cleared on
     * destroy. */
    gl_phrase_t*         phrases;
    uint32_t             phrase_count;
};

/**
 * @brief Find an existing lexicon entry by form, or create a fresh one.
 *
 * Used by the primary implementation and by the persistence sidecar's
 * load path to rehydrate entries while keeping the hash table in sync
 * with vocab_list. Lower-cased on entry. Returns NULL on alloc failure
 * or vocab cap exhaustion.
 */
gl_lexicon_entry_t* gl_internal_lexicon_find_or_create(
    grounded_language_t* gl,
    const char* word);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_GROUNDED_LANGUAGE_INTERNAL_H */
