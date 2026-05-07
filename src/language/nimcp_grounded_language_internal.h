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
 *       field-by-field, which the public API doesn't allow.
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
 * @brief Multi-turn discourse state — small ring buffer of recent
 *        conversational turns. The single rolling `gl_context_t` above
 *        is a recency-weighted blend of these; this buffer keeps the
 *        per-turn breakdown so future coherence / topic-shift checks
 *        can compare individual turns rather than a smeared running
 *        vector.
 *
 * `semantic_vector` is gl-owned (malloc'd at push, freed on destroy /
 * eviction). `head` is the index of the *oldest* turn when the buffer
 * has wrapped (count == capacity); otherwise turns occupy [0, count).
 * Capacity is bounded by GL_DISCOURSE_MAX_TURNS at compile time and
 * runtime-clampable via grounded_language_set_discourse_capacity.
 */
#define GL_DISCOURSE_MAX_TURNS 8

typedef struct {
    float*   semantic_vector;         /**< Owned [semantic_dim], NULL = empty slot */
    uint32_t word_count;
    uint64_t timestamp_ms;
    bool     is_user;
} gl_discourse_turn_t;

typedef struct {
    gl_discourse_turn_t turns[GL_DISCOURSE_MAX_TURNS];
    uint8_t  head;                    /**< Oldest-turn index when full; 0 otherwise */
    uint8_t  count;                   /**< Currently used slots */
    uint8_t  capacity;                /**< 1..GL_DISCOURSE_MAX_TURNS */
} gl_discourse_state_t;

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
     * destroy. phrases_dropped counts the times capacity overflow
     * forced a new phrase to be skipped (without LRU eviction those
     * would just be lost; with eviction they're swapped in for the
     * least-frequent existing entry). */
    gl_phrase_t*         phrases;
    uint32_t             phrase_count;
    uint64_t             phrases_evicted;

    /* D7 — set true while a persistence rehydrate is in progress. The
     * NEW_WORD event publisher and the SNN-bridge spike synthesizer both
     * gate on this. ~30K rehydrated entries per resume otherwise blast
     * NEW_WORD to every subscriber + drive synthesized spike traces over
     * saved STDP weights. Init/runtime always sees is_loading=false. */
    bool                 is_loading;

    /* D1 — process-monotonic virtual time fed to the SNN language bridge
     * by mirror_binding_to_bridge so each synthesized spike pair lives in
     * its own STDP window. Float ms; advanced by 50ms per binding event.
     * Survives across save/load by design — the bridge keeps running
     * windowed STDP after resume. */
    float                snn_virtual_time_ms;

    /* Tier-2 #3 negation polarity. When enabled (default true), comprehend
     * detects negation cues in the tokenized text and inverts the sign of
     * activation_levels[] for the next non-function content word within a
     * GL_NEGATION_WINDOW lookahead. Counters (negation_events,
     * sense_resolutions) live on gl_stats_t — getter copies them out
     * with the rest of the public stats. */
    bool                 enable_negation_inversion;

    /* Tier-2 #6 word-sense disambiguation. When enabled (default false —
     * preserves legacy behaviour), comprehend's per-binding activation is
     * weighted by cosine(intent, binding_concept_features); the most-
     * relevant sense gets the highest weight and off-sense bindings damp.
     * Off-sense weight floor lives at GL_SENSE_OFF_DAMP. */
    bool                 enable_sense_disambiguation;

    /* Tier-2 #7 multi-turn discourse state. Backbone for future coherence
     * / topic-shift checks. The legacy gl->context.context_vector remains
     * the single rolling vector — when discourse turns are pushed it gets
     * recomputed as a recency-weighted blend across the buffer. */
    gl_discourse_state_t discourse;

    /* Engram integration (read-only mode). Borrowed pointer to the
     * brain-level engram system, attached via
     * grounded_language_set_engram_system(). When `engram_enabled` is
     * true and the pointer is non-NULL, comprehend lays down a memory
     * trace at the end of each pass (encode) and consults the engram
     * store mid-pass to blend recalled activations into the result
     * (recall). The pointer is NOT owned, NOT serialized — re-attach
     * after every brain init / load. Default: pointer NULL, flag false. */
    void* engram_system;
    bool  engram_enabled;

    /* TA-2 LGSS attach (currently a stub field — gates not wired in
     * comprehend yet). Borrowed pointer to brain->lgss; set via
     * grounded_language_set_lgss(). NULL = no gates active. The full
     * gate wiring + evaluator forward-decl lives in the comprehend
     * function; this field exists to satisfy referencing call sites
     * that landed ahead of the full TA-2 commit. */
    void* lgss;
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
