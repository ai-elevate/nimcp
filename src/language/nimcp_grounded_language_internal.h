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
