/**
 * @file nimcp_grounded_language_persistence.h
 * @brief Sidecar persistence for the grounded_language module.
 * @date 2026-05-04
 *
 * WHAT: gl_persistence_save() / gl_persistence_load() — flatten the
 *       grounded_language lexicon, syntactic templates, and learning
 *       state to a small binary file alongside the brain checkpoint.
 * WHY:  Without this the trained vocabulary (vocab_list, lexicon hash
 *       table, learned word-class assignments, distributional context
 *       vectors, valence/arousal, syntactic templates) is wiped on every
 *       daemon restart. The language module then stays at zero bindings
 *       no matter how long training runs.
 * HOW:  Mirrors the .kg sidecar pattern. Magic-prefixed binary header,
 *       version field for forward compatibility, length-prefixed strings.
 *       Defensive load: missing file, bad magic, version mismatch,
 *       oversized records — all return -1 cleanly without crashing.
 *
 * SCOPE: Persists only state that survives a restart usefully —
 *        lexicon (forms, bindings, distributional context, valence,
 *        arousal, learned class), templates, stats, rng_state, semantic_dim.
 *        Skipped: semantic_memory pointer (managed elsewhere), the
 *        visual_ctx/auditory_ctx/speech_ctx/column_pool/emotional_ctx
 *        pointers (runtime wiring redone at brain init), the snn_bridge
 *        pointer (runtime, reattached on init).
 *
 * THREAD SAFETY: Caller is responsible for serializing save/load against
 *                concurrent mutation of the lexicon. Currently the brain
 *                save path holds save_load_mutex which suffices.
 */

#ifndef NIMCP_GROUNDED_LANGUAGE_PERSISTENCE_H
#define NIMCP_GROUNDED_LANGUAGE_PERSISTENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decl — keeps the public API SRP-clean: callers don't need to
 * pull in the full grounded_language header just to call save/load. */
struct grounded_language;

/**
 * @brief Save grounded_language state to a sidecar file (best-effort).
 *
 * Writes header magic "NIMCP_GL", version, and the field-by-field
 * serialized lexicon + templates + stats. Truncates / overwrites the
 * destination. Does NOT fsync — caller decides durability.
 *
 * @param gl   Handle to the grounded_language module (NULL: returns -1)
 * @param path Destination file path (NULL: returns -1)
 * @return 0 on success, -1 on any I/O or argument error.
 */
int gl_persistence_save(const struct grounded_language* gl, const char* path);

/**
 * @brief Load grounded_language state from a sidecar file (defensive).
 *
 * Replays the lexicon and templates into the supplied (already-created)
 * handle. The hash table is rebuilt from the loaded vocab_list so that
 * comprehend/respond paths can immediately look words up.
 *
 * The function tolerates: missing file, bad magic, future version,
 * truncated file, oversized strings, malformed records — all return -1
 * without modifying or destroying the handle.
 *
 * @param gl   Handle to the grounded_language module (must be non-NULL
 *             and previously created via grounded_language_create()).
 * @param path Source file path.
 * @return 0 on success, -1 if the file is missing/corrupt or arguments
 *         are invalid.
 */
int gl_persistence_load(struct grounded_language* gl, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_GROUNDED_LANGUAGE_PERSISTENCE_H */
