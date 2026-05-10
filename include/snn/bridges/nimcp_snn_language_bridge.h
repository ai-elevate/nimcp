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
/* Vocab cap for the SNN language bridge. Bumped 16384 → 32768 once the
 * canonical literary corpus + active-learning grounding pushed produced
 * vocab past the 8K-12K band. Each bridge instance allocates O(cap)
 * arrays (word_pops, word_norm_sq, word_emb_cache, word_emb_norm), so
 * every doubling adds ~12-16MB per bridge — trivial at single-bridge
 * scale. Keep aligned with grounded_language's form_hash modulus —
 * mismatches reproduce the silent-capacity-drop bug logged 2026-04-27. */
#define SNN_LANG_MAX_WORD_POPS        32768
#define SNN_LANG_MAX_ATTACHED_POPS    8
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

/* On-disk file format version markers.
 *
 * V2 (legacy, no version field): [magic:u32][snn_lang_config_t blob][...].
 *   Loader detects V2 because the u32 immediately after magic is
 *   `max_concept_pops` (a small bounded number, ≤ SNN_LANG_MAX_CONCEPT_POPS),
 *   never the V3 sentinel.
 *
 * V3 (PA + MQ knobs persisted explicitly):
 *   [magic:u32]
 *   [SNN_LANG_BRIDGE_FILE_V3_SENTINEL:u32]
 *   [SNN_LANG_BRIDGE_FILE_VERSION_V3:u32 = 3]
 *   [snn_lang_config_t blob]
 *   [ext_block_size:u32]   // size of fields below, in bytes
 *   [temperature:f32]
 *   [top_p:f32]
 *   [produce_topk:u32]
 *   [glove_blend:f32]
 *   [intent_persistence:f32]
 *   [word_feedback:f32]
 *   [enable_snn_spike_routing:u8]
 *   [activation_tau_ms:f32]
 *   [use_hyperbolic_embeddings:u8]
 *   [sampling_mode:i32]
 *   [...remainder: populations, bindings...]
 *
 * The ext_block_size lets future readers seek past unknown trailing bytes
 * if/when more knobs are added. */
#define SNN_LANG_BRIDGE_FILE_V3_SENTINEL  0xFFFFFFFEu
#define SNN_LANG_BRIDGE_FILE_VERSION_V2   2u  /* implicit: no version on disk */
#define SNN_LANG_BRIDGE_FILE_VERSION_V3   3u

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
    /* PA-6: sampling knobs for bridge_produce. temperature == 0 keeps the
     * legacy hard-argmax behavior (every produce_word picks the top non-
     * refractory candidate). temperature > 0 enables softmax sampling
     * over the top-K cosine-scored candidates. top_p applies nucleus
     * truncation (keep candidates whose cumulative probability ≥ top_p,
     * default 1.0 = no truncation). produce_topk is the candidate pool
     * size pulled from decode_spikes per word (default 5; max 32). */
    float    temperature;
    float    top_p;
    uint32_t produce_topk;
    /* PA-5: GloVe-aware blend.  decode_spikes ranks each word by
     *   (1 − glove_blend) · cosine(concept_rates, binding_col[w])
     * + glove_blend       · cosine(concept_rates[:emb_dim], glove_emb[w])
     * when an embedding lookup callback is attached. glove_blend = 0
     * (default) preserves Patch-A binding-only behavior. */
    float    glove_blend;
    /* PA-2: autoregressive recurrent decoder controls. Per-step:
     *   state_t   = (1 − word_feedback) · state_{t-1} + word_feedback · w_{t-1}
     *   concept_t = intent_persistence · intent + (1 − intent_persistence) · state_t
     * with state_0 = intent. word_feedback = 0.3 (legacy hard-coded value)
     * and intent_persistence = 0 (default) preserve the prior behavior
     * exactly — concept_acts evolves entirely through state, which decays
     * the original intent away in favor of recently-picked word reverse-
     * encodings. Set intent_persistence > 0 to keep the prompt's intent
     * present at every step (real autoregressive context, not just bag-
     * of-words ranked by drifted activation). */
    float    intent_persistence;
    float    word_feedback;
    /* PA-3: SNN-spike → bridge STDP wiring. enable_snn_spike_routing
     * (default false) gates the entire path. When enabled, drain
     * spike_output from attached Broca/Wernicke pops per global tick
     * and route through concept_spike / word_spike. activation_tau_ms
     * is the decay time constant for concept_pops[].activation and
     * word_pops[].activation — required to be > 0 to prevent the
     * accumulator runaway that previously drove SNN sparsity to 0.00. */
    bool     enable_snn_spike_routing;
    float    activation_tau_ms;
    /* PA-5+: hyperbolic GloVe distance mode. When true *and* the GloVe
     * blend is active (glove_blend > 0 + emb_lookup attached), the
     * GloVe term in decode_spikes is replaced with
     *   1 / (1 + d_H(query, word_emb))
     * where d_H is the Poincaré-ball hyperbolic distance. Larger ⇒ better,
     * matching the cosine sign convention. Vectors are projected into the
     * Poincaré ball via tanh(‖v‖)·v/‖v‖ at query-time. Default false
     * reproduces Euclidean cosine PA-5 behavior bit-for-bit. */
    bool     use_hyperbolic_embeddings;
    /* PA-6+: produce-time sampling mode dispatch.
     *   0 = legacy / argmax (temperature == 0 path)
     *   1 = softmax + nucleus top-p (PA-6, default when temperature > 0)
     *   2 = quantum-Monte-Carlo MCMC sampling over candidate scores.
     * Mode 1 is auto-selected when temperature > 0 and sampling_mode == 0
     * (preserves PA-6 callers). Mode 2 must be set explicitly. */
    int      sampling_mode;
    /* TIER1-A: optional beam-K decoding in snn_language_bridge_produce.
     * produce_beam_width = 1 (default) preserves greedy / current behavior
     * bit-for-bit. > 1 maintains K parallel beams (each with its own state
     * buffer + cumulative log-prob), expands their top-V candidates per
     * step, prunes to K best by length-normalized score
     *   logprob / token_count^0.6
     * and returns the highest-scoring beam's token sequence. Capped at 16. */
    uint32_t produce_beam_width;
    /* TIER1-B: explicit end-of-utterance token. eos_word_pop == UINT32_MAX
     * (default) disables the mechanism. When set to a registered word_pop,
     * sampling that pop in the produce loop halts production cleanly
     * (without appending the EOS token to the output text). */
    uint32_t eos_word_pop;
    /* TIER1-C: n-gram repetition penalty. Per produce step, before
     * sampling, multiplicatively penalize the score of any candidate
     * whose word_pop appears in the last repetition_window picks:
     *   score *= (1 - repetition_penalty)   per match.
     * repetition_penalty == 0 (default) disables; window == 0 falls back
     * to the default (3) when penalty > 0. Penalty clamped to [0, 1]. */
    float    repetition_penalty;
    uint32_t repetition_window;
    /* TB-7: hard length-control on snn_language_bridge_produce.
     *
     * min_produce_words > 0 suppresses an EOS pick (eos_word_pop) until at
     * least that many real words have been emitted — when EOS would
     * otherwise win, the produce loop falls back to the next-best non-EOS
     * candidate and bumps stats.length_min_suppressions. 0 (default) keeps
     * EOS firing immediately as before.
     *
     * max_produce_words > 0 forces the produce loop to terminate cleanly
     * once that many words have been emitted, regardless of EOS state or
     * the legacy hard cap (32). 0 (default) keeps the legacy 32-word cap.
     *
     * Both clamped at runtime to [0, 1024]. Values must satisfy
     * min ≤ max when both are nonzero (rejected by the setter). */
    uint32_t min_produce_words;
    uint32_t max_produce_words;
    /* TC-11 — GPU-decode scaffold flag (NOT YET IMPLEMENTED).
     *
     * Reserved for a future CUDA port of snn_language_bridge_decode_spikes.
     * The decode pass currently runs on CPU; SIMD muladds + cosine
     * normalize + GloVe blend cost ~50µs at vocab=2K, so the PCIe
     * round-trip would dominate any GPU win until vocab grows past
     * ~16K and the binding map past ~100K entries. Flag stays for ABI
     * forward-compat — the decode hot path checks it, logs a one-shot
     * warning if true, and falls through to the CPU path. When the
     * actual kernel ships, this same flag will route to the GPU path.
     *
     * Default false. Setting it currently has no effect except the
     * one-shot informational log. */
    bool     enable_gpu_decode;
    /* CSTDP — comprehend-driven bridge STDP. When false (default),
     * snn_language_bridge_comprehend is the legacy read-only forward
     * pass: tokenize → walk bindings → return concept activations. The
     * bridge's STDP loop never fires during training-time comprehend,
     * so bridge weights stay near initialization regardless of how
     * many texts the trainer feeds it.
     *
     * When true, comprehend additionally fires concept_pop+word_pop
     * spike pairs in production-direction order (concept first, then
     * word δ ms later) for bindings whose pre-existing weight crosses
     * comprehend_stdp_min_weight AND whose accumulated activation for
     * this comprehend call crosses comprehend_stdp_min_activation,
     * then calls a SCOPED apply_stdp variant that walks only the
     * touched concept/word pop pairs (not the full 1.6M-binding
     * hash). Reinforces existing strong bindings on every comprehend
     * — turns lexicon-side training signal into bridge-weight
     * reinforcement without a separate supervised loop.
     *
     * Default OFF — turn on with
     * snn_language_bridge_set_comprehend_stdp_enabled after verifying
     * the bridge isn't dominated by noise bindings. Stage-1 brains
     * with mostly-noise weights would entrench noise. */
    bool     enable_comprehend_stdp;
    float    comprehend_stdp_min_weight;     /* default 0.05 */
    float    comprehend_stdp_min_activation; /* default 0.10 */
    float    comprehend_stdp_lr_scale;       /* multiplied with stdp_learning_rate, default 0.5 */
    /* EOS stopping criterion — opt-in early termination for the produce
     * loop based on intrinsic "thought-complete" signals rather than a
     * trained EOS token.
     *
     * When enable_eos_stopping is true, after each emitted word the
     * produce loop additionally checks two thresholds and stops early
     * (only after at least min_produce_words have been emitted, so the
     * caller's hard length-control floor still wins):
     *
     *   1. Activation drop: ||concept_acts||_2 (post decay+feedback for
     *      the next step) < eos_min_activation. Rationale: when the
     *      original intent has been "spent" by the recurrent state
     *      drift, the brain has nothing more to say.
     *
     *   2. Coherence drop: the just-emitted word's confidence
     *      (word_result.confidence) < eos_min_confidence. Rationale:
     *      below this, the system is just emitting noise — no clearly
     *      preferred word for the current activation.
     *
     * Either threshold being undershot triggers the stop. Default OFF
     * for backward compatibility — bit-for-bit identical output to
     * pre-EOS behavior. */
    bool     enable_eos_stopping;
    float    eos_min_activation;             /* default 0.05 */
    float    eos_min_confidence;             /* default 0.01 */
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
    /* DK-A+: quantum-Shannon entropy-derived confidence.
     * 1 − (H(p) / log2(K)) over the post-softmax candidate distribution at
     * each produced step, averaged across produced words. Peaked posteriors
     * → near 1.0 (high confidence). Flat posteriors → near 0.0 (low). 0 if
     * no posterior was computed (argmax / mode 0 path). spike_confidence
     * stays for backcompat. */
    float    entropy_confidence;
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
    /* PA-3 walkthrough fix: counter for the collision warning fired by
     * snn_language_bridge_attach_snn_pop when n_neurons > bridge cap.
     * Tested by test_attach_overlarge_pop_warns_but_succeeds. */
    uint64_t attach_collision_warnings;
    /* Tier-4 #16: produce-loop latency telemetry. produce_total_us
     * accumulates wall-clock microseconds spent in bridge_produce across
     * the bridge's lifetime; produce_call_count counts every entry to
     * bridge_produce regardless of return code. Both are measured with
     * CLOCK_MONOTONIC and zeroed by snn_language_bridge_reset_stats. */
    uint64_t produce_total_us;
    uint64_t produce_call_count;
    /* TA-2 — LGSS output-gate telemetry. Bumped only when an
     * lgss_context_t has been attached via
     * snn_language_bridge_set_lgss() and a SAFETY_ACTION_DENY fires
     * against the produced text. Stays 0 when no lgss is attached. */
    uint64_t lgss_outputs_blocked;
    /* TA-4: trigram next-token training counter. Incremented once per
     * successful (a, b) → c update (only when grounded_language_learn_next_token_triple
     * applies its LTP/LTD pass; cold-start no-ops do not count). Zeroed by
     * snn_language_bridge_reset_stats. Reads return 0 when trigram learning
     * was never used. */
    uint64_t total_trigram_updates;
    /* TA-3 — reward-modulated STDP telemetry. Bumped once per
     * snn_language_bridge_apply_stdp pass that found a non-trivial DA
     * gate (config.enable_da_modulation == true AND a neuromodulator
     * system is connected). last_da_modulation captures the multiplier
     * applied this pass — 1.0 when modulation is off / DA is at zero,
     * (1 + da*gain) otherwise. Useful for verifying that dopamine is
     * actually reaching the binding-learning loop. Zeroed by
     * snn_language_bridge_reset_stats. */
    uint64_t da_gated_stdp_passes;
    float    last_da_modulation;
    /* TB-7 — produce-time length-control telemetry. */
    uint64_t length_min_suppressions;
    uint64_t length_max_truncations;
    /* TB-8 — streaming-produce telemetry. stream_callbacks_invoked counts
     * every per-token callback fired; stream_aborts counts produce calls
     * that exited early because the callback returned non-zero. */
    uint64_t stream_callbacks_invoked;
    uint64_t stream_aborts;
    /* TC-11 — cumulative wallclock spent in snn_language_bridge_decode_spikes
     * across all calls in nanoseconds. Combined with total_decode_calls,
     * lets consumers compute average decode latency:
     *   avg_us = (decode_total_ns / total_decode_calls) / 1000
     * Surfaces the actual cost of the CPU decode path so operators can
     * make a cost-benefit call on the GPU port (currently deferred). */
    uint64_t decode_total_ns;
    /* CSTDP — count comprehend calls that fired comprehend-STDP and the
     * number of (concept_pop, word_pop) spike pairs injected. Both stay
     * 0 when enable_comprehend_stdp is false. Useful for verifying the
     * comprehend-driven plasticity path is actually firing during a
     * training run. */
    uint64_t comprehend_stdp_passes;
    uint64_t comprehend_stdp_pairs_fired;
    /* Echo-correct: supervised production training. echo_correct_calls
     * bumps on every snn_language_bridge_echo_correct() invocation;
     * echo_correct_pairs counts the total (concept_pop, target_word_pop)
     * bindings strengthened across all calls; echo_correct_target_misses
     * counts calls where the target word wasn't registered in the bridge
     * (caller fed an unfamiliar word). */
    uint64_t echo_correct_calls;
    uint64_t echo_correct_pairs;
    uint64_t echo_correct_target_misses;
    /* EOS stopping telemetry. total_eos_terminations bumps once per
     * produce call where enable_eos_stopping fired (activation or
     * confidence dropped under threshold after min_produce_words).
     * total_max_truncations bumps once per produce call where the loop
     * exited via the implicit/explicit max-words cap (mirrors the
     * existing length_max_truncations counter for consumers that want
     * a single name). Both stay 0 when EOS stopping is disabled. */
    uint64_t total_eos_terminations;
    uint64_t total_max_truncations;
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

/**
 * @brief Attach an LGSS context for output-gate evaluation in produce.
 *
 * WHAT: Wires the brain's Layered Governance Safety System (LGSS) into
 *       snn_language_bridge_produce(). After the spike cascade has
 *       constructed result->text but before that pointer is returned
 *       to the caller, the bridge evaluates a SAFETY_DOMAIN_GOVERNANCE
 *       action context against the safety KB. A SAFETY_ACTION_DENY
 *       result blocks emission: result->text is freed + zeroed,
 *       result->word_count is reset to 0, stats.lgss_outputs_blocked
 *       is bumped, and an LGSS_ACTION_BLOCKED audit event is emitted.
 * WHY:  Symmetric to the comprehend input gate on grounded_language.
 *       Without an output gate, an attacker who poisons the lexicon or
 *       the imagination drive could exfiltrate disallowed content via
 *       produce even when the input gate is intact.
 * HOW:  Borrowed pointer. NOT owned, NOT serialized. NULL detaches.
 *       Typed void* to avoid pulling the LGSS umbrella header into
 *       language consumers (enum collision risk).
 *
 * @param bridge Bridge handle (NULL → -1)
 * @param lgss   lgss_context_t* opaque (NULL detaches)
 * @return 0 on success, -1 on invalid handle
 */
int snn_language_bridge_set_lgss(
    snn_language_bridge_t* bridge,
    void* lgss);

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

/** PA-4: additive weight update on an existing or new binding.
 *
 * Unlike snn_language_bridge_bind() which takes max(old, new), this adds
 * `delta` to the binding weight (clamped to [W_MIN, W_MAX]) and creates
 * the binding if it did not exist. delta may be negative (LTD). Used by
 * the next-token-loss training path to apply small contrastive updates.
 *
 * @return 0 on success, -1 on failure (bridge invalid, pop out of range,
 *         or allocation failure when creating a new binding).
 */
int snn_language_bridge_strengthen_binding(
    snn_language_bridge_t* bridge,
    uint32_t concept_pop,
    uint32_t word_pop,
    float delta);

/** Echo-correct: supervised "produce this word" learning.
 *
 * Given an intent vector (typically the semantic_vector from a fresh
 * comprehend) and a target word form, strengthens every (concept_pop_i →
 * target_word_pop) binding by lr_scale * stdp_lr * intent[i] * a_plus,
 * with optional DA modulation. The activation gate matches CSTDP — only
 * concept_pops with intent[i] >= comprehend_stdp_min_activation get an
 * update, so we don't reinforce on noise.
 *
 * This is the supervised production-side learning loop the bridge has
 * been missing: training updates word→concept bindings (comprehend
 * direction), but never tells the bridge "say THIS word for THAT
 * intent". Without it the produce path samples from noise across the
 * entire 29K-word lexicon. Calling this every Nth training item with
 * (parent_utterance → target_content_word) closes the loop.
 *
 * Returns:
 *    >= 0  number of bindings actually strengthened
 *      -1  bridge invalid
 *      -2  target word not registered in bridge (lexicon hasn't mirrored
 *          this word yet — caller should ground/comprehend first) */
int snn_language_bridge_echo_correct(
    snn_language_bridge_t* bridge,
    const float* intent,
    uint32_t intent_dim,
    const char* target_word_form,
    float lr_scale);

/** PA-4+: Riemannian / sigmoid-reparameterized update on a binding.
 *
 * Treats the binding weight as `w = σ(u)` for an unconstrained latent u,
 * and applies a natural-gradient step in u-space:
 *
 *     Δu          = lr * grad
 *     Δw_effective = σ'(u) * Δu = w * (1 - w) * lr * grad
 *
 * The chain-rule factor `w*(1-w)` is the diagonal Fisher metric for a
 * Bernoulli-like binding and acts as a natural damping near the [0, 1]
 * boundaries — large |grad| no longer over-clips at the edges. Internally
 * the update is computed in u-space (no log/exp needed in the hot path
 * once we have w) and the result is projected back via σ. The binding is
 * created on positive grad if it didn't exist (with weight = σ(lr*grad/2),
 * starting from u=0 ⇔ w=0.5 then taking a half-step). Maintains the
 * cosine norm cache via norm_update.
 *
 * @return 0 on success, -1 on validation failure or allocation failure.
 */
int snn_language_bridge_strengthen_binding_riemannian(
    snn_language_bridge_t* bridge,
    uint32_t concept_pop,
    uint32_t word_pop,
    float grad);

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

/** Tier-4 #15: copy the entire bridge configuration into *out.
 *
 * Operators today have setters for every PA/MQ knob (blend, sampling,
 * glove_blend, autoregressive, spike_routing, hyperbolic_embeddings,
 * sampling_mode) but no consolidated getter — they must read source to
 * answer questions like "is autoregressive enabled? what's the
 * temperature?". This getter copies the live `snn_lang_config_t` out so
 * RPC consumers can introspect any/all fields uniformly.
 *
 * @return 0 on success; -1 if bridge or out is NULL or magic mismatched. */
int snn_language_bridge_get_config(
    const snn_language_bridge_t* bridge,
    snn_lang_config_t* out);

/** Tier-4 #16: average wall-clock microseconds per bridge_produce call.
 *
 * Computed from stats.produce_total_us / stats.produce_call_count. Returns
 * 0.0 when no produce calls have been made (or bridge is invalid) so the
 * caller can divide blindly without a guard. The raw counters live in
 * snn_lang_stats_t (already exposed by snn_language_bridge_get_stats). */
float snn_language_bridge_get_avg_produce_us(
    const snn_language_bridge_t* bridge);

/** Tier-4 #17: explicit RNG seed for deterministic sampling tests.
 *
 * The bridge's xorshift64* RNG state is seeded at create() from
 * (time XOR pointer-mix), which is fine for production but makes tests
 * that exercise softmax/top-p/q-MC sampling flaky on slow runners. This
 * setter writes `seed` directly into the RNG state. xorshift64 requires a
 * non-zero state, so seed=0 is remapped to 1 (caller does not need to
 * special-case it).
 *
 * @return 0 on success; -1 if bridge is NULL or magic mismatched. */
int snn_language_bridge_set_rng_seed(
    snn_language_bridge_t* bridge,
    uint64_t seed);

/** TA-4: toggle trigram next-token learning.
 *
 * Default OFF — preserves PA-4 bigram-only behavior bit-for-bit. When ON,
 * grounded_language_learn_text_bigrams also walks every (w_t, w_{t+1}) → w_{t+2}
 * trigram and applies a learn_next_token_triple update at half the bigram lr.
 *
 * Runtime-only flag (not persisted in the V3 bridge sidecar) — caller must
 * re-enable after each load. This is intentional: trigram learning is a
 * trainer-curriculum knob, not a bridge-state property.
 *
 * @return 0 on success; -1 if bridge is NULL or magic mismatched. */
int snn_language_bridge_set_trigram_learning_enabled(
    snn_language_bridge_t* bridge,
    bool enabled);

/** TA-4: read the trigram-learning runtime flag. Returns false if bridge
 *  is NULL/invalid (matches the "default OFF" contract). */
bool snn_language_bridge_get_trigram_learning_enabled(
    const snn_language_bridge_t* bridge);

/** TB-8: per-token streaming callback. Invoked once per emitted word
 *  during snn_language_bridge_produce. Returning non-zero aborts the
 *  produce loop early (text accumulated so far is preserved in
 *  result->text on return; word_count reflects what was emitted before
 *  the abort). user_data is passed through unchanged.
 *
 *  IMPORTANT: the callback is a strict observer. It MUST NOT hold the
 *  bridge mutex (the caller already does), MUST NOT call any other
 *  snn_language_bridge_* API, and MUST NOT free word_form (borrowed,
 *  valid only for the duration of the call). Treat it as fire-and-
 *  forget — pipe to TTS, log, update a UI, then return. */
typedef int (*snn_lang_stream_callback_t)(
    uint32_t      word_index,        /* 0-based emission index */
    const char*   word_form,         /* borrowed; valid only during callback */
    uint32_t      word_pop,          /* SNN word population id */
    float         confidence,        /* per-token confidence [0,1] */
    void*         user_data);

/** TB-8: attach (or detach) a per-token streaming callback for produce.
 *
 * The callback + user_data are stored on the bridge as per-call runtime
 * state — they are NOT persisted in the V3 sidecar and NOT in the
 * config struct. Pass cb=NULL to detach. Detaching from inside a
 * callback is safe (the next produce call sees the new state).
 *
 * @param bridge    Bridge handle (NULL → -1).
 * @param cb        Streaming callback, or NULL to detach.
 * @param user_data Opaque pointer passed through to every callback
 *                  invocation. Bridge does not free it.
 * @return 0 on success, -1 on invalid handle.
 */
int snn_language_bridge_set_stream_callback(
    snn_language_bridge_t* bridge,
    snn_lang_stream_callback_t cb,
    void* user_data);

/** TA-4 internal: increment the trigram-update counter. Called from
 *  grounded_language_learn_next_token_triple after a successful LTP/LTD
 *  pass so consumers can read total_trigram_updates from snn_lang_stats_t.
 *  No-op on NULL/invalid bridge. */
void snn_language_bridge_inc_trigram_updates(snn_language_bridge_t* bridge);

/** TA-3: toggle dopamine-modulated binding learning.
 *
 * When enabled AND a neuromodulator system is connected via
 * snn_language_bridge_connect_neuromod(), every snn_language_bridge_apply_stdp
 * pass reads the current dopamine concentration once and applies
 *   weight_change *= 1 + dopamine * config.da_modulation_gain
 * to every binding update in the pass. Default config has the gate ON
 * with gain=50, so the pass amounts to identity (×1.0) when no neuromod
 * is connected and tonic-DA modulation otherwise.
 *
 * Runtime-only flag (not persisted in the V3 bridge sidecar) — caller
 * must re-enable after each load. This is intentional: DA modulation is
 * a trainer-curriculum knob, not a bridge-state property.
 *
 * @return 0 on success; -1 if bridge is NULL or magic mismatched. */
int snn_language_bridge_set_da_modulation_enabled(
    snn_language_bridge_t* bridge,
    bool enabled);

/** TA-3: read the DA-modulation runtime flag. Returns false if bridge
 *  is NULL/invalid. */
bool snn_language_bridge_get_da_modulation_enabled(
    const snn_language_bridge_t* bridge);

/** CSTDP: toggle comprehend-driven bridge STDP on/off at runtime.
 *  Default OFF. When enabled, snn_language_bridge_comprehend fires
 *  scoped concept↔word spike pairs and applies STDP to the touched
 *  bindings — so training-time comprehend reinforces bridge weights
 *  rather than just doing a read-only forward pass.
 *
 *  Returns 0 on success, -1 if bridge is NULL/invalid. */
int snn_language_bridge_set_comprehend_stdp_enabled(
    snn_language_bridge_t* bridge,
    bool enabled);

/** CSTDP: read the comprehend-STDP runtime flag. */
bool snn_language_bridge_get_comprehend_stdp_enabled(
    const snn_language_bridge_t* bridge);

/** TA-3: tune the DA → LR scaling.
 *
 * Caps gain to [0, 200] to prevent pathological multipliers when DA
 * spikes (DA in [0,1] × gain × base_lr). gain=0 disables modulation
 * even when the enable flag is true (multiplier is always 1.0). */
int snn_language_bridge_set_da_modulation_gain(
    snn_language_bridge_t* bridge,
    float gain);

/** Get current spike blend factor */
float snn_language_bridge_get_blend(const snn_language_bridge_t* bridge);

/** Get word form string for a given word population index (NULL if invalid/unregistered) */
const char* snn_language_bridge_get_word_form(
    const snn_language_bridge_t* bridge,
    uint32_t word_pop_index);

/** Set spike blend factor [0=all vector, 1=all spike] */
void snn_language_bridge_set_blend(snn_language_bridge_t* bridge, float blend);

/** PA-6: Configure produce-time sampling.
 *
 * @param temperature  0 = argmax (legacy / default). >0 = softmax sampling
 *                     over top-K cosine-scored candidates with this T.
 * @param top_p        Nucleus truncation in [0,1]. 1.0 = no truncation.
 *                     Smaller values keep only the highest-probability mass.
 * @return 0 on success; -1 if bridge invalid or args out of range.
 */
int snn_language_bridge_set_sampling(snn_language_bridge_t* bridge,
                                      float temperature, float top_p);

/** PA-5: word → embedding lookup callback. Caller fills `out_vec` with
 * the embedding (length `out_dim`) for `word_form` and returns 0; returns
 * -1 if the word is not in the embedding table. Called by decode_spikes
 * (lazily, with caching) when glove_blend > 0. Must be thread-safe with
 * respect to the embedding table. */
typedef int (*snn_lang_word_emb_fn)(void* ctx,
                                     const char* word_form,
                                     float* out_vec,
                                     uint32_t out_dim);

/** PA-5: attach embedding lookup. Until this is called, glove_blend has
 * no effect. emb_dim must equal the prefix length of concept_rates that
 * carries the embedding signal (in NIMCP, semantic_dim == emb_dim, so
 * pass gl->semantic_dim). The first call also allocates the bridge's
 * word_emb cache (sized word_pops_capacity × emb_dim). Pass NULL fn to
 * detach, which frees the cache. */
int snn_language_bridge_set_embedding_lookup(snn_language_bridge_t* bridge,
                                              snn_lang_word_emb_fn fn,
                                              void* ctx,
                                              uint32_t emb_dim);

/** PA-5: set the GloVe blend coefficient at runtime. blend in [0, 1];
 * 0 = binding-only (PA-1 default), 1 = embedding-only ranking.
 * Returns -1 if bridge invalid or blend out of range. */
int snn_language_bridge_set_glove_blend(snn_language_bridge_t* bridge,
                                         float blend);

/** PA-5: invalidate the per-word embedding cache. Call after the
 * embedding table changes (rare — only on retraining or model swap). */
int snn_language_bridge_invalidate_emb_cache(snn_language_bridge_t* bridge);

/** PA-5+: select the GloVe distance metric.
 *
 * @param enabled  false (default) → Euclidean cosine (PA-5 legacy).
 *                 true → Poincaré-ball hyperbolic distance, mapped to
 *                        1 / (1 + d_H(query, word_emb)) so the score
 *                        keeps the "larger is better" convention.
 * Only takes effect when an embedding lookup is attached and
 * glove_blend > 0. Switching the mode invalidates the per-word emb cache
 * (the projected hyperbolic representation is cached alongside).
 *
 * @return 0 on success, -1 if bridge invalid.
 */
int snn_language_bridge_set_hyperbolic_embeddings(snn_language_bridge_t* bridge,
                                                   bool enabled);

/** PA-6+: select produce-time sampling mode.
 *
 * @param mode  0 = legacy (argmax when temperature == 0; softmax+top-p
 *              otherwise — equivalent to PA-6 dispatch).
 *              1 = force softmax + nucleus top-p (PA-6).
 *              2 = quantum-Monte-Carlo MCMC sampling (q-MC).
 * Modes 1 and 2 require a pre-set temperature > 0 to seed the
 * candidate distribution.
 *
 * @return 0 on success, -1 if bridge invalid or mode out of range.
 */
int snn_language_bridge_set_sampling_mode(snn_language_bridge_t* bridge,
                                            int mode);

/** TIER1-A: configure optional beam-K decoding in snn_language_bridge_produce.
 *
 * @param k  Beam width. 1 (default) = greedy / legacy bit-for-bit. > 1 keeps
 *           K parallel beams with cumulative log-prob and length-normalized
 *           ranking (logprob / token_count^0.6). Capped at 16. 0 is treated
 *           as 1.
 *
 * @return 0 on success, -1 if bridge invalid.
 */
int snn_language_bridge_set_beam_width(snn_language_bridge_t* bridge,
                                        uint32_t k);

/** TIER1-B: register the end-of-utterance word_pop. When sampled inside
 * the produce loop, generation halts cleanly (the EOS pop itself is NOT
 * appended to the output text).
 *
 * @param pop  Word_pop index, or UINT32_MAX (default) to disable EOS.
 * @return 0 on success, -1 if bridge invalid.
 */
int snn_language_bridge_set_eos_word_pop(snn_language_bridge_t* bridge,
                                          uint32_t pop);

/** TB-7: hard length control on snn_language_bridge_produce.
 *
 * @param min_words  When > 0, EOS picks (eos_word_pop) are suppressed and
 *                   replaced with the next-best non-EOS candidate until at
 *                   least this many words have been emitted. Each
 *                   suppression bumps stats.length_min_suppressions.
 *                   0 (default) preserves legacy EOS-first behavior.
 * @param max_words  When > 0, the produce loop terminates cleanly after
 *                   this many words have been emitted, overriding the
 *                   legacy implicit 32-word cap. Hitting the cap bumps
 *                   stats.length_max_truncations once. 0 (default) keeps
 *                   the legacy 32-word cap.
 *
 * Validation: NULL bridge → -1. Each value clamped to [0, 1024]. When BOTH
 * are nonzero, min_words > max_words is rejected (-1, no state change).
 * Either value may be 0 (disabled sentinel) regardless of the other.
 *
 * @return 0 on success, -1 on validation failure.
 */
int snn_language_bridge_set_length_control(snn_language_bridge_t* bridge,
                                            uint32_t min_words,
                                            uint32_t max_words);

/** TB-7: read the current length-control settings.
 *
 * Copies config.min_produce_words / config.max_produce_words into the
 * out-pointers (NULL out-pointers are skipped). Returns -1 on invalid
 * bridge.
 */
int snn_language_bridge_get_length_control(const snn_language_bridge_t* bridge,
                                            uint32_t* min_words,
                                            uint32_t* max_words);

/** TIER1-C: configure the n-gram repetition penalty applied per produce step.
 *
 * Before scoring/sampling candidates, every candidate whose word_pop appears
 * in the last `window` picks has its score multiplied by `(1 - penalty)` for
 * each match. With penalty = 0 (default) the path is a no-op.
 *
 * @param penalty  In [0, 1]. 0 disables. Clamped if out of range.
 * @param window   Look-back length. 0 falls back to 3 when penalty > 0.
 * @return 0 on success, -1 if bridge invalid.
 */
int snn_language_bridge_set_repetition_penalty(snn_language_bridge_t* bridge,
                                                 float penalty,
                                                 uint32_t window);

/** PA-2: configure the autoregressive recurrent decoder.
 *
 * @param intent_persistence  In [0, 1]. 0 (default) = legacy behavior — the
 *                            original intent decays exponentially across
 *                            the produce loop as state evolves toward the
 *                            most recent words. 1 = pure non-recurrent
 *                            (intent stays full-strength every step,
 *                            ignoring state). Values in between blend.
 * @param word_feedback       In [0, 1]. How aggressively each just-picked
 *                            word reshapes the recurrent state. Default
 *                            0.3 (matches the legacy hard-coded blend).
 *                            Higher = stronger context dependence.
 * @return 0 on success; -1 if bridge invalid or args out of range.
 */
int snn_language_bridge_set_autoregressive(snn_language_bridge_t* bridge,
                                            float intent_persistence,
                                            float word_feedback);

/** PA-3: role of an attached SNN population. CONCEPT routes spikes through
 * snn_language_bridge_concept_spike (Wernicke / arcuate / comprehension
 * tier). WORD routes through snn_language_bridge_word_spike (Broca /
 * production tier). */
typedef enum {
    SNN_LANG_POP_ROLE_CONCEPT = 0,
    SNN_LANG_POP_ROLE_WORD    = 1
} snn_lang_pop_role_t;

/** PA-3: configure SNN-spike routing.
 *
 * @param enabled    Master gate. False (default) disables the entire path
 *                   to prevent the previously-observed sparsity collapse.
 * @param tau_ms     Per-tick decay time constant for activation accumulators.
 *                   Must be > 0 when enabled is true. Suggested 200 ms.
 * @return 0 on success; -1 if bridge invalid or tau invalid given enabled.
 */
int snn_language_bridge_set_snn_spike_routing(snn_language_bridge_t* bridge,
                                               bool enabled, float tau_ms);

/** PA-3: register an SNN population for spike routing through this bridge.
 * Up to SNN_LANG_MAX_ATTACHED_POPS attached at once. Re-attaching with a
 * known pop_id updates the role.
 * @return 0 on success; -1 if bridge invalid or attach table full. */
int snn_language_bridge_attach_snn_pop(snn_language_bridge_t* bridge,
                                        int snn_pop_id,
                                        uint32_t n_neurons,
                                        snn_lang_pop_role_t role);

/** PA-3: drain spike_output for one attached SNN pop and route each fired
 * neuron through concept_spike or word_spike according to its registered
 * role. Bridge-side neuron→pop mapping is `neuron_idx % MAX_*_POPS`.
 * @param spike_output   spike_output[] from the SNN pop, length n_neurons.
 * @param current_time_ms wallclock timestamp of the current tick (for STDP).
 * @return 0 on success; -1 if pop_id not registered or bridge invalid. */
int snn_language_bridge_drain_pop_spikes(snn_language_bridge_t* bridge,
                                          int snn_pop_id,
                                          const float* spike_output,
                                          uint32_t n_neurons,
                                          float current_time_ms);

/** PA-3: per-tick decay on activation accumulators. Must be called once
 * per global tick (cadence ~10 ms) when spike routing is enabled, or the
 * activations diverge.
 * @return 0 on success; -1 if bridge invalid. */
int snn_language_bridge_tick(snn_language_bridge_t* bridge, float dt_ms);

/** PA-3: iterate attached SNN pops. Returns -1 on invalid bridge or out-of-
 * range index; 0 with `*pop_id < 0` indicates an empty slot to skip. */
int snn_language_bridge_get_attached_pop(const snn_language_bridge_t* bridge,
                                          uint32_t index,
                                          int* out_pop_id,
                                          uint32_t* out_n_neurons,
                                          snn_lang_pop_role_t* out_role);

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
