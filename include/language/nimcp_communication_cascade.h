/**
 * @file nimcp_communication_cascade.h
 * @brief Multi-region production cascade for language output.
 *
 * Reframes language production as a coordinated multi-region cascade,
 * not a bridge softmax. Each stage reads state from one or more
 * cognitive modules (hypothalamus, PFC, WM, ToM, hippocampus, semantic
 * memory) and contributes a section of the production_cascade_state_t.
 *
 * Phase 2A skeleton: all 9 stages exist; intent-formation stages (1-5)
 * read real module state and combine into a weighted content_intent
 * vector; output stages (6-9) currently pass through to the SNN
 * language bridge while we build the GL→Broca lexicon mirror (Phase
 * 2C). Each stage is independently testable and can be enabled or
 * disabled via the cascade config.
 *
 * Design: see docs/claude/communication-cascade-plan.md.
 */
#ifndef NIMCP_COMMUNICATION_CASCADE_H
#define NIMCP_COMMUNICATION_CASCADE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "core/brain/regions/broca/nimcp_pragmatics_processor.h"  /* speech_act_type_t */
#include "core/brain/regions/hippocampus/nimcp_hippocampus_adapter.h" /* retrieval_result_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decl — full struct lives in nimcp_brain_internal.h. */
struct brain_struct;
typedef struct brain_struct* brain_t;

/* Forward decl — full struct lives in nimcp_snn_language_bridge.h.
 * Cascade Stage 11 (self-train) calls the bridge's plasticity API. */
struct snn_language_bridge;

/* Per-stage enable bits. Default (0) = all stages on. */
typedef enum {
    CASCADE_STAGE_WERNICKE      = 1u << 0,  /* Stage 0 — input comprehension */
    CASCADE_STAGE_DRIVE         = 1u << 1,
    CASCADE_STAGE_GOAL          = 1u << 2,
    CASCADE_STAGE_LISTENER      = 1u << 3,
    CASCADE_STAGE_EPISODIC      = 1u << 4,
    CASCADE_STAGE_CONTENT       = 1u << 5,
    CASCADE_STAGE_LEXICAL       = 1u << 6,
    CASCADE_STAGE_SYNTACTIC     = 1u << 7,
    CASCADE_STAGE_SELF_COMP     = 1u << 8,  /* Stage 8 (Phase 2D-B) — Wernicke validates own output */
    CASCADE_STAGE_PHONOLOGICAL  = 1u << 9,
    CASCADE_STAGE_MOTOR         = 1u << 10,
    CASCADE_STAGE_SELF_FEEDBACK = 1u << 11, /* Stage 9 — write produced utterance back to WM + cognitive bus */
    CASCADE_STAGE_SPEECH_REPAIR = 1u << 12, /* Stage 10 (Item 5) — perturbation-retry on low self_match */
    CASCADE_STAGE_PROSODY       = 1u << 13, /* Wave 2 Item 9 — FNO-shaped prosodic contour (F0/dur/intensity) */
    CASCADE_STAGE_SELF_TRAIN    = 1u << 14, /* Stage 11 (Wave 2 Item #10) — reward-modulated bridge learning */
    CASCADE_STAGE_ALL           = 0x7FFF /* bits 0..14 */
} cascade_stage_mask_t;

/* Single struct accumulates stage outputs. Allocated by caller; the
 * orchestrator allocates content_intent and utterance, owns them, and
 * frees them on cleanup. */
typedef struct {
    /* Stage 0: Wernicke comprehension of input prompt — phrase
     * structure analysis, speech-act classification, ambiguity detection.
     * Empty/zero values when prompt is NULL or Wernicke isn't attached. */
    bool     wernicke_parsed;          /* true if syntactic_parse_sentence succeeded */
    bool     prompt_is_question;       /* derived from wh-words / aux inversion / "?" */
    bool     prompt_is_imperative;     /* missing-subject + leading verb */
    bool     prompt_is_garden_path;    /* set when parser flagged ambiguity */
    uint32_t prompt_word_count;        /* # words after tokenization */
    float    prompt_complexity;        /* 0..1, normalized syntactic complexity */
    /* Subject / verb / object slots — populated when the parse identifies
     * thematic roles. Each is either a tokenized word string from the
     * prompt or "" if the role wasn't filled. Phase 2D-A v1 uses simple
     * head-finding rather than full thematic-role assignment. */
    char     prompt_subject[32];
    char     prompt_verb[32];
    char     prompt_object[32];

    /* Stage 1: Drive — read from hypothalamus / insula / amygdala */
    float    drive_magnitude;     /* 0..1, strength of urge to communicate */
    float    drive_valence;       /* -1..1, approach vs avoid */
    float    drive_arousal;       /* 0..1, calm vs urgent */
    uint8_t  dominant_drive;      /* hypo_drive_type_t value */

    /* Stage 2: Goal — read from PFC + WM */
    speech_act_type_t act_type;
    uint64_t target_concept_id;       /* primary concept being expressed */
    uint64_t topic_concept_ids[8];    /* related concepts from WM + goals */
    uint32_t topic_count;
    float    goal_priority;            /* 0..1 from PFC top goal */

    /* Stage 3: Listener — read from ToM */
    bool     listener_known;
    float    listener_belief_confidence; /* tom_belief_t.confidence */
    float    listener_emotion_valence;   /* simplified from tom_emotion_t */
    float    audience_familiarity;       /* 0..1 */

    /* Stage 4: Episodic — read from hippocampus (similarity search) */
    uint64_t episodic_concept_ids[16];
    float    episodic_relevances[16];
    uint32_t episodic_count;
    /* Phase 2B: actual retrieved memories. Owned by cascade state —
     * cascade_state_cleanup frees memories[] and similarities[] inside.
     * Used by stage_content to lift feature vectors into the intent. */
    retrieval_result_t episodic_retrieval;

    /* Stage 5: Content — combined intent vector that drives the bridge */
    float*   content_intent;       /* allocated; semantic_dim entries */
    uint32_t content_dim;
    float    content_confidence;   /* 0..1 — how cohered the cascade is */

    /* Stages 6-9: filled by lexical/syntactic/phonological/motor */
    char*    utterance;            /* allocated; final text */
    uint32_t word_count;
    float    fluency;
    float    syntactic_validity;   /* 0 if Broca rejected, 1 if grammatical */

    /* Stage 8 (Phase 2D-B): Wernicke validation of brain's own output —
     * the sensorimotor loop. After Broca produces, we comprehend the
     * utterance via Wernicke + grounded_language_comprehend and compare
     * the derived semantic vector to content_intent. High match = the
     * production path successfully encoded the intent. Low match = the
     * brain said something but it's not what it meant. */
    bool     self_parsed;             /* true if Wernicke parse_sentence succeeded */
    float    self_complexity;         /* 0..1, syntactic complexity of own output */
    float    self_match;              /* 0..1, cosine sim between intent and re-comprehended utterance */
    float    self_grammaticality;     /* 0 / 0.5 / 1.0 — derived from parse + grammaticality flag */

    /* Diagnostics */
    uint32_t stages_completed;
    uint32_t stages_failed;
    uint32_t stages_skipped;       /* per-stage skip mask */
    char     failure_reason[128];

    /* Pragmatics — APPENDED at end of struct to avoid ABI shift on
     * existing fields. Set by cascade Stage 2 (Goal) when Broca's
     * pragmatics processor classifies the prompt as an indirect speech
     * act ("Can you pass the salt?" surface=question, indirect=REQUEST).
     * When true, act_type is overridden to SPEECH_ACT_REQUEST or
     * SPEECH_ACT_COMMAND so downstream stages produce a request-shaped
     * response rather than a literal yes/no answer. */
    bool     pragmatic_is_indirect;

    /* Stage 10 (Item 5) — speech repair retry. APPENDED at end of struct
     * to avoid ABI shift on existing fields. When self_match (Phase 2D-B)
     * falls below a threshold, the orchestrator re-runs lexical+syntactic
     * +self_comp with a small Gaussian perturbation to content_intent and
     * keeps the best-scoring candidate. Diagnostics:
     *   repair_attempts   — how many retry rounds were executed (0..N)
     *   best_self_match   — best score seen across retries (>= original)
     *   best_utterance    — heap-allocated copy of best-scoring utterance;
     *                       owned by cascade state, freed in cleanup. */
    uint32_t repair_attempts;
    float    best_self_match;
    char*    best_utterance;

    /* Stage 9 (Wave 2 Item 7) — phonological output. APPENDED at end of
     * struct to avoid ABI shift on existing fields. Converts state->utterance
     * (Broca-rendered text from Stage 7) into a phoneme sequence using a
     * lightweight rule-based English G2P, then runs the sequence through
     * Broca's phonological_processor_t to obtain syllable diagnostics.
     *
     * Diagnostics:
     *   phoneme_count     — number of phonemes emitted (0 when skipped)
     *   phoneme_sequence  — heap-allocated array of uint8_t phoneme codes
     *                       (ASCII-letter encoding: 'a'..'z' for vowels +
     *                       consonants, ' ' for word boundaries, plus
     *                       digraph sentinels '$' (sh), '&' (ch), '#' (th),
     *                       '@' (ng), '%' (ph→f)). Owned by cascade state,
     *                       freed in cleanup.
     *   syllable_count    — syllables generated by the phonological processor
     *                       (0 if syllabification skipped or failed).
     *   phon_voiced_ratio — fraction of phonemes flagged voiced (0..1);
     *                       a coarse diagnostic for downstream prosody. */
    uint32_t phoneme_count;
    uint8_t* phoneme_sequence;
    uint32_t syllable_count;
    float    phon_voiced_ratio;

    /* Stage 11 (Wave 2 Item 9) — prosodic contour. APPENDED at end of
     * struct to avoid ABI shift on existing fields. Maps the cascade's
     * emotional/syntactic feature vector (drive_arousal, drive_valence,
     * act_type, prompt_is_question, self_grammaticality) onto a
     * per-syllable F0/duration/intensity contour using a spectral
     * (FNO-style) basis. Intent → contour rules:
     *   - drive_arousal     ↑ → wider pitch range (Hz_peak - Hz_floor)
     *   - drive_valence     ↑ → higher baseline F0 (approach prosody)
     *   - act_type=QUESTION → final-rise contour (last syllable +30%)
     *   - act_type=COMMAND  → emphasis-front, falling tail
     *   - act_type=DECLARE  → declination (gradual fall)
     *   - self_grammaticality → confidence boost on intensity dB
     *
     * Allocated arrays are owned by the cascade state — cleanup frees them.
     * All three vectors are sized to prosody_syllable_count entries.
     *
     * Diagnostics:
     *   prosody_pitch_hz       — per-syllable F0 in Hz (typical 80..400)
     *   prosody_duration_ms    — per-syllable duration in ms (typical 50..200)
     *   prosody_intensity_db   — per-syllable amplitude in dB (typical 50..90)
     *   prosody_syllable_count — number of syllables in contour (0 when skipped)
     *   prosody_mean_f0        — mean of pitch_hz, summary diagnostic
     *   prosody_pitch_range    — max(pitch_hz) - min(pitch_hz), summary diagnostic */
    float*   prosody_pitch_hz;
    float*   prosody_duration_ms;
    float*   prosody_intensity_db;
    uint32_t prosody_syllable_count;
    float    prosody_mean_f0;
    float    prosody_pitch_range;

    /* Stage 12 (Wave 2 Item #10) — reward-modulated SNN bridge training.
     * APPENDED at end of struct to avoid ABI shift on existing fields.
     * When the runtime flag brain->cascade_self_train_enabled is true AND
     * stage_self_comprehension produced a valid self_match, the cascade
     * computes reward = (self_match - baseline), updates the per-brain EMA
     * baseline, and applies snn_language_bridge_echo_correct() to each
     * produced word with lr_scale proportional to the reward sign and
     * magnitude. Diagnostics surface here for trainers + tests:
     *   train_applied — true iff the bridge plasticity hook ran.
     *   train_reward  — the reward signal that was actually applied
     *                   (self_match - baseline_before_update). May be
     *                   negative when the brain underperforms its
     *                   recent average. */
    bool  train_applied;
    float train_reward;
} production_cascade_state_t;

/**
 * @brief Run the full 9-stage production cascade and produce an utterance.
 *
 * @param brain          Brain handle (internal pointer).
 * @param prompt_or_null Optional input text. If non-null, the prompt is
 *                       comprehended first and seeds stage 1's intent
 *                       vector — mimics responding to a question. If
 *                       null, the cascade runs purely from internal
 *                       state — mimics spontaneous speech.
 * @param stage_mask     Bitmask of cascade_stage_mask_t enabling
 *                       specific stages. 0 = CASCADE_STAGE_ALL.
 * @param out_state      Caller-owned struct populated by the cascade.
 *                       Free with cascade_state_cleanup().
 *
 * @return 0 on success; -1 on fatal failure. Per-stage failures are
 *         non-fatal — they're recorded in out_state->stages_failed and
 *         the cascade continues with whatever signal is available.
 */
int communication_cascade_run(
    brain_t brain,
    const char* prompt_or_null,
    uint32_t stage_mask,
    production_cascade_state_t* out_state);

/** Free heap-owned fields in the cascade state. */
void cascade_state_cleanup(production_cascade_state_t* state);

/**
 * @brief Wave 2 Item #10 — toggle the cascade self-training hook.
 *
 * When enabled, the orchestrator runs Stage 11
 * (cascade_stage_self_train) after Stage 8 self-comprehension and
 * before Stage 9 self-feedback. Stage 11 calls the SNN language
 * bridge's echo-correct plasticity API on every produced word with
 * a learning rate proportional to (self_match - baseline), driving
 * three-factor learning where the bridge's per-binding eligibility
 * traces are the synaptic memory, dopamine is the bridge's optional
 * neuromodulator gain, and (self_match - baseline) is the reward
 * prediction error.
 *
 * Default OFF. Maps directly to brain->cascade_self_train_enabled.
 * Returns 0 on success, -1 on invalid brain pointer.
 */
int communication_cascade_set_self_train_enabled(brain_t brain, bool enabled);

/** Read the self-train flag. Returns false on NULL brain. */
bool communication_cascade_get_self_train_enabled(brain_t brain);

/**
 * @brief Configure the self-train EMA + LR-scale tunables.
 *
 * @param alpha    EMA mixing rate for the running baseline; clamped to
 *                 [0, 1]. Default 0.05 (slow averaging — ~20-call window).
 *                 0 freezes the baseline at its current value; 1 makes
 *                 every call its own baseline (reward becomes 0).
 * @param lr_scale Multiplier applied to echo_correct lr_scale. Clamped
 *                 to [0, 10]. Default 1.0. Set < 1 to attenuate early
 *                 training, > 1 for fast imprinting (with caveats —
 *                 large values amplify negative-reward LTD via the
 *                 LTD branch of strengthen_binding).
 *
 * Returns 0 on success, -1 on invalid brain pointer.
 */
int communication_cascade_set_self_train_tunables(brain_t brain,
                                                    float alpha,
                                                    float lr_scale);

/**
 * @brief Wave 2 Item #10 — pure helper that applies the reward signal.
 *
 * Exposed for testability — the production cascade orchestrator calls
 * this internally as Stage 11, but unit tests can drive it directly
 * against a bridge + handcrafted utterance + intent. No phantom-API
 * risk: the helper composes only existing bridge APIs
 * (snn_language_bridge_echo_correct) plus a memory-safe tokenizer.
 *
 * @param bridge     SNN language bridge with concept-word bindings.
 *                   May be NULL — returns 0 (no-op) with *out_reward=0.
 * @param intent     Content intent vector that drove the produce.
 * @param intent_dim Length of intent.
 * @param utterance  Whitespace-separated produced text. NULL or empty
 *                   skips with *out_reward=0.
 * @param self_match Phase 2D-B re-comprehension cosine. NaN/Inf skip.
 * @param baseline_inout  In/out EMA baseline. Read once to compute the
 *                        reward, then updated by alpha-mixing self_match.
 * @param alpha      EMA mixing rate, clamped to [0,1].
 * @param lr_scale   Multiplier applied to echo_correct lr_scale.
 * @param out_reward Optional — set to the reward signal that was applied
 *                   (self_match - baseline_before_update). Always defined
 *                   when the call returns; 0 when the call short-circuits.
 *
 * Returns the number of bindings strengthened across all produced words,
 * or 0 when nothing fired (skip path). Never returns negative.
 */
int cascade_apply_self_train_reward(
    struct snn_language_bridge* bridge,
    const float* intent,
    uint32_t intent_dim,
    const char* utterance,
    float self_match,
    float* baseline_inout,
    float alpha,
    float lr_scale,
    float* out_reward);

/* ---------- Full diagnostic snapshot for Python/RPC introspection. ----------
 *
 * The legacy diag impl exposes only 8 of ~50 cascade fields. Trainers and
 * monitoring scripts need the rest: per-stage drives, listener inference,
 * episodic/content state, self_match, repair retries, prosody contours,
 * train reward, failure reasons. This struct captures ALL scalars; the
 * companion impl ALSO malloc-copies the per-syllable / per-phoneme arrays
 * into out-pointers that the caller frees.
 *
 * Field order is contiguous-by-stage for readability — it is NOT an ABI
 * contract. Callers should access by name. New fields APPEND at the end. */
typedef struct {
    /* Stage 0 — Wernicke input comprehension */
    int      wernicke_parsed;
    int      prompt_is_question;
    int      prompt_is_imperative;
    int      prompt_is_garden_path;
    uint32_t prompt_word_count;
    float    prompt_complexity;
    char     prompt_subject[32];
    char     prompt_verb[32];
    char     prompt_object[32];

    /* Stage 1 — Drive */
    float    drive_magnitude;
    float    drive_valence;
    float    drive_arousal;
    uint8_t  dominant_drive;

    /* Stage 2 — Goal + Pragmatics */
    uint8_t  act_type;
    int      pragmatic_is_indirect;
    uint32_t topic_count;
    float    goal_priority;

    /* Stage 3 — Listener (ToM) */
    int      listener_known;
    float    listener_belief_confidence;
    float    listener_emotion_valence;
    float    audience_familiarity;

    /* Stage 4 — Episodic */
    uint32_t episodic_count;

    /* Stage 5 — Content intent */
    uint32_t content_dim;
    float    content_confidence;

    /* Stages 6-7 — Lexical + Syntactic */
    uint32_t word_count;
    float    fluency;
    float    syntactic_validity;

    /* Stage 8 — Self-comprehension */
    int      self_parsed;
    float    self_complexity;
    float    self_match;
    float    self_grammaticality;

    /* Stage 9 — Phonological */
    uint32_t phoneme_count;
    uint32_t syllable_count;
    float    phon_voiced_ratio;

    /* Stage 12 — Speech repair */
    uint32_t repair_attempts;
    float    best_self_match;

    /* Stage 13 — Prosody (scalar summaries; arrays in out-pointers below) */
    uint32_t prosody_syllable_count;
    float    prosody_mean_f0;
    float    prosody_pitch_range;

    /* Stage 14 — Self-train */
    int      train_applied;
    float    train_reward;

    /* Diagnostics */
    uint32_t stages_completed;
    uint32_t stages_failed;
    uint32_t stages_skipped;
    char     failure_reason[128];
} nimcp_cascade_diag_full_t;

/**
 * @brief Run cascade and snapshot ALL state into the diag struct + arrays.
 *
 * @param brain                Brain handle (internal pointer).
 * @param prompt_or_null       Same as communication_cascade_run.
 * @param out_utterance        Caller buffer for the final utterance text;
 *                             may be NULL if caller doesn't need it.
 * @param out_text_max         Capacity of out_utterance.
 * @param out_best_utterance   Caller buffer for the speech-repair best
 *                             candidate (NULL if no repair fired).
 * @param out_best_max         Capacity of out_best_utterance.
 * @param out                  Populated with all scalars on rc=0.
 * @param out_phoneme_sequence If non-NULL, set to a heap-allocated copy
 *                             of state->phoneme_sequence sized
 *                             out->phoneme_count bytes. **Caller frees
 *                             via nimcp_free**. Set to NULL when
 *                             phoneme_count is 0.
 * @param out_prosody_pitch_hz / out_prosody_duration_ms / out_prosody_intensity_db
 *                             Same ownership semantics — heap-allocated
 *                             arrays sized out->prosody_syllable_count
 *                             floats each. **Caller frees via nimcp_free**.
 *                             NULL when prosody_syllable_count is 0.
 *
 * @return 0 on success; -1 on fatal failure. On failure all out-arrays
 *         remain NULL and the struct is zero-initialized.
 */
int nimcp_brain_produce_cascade_diag_full_impl(
    brain_t brain,
    const char* prompt_or_null,
    char* out_utterance,
    uint32_t out_text_max,
    char* out_best_utterance,
    uint32_t out_best_max,
    nimcp_cascade_diag_full_t* out,
    uint8_t** out_phoneme_sequence,
    float**   out_prosody_pitch_hz,
    float**   out_prosody_duration_ms,
    float**   out_prosody_intensity_db);

#ifdef __cplusplus
}
#endif

#endif  /* NIMCP_COMMUNICATION_CASCADE_H */
