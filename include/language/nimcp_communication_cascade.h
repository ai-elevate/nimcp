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
    CASCADE_STAGE_PHONOLOGICAL  = 1u << 8,
    CASCADE_STAGE_MOTOR         = 1u << 9,
    CASCADE_STAGE_ALL           = 0x3FF
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

    /* Diagnostics */
    uint32_t stages_completed;
    uint32_t stages_failed;
    uint32_t stages_skipped;       /* per-stage skip mask */
    char     failure_reason[128];
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

#ifdef __cplusplus
}
#endif

#endif  /* NIMCP_COMMUNICATION_CASCADE_H */
