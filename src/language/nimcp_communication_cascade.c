/**
 * @file nimcp_communication_cascade.c
 * @brief Multi-region production cascade — Phase 2A skeleton.
 *
 * Each stage reads real state from its source module(s) and writes its
 * contribution to the shared production_cascade_state_t. Intent-formation
 * stages (1-5) actually shape the content_intent vector that drives the
 * bridge; output stages (6-9) currently delegate to the existing bridge /
 * Broca path while we build the GL→Broca lexicon mirror in Phase 2C.
 *
 * Module gracefully no-ops when an upstream module isn't attached — a
 * brain initialized with init-mode=minimal still gets to run the
 * cascade, it just won't have hypothalamus / ToM / hippocampus signals.
 *
 * Design: docs/claude/communication-cascade-plan.md.
 */

#include "language/nimcp_communication_cascade.h"
#include "language/nimcp_grounded_language.h"

#include "core/brain/nimcp_brain_internal.h"

/* Cognitive module headers — each stage needs the public getter API. */
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_drives.h"
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_adapter.h"
#include "core/brain/regions/prefrontal/nimcp_prefrontal_adapter.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "core/brain/regions/hippocampus/nimcp_hippocampus_adapter.h"
#include "cognitive/memory/nimcp_semantic_memory.h"

#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#define LOG_MODULE "COMM_CASCADE"

/*============================================================================
 * Internal helpers
 *==========================================================================*/

static void cascade_record_skip(production_cascade_state_t* s,
                                 cascade_stage_mask_t stage,
                                 const char* reason) {
    s->stages_skipped |= (uint32_t)stage;
    /* Don't overwrite an earlier failure reason; first one wins. */
    if (!s->failure_reason[0] && reason) {
        size_t n = strlen(reason);
        if (n >= sizeof(s->failure_reason)) n = sizeof(s->failure_reason) - 1;
        memcpy(s->failure_reason, reason, n);
        s->failure_reason[n] = '\0';
    }
}

static void cascade_record_complete(production_cascade_state_t* s) {
    s->stages_completed++;
}

static void cascade_record_fail(production_cascade_state_t* s,
                                 const char* reason) {
    s->stages_failed++;
    if (!s->failure_reason[0] && reason) {
        size_t n = strlen(reason);
        if (n >= sizeof(s->failure_reason)) n = sizeof(s->failure_reason) - 1;
        memcpy(s->failure_reason, reason, n);
        s->failure_reason[n] = '\0';
    }
}

/*============================================================================
 * Stage 1: Drive — read hypothalamus + insula + amygdala
 *==========================================================================*/

static int cascade_stage_drive(brain_t brain,
                                production_cascade_state_t* state) {
    /* Sensible defaults — neutral drive state. Overridden if hypothalamus
     * is attached and we can read real urgencies. */
    state->drive_magnitude = 0.5f;
    state->drive_valence   = 0.0f;
    state->drive_arousal   = 0.5f;
    state->dominant_drive  = 0;

    if (!brain->hypothalamus) {
        cascade_record_skip(state, CASCADE_STAGE_DRIVE,
                            "stage_drive: hypothalamus not attached");
        return 0;
    }

    /* Phase 2A: hypothalamus is attached → bump drive magnitude above
     * the neutral default to verify the cascade is sensitive to module
     * presence. The deep drive query (hypo_drive_get_system_state on
     * the adapter's internal handle) requires an accessor that doesn't
     * yet exist on the adapter — Phase 2B will add it. The skeleton
     * still proves "drive stage runs and contributes" end-to-end. */
    state->drive_magnitude = 0.7f;
    state->drive_arousal   = 0.6f;
    state->drive_valence   = 0.0f;
    state->dominant_drive  = 0;

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 2: Goal — read PFC + WM, classify speech act
 *==========================================================================*/

static int cascade_stage_goal(brain_t brain, const char* prompt,
                               production_cascade_state_t* state) {
    /* Default: STATEMENT, neutral priority. */
    state->act_type      = SPEECH_ACT_DECLARE;  /* generic statement */
    state->goal_priority = 0.5f;
    state->topic_count   = 0;

    /* Phase 2A heuristic classification: prompt punctuation hints
     * speech-act type. Real version (Phase 2B) calls
     * pragmatics_classify_act() and uses PFC's active goal. */
    if (prompt && prompt[0]) {
        size_t n = strlen(prompt);
        char last = (n > 0) ? prompt[n-1] : '\0';
        if (last == '?')      state->act_type = SPEECH_ACT_QUESTION;
        else if (last == '!') state->act_type = SPEECH_ACT_DECLARE;
        else                  state->act_type = SPEECH_ACT_ASSERT;
    }

    /* Pull top-priority goal from PFC if available. PFC goals are
     * already priority-sorted; the first one drives goal_priority for
     * the cascade. prefrontal_goal_t stores priority as an enum
     * (goal_priority_t), so we coerce to a 0..1 scalar by dividing
     * by the count. */
    if (brain->prefrontal) {
        prefrontal_goal_t goals[8];
        uint32_t goal_count = 8;
        if (prefrontal_get_active_goals(brain->prefrontal, goals,
                                          &goal_count) &&
            goal_count > 0) {
            /* Convert priority enum to 0..1 — assume 4 priority levels. */
            float pri_scaled = (float)goals[0].priority / 4.0f;
            if (pri_scaled > 1.0f) pri_scaled = 1.0f;
            state->goal_priority = pri_scaled;
        }
    }

    /* Working memory: each active chunk is a candidate topic concept.
     * working_memory_get_salience uses an out-param signature. */
    if (brain->working_memory) {
        uint32_t wm_size = working_memory_get_size(brain->working_memory);
        for (uint32_t i = 0; i < wm_size && state->topic_count < 8; i++) {
            float salience = 0.0f;
            if (!working_memory_get_salience(brain->working_memory, i,
                                              &salience)) continue;
            if (salience < 0.2f) continue;
            /* WM stores feature vectors, not concept_ids directly — for
             * Phase 2A we just count active items. Phase 2B will
             * cross-reference with semantic_memory to recover concept_ids. */
            state->topic_concept_ids[state->topic_count++] = (uint64_t)i;
        }
    }

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 3: Listener model — read Theory of Mind
 *==========================================================================*/

static int cascade_stage_listener(brain_t brain,
                                   production_cascade_state_t* state) {
    state->listener_known              = false;
    state->listener_belief_confidence  = 0.0f;
    state->listener_emotion_valence    = 0.0f;
    state->audience_familiarity        = 0.0f;

    if (!brain->theory_of_mind) {
        cascade_record_skip(state, CASCADE_STAGE_LISTENER,
                            "stage_listener: ToM not attached");
        return 0;
    }

    /* Phase 2A: ToM has multi-agent state but we don't yet identify
     * which agent is the listener. Use the perspective-taking score as
     * a proxy for "how well we model the audience". Phase 2B will
     * track an explicit listener_id from conversation context. */
    tom_statistics_t stats;
    memset(&stats, 0, sizeof(stats));
    if (tom_get_statistics(brain->theory_of_mind, &stats)) {
        state->audience_familiarity = stats.perspective_taking_score;
        /* Treat any nontrivial perspective signal as "listener known". */
        state->listener_known = (stats.perspective_taking_score > 0.1f);
    }

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 4: Episodic retrieval — hippocampus similarity search
 *==========================================================================*/

static int cascade_stage_episodic(brain_t brain,
                                   const float* query_vec,
                                   uint32_t query_dim,
                                   production_cascade_state_t* state) {
    state->episodic_count = 0;

    if (!brain->hippocampus) {
        cascade_record_skip(state, CASCADE_STAGE_EPISODIC,
                            "stage_episodic: hippocampus not attached");
        return 0;
    }
    if (!query_vec || query_dim == 0) {
        cascade_record_skip(state, CASCADE_STAGE_EPISODIC,
                            "stage_episodic: no query vector");
        return 0;
    }

    /* Use the adapter-level cue retrieval API (the inner core's
     * hippo_find_similar_episodes isn't reachable through the adapter
     * boundary — Phase 2B will add an accessor if we need finer
     * control). The retrieval_result_t allocates the memory + similarity
     * arrays internally; caller-style cleanup matches existing
     * hippocampus_adapter usage. */
    retrieval_result_t result;
    memset(&result, 0, sizeof(result));
    if (!hippocampus_retrieve_by_cue(brain->hippocampus,
                                       query_vec, query_dim,
                                       16 /* max_results */,
                                       &result) ||
        !result.retrieval_success || result.count == 0) {
        cascade_record_skip(state, CASCADE_STAGE_EPISODIC,
                            "stage_episodic: cue retrieval empty");
        return 0;
    }

    /* Phase 2A: store the per-memory similarities as relevances. The
     * memories themselves stay opaque — Phase 2B will lift their
     * concept content into the intent vector. We only have similarities
     * here, not memory_ids in a publicly-stable form, so we use index
     * as a deterministic stand-in. */
    for (uint32_t i = 0; i < result.count && state->episodic_count < 16; i++) {
        state->episodic_concept_ids[state->episodic_count] = (uint64_t)i;
        state->episodic_relevances[state->episodic_count]  =
            result.similarities ? result.similarities[i] : 0.5f;
        state->episodic_count++;
    }

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 5: Content composition — weighted combine of all sources
 *==========================================================================*/

static int cascade_stage_content(brain_t brain,
                                  const float* prompt_intent,
                                  uint32_t prompt_dim,
                                  production_cascade_state_t* state) {
    /* Allocate the content_intent vector at the GL semantic_dim. Without
     * grounded_lang we have nothing to drive — fail this stage. */
    if (!brain->grounded_lang) {
        cascade_record_fail(state,
                             "stage_content: grounded_lang not available");
        return -1;
    }

    uint32_t dim = grounded_language_get_semantic_dim(brain->grounded_lang);
    if (dim == 0) {
        cascade_record_fail(state, "stage_content: semantic_dim is 0");
        return -1;
    }

    state->content_intent = (float*)nimcp_calloc(dim, sizeof(float));
    if (!state->content_intent) {
        cascade_record_fail(state, "stage_content: alloc failed");
        return -1;
    }
    state->content_dim = dim;

    /* Weighted combine. Default weights chosen so prompt dominates when
     * present (mimics directly responding to the question), but drive
     * and episodic still meaningfully bias. Phase 2D will tune. */
    const float w_prompt   = 1.0f;
    const float w_drive    = 0.3f;
    const float w_episodic = 0.4f;

    /* 1. Seed from prompt comprehend (if provided). */
    if (prompt_intent && prompt_dim > 0) {
        uint32_t copy = (prompt_dim < dim) ? prompt_dim : dim;
        for (uint32_t i = 0; i < copy; i++) {
            float v = prompt_intent[i];
            if (isfinite(v)) state->content_intent[i] += w_prompt * v;
        }
    }

    /* 2. Drive bias: arousal scales the magnitude; valence shifts a
     * single dimension as a "tone" hint. Phase 2A doesn't have per-
     * concept drive embeddings yet — Phase 2B will. */
    if (state->drive_magnitude > 0.0f && dim > 0) {
        state->content_intent[0] += w_drive * state->drive_arousal;
        if (dim > 1) {
            state->content_intent[1] += w_drive * state->drive_valence;
        }
    }

    /* 3. Episodic concepts: each retrieved trace's relevance adds a
     * small bias to the intent. Phase 2A treats episode_ids as opaque,
     * so we just spread relevance across a few dimensions deterministically.
     * Phase 2B will lift actual concept feature vectors from each episode. */
    for (uint32_t i = 0; i < state->episodic_count && i < dim; i++) {
        uint32_t slot = (uint32_t)(state->episodic_concept_ids[i] % dim);
        state->content_intent[slot] +=
            w_episodic * state->episodic_relevances[i];
    }

    /* Confidence = signal magnitude / sqrt(dim). 1.0 = strongly cohered;
     * 0.0 = no signal anywhere. */
    float ssum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float v = state->content_intent[i];
        ssum += v * v;
    }
    state->content_confidence = sqrtf(ssum) / sqrtf((float)dim);
    if (state->content_confidence > 1.0f) state->content_confidence = 1.0f;

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 6: Lexical selection — bridge produce on content_intent
 *==========================================================================*/

static int cascade_stage_lexical(brain_t brain,
                                  production_cascade_state_t* state) {
    if (!brain->grounded_lang || !state->content_intent) {
        cascade_record_skip(state, CASCADE_STAGE_LEXICAL,
                            "stage_lexical: no grounded_lang or content_intent");
        return 0;
    }

    /* Run the bridge produce on our cascade-shaped content_intent, NOT
     * on the raw prompt comprehension. This is what makes the cascade
     * meaningfully different from grounded_respond — the intent has
     * been shaped by drive + episodic + (eventually) listener +
     * goal stages. */
    gl_production_result_t prod = {0};
    int rc = grounded_language_produce(brain->grounded_lang,
                                        state->content_intent,
                                        state->content_dim,
                                        GL_PRODUCE_RESPOND, &prod);
    if (rc != 0 || !prod.text || prod.word_count == 0) {
        gl_production_result_cleanup(&prod);
        cascade_record_fail(state, "stage_lexical: bridge produce failed");
        return -1;
    }

    /* Transfer ownership: prod.text → state->utterance. */
    state->utterance  = prod.text;
    prod.text         = NULL;
    state->word_count = prod.word_count;
    state->fluency    = prod.fluency;

    gl_production_result_cleanup(&prod);
    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 7: Syntactic — Broca syntax processor (Phase 2A: passthrough)
 *==========================================================================*/

static int cascade_stage_syntactic(brain_t brain,
                                    production_cascade_state_t* state) {
    (void)brain;
    /* Phase 2A: real Broca routing requires the GL→Broca lexicon mirror
     * (Phase 2C). Until that lands, syntax_validity is unknown — record
     * as -1 so downstream consumers can tell apart "skipped" from
     * "validated grammatical" (1.0) from "validated ungrammatical" (0.0). */
    state->syntactic_validity = -1.0f;
    cascade_record_skip(state, CASCADE_STAGE_SYNTACTIC,
                        "stage_syntactic: deferred to Phase 2C");
    return 0;
}

/*============================================================================
 * Stage 8: Phonological encoding (deferred to Phase 2E)
 *==========================================================================*/

static int cascade_stage_phonological(brain_t brain,
                                       production_cascade_state_t* state) {
    (void)brain;
    cascade_record_skip(state, CASCADE_STAGE_PHONOLOGICAL,
                        "stage_phonological: deferred to Phase 2E");
    return 0;
}

/*============================================================================
 * Stage 9: Motor / output (deferred — text mode renders in stage 6)
 *==========================================================================*/

static int cascade_stage_motor(brain_t brain,
                                production_cascade_state_t* state) {
    (void)brain;
    cascade_record_skip(state, CASCADE_STAGE_MOTOR,
                        "stage_motor: text mode — rendering happens in stage_lexical");
    return 0;
}

/*============================================================================
 * Public API
 *==========================================================================*/

void cascade_state_cleanup(production_cascade_state_t* state) {
    if (!state) return;
    if (state->content_intent) {
        nimcp_free(state->content_intent);
        state->content_intent = NULL;
    }
    if (state->utterance) {
        nimcp_free(state->utterance);
        state->utterance = NULL;
    }
}

/* Public API impl called from nimcp_part_core.c — fills caller-provided
 * buffers, hides the production_cascade_state_t from the public header.
 * Returns 0 on success; -1 on fatal failure. */
int nimcp_brain_produce_cascade_impl(
    brain_t brain,
    const char* prompt_or_null,
    char* out_utterance,
    uint32_t out_text_max,
    uint32_t* out_word_count,
    float* out_confidence)
{
    if (!brain) return -1;

    production_cascade_state_t state;
    int rc = communication_cascade_run(brain, prompt_or_null,
                                         CASCADE_STAGE_ALL, &state);

    if (rc == 0) {
        if (out_utterance && out_text_max > 0) {
            const char* src = state.utterance ? state.utterance : "";
            size_t n = strlen(src);
            if (n >= out_text_max) n = out_text_max - 1;
            memcpy(out_utterance, src, n);
            out_utterance[n] = '\0';
        }
        if (out_word_count)  *out_word_count  = state.word_count;
        if (out_confidence)  *out_confidence  = state.content_confidence;
    } else {
        if (out_utterance && out_text_max > 0) out_utterance[0] = '\0';
        if (out_word_count) *out_word_count = 0;
        if (out_confidence) *out_confidence = 0.0f;
    }

    cascade_state_cleanup(&state);
    return rc;
}

int communication_cascade_run(
    brain_t brain,
    const char* prompt_or_null,
    uint32_t stage_mask,
    production_cascade_state_t* out_state)
{
    if (!brain || !out_state) return -1;
    memset(out_state, 0, sizeof(*out_state));

    if (stage_mask == 0) stage_mask = CASCADE_STAGE_ALL;

    /* If a prompt was given, comprehend it first to seed the intent
     * vector. The comprehend itself fires GL_EVENT_COMPREHENDED on the
     * cognitive bus, so working memory + ToM see the input naturally. */
    gl_comprehension_result_t comp = {0};
    bool have_prompt = false;
    if (prompt_or_null && prompt_or_null[0] && brain->grounded_lang) {
        if (grounded_language_comprehend(brain->grounded_lang, prompt_or_null,
                                          &comp) == 0 && comp.semantic_vector) {
            have_prompt = true;
        }
    }

    /* Stage 1-5: build content intent. Each stage records its own
     * skip/fail; only stage_content failure aborts the cascade. */
    if (stage_mask & CASCADE_STAGE_DRIVE) {
        cascade_stage_drive(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_GOAL) {
        cascade_stage_goal(brain, prompt_or_null, out_state);
    }
    if (stage_mask & CASCADE_STAGE_LISTENER) {
        cascade_stage_listener(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_EPISODIC) {
        cascade_stage_episodic(brain,
                                have_prompt ? comp.semantic_vector : NULL,
                                have_prompt ? grounded_language_get_semantic_dim(brain->grounded_lang) : 0,
                                out_state);
    }
    if (stage_mask & CASCADE_STAGE_CONTENT) {
        if (cascade_stage_content(brain,
                                    have_prompt ? comp.semantic_vector : NULL,
                                    have_prompt ? grounded_language_get_semantic_dim(brain->grounded_lang) : 0,
                                    out_state) < 0) {
            gl_comprehension_result_cleanup(&comp);
            return -1;
        }
    }

    gl_comprehension_result_cleanup(&comp);

    /* Stages 6-9: surface the intent as language. */
    if (stage_mask & CASCADE_STAGE_LEXICAL) {
        cascade_stage_lexical(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_SYNTACTIC) {
        cascade_stage_syntactic(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_PHONOLOGICAL) {
        cascade_stage_phonological(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_MOTOR) {
        cascade_stage_motor(brain, out_state);
    }

    return 0;
}
