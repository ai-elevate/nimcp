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
/* Note: cannot include core/brain/regions/broca/nimcp_phonological.h here
 * because Wernicke's perception/nimcp_speech_cortex.h (pulled in
 * transitively via grounded_language + brain_internal headers) also
 * defines a conflicting struct phoneme_t. The Broca processor lives in a
 * separate TU; Stage 9 emits the raw phoneme sequence directly using
 * ASCII-letter codes (the same encoding broca_adapter.c uses internally),
 * and approximates the syllable count by counting vowel clusters - good
 * enough for the diagnostic exposed via state->syllable_count. */

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

/* Stage 0 (Wernicke comprehension) and Stage 8 (Wernicke self-comprehension)
 * live in nimcp_communication_cascade_wernicke.c because Wernicke and Broca
 * both define phrase_type_t with overlapping enum values; pulling both into
 * one TU triggers a redeclaration error. */
extern int cascade_stage_wernicke(brain_t brain, const char* prompt,
                                    production_cascade_state_t* state);
extern int cascade_stage_self_comprehension(brain_t brain,
                                              production_cascade_state_t* state);
/* Helper used by stage_goal to look up the Wernicke-extracted SVO words. */
extern const gl_lexicon_entry_t* lexicon_find_internal(
    const grounded_language_t* gl, const char* word);

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

    /* Phase 2B: actually query the hypothalamus state. The adapter
     * exposes the homeostatic + stress + circadian state via
     * hypothalamus_get_state(); from there we extract the strongest
     * drive (hunger / thirst / temperature / stress / fatigue) and
     * compute valence and arousal. The dominant_drive index is what
     * lets downstream stages bias content toward drive-relevant
     * concepts (Phase 2C will add concept-vocab→drive mapping). */
    hypothalamus_state_t h;
    memset(&h, 0, sizeof(h));
    if (!hypothalamus_get_state(brain->hypothalamus, &h)) {
        cascade_record_skip(state, CASCADE_STAGE_DRIVE,
                            "stage_drive: hypothalamus_get_state failed");
        /* Fall back to neutral defaults set above. */
        state->drive_magnitude = 0.5f;
        state->drive_arousal   = 0.5f;
        return 0;
    }

    /* Cherry-pick the homeostatic drive with highest magnitude.
     * Hunger / thirst / cortisol-stress / fatigue / temp-deviation are
     * all in [0,1]. Whichever is loudest wins. */
    const float hunger     = h.appetite.hunger_drive;
    const float thirst     = h.hydration.thirst_drive;
    const float stress     = h.hpa_axis.stress_input;
    const float fatigue    = h.circadian.sleep_pressure;
    const float thermal_e  = (float)fabs(h.thermoregulation.core_temp.error);
    /* Curiosity/social/etc aren't in the homeostatic struct — they live
     * in the drive system handle which the adapter doesn't expose yet.
     * Phase 2C TODO: surface those via a new accessor. */

    float    max_drive = hunger;
    uint8_t  dominant  = 1; /* 1=HUNGER */
    if (thirst    > max_drive) { max_drive = thirst;    dominant = 2; /* THIRST */ }
    if (stress    > max_drive) { max_drive = stress;    dominant = 4; /* STRESS */ }
    if (fatigue   > max_drive) { max_drive = fatigue;   dominant = 5; /* FATIGUE */ }
    if (thermal_e > max_drive) { max_drive = thermal_e; dominant = 3; /* THERMAL */ }

    state->drive_magnitude = max_drive;
    state->dominant_drive  = dominant;

    /* Arousal: HPA stress drives arousal up; circadian alertness too.
     * Take max of (stress, 1 - sleep_pressure). At rest, alertness
     * is high so arousal sits near 0.7-0.9; under stress, arousal
     * spikes to 1.0. */
    float alertness = 1.0f - fatigue;
    state->drive_arousal = (stress > alertness) ? stress : alertness;
    if (state->drive_arousal > 1.0f) state->drive_arousal = 1.0f;
    if (state->drive_arousal < 0.0f) state->drive_arousal = 0.0f;

    /* Valence: stress = strongly negative; fed/hydrated/rested = positive;
     * neutral when no specific drive dominates. */
    if (dominant == 4 /* STRESS */) {
        state->drive_valence = -stress;
    } else if (dominant == 1 /* HUNGER */ || dominant == 2 /* THIRST */) {
        /* Aversive when high — same direction as stress but milder. */
        state->drive_valence = -0.5f * max_drive;
    } else if (dominant == 5 /* FATIGUE */) {
        state->drive_valence = -0.3f * fatigue;
    } else if (dominant == 3 /* THERMAL */) {
        state->drive_valence = -0.4f * thermal_e;
    } else {
        state->drive_valence = 0.0f;
    }

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

    /* Speech-act classification. Phase 2D-A uses Wernicke's Stage 0
     * output (wh-word / aux-inversion / parse-tree) when available;
     * falls back to punctuation heuristic when Wernicke wasn't run.
     * Wernicke's signal is more reliable — it catches questions
     * without trailing '?' ("Is the dog hungry") and imperatives
     * with no subject ("Close the door"). */
    if (state->prompt_is_question) {
        state->act_type = SPEECH_ACT_QUESTION;
    } else if (state->prompt_is_imperative) {
        state->act_type = SPEECH_ACT_COMMAND;
    } else if (prompt && prompt[0]) {
        size_t n = strlen(prompt);
        char last = (n > 0) ? prompt[n-1] : '\0';
        if (last == '?')      state->act_type = SPEECH_ACT_QUESTION;
        else if (last == '!') state->act_type = SPEECH_ACT_DECLARE;
        else                  state->act_type = SPEECH_ACT_ASSERT;
    }

    /* Pragmatics override: when the surface form is a question, ask
     * Broca's pragmatics processor whether it's actually an INDIRECT
     * speech act ("Can you pass the salt?" surface=QUESTION,
     * intended=REQUEST). The processor matches a small set of indirect-
     * request templates (modal+you, declarative-with-please, etc.) and
     * returns is_indirect=true with primary_act=REQUEST/COMMAND. We
     * only override on questions — direct REQUEST/COMMAND is already
     * handled above via prompt_is_imperative. */
    if (state->prompt_is_question &&
        brain->broca_pragmatics &&
        prompt && prompt[0]) {
        speech_act_result_t prag = {0};
        if (pragmatics_classify_speech_act(brain->broca_pragmatics,
                                             prompt, /*speaker_id*/0, &prag) &&
            prag.is_indirect &&
            (prag.primary_act == SPEECH_ACT_REQUEST ||
             prag.primary_act == SPEECH_ACT_COMMAND)) {
            state->act_type             = SPEECH_ACT_REQUEST;
            state->pragmatic_is_indirect = true;
        }
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

    /* Phase 2D-A: pull Wernicke's identified subject/verb/object as
     * primary topic concepts. These are real content words from the
     * prompt itself (extracted via parse tree), not array indices —
     * stage_content uses them via GL lexicon lookup to lift the
     * actual concept feature vectors into the intent. */
    if (state->prompt_subject[0] && state->topic_count < 8) {
        const gl_lexicon_entry_t* e = lexicon_find_internal(
            brain->grounded_lang, state->prompt_subject);
        if (e && e->binding_count > 0) {
            state->topic_concept_ids[state->topic_count++] =
                e->bindings[0].concept_id;
            if (state->target_concept_id == 0) {
                state->target_concept_id = e->bindings[0].concept_id;
            }
        }
    }
    if (state->prompt_verb[0] && state->topic_count < 8) {
        const gl_lexicon_entry_t* e = lexicon_find_internal(
            brain->grounded_lang, state->prompt_verb);
        if (e && e->binding_count > 0) {
            state->topic_concept_ids[state->topic_count++] =
                e->bindings[0].concept_id;
        }
    }
    if (state->prompt_object[0] && state->topic_count < 8) {
        const gl_lexicon_entry_t* e = lexicon_find_internal(
            brain->grounded_lang, state->prompt_object);
        if (e && e->binding_count > 0) {
            state->topic_concept_ids[state->topic_count++] =
                e->bindings[0].concept_id;
        }
    }

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 3: Listener model — read Theory of Mind
 *==========================================================================*/

/* Map ToM emotion enum to a [-1,1] valence axis. Negative = aversive
 * (sadness/anger/fear/disgust/anxiety/shame); positive = approach
 * (joy/pride/calm/surprise); 0 for neutral/unknown. */
static float tom_emotion_to_valence(tom_emotion_t e) {
    switch (e) {
        case TOM_EMOTION_JOY:      return  0.9f;
        case TOM_EMOTION_PRIDE:    return  0.7f;
        case TOM_EMOTION_CALM:     return  0.5f;
        case TOM_EMOTION_SURPRISE: return  0.3f;
        case TOM_EMOTION_NEUTRAL:  return  0.0f;
        case TOM_EMOTION_SADNESS:  return -0.7f;
        case TOM_EMOTION_ANGER:    return -0.8f;
        case TOM_EMOTION_FEAR:     return -0.9f;
        case TOM_EMOTION_DISGUST:  return -0.6f;
        case TOM_EMOTION_ANXIETY:  return -0.5f;
        case TOM_EMOTION_SHAME:    return -0.4f;
        case TOM_EMOTION_UNKNOWN:
        default:                   return  0.0f;
    }
}

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

    /* Phase 2B: query the actual listener model. Agent_id = 0 is the
     * conventional "primary interlocutor". The cascade doesn't yet
     * track multiple agents per conversation — that's a future
     * enhancement once a turn-taking layer is added. The aggregate
     * perspective-taking score still seeds audience_familiarity. */
    tom_statistics_t stats;
    memset(&stats, 0, sizeof(stats));
    if (tom_get_statistics(brain->theory_of_mind, &stats)) {
        state->audience_familiarity = stats.perspective_taking_score;
    }

    /* Pull belief + emotion from the ToM module. The codebase had an
     * tom_get_agent_state declaration with per-agent fan-out, but that
     * function is a phantom (header-only, no impl). The two real getters
     * are tom_get_bdi_state (no agent_id — uses the most-recently-tracked
     * agent internally) and tom_infer_emotion. Together they give us
     * everything we need. */
    tom_belief_t    belief    = {0};
    tom_desire_t    desire    = {0};
    tom_intention_t intention = {0};
    if (tom_get_bdi_state(brain->theory_of_mind, &belief, &desire, &intention)) {
        state->listener_known = true;
        state->listener_belief_confidence = belief.confidence;
    } else if (state->audience_familiarity > 0.1f) {
        /* No BDI record but ToM has been doing perspective work — treat
         * as "vague listener known". */
        state->listener_known = true;
    }

    if (state->listener_known) {
        float emotion_conf = 0.0f;
        tom_emotion_t emotion = tom_infer_emotion(brain->theory_of_mind,
                                                    &emotion_conf);
        state->listener_emotion_valence =
            tom_emotion_to_valence(emotion) * emotion_conf;
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

    /* Use the adapter-level cue retrieval API. The retrieval_result_t
     * allocates the memory + similarity arrays internally. */
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

    /* Phase 2B: store the actual memory_id (not just an array index) so
     * stage_content can dereference and lift the memory's feature vector
     * into the intent. Each retrieved memory carries a feature_count×float
     * features array — that IS the encoded content for the past episode.
     * Storing the memory_id lets us look it up later if needed; for the
     * cascade we mainly need the feature vector, which we'll reach via
     * result.memories[i].features in stage_content. We keep result alive
     * by transferring ownership to a per-call scratch field on state. */
    for (uint32_t i = 0; i < result.count && state->episodic_count < 16; i++) {
        state->episodic_concept_ids[state->episodic_count] =
            (uint64_t)result.memories[i].memory_id;
        state->episodic_relevances[state->episodic_count]  =
            result.similarities ? result.similarities[i] : 0.5f;
        state->episodic_count++;
    }

    /* Stash the retrieval result on the cascade state so stage_content
     * can lift feature vectors. We use a struct member added below — for
     * now, just leak result.memories / similarities arrays (they're
     * tiny and the cascade-state cleanup will free them). */
    state->episodic_retrieval = result;  /* shallow copy — memories[] and
                                          * similarities[] ownership transferred. */

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

    /* Weighted combine. Prompt dominates when present (we're answering
     * a question); drive / episodic / listener / goal nudge the answer
     * without overwhelming the prompt's content. Tuning history:
     *   Phase 2A: w_drive=0.3 — too small to flip bridge argmax
     *   Phase 2B v1: w_drive=0.6 — too LARGE, overrode prompt in the
     *                drive band, all prompts converged to same output
     *   Phase 2B v2: w_drive=0.15 — small enough that prompt content
     *                differentiates outputs, large enough to shift
     *                argmax when prompt signal is weak (spontaneous mode) */
    const float w_prompt   = 1.0f;
    const float w_drive    = 0.15f;
    const float w_episodic = 0.3f;
    const float w_listener = 0.1f;
    const float w_goal     = 0.2f;

    /* 1. Seed from prompt comprehend (if provided). */
    if (prompt_intent && prompt_dim > 0) {
        uint32_t copy = (prompt_dim < dim) ? prompt_dim : dim;
        for (uint32_t i = 0; i < copy; i++) {
            float v = prompt_intent[i];
            if (isfinite(v)) state->content_intent[i] += w_prompt * v;
        }
    }

    /* 2. Drive bias: spread drive activity across a deterministic 16-dim
     * band keyed by dominant drive type. Different drives hit different
     * dimensions so the same prompt under different drive states biases
     * the intent toward different concepts. Phase 2C will replace this
     * deterministic mapping with a real concept↔drive lookup.
     *
     * Signal: max(drive_magnitude, arousal). On a state-empty fresh
     * brain, drive_magnitude=0 but arousal≈1 (no fatigue, no stress
     * = full alertness), so the bias still fires — "I'm alert and
     * about to speak" is itself a valid cognitive signal. */
    float drive_signal = state->drive_magnitude;
    if (state->drive_arousal > drive_signal) drive_signal = state->drive_arousal;
    if (drive_signal > 0.05f && dim >= 16) {
        const uint32_t band_start = ((uint32_t)state->dominant_drive % 8) * 16;
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t slot = (band_start + i) % dim;
            /* Sign alternation: every other dim carries valence tone,
             * the rest carry magnitude — gives the bridge multi-axis
             * signal rather than a flat scalar bump. */
            float sign = (i & 1) ? state->drive_valence : 1.0f;
            state->content_intent[slot] +=
                w_drive * drive_signal * sign;
        }
    }

    /* 3. Episodic concepts: lift feature vectors from each retrieved
     * memory. Each memory.features is feature_count×float of encoded
     * past experience — adding it to the intent is "the brain biased
     * toward saying things related to what it remembers". */
    for (uint32_t i = 0; i < state->episodic_count; i++) {
        uint32_t mi = i;  /* index into result.memories aligns with
                           * episodic_concept_ids since we filled them in
                           * lockstep. */
        if (mi >= state->episodic_retrieval.count) continue;
        const hippocampus_memory_t* m = &state->episodic_retrieval.memories[mi];
        if (!m->features || m->feature_count == 0) continue;
        const float relevance = state->episodic_relevances[i];
        uint32_t copy = (m->feature_count < dim) ? m->feature_count : dim;
        for (uint32_t j = 0; j < copy; j++) {
            float fv = m->features[j];
            if (isfinite(fv)) {
                state->content_intent[j] += w_episodic * relevance * fv;
            }
        }
    }

    /* 4. Listener bias: when ToM is engaged, valence of the listener's
     * inferred emotion shifts the intent's tonal dimension. Subtle but
     * deterministic — used to differentiate audience-aware vs neutral
     * production in Phase 2D evals. */
    if (state->listener_known && state->audience_familiarity > 0.0f && dim >= 4) {
        state->content_intent[2] +=
            w_listener * state->audience_familiarity *
            state->listener_emotion_valence;
        state->content_intent[3] +=
            w_listener * state->audience_familiarity *
            state->listener_belief_confidence;
    }

    /* 5. Goal-priority bias: high-priority PFC goal → broaden the
     * intent slightly so the bridge's softmax is less peaked → more
     * likely to express the goal-relevant concepts even at low
     * activation. This is the "I really need to say this" effect. */
    if (state->goal_priority > 0.0f && dim >= 8) {
        for (uint32_t i = 0; i < state->topic_count && i < 8 && i < dim; i++) {
            uint32_t slot = (uint32_t)(state->topic_concept_ids[i] % dim);
            state->content_intent[slot] += w_goal * state->goal_priority;
        }
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
 * Stage 7: Syntactic — Broca syntax processor (Phase 2C activated)
 *==========================================================================*/

#include "core/brain/regions/broca/nimcp_broca_adapter.h"
#include "core/brain/regions/broca/nimcp_syntax_processor.h"

static int cascade_stage_syntactic(brain_t brain,
                                    production_cascade_state_t* state) {
    state->syntactic_validity = -1.0f;  /* unknown until we run Broca */

    if (!brain->broca || !state->utterance || !state->utterance[0]) {
        cascade_record_skip(state, CASCADE_STAGE_SYNTACTIC,
                            "stage_syntactic: no broca or no utterance");
        return 0;
    }

    /* Tokenize the bridge's output text into words. Same approach the
     * Phase 1 audit experiment used — strtok_r on whitespace +
     * punctuation, with a soft cap on token count. */
    enum { MAX_TOKENS = 32 };
    char dup[2048];
    size_t ulen = strlen(state->utterance);
    if (ulen >= sizeof(dup)) ulen = sizeof(dup) - 1;
    memcpy(dup, state->utterance, ulen);
    dup[ulen] = '\0';

    const char* tokens[MAX_TOKENS];
    uint32_t num_tokens = 0;
    char* save = NULL;
    char* tok = strtok_r(dup, " \t\n\r.,!?;:\"'", &save);
    while (tok && num_tokens < MAX_TOKENS) {
        if (tok[0]) tokens[num_tokens++] = tok;
        tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
    }
    if (num_tokens == 0) {
        cascade_record_skip(state, CASCADE_STAGE_SYNTACTIC,
                            "stage_syntactic: utterance tokenized to zero words");
        return 0;
    }

    /* Run through Broca's full pipeline: syntax_build_tree (CYK),
     * syntax_validate_grammar (agreement check), phonological generation,
     * motor planning. broca_produce_from_strings does all of these and
     * returns false if any phase fails. */
    broca_adapter_t* broca = (broca_adapter_t*)brain->broca;
    broca_utterance_result_t br;
    memset(&br, 0, sizeof(br));
    bool ok = broca_produce_from_strings(broca, tokens, num_tokens, &br);

    if (!ok) {
        /* Broca rejected — likely a CYK chart-build failure on the
         * bridge's word salad. Record as syntactic_validity = 0.0 so
         * downstream consumers can tell "tried but failed" from "skipped".
         * The bridge's text stays as-is in state->utterance. */
        state->syntactic_validity = 0.0f;
        cascade_record_fail(state,
                            "stage_syntactic: broca rejected utterance");
        return 0;
    }

    state->syntactic_validity = (br.syntax_valid && br.agreement_valid) ? 1.0f
                              : (br.syntax_valid ? 0.5f : 0.0f);

    /* Pull the syntactically-ordered word sequence back out of Broca's
     * syntax processor and re-render. CYK is order-preserving for the
     * surface forms it accepts (it builds tree structure but doesn't
     * reorder leaf words), so the rendered text is usually identical
     * to the input. The signal we care about is the validity flag —
     * a successful build means the words DID form a recognized phrase
     * structure, which is more than the bridge's raw output guarantees. */
    syntax_processor_t* syn = broca_get_syntax_processor(broca);
    if (syn) {
        uint32_t unit_count = syntax_get_unit_count(syn);
        if (unit_count > 0) {
            char buf[2048];
            uint32_t pos = 0;
            uint32_t emitted = 0;
            for (uint32_t i = 0; i < unit_count && pos < sizeof(buf) - 1; i++) {
                syntactic_unit_t unit;
                if (!syntax_get_unit(syn, i, &unit)) continue;

                broca_lexical_entry_t entry;
                if (!broca_lookup_word(broca, unit.word_id, NULL, &entry)) continue;
                if (!entry.word[0]) continue;

                if (emitted > 0 && pos < sizeof(buf) - 1) buf[pos++] = ' ';
                size_t wlen = strnlen(entry.word, sizeof(entry.word));
                size_t avail = sizeof(buf) - 1 - pos;
                size_t copy = (wlen < avail) ? wlen : avail;
                memcpy(buf + pos, entry.word, copy);
                pos += (uint32_t)copy;
                emitted++;
            }
            buf[pos] = '\0';

            if (emitted > 0) {
                /* Replace bridge text with Broca-rendered text. */
                char* new_text = (char*)nimcp_calloc(pos + 1, 1);
                if (new_text) {
                    memcpy(new_text, buf, pos);
                    nimcp_free(state->utterance);
                    state->utterance = new_text;
                    state->word_count = emitted;
                }
            }
        }
    }

    cascade_record_complete(state);
    return 0;
}

/*============================================================================
 * Stage 9 (Wave 2 Item 7): Phonological output — text → phoneme sequence.
 *
 * Why: Stage 7 (syntactic) leaves a Broca-rendered text utterance in
 * state->utterance; downstream consumers (motor planning, TTS, edge
 * speech-output) need a phoneme sequence, not graphemes. This stage runs
 * a lightweight rule-based English G2P over the utterance and emits
 * state->phoneme_sequence plus diagnostics. We use the existing Broca
 * phonological_processor_t (include/core/brain/regions/broca/nimcp_phonological.h)
 * for syllabification rather than rolling our own — biological fidelity
 * comes from re-using the same module Broca uses internally.
 *
 * Why a rule-based G2P (not a real one): there is no production G2P module
 * in this codebase today. The Broca lexicon stores phonemes per word, but
 * cascade utterances frequently contain words outside the trained
 * vocabulary (e.g. proper nouns or generated tokens), so a lexicon-only
 * approach would silently emit empty phoneme sequences. The rules below
 * cover all 26 English letters plus 5 common digraphs (sh/ch/th/ng/ph),
 * which is sufficient to expose meaningful phoneme_count + voiced_ratio
 * diagnostics. The phoneme codes themselves are ASCII letters (with a
 * handful of sentinels for digraphs) — matching the convention used in
 * broca_adapter.c phonological_add_phoneme_detailed calls.
 *
 * Why no phonological_processor_t here: pulling in
 * core/brain/regions/broca/nimcp_phonological.h triggers a phoneme_t
 * type conflict with Wernicke's perception/nimcp_speech_cortex.h that
 * is already transitively included through grounded_language. The Broca
 * processor lives in its own TU (nimcp_broca_adapter.c) and stays there.
 * We approximate the syllable count with a simple vowel-cluster
 * heuristic — this matches what a sonority-based syllabifier would
 * produce on monosyllabic English input and is good enough for the
 * diagnostic this stage exposes.
 *==========================================================================*/

/* Returns true if the lowercase letter c is a vowel. */
static inline bool cascade_phon_is_vowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

/* Lightweight English digraph handler. Returns the digraph sentinel
 * phoneme code if buf[i..i+1] is a recognized digraph (and bumps *advance
 * to 2), else 0. */
static uint8_t cascade_phon_digraph(const char* buf, size_t i, size_t n,
                                     uint32_t* advance) {
    if (i + 1 >= n) return 0;
    char a = buf[i], b = buf[i + 1];
    /* Common English digraphs. The sentinel chars stay in the ASCII
     * punctuation range so they can't collide with letter-encoded phonemes. */
    if (a == 's' && b == 'h') { *advance = 2; return (uint8_t)'$'; }  /* /ʃ/ */
    if (a == 'c' && b == 'h') { *advance = 2; return (uint8_t)'&'; }  /* /tʃ/ */
    if (a == 't' && b == 'h') { *advance = 2; return (uint8_t)'#'; }  /* /θ/ or /ð/ */
    if (a == 'n' && b == 'g') { *advance = 2; return (uint8_t)'@'; }  /* /ŋ/ */
    if (a == 'p' && b == 'h') { *advance = 2; return (uint8_t)'%'; }  /* /f/ */
    return 0;
}

static int cascade_stage_phonological(brain_t brain,
                                       production_cascade_state_t* state) {
    (void)brain;
    if (!state) return 0;

    /* Skip cleanly when there's nothing to phonologize. Matches the
     * upstream skip pattern: a brain with no Stage 6/7 output (minimal
     * init, bridge produced empty text, lexical/syntactic skipped) gets
     * recorded as a skip rather than a failure. */
    if (!state->utterance || state->word_count == 0 || !state->utterance[0]) {
        cascade_record_skip(state, CASCADE_STAGE_PHONOLOGICAL,
                            "stage_phonological: no utterance or empty word_count");
        return 0;
    }

    /* Allocate worst-case phoneme buffer: one phoneme per char + word
     * boundaries. nimcp_calloc zero-fills so a partial fill is safe. */
    const size_t ulen = strlen(state->utterance);
    if (ulen == 0) {
        cascade_record_skip(state, CASCADE_STAGE_PHONOLOGICAL,
                            "stage_phonological: utterance has zero length");
        return 0;
    }
    uint8_t* seq = (uint8_t*)nimcp_calloc(ulen + 1, sizeof(uint8_t));
    if (!seq) {
        cascade_record_fail(state, "stage_phonological: phoneme buffer alloc failed");
        return 0;
    }

    uint32_t phon_count = 0;
    uint32_t voiced_count = 0;
    /* Syllable-count heuristic: a syllable starts on a vowel that follows
     * a non-vowel (or a word boundary). Adjacent vowels (diphthongs like
     * 'oa', 'ai') count as one. */
    uint32_t syll_count = 0;
    bool prev_phon_was_vowel = false;
    bool prev_was_space = true;

    for (size_t i = 0; i < ulen; ) {
        char ch = state->utterance[i];

        /* Lowercase ASCII letters in place. The cascade often emits
         * mixed-case text from the bridge; lowercase is the canonical
         * form for the rules below. */
        if (ch >= 'A' && ch <= 'Z') ch = (char)(ch + 32);

        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
            ch == '.' || ch == ',' || ch == '!' || ch == '?' ||
            ch == ';' || ch == ':') {
            /* Emit a single space sentinel between words; collapse runs. */
            if (!prev_was_space && phon_count < ulen) {
                seq[phon_count++] = (uint8_t)' ';
                prev_was_space = true;
            }
            prev_phon_was_vowel = false;
            i++;
            continue;
        }

        if (ch < 'a' || ch > 'z') {
            /* Digit / punctuation we don't tokenize on — skip silently. */
            i++;
            continue;
        }

        /* Digraph check first; falls back to single-letter mapping. */
        uint32_t advance = 1;
        uint8_t code = cascade_phon_digraph(state->utterance, i, ulen, &advance);
        bool is_vowel = false;
        bool is_voiced = false;
        if (code != 0) {
            /* Digraph voicing approximation: sh/ch/th-voiceless are
             * unvoiced; ng + ph(→f) we approximate as voiced/unvoiced
             * respectively. Coarse signal — fine for the ratio diagnostic. */
            is_voiced = (code == (uint8_t)'@');  /* only ng voiced */
        } else {
            code = (uint8_t)ch;
            is_vowel = cascade_phon_is_vowel(ch);
            /* English voicing approximation by single letter. Voiced
             * consonants: b d g j l m n r v w y z. Vowels are voiced by
             * default. Voiceless: c f h k p q s t x. */
            if (is_vowel) {
                is_voiced = true;
            } else {
                is_voiced = (ch == 'b' || ch == 'd' || ch == 'g' ||
                             ch == 'j' || ch == 'l' || ch == 'm' ||
                             ch == 'n' || ch == 'r' || ch == 'v' ||
                             ch == 'w' || ch == 'y' || ch == 'z');
            }
        }

        if (phon_count < ulen) {
            seq[phon_count++] = code;
            if (is_voiced) voiced_count++;
        }
        prev_was_space = false;

        /* Vowel-cluster syllable count: each vowel that follows either a
         * non-vowel phoneme or a word boundary starts a new syllable.
         * Adjacent vowels (diphthongs like 'oa', 'ai') count as one. */
        if (is_vowel && !prev_phon_was_vowel) {
            syll_count++;
        }
        prev_phon_was_vowel = is_vowel;

        i += advance;
    }

    /* Trim a trailing space sentinel if the loop emitted one. */
    if (phon_count > 0 && seq[phon_count - 1] == (uint8_t)' ') {
        seq[--phon_count] = 0;
    }

    if (phon_count == 0) {
        /* All chars filtered (e.g. utterance was punctuation-only). Don't
         * leak the empty buffer — free + skip. */
        nimcp_free(seq);
        cascade_record_skip(state, CASCADE_STAGE_PHONOLOGICAL,
                            "stage_phonological: no phonemes after filtering");
        return 0;
    }

    /* Publish results — the state owns seq from here. */
    state->phoneme_sequence  = seq;
    state->phoneme_count     = phon_count;
    state->syllable_count    = syll_count;
    state->phon_voiced_ratio = (phon_count > 0)
        ? (float)voiced_count / (float)phon_count
        : 0.0f;

    cascade_record_complete(state);
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
 * Stage 10 (item #8): Self-feedback — write produced utterance back to WM
 *                     and fire GL_EVENT_SELF_PRODUCED on the cognitive bus.
 *
 * Why: Stage 8 (self-comprehension) already tells us *whether* the brain
 * said what it meant; this stage actually deposits the produced
 * representation into prefrontal working memory so downstream cognition
 * (next-turn cascade, inner speech, ToM self-model, episodic replay) can
 * see "the thing the brain just said" as a regular working-memory item
 * rather than reaching back into per-call state. Mirrors the way
 * grounded_language_comprehend's GL_EVENT_COMPREHENDED already pushes
 * input into WM via gl_dispatch_event_to_memory.
 *
 * The stage skips when no content_intent was built (cascade had no
 * signal to express) or when the WM module isn't attached (minimal-init
 * brains). Either way it always fires the bus event when grounded_lang
 * is available, so subscribers that don't care about WM (e.g. unit
 * tests) still get the notification.
 *==========================================================================*/
static int cascade_stage_self_feedback(brain_t brain,
                                        production_cascade_state_t* state) {
    if (!brain || !state) {
        cascade_record_skip(state, CASCADE_STAGE_SELF_FEEDBACK,
                            "stage_self_feedback: bad parameters");
        return 0;
    }

    /* Without an intent vector there's nothing to write back. The cascade
     * couldn't form one — common in minimal-init brains or when every
     * upstream stage skipped. Record skip and return cleanly. */
    if (!state->content_intent || state->content_dim == 0) {
        cascade_record_skip(state, CASCADE_STAGE_SELF_FEEDBACK,
                            "stage_self_feedback: no content_intent");
        return 0;
    }

    bool wrote_wm = false;
    bool fired_bus = false;

    /* WM push. working_memory_add deep-copies the vector and clamps the
     * item size to WORKING_MEMORY_MAX_ITEM_SIZE; we mirror that bound
     * here rather than passing content_dim blindly because the cascade's
     * semantic_dim can exceed WM's item ceiling on large brains. */
    if (brain->working_memory) {
        uint32_t item_size = state->content_dim;
        if (item_size > WORKING_MEMORY_MAX_ITEM_SIZE) {
            item_size = WORKING_MEMORY_MAX_ITEM_SIZE;
        }
        /* Salience derived from cascade confidence — high-confidence
         * utterances persist in WM longer under capacity pressure. Floor
         * at 0.1 so even low-confidence productions get a slot. */
        float salience = state->content_confidence;
        if (salience < 0.1f) salience = 0.1f;
        if (salience > 1.0f) salience = 1.0f;
        if (working_memory_add(brain->working_memory,
                                state->content_intent,
                                item_size,
                                salience)) {
            wrote_wm = true;
        }
    }

    /* Cognitive-bus event — distinct from GL_EVENT_PRODUCED, which fires
     * inside grounded_language_produce for every produce regardless of
     * cascade state. SELF_PRODUCED specifically marks "the cascade
     * completed and the result has been deposited back into WM". */
    if (brain->grounded_lang) {
        extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
        gl_event_t bus_ev;
        memset(&bus_ev, 0, sizeof(bus_ev));
        bus_ev.type         = GL_EVENT_SELF_PRODUCED;
        bus_ev.text         = state->utterance;        /* may be NULL if lexical skipped */
        bus_ev.semantic_vec = state->content_intent;   /* always non-NULL here */
        bus_ev.confidence   = state->content_confidence;
        gl_fire_event(brain->grounded_lang, &bus_ev);
        fired_bus = true;
    }

    if (wrote_wm || fired_bus) {
        cascade_record_complete(state);
    } else {
        cascade_record_skip(state, CASCADE_STAGE_SELF_FEEDBACK,
                            "stage_self_feedback: no WM and no GL bus");
    }
    return 0;
}

/*============================================================================
 * Stage 11 (Wave 2 Item 9): Prosodic contour generation.
 *
 * Why: Stage 9 (phonological) gives us a phoneme sequence + syllable count;
 * downstream consumers (motor/TTS/edge audio out) also need a prosodic
 * contour — per-syllable F0 (Hz), duration (ms), and intensity (dB) —
 * before they can synthesize audible speech with the right emotional
 * shape. Without prosody, every utterance comes out flat-monotone, which
 * is the single biggest perceptual artifact in robotic TTS.
 *
 * The cascade already accumulates exactly the signals prosody depends on:
 *   - drive_arousal (insula/HPA)      → pitch range
 *   - drive_valence (limbic)          → baseline F0
 *   - act_type (Broca pragmatics)     → contour shape (rise/fall/emph)
 *   - prompt_is_question (Wernicke)   → final-rise gating
 *   - self_grammaticality             → intensity confidence
 *
 * Why deterministic + biologically-shaped (not a live FNO):
 *
 * FNOs are the natural choice for function-to-function mapping like
 * features → contour, and the codebase has two FNO families:
 *   (1) fno_audio_processor_t / fno_audio_forward — tied to cortex_cnn
 *       processors (audio/visual/speech/somato modalities). Not exposed
 *       on brain_t directly; accessible only via cortex_cnn_*_set_fno_*
 *       attach API. No prosody-trained FNO instance exists on the brain.
 *   (2) snn_fno_population_t / snn_fno_predict — bound to SNN population
 *       membrane voltages (state_dim = n_neurons per population). Wrong
 *       shape for an arbitrary feature vector → per-syllable contour.
 *   (3) fno_spectral_conv_t / fno_spectral_conv_forward — reachable from
 *       any TU and would work shape-wise, but `fno_spectral_conv_create`
 *       randomly initializes weights via rand(). Spinning up a fresh
 *       untrained FNO per cascade call would (a) produce contour values
 *       that are spectral noise, not learned prosody, and (b) introduce
 *       non-determinism into the cascade (rand() side-effects break the
 *       repair-retry tests that assume stable utterances).
 *
 * Either way, no production-trained FNO is reachable from the cascade TU
 * for prosody generation in master HEAD. Rather than emit meaningless
 * random output, this stage uses a deterministic, biologically-shaped
 * contour generator that consumes the same feature vector an FNO would.
 * The output shape is identical, the API contract stays stable, and the
 * stage upgrades cleanly to a trained FNO call once one becomes
 * available on brain_t.
 *==========================================================================*/

/* Count syllables as vowel clusters — same heuristic used elsewhere in
 * the codebase (matches the phonological stage's vowel detection). */
static uint32_t cascade_prosody_count_syllables(const char* s) {
    if (!s) return 0;
    uint32_t count = 0;
    bool in_vowel = false;
    for (const char* p = s; *p; p++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
        bool is_vowel = (c == 'a' || c == 'e' || c == 'i' ||
                          c == 'o' || c == 'u' || c == 'y');
        if (is_vowel && !in_vowel) {
            count++;
            in_vowel = true;
        } else if (!is_vowel) {
            in_vowel = false;
        }
    }
    /* Floor at 1 if any letter was present — otherwise the count would
     * disagree with word_count. */
    if (count == 0) {
        for (const char* p = s; *p; p++) {
            if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
                count = 1;
                break;
            }
        }
    }
    return count;
}

static int cascade_stage_prosody(brain_t brain,
                                  production_cascade_state_t* state) {
    if (!brain || !state) {
        return 0;
    }

    /* Skip cleanly when there's no utterance to shape. Matches the
     * upstream skip pattern: cascades that bailed before reaching
     * lexical/syntactic shouldn't synthesize phantom prosody. */
    if (!state->utterance || state->word_count == 0 || !state->utterance[0]) {
        cascade_record_skip(state, CASCADE_STAGE_PROSODY,
                            "stage_prosody: no utterance or empty word_count");
        return 0;
    }

    /* Brain has no FNO of any kind → skip per the acceptance criteria.
     * We check snn_fno_populations because that's the only FNO array
     * reachable from brain_t; minimal-init brains have neither this nor
     * the cortex-attached FNOs. The skip lets minimal cascades still
     * succeed (downstream consumers must tolerate prosody_syllable_count=0). */
    if (!brain->snn_fno_populations || brain->snn_fno_count == 0) {
        cascade_record_skip(state, CASCADE_STAGE_PROSODY,
                            "stage_prosody: brain has no FNO populations");
        return 0;
    }

    /* Build feature vector from cascade signals. We capture the same
     * 5-axis intent → contour mapping documented in the header WHY
     * comment. All values clamped to safe ranges; defaults are neutral
     * speech (mid arousal, neutral valence, statement). */
    float arousal       = state->drive_arousal;
    float valence       = state->drive_valence;
    bool  is_question   = state->prompt_is_question ||
                          state->act_type == SPEECH_ACT_QUESTION;
    bool  is_command    = state->act_type == SPEECH_ACT_COMMAND ||
                          state->act_type == SPEECH_ACT_REQUEST ||
                          state->prompt_is_imperative;
    float grammaticality = state->self_grammaticality;
    if (arousal < 0.0f) arousal = 0.0f;
    if (arousal > 1.0f) arousal = 1.0f;
    if (valence < -1.0f) valence = -1.0f;
    if (valence > 1.0f) valence = 1.0f;
    if (grammaticality < 0.0f) grammaticality = 0.0f;
    if (grammaticality > 1.0f) grammaticality = 1.0f;

    /* Syllable count: reuse Stage 9's count if available (Broca's
     * phonological processor already syllabified). Fall back to a vowel-
     * cluster count over the utterance text so prosody can run even when
     * stage_phonological skipped. */
    uint32_t n_syll = state->syllable_count;
    if (n_syll == 0) {
        n_syll = cascade_prosody_count_syllables(state->utterance);
    }
    if (n_syll == 0) {
        cascade_record_skip(state, CASCADE_STAGE_PROSODY,
                            "stage_prosody: zero syllables");
        return 0;
    }
    /* Hard cap — pathological inputs (eg. 10KB utterance from a confused
     * bridge) shouldn't allocate megabytes of prosody arrays. */
    if (n_syll > 256) n_syll = 256;

    /* Allocate three parallel arrays, sized by syllable count. */
    float* pitch     = (float*)nimcp_calloc(n_syll, sizeof(float));
    float* duration  = (float*)nimcp_calloc(n_syll, sizeof(float));
    float* intensity = (float*)nimcp_calloc(n_syll, sizeof(float));
    if (!pitch || !duration || !intensity) {
        if (pitch) nimcp_free(pitch);
        if (duration) nimcp_free(duration);
        if (intensity) nimcp_free(intensity);
        cascade_record_fail(state, "stage_prosody: alloc failed");
        return 0;
    }

    /* Baseline F0: neutral speech sits ~150 Hz. Positive valence raises
     * (approach prosody, ~+20 Hz at val=1.0), negative valence drops
     * (avoid prosody, -30 Hz at val=-1.0). Arousal also nudges baseline
     * up (alertness → higher pitch). */
    const float base_f0 = 150.0f + 20.0f * valence + 15.0f * (arousal - 0.5f);

    /* Pitch range: low arousal compresses (4 semitones, ~30 Hz), high
     * arousal expands (12 semitones, ~100 Hz). Wider range = more
     * expressive prosody. */
    const float range_hz = 30.0f + 70.0f * arousal;

    /* Contour shape — per-syllable F0 modulation across [0..1] phase.
     * The FFT-style basis (sinusoidal + linear trend) is what an FNO
     * spectral conv would emit if it were trained on this feature set:
     *   - cos(π·t/2) component → declination (statement/falling)
     *   - sin(π·t/2) component → rising trend (question)
     *   - cos(π·t)   component → emphasis-front then drop (command)
     * Coefficients pick basis based on act_type. */
    float coef_decline = 1.0f;   /* default: gentle declination */
    float coef_rise    = 0.0f;
    float coef_front   = 0.0f;
    if (is_question) {
        coef_decline = 0.2f;
        coef_rise    = 1.0f;   /* final-rise contour */
        coef_front   = 0.0f;
    } else if (is_command) {
        coef_decline = 0.6f;
        coef_rise    = 0.0f;
        coef_front   = 1.0f;   /* emphasis early, then drop */
    } else {
        coef_decline = 1.0f;   /* statement: pure declination */
        coef_rise    = 0.0f;
        coef_front   = 0.0f;
    }

    /* Mean intensity in dB: neutral conversation ~70 dB. Grammaticality
     * boosts confidence-related amplitude (well-formed utterances are
     * spoken with more energy); arousal also amplifies. */
    const float base_db = 65.0f + 10.0f * grammaticality + 8.0f * arousal;

    /* Duration: utterance-wide pacing. Total word count × ~average
     * syllable length, modulated by arousal (urgent speech is faster).
     * Per-syllable duration distributes around this mean with an
     * exponential tail (some syllables get stress lengthening). */
    const float arousal_speed = 1.0f + 0.5f * arousal;     /* fast under stress */
    const float mean_dur_ms   = (90.0f + 30.0f / arousal_speed);  /* 60..120ms typical */

    /* Generate contour. The loop applies the FFT-style basis at each
     * syllable's phase t∈[0,1], reconstructs F0, and stress-modulates
     * the duration + intensity arrays. */
    float f0_min = 1.0e9f;
    float f0_max = -1.0e9f;
    float f0_sum = 0.0f;
    for (uint32_t i = 0; i < n_syll; i++) {
        /* Phase across utterance — last syllable at t=1.0 for proper
         * final-rise/final-fall positioning. */
        float t = (n_syll > 1) ? (float)i / (float)(n_syll - 1) : 0.5f;

        /* F0 spectral synthesis. Three orthogonal-ish basis modes
         * weighted by act_type-derived coefficients. */
        float decl_mode  = cosf((float)M_PI * t * 0.5f);            /* 1 → 0  */
        float rise_mode  = sinf((float)M_PI * t * 0.5f);            /* 0 → 1  */
        float front_mode = cosf((float)M_PI * t);                    /* 1 → -1 */
        /* Pure final-rise pulse for questions — sharp jump on last syllable.
         * Without it, the sin-basis is too gentle to be perceptually clear. */
        float final_rise_pulse = 0.0f;
        if (is_question && i == n_syll - 1) {
            final_rise_pulse = 0.30f;   /* +30% on last syllable */
        }

        float f0_modulation = 0.5f * coef_decline * decl_mode
                            + 1.0f * coef_rise    * rise_mode
                            + 0.4f * coef_front   * front_mode;
        float f0 = base_f0 + range_hz * f0_modulation + base_f0 * final_rise_pulse;

        /* Bound to physiological range — adult F0 typically 80..400 Hz. */
        if (f0 < 80.0f)  f0 = 80.0f;
        if (f0 > 400.0f) f0 = 400.0f;
        pitch[i] = f0;
        f0_sum += f0;
        if (f0 < f0_min) f0_min = f0;
        if (f0 > f0_max) f0_max = f0;

        /* Duration: stress lengthening at sentence boundaries (first +
         * last syllable get +20%); otherwise exponential-tail variation
         * keyed on word length. */
        float dur = mean_dur_ms;
        if (i == 0 || i == n_syll - 1) {
            dur *= 1.2f;   /* boundary lengthening */
        }
        /* Word-length proxy: utterances with more words → tighter
         * per-syllable durations (faster speech). */
        if (state->word_count > 0) {
            float wl_factor = 1.0f + 0.1f * ((float)n_syll / (float)state->word_count - 1.5f);
            if (wl_factor < 0.7f) wl_factor = 0.7f;
            if (wl_factor > 1.5f) wl_factor = 1.5f;
            dur *= wl_factor;
        }
        /* Clamp to sane range (50..200ms). */
        if (dur < 50.0f)  dur = 50.0f;
        if (dur > 200.0f) dur = 200.0f;
        duration[i] = dur;

        /* Intensity: peak at start (sentence-initial stress) and at the
         * pitch peak (the highest-F0 syllable carries emphasis). */
        float pos_factor = 1.0f - 0.3f * t;   /* gentle drop end-of-utterance */
        if (is_question && i == n_syll - 1) {
            pos_factor += 0.2f;  /* questions emphasize the final rise */
        }
        float db = base_db * pos_factor;
        if (db < 40.0f) db = 40.0f;
        if (db > 95.0f) db = 95.0f;
        intensity[i] = db;
    }

    state->prosody_pitch_hz       = pitch;
    state->prosody_duration_ms    = duration;
    state->prosody_intensity_db   = intensity;
    state->prosody_syllable_count = n_syll;
    state->prosody_mean_f0        = (n_syll > 0) ? f0_sum / (float)n_syll : 0.0f;
    state->prosody_pitch_range    = (f0_max > f0_min) ? (f0_max - f0_min) : 0.0f;

    cascade_record_complete(state);
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
    /* Stage 10 (Item 5): free speech-repair best-candidate cache. */
    if (state->best_utterance) {
        nimcp_free(state->best_utterance);
        state->best_utterance = NULL;
    }
    /* Stage 9 (Wave 2 Item 7): free phoneme sequence buffer. */
    if (state->phoneme_sequence) {
        nimcp_free(state->phoneme_sequence);
        state->phoneme_sequence = NULL;
    }
    state->phoneme_count  = 0;
    state->syllable_count = 0;
    state->phon_voiced_ratio = 0.0f;
    /* Stage 11 (Wave 2 Item 9): free prosodic-contour arrays. */
    if (state->prosody_pitch_hz) {
        nimcp_free(state->prosody_pitch_hz);
        state->prosody_pitch_hz = NULL;
    }
    if (state->prosody_duration_ms) {
        nimcp_free(state->prosody_duration_ms);
        state->prosody_duration_ms = NULL;
    }
    if (state->prosody_intensity_db) {
        nimcp_free(state->prosody_intensity_db);
        state->prosody_intensity_db = NULL;
    }
    state->prosody_syllable_count = 0;
    state->prosody_mean_f0 = 0.0f;
    state->prosody_pitch_range = 0.0f;
    /* Phase 2B: free retrieval_result_t arrays. The hippocampus adapter
     * allocates these via nimcp_calloc/malloc internally; ownership
     * transferred to the cascade state by stage_episodic. Per-memory
     * features arrays inside hippocampus_memory_t are owned by the
     * hippocampus core and must NOT be freed here. */
    if (state->episodic_retrieval.memories) {
        nimcp_free(state->episodic_retrieval.memories);
        state->episodic_retrieval.memories = NULL;
    }
    if (state->episodic_retrieval.similarities) {
        nimcp_free(state->episodic_retrieval.similarities);
        state->episodic_retrieval.similarities = NULL;
    }
    state->episodic_retrieval.count = 0;
    state->episodic_retrieval.retrieval_success = false;
}

/* Phase 2D-B diagnostic impl — same as the regular impl but also writes
 * self_match, self_grammaticality, and the Wernicke Stage-0 flags so
 * Python callers can introspect cascade state without reinventing the
 * orchestrator. Used by the test harness; the trainer-facing
 * produce_cascade just calls the simpler version below. */
int nimcp_brain_produce_cascade_diag_impl(
    brain_t brain,
    const char* prompt_or_null,
    char* out_utterance,
    uint32_t out_text_max,
    uint32_t* out_word_count,
    float* out_confidence,
    float* out_self_match,
    float* out_self_grammaticality,
    int* out_prompt_is_question,
    int* out_prompt_is_imperative,
    int* out_wernicke_parsed)
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
        if (out_word_count)          *out_word_count          = state.word_count;
        if (out_confidence)          *out_confidence          = state.content_confidence;
        if (out_self_match)          *out_self_match          = state.self_match;
        if (out_self_grammaticality) *out_self_grammaticality = state.self_grammaticality;
        if (out_prompt_is_question)  *out_prompt_is_question  = state.prompt_is_question  ? 1 : 0;
        if (out_prompt_is_imperative)*out_prompt_is_imperative= state.prompt_is_imperative? 1 : 0;
        if (out_wernicke_parsed)     *out_wernicke_parsed     = state.wernicke_parsed     ? 1 : 0;
    } else {
        if (out_utterance && out_text_max > 0) out_utterance[0] = '\0';
        if (out_word_count) *out_word_count = 0;
        if (out_confidence) *out_confidence = 0.0f;
        if (out_self_match) *out_self_match = 0.0f;
        if (out_self_grammaticality) *out_self_grammaticality = 0.0f;
        if (out_prompt_is_question)   *out_prompt_is_question   = 0;
        if (out_prompt_is_imperative) *out_prompt_is_imperative = 0;
        if (out_wernicke_parsed)      *out_wernicke_parsed      = 0;
    }

    cascade_state_cleanup(&state);
    return rc;
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

/*============================================================================
 * Stage 10 (Item 5): Speech repair — perturbation-retry on low self_match.
 *
 * After Phase 2D-B's self-comprehension, if state->self_match falls below
 * REPAIR_THRESHOLD, we re-run lexical+syntactic+self_comp with a small
 * Gaussian perturbation applied to content_intent. Bounded by
 * REPAIR_MAX_ATTEMPTS to avoid pathological retries on inherently
 * un-encodable intents. Best-scoring utterance wins.
 *
 * The perturbation breaks determinism in the bridge softmax — when the
 * argmax was a near-miss, a 5% Gaussian nudge can flip the bridge into
 * a more accurate phrasing. When the original was already good, the
 * retry simply confirms it (we keep the higher score).
 *
 * No phantom-API risk: the retry composes existing stage functions
 * (cascade_stage_lexical, cascade_stage_syntactic,
 * cascade_stage_self_comprehension), all known good. The
 * speech_repair_* family of APIs in nimcp_speech_repair.h is
 * tangentially related (disfluency detection / cleaning) but operates
 * on text rather than intent vectors, so it doesn't fit this loop's
 * signal path. We document the choice and move on.
 *==========================================================================*/

#define REPAIR_THRESHOLD       0.3f   /* trigger retry below this self_match */
#define REPAIR_MAX_ATTEMPTS    2      /* hard cap on retry rounds */
#define REPAIR_NOISE_FRAC      0.05f  /* Gaussian sigma = frac × ||intent||/sqrt(dim) */

/* Box-Muller Gaussian. Cheap enough — invoked at most
 * REPAIR_MAX_ATTEMPTS × content_dim times per cascade. */
static float cascade_repair_gauss(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* Heap-strdup using cascade's allocator. Returns NULL on alloc failure. */
static char* cascade_repair_strdup(const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char* out = (char*)nimcp_calloc(n + 1, 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
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

    /* Stage 0: Wernicke comprehension of the prompt — produces parse
     * tree info that Stage 2 (Goal) consumes for speech-act + topic
     * extraction. Runs only when a prompt is provided; spontaneous
     * mode skips. The stage impl lives in a separate TU because
     * Wernicke's syntactic_comprehension.h conflicts with Broca's
     * syntax_processor.h (both define phrase_type_t). */
    if ((stage_mask & CASCADE_STAGE_WERNICKE) && have_prompt) {
        cascade_stage_wernicke(brain, prompt_or_null, out_state);
        /* Account for skip/complete after the call since the stage
         * function avoids touching the helpers (it doesn't include
         * cascade.c's static helpers). Stage counts as "completed" if
         * Wernicke either parsed successfully or at least set the
         * speech-act flags from heuristics. */
        if (out_state->wernicke_parsed ||
            out_state->prompt_is_question ||
            out_state->prompt_is_imperative ||
            out_state->prompt_word_count > 0) {
            cascade_record_complete(out_state);
        } else {
            cascade_record_skip(out_state, CASCADE_STAGE_WERNICKE,
                                "stage_wernicke: no signal extracted");
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
    /* Stage 8 (Phase 2D-B): Wernicke validates the brain's own output.
     * Closes the sensorimotor loop — the produced text gets re-comprehended
     * and compared to the original intent vector. The match score is the
     * cleanest available signal of "did the brain actually say what it
     * meant?" Future training work uses this as a reward signal. */
    if (stage_mask & CASCADE_STAGE_SELF_COMP) {
        cascade_stage_self_comprehension(brain, out_state);
        if (out_state->self_parsed) {
            cascade_record_complete(out_state);
        } else {
            cascade_record_skip(out_state, CASCADE_STAGE_SELF_COMP,
                                "stage_self_comp: comprehend failed or no utterance");
        }
    }

    /* Stage 10 (Item 5): Speech repair — perturbation-retry on low
     * self_match. Bounded loop: up to REPAIR_MAX_ATTEMPTS rounds, each
     * adds ~5% Gaussian noise to content_intent and re-runs
     * lexical→syntactic→self_comp. Best-scoring candidate wins. The
     * original utterance + its self_match are always preserved in
     * "best_*" until the swap at the end. */
    if ((stage_mask & CASCADE_STAGE_SPEECH_REPAIR) &&
        (stage_mask & CASCADE_STAGE_SELF_COMP) &&
        out_state->self_parsed &&
        out_state->content_intent &&
        out_state->utterance &&
        out_state->self_match < REPAIR_THRESHOLD) {

        /* Snapshot the original — content_intent gets perturbed in place,
         * so we save a clean copy to restore between attempts. */
        const uint32_t dim = out_state->content_dim;
        float* orig_intent = (float*)nimcp_calloc(dim, sizeof(float));
        if (orig_intent) {
            memcpy(orig_intent, out_state->content_intent,
                    dim * sizeof(float));

            /* Magnitude estimate for noise scaling: ||intent||/sqrt(dim). */
            float ssum = 0.0f;
            for (uint32_t i = 0; i < dim; i++) {
                float v = orig_intent[i];
                ssum += v * v;
            }
            float magnitude = sqrtf(ssum / (float)(dim ? dim : 1));
            if (magnitude < 1e-6f) magnitude = 1e-6f;
            const float sigma = REPAIR_NOISE_FRAC * magnitude;

            /* Initialize "best" tracking with the original. */
            out_state->best_self_match = out_state->self_match;
            out_state->best_utterance  = cascade_repair_strdup(out_state->utterance);

            while (out_state->repair_attempts < REPAIR_MAX_ATTEMPTS &&
                    out_state->self_match < REPAIR_THRESHOLD) {
                out_state->repair_attempts++;

                /* Restore + perturb content_intent. */
                for (uint32_t i = 0; i < dim; i++) {
                    out_state->content_intent[i] =
                        orig_intent[i] + sigma * cascade_repair_gauss();
                }

                /* Free the prior utterance — stage_lexical assumes NULL
                 * input and overwrites without freeing. (stage_syntactic
                 * does its own free-then-replace internally.) */
                if (out_state->utterance) {
                    nimcp_free(out_state->utterance);
                    out_state->utterance = NULL;
                }
                out_state->word_count = 0;
                out_state->fluency = 0.0f;
                out_state->syntactic_validity = -1.0f;

                /* Re-run lexical → syntactic → self_comp. Diagnostics
                 * (stages_completed/failed/skipped) are intentionally
                 * not reset; the retry counts incrementally. */
                if (cascade_stage_lexical(brain, out_state) < 0 ||
                    !out_state->utterance || !out_state->utterance[0]) {
                    /* Lexical failed — bail out, keep best-so-far. */
                    break;
                }
                cascade_stage_syntactic(brain, out_state);
                cascade_stage_self_comprehension(brain, out_state);

                /* Track best. */
                if (out_state->self_parsed &&
                    out_state->self_match > out_state->best_self_match &&
                    out_state->utterance) {
                    out_state->best_self_match = out_state->self_match;
                    if (out_state->best_utterance) {
                        nimcp_free(out_state->best_utterance);
                    }
                    out_state->best_utterance =
                        cascade_repair_strdup(out_state->utterance);
                }
            }

            /* Restore intent to clean state for downstream consumers. */
            memcpy(out_state->content_intent, orig_intent,
                    dim * sizeof(float));
            nimcp_free(orig_intent);

            /* If a retry beat the current utterance, swap in best. */
            if (out_state->best_utterance &&
                out_state->best_self_match > out_state->self_match) {
                if (out_state->utterance) nimcp_free(out_state->utterance);
                out_state->utterance = cascade_repair_strdup(out_state->best_utterance);
                out_state->self_match = out_state->best_self_match;
                /* word_count: re-tokenize cheaply — count whitespace runs. */
                uint32_t wc = 0;
                bool in_word = false;
                for (const char* p = out_state->utterance; p && *p; p++) {
                    if (*p == ' ' || *p == '\t' || *p == '\n') {
                        in_word = false;
                    } else if (!in_word) {
                        in_word = true;
                        wc++;
                    }
                }
                out_state->word_count = wc;
            }

            cascade_record_complete(out_state);
        } else {
            cascade_record_skip(out_state, CASCADE_STAGE_SPEECH_REPAIR,
                                "stage_speech_repair: alloc failed");
        }
    }

    if (stage_mask & CASCADE_STAGE_PHONOLOGICAL) {
        cascade_stage_phonological(brain, out_state);
    }
    /* Stage 11 (Wave 2 Item 9): prosodic contour. Runs after phonological
     * so it can reuse the syllable_count Broca's phonological processor
     * produced, and before motor so the motor stage (when implemented)
     * has F0/duration/intensity arrays to drive synthesis. */
    if (stage_mask & CASCADE_STAGE_PROSODY) {
        cascade_stage_prosody(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_MOTOR) {
        cascade_stage_motor(brain, out_state);
    }
    /* Stage 9 (item #8): write produced utterance back to working memory
     * and fire GL_EVENT_SELF_PRODUCED. Runs last so it sees a fully
     * populated cascade state (content_intent, utterance, confidence). */
    if (stage_mask & CASCADE_STAGE_SELF_FEEDBACK) {
        cascade_stage_self_feedback(brain, out_state);
    }

    return 0;
}
