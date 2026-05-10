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

/* Stage 0 (Wernicke comprehension) lives in nimcp_communication_cascade_wernicke.c
 * because Wernicke and Broca both define phrase_type_t with overlapping enum
 * values; pulling both into one TU triggers a redeclaration error. */
extern int cascade_stage_wernicke(brain_t brain, const char* prompt,
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
    if (stage_mask & CASCADE_STAGE_PHONOLOGICAL) {
        cascade_stage_phonological(brain, out_state);
    }
    if (stage_mask & CASCADE_STAGE_MOTOR) {
        cascade_stage_motor(brain, out_state);
    }

    return 0;
}
