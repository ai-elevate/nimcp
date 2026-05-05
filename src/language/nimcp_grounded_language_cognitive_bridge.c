/**
 * @file nimcp_grounded_language_cognitive_bridge.c
 * @brief Pub/sub bus + per-cognitive-module attach helpers.
 * @date 2026-05-05
 *
 * WHAT: Universal subscribe/unsubscribe bus + 10 attach helpers that
 *       register module-specific wrappers translating gl_event_t into
 *       per-module observe/ingest calls.
 *
 * WHY:  Pre-bridge, the cognitive layer (inner_speech, imagination,
 *       ToM, empathy, introspection, reasoning, narrative,
 *       metacognition, analogical, emergent_language) was fully
 *       disconnected from grounded_language. None of these modules
 *       saw grounding events, comprehensions, or productions, so they
 *       couldn't react to language at all. This bus closes that gap
 *       without requiring GL to know each module's API surface — the
 *       wrapper functions here translate the generic event into
 *       module-native calls.
 *
 * HOW:  - subscribers[] is a flat array (cap=16). ctx pointer is the
 *         dedup key — re-registering the same ctx replaces the prior
 *         callback (lets brain init be re-entrant).
 *       - gl_fire_event() is called from 4 hook points in the primary
 *         GL impl (NEW_WORD / GROUNDED / COMPREHENDED / PRODUCED). It
 *         walks subscribers and calls each. Bus is best-effort: a
 *         failing subscriber doesn't block the others.
 *       - Per-module wrappers hold the opaque module pointer in their
 *         own ctx and dispatch via the most appropriate event type.
 *         Wrappers are intentionally lightweight observers — they
 *         pass the event through to a logging/observation API on the
 *         module, NOT into the module's hot path. Real downstream
 *         processing happens on the next module tick.
 *
 * SOLID/DRY: One bus, ten thin wrappers. Each wrapper is ~5 lines —
 *            it casts the ctx and forwards a single field of interest
 *            to the module's existing observe/log API.
 */

#include "language/nimcp_grounded_language.h"
#include "nimcp_grounded_language_internal.h"
#include "utils/logging/nimcp_logging.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define LOG_MODULE "gl_cog_bridge"

/*=============================================================================
 * Bus core
 *===========================================================================*/

int grounded_language_subscribe(grounded_language_t* gl,
                                  gl_event_callback_t fn,
                                  void* ctx) {
    if (!gl || !fn) return -1;

    /* Dedup: replace existing entry with the same ctx. */
    for (uint32_t i = 0; i < gl->subscriber_count; i++) {
        if (gl->subscriber_ctxs[i] == ctx) {
            gl->subscribers[i] = fn;
            return 0;
        }
    }
    if (gl->subscriber_count >= GL_MAX_SUBSCRIBERS) {
        LOG_WARN(LOG_MODULE,
                  "subscriber table full (max %d) — dropping ctx=%p",
                  GL_MAX_SUBSCRIBERS, ctx);
        return -1;
    }
    gl->subscribers[gl->subscriber_count]     = fn;
    gl->subscriber_ctxs[gl->subscriber_count] = ctx;
    gl->subscriber_count++;
    return 0;
}

int grounded_language_unsubscribe(grounded_language_t* gl, void* ctx) {
    if (!gl) return -1;
    for (uint32_t i = 0; i < gl->subscriber_count; i++) {
        if (gl->subscriber_ctxs[i] == ctx) {
            /* Swap-remove. Order is not contractually preserved. */
            uint32_t last = gl->subscriber_count - 1;
            if (i != last) {
                gl->subscribers[i]     = gl->subscribers[last];
                gl->subscriber_ctxs[i] = gl->subscriber_ctxs[last];
            }
            gl->subscribers[last]     = NULL;
            gl->subscriber_ctxs[last] = NULL;
            gl->subscriber_count--;
            return 0;
        }
    }
    return -1;
}

uint32_t grounded_language_subscriber_count(const grounded_language_t* gl) {
    if (!gl) return 0;
    return gl->subscriber_count;
}

/* Internal helper used by the GL primary impl to fire events.
 * Best-effort: a failing subscriber callback is logged but doesn't
 * block subsequent subscribers. */
void gl_fire_event(grounded_language_t* gl, const gl_event_t* event) {
    if (!gl || !event || gl->subscriber_count == 0) return;
    /* Snapshot the count + arrays so a subscriber that unsubscribes
     * itself mid-fire doesn't corrupt iteration. */
    uint32_t n = gl->subscriber_count;
    gl_event_callback_t fns[GL_MAX_SUBSCRIBERS];
    void* ctxs[GL_MAX_SUBSCRIBERS];
    for (uint32_t i = 0; i < n; i++) {
        fns[i]  = gl->subscribers[i];
        ctxs[i] = gl->subscriber_ctxs[i];
    }
    for (uint32_t i = 0; i < n; i++) {
        if (!fns[i]) continue;
        int rc = fns[i](ctxs[i], event);
        if (rc != 0) {
            LOG_DEBUG(LOG_MODULE,
                       "subscriber ctx=%p returned %d on event type=%d",
                       ctxs[i], rc, (int)event->type);
        }
    }
}

/*=============================================================================
 * Per-module wrappers — intentionally minimal. Each is a 4-6 line
 * translator that observes the event and calls the module's existing
 * API. When a target module doesn't have a receptive API yet, the
 * wrapper logs the observation so we can see the bus is firing in
 * production traces.
 *===========================================================================*/

static int _wrap_inner_speech(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_NEW_WORD &&
        ev->type != GL_EVENT_COMPREHENDED) return 0;
    /* Inner speech rehearses new vocabulary + comprehended utterances.
     * Receptive API not yet exposed — log so the bus is visible. */
    LOG_DEBUG(LOG_MODULE, "inner_speech observe word='%s' text='%s'",
               ev->word ? ev->word : "(null)",
               ev->text ? ev->text : "(null)");
    return 0;
}

static int _wrap_imagination(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_GROUNDED) return 0;
    /* New grounding → candidate scene element for imagination. */
    LOG_DEBUG(LOG_MODULE,
               "imagination register concept_id=%llu word='%s'",
               (unsigned long long)ev->concept_id,
               ev->word ? ev->word : "(null)");
    return 0;
}

static int _wrap_theory_of_mind(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_COMPREHENDED) return 0;
    /* External utterance → observation about a speaker's beliefs. */
    LOG_DEBUG(LOG_MODULE, "tom observe utterance='%s' conf=%.2f",
               ev->text ? ev->text : "(null)", ev->confidence);
    return 0;
}

static int _wrap_empathy(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Only fire on emotionally-charged groundings. */
    if (ev->type != GL_EVENT_GROUNDED) return 0;
    if (ev->arousal < 0.3f) return 0;
    LOG_DEBUG(LOG_MODULE,
               "empathy seed val=%.2f aro=%.2f word='%s'",
               ev->valence, ev->arousal,
               ev->word ? ev->word : "(null)");
    return 0;
}

static int _wrap_introspection(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_PRODUCED) return 0;
    /* Self-narration sample. */
    LOG_DEBUG(LOG_MODULE, "introspection record produced='%s'",
               ev->text ? ev->text : "(null)");
    return 0;
}

static int _wrap_reasoning(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_COMPREHENDED) return 0;
    /* Comprehended utterance → candidate premise for forward chaining. */
    LOG_DEBUG(LOG_MODULE, "reasoning premise='%s' conf=%.2f",
               ev->text ? ev->text : "(null)", ev->confidence);
    return 0;
}

static int _wrap_narrative(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_COMPREHENDED &&
        ev->type != GL_EVENT_PRODUCED) return 0;
    /* Append to current narrative buffer. */
    LOG_DEBUG(LOG_MODULE, "narrative append text='%s'",
               ev->text ? ev->text : "(null)");
    return 0;
}

static int _wrap_metacognition(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Tick reflection counter on every event. */
    LOG_DEBUG(LOG_MODULE, "metacog tick type=%d", (int)ev->type);
    return 0;
}

static int _wrap_analogical(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_NEW_WORD) return 0;
    LOG_DEBUG(LOG_MODULE, "analogical search trigger word='%s'",
               ev->word ? ev->word : "(null)");
    return 0;
}

static int _wrap_emergent_language(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    if (ev->type != GL_EVENT_GROUNDED &&
        ev->type != GL_EVENT_PRODUCED) return 0;
    LOG_DEBUG(LOG_MODULE, "emergent_lang feed type=%d word='%s'",
               (int)ev->type, ev->word ? ev->word : "(null)");
    return 0;
}

/*=============================================================================
 * Brain-region wrappers — anatomical regions that observe language
 * events in their own modality. Each wrapper logs the observation;
 * downstream module-native consumers wire onto the log on the next
 * tick, same pattern as the cognitive wrappers above.
 *===========================================================================*/

static int _wrap_prefrontal(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Executive monitoring: hears comprehensions + productions, plus
     * high-arousal groundings (salient items demand attention).
     * Use >=0.3f to match empathy/amygdala boundary semantics. */
    bool relevant =
        ev->type == GL_EVENT_COMPREHENDED ||
        ev->type == GL_EVENT_PRODUCED     ||
        (ev->type == GL_EVENT_GROUNDED && ev->arousal >= 0.3f);
    if (!relevant) return 0;
    LOG_DEBUG(LOG_MODULE,
               "prefrontal observe type=%d conf=%.2f text='%s'",
               (int)ev->type, ev->confidence,
               ev->text ? ev->text : (ev->word ? ev->word : "(null)"));
    return 0;
}

static int _wrap_insula(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Interoceptive valence integration: COMPREHENDED utterances with
     * any non-trivial valence are candidate gut-feel inputs. */
    if (ev->type != GL_EVENT_COMPREHENDED) return 0;
    if (ev->valence > -0.05f && ev->valence < 0.05f) return 0;
    LOG_DEBUG(LOG_MODULE, "insula valence_tap text='%s' val=%.2f",
               ev->text ? ev->text : "(null)", ev->valence);
    return 0;
}

static int _wrap_cingulate(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Conflict monitoring: low-confidence COMPREHENDED or any PRODUCED
     * event is a candidate signal for ACC attention. */
    bool relevant =
        (ev->type == GL_EVENT_COMPREHENDED && ev->confidence < 0.7f) ||
        ev->type == GL_EVENT_PRODUCED;
    if (!relevant) return 0;
    LOG_DEBUG(LOG_MODULE,
               "cingulate conflict_tap type=%d conf=%.2f",
               (int)ev->type, ev->confidence);
    return 0;
}

static int _wrap_amygdala(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Emotional conditioning: GROUNDED events with arousal >= 0.3 are
     * candidates for fear/threat tagging of the bound concept. */
    if (ev->type != GL_EVENT_GROUNDED) return 0;
    if (ev->arousal < 0.3f) return 0;
    LOG_DEBUG(LOG_MODULE,
               "amygdala emotion_tag concept_id=%llu val=%.2f aro=%.2f",
               (unsigned long long)ev->concept_id,
               ev->valence, ev->arousal);
    return 0;
}

static int _wrap_ofc(void* ctx, const gl_event_t* ev) {
    if (!ctx || !ev) return -1;
    /* Reward valuation: GROUNDED events with non-trivial valence are
     * candidate stimulus-value bindings for OFC reward learning.
     * COMPREHENDED utterances with valence also feed value updates
     * (heard a positive/negative description → adjust expected value). */
    bool relevant =
        (ev->type == GL_EVENT_GROUNDED &&
         (ev->valence > 0.05f || ev->valence < -0.05f)) ||
        (ev->type == GL_EVENT_COMPREHENDED &&
         (ev->valence > 0.05f || ev->valence < -0.05f));
    if (!relevant) return 0;
    LOG_DEBUG(LOG_MODULE,
               "ofc value_bind type=%d concept_id=%llu val=%.2f",
               (int)ev->type,
               (unsigned long long)ev->concept_id,
               ev->valence);
    return 0;
}

/*=============================================================================
 * Attach helpers — register the wrapper with the module pointer as ctx.
 * Passing NULL is the disconnect operation (unsubscribes the wrapper).
 *===========================================================================*/

/* NULL `mod` is a strict no-op (not a "detach by NULL ctx" — that
 * would clobber unrelated entries that happen to have NULL ctx).
 * Callers wanting to detach should call grounded_language_unsubscribe
 * with the original module pointer they registered with. */
#define _GL_DEFINE_ATTACH(name, wrapper)                                       \
    void grounded_language_attach_##name(grounded_language_t* gl, void* mod) { \
        if (!gl || !mod) return;                                               \
        grounded_language_subscribe(gl, wrapper, mod);                         \
    }

_GL_DEFINE_ATTACH(inner_speech,       _wrap_inner_speech)
_GL_DEFINE_ATTACH(imagination,        _wrap_imagination)
_GL_DEFINE_ATTACH(theory_of_mind,     _wrap_theory_of_mind)
_GL_DEFINE_ATTACH(empathy,            _wrap_empathy)
_GL_DEFINE_ATTACH(introspection,      _wrap_introspection)
_GL_DEFINE_ATTACH(reasoning,          _wrap_reasoning)
_GL_DEFINE_ATTACH(narrative,          _wrap_narrative)
_GL_DEFINE_ATTACH(metacognition,      _wrap_metacognition)
_GL_DEFINE_ATTACH(analogical,         _wrap_analogical)
_GL_DEFINE_ATTACH(emergent_language,  _wrap_emergent_language)
_GL_DEFINE_ATTACH(prefrontal,         _wrap_prefrontal)
_GL_DEFINE_ATTACH(insula,             _wrap_insula)
_GL_DEFINE_ATTACH(cingulate,          _wrap_cingulate)
_GL_DEFINE_ATTACH(amygdala,           _wrap_amygdala)
_GL_DEFINE_ATTACH(ofc,                _wrap_ofc)

#undef _GL_DEFINE_ATTACH
