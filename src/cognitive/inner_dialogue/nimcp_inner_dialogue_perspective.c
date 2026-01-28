/**
 * @file nimcp_inner_dialogue_perspective.c
 * @brief Perspective Registry and Built-in Stub Perspectives
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Manages cognitive perspective registration, scheduling, and built-in stubs
 * WHY:  Engine needs a registry to iterate perspectives and select which speaks next
 * HOW:  Array-based registry with priority computation and built-in formulate stubs
 *
 * BIOLOGICAL: Each registered perspective represents a distinct cortical network
 * contributing its specialised "voice" to the internal conversation.
 *
 * @author NIMCP Development Team
 */

#include "cognitive/inner_dialogue/nimcp_inner_dialogue_perspective.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

static nimcp_health_agent_t* g_perspective_health_agent = NULL;

void inner_dialogue_perspective_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_perspective_health_agent = agent;
}

static inline void perspective_heartbeat(const char* op, float progress) {
    if (g_perspective_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_perspective_health_agent, op, progress);
    }
}

/** @brief Send heartbeat from perspective module (instance-level) */
static inline void perspective_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_perspective_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_perspective_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_perspective_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


/* ============================================================================
 * Perspective Type String Table
 * ============================================================================ */

static const char* s_perspective_type_names[] = {
    "ANALYTICAL",
    "EMOTIONAL",
    "CRITICAL",
    "CREATIVE",
    "MEMORY",
    "ETHICAL",
    "METACOGNITIVE"
};

const char* perspective_type_to_string(perspective_type_t type) {
    if ((unsigned)type < PERSPECTIVE_BUILTIN_COUNT) {
        return s_perspective_type_names[(unsigned)type];
    }
    return "CUSTOM/UNKNOWN";
}

/* ============================================================================
 * Registry Lifecycle
 * ============================================================================ */

int inner_dialogue_perspective_registry_init(
    inner_dialogue_perspective_registry_t* registry) {
    if (!registry) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL,
                              "inner_dialogue_perspective: registry_init called with NULL");
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL;
    }
    memset(registry, 0, sizeof(inner_dialogue_perspective_registry_t));
    NIMCP_LOGGING_DEBUG("inner_dialogue_perspective: registry initialised");
    perspective_heartbeat("perspective_registry_init", 1.0f);
    return 0;
}

void inner_dialogue_perspective_registry_clear(
    inner_dialogue_perspective_registry_t* registry) {
    if (!registry) return;
    NIMCP_LOGGING_DEBUG("inner_dialogue_perspective: clearing registry (%u perspectives)",
                        registry->count);
    memset(registry, 0, sizeof(inner_dialogue_perspective_registry_t));
}

/* ============================================================================
 * Registration Implementation
 * ============================================================================ */

int inner_dialogue_perspective_register(
    inner_dialogue_perspective_registry_t* registry,
    const inner_dialogue_perspective_desc_t* desc) {
    if (!registry) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL,
                              "inner_dialogue_perspective: register called with NULL registry");
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL;
    }
    if (!desc) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL,
                              "inner_dialogue_perspective: register called with NULL descriptor");
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL;
    }
    if (!desc->formulate) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_INVALID,
                              "inner_dialogue_perspective: formulate callback is NULL for '%s'",
                              desc->name);
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_INVALID;
    }

    /* Check for duplicate type */
    for (uint32_t i = 0; i < INNER_DIALOGUE_MAX_PERSPECTIVES; i++) {
        if (registry->entries[i].registered &&
            registry->entries[i].desc.type == desc->type) {
            NIMCP_LOGGING_WARN("inner_dialogue_perspective: duplicate type %u (%s)",
                               (unsigned)desc->type, desc->name);
            return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_DUPLICATE;
        }
    }

    /* Check capacity */
    if (registry->count >= INNER_DIALOGUE_MAX_PERSPECTIVES) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_FULL,
                              "inner_dialogue_perspective: registry full (%u/%u)",
                              registry->count, (unsigned)INNER_DIALOGUE_MAX_PERSPECTIVES);
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_FULL;
    }

    /* Find first free slot */
    for (uint32_t i = 0; i < INNER_DIALOGUE_MAX_PERSPECTIVES; i++) {
        if (!registry->entries[i].registered) {
            memcpy(&registry->entries[i].desc, desc,
                   sizeof(inner_dialogue_perspective_desc_t));
            registry->entries[i].registered = true;
            registry->entries[i].turns_produced = 0;
            registry->entries[i].turns_skipped = 0;
            registry->entries[i].cumulative_confidence = 0.0f;
            registry->entries[i].cumulative_relevance = 0.0f;
            registry->entries[i].last_turn_timestamp_us = 0;
            registry->count++;

            NIMCP_LOGGING_INFO("inner_dialogue_perspective: registered '%s' (type=%u, priority=%.2f) [%u/%u]",
                               desc->name, (unsigned)desc->type,
                               (double)desc->base_priority,
                               registry->count, (unsigned)INNER_DIALOGUE_MAX_PERSPECTIVES);
            perspective_heartbeat("perspective_register", 1.0f);
            return 0;
        }
    }

    /* Should not reach here if count < MAX */
    return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_FULL;
}

int inner_dialogue_perspective_unregister(
    inner_dialogue_perspective_registry_t* registry,
    perspective_type_t type) {
    if (!registry) {
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL;
    }
    for (uint32_t i = 0; i < INNER_DIALOGUE_MAX_PERSPECTIVES; i++) {
        if (registry->entries[i].registered &&
            registry->entries[i].desc.type == type) {
            NIMCP_LOGGING_INFO("inner_dialogue_perspective: unregistered '%s' (type=%u)",
                               registry->entries[i].desc.name, (unsigned)type);
            memset(&registry->entries[i], 0, sizeof(inner_dialogue_perspective_entry_t));
            registry->count--;
            return 0;
        }
    }
    return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NOT_FOUND;
}

const inner_dialogue_perspective_entry_t* inner_dialogue_perspective_find(
    const inner_dialogue_perspective_registry_t* registry,
    perspective_type_t type) {
    if (!registry) return NULL;
    for (uint32_t i = 0; i < INNER_DIALOGUE_MAX_PERSPECTIVES; i++) {
        if (registry->entries[i].registered &&
            registry->entries[i].desc.type == type) {
            return &registry->entries[i];
        }
    }
    return NULL;
}

uint32_t inner_dialogue_perspective_count(
    const inner_dialogue_perspective_registry_t* registry) {
    return registry ? registry->count : 0;
}

/* ============================================================================
 * Priority Scheduling
 * ============================================================================ */

/**
 * @brief Clamp float to [min, max] (local utility matching nimcp_clamp_f pattern)
 */
static inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

float inner_dialogue_perspective_compute_priority(
    const inner_dialogue_perspective_entry_t* entry,
    const perspective_turn_context_t* context,
    uint32_t current_turn_number) {
    if (!entry || !context) {
        return 0.0f;
    }

    float base = entry->desc.base_priority;

    /* Relevance factor: check_relevance callback or 1.0 */
    float relevance = 1.0f;
    if (entry->desc.check_relevance) {
        relevance = entry->desc.check_relevance(context);
        relevance = clamp_f(relevance, 0.0f, 1.0f);
    }

    /* Urgency boost */
    float urgency_factor = 1.0f + clamp_f(context->urgency, 0.0f, 1.0f);

    /* Recency penalty — avoid same perspective speaking consecutive turns
     * Penalty decays linearly over last 4 turns */
    float recency_penalty = 0.0f;
    if (context->history && current_turn_number > 0) {
        for (uint32_t i = 0; i < 4 && i < inner_dialogue_turn_history_count(context->history); i++) {
            const inner_dialogue_turn_t* t = inner_dialogue_turn_history_get_at(
                context->history, i);
            if (t && t->perspective_idx < INNER_DIALOGUE_MAX_PERSPECTIVES) {
                /* Find if this turn's perspective matches our entry */
                if (t->perspective_idx == (uint32_t)entry->desc.type) {
                    recency_penalty += 0.25f * (1.0f - (float)i * 0.25f);
                }
            }
        }
    }
    recency_penalty = clamp_f(recency_penalty, 0.0f, 0.8f);

    /* Fairness: bonus for underrepresented perspectives */
    float fairness = 1.0f;
    if (current_turn_number > 0 && entry->turns_produced < current_turn_number) {
        float expected_share = 1.0f / (float)(context->history ?
            inner_dialogue_turn_history_count(context->history) : 1);
        float actual_share = (current_turn_number > 0) ?
            (float)entry->turns_produced / (float)current_turn_number : 0.0f;
        if (actual_share < expected_share) {
            fairness = 1.0f + (expected_share - actual_share);
        }
    }

    float effective = base * relevance * urgency_factor *
                      (1.0f - recency_penalty) * fairness;

    NIMCP_LOGGING_TRACE("inner_dialogue_perspective: priority for '%s': "
                        "base=%.2f rel=%.2f urg=%.2f rec=%.2f fair=%.2f → eff=%.3f",
                        entry->desc.name, (double)base, (double)relevance,
                        (double)urgency_factor, (double)recency_penalty,
                        (double)fairness, (double)effective);
    return effective;
}

int inner_dialogue_perspective_select_next(
    const inner_dialogue_perspective_registry_t* registry,
    const perspective_turn_context_t* context,
    uint32_t turn_number) {
    if (!registry || !context || registry->count == 0) {
        return -1;
    }

    float best_priority = -1.0f;
    int best_idx = -1;

    for (uint32_t i = 0; i < INNER_DIALOGUE_MAX_PERSPECTIVES; i++) {
        if (!registry->entries[i].registered) continue;

        float prio = inner_dialogue_perspective_compute_priority(
            &registry->entries[i], context, turn_number);
        if (prio > best_priority) {
            best_priority = prio;
            best_idx = (int)i;
        }
    }

    if (best_idx >= 0) {
        NIMCP_LOGGING_DEBUG("inner_dialogue_perspective: selected '%s' (idx=%d, priority=%.3f)",
                            registry->entries[best_idx].desc.name,
                            best_idx, (double)best_priority);
    } else {
        NIMCP_LOGGING_WARN("inner_dialogue_perspective: no perspective available for selection");
    }

    perspective_heartbeat("perspective_select", 1.0f);
    return best_idx;
}

/* ============================================================================
 * Built-in Stub Formulate Callbacks
 * ============================================================================ */

/**
 * @brief Generic stub formulate that fills a template assertion
 *
 * Each built-in perspective uses this with different content templates.
 * Real implementations will be wired to actual cognitive modules.
 */
static bool stub_formulate_generic(const perspective_turn_context_t* context,
                                    inner_dialogue_turn_t* output,
                                    perspective_type_t type,
                                    const char* prefix) {
    if (!context || !output) return false;

    output->perspective_idx = (uint32_t)type;
    output->act = DIALOGUE_ACT_ASSERT;
    output->confidence = 0.5f;
    output->relevance = 0.5f;
    output->novelty = 0.3f;
    output->agreement_with_prior = 0.5f;
    output->emotional_valence = 0.0f;
    output->references_turn_id = 0;

    if (context->last_turn) {
        output->references_turn_id = context->last_turn->turn_id;
        /* Vary act based on context */
        if (context->turn_number > 3) {
            output->act = DIALOGUE_ACT_ELABORATE;
        }
    }

    int written = snprintf(output->content, INNER_DIALOGUE_TURN_MAX_CONTENT,
                           "[%s stub] Considering topic: %s (turn %u)",
                           prefix, context->topic ? context->topic : "(none)",
                           context->turn_number);
    output->content_len = (written > 0) ? (uint32_t)written : 0;

    NIMCP_LOGGING_TRACE("inner_dialogue_perspective: stub '%s' formulated turn", prefix);
    return true;
}

/* Individual built-in formulate callbacks */

static bool formulate_analytical(const perspective_turn_context_t* ctx,
                                  inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_ANALYTICAL, "Analytical")) {
        return false;
    }
    out->confidence = 0.65f;
    out->emotional_valence = 0.0f;  /* Analytical is emotionally neutral */
    return true;
}

static bool formulate_emotional(const perspective_turn_context_t* ctx,
                                 inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_EMOTIONAL, "Emotional")) {
        return false;
    }
    out->confidence = 0.55f;
    out->emotional_valence = ctx->emotional_temperature * 0.8f;
    return true;
}

static bool formulate_critical(const perspective_turn_context_t* ctx,
                                inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_CRITICAL, "Critical")) {
        return false;
    }
    out->act = DIALOGUE_ACT_CHALLENGE;
    out->confidence = 0.60f;
    out->agreement_with_prior = 0.3f;  /* Critical voice tends to disagree */
    return true;
}

static bool formulate_creative(const perspective_turn_context_t* ctx,
                                inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_CREATIVE, "Creative")) {
        return false;
    }
    out->act = DIALOGUE_ACT_REFRAME;
    out->confidence = 0.45f;
    out->novelty = 0.8f;  /* Creative perspective produces high novelty */
    return true;
}

static bool formulate_memory(const perspective_turn_context_t* ctx,
                              inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_MEMORY, "Memory")) {
        return false;
    }
    out->act = DIALOGUE_ACT_ELABORATE;
    out->confidence = 0.70f;
    out->novelty = 0.2f;  /* Memory recalls known information */
    return true;
}

static bool formulate_ethical(const perspective_turn_context_t* ctx,
                               inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_ETHICAL, "Ethical")) {
        return false;
    }
    out->act = DIALOGUE_ACT_WARN;
    out->confidence = 0.60f;
    return true;
}

static bool formulate_metacognitive(const perspective_turn_context_t* ctx,
                                     inner_dialogue_turn_t* out) {
    if (!stub_formulate_generic(ctx, out, PERSPECTIVE_METACOGNITIVE, "Metacognitive")) {
        return false;
    }
    out->act = DIALOGUE_ACT_INTROSPECT;
    out->confidence = 0.55f;

    /* Metacognitive perspective monitors the dialogue itself */
    if (ctx->history && inner_dialogue_turn_history_count(ctx->history) > 2) {
        float entropy = inner_dialogue_turn_history_act_entropy(ctx->history, 0);
        if (entropy >= 0.0f && entropy < 1.5f) {
            /* Low diversity — flag as concern */
            int written = snprintf(out->content, INNER_DIALOGUE_TURN_MAX_CONTENT,
                "[Metacognitive stub] Low dialogue diversity detected (entropy=%.2f). "
                "Consider broadening the discussion.",
                (double)entropy);
            out->content_len = (written > 0) ? (uint32_t)written : 0;
        }
    }
    return true;
}

/* ============================================================================
 * Built-in Registration
 * ============================================================================ */

int inner_dialogue_register_builtin_perspectives(
    inner_dialogue_perspective_registry_t* registry) {
    if (!registry) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL,
                              "inner_dialogue_perspective: register_builtins NULL registry");
        return NIMCP_INNER_DIALOGUE_PERSPECTIVE_ERROR_NULL;
    }

    NIMCP_LOGGING_INFO("inner_dialogue_perspective: registering %u built-in perspectives",
                       (unsigned)PERSPECTIVE_BUILTIN_COUNT);

    struct {
        perspective_type_t type;
        const char* name;
        float priority;
        perspective_formulate_fn formulate;
    } builtins[] = {
        { PERSPECTIVE_ANALYTICAL,    "Analytical",    0.7f, formulate_analytical    },
        { PERSPECTIVE_EMOTIONAL,     "Emotional",     0.6f, formulate_emotional     },
        { PERSPECTIVE_CRITICAL,      "Critical",      0.65f, formulate_critical      },
        { PERSPECTIVE_CREATIVE,      "Creative",      0.5f, formulate_creative      },
        { PERSPECTIVE_MEMORY,        "Memory",        0.55f, formulate_memory        },
        { PERSPECTIVE_ETHICAL,       "Ethical",        0.6f, formulate_ethical        },
        { PERSPECTIVE_METACOGNITIVE, "Metacognitive", 0.5f, formulate_metacognitive },
    };

    uint32_t num_builtins = sizeof(builtins) / sizeof(builtins[0]);
    for (uint32_t i = 0; i < num_builtins; i++) {
        inner_dialogue_perspective_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.type = builtins[i].type;
        strncpy(desc.name, builtins[i].name, INNER_DIALOGUE_PERSPECTIVE_NAME_LEN - 1);
        desc.base_priority = builtins[i].priority;
        desc.formulate = builtins[i].formulate;
        desc.check_relevance = NULL;  /* All builtins always relevant (stub) */
        desc.observe = NULL;
        desc.user_data = NULL;

        int rc = inner_dialogue_perspective_register(registry, &desc);
        if (rc != 0) {
            NIMCP_LOGGING_ERROR("inner_dialogue_perspective: failed to register '%s' (rc=%d)",
                                builtins[i].name, rc);
            return rc;
        }
    }

    NIMCP_LOGGING_INFO("inner_dialogue_perspective: all %u built-in perspectives registered",
                       num_builtins);
    perspective_heartbeat("register_builtins", 1.0f);
    return 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void perspective_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_perspective_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int perspective_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "perspective_training_begin: NULL argument");
        return -1;
    }
    perspective_heartbeat_instance(NULL, "perspective_training_begin", 0.0f);
    return 0;
}

int perspective_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "perspective_training_end: NULL argument");
        return -1;
    }
    perspective_heartbeat_instance(NULL, "perspective_training_end", 1.0f);
    return 0;
}

int perspective_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "perspective_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    perspective_heartbeat_instance(NULL, "perspective_training_step", progress);
    return 0;
}
