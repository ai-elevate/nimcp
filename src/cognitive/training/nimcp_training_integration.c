/**
 * @file nimcp_training_integration.c
 * @brief Training Integration Layer - Implementation
 *
 * WHAT: Simplified wrapper around basal ganglia, medulla, symbolic logic,
 *       reasoning chain, and information forager for the training pipeline
 * WHY:  The training pipeline needs clean, simple APIs to interact with
 *       brain subsystems without managing their individual complexities
 * HOW:  Each function guards against NULL brain and missing subsystems,
 *       returning sensible defaults when subsystems are unavailable
 *
 * @author NIMCP Development Team
 * @date 2026-02-25
 * @version 1.0.0
 */

#include "cognitive/training/nimcp_training_integration.h"

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/subcortical/nimcp_basal_ganglia.h"
#include "core/brain/subcortical/nimcp_basal_ganglia_enhanced.h"
#include "core/medulla/nimcp_medulla.h"
#include "cognitive/reasoning/nimcp_reasoning_chain.h"
#include "cognitive/reasoning/nimcp_forward_chaining.h"
#include "cognitive/reasoning/nimcp_backward_chaining.h"
#include "cognitive/reasoning/nimcp_knowledge_base_interface.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE "training_integration"

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Simple FNV-1a hash for domain string to uint32_t context ID
 *
 * WHAT: Hash a domain string into a 32-bit integer
 * WHY:  BG habits use uint32_t trigger_context; we need to map string domains
 * HOW:  FNV-1a algorithm - fast, well-distributed for short strings
 */
static uint32_t hash_domain(const char* domain) {
    if (!domain) { return 0; }
    uint32_t hash = 2166136261u; /* FNV offset basis */
    while (*domain) {
        hash ^= (uint8_t)*domain++;
        hash *= 16777619u; /* FNV prime */
    }
    return hash;
}

/**
 * @brief Clamp a float to [min, max]
 */
static inline float clampf(float val, float lo, float hi) {
    if (val < lo) { return lo; }
    if (val > hi) { return hi; }
    return val;
}

/**
 * @brief Get the core basal ganglia from brain, or NULL if unavailable
 */
static basal_ganglia_t* get_bg_core(brain_t brain) {
    if (!brain) { return NULL; }
    if (!brain->basal_ganglia_enabled) { return NULL; }
    if (!brain->basal_ganglia) { return NULL; }
    return bg_enhanced_get_core(brain->basal_ganglia);
}

/*=============================================================================
 * REASONING ENGINE CACHE
 *
 * Since we cannot modify brain_internal.h, we use a simple static cache
 * that stores one reasoning engine at a time. The training pipeline
 * typically operates on a single brain, so this is sufficient.
 *===========================================================================*/

static reasoning_engine_t* g_cached_engine = NULL;
static brain_t g_cached_brain = NULL;
static uint32_t g_last_reasoning_steps = 0;

/*=============================================================================
 * BASAL GANGLIA TRAINING INTEGRATION
 *===========================================================================*/

int brain_ti_update_reward(brain_t brain, float actual_reward, float expected_reward) {
    if (!brain) { return -1; }
    if (!brain->basal_ganglia_enabled || !brain->basal_ganglia) {
        return -1;
    }

    /* Use enhanced BG reward processing which updates all subsystems */
    int rc = bg_enhanced_process_reward(brain->basal_ganglia,
                                        actual_reward, expected_reward);
    if (rc != 0) {
        NIMCP_LOGGING_WARN("training_integration: bg_enhanced_process_reward "
                           "failed (rc=%d)", rc);
        return -1;
    }

    return 0;
}

float brain_ti_get_conflict(brain_t brain) {
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return 0.0f; }
    return basal_ganglia_get_conflict(bg);
}

int brain_ti_get_mode(brain_t brain) {
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return 0; } /* BG_MODE_GOAL_DIRECTED */
    return (int)basal_ganglia_get_mode(bg);
}

float brain_ti_get_dopamine(brain_t brain) {
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return 0.5f; } /* BG_DOPAMINE_BASELINE */
    return basal_ganglia_get_dopamine(bg);
}

float brain_ti_get_rpe(brain_t brain) {
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return 0.0f; }
    return basal_ganglia_get_rpe(bg);
}

int brain_ti_register_habit(brain_t brain, const char* domain, uint32_t action_id) {
    if (!domain) { return -1; }
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return -1; }

    uint32_t context = hash_domain(domain);
    uint32_t habit_id = 0;
    int rc = basal_ganglia_register_habit(bg, context, action_id, &habit_id);
    if (rc != 0) {
        NIMCP_LOGGING_WARN("training_integration: register_habit failed for "
                           "domain='%s' action=%u (rc=%d)", domain, action_id, rc);
        return -1;
    }

    NIMCP_LOGGING_DEBUG("training_integration: registered habit %u for "
                        "domain='%s' (context=0x%08x, action=%u)",
                        habit_id, domain, context, action_id);
    return (int)habit_id;
}

int brain_ti_check_habit(brain_t brain, const char* domain) {
    if (!domain) { return -1; }
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return -1; }

    uint32_t context = hash_domain(domain);
    uint32_t action_id = 0;
    bool found = basal_ganglia_check_habit(bg, context, &action_id);
    if (!found) { return -1; }

    return (int)action_id;
}

int brain_ti_strengthen_habit(brain_t brain, int habit_id, bool success) {
    if (habit_id < 0) { return -1; }
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return -1; }

    int rc = basal_ganglia_strengthen_habit(bg, (uint32_t)habit_id, success);
    if (rc != 0) {
        NIMCP_LOGGING_WARN("training_integration: strengthen_habit failed for "
                           "habit_id=%d (rc=%d)", habit_id, rc);
        return -1;
    }

    return 0;
}

float brain_ti_get_habit_strength(brain_t brain, int habit_id) {
    if (habit_id < 0) { return -1.0f; }
    basal_ganglia_t* bg = get_bg_core(brain);
    if (!bg) { return -1.0f; }
    return basal_ganglia_get_habit_strength(bg, (uint32_t)habit_id);
}

/*=============================================================================
 * MEDULLA TRAINING INTEGRATION
 *===========================================================================*/

float brain_ti_get_arousal(brain_t brain) {
    if (!brain) { return 0.5f; }
    if (!brain->medulla_enabled || !brain->medulla) { return 0.5f; }
    float arousal = medulla_get_arousal_level(brain->medulla);
    /* medulla_get_arousal_level returns -1.0f on error */
    if (arousal < 0.0f) { return 0.5f; }
    return arousal;
}

int brain_ti_get_circadian_phase(brain_t brain) {
    if (!brain) { return 0; }
    if (!brain->medulla_enabled || !brain->medulla) { return 0; }
    return (int)medulla_get_circadian_phase(brain->medulla);
}

int brain_ti_boost_arousal(brain_t brain, float delta) {
    if (!brain) { return -1; }
    if (!brain->medulla_enabled || !brain->medulla) { return -1; }
    int rc = medulla_boost_arousal(brain->medulla, delta);
    if (rc != 0) {
        NIMCP_LOGGING_WARN("training_integration: boost_arousal failed "
                           "(delta=%.3f, rc=%d)", delta, rc);
        return -1;
    }
    return 0;
}

float brain_ti_get_circadian_efficiency(brain_t brain) {
    if (!brain) { return 1.0f; }
    if (!brain->medulla_enabled || !brain->medulla) { return 1.0f; }

    circadian_phase_t phase = medulla_get_circadian_phase(brain->medulla);

    /*
     * Map circadian phase to learning efficiency multiplier.
     *
     * The medulla uses the circadian_phase_t enum from nimcp_medulla.h:
     *   CIRCADIAN_PHASE_EARLY_MORNING = 0  (06:00-09:00)
     *   CIRCADIAN_PHASE_MORNING       = 1  (09:00-12:00)
     *   CIRCADIAN_PHASE_AFTERNOON     = 2  (12:00-15:00)
     *   CIRCADIAN_PHASE_EVENING       = 3  (15:00-18:00)
     *   CIRCADIAN_PHASE_LATE_EVENING  = 4  (18:00-21:00)
     *   CIRCADIAN_PHASE_NIGHT         = 5  (21:00-24:00)
     *   CIRCADIAN_PHASE_DEEP_NIGHT    = 6  (00:00-03:00)
     *   CIRCADIAN_PHASE_PRE_DAWN      = 7  (03:00-06:00)
     */
    static const float phase_efficiency[] = {
        1.2f,   /* 0: EARLY_MORNING - rising, good */
        1.5f,   /* 1: MORNING      - peak alertness, best for learning */
        1.0f,   /* 2: AFTERNOON    - post-lunch dip recovery */
        1.2f,   /* 3: EVENING      - second peak */
        0.9f,   /* 4: LATE_EVENING - declining */
        0.85f,  /* 5: NIGHT        - low arousal */
        0.85f,  /* 6: DEEP_NIGHT   - minimal arousal */
        0.8f    /* 7: PRE_DAWN     - minimum efficiency */
    };

    int idx = (int)phase;
    if (idx < 0 || idx >= 8) { return 1.0f; }
    return phase_efficiency[idx];
}

/*=============================================================================
 * SYMBOLIC LOGIC TRAINING INTEGRATION
 *===========================================================================*/

/**
 * @brief Check if symbolic logic engine is available on this brain
 */
static bool logic_available(brain_t brain) {
    if (!brain) { return false; }
    if (!brain->symbolic_logic) { return false; }
    return true;
}

bool brain_ti_add_fact(brain_t brain, const char* fact_str, float salience) {
    if (!logic_available(brain)) { return false; }
    if (!fact_str) { return false; }
    return brain_add_fact(brain, fact_str, salience);
}

bool brain_ti_add_rule(brain_t brain, const char* rule_str, float priority) {
    if (!logic_available(brain)) { return false; }
    if (!rule_str) { return false; }
    return brain_add_rule(brain, rule_str, priority);
}

int brain_ti_forward_chain(brain_t brain, uint32_t max_iterations) {
    if (!logic_available(brain)) { return 0; }
    if (max_iterations == 0) { max_iterations = 100; }

    forward_chain_result_t result;
    memset(&result, 0, sizeof(result));
    bool ok = brain_forward_chain(brain, max_iterations, &result);
    if (!ok) { return 0; }

    int derived = (int)result.num_new_facts;

    if (derived > 0) {
        NIMCP_LOGGING_DEBUG("training_integration: forward_chain derived %d "
                            "new facts in %u iterations (conf=%.3f)",
                            derived, result.iterations_performed,
                            result.confidence);
    }

    forward_chain_free_result(&result);
    return derived;
}

float brain_ti_backward_chain(brain_t brain, const char* goal_str) {
    if (!logic_available(brain)) { return -1.0f; }
    if (!goal_str) { return -1.0f; }

    backward_chain_result_t result;
    memset(&result, 0, sizeof(result));
    bool ok = brain_backward_chain(brain, goal_str, &result);

    if (!ok) {
        /* Could be parse error or genuine failure to prove */
        backward_chain_free_result(&result);
        return -1.0f;
    }

    float confidence = result.proven ? result.confidence : 0.0f;

    if (result.proven) {
        NIMCP_LOGGING_DEBUG("training_integration: backward_chain proved '%s' "
                            "with confidence=%.3f (%u steps, depth=%u)",
                            goal_str, result.confidence,
                            result.num_steps, result.depth_reached);
    }

    backward_chain_free_result(&result);
    return confidence;
}

int brain_ti_query_knowledge(brain_t brain, const char* query_str) {
    if (!logic_available(brain)) { return 0; }
    if (!query_str) { return 0; }

    kb_query_result_t result;
    memset(&result, 0, sizeof(result));
    bool ok = brain_query_knowledge(brain, query_str, &result);
    if (!ok) { return 0; }

    int matches = result.num_matches;
    kb_free_query_result(&result);
    return matches;
}

int brain_ti_get_logic_stats(brain_t brain, brain_ti_logic_stats_t* stats) {
    if (!stats) { return -1; }
    memset(stats, 0, sizeof(*stats));

    if (!logic_available(brain)) { return -1; }

    /* Get base logic engine stats directly from symbolic_logic_t */
    logic_stats_t ls;
    memset(&ls, 0, sizeof(ls));
    if (symbolic_logic_get_stats(brain->symbolic_logic, &ls)) {
        stats->total_facts = ls.facts_stored;
        stats->total_rules = ls.rules_applied;
    }

    /* Get forward chaining stats */
    uint32_t fc_iters = 0;
    uint32_t fc_derived = 0;
    uint64_t fc_time = 0;
    if (brain_get_forward_chain_stats(brain, &fc_iters, &fc_derived, &fc_time)) {
        stats->facts_derived = fc_derived;
    }

    /* Get backward chaining stats */
    uint32_t bc_attempted = 0;
    uint32_t bc_succeeded = 0;
    float bc_avg_depth = 0.0f;
    if (brain_get_backward_chain_stats(brain, &bc_attempted, &bc_succeeded,
                                       &bc_avg_depth)) {
        stats->proofs_completed = bc_succeeded;
        stats->proofs_failed = bc_attempted - bc_succeeded;
    }

    /* Use the knowledge base interface for accurate counts */
    stats->total_facts = brain_get_fact_count(brain);
    stats->total_rules = brain_get_rule_count(brain);

    return 0;
}

/*=============================================================================
 * REASONING CHAIN TRAINING INTEGRATION
 *===========================================================================*/

int brain_ti_init_reasoning(brain_t brain) {
    if (!brain) { return -1; }

    /* If already cached for this brain, reuse */
    if (g_cached_engine && g_cached_brain == brain) {
        return 0;
    }

    /* Destroy any existing cached engine for a different brain */
    if (g_cached_engine) {
        NIMCP_LOGGING_INFO("training_integration: replacing cached reasoning "
                           "engine (was for different brain)");
        reasoning_engine_destroy(g_cached_engine);
        g_cached_engine = NULL;
        g_cached_brain = NULL;
        g_last_reasoning_steps = 0;
    }

    /* Create new engine with defaults */
    reasoning_engine_config_t config = reasoning_engine_default_config();
    g_cached_engine = reasoning_engine_create(&config);
    if (!g_cached_engine) {
        NIMCP_LOGGING_ERROR("training_integration: failed to create reasoning "
                            "engine");
        return -1;
    }

    /* Connect to brain */
    int rc = reasoning_engine_connect_brain(g_cached_engine, brain);
    if (rc != 0) {
        NIMCP_LOGGING_ERROR("training_integration: failed to connect reasoning "
                            "engine to brain (rc=%d)", rc);
        reasoning_engine_destroy(g_cached_engine);
        g_cached_engine = NULL;
        return -1;
    }

    g_cached_brain = brain;
    g_last_reasoning_steps = 0;

    NIMCP_LOGGING_INFO("training_integration: reasoning engine initialized "
                       "and connected to brain");
    return 0;
}

float brain_ti_reason(brain_t brain, const char* query) {
    if (!brain || !query) { return -1.0f; }

    /* Check cache */
    if (!g_cached_engine || g_cached_brain != brain) {
        NIMCP_LOGGING_WARN("training_integration: brain_ti_reason called "
                           "without brain_ti_init_reasoning");
        return -1.0f;
    }

    reasoning_chain_t chain;
    reasoning_chain_init(&chain);

    int rc = reasoning_engine_reason(g_cached_engine, query, &chain);
    if (rc != 0) {
        NIMCP_LOGGING_WARN("training_integration: reasoning failed for "
                           "query='%.80s' (rc=%d)", query, rc);
        reasoning_chain_cleanup(&chain);
        g_last_reasoning_steps = 0;
        return -1.0f;
    }

    float confidence = reasoning_chain_get_confidence(&chain);
    g_last_reasoning_steps = reasoning_chain_get_num_steps(&chain);

    NIMCP_LOGGING_DEBUG("training_integration: reasoning completed for "
                        "'%.80s': confidence=%.3f, steps=%u",
                        query, confidence, g_last_reasoning_steps);

    reasoning_chain_cleanup(&chain);
    return confidence;
}

uint32_t brain_ti_get_reasoning_steps(brain_t brain) {
    if (!brain) { return 0; }
    if (g_cached_brain != brain) { return 0; }
    return g_last_reasoning_steps;
}

void brain_ti_destroy_reasoning(brain_t brain) {
    /* Clear cache if brain matches or if brain is NULL (force clear) */
    if (!brain || g_cached_brain == brain) {
        if (g_cached_engine) {
            NIMCP_LOGGING_INFO("training_integration: destroying cached "
                               "reasoning engine");
            reasoning_engine_destroy(g_cached_engine);
        }
        g_cached_engine = NULL;
        g_cached_brain = NULL;
        g_last_reasoning_steps = 0;
    }
}

/*=============================================================================
 * COMBINED TRAINING HELPERS
 *===========================================================================*/

float brain_ti_compute_adaptive_lr(brain_t brain, float base_lr) {
    if (!brain) { return base_lr; }

    /* Arousal factor: 0.5 + 0.5 * arousal  (0.5x at zero, 1.0x at full) */
    float arousal = brain_ti_get_arousal(brain);
    float arousal_factor = 0.5f + 0.5f * arousal;

    /* Circadian factor: phase-based efficiency multiplier */
    float circadian_factor = brain_ti_get_circadian_efficiency(brain);

    /* RPE bonus: clamp(rpe * 0.2, -0.2, 0.3) */
    float rpe = brain_ti_get_rpe(brain);
    float rpe_bonus = clampf(rpe * 0.2f, -0.2f, 0.3f);

    float adaptive_lr = base_lr * arousal_factor * circadian_factor *
                        (1.0f + rpe_bonus);

    NIMCP_LOGGING_DEBUG("training_integration: adaptive_lr=%.6f "
                        "(base=%.6f, arousal=%.3f[x%.3f], "
                        "circadian=x%.3f, rpe=%.3f[bonus=%.3f])",
                        adaptive_lr, base_lr, arousal, arousal_factor,
                        circadian_factor, rpe, rpe_bonus);

    return adaptive_lr;
}

int brain_ti_post_batch_update(brain_t brain, float accuracy,
                               float expected_accuracy, const char* domain) {
    if (!brain) { return -1; }

    /* Step 1: Update BG reward with accuracy as reward signal */
    brain_ti_update_reward(brain, accuracy, expected_accuracy);

    /* Step 2: Check for habitual behavior in this domain */
    if (domain) {
        int action_id = brain_ti_check_habit(brain, domain);
        if (action_id >= 0) {
            /* Habit exists -- check its strength */
            /* We need the habit_id, not the action_id, for strength query.
             * Since we only have action_id from check_habit, we look up
             * the habit by iterating. For simplicity, we use the BG core
             * habit system directly. */
            basal_ganglia_t* bg = get_bg_core(brain);
            if (bg) {
                uint32_t context = hash_domain(domain);
                /* Find the habit_id for this context */
                for (uint32_t i = 0; i < bg->num_habits; i++) {
                    if (bg->habits[i].trigger_context == context) {
                        float strength = basal_ganglia_get_habit_strength(
                            bg, bg->habits[i].habit_id);
                        if (strength > 0.7f) {
                            NIMCP_LOGGING_INFO("training_integration: domain "
                                               "'%s' in habitual mode "
                                               "(strength=%.3f, action=%d)",
                                               domain, strength, action_id);
                        }

                        /* Step 3: Strengthen habit if accuracy exceeds
                         * expected */
                        if (accuracy > expected_accuracy) {
                            basal_ganglia_strengthen_habit(
                                bg, bg->habits[i].habit_id, true);
                            NIMCP_LOGGING_DEBUG("training_integration: "
                                                "strengthened habit for "
                                                "domain='%s' (accuracy %.3f "
                                                "> expected %.3f)",
                                                domain, accuracy,
                                                expected_accuracy);
                        }
                        break;
                    }
                }
            }
        }
    }

    return 0;
}
