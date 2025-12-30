//=============================================================================
// nimcp_brain_quantum_reasoning.c - Brain-Level Quantum Reasoning Implementation
//=============================================================================

#include "core/brain/inference/nimcp_brain_quantum_reasoning.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

//=============================================================================
// Internal State
//=============================================================================

/**
 * @brief Brain quantum reasoning context
 *
 * Stored in brain_struct as opaque pointer
 */
typedef struct brain_qreason_ctx_s {
    qreason_t reasoner;              /**< Underlying quantum reasoner */
    brain_qreason_config_t config;   /**< Configuration */
    brain_qreason_stats_t stats;     /**< Statistics */

    /* Modulation state */
    float fatigue_level;             /**< Current fatigue [0,1] */
    float stress_level;              /**< Current stress [0,1] */

    /* Computed modulation factors */
    float effective_max_iterations;  /**< Modulated max iterations */
    float effective_min_confidence;  /**< Modulated confidence threshold */
} brain_qreason_ctx_t;

//=============================================================================
// Configuration API
//=============================================================================

brain_qreason_config_t brain_qreason_default_config(void) {
    return (brain_qreason_config_t){
        .enabled = true,
        .max_grover_iterations = 0,      /* Auto-compute */
        .max_inference_depth = 20,
        .min_confidence = 0.5f,
        .use_ternary_logic = true,
        .enable_interference = true,
        .integrate_with_executive = true,
        .integrate_with_parietal = true,
        .fatigue_sensitivity = 0.5f,
        .stress_sensitivity = 0.5f
    };
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_qreason_ctx_t* get_ctx(brain_t brain) {
    if (!brain) return NULL;
    struct brain_struct* b = (struct brain_struct*)brain;
    return (brain_qreason_ctx_t*)b->quantum_reasoner;
}

static void update_modulation(brain_qreason_ctx_t* ctx) {
    if (!ctx) return;

    /* Fatigue reduces max iterations */
    float fatigue_factor = 1.0f - (ctx->fatigue_level * ctx->config.fatigue_sensitivity);
    if (fatigue_factor < 0.2f) fatigue_factor = 0.2f;

    /* Stress increases confidence threshold (more conservative) */
    float stress_factor = 1.0f + (ctx->stress_level * ctx->config.stress_sensitivity * 0.5f);
    if (stress_factor > 2.0f) stress_factor = 2.0f;

    ctx->effective_max_iterations = (float)ctx->config.max_grover_iterations * fatigue_factor;
    ctx->effective_min_confidence = ctx->config.min_confidence * stress_factor;
    if (ctx->effective_min_confidence > 0.95f) {
        ctx->effective_min_confidence = 0.95f;
    }
}

//=============================================================================
// Lifecycle API
//=============================================================================

bool nimcp_brain_factory_init_quantum_reasoning(brain_t brain,
                                                  const brain_qreason_config_t* config) {
    if (!brain) {
        NIMCP_LOGGING_ERROR("Cannot init quantum reasoning: NULL brain");
        return false;
    }

    struct brain_struct* b = (struct brain_struct*)brain;

    /* Skip if already initialized */
    if (b->quantum_reasoner != NULL) {
        NIMCP_LOGGING_WARN("Quantum reasoning already initialized");
        return true;
    }

    NIMCP_LOGGING_INFO("Initializing brain quantum reasoning subsystem...");

    /* Allocate context */
    brain_qreason_ctx_t* ctx = nimcp_malloc(sizeof(brain_qreason_ctx_t));
    if (!ctx) {
        NIMCP_LOGGING_ERROR("Failed to allocate quantum reasoning context");
        return false;
    }
    memset(ctx, 0, sizeof(brain_qreason_ctx_t));

    /* Apply configuration */
    ctx->config = config ? *config : brain_qreason_default_config();

    /* Create underlying quantum reasoner */
    qreason_config_t qconfig = qreason_default_config();
    qconfig.max_grover_iterations = ctx->config.max_grover_iterations;
    qconfig.max_inference_depth = ctx->config.max_inference_depth;
    qconfig.min_confidence = ctx->config.min_confidence;
    qconfig.use_ternary_logic = ctx->config.use_ternary_logic;
    qconfig.enable_interference = ctx->config.enable_interference;

    ctx->reasoner = qreason_create(&qconfig);
    if (!ctx->reasoner) {
        NIMCP_LOGGING_ERROR("Failed to create quantum reasoner");
        nimcp_free(ctx);
        return false;
    }

    /* Initialize modulation */
    ctx->fatigue_level = 0.0f;
    ctx->stress_level = 0.0f;
    update_modulation(ctx);

    /* Store in brain */
    b->quantum_reasoner = ctx;
    b->quantum_reasoning_enabled = ctx->config.enabled;

    NIMCP_LOGGING_INFO("Brain quantum reasoning initialized successfully");
    NIMCP_LOGGING_DEBUG("  - Max Grover iterations: %u",
        ctx->config.max_grover_iterations == 0 ? 10 : ctx->config.max_grover_iterations);
    NIMCP_LOGGING_DEBUG("  - Ternary logic: %s", ctx->config.use_ternary_logic ? "enabled" : "disabled");
    NIMCP_LOGGING_DEBUG("  - Interference: %s", ctx->config.enable_interference ? "enabled" : "disabled");

    return true;
}

void nimcp_brain_qreason_destroy(brain_t brain) {
    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;
    brain_qreason_ctx_t* ctx = (brain_qreason_ctx_t*)b->quantum_reasoner;

    if (ctx) {
        if (ctx->reasoner) {
            qreason_destroy(ctx->reasoner);
        }
        nimcp_free(ctx);
        b->quantum_reasoner = NULL;
        b->quantum_reasoning_enabled = false;
        NIMCP_LOGGING_DEBUG("Brain quantum reasoning destroyed");
    }
}

bool nimcp_brain_qreason_is_enabled(brain_t brain) {
    if (!brain) return false;
    struct brain_struct* b = (struct brain_struct*)brain;
    return b->quantum_reasoning_enabled && b->quantum_reasoner != NULL;
}

int nimcp_brain_qreason_set_enabled(brain_t brain, bool enabled) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;
    brain_qreason_ctx_t* ctx = get_ctx(brain);

    if (!ctx) {
        if (enabled) {
            /* Initialize if enabling and not yet created */
            return nimcp_brain_factory_init_quantum_reasoning(brain, NULL) ? 0 : -1;
        }
        return 0;
    }

    ctx->config.enabled = enabled;
    b->quantum_reasoning_enabled = enabled;
    return 0;
}

//=============================================================================
// Query API
//=============================================================================

int nimcp_brain_qreason_solve_sat(brain_t brain,
                                   const brain_reasoning_query_t* query,
                                   brain_reasoning_result_t* result) {
    if (!brain || !query || !result) return -1;

    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) {
        NIMCP_LOGGING_WARN("Quantum reasoning not available");
        return -1;
    }

    if (!ctx->config.enabled) {
        NIMCP_LOGGING_DEBUG("Quantum reasoning disabled, skipping");
        return -1;
    }

    memset(result, 0, sizeof(*result));

    uint64_t start_time = nimcp_time_get_us();

    /* Solve using quantum reasoner */
    qreason_result_t qresult = {0};
    int status = qreason_solve_sat(ctx->reasoner, &query->cnf, &qresult);

    uint64_t end_time = nimcp_time_get_us();

    if (status != 0) {
        NIMCP_LOGGING_ERROR("Quantum SAT solve failed");
        ctx->stats.timeout_count++;
        return -1;
    }

    /* Populate result */
    result->satisfiable = qresult.satisfiable;
    result->satisfaction_probability = qresult.satisfaction_prob;
    result->grover_iterations = qresult.grover_iterations;
    result->num_variables = query->cnf.n_variables;
    result->solve_time_us = end_time - start_time;

    /* Copy assignment and compute average confidence */
    float total_confidence = 0.0f;
    uint32_t conf_count = 0;
    for (uint32_t i = 0; i < query->cnf.n_variables && i < BRAIN_QREASON_MAX_VARIABLES; i++) {
        result->assignment[i] = qresult.assignment[i];
        if (qresult.confidences[i] > 0.0f) {
            total_confidence += qresult.confidences[i];
            conf_count++;
        }
    }
    result->confidence = conf_count > 0 ? total_confidence / conf_count : qresult.satisfaction_prob;

    /* Estimate quantum speedup: √(2^N) vs 2^N */
    if (query->cnf.n_variables > 0) {
        float classical_ops = (float)(1u << query->cnf.n_variables);
        float quantum_ops = sqrtf(classical_ops);
        result->quantum_speedup = classical_ops / quantum_ops;
    }

    /* Update statistics */
    ctx->stats.total_queries++;
    if (result->satisfiable) {
        ctx->stats.satisfiable_count++;
    } else {
        ctx->stats.unsatisfiable_count++;
    }
    ctx->stats.total_grover_iterations += result->grover_iterations;
    ctx->stats.avg_grover_iterations =
        (float)ctx->stats.total_grover_iterations / (float)ctx->stats.total_queries;
    ctx->stats.avg_solve_time_us =
        (ctx->stats.avg_solve_time_us * (ctx->stats.total_queries - 1) + result->solve_time_us)
        / ctx->stats.total_queries;
    ctx->stats.avg_confidence =
        (ctx->stats.avg_confidence * (ctx->stats.total_queries - 1) + result->confidence)
        / ctx->stats.total_queries;

    NIMCP_LOGGING_DEBUG("SAT solve: %s (conf=%.2f, iters=%u, time=%luus)",
        result->satisfiable ? "SAT" : "UNSAT",
        result->confidence,
        result->grover_iterations,
        (unsigned long)result->solve_time_us);

    return 0;
}

int nimcp_brain_qreason_check_goal_feasibility(brain_t brain,
                                                 uint32_t goal_id,
                                                 const uint32_t* constraints,
                                                 uint32_t num_constraints,
                                                 bool* feasible,
                                                 float* confidence) {
    if (!brain || !feasible || !confidence) return -1;

    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    /* Build CNF from goal and constraints */
    brain_reasoning_query_t query = {0};
    snprintf(query.description, sizeof(query.description),
             "Goal %u feasibility", goal_id);

    /* Goal variable must be true */
    query.cnf.n_variables = num_constraints + 1;
    query.cnf.n_clauses = num_constraints + 1;

    /* Clause 0: goal must be true */
    query.cnf.clauses[0].n_literals = 1;
    query.cnf.clauses[0].literals[0].variable = goal_id;
    query.cnf.clauses[0].literals[0].negated = false;

    /* Additional clauses from constraints */
    for (uint32_t i = 0; i < num_constraints && i < QREASON_MAX_CLAUSES - 1; i++) {
        query.cnf.clauses[i + 1].n_literals = 1;
        query.cnf.clauses[i + 1].literals[0].variable = constraints[i];
        query.cnf.clauses[i + 1].literals[0].negated = false;
    }

    brain_reasoning_result_t result;
    int status = nimcp_brain_qreason_solve_sat(brain, &query, &result);

    if (status != 0) {
        *feasible = false;
        *confidence = 0.0f;
        return -1;
    }

    *feasible = result.satisfiable;
    *confidence = result.confidence;

    return 0;
}

int nimcp_brain_qreason_query_ternary(brain_t brain,
                                        uint32_t variable,
                                        qreason_truth_t* value,
                                        float* confidence) {
    if (!brain || !value || !confidence) return -1;

    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    *value = qreason_get_fact(ctx->reasoner, variable, confidence);
    return 0;
}

//=============================================================================
// Knowledge Base API
//=============================================================================

int nimcp_brain_qreason_set_fact(brain_t brain,
                                   uint32_t variable,
                                   qreason_truth_t value,
                                   float confidence) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    return qreason_set_fact(ctx->reasoner, variable, value, confidence);
}

int nimcp_brain_qreason_get_fact(brain_t brain,
                                   uint32_t variable,
                                   qreason_truth_t* value,
                                   float* confidence) {
    if (!value || !confidence) return -1;

    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    *value = qreason_get_fact(ctx->reasoner, variable, confidence);
    return 0;
}

int nimcp_brain_qreason_add_rule(brain_t brain,
                                   const uint32_t* antecedents,
                                   uint32_t num_antecedents,
                                   uint32_t consequent,
                                   float confidence) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    return qreason_add_rule(ctx->reasoner, antecedents, num_antecedents,
                            consequent, confidence);
}

int nimcp_brain_qreason_clear_kb(brain_t brain) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx || !ctx->reasoner) return -1;

    qreason_clear_facts(ctx->reasoner);
    qreason_clear_rules(ctx->reasoner);
    return 0;
}

//=============================================================================
// Modulation API
//=============================================================================

int nimcp_brain_qreason_set_fatigue(brain_t brain, float fatigue_level) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx) return -1;

    ctx->fatigue_level = fatigue_level < 0.0f ? 0.0f :
                         (fatigue_level > 1.0f ? 1.0f : fatigue_level);
    update_modulation(ctx);

    NIMCP_LOGGING_DEBUG("Quantum reasoning fatigue set to %.2f (eff_iters=%.0f)",
        ctx->fatigue_level, ctx->effective_max_iterations);

    return 0;
}

int nimcp_brain_qreason_set_stress(brain_t brain, float stress_level) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx) return -1;

    ctx->stress_level = stress_level < 0.0f ? 0.0f :
                        (stress_level > 1.0f ? 1.0f : stress_level);
    update_modulation(ctx);

    NIMCP_LOGGING_DEBUG("Quantum reasoning stress set to %.2f (eff_conf=%.2f)",
        ctx->stress_level, ctx->effective_min_confidence);

    return 0;
}

//=============================================================================
// Statistics API
//=============================================================================

int nimcp_brain_qreason_get_stats(brain_t brain, brain_qreason_stats_t* stats) {
    if (!stats) return -1;

    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (!ctx) {
        memset(stats, 0, sizeof(*stats));
        return -1;
    }

    *stats = ctx->stats;
    return 0;
}

void nimcp_brain_qreason_reset_stats(brain_t brain) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    if (ctx) {
        memset(&ctx->stats, 0, sizeof(ctx->stats));
    }
}

//=============================================================================
// Internal Access
//=============================================================================

qreason_t nimcp_brain_qreason_get_handle(brain_t brain) {
    brain_qreason_ctx_t* ctx = get_ctx(brain);
    return ctx ? ctx->reasoner : NULL;
}
